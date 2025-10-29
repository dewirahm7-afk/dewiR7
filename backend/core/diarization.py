# backend/core/diarization.py
#
# Tugas:
# - Build argumen untuk dracin_gender.py
# - Jalankan dracin_gender.main() di thread terpisah
# - Pindahkan output timeline ke root workdir
# - Kembalikan path hasil ke caller (processor, endpoint)
#
# Sekarang support 3 mode gender:
#   reference  -> butuh male_ref/female_ref, voting per SPEAKER
#   hf_svm     -> ECAPA+SVM custom, per SEGMENT
#   wav2vec2   -> audEERING wav2vec2 age+gender, per SEGMENT
#
# Catatan:
# - Semua model dijalankan offline (kecuali diarization pyannote yang tetap butuh HF token).
# - global linking (merge speaker lokal jadi SPK_01, SPK_02) bisa ditambahkan di processor.py
#   setelah kami return data ini.
#
# ---------------------------------------------------------------------

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import sys
import shutil
import json
import traceback

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class DiarizationEngine:
    """
    Jalankan diarization + gender tagging via dracin_gender.py.
    Selalu return dict:
      sukses:
        {
          "success": True,
          "data": {
            "speaker_timeline_raw": "<abs path>",
            "gender_timeline": "<abs path>"
          }
        }
      gagal:
        { "success": False, "error": "<pesan>" }
    """

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1)

    async def process(self, session, config: dict, progress_callback=None) -> dict:
        workdir = Path(session.workdir)
        workdir.mkdir(parents=True, exist_ok=True)

        def task():
            try:
                # import lokal supaya path ROOT sudah ditambah
                from dracin_gender import main as gender_main
                import torch
                import torchaudio

                # ------------ helper util lokal ------------
                AUDIO_PATS = ("*.wav", "*.mp3", "*.flac", "*.m4a", "*.ogg")

                def list_audio(p: Path):
                    out = []
                    if p.is_dir():
                        for pat in AUDIO_PATS:
                            out += sorted(p.glob(pat))
                    elif p.is_file():
                        out = [p]
                    return out

                def make_bank(ref_input: str, label: str) -> str:
                    """
                    Bangun <label>_bank.wav (16k mono) di workdir dari folder/file.
                    Ini dipakai gender_mode="reference".
                    """
                    src = Path(os.path.expandvars(ref_input)).expanduser()
                    files = list_audio(src)
                    if not files:
                        raise FileNotFoundError(f"Tidak ada file audio di: {src}")

                    chunks = []
                    meta_samples = []
                    for f in files:
                        wav, sr = torchaudio.load(str(f))
                        # stereo -> mono
                        if wav.dim() == 2 and wav.size(0) > 1:
                            wav = wav.mean(dim=0, keepdim=True)
                        wav = wav.squeeze(0)
                        # resample -> 16k
                        if sr != 16000:
                            wav = torchaudio.transforms.Resample(sr, 16000)(wav)
                        chunks.append(wav)
                        meta_samples.append({
                            "src": str(f),
                            "dur_sec": float(wav.numel()) / 16000.0,
                        })

                    full = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
                    out_wav = workdir / f"{label}_bank.wav"
                    torchaudio.save(
                        str(out_wav),
                        full.unsqueeze(0),
                        16000,
                        encoding="PCM_S",
                        bits_per_sample=16,
                    )

                    cache = {"files": meta_samples}
                    (workdir / f"{label}_bank_cache.json").write_text(
                        json.dumps(cache, indent=2), encoding="utf-8"
                    )
                    return str(out_wav)

                def latest(pattern: str):
                    items = sorted(
                        workdir.rglob(pattern),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    return items[0] if items else None

                def bring_to_root(p: Path) -> Path:
                    """
                    Pastikan file penting akhirnya ada di root workdir
                    (bukan di subfolder random).
                    """
                    if p and p.parent != workdir:
                        dst = workdir / p.name
                        try:
                            shutil.move(str(p), str(dst))
                        except Exception:
                            dst = p
                        return dst
                    return p

                # ------------ ambil config dari frontend ------------
                gender_mode   = config.get("gender_mode", "hf_svm")  # "reference" / "hf_svm" / "wav2vec2"
                top_n         = int(config.get("top_n", 5))

                hf_token      = (config.get("hf_token") or "").strip()
                use_gpu       = bool(config.get("use_gpu", True))

                # hanya dipakai hf_svm / wav2vec2:
                min_vote      = float(config.get("min_vote", 0.7))
                min_len_sec   = float(config.get("min_len_sec", 0.3))

                # hanya dipakai reference:
                male_ref_in   = config.get("male_ref", "")
                female_ref_in = config.get("female_ref", "")

                # session.wav_16k harus sudah diisi waktu Extract Audio
                audio_arg = Path(session.wav_16k).name

                # ------------ siapkan sys.argv buat dracin_gender.main() ------------
                argv_backup = sys.argv[:]
                sys.argv = [
                    "dracin_gender",
                    "--audio", audio_arg,
                    "--gender_mode", gender_mode,
                    "--outdir", ".",           # output langsung ke workdir
                    "--top_n", str(top_n),
                ]

                if gender_mode == "reference":
                    male_ref_wav   = make_bank(male_ref_in,   "male")
                    female_ref_wav = make_bank(female_ref_in, "female")

                    sys.argv += [
                        "--male_ref",   Path(male_ref_wav).name,
                        "--female_ref", Path(female_ref_wav).name,
                        # mode reference TIDAK pakai --min_vote/min_len_sec
                    ]

                elif gender_mode in ("hf_svm", "wav2vec2"):
                    # dua mode ini butuh threshold per-segmen
                    sys.argv += [
                        "--min_vote",    str(min_vote),
                        "--min_len_sec", str(min_len_sec),
                    ]

                else:
                    # kalau user kirim mode aneh
                    raise ValueError(f"Unsupported gender_mode: {gender_mode}")

                if hf_token:
                    sys.argv += ["--hf_token", hf_token]
                if use_gpu:
                    sys.argv += ["--use_gpu"]

                # ------------ jalankan dracin_gender di workdir ------------
                old_cwd = os.getcwd()
                os.chdir(workdir)
                try:
                    try:
                        gender_main()
                    except SystemExit as e:
                        # argparse di dracin_gender bisa raise SystemExit(code=2) kalau argumen salah
                        return {
                            "success": False,
                            "error": f"dracin_gender exited with code {getattr(e, 'code', None)}",
                        }
                finally:
                    os.chdir(old_cwd)
                    sys.argv = argv_backup

                # ------------ ambil output yg baru dibuat ------------
                spk_raw = latest("*_speaker_timeline_raw.json")
                gen_tl  = latest("*_gender_timeline.json")

                if not spk_raw or not gen_tl:
                    return {
                        "success": False,
                        "error": "dracin_gender selesai tapi file timeline tidak ketemu",
                    }

                spk_raw = bring_to_root(spk_raw)
                gen_tl  = bring_to_root(gen_tl)

                data = {
                    "speaker_timeline_raw": str(spk_raw),
                    "gender_timeline":      str(gen_tl),
                }

                return {"success": True, "data": data}

            except BaseException as e:
                return {
                    "success": False,
                    "error": f"{e}\n{traceback.format_exc()}",
                }

        # jalankan blocking task() di thread pool
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(self.executor, task)

        if not isinstance(result, dict):
            return {
                "success": False,
                "error": f"Engine returned invalid result: {result!r}",
            }

        return result
