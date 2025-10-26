# backend/core/diarization.py

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import sys
import traceback
import shutil
import json

# ──────────────────────────────────────────────────────────────────────────────
# PYTHONPATH bootstrap: pastikan modul di repo root (selevel "backend/") bisa di-import
# ──────────────────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class DiarizationEngine:
    """Menjalankan diarization persis alur GUI lama:
       - Male/Female Reference boleh folder (dikompilasi jadi bank .wav) atau file langsung
       - Semua output (segments/speakers/SRT) dipastikan berada di root workdir session
       - Selalu mengembalikan dict: {success: bool, data|error}
    """

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1)

    async def process(self, session, config: dict, progress_callback=None) -> dict:
        """SELALU return dict:
           - sukses: {"success": True,  "data": {"segjson": "<path>", "spkjson": "<path>", "srt": "<path>|None"}}
           - gagal : {"success": False, "error": "<pesan>"}
        """
        workdir = Path(session.workdir)
        workdir.mkdir(parents=True, exist_ok=True)

        def task():
            try:
                # import ditunda agar PYTHONPATH di atas sudah aktif
                from dracin_gender import main as gender_main
                import torch
                import torchaudio

                # ── helper ───────────────────────────────────────────────────────
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
                    """Bangun <label>_bank.wav (16k mono) di workdir dari folder/file."""
                    src = Path(os.path.expandvars(ref_input)).expanduser()
                    files = list_audio(src)
                    if not files:
                        raise FileNotFoundError(f"Tidak ada file audio di: {src}")

                    chunks = []
                    meta_samples = []
                    for f in files:
                        wav, sr = torchaudio.load(str(f))
                        if wav.dim() == 2 and wav.size(0) > 1:
                            wav = wav.mean(dim=0, keepdim=True)  # mono
                        if sr != 16000:
                            wav = torchaudio.functional.resample(wav, sr, 16000)
                        chunks.append(wav)
                        meta_samples.append(int(wav.shape[-1]))

                    bank = torch.cat(chunks, dim=-1)  # concat time
                    out_wav = workdir / f"{label}_bank.wav"
                    torchaudio.save(str(out_wav), bank, 16000)

                    cache = {
                        "sources": [str(f) for f in files],
                        "sr": 16000,
                        "samples_per_chunk": meta_samples,
                    }
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
                    if p and p.parent != workdir:
                        dst = workdir / p.name
                        try:
                            shutil.move(str(p), str(dst))
                        except Exception:
                            dst = p  # fallback
                        return dst
                    return p

                # ── siapkan referensi (folder -> bank; file -> tetap diproses sebagai bank 1 file) ──
                male_ref_in = config["male_ref"]
                female_ref_in = config["female_ref"]
                male_ref_wav = make_bank(male_ref_in, "male")
                female_ref_wav = make_bank(female_ref_in, "female")

                # ── susun argv untuk dracin_gender (pakai underscore, outdir='.') ──
                audio_arg = Path(session.wav_16k).name
                male_ref_arg = Path(male_ref_wav).name
                female_ref_arg = Path(female_ref_wav).name

                top_n = int(config.get("top_n", 6))
                hf_token = (config.get("hf_token") or "").strip()
                use_gpu = bool(config.get("use_gpu", True))

                argv_backup = sys.argv[:]
                sys.argv = [
                    "dracin_gender",
                    "--audio",
                    audio_arg,
                    "--male_ref",
                    male_ref_arg,
                    "--female_ref",
                    female_ref_arg,
                    "--outdir",
                    ".",  # sangat penting: tulis ke CWD (workdir)
                    "--top_n",
                    str(top_n),
                ]
                if hf_token:
                    sys.argv += ["--hf_token", hf_token]
                if use_gpu:
                    sys.argv += ["--use_gpu"]

                # ── jalankan di CWD = workdir ───────────────────────────────────
                old_cwd = os.getcwd()
                os.chdir(workdir)
                try:
                    try:
                        gender_main()  # TANPA argumen; argparse baca sys.argv
                    except SystemExit as e:
                        # argparse exit (kode 2 jika argumen tidak valid)
                        return {
                            "success": False,
                            "error": f"dracin_gender exited with code {getattr(e, 'code', None)}",
                        }
                finally:
                    os.chdir(old_cwd)
                    sys.argv = argv_backup

                # ── ambil output terbaru (JSON + SRT) dan pastikan di root workdir ──
                seg = latest("*_gender_*_segments.json")
                spk = latest("*_gender_*_speakers.json")
                srt = latest("*_gender_*.srt")  # opsional

                if not seg or not spk:
                    return {
                        "success": False,
                        "error": "Diarization selesai tapi file output tidak ditemukan.",
                    }

                seg = bring_to_root(seg)
                spk = bring_to_root(spk)
                srt = bring_to_root(srt) if srt else None

                data = {"segjson": str(seg), "spkjson": str(spk)}
                if srt:
                    data["srt"] = str(srt)
                return {"success": True, "data": data}

            except BaseException as e:
                return {"success": False, "error": f"{e}\n{traceback.format_exc()}"}

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(self.executor, task)

        # guard: jangan pernah return None
        if not isinstance(result, dict):
            return {"success": False, "error": f"Engine returned invalid result: {result!r}"}
        return result
