# ------------------------------------------------------------
# dracin_gender.py
#
# Diarization (pyannote 3.x)
# + Klasifikasi gender per speaker dengan 2 MODE:
#
#   1. reference  -> pakai bank male_ref / female_ref (cara lama)
#   2. hf_svm     -> pakai model pretrained griko/gender_cls_svm_ecapa_voxceleb
#                    lewat core.gender_hf_svm (tanpa sample manual)
#
# Output:
#   *_gender_YYYYMMDD_HHMMSS_rand.srt
#   *_segments.json
#   *_speakers.json
#
# Catatan:
#  - hf_svm pakai voting beberapa segmen terpanjang speaker
#    + min_len_sec (abaikan segmen super pendek)
#    + min_vote    (kalau mayoritas lemah -> "Unknown")
#
#  - reference pakai cos_sim ke male_ref / female_ref
#    masih support supaya mode lama tidak hilang.
#
# ------------------------------------------------------------

import argparse, os, json, random
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np

from pyannote.audio import Pipeline

# ECAPA buat mode "reference"
from speechbrain.inference.speaker import EncoderClassifier

# patch torch.load buat pyannote checkpoints (biar gak error weights_only)
from torch.serialization import add_safe_globals
import omegaconf
add_safe_globals([omegaconf.listconfig.ListConfig, torch.torch_version.TorchVersion])

_original_torch_load = torch.load
def patched_torch_load(f, map_location=None, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(f, map_location=map_location, **kwargs)

torch.load = patched_torch_load

# MODE BARU (hf_svm)
# -> ini fungsi yang kita tulis di backend/core/gender_hf_svm.py
from core.gender_hf_svm import classify_speaker_gender as hf_gender_vote


# ---------- Utils umum ----------
def hhmmssms(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)

def load_wav(path: Path, target_sr: int = 16000, mono: bool = True):
    """
    Load audio jadi mono 16kHz.
    return: (wav_1d[T], sr=16000)
    """
    wav, sr = torchaudio.load(str(path))
    if mono and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
        sr = target_sr
    return wav.squeeze(0), sr  # [T], 16000

def slice_wav(wav_1d: torch.Tensor, sr: int, t0: float, t1: float):
    """
    Ambil potongan [t0, t1) detik dari waveform 16k mono.
    """
    i0 = max(0, int(round(t0 * sr)))
    i1 = min(wav_1d.numel(), int(round(t1 * sr)))
    if i1 <= i0:
        i1 = min(wav_1d.numel(), i0 + 1)
    return wav_1d[i0:i1]


# ---------- ECAPA wrapper buat MODE "reference" ----------
class ECAPAEmbedder:
    """
    Ambil embedding speaker (ECAPA-TDNN dari SpeechBrain),
    dengan auto-padding segmen pendek supaya BatchNorm aman.
    """
    def __init__(self, device: torch.device, min_sec: float = 1.2, sr: int = 16000):
        self.device = device
        self.min_samples = int(min_sec * sr)
        self.sr = sr

        self.enc = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": str(device)}
        )

    @torch.inference_mode()
    def __call__(self, wav_1d: torch.Tensor) -> np.ndarray:
        # wav_1d: torch.Tensor [T] pada 16k
        if wav_1d.dim() != 1:
            wav_1d = wav_1d.reshape(-1)

        T = wav_1d.numel()
        if T < self.min_samples:
            # zero pad biar panjang minimal
            pad_total = self.min_samples - T
            left = pad_total // 2
            right = pad_total - left
            wav_1d = F.pad(
                wav_1d.unsqueeze(0),
                (left, right),
                mode="constant",
                value=0.0
            ).squeeze(0)

        x = wav_1d.float().unsqueeze(0).to(self.device)  # [1, T]
        emb = self.enc.encode_batch(x).squeeze(0).squeeze(0)  # [D]
        return emb.detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()

    # --- input wajib ---
    ap.add_argument("--audio", required=True,
                    help="Path WAV vokal 16k mono (kalau bukan 16k mono nanti di-resample)")
    ap.add_argument("--outdir", required=True)

    # --- diarization ---
    ap.add_argument("--hf_token",
        default=os.getenv("HF_TOKEN")
             or os.getenv("HUGGINGFACE_TOKEN")
             or os.getenv("HF_API_TOKEN"),
        help="HuggingFace token untuk pyannote/speaker-diarization-3.1")
    ap.add_argument("--use_gpu", action="store_true",
        help="pakai CUDA kalau ada")

    # --- mode gender ---
    ap.add_argument("--gender_mode",
        choices=["reference","hf_svm"],
        default="hf_svm",
        help="reference = pakai bank male/female, hf_svm = pakai model pretrained HF")

    # argumen KHUSUS mode reference:
    ap.add_argument("--male_ref", default="",
        help="Folder/file contoh suara male (only used if gender_mode=reference)")
    ap.add_argument("--female_ref", default="",
        help="Folder/file contoh suara female (only used if gender_mode=reference)")

    # argumen umum / voting:
    ap.add_argument("--top_n", type=int, default=5,
        help="ambil N segmen terpanjang per speaker untuk menilai gender")

    # argumen KHUSUS mode hf_svm:
    ap.add_argument("--min_vote", type=float, default=0.6,
        help="hf_svm: kalau majority < min_vote => 'Unknown'")
    ap.add_argument("--min_len_sec", type=float, default=1.0,
        help="hf_svm: abaikan segmen < detik ini (teriakan sangat pendek, interupsi kecil)")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # pilih device
    device = torch.device("cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu")
    print("Device:", device)

    # --- siapkan audio utama (vokal film sudah dipisah) ---
    wav, sr = load_wav(Path(args.audio), target_sr=16000, mono=True)  # [T], sr=16k
    full_dur = float(wav.numel()) / 16000.0

    # --- jalankan diarization ---
    if not args.hf_token:
        raise RuntimeError("HuggingFace token diperlukan (--hf_token) untuk pyannote/speaker-diarization-3.1")

    print("Loading diarization pipeline...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=args.hf_token
    )
    try:
        pipeline.to(device)
    except Exception:
        # beberapa versi pyannote pipeline gak punya .to() -- aman
        pass

    print("Running diarization…")
    diar = pipeline({"audio": str(args.audio)})

    # kumpulkan semua segmen: [{start, end, speaker}, ...]
    segs = []
    for turn, _, spk_label in diar.itertracks(yield_label=True):
        t0, t1 = float(turn.start), float(turn.end)
        if t1 > t0:
            segs.append({
                "start": t0,
                "end": t1,
                "speaker": spk_label,
            })

    # group per speaker lokal
    by_spk = defaultdict(list)
    for seg in segs:
        by_spk[seg["speaker"]].append(seg)

    # sort segmen tiap speaker (paling panjang dulu)
    for spk in by_spk:
        by_spk[spk].sort(key=lambda x: (x["end"] - x["start"]), reverse=True)

    # -------------------------------------------------
    # BAGIAN INI YANG KAMU TANYA:
    # "setelah punya segs dan by_spk, kita tetapkan gender per speaker"
    #
    # Sekarang jadi cabang:
    #   if gender_mode == "reference": (cara lama, cosine ke male_ref/female_ref)
    #   else:                         (hf_svm, pakai model griko)
    # -------------------------------------------------

    speakers = {}

    if args.gender_mode == "reference":
        # ========== MODE LAMA ==========
        # 1) siapkan embedder ECAPA
        embedder = ECAPAEmbedder(device=device, min_sec=1.2, sr=16000)

        # 2) load male_ref dan female_ref -> embed
        #    NOTE: di versi kamu sebelumnya male_ref/female_ref bisa folder.
        #    Di sini aku anggap file satu. Kalau folder, kamu bisa ambil 1 sample
        #    atau rata-rata semua file di folder tsb.
        mref_wav, _ = load_wav(Path(args.male_ref), target_sr=16000, mono=True)
        fref_wav, _ = load_wav(Path(args.female_ref), target_sr=16000, mono=True)
        mref_emb = embedder(mref_wav.to(device))
        fref_emb = embedder(fref_wav.to(device))

        # 3) loop tiap speaker lokal
        for spk, seg_list in by_spk.items():

            # ambil N segmen terpanjang
            picks = seg_list[: max(1, args.top_n)]
            embs = []
            for seginfo in picks:
                ch = slice_wav(wav, 16000, seginfo["start"], seginfo["end"])
                embs.append(embedder(ch.to(device)))

            # centroid speaker
            if len(embs) == 1:
                spk_emb = embs[0]
            else:
                spk_emb = np.mean(np.stack(embs, 0), axis=0)

            sm = cos_sim(spk_emb, mref_emb)
            sf = cos_sim(spk_emb, fref_emb)
            margin = abs(sm - sf)

            # threshold lama kamu (aku pakai angka kamu yang terakhir naik jadi sedikit ketat)
            min_confidence = 0.17
            if margin < min_confidence:
                gender = "Unknown"
            else:
                gender = "Male" if sm >= sf else "Female"

            speakers[spk] = {
                "gender": gender,
                "score_m": sm,
                "score_f": sf,
                "margin": margin,
                "mode": "reference",
            }

    else:
        # ========== MODE BARU: HF SVM ==========
        # kita pakai model griko/gender_cls_svm_ecapa_voxceleb
        # via core.gender_hf_svm.hf_gender_vote
        #
        # voting multi-segmen + fallback Unknown
        #
        for spk, seg_list in by_spk.items():
            # ambil N segmen terpanjang dari speaker ini
            top_spans = [
                (seginfo["start"], seginfo["end"])
                for seginfo in seg_list[: max(1, args.top_n)]
            ]

            info = hf_gender_vote(
                wav_path=Path(args.audio),
                speaker_segments=top_spans,
                device=("cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu"),
                max_samples=len(top_spans),
                min_len_sec=args.min_len_sec,
                min_vote=args.min_vote,
            )
            # info: {gender:"Male"/"Female"/"Unknown", male_votes, female_votes, samples_used}

            speakers[spk] = {
                "gender": info["gender"],
                "male_votes": info["male_votes"],
                "female_votes": info["female_votes"],
                "samples_used": info["samples_used"],
                "mode": "hf_svm",
            }

    # -------------------------------------------------
    # Tulis output (.srt, *_segments.json, *_speakers.json)
    # -------------------------------------------------

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rand4 = random.randint(1000, 9999)
    stem = Path(args.audio).stem

    srt_path = outdir / f"{stem}_gender_{stamp}_{rand4}.srt"
    with srt_path.open("w", encoding="utf-8") as f:
        for i, seginfo in enumerate(segs, start=1):
            spk = seginfo["speaker"]
            g = speakers.get(spk, {}).get("gender", "Unknown")
            f.write(
                f"{i}\n"
                f"{hhmmssms(seginfo['start'])} --> {hhmmssms(seginfo['end'])}\n"
                f"[{g}] (Speaker {spk})\n\n"
            )

    seg_json = outdir / f"{stem}_gender_{stamp}_{rand4}_segments.json"
    with seg_json.open("w", encoding="utf-8") as f:
        seg_dump = []
        for seginfo in segs:
            spk = seginfo["speaker"]
            g = speakers.get(spk, {}).get("gender", "Unknown")
            seg_dump.append({
                "start": seginfo["start"],
                "end": seginfo["end"],
                "speaker": spk,
                "gender": g,
            })
        json.dump(
            {"segments": seg_dump, "duration": full_dur},
            f,
            ensure_ascii=False,
            indent=2
        )

    spk_json = outdir / f"{stem}_gender_{stamp}_{rand4}_speakers.json"
    with spk_json.open("w", encoding="utf-8") as f:
        json.dump({"speakers": speakers}, f, ensure_ascii=False, indent=2)

    print("SRT :", srt_path)
    print("SEGS:", seg_json)
    print("SPKS:", spk_json)


if __name__ == "__main__":
    main()