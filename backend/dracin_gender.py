# dracin_gender.py
# ------------------------------------------------------------
# Diarization (pyannote 3.x) + Gender classification pakai
# SpeechBrain ECAPA embeddings (robust, Torch 2.x friendly).
# - CUDA otomatis (--use_gpu)
# - Pad segmen pendek => aman BatchNorm
# - Output: *_gender_YYYYMMDD_HHMMSS_rand.srt / _segments.json / _speakers.json
# ------------------------------------------------------------

import argparse, os, json, random
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np

from pyannote.audio import Pipeline
from speechbrain.inference.speaker import EncoderClassifier
from torch.serialization import add_safe_globals

# Allowlist classes used in pyannote checkpoints
import omegaconf
add_safe_globals([omegaconf.listconfig.ListConfig, torch.torch_version.TorchVersion])

_original_torch_load = torch.load
def patched_torch_load(f, map_location=None, **kwargs):
    # pastikan weights_only=False dipaksa
    kwargs["weights_only"] = False
    return _original_torch_load(f, map_location=map_location, **kwargs)

torch.load = patched_torch_load

# ---------- Utils ----------
def hhmmssms(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)

def load_wav(path: Path, target_sr: int = 16000, mono=True):
    wav, sr = torchaudio.load(str(path))
    if mono and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
        sr = target_sr
    return wav.squeeze(0), sr  # [T], sr

def slice_wav(wav_1d: torch.Tensor, sr: int, t0: float, t1: float):
    i0 = max(0, int(round(t0 * sr)))
    i1 = min(wav_1d.numel(), int(round(t1 * sr)))
    if i1 <= i0:
        i1 = min(wav_1d.numel(), i0 + 1)
    return wav_1d[i0:i1]


# ---------- ECAPA Embedder (SpeechBrain) ----------
class ECAPAEmbedder:
    def __init__(self, device: torch.device, min_sec: float = 1.2, sr: int = 16000):
        self.device = device
        self.min_samples = int(min_sec * sr)
        # Load encoder
        # Catatan: SpeechBrain 1.0 memindah modul ke `speechbrain.inference.*`
        self.enc = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": str(device)}
        )
        self.sr = sr

    @torch.inference_mode()
    def __call__(self, wav_1d: torch.Tensor) -> np.ndarray:
        if wav_1d.dim() != 1:
            wav_1d = wav_1d.view(-1)

        T = wav_1d.numel()
        if T < self.min_samples:
            # pad zero (lebih aman untuk reflect saat T terlalu pendek)
            pad_total = self.min_samples - T
            left = pad_total // 2
            right = pad_total - left
            wav_1d = F.pad(wav_1d.unsqueeze(0), (left, right), mode="constant", value=0.0).squeeze(0)

        # encode_batch butuh shape [B, T]
        x = wav_1d.float().unsqueeze(0).to(self.device)
        emb = self.enc.encode_batch(x).squeeze(0).squeeze(0)  # [D]
        return emb.detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="Path WAV 16k mono (atau akan di-resample dulu)")
    ap.add_argument("--male_ref", required=True, help="Contoh suara Male (wav)")
    ap.add_argument("--female_ref", required=True, help="Contoh suara Female (wav)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--top_n", type=int, default=3, help="Ambil N segmen terpanjang per speaker")
    ap.add_argument("--hf_token", default=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_API_TOKEN"))
    ap.add_argument("--use_gpu", action="store_true", help="Pakai CUDA kalau tersedia")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu")
    print("Device:", device)

    # Load (dan pastikan) audio full 16k mono
    wav, sr = load_wav(Path(args.audio), target_sr=16000, mono=True)  # [T], 16k
    full_dur = float(wav.numel()) / 16000.0

    # ---- Diarization ----
    if not args.hf_token:
        raise RuntimeError("HuggingFace token diperlukan (--hf_token) untuk pyannote/speaker-diarization-3.1")

    print("Loading diarization pipeline...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=args.hf_token)
    try:
        pipeline.to(device)
    except Exception:
        # beberapa build tidak mendukung .to(device); tetap jalan di CPU
        pass

    print("Running diarization…")
    diar = pipeline({"audio": str(args.audio)})

    segs = []
    for turn, _, speaker in diar.itertracks(yield_label=True):
        t0, t1 = float(turn.start), float(turn.end)
        if t1 > t0:
            segs.append({"start": t0, "end": t1, "speaker": speaker})

    # ---- Embedding engine (SpeechBrain ECAPA) ----
    embedder = ECAPAEmbedder(device=device, min_sec=1.2, sr=16000)

    mref_wav, _ = load_wav(Path(args.male_ref), target_sr=16000, mono=True)
    fref_wav, _ = load_wav(Path(args.female_ref), target_sr=16000, mono=True)
    mref_emb = embedder(mref_wav.to(device))
    fref_emb = embedder(fref_wav.to(device))

    # ---- Agregasi per speaker (top-N segmen terpanjang) ----
    from collections import defaultdict
    by_spk = defaultdict(list)
    for s in segs:
        by_spk[s["speaker"]].append(s)
    for spk in by_spk:
        by_spk[spk].sort(key=lambda x: (x["end"] - x["start"]), reverse=True)

    speakers = {}
    for spk, lst in by_spk.items():
        picks = lst[: max(1, args.top_n)]
        embs = []
        for s in picks:
            ch = slice_wav(wav, 16000, s["start"], s["end"])
            embs.append(embedder(ch.to(device)))
        spk_emb = embs[0] if len(embs) == 1 else np.mean(np.stack(embs, 0), 0)
        
        # === GANTI BAGIAN INI ===
        sm = cos_sim(spk_emb, mref_emb)
        sf = cos_sim(spk_emb, fref_emb)
        margin = abs(sm - sf)
        
        # Improved gender classification dengan confidence yang lebih tinggi
        min_confidence = 0.17  # Dari 0.10 menjadi 0.15 (lebih ketat)
        if margin < min_confidence:
            gender = "Unknown"
        else:
            gender = "Male" if sm >= sf else "Female"
        
        speakers[spk] = {"gender": gender, "score_m": sm, "score_f": sf, "margin": margin}

    # ---- Tulis outputs ----
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rand4 = random.randint(1000, 9999)
    stem = Path(args.audio).stem

    srt_path = outdir / f"{stem}_gender_{stamp}_{rand4}.srt"
    with srt_path.open("w", encoding="utf-8") as f:
        for i, s in enumerate(segs, start=1):
            spk = s["speaker"]; g = speakers.get(spk, {}).get("gender", "Unknown")
            f.write(f"{i}\n{hhmmssms(s['start'])} --> {hhmmssms(s['end'])}\n[{g}] (Speaker {spk})\n\n")

    seg_json = outdir / f"{stem}_gender_{stamp}_{rand4}_segments.json"
    with seg_json.open("w", encoding="utf-8") as f:
        seg_dump = []
        for s in segs:
            spk = s["speaker"]; g = speakers.get(spk, {}).get("gender", "Unknown")
            seg_dump.append({"start": s["start"], "end": s["end"], "speaker": spk, "gender": g})
        json.dump({"segments": seg_dump, "duration": full_dur}, f, ensure_ascii=False, indent=2)

    spk_json = outdir / f"{stem}_gender_{stamp}_{rand4}_speakers.json"
    with spk_json.open("w", encoding="utf-8") as f:
        json.dump({"speakers": speakers}, f, ensure_ascii=False, indent=2)

    print("SRT :", srt_path)
    print("SEGS:", seg_json)
    print("SPKS:", spk_json)


if __name__ == "__main__":
    main()