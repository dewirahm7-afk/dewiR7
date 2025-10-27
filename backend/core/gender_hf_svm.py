from __future__ import annotations
import json, joblib, numpy as np, torch, torchaudio
from pathlib import Path
from typing import List, Tuple, Dict
import pandas as pd
from speechbrain.pretrained import EncoderClassifier

# Lokasi model yang sudah kamu taruh:
# backend/models/config.json
# backend/models/scaler.joblib
# backend/models/svm_model.joblib
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

_ENCODER = None        # ECAPA encoder (SpeechBrain)
_SCALER = None         # pipeline preprocessing (ColumnTransformer + scaler)
_SVM = None            # classifier SVM
_LABELS = None         # ex: ["female","male"]


def _device(dev="auto"):
    if dev == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return dev


def _lazy_load(dev="auto") -> str:
    """
    Load semua komponen sekali:
    - ECAPA encoder (speechbrain)
    - scaler.joblib  (preprocessing pipeline)
    - svm_model.joblib (SVM classifier)
    - labels dari config.json
    """
    global _ENCODER, _SCALER, _SVM, _LABELS

    dev = _device(dev)

    # 1. ECAPA embedder
    if _ENCODER is None:
        _ENCODER = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": dev},
        )

    # 2. scaler, svm, labels
    if _SCALER is None:
        _SCALER = joblib.load(MODEL_DIR / "scaler.joblib")

    if _SVM is None:
        _SVM = joblib.load(MODEL_DIR / "svm_model.joblib")

    if _LABELS is None:
        cfg_path = MODEL_DIR / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        _LABELS = cfg.get("labels", ["female", "male"])

    return dev


def _load_wav_16k(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    # merge stereo -> mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)
    # resample -> 16k
    if sr != 16000:
        wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, 16000).squeeze(0)
    return wav.contiguous()


def _slice(wav: torch.Tensor, t0: float, t1: float, sr: int = 16000) -> torch.Tensor:
    i0 = max(0, int(round(t0 * sr)))
    i1 = min(wav.numel(), int(round(t1 * sr)))
    if i1 <= i0:
        i1 = min(wav.numel(), i0 + 1)
    return wav[i0:i1]


def _embed(seg: torch.Tensor, dev: str) -> np.ndarray:
    """
    Ambil embedding ECAPA dari potongan suara -> np.ndarray[d] (float32)
    Biasanya d ~ 192 dimensi.
    """
    x = seg.unsqueeze(0).to(dev)          # [1, T]
    with torch.no_grad():
        e = _ENCODER.encode_batch(x)      # [1,1,D]
    return e.squeeze().cpu().numpy().astype("float32")


def _predict(emb: np.ndarray) -> str:
    """
    Satu embedding ECAPA -> label "male"/"female".

    Trik penting:
    - scaler.joblib (ColumnTransformer/Pipeline) EXPECT kolom named:
      "0_speechbrain_embedding", "1_speechbrain_embedding", ..., dst.
    - Jadi kita bikin DataFrame 1 baris dengan nama kolom seperti itu.
    - Kita isi kolom i_speechbrain_embedding = emb[i].
    """

    global _SCALER, _SVM, _LABELS

    dim = emb.shape[0]
    feat_names = [f"{i}_speechbrain_embedding" for i in range(dim)]

    row_vals = {feat_names[i]: float(emb[i]) for i in range(dim)}

    # DataFrame 1 baris, kolom persis seperti yang diharapkan ColumnTransformer
    X_df = pd.DataFrame([row_vals], columns=feat_names)

    # jalankan preprocessing
    z = _SCALER.transform(X_df)  # -> array fit buat SVM

    # klasifikasi
    idx_cls = int(_SVM.predict(z)[0])  # 0/1
    return _LABELS[idx_cls]            # contoh: "female"/"male"


def classify_speaker_gender(
    wav_path: Path,
    speaker_segments: List[Tuple[float, float]],
    *,
    device: str = "auto",
    max_samples: int = 1,
    min_len_sec: float = 0.3,
    min_vote: float = 0.7,
) -> Dict[str, object]:
    """
    Majority vote dari beberapa segmen -> "Male"/"Female"/"Unknown"
    - wav_path: path WAV 16k mono hasil ekstrak vokal film
    - speaker_segments: [(start_sec, end_sec), ...] semua segmen speaker ini
    - max_samples: ambil beberapa segmen TERPANJANG buat voting
    - min_len_sec: skip segmen super pendek (teriak 0.2s dsb)
    - min_vote: minimal proporsi mayoritas agar fix "Male"/"Female"
    """

    dev = _lazy_load(device)
    wav = _load_wav_16k(Path(wav_path))

    # pilih segmen terpanjang dulu biar representatif
    spans = sorted(
        speaker_segments,
        key=lambda ab: (ab[1] - ab[0]),
        reverse=True,
    )

    votes = []
    for (a, b) in spans:
        dur = b - a
        if dur < min_len_sec:
            continue
        if len(votes) >= max_samples:
            break

        seg = _slice(wav, a, b)
        emb = _embed(seg, dev)        # np.array [D]
        votes.append(_predict(emb))   # "female"/"male"

    male = sum(v == "male" for v in votes)
    female = sum(v == "female" for v in votes)
    n = len(votes)

    if n == 0:
        return {
            "gender": "Unknown",
            "male_votes": 0,
            "female_votes": 0,
            "samples_used": 0,
        }

    # mayoritas
    if male >= female:
        best_label = "male"
        best_count = male
    else:
        best_label = "female"
        best_count = female

    confidence = best_count / n

    if confidence >= min_vote:
        final_gender = "Male" if best_label == "male" else "Female"
    else:
        final_gender = "Unknown"

    return {
        "gender": final_gender,
        "male_votes": male,
        "female_votes": female,
        "samples_used": n,
    }
