# backend/core/gender_w2v2_agegender.py
#
# Mode gender "wav2vec2":
#   - pakai model lokal audEERING "wav2vec2-large-robust-24-ft-age-gender"
#   - kita simpan file2 model di:
#         dracindub_web/frontend/model/
#     (config.json, pytorch_model.bin / model.safetensors, preprocessor_config.json, vocab.json)
#
# Fungsi utama yg dipakai dracin_gender.py:
#   classify_segment_gender_w2v2(...)
#
# Dia juga expose classify_gender_w2v2(...), yg bisa voting multi segmen.
#
# Return format SELALU:
# {
#   "gender": "Male" / "Female" / "Unknown",
#   "male_votes": int,
#   "female_votes": int,
#   "samples_used": int
# }
#
# Catatan penting:
# - "wav2vec2" ini output label usia+gender ("female_adult", dsb). Kita map jadi Male/Female.
# - kalau confidence mayoritas < min_confidence => "Unknown"
# - min_len_sec skip potongan super pendek
#
# Wrapper classify_segment_gender_w2v2() mendukung 2 gaya:
#   1) start_s=..., end_s=... (1 segmen)
#   2) speaker_segments=[(s,e), ...]       (beberapa segmen)
#
# supaya kompatibel sama dracin_gender.py branch wav2vec2.
# ---------------------------------------------------------------------

from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForAudioClassification

# lokasi model lokal:
#   backend/core/gender_w2v2_agegender.py
#   parents[0] = core
#   parents[1] = backend
#   parents[2] = dracindub_web
MODEL_W2V2_DIR = Path(__file__).resolve().parents[2] / "frontend" / "model"

_W2V2_PROCESSOR = None
_W2V2_MODEL = None
_W2V2_DEVICE = None


def _pick_device(dev: str = "auto") -> str:
    if dev == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return dev


def _lazy_load_w2v2(device: str = "auto"):
    """
    Lazy-load processor & model sekali saja dari MODEL_W2V2_DIR.
    """
    global _W2V2_PROCESSOR, _W2V2_MODEL, _W2V2_DEVICE

    dev = _pick_device(device)

    if _W2V2_PROCESSOR is None or _W2V2_MODEL is None:
        if not MODEL_W2V2_DIR.exists():
            raise FileNotFoundError(
                f"Model folder not found: {MODEL_W2V2_DIR}\n"
                "Pastikan ada: config.json, pytorch_model.bin / model.safetensors, "
                "preprocessor_config.json, vocab.json"
            )

        _W2V2_PROCESSOR = AutoProcessor.from_pretrained(str(MODEL_W2V2_DIR))
        _W2V2_MODEL = AutoModelForAudioClassification.from_pretrained(
            str(MODEL_W2V2_DIR)
        )

    # pindahkan model ke device target (kalau berubah)
    if _W2V2_DEVICE != dev:
        _W2V2_MODEL.to(dev)
        _W2V2_DEVICE = dev

    return _W2V2_PROCESSOR, _W2V2_MODEL, dev


def _load_wav_16k(path: Path) -> torch.Tensor:
    """
    Load audio -> mono 16kHz tensor [T].
    """
    wav, sr = torchaudio.load(str(path))

    # stereo -> mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)

    # resample ke 16k
    if sr != 16000:
        wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, 16000).squeeze(0)

    return wav.contiguous()


def _slice_wave(wav: torch.Tensor, t0: float, t1: float, sr: int = 16000) -> torch.Tensor:
    """
    Ambil potongan [t0, t1) detik dari wav16k → Tensor [Tseg].
    """
    i0 = max(0, int(round(t0 * sr)))
    i1 = min(wav.numel(), int(round(t1 * sr)))
    if i1 <= i0:
        i1 = min(wav.numel(), i0 + 1)
    return wav[i0:i1]


@torch.inference_mode()
def _predict_one(
    seg_wave: torch.Tensor,
    processor,
    model,
    dev: str,
) -> str:
    """
    1 potongan audio -> label model HF (misal 'female_adult').
    Kita normalisasi ke 'Male'/'Female'/'Unknown'.
    """
    inputs = processor(
        seg_wave.numpy(),
        sampling_rate=16000,
        return_tensors="pt",
    )
    # kirim tensor ke device
    for k in inputs:
        inputs[k] = inputs[k].to(dev)

    out = model(**inputs)
    logits = out.logits  # [1, num_labels]
    probs = torch.softmax(logits, dim=-1)[0]
    idx = int(torch.argmax(probs).item())

    try:
        raw_label = model.config.id2label[idx]
    except Exception:
        raw_label = str(idx)

    rl = raw_label.lower()
    if "female" in rl:
        return "Female"
    if "male" in rl:
        return "Male"
    return "Unknown"


def _majority(votes: List[str], min_vote: float) -> Dict[str, object]:
    """
    Voting mayoritas -> gender final + statistik.
    """
    male_count = sum(v == "Male" for v in votes)
    female_count = sum(v == "Female" for v in votes)
    n = len(votes)

    if n == 0:
        return {
            "gender": "Unknown",
            "male_votes": 0,
            "female_votes": 0,
            "samples_used": 0,
        }

    if male_count >= female_count:
        maj_label = "Male"
        maj_count = male_count
    else:
        maj_label = "Female"
        maj_count = female_count

    confidence = maj_count / n
    if confidence < min_vote:
        maj_label = "Unknown"

    return {
        "gender": maj_label,
        "male_votes": male_count,
        "female_votes": female_count,
        "samples_used": n,
    }


def classify_gender_w2v2(
    wav_path: Path,
    speaker_segments: List[Tuple[float, float]],
    *,
    device: str = "auto",
    max_samples: int = 5,
    min_len_sec: float = 0.3,
    min_vote: float = 0.7,
) -> Dict[str, object]:
    """
    Analisa beberapa segmen (speaker_segments) pakai majority vote.
    Cocok kalau kamu mau kumpulin beberapa potongan 1 speaker,
    lalu tentukan gender mayoritas speaker itu.
    """
    processor, model, dev = _lazy_load_w2v2(device)
    wav = _load_wav_16k(Path(wav_path))

    # pilih segmen terpanjang dulu -> representatif
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

        clip = _slice_wave(wav, a, b)
        g = _predict_one(clip, processor, model, dev)  # "Male"/"Female"/"Unknown"
        votes.append(g)

    return _majority(votes, min_vote)


def classify_segment_gender_w2v2(
    *,
    wav_path: Path,
    start_s: Optional[float] = None,
    end_s: Optional[float] = None,
    speaker_segments: Optional[List[Tuple[float, float]]] = None,
    device: str = "auto",
    min_len_sec: float = 0.3,
    min_confidence: float = 0.7,
    max_samples: int = 5,
) -> Dict[str, object]:
    """
    Wrapper kompatibel buat dracin_gender.py MODE 'wav2vec2'.

    Dua cara pakai:
      1) start_s=..., end_s=... (single segmen)
      2) speaker_segments=[(s,e), ...] (multi segmen untuk voting)

    Kita panggil classify_gender_w2v2(...) di bawah.
    """

    if speaker_segments is not None:
        spans = speaker_segments
    elif start_s is not None and end_s is not None:
        spans = [(float(start_s), float(end_s))]
    else:
        raise ValueError("Must provide either start_s+end_s or speaker_segments")

    return classify_gender_w2v2(
        wav_path=wav_path,
        speaker_segments=spans,
        device=device,
        max_samples=max_samples,
        min_len_sec=min_len_sec,
        min_vote=min_confidence,
    )
