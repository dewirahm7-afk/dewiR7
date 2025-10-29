# ------------------------------------------------------------
# dracin_gender.py (3-mode version)
#
# gender_mode:
#   - "reference" : pakai referensi male_ref/female_ref (lama, voting per speaker)
#   - "hf_svm"    : ECAPA+SVM custom kamu, per SEGMENT
#   - "wav2vec2"  : audEERING wav2vec2-large-robust-24-ft-age-gender, per SEGMENT :contentReference[oaicite:6]{index=6}
#
# Output file:
#   <stem>_speaker_timeline_raw.json      # diarization hasil pyannote
#   <stem>_gender_timeline.json           # gender per segmen (tanpa speaker_local)
#   <stem>_speakers_stats.json            # ringkasan mayoritas per speaker_local
#
# NOTE:
# - timeline gender dipakai Auto-Sync V5 → jadi ini sakral, kita gak tebak mayoritas.
# - speaker_stats cuma buat debug/global-link (bisa beda).
# ------------------------------------------------------------

import argparse, os, json
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np

from pyannote.audio import Pipeline
from speechbrain.inference.speaker import EncoderClassifier

# patch torch.load buat pyannote checkpoints (biar gak error weights_only=True)
from torch.serialization import add_safe_globals
import omegaconf
add_safe_globals([
    omegaconf.listconfig.ListConfig,
    torch.torch_version.TorchVersion
])

_original_torch_load = torch.load
def patched_torch_load(f, map_location=None, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(f, map_location=map_location, **kwargs)
torch.load = patched_torch_load

# MODE 2 (hf_svm) helper
from core.gender_hf_svm import classify_speaker_gender as hf_gender_vote

# MODE 3 (wav2vec2) helper
from core.gender_w2v2_agegender import classify_segment_gender_w2v2

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
        if wav_1d.dim() != 1:
            wav_1d = wav_1d.reshape(-1)

        T = wav_1d.numel()
        if T < self.min_samples:
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
        choices=["reference","hf_svm","wav2vec2"],
        default="hf_svm",
        help=(
            "reference  = cosine vs male_ref/female_ref per SPEAKER (lama)"
            ", hf_svm   = ECAPA+SVM per SEGMENT"
            ", wav2vec2 = audEERING wav2vec2 per SEGMENT"
        )
    )

    # argumen KHUSUS mode reference:
    ap.add_argument("--male_ref", default="",
        help="Path contoh suara male (hanya dipakai kalau gender_mode=reference)")
    ap.add_argument("--female_ref", default="",
        help="Path contoh suara female (hanya dipakai kalau gender_mode=reference)")

    # argumen umum / voting / threshold:
    ap.add_argument("--top_n", type=int, default=5,
        help="(reference only) ambil N segmen terpanjang per speaker buat centroid gender")
    ap.add_argument("--min_vote", type=float, default=0.6,
        help="hf_svm & wav2vec2: min confidence utk fix 'Male'/'Female', else 'Unknown'")
    ap.add_argument("--min_len_sec", type=float, default=1.0,
        help="hf_svm & wav2vec2: abaikan segmen < detik ini (teriakan super pendek)")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu")
    print("Device:", device)

    # --- siapkan audio utama (track voice 16k mono) ---
    wav, sr = load_wav(Path(args.audio), target_sr=16000, mono=True)  # [T], sr=16k
    full_dur = float(wav.numel()) / 16000.0

    # --- jalankan diarization (pyannote/speaker-diarization-3.1) ---
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
        pass  # beberapa versi pipeline gak punya .to()

    print("Running diarization…")
    diar = pipeline({"audio": str(args.audio)})

    segs = []
    for turn, _, spk_label in diar.itertracks(yield_label=True):
        t0, t1 = float(turn.start), float(turn.end)
        if t1 > t0:
            segs.append({
                "start": t0,
                "end": t1,
                "speaker": spk_label,   # "SPEAKER_00", ...
            })

    # group per speaker_local (buat ringkasan stats nanti)
    by_spk = defaultdict(list)
    for seg in segs:
        by_spk[seg["speaker"]].append(seg)

    for spk in by_spk:
        by_spk[spk].sort(key=lambda x: (x["end"] - x["start"]), reverse=True)

    # ==================================================
    # GENDER ESTIMATION
    # ==================================================

    # segment_gender_map:
    #   list of { "start":float, "end":float, "gender": "Male"/"Female"/"Unknown", "speaker_local": str }
    segment_gender_map = []

    # speakers (ringkasan per speaker_local)
    speakers = {}

    if args.gender_mode == "reference":
        # ---------- MODE REFERENCE (lama, centroid per speaker) ----------
        embedder = ECAPAEmbedder(device=device, min_sec=1.2, sr=16000)

        # siapkan embedding referensi
        mref_wav, _ = load_wav(Path(args.male_ref), target_sr=16000, mono=True)
        fref_wav, _ = load_wav(Path(args.female_ref), target_sr=16000, mono=True)
        mref_emb = embedder(mref_wav.to(device))
        fref_emb = embedder(fref_wav.to(device))

        # tentukan gender speaker_global dengan cosine sim
        for spk, seg_list in by_spk.items():
            picks = seg_list[: max(1, args.top_n)]
            embs = []
            for seginfo in picks:
                ch = slice_wav(wav, 16000, seginfo["start"], seginfo["end"])
                embs.append(embedder(ch.to(device)))

            if len(embs) == 1:
                spk_emb = embs[0]
            else:
                spk_emb = np.mean(np.stack(embs, 0), axis=0)

            sm = cos_sim(spk_emb, mref_emb)
            sf = cos_sim(spk_emb, fref_emb)
            margin = abs(sm - sf)

            min_confidence = 0.17
            if margin < min_confidence:
                gender_label = "Unknown"
            else:
                gender_label = "Male" if sm >= sf else "Female"

            speakers[spk] = {
                "gender": gender_label,
                "score_m": sm,
                "score_f": sf,
                "margin": margin,
                "mode": "reference",
            }

        # turunkan gender speaker_global itu ke semua potongan speaker tsb
        for seg in segs:
            g = speakers.get(seg["speaker"], {}).get("gender", "Unknown")
            segment_gender_map.append({
                "start": seg["start"],
                "end": seg["end"],
                "gender": g,
                "speaker_local": seg["speaker"],
            })

    elif args.gender_mode == "hf_svm":
        # ---------- MODE HF_SVM (ECAPA+SVM custom kita) ----------
        for seg in segs:
            info = hf_gender_vote(
                wav_path=Path(args.audio),
                speaker_segments=[(seg["start"], seg["end"])],
                device=("cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu"),
                max_samples=1,                 # per segmen langsung
                min_len_sec=args.min_len_sec,  # skip segmen super pendek
                min_vote=args.min_vote,        # min confidence
            )
            segment_gender_map.append({
                "start": seg["start"],
                "end": seg["end"],
                "gender": info["gender"],      # "Male"/"Female"/"Unknown"
                "speaker_local": seg["speaker"],
            })

    elif args.gender_mode == "wav2vec2":
        # ---------- MODE WAV2VEC2 (audEERING age+gender model) ----------
        for seg in segs:
            info = classify_segment_gender_w2v2(
                wav_path=Path(args.audio),
                start_s=seg["start"],
                end_s=seg["end"],
                device=("cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu"),
                min_len_sec=args.min_len_sec,
                min_confidence=args.min_vote,
            )
            segment_gender_map.append({
                "start": seg["start"],
                "end": seg["end"],
                "gender": info["gender"],      # "Male"/"Female"/"Unknown"
                "speaker_local": seg["speaker"],
            })

    else:
        raise RuntimeError(f"gender_mode '{args.gender_mode}' tidak dikenal")

    # -------------------------------------------------
    # Build speakers[] ringkasan mayoritas per speaker_local
    # (ini hanya buat debug / global link UI)
    # -------------------------------------------------
    tmp_votes = defaultdict(lambda: {"male":0.0, "female":0.0, "unknown":0.0, "dur":0.0})
    for item in segment_gender_map:
        spk = item["speaker_local"]
        g   = (item["gender"] or "Unknown").lower()
        dur = max(item["end"] - item["start"], 0.0)

        tmp_votes[spk]["dur"] += dur
        if g == "male":
            tmp_votes[spk]["male"] += dur
        elif g == "female":
            tmp_votes[spk]["female"] += dur
        else:
            tmp_votes[spk]["unknown"] += dur

    for spk, stat in tmp_votes.items():
        m = stat["male"]
        f = stat["female"]
        u = stat["unknown"]

        if m == 0 and f == 0:
            final_g = "Unknown"
        elif m >= f and m >= u:
            final_g = "Male"
        elif f >= m and f >= u:
            final_g = "Female"
        else:
            final_g = "Unknown"

        speakers[spk] = {
            "gender": final_g,
            "male_dur": m,
            "female_dur": f,
            "unknown_dur": u,
            "mode": (
                "reference" if args.gender_mode=="reference"
                else ("hf_svm_segmentwise" if args.gender_mode=="hf_svm"
                      else "wav2vec2_segmentwise")
            ),
        }

    # -------------------------------------------------
    # OUTPUT SECTION
    # -------------------------------------------------

    stem = Path(args.audio).stem
    speaker_raw_path = outdir / f"{stem}_speaker_timeline_raw.json"
    gender_tl_path   = outdir / f"{stem}_gender_timeline.json"
    speakers_path    = outdir / f"{stem}_speakers_stats.json"

    # 1) speaker timeline mentah
    with speaker_raw_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "segments": [
                    {
                        "start": seg["start"],
                        "end": seg["end"],
                        "speaker": seg["speaker"],  # "SPEAKER_00"
                    }
                    for seg in segs
                ],
                "duration": full_dur,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 2) gender timeline FINAL buat Auto Sync V5
    #    (tanpa speaker_local, pure {start,end,gender})
    gender_segments_export = []
    for item in segment_gender_map:
        gender_segments_export.append({
            "start":  item["start"],
            "end":    item["end"],
            "gender": item["gender"],
        })

    with gender_tl_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "segments": gender_segments_export,
                "duration": full_dur,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 3) speaker summary debug
    with speakers_path.open("w", encoding="utf-8") as f:
        json.dump(
            speakers,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("RAW_SPK_TIMELINE:", speaker_raw_path)
    print("GENDER_TIMELINE :", gender_tl_path)
    print("SPEAKERS_STATS  :", speakers_path)


if __name__ == "__main__":
    main()
