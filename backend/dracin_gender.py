# ------------------------------------------------------------
# dracin_gender.py (REVISED VERSION)
#
# Pipeline:
#   1. Diarization (pyannote 3.x) -> potongan segmen {start,end,speaker_local}
#   2. Gender classification per-segmen langsung dari audio (hf_svm mode)
#
# Output:
#   <stem>_speaker_timeline_raw.json
#       segments: [ {start, end, speaker} ... ]
#
#   <stem>_gender_timeline.json
#       segments: [ {start, end, gender, speaker_local} ... ]
#
# Catatan:
# - DULU: gender ditentukan per speaker_local (by_spk) lalu ditempel ke semua segmen.
#         Itu bikin kasus "speaker_19 = male semua", walau ada cewek.
# - SEKARANG: gender diklasifikasi untuk SETIAP segmen langsung.
#         Jadi tiap baris subtitle nanti bisa punya gender yang akurat per waktu,
#         tanpa ikut rusaknya cluster speaker.
#
# - Masih ada summary per speaker (speakers{}), tapi itu hanya statistik / debug,
#   bukan sumber kebenaran final untuk timeline.
#
# ------------------------------------------------------------

import argparse, os, json
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
# -> ini fungsi dari core/gender_hf_svm.py
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
        help="(mode lama) ambil N segmen terpanjang per speaker untuk voting gender")

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
                "speaker": spk_label,   # <- label speaker lokal pyannote
            })

    # group per speaker lokal (buat ringkasan statistik nanti)
    by_spk = defaultdict(list)
    for seg in segs:
        by_spk[seg["speaker"]].append(seg)

    # sort segmen tiap speaker (paling panjang dulu)
    for spk in by_spk:
        by_spk[spk].sort(key=lambda x: (x["end"] - x["start"]), reverse=True)

    # -------------------------------------------------
    # GENDER ESTIMATION
    #
    # MODE "reference": lama, voting per speaker
    # MODE "hf_svm":    BARU, gender per segmen langsung
    # -------------------------------------------------

    # speakers{} sekarang hanya ringkasan statistik per speaker_local
    speakers = {}

    if args.gender_mode == "reference":
        # ========== MODE LAMA (masih disimpan buat backward compat) ==========

        embedder = ECAPAEmbedder(device=device, min_sec=1.2, sr=16000)

        # load male_ref & female_ref, ambil embedding
        mref_wav, _ = load_wav(Path(args.male_ref), target_sr=16000, mono=True)
        fref_wav, _ = load_wav(Path(args.female_ref), target_sr=16000, mono=True)
        mref_emb = embedder(mref_wav.to(device))
        fref_emb = embedder(fref_wav.to(device))

        for spk, seg_list in by_spk.items():
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

        # Di mode reference lama, kita TIDAK punya prediksi gender per segmen.
        # Kita akan isi gender per segmen pakai speakers[seg.speaker]["gender"].

        segment_gender_map = []
        for seg in segs:
            g = speakers.get(seg["speaker"], {}).get("gender", "Unknown")
            segment_gender_map.append({
                "start": seg["start"],
                "end": seg["end"],
                "gender": g,
                "speaker_local": seg["speaker"],
            })

    else:
        # ========== MODE BARU: HF SVM ==========
        #
        # Perbaikan BESAR:
        #   - Kita klasifikasi gender LANGSUNG PER SEGMEN,
        #     bukan voting sekali per speaker lalu tembak ke semua segmen.
        #
        #   - Setelah itu baru kita bikin summary per speaker_local (optional),
        #     hanya untuk info/debug, bukan sumber kebenaran.
        #

        # 1) Gender PER SEGMEN
        segment_gender_map = []
        for seg in segs:
            this_span = [(seg["start"], seg["end"])]
            info = hf_gender_vote(
                wav_path=Path(args.audio),
                speaker_segments=this_span,
                device=("cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu"),
                max_samples=1,           # kita cuma kasih 1 span ini
                min_len_sec=args.min_len_sec,
                min_vote=args.min_vote,
            )
            # info["gender"] ∈ {"Male","Female","Unknown"}

            segment_gender_map.append({
                "start": seg["start"],
                "end": seg["end"],
                "gender": info["gender"],
            })

        # 2) Summary per speaker_local (opsional)
        #    Ini buat manusia lihat statistik, bukan buat nempel gender global.
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
                "gender": final_g,  # RINGKASAN MAYORITAS, boleh salah di lokal
                "male_dur": m,
                "female_dur": f,
                "unknown_dur": u,
                "mode": "hf_svm_segmentwise",
            }

    # -------------------------------------------------
    # OUTPUT SECTION
    # -------------------------------------------------

    stem = Path(args.audio).stem  # contoh: "source_video_16k"
    speaker_raw_path = outdir / f"{stem}_speaker_timeline_raw.json"
    gender_tl_path   = outdir / f"{stem}_gender_timeline.json"
    speakers_path    = outdir / f"{stem}_speakers_stats.json"

    # 1) speaker timeline mentah (hasil diarization pyannote)
    #    -> cuma start/end + speaker lokal (apa kata pyannote langsung)
    with speaker_raw_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "segments": [
                    {
                        "start": seg["start"],
                        "end": seg["end"],
                        "speaker": seg["speaker"],   # contoh: "SPEAKER_00"
                    }
                    for seg in segs
                ],
                "duration": full_dur,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 2) gender timeline mentah PER SEGMEN
    #    -> ini krusial buat editor. Gender TIDAK LAGI dipaksa sama untuk seluruh speaker.
    with gender_tl_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "segments": [
                    {
                        "start": item["start"],
                        "end": item["end"],
                        "gender": item["gender"],          # "Male"/"Female"/"Unknown"
                        "speaker_local": item["speaker_local"],
                    }
                    for item in segment_gender_map
                ],
                "duration": full_dur,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 3) speaker summary (opsional, hanya buat debug / info UI tambahan)
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
