# backend/core/processor.py

import asyncio
import time
import json
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from core.session_manager import SessionManager
from core.translate import TranslateEngine


# ----------------- Helper kecil untuk diarization linking -----------------

def _parse_time(t):
    if isinstance(t, (int, float)):
        return float(t)
    if isinstance(t, str):
        t = t.strip()
        if not t:
            return 0.0
        p = t.replace(",", ".").split(":")
        try:
            if len(p) == 3:
                return int(p[0]) * 3600 + int(p[1]) * 60 + float(p[2])
            if len(p) == 2:
                return int(p[0]) * 60 + float(p[1])
            return float(p[0])
        except Exception:
            return 0.0
    return 0.0


def _safe_speaker_key(seg):
    for k in ("speaker", "spk", "spkid", "spk_id", "label"):
        if k in seg:
            return k
    return None


def _gather_samples(segments, samples_per_spk=8, min_dur=1.0):
    """
    Ambil potongan yang cukup panjang per speaker untuk bikin centroid embedding.
    """
    by = {}
    for s in segments:
        k = _safe_speaker_key(s)
        if not k:
            continue
        sp = s[k]
        st = _parse_time(s.get("start", 0))
        en = _parse_time(s.get("end", st))
        if en - st < min_dur:
            continue
        by.setdefault(sp, []).append((st, en, en - st))

    for sp, lst in by.items():
        lst.sort(key=lambda x: x[2], reverse=True)
        by[sp] = [(a, b) for a, b, _ in lst[:samples_per_spk]]

    return by


def _cos(a, b):
    return float((a * b).sum()) / (
        float(np.linalg.norm(a)) * float(np.linalg.norm(b)) + 1e-8
    )


def _extract_embeddings_ecapa(wav16k_path, time_spans, device="auto"):
    """
    Ambil embedding speaker per potongan waktu.
    Import torch / speechbrain berat dipindah ke sini supaya TIDAK dieksekusi saat startup server.
    """
    import torch, torchaudio
    from huggingface_hub import snapshot_download
    from speechbrain.inference import EncoderClassifier  # heavy, lazy import

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    local_dir = Path(wav16k_path).parent / ".sb_model_ecapa"
    try:
        snapshot_download(
            "speechbrain/spkrec-ecapa-voxceleb",
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,  # copy, bukan symlink
            token=False,
        )
    except Exception as e:
        print(f"[ECAPA] snapshot_download warn: {e}")

    classifier = EncoderClassifier.from_hparams(
        source=str(local_dir),
        run_opts={"device": device},
    )

    wav, sr = torchaudio.load(str(wav16k_path))
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000

    embs = []
    for (start, end) in time_spans:
        s = max(0, int(start * sr))
        e = min(wav.shape[1], int(end * sr))
        x = (
            torch.nn.functional.pad(wav[:, s:e], (0, max(0, sr - (e - s))))
            if e - s < sr
            else wav[:, s:e]
        )
        with torch.no_grad():
            v = classifier.encode_batch(x).squeeze().cpu().numpy()
        v = v / (np.linalg.norm(v) + 1e-8)
        embs.append(v.astype(np.float32))

    if embs:
        return np.stack(embs, axis=0)
    return np.zeros((0, 192), np.float32)


def _global_link_speakers(
    seg_path: Path,
    spk_path: Path,
    wav16k_path: Path,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    link_threshold: float = 0.93,
    samples_per_spk: int = 8,
    min_sample_dur: float = 1.5,
    device: str = "auto",
) -> Tuple[dict, dict]:
    """
    Satukan speaker ID lokal jadi global konsisten (SPK_01, SPK_02, ...),
    TAPI sekarang kita cegah laki/perempuan ketimpa jadi satu cluster.
    """

    # --- load json awal ---
    with open(seg_path, "r", encoding="utf-8") as f:
        seg = json.load(f)
    with open(spk_path, "r", encoding="utf-8") as f:
        spk = json.load(f)

    segments = seg.get("segments") if isinstance(seg, dict) else seg
    if not isinstance(segments, list):
        raise ValueError("segments json must be a list or have 'segments' list")

    # --- helper buat ambil tabel speaker -> gender dari output dracin_gender ---
    def _norm_spk_tbl(spk_obj):
        # nyamain struktur spk agar jadi dict {local_spk_id: {...info...}}
        if isinstance(spk_obj, dict) and "speakers" in spk_obj:
            tbl = spk_obj["speakers"]
        else:
            tbl = spk_obj
        if isinstance(tbl, list):
            out = {}
            for it in tbl:
                spid = it.get("id") or it.get("label") or it.get("name")
                if spid:
                    out[spid] = {
                        k: v
                        for k, v in it.items()
                        if k not in ("id", "label", "name")
                    }
            return out
        elif isinstance(tbl, dict):
            return tbl
        return {}

    spk_tbl = _norm_spk_tbl(spk)

    # Ambil sampel durasi panjang per speaker lokal
    samples = _gather_samples(
        segments,
        samples_per_spk=samples_per_spk,
        min_dur=min_sample_dur,
    )
    local_spks = list(samples.keys())
    if not local_spks:
        return seg, spk

    # Hitung centroid embedding per speaker lokal
    centroids = {}
    dur_by = {}
    for sp_local in local_spks:
        e = _extract_embeddings_ecapa(wav16k_path, samples[sp_local], device=device)
        if e.shape[0] == 0:
            continue
        c = e.mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-8)
        centroids[sp_local] = c
        dur_by[sp_local] = sum(b - a for a, b in samples[sp_local])

    # cluster global
    global_ids = []      # ["SPK_01", ...]
    global_vecs = []     # centroid gabungan untuk cluster tsb
    global_wgts = []     # total durasi gabungan
    cluster_gender = []  # gender cluster saat ini ("male"/"female"/"unknown")
    mapping = {}         # speaker lokal -> SPK_xx

    # urutkan speaker lokal dari durasi terpanjang dulu jadi anchor
    for sp_local in sorted(local_spks, key=lambda x: dur_by.get(x, 0), reverse=True):
        v = centroids.get(sp_local)

        # gender lokal dari hasil dracin_gender
        g_local = (
            (spk_tbl.get(sp_local, {}).get("gender") or "unknown").lower()
        )
        if g_local not in ("male", "female", "unknown"):
            g_local = "unknown"

        # kalau speaker ini gak punya centroid (aneh tapi mungkin)
        if v is None:
            gid = f"SPK_{len(global_ids)+1:02d}"
            global_ids.append(gid)
            global_vecs.append(None)
            global_wgts.append(float(dur_by.get(sp_local, 0.0)))
            cluster_gender.append(g_local)
            mapping[sp_local] = gid
            continue

        # cari cluster terbaik yg mirip
        best_i, best_sim = None, -1.0
        for i, gvec in enumerate(global_vecs):
            if gvec is None:
                continue

            # --- GENDER GUARD: kalau dua2nya tahu gender dan beda, skip compare
            g_cluster = cluster_gender[i]
            if (
                g_local in ("male", "female")
                and g_cluster in ("male", "female")
                and g_local != g_cluster
            ):
                continue  # jangan merge cowok ke cluster cewek atau sebaliknya

            s = _cos(v, gvec)
            if s > best_sim:
                best_sim = s
                best_i = i

        # cek threshold similarity
        if best_i is not None and best_sim >= link_threshold:
            # merge ke cluster existing
            gvec = global_vecs[best_i]
            w_old = float(global_wgts[best_i])
            w_new = float(dur_by.get(sp_local, 0.0))
            new = (gvec * (w_old + 1e-8) + v * (w_new + 1e-8))
            new = new / (np.linalg.norm(new) + 1e-8)

            global_vecs[best_i] = new
            global_wgts[best_i] = w_old + w_new

            # jangan overwrite gender cluster kalau cluster_gender[best_i] sudah punya male/female
            if cluster_gender[best_i] in ("unknown", None, "") and g_local in ("male","female"):
                cluster_gender[best_i] = g_local

            mapping[sp_local] = global_ids[best_i]
        else:
            # bikin cluster baru
            gid = f"SPK_{len(global_ids)+1:02d}"
            global_ids.append(gid)
            global_vecs.append(v)
            global_wgts.append(float(dur_by.get(sp_local, 0.0)))
            cluster_gender.append(g_local)
            mapping[sp_local] = gid

    # --- (opsional) enforce max_speakers sama seperti kode lama ---
    #   (disini bisa dibiarkan sama seperti implementasi kamu sekarang,
    #    termasuk nearest_pair merge, tapi kalau merge di sini
    #    sebaiknya juga hormati gender guard. Bisa kamu copy logic lama
    #    + tambahin check gender sama seperti di atas.)

    # Apply mapping ke semua segment
    seg_key = _safe_speaker_key(segments[0]) or "speaker"
    for s in segments:
        sp_val = s.get(seg_key)
        if isinstance(sp_val, list):
            s[seg_key] = [mapping.get(x, x) for x in sp_val]
        elif isinstance(sp_val, str):
            s[seg_key] = mapping.get(sp_val, sp_val)

    # Build tabel speakers final (mirip kode lama)
    by_global = {}
    for loc, gid in mapping.items():
        by_global.setdefault(gid, []).append(loc)

    merged = {}
    for gid, locals_ in by_global.items():
        info = {}
        for k in ("gender", "voice", "notes", "age", "accent"):
            for x in locals_:
                val = spk_tbl.get(x, {}).get(k)
                if val not in (None, "", "unknown"):
                    info[k] = val
                    break
        # fallback gender kalau belum ada
        if "gender" not in info or info["gender"] in (None, "", "unknown"):
            info["gender"] = cluster_gender[global_ids.index(gid)] or "unknown"

        merged[gid] = info

    return {"segments": segments}, {"speakers": merged}



# ----------------- ProcessingManager -----------------

class ProcessingManager:
    """
    Ini manager status + operasi berat per session.
    Versi ini sengaja dibuat RINGAN saat import:
    - Tidak load model besar di __init__
    - Tidak cetak "Project root: D:\\..." saat startup
    """

    def __init__(self):
        self.session_manager = SessionManager(Path("workspaces"))
        self.executor = ThreadPoolExecutor(max_workers=4)

        # engine berat (diarization, tts, dll) akan diload belakangan
        self._lock = threading.Lock()
        self._engines_ready = False
        self._diarization_engine = None
        self._tts_engine = None

    def _ensure_engines(self):
        """
        Dipanggil hanya ketika butuh diarization / export.
        Ini yang akan import torch/dll. BUKAN saat startup server.
        """
        if self._engines_ready:
            return
        with self._lock:
            if self._engines_ready:
                return

            from core.diarization import DiarizationEngine
            from core.tts_export import TTSExportEngine

            self._diarization_engine = DiarizationEngine()
            self._tts_engine = TTSExportEngine()
            self._engines_ready = True

    # ---- Session lifecycle ----

    async def create_session(self, session_data: Dict[str, Any]) -> str:
        session = self.session_manager.create_session(
            session_data["video_name"],
            session_data["srt_name"],
        )
        await self._notify_session_update(session.id)
        return session.id

    async def upload_files(
        self,
        session_id: str,
        video_file: bytes,
        srt_file: bytes,
    ):
        session = self.session_manager.get_session(session_id)
        if not session:
            raise ValueError("Session not found")

        workdir = session.workdir

        # simpan video
        if video_file:
            video_path = workdir / "source_video.mp4"
            video_path.write_bytes(video_file)
            self.session_manager.update_session(session_id, video_path=video_path)

        # simpan SRT
        if srt_file:
            srt_path = workdir / "source_subtitles.srt"
            srt_path.write_bytes(srt_file)
            self.session_manager.update_session(session_id, srt_path=srt_path)

        self.session_manager.update_session(
            session_id,
            status="files_uploaded",
        )
        await self._notify_session_update(session_id)

    async def generate_workdir(self, session_id: str):
        session = self.session_manager.get_session(session_id)
        if not session:
            raise ValueError("Session not found")

        self.session_manager.update_session(
            session_id,
            status="generating_workdir",
            current_step="Creating workspace structure",
        )
        await self._notify_session_update(session_id)

        def _generate():
            try:
                # copy SRT / rename
                if session.srt_path:
                    video_stem = (
                        session.video_path.stem if session.video_path else "source"
                    )
                    target_srt = session.workdir / f"{video_stem}.srt"
                    if session.srt_path != target_srt:
                        shutil.copy2(session.srt_path, target_srt)
                    self.session_manager.update_session(
                        session_id,
                        srt_path=target_srt,
                    )

                # tulis session.json
                session_info = {
                    "video": str(session.video_path) if session.video_path else None,
                    "srt_source": str(session.srt_path),
                    "workdir": str(session.workdir),
                    "created_at": time.time(),
                }
                (session.workdir / "session.json").write_text(
                    json.dumps(session_info, indent=2)
                )

                return True
            except Exception as e:
                return str(e)

        result = await asyncio.get_event_loop().run_in_executor(
            self.executor, _generate
        )

        if result is True:
            self.session_manager.update_session(
                session_id,
                status="workdir_ready",
                progress=20,
            )
        else:
            self.session_manager.update_session(
                session_id,
                status="error",
                error=result,
            )

        await self._notify_session_update(session_id)

    # ---- Audio extraction ----

    async def extract_audio(self, session_id: str):
        """
        Versi ringan:
        - Ambil audio dari source_video.mp4
        - Downmix ke mono 16kHz -> source_video_16k.wav
        - Simpan path di session.wav_16k
        (Tanpa import dracindub / engine lama)
        """
        session = self.session_manager.get_session(session_id)
        if not session or not session.video_path:
            raise ValueError("Session or video not found")

        self.session_manager.update_session(
            session_id,
            status="extracting_audio",
            current_step="Extracting 16kHz mono audio",
        )
        await self._update_progress(session_id, 30, "Extracting audio 16kHz")

        def _extract():
            try:
                workdir = session.workdir
                src_video = session.video_path
                wav16_path = workdir / "source_video_16k.wav"

                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(src_video),
                    "-ac",
                    "1",        # mono
                    "-ar",
                    "16000",    # 16 kHz
                    str(wav16_path),
                ]
                subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

                self.session_manager.update_session(
                    session_id,
                    wav_16k=wav16_path,
                )
                return True
            except Exception as e:
                return str(e)

        result = await asyncio.get_event_loop().run_in_executor(
            self.executor, _extract
        )

        if result is True:
            self.session_manager.update_session(
                session_id,
                status="audio_ready",
                progress=40,
                current_step="Audio ready (16kHz mono)",
            )
        else:
            self.session_manager.update_session(
                session_id,
                status="error",
                error=result,
            )

        await self._notify_session_update(session_id)

    # ---- Diarization ----

    async def run_diarization(self, session_id: str, cfg: dict):
        """
        Jalankan DiarizationEngine(), lalu lakukan global speaker linking
        (SPK_01, SPK_02, ...) supaya ID speaker konsisten sepanjang video.
        """
        self._ensure_engines()

        session = self.get_session(session_id)
        if not session:
            raise ValueError("Session not found")
        if not getattr(session, "wav_16k", None):
            raise ValueError("16kHz audio not found. Run extract-audio first")

        engine = self._diarization_engine
        result = await engine.process(session, cfg, lambda *a, **k: None)
        if not isinstance(result, dict):
            raise RuntimeError(f"Engine returned invalid result: {result!r}")
        if not result.get("success"):
            raise RuntimeError(result.get("error", "Diarization failed"))

        data = result["data"]
        seg_path = Path(data["segjson"])
        spk_path = Path(data["spkjson"])
        srt_path = Path(data["srt"]) if data.get("srt") else None

        cfg = cfg or {}
        if bool(cfg.get("link_global", True)):
            try:
                linked_seg, linked_spk = _global_link_speakers(
                    seg_path,
                    spk_path,
                    Path(session.wav_16k),
                    min_speakers=cfg.get("min_speakers"),
                    max_speakers=cfg.get("max_speakers"),
                    link_threshold=float(cfg.get("link_threshold", 0.93)),
                    samples_per_spk=int(cfg.get("samples_per_spk", 8)),
                    min_sample_dur=float(cfg.get("min_sample_dur", 1.0)),
                    device="auto",
                )

                seg_link = seg_path.with_name(seg_path.stem + "_linked.json")
                spk_link = spk_path.with_name(spk_path.stem + "_linked.json")
                seg_link.write_text(
                    json.dumps(linked_seg, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                spk_link.write_text(
                    json.dumps(linked_spk, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                # kompat ke nama lama (_segments.json / _speakers.json)
                seg_compat = seg_link.with_name(
                    seg_link.name.replace(
                        "_segments_linked.json", "_segments.json"
                    )
                )
                spk_compat = spk_link.with_name(
                    spk_link.name.replace(
                        "_speakers_linked.json", "_speakers.json"
                    )
                )
                shutil.copyfile(seg_link, seg_compat)
                shutil.copyfile(spk_link, spk_compat)
                print(
                    "[GlobalLink] compat copies ->",
                    seg_compat.name,
                    spk_compat.name,
                )
                seg_path, spk_path = seg_link, spk_link
                print(
                    "[GlobalLink] done ->",
                    seg_link.name,
                    spk_link.name,
                )
            except Exception as e:
                print(f"[GlobalLink] skipped: {e}")

        self.session_manager.update_session(
            session_id,
            segjson=seg_path,
            spkjson=spk_path,
            srtpath=srt_path,
            status="diarization_complete",
            progress=100,
            current_step="Diarization done",
        )

        await self._notify_session_update(session_id)

        return {
            "segments_path": seg_path,
            "speakers_path": spk_path,
            "srt_path": srt_path,
        }

    # ---- Export TTS (placeholder lama) ----

    async def run_tts_export(self, session_id: str, tts_config: Dict[str, Any]):
        """
        Ini placeholder.
        Pipeline export final MP4 kamu sekarang kan sudah ada di endpoint
        /api/session/{id}/export/build (chunked mix, center_cut, dll),
        jadi di sini kita gak paksakan lagi panggil engine lama.
        """
        self.session_manager.update_session(
            session_id,
            status="tts_export",
            current_step="Export via new pipeline not implemented here",
        )
        await self._notify_session_update(session_id)

        self.session_manager.update_session(
            session_id,
            status="error",
            error="run_tts_export is deprecated in web backend",
        )
        await self._notify_session_update(session_id)

    # ---- Util internal ----

    async def _update_progress(self, session_id: str, progress: int, message: str):
        session = self.session_manager.get_session(session_id)
        if session:
            self.session_manager.update_session(
                session_id,
                progress=progress,
                current_step=message,
            )
            await self._notify_session_update(session_id)

    async def _notify_session_update(self, session_id: str):
        """
        Broadcast state session ke semua client yang join di websocket.
        Import websocket_manager nya LAZY biar gak berat saat startup.
        """
        session = self.session_manager.get_session(session_id)
        if session:
            from api.websockets import websocket_manager
            await websocket_manager.broadcast_to_session(
                session_id,
                {
                    "type": "session_update",
                    "data": self._serialize_session(session),
                },
            )

    def _serialize_session(self, session) -> Dict[str, Any]:
        import dataclasses
        serialized = dataclasses.asdict(session)
        for key, value in serialized.items():
            if isinstance(value, Path):
                serialized[key] = str(value)
        return serialized

    def get_session(self, session_id: str):
        return self.session_manager.get_session(session_id)

    def list_sessions(self):
        return self.session_manager.list_sessions()

    async def run_translate(self, session_id: Optional[str], cfg: Dict[str, Any]):
        """
        Panggil TranslateEngine (DeepSeek style) langsung dari core/translate.py
        tanpa ikut engine lama.
        """
        session = None
        if session_id:
            session = self.session_manager.get_session(session_id)
            if not session:
                raise RuntimeError("Session not found")

        engine = TranslateEngine()
        result = await engine.process(session, cfg)

        if not isinstance(result, dict):
            raise RuntimeError(f"Engine returned invalid result: {result!r}")
        if not result.get("success"):
            raise RuntimeError(result.get("error", "Translate failed"))

        data = result["data"]

        # update session SRT kalau output ada
        if session and data.get("output_path"):
            out = Path(data["output_path"])
            if out.exists():
                self.session_manager.update_session(
                    session_id,
                    srtpath=str(out),
                    status="translate_complete",
                    progress=90,
                    current_step="Translate done",
                )
                await self._notify_session_update(session_id)

        return data


# global singleton buat Depends()
processing_manager = ProcessingManager()

def get_processing_manager():
    return processing_manager
