# backend/core/processor.py

import asyncio
import time
import json
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
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

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    num = float((a * b).sum())
    den = float(np.linalg.norm(a)) * float(np.linalg.norm(b)) + 1e-8
    return num / den

def _gather_samples(spk_spans: Dict[str, List[Tuple[float, float]]], samples_per_spk: int = 3):
    """
    Ambil beberapa potongan berdurasi terpanjang per speaker (local).
    spk_spans: { 'SPEAKER_00': [(start,end), ...], ... }
    return: { spk: [(start,end), ... <= samples_per_spk] }
    """
    by = {}
    for spk, spans in spk_spans.items():
        lst = sorted(spans, key=lambda ab: (ab[1] - ab[0]), reverse=True)
        by[spk] = lst[:samples_per_spk]
    return by

def _extract_embeddings_ecapa(
    wav16k_path: str,
    spans: List[Tuple[float, float]],
    device: str = "auto",
):
    """
    Ambil embedding speaker per potongan waktu suara pakai ECAPA (SpeechBrain).
    Kita ambil beberapa potongan (start,end) detik, convert ke mono 16 kHz,
    terus encode_batch() → embedding vektor. Lalu kita normalisasi.

    Return:
        np.ndarray shape [N, D]  (tiap baris = 1 potongan suara)
        Kalau nggak ada apa-apa → array shape [0, 192]
    """

    import torch, torchaudio
    from huggingface_hub import snapshot_download
    from speechbrain.inference import EncoderClassifier  # heavy, lazy import

    # pilih device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # download / cache model ECAPA ke folder kerja session
    local_dir = Path(wav16k_path).parent / ".sb_model_ecapa"
    try:
        snapshot_download(
            "speechbrain/spkrec-ecapa-voxceleb",
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,  # copy instead of symlink (lebih aman di Windows)
            token=False,
        )
    except Exception as e:
        print(f"[ECAPA] snapshot_download warn: {e}")

    # load classifier ECAPA dari local cache
    classifier = EncoderClassifier.from_hparams(
        source=str(local_dir),
        run_opts={"device": device},
    )

    # load audio 16 kHz mono
    wav, sr = torchaudio.load(str(wav16k_path))
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000

    # wav shape sekarang [C, T]; kita mau [1, T] mono
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    # pastikan [1, T]
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)

    embs = []

    for (start_sec, end_sec) in spans:
        # konversi detik → sample index
        s_idx = max(0, int(start_sec * sr))
        e_idx = min(wav.shape[1], int(end_sec * sr))
        if e_idx <= s_idx:
            continue

        chunk = wav[:, s_idx:e_idx]  # [1, Tchunk]

        # Pastikan potongan minimal panjang ~1 detik biar BN di ECAPA aman
        if chunk.shape[1] < sr:
            pad_needed = sr - chunk.shape[1]
            # pad kanan
            chunk = torch.nn.functional.pad(chunk, (0, pad_needed))

        with torch.no_grad():
            vec = classifier.encode_batch(chunk.to(device))  # [1,1,D]
        vec = vec.squeeze().cpu().numpy().astype(np.float32)

        # L2-normalize supaya konsisten
        norm = np.linalg.norm(vec) + 1e-8
        vec = (vec / norm).astype(np.float32)

        embs.append(vec)

    if not embs:
        # fallback kosong
        return np.zeros((0, 192), dtype=np.float32)

    return np.stack(embs, axis=0).astype(np.float32)


# -----------------------------------------------------------------------------
# SINGLE SOURCE OF TRUTH: global speaker linking
# -----------------------------------------------------------------------------
def _global_link_speakers(
    raw_path: Path,
    wav16k_path: Path,
    *,
    link_threshold: float = 0.93,
    samples_per_spk: int = 8,
    min_sample_dur: float = 1.0,   # boleh dipakai kalau mau filter span terlalu pendek
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    device: str = "auto",
):
    """
    Baca <stem>_speaker_timeline_raw.json, ambil embedding ECAPA per speaker lokal,
    cluster speaker lokal jadi speaker global (SPK_01, SPK_02, ...).

    Kemudian enforce max_speakers:
    - urutkan cluster berdasarkan total durasi bicara,
    - keep top-N,
    - sisanya merge ke cluster terdekat (cosine highest).

    Return:
      linked_obj = {
         "segments": [
            {"start_s": float, "end_s": float, "speaker": "SPK_01"},
            ...
         ],
         "duration": <float>
      }
      debug_map = {
         "local2global": {"SPEAKER_00": "SPK_01", ...},
         "cluster_dur": {...},
         "grouped_locals": {...}  # speaker lokal per cluster
      }
    """

    import numpy as np
    from collections import defaultdict

    # --- 1. load timeline mentah
    raw_json = json.loads(raw_path.read_text(encoding="utf-8"))
    segs_raw = raw_json.get("segments", [])
    total_dur = float(raw_json.get("duration", 0.0) or 0.0)

    # Helper buat ambil nama speaker lokal di seg lama
    def _get_local_spk(seg):
        key = _safe_speaker_key(seg)
        return seg.get(key, "") if key else ""

    # Kumpulin semua span per speaker lokal
    spans_by_spk = defaultdict(list)  # { "SPEAKER_00": [(s,e), ...] }
    for seg in segs_raw:
        s0 = float(seg.get("start") or seg.get("start_s") or 0.0)
        s1 = float(seg.get("end")   or seg.get("end_s")   or 0.0)
        if s1 <= s0:
            continue
        sp_local = _get_local_spk(seg)
        if not sp_local:
            continue
        # bisa skip segmen super pendek kalau mau pakai min_sample_dur
        if (s1 - s0) < float(min_sample_dur or 0.0):
            continue
        spans_by_spk[sp_local].append((s0, s1))

    # Ambil beberapa potongan terpanjang per speaker lokal
    sample_spans = _gather_samples(spans_by_spk, samples_per_spk)

    # Ekstrak embedding ECAPA utk setiap speaker lokal
    emb_avg_by_local = {}
    for sp_local, spans in sample_spans.items():
        emb_mat = _extract_embeddings_ecapa(str(wav16k_path), spans, device=device)
        if emb_mat.shape[0] > 0:
            emb_avg_by_local[sp_local] = emb_mat.mean(axis=0).astype(np.float32)

    locals_list = list(emb_avg_by_local.keys())
    if not locals_list:
        # fallback trivial: kasih nama SPK_01 ke semua
        linked_segments = []
        for seg in segs_raw:
            s0 = float(seg.get("start") or seg.get("start_s") or 0.0)
            s1 = float(seg.get("end")   or seg.get("end_s")   or 0.0)
            if s1 <= s0:
                continue
            sp_loc = _get_local_spk(seg) or "SPK_01"
            linked_segments.append({
                "start_s": s0,
                "end_s": s1,
                "speaker": "SPK_01" if sp_loc else "SPK_01",
            })
        return (
            {"segments": linked_segments, "duration": total_dur},
            {
                "local2global": {sp: "SPK_01" for sp in spans_by_spk.keys()},
                "cluster_dur": {},
                "grouped_locals": {"SPK_01": locals_list},
            }
        )

    # --- 2. Cluster local speakers pakai union-find + cosine sim threshold
    parent = {sp: sp for sp in locals_list}

    def uf_find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def uf_union(a, b):
        ra = uf_find(a)
        rb = uf_find(b)
        if ra != rb:
            parent[rb] = ra

    for i, a in enumerate(locals_list):
        for j in range(i + 1, len(locals_list)):
            b = locals_list[j]
            sim = _cos(emb_avg_by_local[a], emb_avg_by_local[b])
            if sim >= float(link_threshold):
                uf_union(a, b)

    # grouped_locals: root -> [local speakers...]
    grouped_locals = defaultdict(list)
    for sp in locals_list:
        grouped_locals[uf_find(sp)].append(sp)

    # centroid per cluster
    cluster_centroid = {}
    for root, members in grouped_locals.items():
        vecs = [emb_avg_by_local[m] for m in members if m in emb_avg_by_local]
        if not vecs:
            continue
        cluster_centroid[root] = np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)

    # durasi bicara per speaker lokal
    dur_by_local = {}
    for sp_local, lst in spans_by_spk.items():
        tot = 0.0
        for (s0, s1) in lst:
            tot += max(0.0, s1 - s0)
        dur_by_local[sp_local] = tot

    # durasi bicara per cluster
    cluster_dur = {}
    for root, members in grouped_locals.items():
        tot = 0.0
        for m in members:
            tot += dur_by_local.get(m, 0.0)
        cluster_dur[root] = tot

    # sort cluster by talk duration desc
    clusters_sorted = sorted(cluster_dur.items(), key=lambda kv: kv[1], reverse=True)
    cluster_roots_sorted = [c for (c, _dur) in clusters_sorted]

    # --- 3. Enforce max_speakers
    kept_roots = list(cluster_roots_sorted)
    if max_speakers is not None and len(kept_roots) > max_speakers:
        base_roots = kept_roots[:max_speakers]
        extra_roots = kept_roots[max_speakers:]

        # merge semua extra_roots ke base terdekat
        for rc in extra_roots:
            if rc not in cluster_centroid:
                continue
            # pilih base dg cosine tertinggi
            best_base = max(
                base_roots,
                key=lambda br: _cos(cluster_centroid[rc], cluster_centroid[br])
            )
            # gabungkan member rc ke grouped_locals[best_base]
            grouped_locals[best_base].extend(grouped_locals[rc])
            cluster_dur[best_base] += cluster_dur[rc]

        kept_roots = base_roots

    # kasih nama SPK_01, SPK_02, ... hanya utk kept_roots
    root2global = {}
    for idx, root in enumerate(kept_roots, start=1):
        root2global[root] = f"SPK_{idx:02d}"

    # assign local2global
    local2global = {}
    for root, members in grouped_locals.items():
        if root in root2global:
            gname = root2global[root]
        else:
            # cluster yg di-merge -> pilih base yg paling mirip centroidnya
            if root in cluster_centroid and kept_roots:
                best_base = max(
                    kept_roots,
                    key=lambda br: _cos(cluster_centroid[root], cluster_centroid[br])
                )
                gname = root2global[best_base]
            else:
                # fallback
                gname = "SPK_99"
        for m in members:
            local2global[m] = gname

    # --- 4. rewrite timeline segmen ke speaker global
    linked_segments = []
    for seg in segs_raw:
        s0 = float(seg.get("start") or seg.get("start_s") or 0.0)
        s1 = float(seg.get("end")   or seg.get("end_s")   or 0.0)
        if s1 <= s0:
            continue
        sp_local = _get_local_spk(seg)
        sp_global = local2global.get(sp_local, sp_local or "SPK_00")
        linked_segments.append({
            "start_s": s0,
            "end_s": s1,
            "speaker": sp_global,
        })

    linked_obj = {
        "segments": linked_segments,
        "duration": total_dur,
    }

    debug_map = {
        "local2global": local2global,
        "cluster_dur": cluster_dur,
        "grouped_locals": dict(grouped_locals),
    }

    return linked_obj, debug_map


# Backward-compat alias (biar kode lama yg masih manggil nama lama gak meledak):
def _global_link_speakers_from_raw(
    raw_path: Path,
    wav16k_path: Path,
    link_threshold: float = 0.93,
    samples_per_spk: int = 8,
    min_sample_dur: float = 1.0,
    device: str = "auto",
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
):
    return _global_link_speakers(
        raw_path,
        wav16k_path,
        link_threshold=link_threshold,
        samples_per_spk=samples_per_spk,
        min_sample_dur=min_sample_dur,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        device=device,
    )

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
        Jalankan DiarizationEngine() -> hasilkan:
          - <stem>_speaker_timeline_raw.json
          - <stem>_gender_timeline.json

        Lalu (opsional link_global):
          - bikin <stem>_speaker_timeline_linked.json
            dengan ID global SPK_01, SPK_02, ... yang stabil di seluruh video.

        Kita TIDAK lagi pakai *_gender_segments.json / *_gender_speakers.json.
        """

        # 1. pastikan engine diarization sudah kebuka
        self._ensure_engines()

        # 2. ambil session + cek audio 16k
        session = self.get_session(session_id)
        if not session:
            raise ValueError("Session not found")
        if not getattr(session, "wav_16k", None):
            raise ValueError("16kHz audio not found. Run extract-audio first")

        # 3. jalankan DiarizationEngine.process()
        # engine.process() sekarang harus balikin:
        #   {
        #     "success": True,
        #     "data": {
        #        "speaker_timeline_raw": "<path/..._speaker_timeline_raw.json>",
        #        "gender_timeline":      "<path/..._gender_timeline.json>"
        #     }
        #   }
        engine = self._diarization_engine
        result = await engine.process(session, cfg, lambda *a, **k: None)

        if not isinstance(result, dict):
            raise RuntimeError(f"Engine returned invalid result: {result!r}")
        if not result.get("success"):
            raise RuntimeError(result.get("error", "Diarization failed"))

        data = result["data"]

        raw_path_str    = data.get("speaker_timeline_raw")
        gender_path_str = data.get("gender_timeline")

        if not raw_path_str or not gender_path_str:
            raise RuntimeError("Diarization result missing timeline paths")

        raw_path    = Path(raw_path_str).resolve()
        gender_path = Path(gender_path_str).resolve()

        if not raw_path.exists():
            raise RuntimeError("speaker_timeline_raw.json not found on disk")
        if not gender_path.exists():
            raise RuntimeError("gender_timeline.json not found on disk")

        # 4. Tentukan path linked (ID global SPK_01, SPK_02, ...)
        # setelah dapet raw_path, gender_path
        linked_path = raw_path.with_name(
            raw_path.name.replace("_speaker_timeline_raw", "_speaker_timeline_linked")
        )

        if str(cfg.get("link_global", "true")).lower() == "true":
            linked_obj, debug_map = _global_link_speakers(
                raw_path,
                Path(session.wav_16k),
                link_threshold=float(cfg.get("link_threshold", 0.93)),
                samples_per_spk=int(cfg.get("samples_per_spk", 8)),
                min_sample_dur=float(cfg.get("min_sample_dur", 1.0)),
                min_speakers=cfg.get("min_speakers"),
                max_speakers=cfg.get("max_speakers"),
                device=("cuda" if cfg.get("use_gpu") else "cpu"),
            )

            # tulis linked timeline
            linked_path.write_text(
                json.dumps(linked_obj, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )

            # optional: mapping debug
            debug_map_path = linked_path.with_name(
                linked_path.name.replace(
                    "_speaker_timeline_linked",
                    "_speaker_timeline_linked_mapping"
                )
            )
            debug_map_path.write_text(
                json.dumps(debug_map, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        else:
            # kalau user matiin link_global -> anggap "linked" = raw
            linked_path = raw_path

        # lalu update session metadata (ini udah ada di kode kamu)
        self.session_manager.update_session(
            session_id,
            status="diar_done",
            progress=80,
            current_step="Diarization done",
            speaker_timeline_raw=str(raw_path),
            speaker_timeline_linked=str(linked_path),
            gender_timeline=str(gender_path),
        )


        # 5. update status session biar UI tau path2 terbaru
        self.session_manager.update_session(
            session_id,
            speaker_timeline_raw=str(raw_path),
            speaker_timeline_linked=str(linked_path),
            gender_timeline=str(gender_path),
            status="diarization_complete",
            progress=100,
            current_step="Diarization done",
        )

        await self._notify_session_update(session_id)

        # 6. return buat endpoint /api/session/{id}/diarization
        return {
            "speaker_timeline_raw":      str(raw_path),
            "speaker_timeline_linked":   str(linked_path),
            "gender_timeline":           str(gender_path),
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

    def delete_session(self, session_id: str):
        """
        Hapus session dari SessionManager + hapus folder workdir di disk.
        """
        session = self.session_manager.get_session(session_id)
        if not session:
            raise RuntimeError(f"Unknown session_id {session_id}")

        # simpan path sebelum di-pop
        workdir = getattr(session, "workdir", None)

        # buang dari SessionManager
        self.session_manager.delete_session(session_id)

        # hapus folder fisik (jika ada)
        if workdir:
            try:
                shutil.rmtree(workdir, ignore_errors=True)
            except Exception as e:
                print(f"[WARN] gagal hapus workspace {workdir}: {e}")
                
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
