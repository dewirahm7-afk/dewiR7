# backend/api/endpoints.py
from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    HTTPException,
    Depends,
    Response,
    Query,
    Body,
    Request,
)
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    StreamingResponse,
)
from pathlib import Path

from api.websockets import websocket_manager  # jika dipakai di tempat lain
from core.processor import processing_manager
from core.session_manager import session_manager

from pydantic import BaseModel
import json, re, tempfile, zipfile, shutil, subprocess, math, os
from typing import List, Optional, Literal, Dict, Any

from starlette.responses import StreamingResponse, Response
import asyncio, json, time
from urllib.parse import quote

router = APIRouter()

# --- Dependencies -------------------------------------------------------------

def get_processing_manager():
    return processing_manager

def get_session_manager():
    return session_manager


# =============================================================================
# TAB 1: SRT Processing (Create session, upload, workdir, audio, diarization)
# =============================================================================

@router.post("/api/session/create")
async def create_session(
    video_name: str = Form(...),
    srt_name: str = Form(...),
    pm = Depends(get_processing_manager),
):
    try:
        session_data = {
            "video_name": video_name,
            "srt_name": srt_name,
        }
        session_id = await pm.create_session(session_data)
        return JSONResponse({"session_id": session_id, "status": "created"})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/session/{session_id}/upload")
async def upload_files(
    session_id: str,
    video: Optional[UploadFile] = File(None),
    srt: UploadFile = File(...),
    pm = Depends(get_processing_manager),
):
    try:
        video_content = await video.read() if video else None
        srt_content = await srt.read()
        await pm.upload_files(session_id, video_content, srt_content)
        return JSONResponse({"status": "files_uploaded"})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/session/{session_id}/generate-workdir")
async def generate_workdir(
    session_id: str,
    pm = Depends(get_processing_manager),
):
    try:
        await pm.generate_workdir(session_id)
        return JSONResponse({"status": "workdir_generated"})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/session/{session_id}/extract-audio")
async def extract_audio(
    session_id: str,
    vocal_only: str = Form("true"),
    prefer: str = Form("demucs"),
    pm = Depends(get_processing_manager),
):
    """
    Demucs-only extractor.

    Output minimal:
      - source_video_16k.wav : vocals only, mono 16 kHz (untuk diarization / pipeline)
      - instrument.wav       : BGM only (hasil no_vocals dari Demucs)

    Tidak menyisakan file lain (hemat disk).
    """
    import shutil, subprocess, os
    from pathlib import Path

    def _which(cmd: str) -> bool:
        return shutil.which(cmd) is not None

    def _run(cmd: list) -> tuple[int, str, str]:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return p.returncode, p.stdout, p.stderr

    def _to_16k_mono(inp: Path, outp: Path, gain_db: float = 3.0) -> None:
        """
        - pan=mono equal-power (0.707) → jaga persepsi loudness
        - volume=+3dB (bisa ubah), alimiter untuk cegah clipping
        """
        af = f"pan=mono|c0=0.707*FL+0.707*FR,volume={gain_db}dB,alimiter=limit=0.96"
        rc, _, err = _run([
            "ffmpeg","-y","-i", str(inp),
            "-af", af, "-ac","1","-ar","16000","-c:a","pcm_s16le", str(outp)
        ])
        if rc != 0:
            raise HTTPException(status_code=500, detail=f"ffmpeg resample failed:\n{err}")

    try:
        sess = pm.get_session(session_id)
        if not sess:
            raise HTTPException(status_code=404, detail="Session not found")
        if not getattr(sess, "video_path", None):
            raise HTTPException(status_code=404, detail="Video not found in session")

        ws = Path(sess.workdir).resolve()
        ws.mkdir(parents=True, exist_ok=True)

        vid = Path(sess.video_path).resolve()
        if not vid.exists():
            raise HTTPException(status_code=404, detail="Video file not found")

        # === 0) Cek Demucs tersedia ===
        demucs_cmd = None
        if _which("demucs"):
            demucs_cmd = ["demucs"]
        elif _which("python"):
            # fallback: python -m demucs
            demucs_cmd = ["python", "-m", "demucs"]
        if demucs_cmd is None:
            raise HTTPException(status_code=500, detail="Demucs not installed or not on PATH")

        # === 1) Jalankan Demucs langsung dari file video (hemat disk) ===
        outdir = ws / "_sep_demucs"
        outdir.mkdir(exist_ok=True)
        rc, _, err = _run(demucs_cmd + ["--two-stems=vocals", "-o", str(outdir), str(vid)])
        if rc != 0:
            raise HTTPException(status_code=500, detail=f"Demucs failed:\n{err}")

        # === 2) Ambil kedua stem secara eksplisit ===
        #    - vocals.wav       -> source_video_16k.wav (mono 16k)
        #    - no_vocals.wav    -> instrument.wav
        vocals_path = None
        inst_path = None
        # pilih kandidat terbaru jika lebih dari satu (misal multiple runs)
        cands_voc = sorted(list(outdir.rglob("vocals.wav")), key=lambda p: p.stat().st_mtime, reverse=True)
        cands_no  = sorted(list(outdir.rglob("no_vocals.wav")), key=lambda p: p.stat().st_mtime, reverse=True)
        if cands_voc:
            vocals_path = cands_voc[0]
        if cands_no:
            inst_path = cands_no[0]
        if vocals_path is None or inst_path is None:
            raise HTTPException(status_code=500, detail="Demucs output not found: vocals.wav / no_vocals.wav")

        # === 3) Tulis final minimal ===
        out_16k = ws / "source_video_16k.wav"
        out_bgm = ws / "instrument.wav"

        _to_16k_mono(vocals_path, out_16k)      # convert -> mono 16k
        # pindah (hemat disk). Jika beda drive, shutil.move akan copy+delete.
        if out_bgm.exists():
            try: out_bgm.unlink()
            except: pass
        shutil.move(str(inst_path), str(out_bgm))

        # >>>> TAMBAHKAN INI <<<<
        pm.session_manager.update_session(
            session_id,
            wav_16k=out_16k,                    # agar diarization tidak error "16kHz audio not found"
            status='audio_ready',
            progress=40,
            current_step='Audio extracted (Demucs)'
        )
        await pm._notify_session_update(session_id)
        # <<<< SAMPAI SINI <<<<

        # (opsional) bersihkan folder demucs untuk hemat disk:
        # import shutil as _sh; _sh.rmtree(outdir, ignore_errors=True)

        return JSONResponse({
            "status": "audio_extracted",
            "vocal_isolation": "demucs",
            "paths": {
                "vocals_16k": str(out_16k),
                "instrument": str(out_bgm),
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        raise HTTPException(status_code=400, detail=f"{e}\n{traceback.format_exc()}")


@router.post("/api/session/{session_id}/diarization")
async def run_diarization(
    session_id: str,

    # =========================
    # GENDER CONFIG
    # =========================
    # dropdown baru di frontend:
    #   "reference"  -> mode lama (butuh male_ref/female_ref)
    #   "hf_svm"     -> mode baru (pakai model griko)
    #
    # default "reference" = aman buat UI lama
    gender_mode: str = Form("reference"),

    # mode lama: wajib, mode baru: boleh kosong
    male_ref: Optional[str] = Form(None),
    female_ref: Optional[str] = Form(None),

    # mode baru hf_svm: parameter voting
    # (UI lama tidak mengirim field ini, tapi kita kasih default supaya tidak error)
    min_vote: float = Form(0.7),
    min_len_sec: float = Form(0.5),

    # berapa segmen terpanjang per speaker yg dipakai buat gender
    top_n: int = Form(5),

    # =========================
    # DIARIZATION / DEVICE
    # =========================
    hf_token: str = Form(...),
    use_gpu: bool = Form(True),

    # =========================
    # GLOBAL SPEAKER LINKING
    # =========================
    link_global: str = Form("true"),        # "true"/"false"
    link_threshold: float = Form(0.93),
    samples_per_spk: int = Form(8),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
    min_sample_dur: float = Form(1.5),

    pm = Depends(get_processing_manager),
):
    """
    Endpoint ini sekarang support 2 mode gender:
    - reference : pakai bank male_ref/female_ref (versi lama)
    - hf_svm    : pakai model pretrained HF (griko/gender_cls_svm_ecapa_voxceleb)
                  -> tidak butuh male_ref/female_ref
                  -> pakai voting min_vote/min_len_sec
    """

    try:
        # siapkan config lengkap buat ProcessingManager
        cfg = {
            # --- mode klasifikasi gender:
            "gender_mode": gender_mode,           # "reference" atau "hf_svm"

            # untuk mode "reference" (lama)
            "male_ref": male_ref or "",
            "female_ref": female_ref or "",

            # untuk mode "hf_svm" (baru)
            "min_vote": float(min_vote),
            "min_len_sec": float(min_len_sec),

            # umum untuk kedua mode
            "top_n": int(top_n),

            # diarization requirement
            "hf_token": hf_token,
            "use_gpu": bool(use_gpu),

            # global linking config
            "link_global": (link_global or "true").lower() == "true",
            "link_threshold": float(link_threshold),
            "samples_per_spk": int(samples_per_spk),
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
            "min_sample_dur": float(min_sample_dur),
        }

        # jalankan pipeline diarization + gender + (opsional) global linking
        result = await pm.run_diarization(session_id, cfg)

        # ProcessingManager diharapkan balikin dict minimal:
        # {
        #   "segments_path": <path ke *_segments.json final>,
        #   "speakers_path": <path ke *_speakers.json final>
        # }

        return JSONResponse(
            {
                "status": "diarization_completed",
                "segments_path": str(result["segments_path"]),
                "speakers_path": str(result["speakers_path"]),
            }
        )

    except Exception as e:
        import traceback
        raise HTTPException(
            status_code=400,
            detail=f"{e}\n{traceback.format_exc()}",
        )

# Info & utilities used across tabs
@router.get("/api/session/{session_id}")
async def get_session(
    session_id: str,
    pm = Depends(get_processing_manager),
):
    try:
        session = pm.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return JSONResponse(pm._serialize_session(session))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/session/{session_id}/delete")
async def delete_session(session_id: str, pm = Depends(get_processing_manager)):
    """
    Hapus session dari memori + hapus folder workspace fisiknya.
    Dipakai tombol Delete di tab SRT Processing.
    """
    try:
        pm.delete_session(session_id)
        return JSONResponse({"status": "deleted", "session_id": session_id})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def iter_file_range(path: Path, start: int, end: int, chunk_size: int = 1024 * 1024):
    with open(path, "rb") as f:
        f.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            data = f.read(min(chunk_size, remaining))
            if not data:
                break
            remaining -= len(data)
            yield data
            
@router.get("/api/session/{session_id}/video")
async def get_video(session_id: str, request: Request, pm = Depends(get_processing_manager)):
    session = pm.get_session(session_id)
    if not session or not session.video_path:
        raise HTTPException(status_code=404, detail="Video not found")

    file_path = Path(session.video_path)
    file_size = file_path.stat().st_size
    range_header = request.headers.get("range")

    # Jika browser minta partial (seek)
    if range_header:
        # contoh: "bytes=12345-67890" atau "bytes=12345-"
        m = re.match(r"bytes=(\d+)-(\d*)", range_header)
        if m:
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else file_size - 1
            end = min(end, file_size - 1)
            headers = {
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(end - start + 1),
                "Content-Type": "video/mp4",
            }
            return StreamingResponse(
                iter_file_range(file_path, start, end),
                status_code=206,
                headers=headers,
            )

    # fallback: kirim full file tapi tetap deklarasi Accept-Ranges
    resp = FileResponse(str(file_path), media_type="video/mp4", filename=file_path.name)
    resp.headers["Accept-Ranges"] = "bytes"
    return resp

@router.get("/api/sessions")
async def list_sessions(pm = Depends(get_processing_manager)):
    try:
        sessions = pm.list_sessions()
        # Frontend ada dua varian: kadang expects array langsung, kadang {sessions:[]}
        # Kita kirim array langsung sesuai penggunaan di Editing tab.
        return JSONResponse(sessions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint lama yang dipakai sebagian UI untuk load contoh data
@router.get("/api/session/{session_id}/editing-data")
async def get_editing_data(
    session_id: str,
    pm = Depends(get_processing_manager),
):
    try:
        session = pm.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        mock_data = {
            "entries": [
                {
                    "index": 1,
                    "start": 0.0,
                    "end": 5.0,
                    "text": "Sample subtitle text",
                    "speaker": "Speaker1",
                    "gender": "Male",
                    "duration": 5.0,
                }
            ],
            "video_url": f"/api/session/{session_id}/video",
        }
        return JSONResponse(mock_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/test")
async def test_endpoint():
    return JSONResponse({"message": "API is working!", "status": "success"})


# =============================================================================
# TAB 2: Translate (Auto & Manual) + Get/Save SRT
# =============================================================================

@router.get("/api/session/{session_id}/srt")
def get_session_srt(
    session_id: str,
    prefer: str = "auto",  # "auto" | "original" | "gender" | "translated"
    pm = Depends(get_processing_manager),
):
    """
    Mengembalikan isi SRT dari workspace session dgn prioritas sesuai `prefer`.

    prefer=original:
        1) source_subtitles.srt
        2) source_video.srt
    prefer=gender:
        1) *_gender_*.srt (latest, root & subfolder)
    prefer=translated:
        1) translated_latest.srt
        2) translated_*.srt (latest)
    prefer=auto (default):
        1) session.srtpath (jika valid & di dalam workdir)
        2) *_gender_*.srt (latest)
        3) source_subtitles.srt
        4) source_video.srt
    """
    session = pm.session_manager.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    workdir = Path(session.workdir).resolve()
    if not workdir.exists():
        raise HTTPException(404, "Workspace not found")

    def in_workdir(p: Path) -> bool:
        try:
            rp = p.resolve()
            return rp.exists() and (workdir == rp or workdir in rp.parents)
        except Exception:
            return False

    def latest(patterns):
        cands = []
        for pat in patterns:
            cands += list(workdir.glob(pat))
            cands += list(workdir.glob(f"**/{pat}"))
        uniq = {str(p.resolve()): p for p in cands}.values()
        sorted_cands = sorted(uniq, key=lambda p: p.stat().st_mtime, reverse=True)
        return sorted_cands[0] if sorted_cands else None

    prefer = (prefer or "auto").lower().strip()
    srt_path: Optional[Path] = None

    if prefer == "original":
        srt_path = workdir / "source_subtitles.srt"
        if not srt_path.exists():
            srt_path = workdir / "source_video.srt"
        if not srt_path.exists():
            srt_path = None

    elif prefer == "gender":
        srt_path = latest(["*_gender_*.srt"])

    elif prefer == "translated":
        srt_path = workdir / "translated_latest.srt"
        if not srt_path.exists():
            srt_path = latest(["translated_*.srt"])

    else:  # auto
        hint = getattr(session, "srtpath", None)
        if hint:
            p = Path(hint)
            if in_workdir(p):
                srt_path = p

        if not srt_path:
            srt_path = latest(["*_gender_*.srt"])

        if not srt_path:
            p = workdir / "source_subtitles.srt"
            if p.exists():
                srt_path = p

        if not srt_path:
            p = workdir / "source_video.srt"
            if p.exists():
                srt_path = p

    if not srt_path or not in_workdir(srt_path):
        raise HTTPException(404, "SRT not found for this session")

    return Response(
        srt_path.read_text(encoding="utf-8"),
        media_type="text/plain; charset=utf-8",
    )


@router.post("/api/session/{session_id}/translate")
async def translate_session(
    session_id: str,
    api_key: str = Form(""),
    target_lang: str = Form("id"),
    engine: str = Form("llm"),
    temperature: float = Form(0.1),
    top_p: float = Form(0.3),
    batch: int = Form(20),
    workers: int = Form(1),
    timeout: int = Form(120),
    autosave: str = Form("true"),
    srt_text: str = Form(""),
    mode: str = Form("auto"),
    prefer: str = Form("original"),
    pm = Depends(get_processing_manager),
):
    cfg = {
        "api_key": api_key,
        "target_lang": target_lang,
        "engine": engine,
        "temperature": temperature,
        "top_p": top_p,
        "batch": batch,
        "workers": workers,
        "timeout": timeout,
        "autosave": autosave.lower() == "true",
        "mode": mode,
        "srt_text": srt_text,
        "prefer": prefer,
    }
    data = await pm.run_translate(session_id, cfg)
    return JSONResponse(data)

# === Tambahkan di bawah endpoint /api/session/{id}/translate (atau berdampingan) ===
@router.post("/api/session/{session_id}/translate/stream")
async def translate_stream(
    session_id: str,
    api_key: str = Form(""),
    target_lang: str = Form("id"),
    engine: str = Form("llm"),
    temperature: float = Form(0.1),
    top_p: float = Form(0.3),
    batch: int = Form(20),
    workers: int = Form(1),
    timeout: int = Form(120),
    autosave: str = Form("true"),
    srt_text: str = Form(""),
    mode: str = Form("dubbing"),
    prefer: str = Form("original"),
    only_indices: str = Form(""),
    typing_delay_ms: int = Form(20),   # <<< TAMBAHKAN INI
    pm = Depends(get_processing_manager),
):
    session = pm.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    from core.translate import TranslateEngine
    eng = TranslateEngine()

    # Ambil SRT dari workspace (atau dari srt_text jika dikirim)
    text, workdir = eng._resolve_input_srt(session, {"prefer": prefer, "srt_text": srt_text})
    items = eng._parse_srt(text)
    if not items:
        raise HTTPException(400, "SRT is empty or cannot be parsed")

    # Filter indeks jika diminta
    only = set()
    if only_indices.strip():
        for x in re.split(r"[,\s]+", only_indices.strip()):
            if x.isdigit():
                only.add(int(x))
    items_to_process = [it for it in items if not only or it["index"] in only]

    # Buffer hasil untuk autosave
    trans_buf = [""] * len(items)
    idx_pos = {it["index"]: i for i, it in enumerate(items)}

    async def eventgen():
        try:
            yield (json.dumps({"type": "begin", "total": len(items_to_process)}) + "\n").encode()

            # === STREAM PER CHUNK (bukan per item) ==========================
            # G = ukuran chunk dari GUI; batasi biar aman (mis. ≤ 50)
            G = max(1, min(int(batch or 10), 50))
            W = max(1, int(workers))
            done = 0

            for off in range(0, len(items_to_process), G):
                chunk = items_to_process[off:off + G]

                # bagi ke sub-batch paralel sesuai workers
                internal_bs = max(1, (len(chunk) + W - 1) // W)

                results = eng._translate_items_with_deepseek(
                    items=chunk,
                    api_key=api_key,
                    style=mode,
                    target_lang=target_lang,
                    temperature=float(temperature),
                    top_p=float(top_p),
                    batch_size=int(internal_bs),
                    workers=W,
                    timeout=int(timeout),
                )

                # --- pindahkan delay ke level ITEM (bukan setelah chunk) ---
                delay_s = max(0.0, min(float(typing_delay_ms), 200.0) / 1000.0)

                for item, t in zip(chunk, results):
                    # simpan ke buffer (autosave)
                    pos = idx_pos.get(item["index"])
                    if pos is not None:
                        trans_buf[pos] = t or ""

                    # kirim satu hasil (NDJSON)
                    yield (json.dumps({
                        "type": "result",
                        "index": item["index"],
                        "timestamp": f'{item["start"]} --> {item["end"]}',
                        "original_text": item["text"],
                        "translation": t or ""
                    }) + "\n").encode()

                    done += 1
                    # progress per item
                    yield (json.dumps({
                        "type": "progress",
                        "done": done,
                        "total": len(items_to_process)
                    }) + "\n").encode()

                    # delay kosmetik antar item (efek "mengetik")
                    if delay_s > 0.0:
                        await asyncio.sleep(delay_s)

                # autosave tiap selesai satu chunk (aman & jarang I/O)
                if autosave.lower() == "true":
                    try:
                        srt_out = eng._build_srt_with_trans(items, trans_buf)
                        (workdir / "translated_latest.srt").write_text(srt_out, encoding="utf-8")
                    except Exception:
                        pass

                # beri napas event loop (tanpa delay kosmetik lagi di level chunk)
                await asyncio.sleep(0)
            # =================================================================

            yield (json.dumps({"type": "end"}) + "\n").encode()
        except asyncio.CancelledError:
            return

    return StreamingResponse(
        eventgen(),
        media_type="application/x-ndjson; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "Content-Type": "application/x-ndjson; charset=utf-8",
            "Transfer-Encoding": "chunked",
        },
    )

@router.post("/api/translate/stream")
async def translate_stream_manual(
    api_key: str = Form(""),
    target_lang: str = Form("id"),
    engine: str = Form("llm"),
    temperature: float = Form(0.1),
    top_p: float = Form(0.3),
    batch: int = Form(20),            # dihormati utk ukuran chunk
    workers: int = Form(1),           # dihormati utk paralel per chunk
    timeout: int = Form(120),
    srt_text: str = Form(...),        # WAJIB untuk manual
    mode: str = Form("dubbing"),
    typing_delay_ms: int = Form(20),
    only_indices: str = Form(""),
):
    from core.translate import TranslateEngine
    eng = TranslateEngine()

    # Parse SRT dari text
    items = eng._parse_srt(srt_text)
    if not items:
        raise HTTPException(400, "SRT is empty or cannot be parsed")

    # Filter indeks kalau ada
    only = set()
    if only_indices.strip():
        for x in re.split(r"[,\s]+", only_indices.strip()):
            if x.isdigit():
                only.add(int(x))
    items_to_process = [it for it in items if not only or it["index"] in only]

    async def eventgen():
        try:
            yield (json.dumps({"type": "begin", "total": len(items_to_process)}) + "\n").encode()

            # === STREAM PER CHUNK (bukan per item) ==========================
            G = max(1, min(int(batch or 10), 50))
            W = max(1, int(workers))
            done = 0

            for off in range(0, len(items_to_process), G):
                chunk = items_to_process[off:off + G]
                internal_bs = max(1, (len(chunk) + W - 1) // W)

                results = eng._translate_items_with_deepseek(
                    items=chunk,
                    api_key=api_key,
                    style=mode,
                    target_lang=target_lang,
                    temperature=float(temperature),
                    top_p=float(top_p),
                    batch_size=int(internal_bs),
                    workers=W,
                    timeout=int(timeout),
                )

                delay_s = max(0.0, min(float(typing_delay_ms), 200.0) / 1000.0)

                for item, t in zip(chunk, results):
                    yield (json.dumps({
                        "type": "result",
                        "index": item["index"],
                        "timestamp": f'{item["start"]} --> {item["end"]}',
                        "original_text": item["text"],
                        "translation": t or ""
                    }) + "\n").encode()

                    done += 1
                    yield (json.dumps({
                        "type": "progress",
                        "done": done,
                        "total": len(items_to_process)
                    }) + "\n").encode()

                    if delay_s > 0.0:
                        await asyncio.sleep(delay_s)
            # =================================================================

            yield (json.dumps({"type": "end"}) + "\n").encode()
        except asyncio.CancelledError:
            return

    return StreamingResponse(
        eventgen(),
        media_type="application/x-ndjson; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "Content-Type": "application/x-ndjson; charset=utf-8",
            "Transfer-Encoding": "chunked",
        },
    )

@router.post("/api/translate")
async def translate_manual(
    api_key: str = Form(""),
    target_lang: str = Form("id"),
    engine: str = Form("llm"),
    temperature: float = Form(0.1),
    top_p: float = Form(0.3),
    batch: int = Form(20),
    workers: int = Form(1),
    timeout: int = Form(120),
    autosave: str = Form("true"),
    srt_text: str = Form(...),  # wajib untuk manual
    mode: str = Form("manual"),
    pm = Depends(get_processing_manager),
):
    cfg = {
        "api_key": api_key,
        "target_lang": target_lang,
        "engine": engine,
        "temperature": temperature,
        "top_p": top_p,
        "batch": batch,
        "workers": workers,
        "timeout": timeout,
        "autosave": autosave.lower() == "true",
        "mode": mode,
        "srt_text": srt_text,
    }
    data = await pm.run_translate(None, cfg)
    return JSONResponse(data)


@router.post("/api/session/{session_id}/save-srt")
async def save_translated_srt(
    session_id: str,
    srt_text: str = Form(...),
    filename: str = Form("translated_latest.srt"),
    pm = Depends(get_processing_manager),
):
    session = pm.session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    workdir = Path(session.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    out_path = workdir / filename
    out_path.write_text(srt_text, encoding="utf-8")

    pm.session_manager.update_session(session_id, srtpath=str(out_path))
    return JSONResponse({"ok": True, "path": str(out_path)})


# =============================================================================
# TAB 3: Editing (load rows, save rows, export SRT by gender / zip)
# =============================================================================

def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore") if p and p.exists() else ""

# ==== SRT utilities (replace the existing ones) ==============================
_TIME_RE = re.compile(r"(\d{2}:\d{2}:\d{2})[,.](\d{3})")
_TIME_LINE_RE = re.compile(
    r"(?P<s>\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(?P<e>\d{2}:\d{2}:\d{2}[,\.]\d{3})"
)

def _norm_ts(ts: str) -> str:
    # pastikan milidetik pakai koma
    return ts.replace(".", ",")

def _clean_srt_text(s: str) -> str:
    if not s:
        return ""
    s = s.lstrip("\ufeff")                # BOM
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # buang header WEBVTT jika ada
    lines = s.split("\n")
    if lines and lines[0].strip().upper().startswith("WEBVTT"):
        lines = lines[1:]
    return "\n".join(lines)

def _t2s(ts: str) -> float:
    ts = _norm_ts(ts)
    m = re.match(r"(?P<h>\d\d):(?P<m>\d\d):(?P<s>\d\d),(?P<ms>\d{3})", ts.strip())
    if not m:
        return 0.0
    h, m_, s, ms = int(m["h"]), int(m["m"]), int(m["s"]), int(m["ms"])
    return h * 3600 + m_ * 60 + s + ms / 1000.0

def _parse_srt(srt_text: str) -> List[dict]:
    """
    Parser yang toleran: menerima blok tanpa indeks, '.' atau ',' untuk ms,
    dan mengabaikan header WEBVTT/BOM.
    """
    s = _clean_srt_text(srt_text)
    if not s.strip():
        return []

    blocks, cur = [], []
    for line in s.split("\n"):
        if line.strip():
            cur.append(line)
        else:
            if cur:
                blocks.append(cur)
                cur = []
    if cur:
        blocks.append(cur)

    out, autono = [], 1
    for b in blocks:
        if not b:
            continue

        # Jika baris pertama angka, itu index; kalau tidak, kita auto-number
        k = 0
        try:
            idx = int(b[0].strip())
            k = 1
        except Exception:
            idx = autono

        # Cari baris timecode di dalam blok mulai dari k
        tline = None
        for i in range(k, len(b)):
            m = _TIME_LINE_RE.search(b[i])
            if m:
                tline = (i, m.group("s"), m.group("e"))
                break
        if not tline:
            autono += 1
            continue

        i, s0, e0 = tline
        s1, e1 = _norm_ts(s0), _norm_ts(e0)
        text = "\n".join(b[i + 1 :]).strip()

        out.append({
            "index": idx,
            "start": s1,
            "end": e1,
            "text": text,
            "start_s": _t2s(s1),
            "end_s": _t2s(e1),
        })
        autono += 1

    return out

def _pick_translation_file(workdir: Path) -> Optional[Path]:
    """
    Prioritas:
      1) edited_translated.srt
      2) translated_latest.srt
      3) translated.srt   (nama lama)
      4) translated_*.srt (terbaru, termasuk subfolder)
    """
    for name in ["edited_translated.srt", "translated_latest.srt", "translated.srt"]:
        p = workdir / name
        if p.exists():
            return p
    cands = list(workdir.glob("translated_*.srt"))
    cands += list(workdir.glob("**/translated_*.srt"))
    if cands:
        return sorted(set(cands), key=lambda x: x.stat().st_mtime, reverse=True)[0]
    return None

def _load_diarization_segments(workdir: Path) -> List[dict]:
    """
    Cari *_gender_*_segments.json (punya fields: start, end, speaker, gender).
    Jika ada speakers.json, pakai sebagai fallback gender per speaker.
    """
    seg_json = None
    for p in sorted(workdir.glob("*_gender_*_segments.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        seg_json = p; break
    if not seg_json:
        for p in sorted(workdir.glob("**/*_gender_*_segments.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            seg_json = p; break
    segments = []
    speaker_gender = {}

    sp_json = None
    for p in sorted(workdir.glob("*_gender_*_speakers.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        sp_json = p; break
    if not sp_json:
        for p in sorted(workdir.glob("**/*_gender_*_speakers.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            sp_json = p; break

    if sp_json:
        try:
            d = json.loads(sp_json.read_text(encoding="utf-8"))
            # {"speakers": {"SPEAKER_04":{"gender":"Male", ...}, ...}}
            for spk, info in (d.get("speakers") or {}).items():
                g = (info.get("gender") or "Unknown").title()
                speaker_gender[spk] = g
        except Exception:
            pass

    if seg_json:
        d = json.loads(seg_json.read_text(encoding="utf-8"))
        for s in d.get("segments", []):
            gender = s.get("gender")
            spk = s.get("speaker")
            if (not gender) and spk and spk in speaker_gender:
                gender = speaker_gender[spk]
            segments.append({
                "start_s": float(s.get("start", 0.0)),
                "end_s": float(s.get("end", 0.0)),
                "speaker": spk or None,
                "gender": (gender or "Unknown").title()
            })
    return segments

def _best_overlap(seg_list: List[dict], a0: float, a1: float) -> Optional[dict]:
    best, best_ov = None, 0.0
    for s in seg_list:
        b0, b1 = s["start_s"], s["end_s"]
        ov = max(0.0, min(a1, b1) - max(a0, b0))
        if ov > best_ov:
            best_ov, best = ov, s
    return best

# ==== Assignment policy & helpers (baru) =====================================

# Cara memilih speaker untuk 1 baris SRT:
# - "overlap"              : pemenang = segmen dengan overlap terbesar (kondisi lama)
# - "start" / "end"        : segmen yang meliputi start / end baris
# - "midpoint"             : segmen yang meliputi titik tengah baris
# - "majority_then_start"  : jika pemenang overlap >= MAJORITY, ambil dia; kalau tidak, fallback ke segmen yang meliputi START
ASSIGN_POLICY = "adaptive"
MAJORITY = 0.60          # 60% durasi baris
SNAP_MS = 120            # jepit start/end baris ke boundary diarization terdekat jika selisih <= 120ms (0 = matikan)
# === NEW: ambang adaptif (tweak kalau perlu) ===
# konstanta untuk adaptive (sudah kamu punya)
MICRO_MS = 450
LONG_MS  = 2500

def _pick_segment(seg_list, a0: float, a1: float):
    """Pilih segmen untuk interval [a0,a1] pakai ASSIGN_POLICY & MAJORITY."""
    if not seg_list:
        return None
    a0 = float(a0); a1 = float(a1)
    dur = max(1e-6, a1 - a0)
    mode = (ASSIGN_POLICY or "overlap").lower().strip()

    # kandidat overlap maksimum
    best, best_ov = None, 0.0
    for s in seg_list:
        b0, b1 = float(s["start_s"]), float(s["end_s"])
        ov = max(0.0, min(a1, b1) - max(a0, b0))
        if ov > best_ov:
            best, best_ov = s, ov

    # policy sederhana
    if mode == "overlap":
        return best
    if mode in ("start", "end", "midpoint"):
        x = a0 if mode=="start" else (a1 if mode=="end" else (a0+a1)/2.0)
        for s in seg_list:
            if float(s["start_s"]) <= x <= float(s["end_s"]):
                return s
        return best
    if mode == "majority_then_start":
        if best and best_ov/dur >= float(MAJORITY): return best
        for s in seg_list:
            if float(s["start_s"]) <= a0 <= float(s["end_s"]): return s
        return best
    if mode == "majority_then_midpoint":
        if best and best_ov/dur >= float(MAJORITY): return best
        x = (a0+a1)/2.0
        for s in seg_list:
            if float(s["start_s"]) <= x <= float(s["end_s"]): return s
        return best

    # adaptive
    if mode == "adaptive":
        dur_ms = (a1 - a0) * 1000.0
        if dur_ms <= MICRO_MS:
            x = (a0+a1)/2.0
            for s in seg_list:
                if float(s["start_s"]) <= x <= float(s["end_s"]): return s
            return best
        if dur_ms >= LONG_MS:
            if best and best_ov/dur >= float(MAJORITY): return best
            for s in seg_list:
                if float(s["start_s"]) <= a0 <= float(s["end_s"]): return s
            return best
        # medium
        if best and best_ov/dur >= float(MAJORITY): return best
        x = (a0+a1)/2.0
        for s in seg_list:
            if float(s["start_s"]) <= x <= float(s["end_s"]): return s
        return best

    # fallback
    return best

def _collect_boundaries(seg_list):
    """Kumpulkan semua batas (start & end) segmen diarization sebagai list float terurut unik."""
    b = []
    for s in seg_list or []:
        try:
            b.append(float(s["start_s"]))
            b.append(float(s["end_s"]))
        except Exception:
            pass
    # unik + terurut
    return sorted({x for x in b})

def _snap_pair(a0: float, a1: float, boundaries: list, snap_ms: int = SNAP_MS):
    """Jepit (a0,a1) ke boundary terdekat jika jarak <= SNAP_MS; anti kolaps interval."""
    if not boundaries or not snap_ms or snap_ms <= 0:
        return a0, a1
    import bisect
    thr = float(snap_ms) / 1000.0

    def snap(x: float) -> float:
        i = bisect.bisect_left(boundaries, x)
        cand = []
        if i > 0: cand.append(boundaries[i-1])
        if i < len(boundaries): cand.append(boundaries[i])
        for c in cand:
            if abs(c - x) <= thr:
                return c
        return x

    x0, x1 = snap(float(a0)), snap(float(a1))
    # jangan sampai kebalik / nol durasi akibat snap
    if x1 <= x0:
        return a0, a1
    return x0, x1

# endpoints.py  — TAB 3
@router.get("/api/session/{session_id}/editing")
def get_editing(
    session_id: str,
    assign_policy: str = Query("adaptive"),
    majority: float = Query(0.60),
    snap_ms: int = Query(120),

    # --- parameter warning (boleh dibiarkan default) ---
    min_ovl_spk: float = Query(0.25),   # ambang overlap SPEAKER minimal → WEAK kalau di bawah OK
    min_ovl_gen: float = Query(0.20),   # ambang overlap GENDER minimal → WEAK kalau di bawah OK
    ok_frac: float     = Query(0.66),   # ≥ 66% dianggap OK
    very_short_ms: int = Query(300),    # baris ≤ 300ms dianggap mikro (rawan)
    gender_mode: str   = Query("segment_only"),  # eksplisit (hanya untuk kejelasan; tidak mengubah logika)
):
    # override konstanta untuk request ini
    global ASSIGN_POLICY, MAJORITY, SNAP_MS
    ASSIGN_POLICY = (assign_policy or "adaptive").strip()
    MAJORITY = float(majority)
    SNAP_MS = int(snap_ms)

    """
    Load data untuk Tab 3 (Editing).

    Prioritas sumber data:
      1) workspaces/<id>/editing_cache.json   (source of truth setelah Save)
      2) SRT terjemahan (_pick_translation_file): edited_translated.srt /
         translated_latest.srt / translated.srt / translated_*.srt
      3) Fallback ke source_video.srt (translation kosong)

    Logika mapping:
      - Waktu SRT = pilar UI (tidak kita ubah).
      - SPEAKER diambil dari jalur speaker (*_speakers.json), GENDER dari jalur gender (*_segments.json) — TERPISAH.
      - Snapping & policy dipakai untuk memilih segmen, bukan mengubah waktu SRT.
      - Warning dihitung per baris (tidak mengubah keputusan), supaya user bisa review manual.
    """
    d = Path("workspaces") / session_id
    if not d.exists():
        raise HTTPException(404, "Session not found")

    cache_p = d / "editing_cache.json"

    # ------------------------------------------------------------------ #
    # 0) (Optional) baca cache (nanti dipakai untuk merge user edits)
    # ------------------------------------------------------------------ #
    cache_rows = []
    if cache_p.exists():
        try:
            payload = json.loads(cache_p.read_text(encoding="utf-8"))
            cache_rows = payload.get("rows", []) or []
        except Exception:
            cache_rows = []

    # ------------------------------------------------------------------ #
    # 1) PARSE SRT TERJEMAHAN (edited_translated / translated_latest / dst)
    # ------------------------------------------------------------------ #
    tran_rows = []
    tp = _pick_translation_file(d)
    if tp:
        try:
            tran_rows = _parse_srt(_read_text(tp))
        except Exception:
            tran_rows = []

    # ------------------------------------------------------------------ #
    # 2) Fallback: pakai timing dari source_video.srt jika tidak ada terjemahan
    # ------------------------------------------------------------------ #
    if not tran_rows:
        orig = _parse_srt(_read_text(d / "source_video.srt"))
        rows = [{
            "index": o["index"],
            "start": o["start"],
            "end": o["end"],
            "original": "",
            "translation": "",
            "speaker": None,
            "gender": "unknown",
            "notes": "",
            "warn_level": "OK",
            "warn_codes": [],
            "frac_spk": None,
            "frac_gen": None,
            "near_start_ms": None,
            "near_end_ms": None,
            "dur_ms": None,
        } for o in orig]
        return JSONResponse({
            "video": f"/api/session/{session_id}/video",
            "rows": rows,
            "speakers": []
        })

    # ------------------------------------------------------------------ #
    # 3) Map diarization SECARA TERPISAH: speaker vs gender
    # ------------------------------------------------------------------ #
    # 3a) muat dua jalur terpisah
    spk_segs_raw = _load_speaker_segments(d)    # [{start/end, speaker}]
    gen_segs_raw = _load_gender_segments(d)     # [{start/end, gender}]

    # fallback: kalau salah satu kosong, pakai segs gabungan lama
    base_segs = _load_diarization_segments(d)   # gabungan lama (bisa ada 'speaker' dan 'gender')
    if (not spk_segs_raw) and base_segs:
        tmp = []
        for s in base_segs or []:
            try:
                tmp.append({
                    "start": float(s.get("start", s.get("start_s", 0.0))),
                    "end":   float(s.get("end",   s.get("end_s",   0.0))),
                    "speaker": s.get("speaker"),
                })
            except:
                pass
        spk_segs_raw = tmp

    if (not gen_segs_raw) and base_segs:
        tmp = []
        for s in base_segs or []:
            try:
                g = (s.get("gender") or "unknown").lower()
                if g not in ("male","female","unknown"):
                    g = "unknown"
                tmp.append({
                    "start": float(s.get("start", s.get("start_s", 0.0))),
                    "end":   float(s.get("end",   s.get("end_s",   0.0))),
                    "gender": g,
                })
            except:
                pass
        gen_segs_raw = tmp

    # 3b) NORMALISASI: pastikan semua pakai start_s/end_s (AMAN utk _collect_boundaries/_pick_segment lama)
    spk_segs = _norm_seg_list(spk_segs_raw, keep=["speaker"])
    gen_segs = _norm_seg_list(gen_segs_raw, keep=["gender"])

    # 3c) siapkan batas untuk snapping masing-masing
    spk_bound = _collect_boundaries(spk_segs) if spk_segs else []
    gen_bound = _collect_boundaries(gen_segs) if gen_segs else []

    rows = []
    for t in tran_rows:
        a0 = float(t.get("start_s", 0.0))
        a1 = float(t.get("end_s", 0.0))
        dur_s = max(1e-6, a1 - a0)

        # SNAP terpisah
        a0_spk, a1_spk = _snap_pair(a0, a1, spk_bound, SNAP_MS) if spk_segs else (a0, a1)
        a0_gen, a1_gen = _snap_pair(a0, a1, gen_bound, SNAP_MS) if gen_segs else (a0, a1)

        # PICK segmen terpisah (policy berlaku untuk keduanya)
        s_spk = _pick_segment(spk_segs, a0_spk, a1_spk) if spk_segs else None
        s_gen = _pick_segment(gen_segs, a0_gen, a1_gen) if gen_segs else None

        speaker = (s_spk.get("speaker") if s_spk else None)

        # Gender "segment_only": pakai hasil jalur gender apa adanya
        gg = ((s_gen.get("gender") if s_gen else "unknown") or "unknown").lower()
        if gg not in ("male", "female", "unknown"):
            gg = "unknown"

        rows.append({
            "index": int(t["index"]),
            "start": t["start"],  # UI tetap pakai timestamp SRT; mapping pakai a0*/a1* yang sudah snap
            "end":   t["end"],
            "original": "",
            "translation": t.get("text") or "",
            "speaker": speaker,
            "gender": gg,
            "notes": "",
            # placeholder warning (akan dihitung di step 5)
            "warn_level": "OK",
            "warn_codes": [],
            "frac_spk": None,
            "frac_gen": None,
            "near_start_ms": None,
            "near_end_ms": None,
            "dur_ms": int(round(dur_s*1000)),
        })

    # ------------------------------------------------------------------ #
    # 4) Merge dengan cache jika ada (update field; tambahkan row yang tidak ada)
    # ------------------------------------------------------------------ #
    if cache_rows:
        by_idx = {r["index"]: r for r in rows}
        for r in cache_rows:
            try:
                idx = int(r.get("index") or 0)
            except:
                continue
            if idx in by_idx:
                # timpa value yang pernah diedit user
                for k in ("translation", "speaker", "gender", "notes", "start", "end"):
                    if r.get(k) is not None:
                        if k == "gender":
                            gg = (r.get(k) or "unknown").lower()
                            by_idx[idx][k] = gg if gg in ("male", "female", "unknown") else "unknown"
                        else:
                            by_idx[idx][k] = r.get(k)
            else:
                gg = (r.get("gender") or "unknown").lower()
                if gg not in ("male", "female", "unknown"):
                    gg = "unknown"
                by_idx[idx] = {
                    "index": idx,
                    "start": str(r.get("start") or ""),
                    "end": str(r.get("end") or ""),
                    "original": "",
                    "translation": r.get("translation") or "",
                    "speaker": (r.get("speaker") or None),
                    "gender": gg,
                    "notes": r.get("notes") or "",
                    "warn_level": "OK",
                    "warn_codes": [],
                    "frac_spk": None,
                    "frac_gen": None,
                    "near_start_ms": None,
                    "near_end_ms": None,
                    "dur_ms": None,
                }
        rows = [by_idx[k] for k in sorted(by_idx.keys())]

    # ------------------------------------------------------------------ #
    # 5) HITUNG WARNING per baris (tanpa mengubah keputusan)
    # ------------------------------------------------------------------ #
    def _hms_to_s(h: str) -> float:
        hh, mm, rest = h.split(":")
        ss, ms = rest.replace(".", ",").split(",")
        return (int(hh)*3600 + int(mm)*60 + int(ss)) + (int(ms)/1000.0)

    def _ovl(a0: float, a1: float, b0: float, b1: float) -> float:
        x = min(a1, b1) - max(a0, b0)
        return x if x > 0 else 0.0

    # gabung semua boundary untuk metrik "mepet"
    all_bounds = []
    for ss in spk_segs or []:
        all_bounds.append(float(ss["start_s"])); all_bounds.append(float(ss["end_s"]))
    for gg in gen_segs or []:
        all_bounds.append(float(gg["start_s"])); all_bounds.append(float(gg["end_s"]))

    for r in rows:
        # parse waktu SRT baris
        try:
            a0 = _hms_to_s(r["start"]); a1 = _hms_to_s(r["end"])
        except Exception:
            continue
        dur = max(1e-6, a1 - a0)

        # overlap SPEAKER berdasarkan speaker final di row
        frac_spk = 0.0
        if r.get("speaker") and spk_segs:
            ov = 0.0
            for ss in spk_segs:
                if ss.get("speaker") == r["speaker"]:
                    ov += _ovl(a0, a1, float(ss["start_s"]), float(ss["end_s"]))
            frac_spk = ov / dur

        # overlap GENDER berdasarkan gender final di row
        frac_gen = 0.0
        if r.get("gender") and gen_segs:
            ov = 0.0
            for gg in gen_segs:
                if gg.get("gender") == r["gender"]:
                    ov += _ovl(a0, a1, float(gg["start_s"]), float(gg["end_s"]))
            frac_gen = ov / dur

        # dekat boundary? (mepet ≤ SNAP_MS)
        near_start = min((abs(b - a0) for b in all_bounds), default=None)
        near_end   = min((abs(b - a1) for b in all_bounds), default=None)

        warn_codes = []

        # Speaker quality
        if frac_spk >= ok_frac:
            pass
        elif frac_spk >= min_ovl_spk:
            warn_codes.append("SPK_WEAK")
        else:
            warn_codes.append("SPK_BAD")

        # Gender quality
        if frac_gen >= ok_frac:
            pass
        elif frac_gen >= min_ovl_gen:
            warn_codes.append("GEN_WEAK")
        else:
            warn_codes.append("GEN_BAD")

        # Mepet boundary
        if near_start is not None and near_start <= (SNAP_MS/1000.0): warn_codes.append("MEPET_START")
        if near_end   is not None and near_end   <= (SNAP_MS/1000.0): warn_codes.append("MEPET_END")

        # Mikro
        if (dur*1000.0) <= very_short_ms: warn_codes.append("VERY_SHORT")

        warn_level = "OK"
        if any(c in warn_codes for c in ("SPK_BAD","GEN_BAD")):
            warn_level = "ALERT"
        elif warn_codes:
            warn_level = "WARN"

        r["warn_level"]    = warn_level
        r["warn_codes"]    = warn_codes
        r["frac_spk"]      = round(float(frac_spk), 3)
        r["frac_gen"]      = round(float(frac_gen), 3)
        r["near_start_ms"] = int(round((near_start or 0.0)*1000))
        r["near_end_ms"]   = int(round((near_end   or 0.0)*1000))
        r["dur_ms"]        = int(round(dur*1000))

    # ------------------------------------------------------------------ #
    # 6) Final
    # ------------------------------------------------------------------ #
    speakers = sorted({(r.get("speaker") or "").strip() for r in rows if r.get("speaker")})
    return JSONResponse({
        "video": f"/api/session/{session_id}/video",
        "rows": rows,
        "speakers": speakers
    })


class Row(BaseModel):
    index: int
    start: str
    end: str
    original: Optional[str] = None
    translation: Optional[str] = None
    speaker: Optional[str] = None
    gender: str = "unknown"
    notes: Optional[str] = None


class EditSave(BaseModel):
    rows: list[Row]


@router.post("/api/session/{session_id}/editing")
def post_editing(session_id: str, data: EditSave):
    d = Path("workspaces") / session_id
    d.mkdir(parents=True, exist_ok=True)

    # simpan cache editor
    (d / "editing_cache.json").write_text(
        json.dumps({"rows": [r.dict() for r in data.rows]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # generate SRT versi edited
    srt_lines, n = [], 1
    for r in data.rows:
        srt_lines.append(str(n))
        srt_lines.append(f"{r.start} --> {r.end}")
        srt_lines.append((r.translation or "").strip())
        srt_lines.append("")
        n += 1
    txt = "\n".join(srt_lines).rstrip() + "\n"
    (d / "edited_translated.srt").write_text(txt, encoding="utf-8")
    # opsional: jaga kompatibilitas
    (d / "translated_latest.srt").write_text(txt, encoding="utf-8")

    return {"status": "ok"}

class ExportReq(BaseModel):
    # mode lama: male|female|unknown|all ; mode baru: speaker|speaker_zip
    mode: Literal['male','female','unknown','all','speaker','speaker_zip','full'] = 'male'
    reindex: bool = True
    speaker: Optional[str] = None

def _build_srt(rows, reindex=True):
    out = []
    n = 1
    for r in rows:
        out.append(str(n if reindex else r["index"]))
        out.append(f'{r["start"]} --> {r["end"]}')
        out.append((r.get("translation") or "").strip())
        out.append("")
        n += 1
    return "\r\n".join(out)

def _safe_name(text: str) -> str:
    # aman untuk nama file, hilangkan karakter aneh
    t = (text or "unknown").strip()
    t = re.sub(r"\s+", "_", t)
    t = re.sub(r"[^\w\-\.]+", "", t)
    return t or "unknown"
    
@router.post("/api/session/{session_id}/editing/export")
def export_editing(session_id: str, req: ExportReq):
    d = Path("workspaces") / session_id
    cache_p = d / "editing_cache.json"
    if not cache_p.exists():
        raise HTTPException(400, "No editing data")

    payload = json.loads(cache_p.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])

    def sel_gender(g: str):
        g = (g or "unknown").lower()
        return [r for r in rows if (r.get("gender", "unknown").lower() == g)]

    def sel_speaker(spk: str):
        spk = (spk or "").strip()
        return [r for r in rows if (r.get("speaker") or "").strip() == spk]

    # --- gender (1 file) ---
    if req.mode in ("male", "female", "unknown"):
        srt = _build_srt(sel_gender(req.mode), req.reindex)
        out = d / f"{req.mode}_gender.srt"
        out.write_text(srt, encoding="utf-8")
        return FileResponse(out, media_type="text/plain", filename=out.name)

    # --- speaker (1 file) ---
    elif req.mode == "speaker":
        if not req.speaker:
            raise HTTPException(400, "Missing 'speaker' for mode=speaker")
        group = sel_speaker(req.speaker)
        if not group:
            raise HTTPException(404, f"No rows for speaker: {req.speaker}")
        srt = _build_srt(group, req.reindex)
        safe = _safe_name(req.speaker)
        out = d / f"{safe}.srt"
        out.write_text(srt, encoding="utf-8")
        return FileResponse(out, media_type="text/plain", filename=out.name)

    elif req.mode == "full":
        srt = _build_srt(rows, req.reindex)
        out = d / f"{session_id}_full.srt"
        out.write_text(srt, encoding="utf-8")
        return FileResponse(out, media_type="text/plain", filename=out.name)
    
    # --- speaker_zip (banyak file, zip) ---
    elif req.mode == "speaker_zip":
        speakers = sorted({
            (r.get("speaker") or "").strip()
            for r in rows
            if (r.get("speaker") or "").strip()
        })
        if not speakers:
            raise HTTPException(404, "No speakers in data")

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as zf:
            for spk in speakers:
                group = sel_speaker(spk)
                if not group:
                    continue
                srt = _build_srt(group, req.reindex)
                arc = f"{_safe_name(spk)}.srt"
                zf.writestr(arc, srt)

        return FileResponse(
            tmp.name,
            media_type="application/zip",
            filename=f"{session_id}_srt_by_speaker.zip",
        )

    # --- all (by gender -> zip tiga file) ---
    elif req.mode == "all":
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as zf:
            for g in ("male", "female", "unknown"):
                srt = _build_srt(sel_gender(g), req.reindex)
                zf.writestr(f"{g}_gender.srt", srt)

        return FileResponse(
            tmp.name,
            media_type="application/zip",
            filename=f"{session_id}_srt_by_gender.zip",
        )

    else:
        raise HTTPException(400, f"Unknown export mode: {req.mode}")

# =============================================================================
# TAB 4: TTS & Export (trigger proses dubbing/replace audio)
# =============================================================================

@router.post("/api/session/{session_id}/tts-export")
async def run_tts_export(
    session_id: str,
    tts_engine: str = Form("edge"),
    voice_male: str = Form(...),
    voice_female: str = Form(...),
    voice_unknown: str = Form(...),
    replace_audio: bool = Form(False),
    bg_mode: str = Form("center_cut"),
    pm = Depends(get_processing_manager),
):
    try:
        config = {
            "tts_engine": tts_engine,
            "voice_male": voice_male,
            "voice_female": voice_female,
            "voice_unknown": voice_unknown,
            "replace_audio": replace_audio,
            "bg_mode": bg_mode,
        }
        await pm.run_tts_export(session_id, config)
        return JSONResponse({"status": "tts_export_started"})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ===== Review & Export (CapCut) – Tahap 1,2,3 =====

# -- ambil processing_manager dari projectmu
def get_processing_manager():
    from core.processor import processing_manager
    return processing_manager

# -- util workspace
def _ws(pm, session_id: str) -> Path:
    sess = pm.get_session(session_id)
    if not sess or not getattr(sess, "workdir", None):
        raise HTTPException(404, "Session/workspace tidak ditemukan")
    ws = Path(sess.workdir).resolve()
    if not ws.exists():
        raise HTTPException(404, "Workspace path tidak ada")
    return ws

def _video(pm, session_id: str) -> Path:
    sess = pm.get_session(session_id)
    if not sess or not getattr(sess, "video_path", None):
        raise HTTPException(404, "Video tidak ditemukan")
    p = Path(sess.video_path).resolve()
    if not p.exists():
        raise HTTPException(404, "Video path tidak ada")
    return p

# -- wsfile: serve file dari workdir
@router.get("/api/session/{session_id}/wsfile")
def wsfile(session_id: str, rel: str = Query(...), pm=Depends(get_processing_manager)):
    ws = _ws(pm, session_id)
    p = (ws / rel).resolve()
    if ws != p and ws not in p.parents:
        raise HTTPException(403, "Invalid path")
    if not p.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(str(p))

# ---------- SRT cache -> rows (index,start_ms,end_ms,text,gender) ----------
def _hms_to_ms(h: str) -> int:
    hh, mm, rest = h.split(":")
    ss, ms = rest.replace(".", ",").split(",")
    return (int(hh)*3600 + int(mm)*60 + int(ss))*1000 + int(ms)

def _load_rows(ws: Path) -> List[dict]:
    ec = ws / "editing_cache.json"
    if not ec.exists():
        raise HTTPException(404, "editing_cache.json tidak ada (Load Session dulu di tab Editing)")
    data = json.loads(ec.read_text(encoding="utf-8", errors="ignore"))
    rows = []
    for r in (data.get("rows") or []):
        try:
            rows.append({
                "index": int(r["index"]),
                "start_ms": _hms_to_ms(r["start"]),
                "end_ms": _hms_to_ms(r["end"]),
                "text": r.get("translation") or r.get("text") or "",
                "gender": (r.get("gender") or "").lower()
            })
        except:
            pass
    rows.sort(key=lambda x: x["index"])
    return rows

# ---------- parser CapCut Project (textReading) ----------
def _normalize_project_dir(p: Path) -> Path:
    p = p.resolve()
    if (p / "draft_content.json").exists():
        return p
    cands = [d for d in p.iterdir() if d.is_dir() and (d / "draft_content.json").exists()]
    if not cands:
        raise FileNotFoundError("draft_content.json tidak ditemukan")
    cands.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return cands[0]

def _resolve_capcut_path(base: Path, s: str) -> Path:
    s = (s or "").replace("\\", "/")
    s = re.sub(r"^#+[^/]*draft[^/]*placeholder[^/]*#+/?", "", s, flags=re.I)
    s = re.sub(r"^project://", "", s, flags=re.I)
    p = Path(s)
    if p.is_absolute(): return p
    if s.startswith("/"): s = s[1:]
    return base / s

def _find_wav_in_node(base: Path, n: dict) -> Optional[Path]:
    for k in ("path","local_material_path","local_path","file_path"):
        v = n.get(k)
        if isinstance(v, str) and v.lower().endswith(".wav"):
            return _resolve_capcut_path(base, v)
    for k in ("resource","material_resource","file","source"):
        v = n.get(k)
        if isinstance(v, dict):
            p = _find_wav_in_node(base, v)
            if p: return p
    return None

def _us_or_ms_to_ms(x: Any) -> int:
    if x is None: return 0
    try: x = float(x)
    except: return 0
    return int(x/1000) if x > 1e6 else int(x)

def _parse_capcut_items(project_dir: Path) -> List[dict]:
    prj = _normalize_project_dir(project_dir)
    dc = prj / "draft_content.json"
    data = json.loads(dc.read_text(encoding="utf-8", errors="ignore"))
    base = prj

    mats: Dict[str, dict] = {}
    def walk_collect(n):
        if isinstance(n, dict):
            mid = str(n.get("id") or n.get("material_id") or n.get("materialId") or "")
            wav = _find_wav_in_node(base, n)
            if mid and wav and wav.suffix.lower()==".wav":
                mats[mid] = {
                    "wav": wav,
                    "speaker": n.get("tone_speaker") or n.get("speaker") or "",
                    "name": n.get("name") or n.get("content") or ""
                }
            for v in n.values(): walk_collect(v)
        elif isinstance(n, list):
            for v in n: walk_collect(v)
    walk_collect(data)

    segs: List[dict] = []
    def walk_segments(n):
        if isinstance(n, dict):
            t = (n.get("type") or n.get("track_type") or "").lower()
            if t=="audio" and isinstance(n.get("segments"), list):
                for s in n["segments"]:
                    tid = s.get("material_id") or s.get("materialId") or s.get("refMaterialId")
                    tr  = s.get("target_timerange") or {}
                    st  = _us_or_ms_to_ms(tr.get("start") or tr.get("startUs") or 0)
                    du  = _us_or_ms_to_ms(tr.get("duration") or tr.get("durationUs") or 0)
                    if tid:
                        segs.append({"start_ms": st, "dur_ms": du, "mat_id": str(tid)})
            for v in n.values(): walk_segments(v)
        elif isinstance(n, list):
            for v in n: walk_segments(v)
    walk_segments(data)
    segs.sort(key=lambda x: x["start_ms"])

    items = []
    for s in segs:
        m = mats.get(s["mat_id"])
        if not m: continue
        items.append({
            "start_ms": s["start_ms"],
            "dur_ms": s["dur_ms"],
            "speaker": m.get("speaker",""),
            "name": m.get("name",""),
            "wav": Path(m["wav"])
        })
    return items

# ---------- Tahap-1: IMPORT (copy-only ke raw/all), simpan items + map ----------
# =============================
# CapCut helpers (collect & map)
# =============================

class CapcutImportBody(BaseModel):
    project_dir: str
    offset:     int = 0        # offset ms untuk menggeser textReading
    tol:        int = 1000     # toleransi pencocokan waktu
    pre_cut_ms: int = 0        # trim head (ms) saat menulis WAV workspace
    post_cut_ms:int = 0        # trim tail (ms) saat menulis WAV workspace
    max_speed:  float = 3.0    # (disimpan saja untuk tahap export/build timeline)

def _capcut__pick_srt_for_rows(workdir: Path) -> Optional[Path]:
    # pilih SRT yang jadi dasar rows (urut prioritas)
    for name in ["edited_translated.srt", "translated_latest.srt",
                 "translated.srt", "source_subtitles.srt", "source_video.srt"]:
        p = workdir / name
        if p.exists():
            return p
    # fallback cari translated_*.srt
    cands = list(workdir.glob("translated_*.srt")) + list(workdir.glob("**/translated_*.srt"))
    if cands:
        cands = sorted(cands, key=lambda x: x.stat().st_mtime, reverse=True)
        return cands[0]
    return None

def _capcut__parse_srt_to_rows(srt_text: str) -> List[Dict[str, Any]]:
    # parse SRT sederhana → rows: [{index,start,end,text}], sorted by start
    text = srt_text.replace("\r", "")
    blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]
    out = []
    def _ms(hms, ms):
        h, m, s = map(int, hms.split(":"))
        return h*3600000 + m*60000 + s*1000 + int(ms)
    for blk in blocks:
        lines = [x for x in blk.split("\n") if x.strip() != ""]
        if len(lines) < 2: continue
        # index opsional
        try:
            idx = int(lines[0].strip())
            tline = lines[1]; pos = 2
        except Exception:
            idx = None
            tline = lines[0]; pos = 1
        m = re.search(r"(\d\d:\d\d:\d\d)[,\.](\d{3})\s*-->\s*(\d\d:\d\d:\d\d)[,\.](\d{3})", tline)
        if not m: continue
        s_ms = _ms(m.group(1), m.group(2))
        e_ms = _ms(m.group(3), m.group(4))
        txt  = "\n".join(lines[pos:]).strip()
        if idx is None:
            idx = (out[-1]["index"]+1) if out else 1
        out.append({"index": idx, "start": s_ms, "end": e_ms, "text": txt})
    out.sort(key=lambda r: (r["start"], r["end"], r["index"]))
    return out

def _capcut__collect_textreading_items(project_dir: Path) -> List[Dict[str, Any]]:
    """
    Baca draft_content.json → daftar item TTS:
    Mendukung 2 skema:
     A) materials.textReadings + tracks.segments.targetTimerange (ms)
     B) tracks[type=='audio'].segments.material_id → materials.audios[id].path
        (path mengandung '.../textReading/xxx.wav'; waktu mikrodetik → ms)
    Return: [{start,end,path}], path absolut.
    """
    dc = project_dir / "draft_content.json"
    if not dc.exists(): return []
    try:
        doc = json.loads(dc.read_text(encoding="utf-8"))
    except Exception:
        return []

    items: List[Dict[str, Any]] = []
    mats  = (doc.get("materials") or {})

    # --- VARIAN A: textReadings ---
    id2path_a: Dict[str, Path] = {}
    tr_list = mats.get("textReadings") or mats.get("text_readings") or []
    for m in tr_list:
        mid = str(m.get("id") or m.get("mid") or "")
        pth = m.get("path") or m.get("filePath") or ""
        if not (mid and pth): continue
        p = (project_dir / pth)
        if p.suffix.lower()==".wav" and p.exists():
            id2path_a[mid] = p

    if id2path_a:
        for t in (doc.get("tracks") or []):
            for sg in (t.get("segments") or []):
                mid = str(sg.get("materialId") or sg.get("material_id") or "")
                if not (mid and mid in id2path_a): continue
                tr = sg.get("targetTimerange") or sg.get("target_timerange") or {}
                st = tr.get("start"); du = tr.get("duration")
                if st is None or du is None: continue
                try:
                    st = int(st); du = int(du); en = st+du
                except Exception:
                    continue
                items.append({"start": st, "end": en, "path": id2path_a[mid]})

    # --- VARIAN B: audio tracks (umum di CapCut versi terbaru) ---
    audios = (mats.get("audios") or [])
    id2audio = {str(m.get("id")): m for m in audios if m and m.get("id")}

    def _resolve_audio_path(raw: str) -> Optional[Path]:
        if not raw: return None
        raw = str(raw)
        if "textReading" in raw:
            sub = raw.split("textReading", 1)[1].lstrip("/\\")
            return project_dir / "textReading" / sub
        return project_dir / raw

    for t in (doc.get("tracks") or []):
        if (t.get("type") or "").lower() != "audio":
            continue
        for sg in (t.get("segments") or []):
            mid = str(sg.get("material_id") or sg.get("materialId") or "")
            m = id2audio.get(mid)
            if not m: continue
            p = _resolve_audio_path(m.get("path") or "")
            if not p or p.suffix.lower()!=".wav" or not p.exists():
                continue
            tr = sg.get("target_timerange") or sg.get("targetTimerange") \
                 or sg.get("source_timerange") or {}
            st_us = tr.get("start"); du_us = tr.get("duration")
            if st_us is None or du_us is None: continue
            try:
                st_ms = int(round(int(st_us)/1000))
                du_ms = int(round(int(du_us)/1000))
            except Exception:
                continue
            items.append({"start": st_ms, "end": st_ms+du_ms, "path": p})

    # dedup + sort
    uniq = {}
    for it in items:
        key = (int(it["start"]), int(it["end"]), str(Path(it["path"]).resolve()))
        uniq[key] = it
    items = list(uniq.values())
    items.sort(key=lambda x: (x["start"], x["end"], str(x["path"])))
    return items

def _assign_tts_strict(rows: List[Dict[str, Any]],
                       items: List[Dict[str, Any]],
                       offset_ms: int = 0,
                       tol_ms: int = 1000) -> List[Any]:
    """Greedy 1:1 mapping baris SRT → item TTS by waktu (tahan lewati)."""
    items = sorted(items, key=lambda x: (int(x["start"]), int(x["end"])))
    rows  = sorted(rows,  key=lambda x: (int(x["start"]), int(x["end"]), int(x["index"])))
    maps = []
    j, n = 0, len(items)
    for r in rows:
        rs = int(r["start"])
        while j < n and (int(items[j]["end"]) + offset_ms) < (rs - tol_ms):
            j += 1
        if j < n:
            ds = (int(items[j]["start"]) + offset_ms) - rs
            if abs(ds) <= tol_ms:
                maps.append((r["index"], items[j])); j += 1; continue
        maps.append((r["index"], None))
    return maps

def _fix_off_by_one(rows: List[Dict[str, Any]],
                    items: List[Dict[str, Any]],
                    offset_ms: int = 0,
                    tol_ms: int = 1000) -> List[Any]:
    """Perbaiki pola 'baris pertama None, sisanya beruntun cocok' → rotasi 1."""
    initial = _assign_tts_strict(rows, items, offset_ms, tol_ms)
    if not initial: return initial
    first_none  = (initial[0][1] is None)
    tail_filled = sum(1 for _, it in initial[1:] if it is not None)
    if first_none and tail_filled >= max(1, min(len(items), len(rows)) - 2):
        items_rot = items[1:] + items[:1]
        return _assign_tts_strict(rows, items_rot, offset_ms, tol_ms)
    return initial


@router.post("/api/session/{session_id}/capcut/import")
def capcut_import(
    session_id: str,
    body: CapcutImportBody,
    pm = Depends(get_processing_manager),
):
    """
    Import WAV TTS dari CapCut Project:
      - baca SRT session → rows
      - baca draft_content.json → items (textReading)
      - map rows ↔ items (offset/tol + fix off-by-one)
      - SELALU tulis WAV ke workspace: workspaces/<sid>/capcut/trim/00001.wav dst
      - simpan manifest capcut_map_all.json (dipakai review/preview)
    """
    session = pm.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    workdir = Path(session.workdir).resolve()
    if not workdir.exists():
        raise HTTPException(status_code=404, detail="Workspace not found")

    project_dir = Path(body.project_dir).resolve()
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="Project dir not found")

    # 1) rows dari SRT
    srt_path = _capcut__pick_srt_for_rows(workdir)
    if not srt_path or not srt_path.exists():
        raise HTTPException(status_code=404, detail="SRT for this session not found")
    rows = _capcut__parse_srt_to_rows(srt_path.read_text(encoding="utf-8"))
    if not rows:
        raise HTTPException(status_code=400, detail="Parsed SRT is empty")

    # 2) items dari project (dua skema didukung)
    items = _capcut__collect_textreading_items(project_dir)
    if not items:
        raise HTTPException(status_code=404, detail="Tidak ada WAV textReading pada project ini")

    # 3) mapping
    offset_ms = int(body.offset or 0)
    tol_ms    = int(body.tol or 0)
    mapping   = _fix_off_by_one(rows, items, offset_ms=offset_ms, tol_ms=tol_ms)

    # 4) tulis WAV ke workspace (selalu)
    capcut_dir = workdir / "capcut"
    trim_dir = capcut_dir / "trim"
    trim_dir.mkdir(parents=True, exist_ok=True)

    manifest_items: List[Dict[str, Any]] = []
    copied = 0

    # buat index → row untuk durasi (untuk trimming)
    idx2row = {r["index"]: r for r in rows}

    for row_idx, it in mapping:
        if it is None:
            continue
        src = Path(it["path"]).resolve()
        if not src.exists():
            continue

        # nama konsisten 5 digit (00001.wav)
        out_wav = trim_dir / f"{int(row_idx):05d}.wav"

        # trimming jika diminta, kalau tidak → copy
        pre = int(body.pre_cut_ms or 0)
        post= int(body.post_cut_ms or 0)
        row = idx2row.get(int(row_idx))
        if row:
            slot_ms = max(0, (row["end"] - row["start"]))
        else:
            slot_ms = max(0, int(it["end"]) - int(it["start"]))
        cut_ms = max(0, slot_ms - pre - post)

        if pre>0 or post>0:
            # pakai ffmpeg
            cmd = [
                "ffmpeg","-y",
                "-i", str(src),
                "-ss", f"{pre/1000.0:.3f}",
                "-t",  f"{cut_ms/1000.0:.3f}",
                "-c:a","pcm_s16le",
                str(out_wav)
            ]
            _run(cmd)
        else:
            shutil.copy2(src, out_wav)

        manifest_items.append({
            "row_index": int(row_idx),
            "src": str(src),
            "trim": f"capcut/trim/{out_wav.name}",
            "start_ms": int(it["start"]),
            "end_ms": int(it["end"]),
        })
        copied += 1

    # 5) simpan manifest & metadata import
    (capcut_dir / "capcut_map_all.json").write_text(
        json.dumps({"items": manifest_items}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    pm.session_manager.update_session(
        session_id,
        capcut_import={
            "project_dir": str(project_dir),
            "offset": offset_ms,
            "tol": tol_ms,
            "pre_cut_ms": int(body.pre_cut_ms or 0),
            "post_cut_ms": int(body.post_cut_ms or 0),
            "max_speed": float(body.max_speed or 3.0),
        },
    )

    return JSONResponse({
        "ok": True,
        "mapped": len(manifest_items),
        "total_rows": len(rows),
        "total_items": len(items),
        "copied": copied,
    })

    
# ---------- Tahap-2: Review (rows + tts_path) ----------
@router.get("/api/session/{session_id}/review")
def review_rows(session_id: str, pm = Depends(get_processing_manager)):
    """
    Kembalikan rows untuk tab Review & Export:
      [{index,start,end,text,gender,tts_path}]
    tts_path diambil dari capcut_map_all.json (field 'trim') dan disajikan via wsfile.
    """
    session = pm.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    workdir = Path(session.workdir).resolve()

    # --- ambil rows dari SRT (sama seperti import)
    srt_path = _capcut__pick_srt_for_rows(workdir)
    if not srt_path or not srt_path.exists():
        return JSONResponse({"rows": []})
    rows = _capcut__parse_srt_to_rows(srt_path.read_text(encoding="utf-8"))

    # --- baca manifest capcut
    capcut_dir = workdir / "capcut"
    idx_to_rel: Dict[int, str] = {}
    man = capcut_dir / "capcut_map_all.json"
    if man.exists():
        try:
            m = json.loads(man.read_text(encoding="utf-8"))
            for it in m.get("items", []):
                idx = int(it.get("row_index", -1))
                rel = it.get("trim") or it.get("src")
                if idx >= 0 and rel:
                    idx_to_rel[idx] = str(rel)
        except Exception:
            pass

    # --- format waktu
    def _fmt_ms_local(ms: int) -> str:
        try:
            ms = int(ms)
        except Exception:
            return "00:00,000"
        s = ms // 1000
        h = s // 3600
        m = (s % 3600) // 60
        ss = s % 60
        return f"{h:02d}:{m:02d}:{ss:02d},{ms%1000:03d}"

    out = []
    for r in rows:
        i = int(r["index"])
        rel = idx_to_rel.get(i)
        if rel and rel.startswith("capcut/"):
            rel_q = quote(str(Path(rel)).replace("\\", "/"))
            tts_url = f"/api/session/{session_id}/wsfile?rel={rel_q}"
        else:
            tts_url = None

        # r["start"] dan r["end"] sudah ms (int) dari parser
        out.append({
            "index": i,
            "start": _fmt_ms_local(r["start"]),
            "end":   _fmt_ms_local(r["end"]),
            "start_ms": int(r["start"]),
            "end_ms":   int(r["end"]),
            "text":  r.get("text", ""),
            "gender": r.get("gender", ""),
            "tts_path": tts_url,
        })

    return JSONResponse({"rows": out})


    out = []
    for r in rows:
        i = int(r["index"])
        rel = idx_to_rel.get(i)
        if rel and rel.startswith("capcut/"):
            rel_q = quote(rel.replace("\\","/"))
            tts_url = f"/api/session/{session_id}/wsfile?rel={rel_q}"
        else:
            tts_url = None
        out.append({
            "index": i,
            "start": _fmt_ms_local(r["start"]),
            "end":   _fmt_ms_local(r["end"]),
            "text":  r.get("text",""),
            "gender": "",         # opsional: isi jika kamu punya mapping gender
            "tts_path": tts_url,  # dipakai tombol Play
        })

    return JSONResponse({"rows": out})


# nudge satu baris / global
@router.post("/api/session/{session_id}/timeline/update")
def timeline_update(session_id: str, body: Dict[str, Any], pm=Depends(get_processing_manager)):
    ws = _ws(pm, session_id)
    adjf = ws / "timeline_adjust.json"
    data = json.loads(adjf.read_text()) if adjf.exists() else {"shift": {}}
    shift = data.get("shift") or {}

    if "index" in body:
        idx = str(int(body["index"]))
        delta = int(body.get("delta_ms") or 0)
        shift[idx] = int(shift.get(idx) or 0) + delta

    if "global_delta_ms" in body:
        g = int(body["global_delta_ms"])
        # tambah ke semua yg sudah punya shift, baris lain tetap 0+g (disimpan saat disentuh)
        for k in list(shift.keys()):
            shift[k] = int(shift[k]) + g
        data["global_offset_ms"] = int(data.get("global_offset_ms") or 0) + g

    data["shift"] = shift
    adjf.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True}

# deteksi offset median dari 20 pasang pertama
def _parse_ts_ms(ts: str) -> int:
    m = re.match(r"^\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*$", str(ts) if ts is not None else "")
    if not m: return 0
    h, mm, ss, ms = map(int, m.groups())
    return ((h*3600 + mm*60 + ss) * 1000) + ms

def _row_start_ms(row: Dict[str, Any]) -> int:
    v = row.get("start_ms", None)
    if v is not None:
        try: return int(v)
        except Exception: pass
    return _parse_ts_ms(row.get("start"))
    
@router.post("/api/session/{session_id}/capcut/detect_offset")
def detect_offset(session_id: str, pm = Depends(get_processing_manager)):
    ws = _ws(pm, session_id)

    # Ambil rows (pakai util kamu yang sudah dipakai luas)
    rows = _load_rows(ws)
    if not rows:
        raise HTTPException(400, "Rows kosong / belum load editing")

    # Manifest baru hasil import
    man_new = ws / "capcut" / "capcut_map_all.json"
    items = []
    if man_new.exists():
        data = json.loads(man_new.read_text(encoding="utf-8"))
        items = data.get("items", [])
    else:
        # Fallback ke format lama (biar backward compatible)
        man_old = ws / "capcut_items.json"
        if man_old.exists():
            items = json.loads(man_old.read_text(encoding="utf-8"))
        else:
            raise HTTPException(400, "Import belum dilakukan")

    # Pemetaan by index untuk stabilitas
    idx2row = {}
    for r in rows:
        try:
            idx2row[int(r.get("index"))] = r
        except Exception:
            pass

    deltas = []
    for it in items:
        # di manifest baru: row_index; di lama tidak ada → pakai urutan
        if "row_index" in it:
            idx = int(it.get("row_index", -1))
            r = idx2row.get(idx)
        else:
            # format lama: posisional
            i = len(deltas)
            if i >= len(rows):
                break
            r = rows[i]
        if not r:
            continue

        s_ms = _row_start_ms(r)
        i_ms = int(it.get("start_ms", 0))
        deltas.append(i_ms - s_ms)
        if len(deltas) >= 20:
            break

    if not deltas:
        raise HTTPException(400, "Belum ada data untuk deteksi")

    deltas.sort()
    n = len(deltas)
    med = deltas[n//2] if (n % 2 == 1) else int((deltas[n//2-1] + deltas[n//2]) / 2)
    return {"suggested_offset_ms": med, "pairs_used": n}

# remap strict by-index memakai offset & tolerance
@router.post("/api/session/{session_id}/capcut/remap")
def remap(session_id: str, body: Dict[str, Any], pm = Depends(get_processing_manager)):
    ws = _ws(pm, session_id)

    rows = _load_rows(ws)
    if not rows:
        raise HTTPException(400, "Rows kosong / belum load editing")

    # Ambil manifest (baru dulu, fallback ke lama)
    man_new = ws / "capcut" / "capcut_map_all.json"
    if man_new.exists():
        items = json.loads(man_new.read_text(encoding="utf-8")).get("items", [])
    else:
        man_old = ws / "capcut_items.json"
        if man_old.exists():
            items = json.loads(man_old.read_text(encoding="utf-8"))
        else:
            raise HTTPException(400, "Import belum dilakukan")

    offset = int(body.get("global_offset_ms") or 0)
    tol    = max(0, int(body.get("tolerance_ms") or 200))  # default 200 ms

    # index → row
    idx2row = {}
    for r in rows:
        try:
            idx2row[int(r.get("index"))] = r
        except Exception:
            pass

    matched, unmatched = set(), set()

    # Jika manifest baru (ada row_index): cocokan per index.
    # Jika lama (tanpa row_index): pakai urutan posisional sebagai fallback.
    if items and isinstance(items[0], dict) and "row_index" in items[0]:
        for it in items:
            idx = int(it.get("row_index", -1))
            r = idx2row.get(idx)
            if not r:
                continue
            s_ms = _row_start_ms(r)           # pastikan helper ini ada
            i_ms = int(it.get("start_ms", 0))
            (matched if abs((i_ms + offset) - s_ms) <= tol else unmatched).add(idx)

        # rows yang tidak punya item → unmatched
        for idx in idx2row.keys() - matched - unmatched:
            unmatched.add(idx)
    else:
        # posisional
        k = min(len(rows), len(items))
        for i in range(k):
            r   = rows[i]
            s_ms = _row_start_ms(r)
            i_ms = int(items[i].get("start_ms", 0))
            idx  = int(r.get("index", i + 1))
            (matched if abs((i_ms + offset) - s_ms) <= tol else unmatched).add(idx)
        # sisa rows tanpa pasangan
        for j in range(k, len(rows)):
            unmatched.add(int(rows[j].get("index", j + 1)))

    # Gabungkan dengan map lama supaya nudge tidak hilang
    old = _capcut_load_map(ws)  # pastikan helper load/save map ada
    new_map = {
        "mode": "strict",
        "global_offset_ms": offset,
        "tolerance_ms": tol,
        "matched": sorted(matched),
        "unmatched": sorted(unmatched),
        "global_nudge_ms": int(old.get("global_nudge_ms", 0)),
        "row_nudges": old.get("row_nudges", {}),
    }
    _capcut_save_map(ws, new_map)

    return {"ok": True, "matched": len(matched), "unmatched": len(unmatched), "total_rows": len(rows)}

# preview segmen (trim dulu, fallback raw)
@router.get("/api/session/{session_id}/capcut/segment_audio")
def seg_preview(session_id: str, index: int, pm=Depends(get_processing_manager)):
    ws = _ws(pm, session_id)
    idx = int(index)

    # 1) Coba pakai manifest capcut_map_all.json (paling akurat)
    man = ws / "capcut" / "capcut_map_all.json"
    if man.exists():
        try:
            m = json.loads(man.read_text(encoding="utf-8"))
            for it in m.get("items", []):
                if int(it.get("row_index", -1)) == idx:
                    rel = it.get("trim") or it.get("src")
                    if rel:
                        rel_norm = str(Path(rel).as_posix())
                        return {"file": f"/api/session/{session_id}/wsfile?rel={quote(rel_norm)}"}
        except Exception:
            pass

    # 2) Fallback: tebak lokasi file di beberapa folder yang umum
    candidates = [
        ws / "capcut" / "trim" / f"{idx:05d}.wav",
        ws / "capcut" / "trim" / "all" / f"{idx:05d}.wav",
        ws / "capcut" / "raw" / f"{idx:05d}.wav",
        ws / "capcut" / "raw" / "all" / f"{idx:05d}.wav",
    ]
    for p in candidates:
        if p.exists():
            rel = str(p.resolve().relative_to(ws)).replace("\\", "/")
            return {"file": f"/api/session/{session_id}/wsfile?rel={quote(rel)}"}

    raise HTTPException(404, "Audio not found")

def _pick_tts_dir(ws: Path) -> Optional[Path]:
    for d in [ws / "capcut" / "trim",
              ws / "capcut" / "trim" / "all",
              ws / "capcut" / "raw",
              ws / "capcut" / "raw" / "all"]:
        if d.exists():
            return d
    return None
    
# ---------- Tahap-3: Build timeline & export MP4 (versi 2-pass: part → final) ----------
@router.post("/api/session/{session_id}/export/build")
def export_build(
    session_id: str,
    body: dict = Body(...),
    pm = Depends(get_processing_manager),
):
    """
    Export MP4 dengan strategi 2-pass yang aman buat durasi panjang:

      PASS A (per-chunk → PART WAV):
        - Ambil TTS dalam batch <= chunk_size (default 250)
        - Hitung timing efektif (eff_start_map / eff_slot_map / tempo)
        - Render jadi 1 file `_bus_part_XXX.wav`
          • file ini HANYA sepanjang rentang real chunk tsb (misal ~20 menit),
            bukan 2 jam penuh.
          • tidak digabung ke part sebelumnya
        - Simpan juga base_ms (waktu awal chunk dalam ms di timeline global)

      PASS B (gabung semua part → final MP4):
        - ffmpeg sekali saja:
            bed audio (center_cut / bgm / ori)
          + semua _bus_part_XXX.wav (digeser pakai base_ms masing-masing)
          + video (copy stream)
        - hasil = export.mp4

    Dampak:
      - Tidak lagi re-render audio 2 jam berulang-ulang.
      - Windows tidak buka ribuan file sekaligus → aman.
      - Disk tidak meledak sampai 50GB hanya karena versi bus_cur bertumpuk.
    """

    import json, subprocess, re as _re
    from pathlib import Path
    from urllib.parse import quote
    from typing import Optional, List, Tuple

    # -------------------------------------------------
    # Helper internal lokal
    # -------------------------------------------------
    def _escape_for_amovie(p: Path) -> str:
        """
        Biar aman dipakai di filtergraph amovie='...'
        - Windows path 'D:\\vid\\x.wav' → 'D\\:/vid/x.wav'
        - Escape single quote '
        """
        full_posix = p.resolve().as_posix()
        # escape drive letter (D: -> D\:)
        if _re.match(r"^[A-Za-z]:", full_posix):
            full_posix = full_posix[0] + r"\:" + full_posix[2:]
        # escape single quote
        full_posix = full_posix.replace("'", r"'\''")
        return full_posix

    def _ws_local(pm, sid: str) -> Path:
        try:
            return pm.workspaces_dir / sid
        except Exception:
            return Path("workspaces") / sid

    def _parse_srt_time_to_ms(s: str) -> int:
        m = _re.match(r"(\d{1,2}):(\d{2}):(\d{2})[,\.](\d{1,3})", str(s).strip())
        if not m:
            return 0
        h, m_, s_, ms = map(int, m.groups())
        return ((h * 3600 + m_ * 60 + s_) * 1000) + ms

    def _load_times_dict(ws: Path) -> dict:
        """
        Ambil timing tiap baris subtitle:
        1. capcut/capcut_map_all.json
        2. editing_cache.json
        3. fallback *.srt
        return: {index: (start_ms, end_ms)}
        """
        times = {}

        # 1) capcut_map_all.json
        man = ws / "capcut" / "capcut_map_all.json"
        if man.exists():
            try:
                data = json.loads(man.read_text(encoding="utf-8"))
                for it in data.get("items", []):
                    idx = it.get("row_index", it.get("index", None))
                    if idx is None:
                        continue
                    s = it.get("start_ms")
                    e = it.get("end_ms")
                    if s is None and it.get("start"):
                        s = _parse_srt_time_to_ms(it["start"])
                    if e is None and it.get("end"):
                        e = _parse_srt_time_to_ms(it["end"])
                    if s is not None and e is not None:
                        times[int(idx)] = (int(s), int(e))
            except Exception:
                pass
        if times:
            return times

        # 2) editing_cache.json
        for cand in [ws / "editing_cache.json", ws / "capcut" / "editing_cache.json"]:
            if not cand.exists():
                continue
            try:
                obj = json.loads(cand.read_text(encoding="utf-8"))
                arr = obj.get("rows") if isinstance(obj, dict) else obj
                for r in (arr or []):
                    try:
                        idx = int(r.get("index", 0))
                    except Exception:
                        continue
                    s = _parse_srt_time_to_ms(r.get("start", ""))
                    e = _parse_srt_time_to_ms(r.get("end", ""))
                    if idx and e >= s:
                        times[idx] = (s, e)
                if times:
                    return times
            except Exception:
                pass

        # 3) SRT fallback
        for name in ["source_video.srt", "translated_latest.srt", "translated.srt"]:
            f = ws / name
            if not f.exists():
                continue
            content = f.read_text(encoding="utf-8", errors="ignore")
            blocks = _re.split(r"\r?\n\r?\n+", content.strip())
            for b in blocks:
                lines = [ln for ln in b.splitlines() if ln.strip()]
                if len(lines) < 2:
                    continue
                try:
                    idx = int(lines[0].strip())
                except Exception:
                    continue
                tl = lines[1]
                m = _re.match(
                    r".*?(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3}).*?(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3})",
                    tl,
                )
                if not m:
                    continue
                s = _parse_srt_time_to_ms(m.group(1))
                e = _parse_srt_time_to_ms(m.group(2))
                if e >= s:
                    times[idx] = (s, e)
            if times:
                return times

        return times

    def _ffprobe_duration_ms(p: Path) -> int:
        info = subprocess.check_output(
            [
                "ffprobe",
                "-v", "error",
                "-hide_banner",
                "-show_entries", "format=duration",
                "-of", "json",
                str(p),
            ],
            text=True,
            encoding="utf-8",
        )
        j = json.loads(info)
        sec = float(j.get("format", {}).get("duration", 0.0) or 0.0)
        return int(round(sec * 1000))

    # -------------------------------------------------
    # 1. Persiapan workspace + params dari body
    # -------------------------------------------------
    ws = _ws_local(pm, session_id)
    if not ws.exists():
        raise HTTPException(404, f"Workspace not found: {session_id}")

    src_video = ws / "source_video.mp4"
    if not src_video.exists():
        raise HTTPException(404, "source_video.mp4 tidak ditemukan")

    # Ambil parameter request
    raw_center = body.get("center_cut", False)
    if isinstance(raw_center, str):
        center_cut = raw_center.strip().lower() in ("1", "true", "yes", "on")
    else:
        center_cut = bool(raw_center)

    tts_vol    = float(body.get("tts_vol", 1.0))
    bg_vol     = float(body.get("bg_vol", 0.8))
    audio_br   = str(body.get("audio_bitrate", "128k"))
    out_name   = (body.get("out_name") or "export.mp4").strip() or "export.mp4"
    bgm_path   = (body.get("bgm_path") or "").strip()
    max_atempo = float(body.get("max_atempo", 2.8))
    base_tempo = float(body.get("base_tempo", 1.4))  # kecepatan default kalau gak ketat
    guard_ms   = int(body.get("guard_ms", 15))       # jarak minimal antar baris

    # batas jumlah file TTS yang dibuka sekaligus
    chunk_size = int(body.get("chunk_size", 250) or 250)
    if chunk_size < 1:
        chunk_size = 250
    if chunk_size > 250:
        chunk_size = 250  # user cerita >200 bikin error di Windows

    # shift (per-row nudge tambahan)
    shift = {}
    shiftf = ws / "capcut" / "shift_ms.json"
    if shiftf.exists():
        try:
            shift = json.loads(shiftf.read_text(encoding="utf-8"))
        except Exception:
            shift = {}

    # capcut_map.json → global offset & nudge tiap row
    mapf_main = ws / "capcut_map.json"
    mapf_alt  = ws / "capcut" / "capcut_map.json"
    capcut_map = {"global_offset_ms": 0, "global_nudge_ms": 0, "row_nudges": {}}
    for _cand in [mapf_main, mapf_alt]:
        if _cand.exists():
            try:
                capcut_map = json.loads(_cand.read_text(encoding="utf-8"))
                break
            except Exception:
                pass

    global_offset = int(capcut_map.get("global_offset_ms", 0) or 0)
    global_nudge  = int(capcut_map.get("global_nudge_ms", 0) or 0)
    row_nudges    = {
        int(k): int(v)
        for k, v in (capcut_map.get("row_nudges", {}) or {}).items()
    }

    # timing dict index -> (start_ms, end_ms)
    times = _load_times_dict(ws)

    # direktori TTS
    tts_dir = _pick_tts_dir(ws)
    if not tts_dir:
        raise HTTPException(400, "Tidak ada TTS di workspace (Import dulu).")

    # urutkan file TTS 00001.wav, 00002.wav, ...
    tts_files = sorted(
        tts_dir.glob("*.wav"),
        key=lambda p: int(_re.sub(r"\D", "", p.stem) or "0"),
    )
    if not tts_files:
        raise HTTPException(400, "File TTS (*.wav) tidak ditemukan.")

    # -------------------------------------------------
    # 2. Hitung eff_start_map & eff_slot_map
    #    (batas aman antar kalimat biar gak tumpang tindih)
    # -------------------------------------------------
    safety_ms = 80  # buffer kecil biar gak mepet ujung slot
    meta = []       # {row_idx, eff_start, slot_ms}

    for f in tts_files:
        try:
            row_idx = int(_re.sub(r"\D", "", f.stem) or "0") or 0
        except Exception:
            continue

        s_ms, e_ms = times.get(row_idx, (0, 0))
        if e_ms > s_ms:
            per_row = int(row_nudges.get(row_idx, 0) or 0)
            eff_start = max(
                0,
                int(s_ms)
                + global_offset
                + global_nudge
                + per_row
                + int(shift.get(str(row_idx), 0) or 0),
            )
            slot_ms = max(1, int(e_ms) - int(s_ms))
            meta.append({
                "row_idx": row_idx,
                "eff_start": eff_start,
                "slot_ms": slot_ms,
            })

    # sort by time
    meta.sort(key=lambda x: x["eff_start"])

    eff_start_map = {}
    eff_slot_map  = {}

    for i, cur in enumerate(meta):
        next_start = meta[i + 1]["eff_start"] if i + 1 < len(meta) else None
        hard_end = cur["eff_start"] + cur["slot_ms"]
        if next_start is not None:
            # clamp supaya gak numpuk sama baris berikutnya
            hard_end = min(hard_end, max(0, next_start - guard_ms))
        eff_start_map[cur["row_idx"]] = cur["eff_start"]
        eff_slot_map[cur["row_idx"]]  = max(1, hard_end - cur["eff_start"])

    # -------------------------------------------------
    # 3. PASS A
    #    Render tiap chunk -> _bus_part_XXX.wav (durasi LOKAL, bukan 2 jam)
    # -------------------------------------------------
    # hasil pass A akan kita kumpulkan di parts_info:
    # parts_info = [ (Path('_bus_part_000.wav'), base_ms_000),
    #                (Path('_bus_part_001.wav'), base_ms_001),
    #                ... ]
    parts_info: List[Tuple[Path, int]] = []

    def _render_part(subset_files: List[Path], part_idx: int) -> Tuple[Path, int]:
        """
        subset_files   = sampai 200 file TTS
        part_idx       = urutan chunk (0,1,2,...)
        return (out_wav_path, base_ms)

        base_ms = waktu global (ms) paling awal di subset ini.
        Semua dialog di subset ini akan direlativkan jadi mulai dari 0 (base_ms),
        jadi file result panjangnya kira-kira cuma range chunk tsb (mis. ~20 menit),
        bukan 2 jam full.
        """
        nonlocal ws, tts_vol, base_tempo, max_atempo
        nonlocal times, eff_start_map, eff_slot_map
        nonlocal row_nudges, shift, global_offset, global_nudge, safety_ms

        # 3a. cari base_ms (awal paling kecil di chunk ini)
        base_candidates = []
        for f in subset_files:
            try:
                row_idx = int(_re.sub(r"\D", "", f.stem) or "0")
            except Exception:
                row_idx = None
            if row_idx is None:
                continue
            s_ms, e_ms = times.get(row_idx, (0, 0))
            if e_ms > s_ms:
                per_row = int(row_nudges.get(row_idx, 0) or 0)
                eff_start_ms = eff_start_map.get(
                    row_idx,
                    max(
                        0,
                        int(s_ms)
                        + global_offset
                        + global_nudge
                        + per_row
                        + int(shift.get(str(row_idx), 0) or 0),
                    ),
                )
                base_candidates.append(eff_start_ms)

        base_ms = min(base_candidates) if base_candidates else 0
        if base_ms < 0:
            base_ms = 0

        # 3b. build filtergraph untuk chunk ini
        filter_parts_chunk = []
        amixes_chunk = []

        local_label_idx = 1
        for f in subset_files:
            try:
                row_idx = int(_re.sub(r"\D", "", f.stem) or "0")
            except Exception:
                row_idx = local_label_idx

            s_ms, e_ms = times.get(row_idx, (0, 0))
            s_ms = int(s_ms or 0)
            e_ms = int(e_ms or 0)

            # hitung start efektif global
            per_row = int(row_nudges.get(row_idx, 0) or 0)
            eff_start_ms_global = eff_start_map.get(
                row_idx,
                max(
                    0,
                    s_ms
                    + global_offset
                    + global_nudge
                    + per_row
                    + int(shift.get(str(row_idx), 0) or 0),
                ),
            )

            # offset relatif thd base_ms chunk ini
            eff_start_rel = max(0, eff_start_ms_global - base_ms)

            if e_ms > s_ms:
                # tempo adaptif biar muat slot
                slot_ms_orig = max(1, e_ms - s_ms)
                slot_ms = int(eff_slot_map.get(row_idx, slot_ms_orig))
                dur_ms  = _ffprobe_duration_ms(f) or 0

                slot_eff = max(1, slot_ms - safety_ms)
                needed   = (dur_ms / float(slot_eff)) if dur_ms > 0 else float(base_tempo or 1.2)
                tempo    = max(
                    float(base_tempo or 1.4),
                    min(float(max_atempo or 2.8), needed),
                )
                tempo_ops = _atempo_chain(tempo)  # helper global

                src_esc = _escape_for_amovie(f)
                chain = (
                    f"amovie='{src_esc}',"
                    "aformat=channel_layouts=stereo:sample_rates=48000,"
                )
                if tempo_ops:
                    chain += tempo_ops + ","
                chain += (
                    f"volume={tts_vol:.3f},"
                    "apad,"
                    f"atrim=0:{slot_ms/1000:.6f},"
                    "asetpts=PTS-STARTPTS,"
                    f"adelay=delays={eff_start_rel}|{eff_start_rel}:all=1,"
                    "aresample=48000:async=1:first_pts=0"
                    f"[a{local_label_idx}]"
                )
            else:
                # baris ini gak punya end_ms valid → cuma delay-in saja
                src_esc = _escape_for_amovie(f)
                chain = (
                    f"amovie='{src_esc}',"
                    "aformat=channel_layouts=stereo:sample_rates=48000,"
                    f"volume={tts_vol:.3f},"
                    f"adelay=delays={eff_start_rel}|{eff_start_rel}:all=1,"
                    "aresample=48000:async=1:first_pts=0"
                    f"[a{local_label_idx}]"
                )

            filter_parts_chunk.append(chain)
            amixes_chunk.append(f"[a{local_label_idx}]")
            local_label_idx += 1

        if not amixes_chunk:
            # chunk kosong? balikin None
            raise HTTPException(400, f"Chunk {part_idx} kosong saat render_part().")

        # gabung semua a1,a2,... jadi mixbus
        parts_mix, last_mix = _hier_amix(
            amixes_chunk,
            normalize=0,
            group=32,
            final_label="[mixbus]",
        )
        filter_parts_chunk += parts_mix

        # finishing limiter, resample & reset PTS → [partout]
        filter_parts_chunk.append(
            f"{last_mix}"
            "alimiter=limit=0.95,"
            "aresample=48000:async=1:first_pts=0,"
            "aformat=channel_layouts=stereo:sample_rates=48000,"
            "asetpts=PTS-STARTPTS[partout]"
        )

        # simpan filtergraph chunk buat debug
        fc_text_chunk = ";\n".join(filter_parts_chunk)
        fc_path_chunk = ws / f"_fc_part_{part_idx:03d}.txt"
        fc_path_chunk.write_text(fc_text_chunk, encoding="utf-8", errors="ignore")

        # output .wav part ini
        out_part_path = ws / f"_bus_part_{part_idx:03d}.wav"

        cmd_chunk = [
            "ffmpeg", "-y",
            "-filter_complex_script", str(fc_path_chunk),
            "-map", "[partout]",
            "-c:a", "pcm_s16le",
            "-ar", "48000",
            "-ac", "2",
            str(out_part_path),
        ]

        # catat cmd buat debug
        (ws / f"_last_ffmpeg_part_{part_idx:03d}.txt").write_text(
            " ".join(cmd_chunk) + "\n\n# filter_complex:\n" + fc_text_chunk,
            encoding="utf-8",
            errors="ignore",
        )

        p_chunk = subprocess.run(cmd_chunk, capture_output=True, text=True)
        if p_chunk.returncode != 0:
            (ws / f"_ffmpeg_err_part_{part_idx:03d}.txt").write_text(
                p_chunk.stderr or "",
                encoding="utf-8",
                errors="ignore",
            )
            raise HTTPException(
                500,
                f"ffmpeg gagal di PASS A (chunk {part_idx}). Cek _ffmpeg_err_part_{part_idx:03d}.txt",
            )

        return out_part_path, base_ms

    # jalankan PASS A untuk semua chunk
    for ci_start in range(0, len(tts_files), chunk_size):
        subset = tts_files[ci_start : ci_start + chunk_size]
        chunk_idx = ci_start // chunk_size
        out_wav, base_ms = _render_part(subset, chunk_idx)
        parts_info.append((out_wav, base_ms))

    if not parts_info:
        raise HTTPException(400, "Gagal membangun PART audio TTS (kosong).")

    # -------------------------------------------------
    # 4. PASS B
    #    Gabungkan semua part + bed audio + video → final MP4
    # -------------------------------------------------
    def _final_assemble(
        parts_info: List[Tuple[Path, int]],
        out_name: str,
    ) -> Path:
        """
        parts_info: list of (Path_part_wav, base_ms_global)
        base_ms_global = timestamp (ms) part tsb di timeline global film

        Final step:
          - siapkan 'bed' (audio background):
              * bgm eksternal (bgm_path) + volume bg_vol
              * atau center_cut (ambil SIDE channel dari audio video)
              * atau audio ori tapi diturunin volumenya
          - untuk tiap part: delay seluruh part sesuai base_ms_global
          - hier_amix semua part → [tts_all]
          - amix [tts_all] + [bed] → [aout]
          - mux bareng video (`-c:v copy`)
        """
        nonlocal ws, src_video, bgm_path, center_cut, bg_vol, audio_br

        # rakit daftar input ffmpeg
        # input 0 = video (punya audio ori juga)
        inputs_final = [str(src_video)]

        bgm_input_index = None
        if bgm_path:
            # kalau user kasih BGM eksternal, tambahkan sbg input 1
            p_bgm = Path(bgm_path)
            if not p_bgm.is_absolute():
                p_bgm = (ws / p_bgm).resolve()
            if not p_bgm.exists():
                raise HTTPException(400, f"BGM tidak ditemukan: {p_bgm}")
            inputs_final.append(str(p_bgm))
            bgm_input_index = 1

        # index input pertama untuk part-part
        first_part_input_index = len(inputs_final)

        for (ppath, _base_ms) in parts_info:
            inputs_final.append(str(ppath))

        # sekarang kita bangun filter_complex
        filter_final_parts = []

        # --- BED TRACK ---
        if bgm_input_index is not None:
            # pakai BGM eksternal → volume = bg_vol
            filter_final_parts.append(
                f"[{bgm_input_index}:a]"
                "aformat=channel_layouts=stereo:sample_rates=48000,"
                "alimiter=limit=0.95,"
                f"volume={bg_vol:.2f},"
                "aresample=48000:async=1:first_pts=0,"
                "asetpts=PTS-STARTPTS[bed]"
            )
        else:
            # pakai audio original dari video (0:a)
            if center_cut:
                # pseudo-instrumental ambil side (L-R vs R-L)
                filter_final_parts.append(
                    "[0:a]"
                    "aformat=channel_layouts=stereo:sample_rates=48000,"
                    "pan=stereo|c0=FL-FR|c1=FR-FL,"
                    "alimiter=limit=0.95,"
                    f"volume={bg_vol:.2f},"
                    "aresample=48000:async=1:first_pts=0,"
                    "asetpts=PTS-STARTPTS[bed]"
                )
            else:
                # turunin volume original aja
                filter_final_parts.append(
                    "[0:a]"
                    "aformat=channel_layouts=stereo:sample_rates=48000,"
                    "alimiter=limit=0.95,"
                    "volume=0.35,"
                    "aresample=48000:async=1:first_pts=0,"
                    "asetpts=PTS-STARTPTS[bed]"
                )

        # --- TTS PARTS ---
        part_labels = []
        for idx_part, (ppath, base_ms_here) in enumerate(parts_info):
            inp_idx = first_part_input_index + idx_part
            lbl = f"[p{idx_part}]"

            # tiap part:
            # - normalisasi format
            # - delay seluruh part sesuai base_ms_here (ms) supaya
            #   part ini mendarat di posisi global film
            chain = (
                f"[{inp_idx}:a]"
                "aformat=channel_layouts=stereo:sample_rates=48000,"
                "alimiter=limit=0.95,"
                "aresample=48000:async=1:first_pts=0,"
                "asetpts=PTS-STARTPTS,"
                f"adelay=delays={base_ms_here}|{base_ms_here}:all=1,"
                "aresample=48000:async=1:first_pts=0"
                f"{lbl}"
            )
            filter_final_parts.append(chain)
            part_labels.append(lbl)

        # sekarang hier_amix semua part_labels jadi satu [tts_all]
        if not part_labels:
            raise HTTPException(400, "PASS B gagal: tidak ada part_labels")

        parts_mix, last_mix = _hier_amix(
            part_labels,
            normalize=0,
            group=32,
            final_label="[mix_tts]",
        )
        filter_final_parts += parts_mix

        # finishing agar [mix_tts] → [tts_all]
        filter_final_parts.append(
            f"{last_mix}"
            "alimiter=limit=0.95,"
            "aresample=48000:async=1:first_pts=0,"
            "aformat=channel_layouts=stereo:sample_rates=48000,"
            "asetpts=PTS-STARTPTS[tts_all]"
        )

        # --- gabung TTS + bed ---
        filter_final_parts.append(
            "[tts_all][bed]"
            "amix=inputs=2:normalize=0:dropout_transition=0:duration=longest[mix2]"
        )

        # --- limiter final + siap encode AAC ---
        filter_final_parts.append(
            "[mix2]"
            "alimiter=limit=0.96,"
            "aresample=48000:async=1:first_pts=0,"
            "aformat=channel_layouts=stereo:sample_rates=48000,"
            "asetpts=PTS-STARTPTS[aout]"
        )

        # tulis filter_complex_final ke file (debug)
        fc_text_final = ";\n".join(filter_final_parts)
        fc_final_path = ws / "_filter_complex_final.txt"
        fc_final_path.write_text(fc_text_final, encoding="utf-8", errors="ignore")

        out_path = (ws / (out_name or "export.mp4")).resolve()

        # rakit ffmpeg final cmd
        cmd_final = ["ffmpeg", "-y"]
        for ip in inputs_final:
            cmd_final += ["-i", ip]
        cmd_final += [
            "-filter_complex_script", str(fc_final_path),
            "-map", "0:v:0",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", audio_br,
            "-movflags", "+faststart",
            "-max_interleave_delta", "0",
            "-shortest",
            str(out_path),
        ]

        # simpan log final
        (ws / "_last_ffmpeg.txt").write_text(
            " ".join(cmd_final) + "\n\n# filter_complex_final:\n" + fc_text_final,
            encoding="utf-8",
            errors="ignore",
        )

        p_final = subprocess.run(cmd_final, capture_output=True, text=True)
        if p_final.returncode != 0:
            (ws / "_ffmpeg_err.txt").write_text(
                p_final.stderr or "",
                encoding="utf-8",
                errors="ignore",
            )
            raise HTTPException(
                500,
                "ffmpeg gagal di PASS B (final mux). Cek _ffmpeg_err.txt di workspace."
            )

        return out_path

    # jalankan PASS B
    out_path = _final_assemble(parts_info, out_name)

    # -------------------------------------------------
    # 5. Bersihkan part-part sementara biar hemat disk
    # -------------------------------------------------
    for (ppath, _base_ms) in parts_info:
        try:
            ppath.unlink()
        except Exception:
            pass

    # (opsional, jangan hapus filter graph txt kalau mau debugging:
    #  kamu bisa hapus di sini juga kalau mau super bersih.)

    # -------------------------------------------------
    # 6. Balikkan URL relatif supaya FE bisa download
    # -------------------------------------------------
    rel = (
        str(out_path.resolve().relative_to(ws.resolve()))
        .replace("\\", "/")
        .lstrip("/")
    )
    return {"ok": True, "output": f"/api/session/{session_id}/wsfile?rel={quote(rel)}"}





# === Helper: tempo adaptif & ffmpeg chain ===
def _compute_tempo(tts_ms: int, slot_ms: int, base_tempo: float, max_atempo: float, safety_ms: int = 80) -> float:
    """
    Tempo adaptif:
      slot_eff = max(1, slot_ms - safety_ms)
      needed   = tts_ms / slot_eff
      tempo    = clamp(max(base_tempo, needed), base_tempo, max_atempo)
    Catatan:
      - Set base_tempo=1.2 kalau mau ada guard minimal (tidak terlalu lambat).
      - Set base_tempo=1.0 kalau benar2 ingin 'jika waktu cukup, jangan diapa2in'.
    """
    slot_eff = max(1, int(slot_ms) - int(safety_ms))
    needed   = (float(tts_ms) / float(slot_eff)) if tts_ms > 0 else base_tempo
    tempo    = max(base_tempo, needed)
    if tempo < base_tempo:
        tempo = base_tempo
    if tempo > max_atempo:
        tempo = max_atempo
    # hindari noise artefak di 1.00 ± very small
    if 0.98 <= tempo <= 1.02:
        tempo = 1.0
    return float(tempo)
    
def _atempo_chain(val: float) -> str:
    """Bentuk rangkaian atempo agar tetap dalam batas 0.5..2.0 per langkah."""
    ops = []
    v = float(val)
    # pecah > 2.0
    while v > 2.0 + 1e-6:
        ops.append("atempo=2.0")
        v /= 2.0
    # pecah < 0.5
    while v < 0.5 - 1e-6:
        ops.append("atempo=0.5")
        v *= 2.0
    # dekat 1.0, skip supaya tak tambah artefak
    if abs(v - 1.0) > 0.02:
        ops.append(f"atempo={v:.3f}")
    return ",".join(ops)
    
def _hier_amix(labels, normalize=0, group=32, final_label="[aout]"):
    """
    Mix banyak label audio secara bertingkat biar graph tidak melebar ekstrem.
    labels: list seperti ["[orig]", "[a1]", "[a2]", ...]
    return: (parts, last) → parts = list of filter strings, last = label output terakhir
    """
    if not labels:
        return [], None
    if len(labels) == 1:
        # langsung rename jadi [aout]
        return [f"{labels[0].strip()}anull{final_label}"], final_label

    parts = []
    cur = labels[:]
    round_idx = 0
    gid = 0
    while len(cur) > 1:
        nxt = []
        for i in range(0, len(cur), group):
            chunk = cur[i:i+group]
            gid += 1
            out = f"[mix{round_idx}_{gid}]"
            # contoh: [a1][a2]...[a32]amix=inputs=32:normalize=0[mix0_1]
            parts.append("".join(chunk) + f"amix=inputs={len(chunk)}:normalize={normalize}{out}")
            nxt.append(out)
        cur = nxt
        round_idx += 1
    # cur tinggal satu label
    last = cur[0]
    if last != final_label:
        parts.append(f"{last}anull{final_label}")
        last = final_label
    return parts, last

def _capcut_load_map(ws) -> Dict[str, Any]:
    p = ws / "capcut_map.json"
    if p.exists():
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            m = {}
    else:
        m = {}
    # default fields
    m.setdefault("mode", "strict")
    m.setdefault("global_offset_ms", 0)
    m.setdefault("global_nudge_ms", 0)
    m.setdefault("tolerance_ms", 200)
    m.setdefault("matched", [])
    m.setdefault("unmatched", [])
    m.setdefault("row_nudges", {})  # { "12": +150, "13": -100, ... }
    return m

def _capcut_save_map(ws, m: Dict[str, Any]):
    (ws / "capcut_map.json").write_text(
        json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8"
    )

@router.get("/api/session/{session_id}/capcut/map")
def get_capcut_map(session_id: str, pm=Depends(get_processing_manager)):
    ws = _ws(pm, session_id)
    return _capcut_load_map(ws)
    
# === ADD: Nudge per baris ===
@router.post("/api/session/{session_id}/capcut/nudge_row")
def capcut_nudge_row(session_id: str, body: Dict[str, Any], pm=Depends(get_processing_manager)):
    ws = _ws(pm, session_id)
    idx = int(body.get("index", -1))
    delta = int(body.get("delta_ms", 0))
    if idx < 0 or delta == 0:
        raise HTTPException(400, "index/delta tidak valid")

    m = _capcut_load_map(ws)
    rn = m.get("row_nudges", {})
    cur = int(rn.get(str(idx), 0))
    rn[str(idx)] = cur + delta
    m["row_nudges"] = rn
    _capcut_save_map(ws, m)
    return {"ok": True, "index": idx, "row_nudge_ms": rn[str(idx)], "global_nudge_ms": m.get("global_nudge_ms", 0)}

# === ADD: Nudge global (semua baris) ===
@router.post("/api/session/{session_id}/capcut/nudge_all")
def capcut_nudge_all(session_id: str, body: Dict[str, Any], pm=Depends(get_processing_manager)):
    ws = _ws(pm, session_id)
    delta = int(body.get("delta_ms", 0))
    if delta == 0:
        raise HTTPException(400, "delta tidak valid")

    m = _capcut_load_map(ws)
    m["global_nudge_ms"] = int(m.get("global_nudge_ms", 0)) + delta
    _capcut_save_map(ws, m)
    eff = int(m.get("global_offset_ms", 0)) + int(m.get("global_nudge_ms", 0))
    return {"ok": True, "global_nudge_ms": m["global_nudge_ms"], "effective_offset_ms": eff}

#Helper Segment tab Editing speaker dan gender pisah masing-masing
# === helpers: load speaker-vs-gender secara TERPISAH ===

def _latest(ws: Path, pattern: str) -> Optional[Path]:
    cands = sorted(ws.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None

def _load_json_safely(p: Optional[Path]):
    if not p or not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return []

def _load_speaker_segments(ws: Path):
    """
    Prefer file *_speakers.json (ringkasan per-segmen dari jalur SPEAKER).
    Format keluaran: [{start: float(sec), end: float(sec), speaker: 'SPK_01'}, ...]
    """
    p = _latest(ws, "*_speakers.json")
    arr = _load_json_safely(p)
    out = []
    for it in arr or []:
        try:
            out.append({
                "start": float(it.get("start", it.get("start_s", 0.0))),
                "end":   float(it.get("end", it.get("end_s", 0.0))),
                "speaker": (it.get("speaker") or it.get("spk") or None),
            })
        except:
            pass
    return out

def _load_gender_segments(ws: Path):
    """
    Prefer file *_segments.json (per-segmen dari jalur GENDER).
    Format keluaran: [{start: float(sec), end: float(sec), gender: 'male|female|unknown'}, ...]
    """
    p = _latest(ws, "*_segments.json")
    arr = _load_json_safely(p)
    out = []
    for it in arr or []:
        try:
            g = (it.get("gender") or it.get("gen") or "unknown").lower()
            if g not in ("male", "female", "unknown"):
                g = "unknown"
            out.append({
                "start": float(it.get("start", it.get("start_s", 0.0))),
                "end":   float(it.get("end", it.get("end_s", 0.0))),
                "gender": g,
            })
        except:
            pass
    return out

def _norm_seg_list(seg_list, keep=None):
    """
    Input: list of dict dengan kemungkinan kunci waktu bervariasi: start_s/end_s atau start/end.
    Output: list of dict dengan kunci pasti 'start_s' dan 'end_s', plus field lain (mis. speaker/gender).
    """
    out = []
    for s in seg_list or []:
        try:
            st = s.get("start_s", s.get("start", s.get("start_time")))
            en = s.get("end_s",   s.get("end",   s.get("end_time")))
            if st is None or en is None:
                continue
            item = {"start_s": float(st), "end_s": float(en)}
            if keep:
                for k in keep:
                    if k in s:
                        item[k] = s[k]
            out.append(item)
        except Exception:
            # skip baris yang aneh
            pass
    return out
    
@router.post("/api/session/{session_id}/export/preview-audio")
def export_preview_audio(session_id: str, body: dict = Body(...), pm = Depends(get_processing_manager)):
    """
    Bangun audio preview (tanpa chipmunk):
      - TTS di-atempo adaptif (>= base_tempo, <= max_atempo)
      - Ditrim sesuai slot (durasi minimal 200 ms agar tak 0.001)
      - Digeser pakai eff_start_ms
      - Ditambah 'silence bed' supaya file tak pernah kosong
      - Keluarkan juga per-segmen row_<idx>.m4a untuk Play Segment
    """
    import json, subprocess, re as _re
    from pathlib import Path
    from urllib.parse import quote

    # -------- helper lokal --------
    def _ws_local(pm, sid: str) -> Path:
        try:
            return pm.workspaces_dir / sid
        except Exception:
            return Path("workspaces") / sid

    def _parse_srt_time_to_ms(s: str) -> int:
        m = _re.match(r"(\d{1,2}):(\d{2}):(\d{2})[,\.](\d{1,3})", str(s).strip())
        if not m: return 0
        h, m_, s_, ms = map(int, m.groups())
        return ((h*3600 + m_*60 + s_) * 1000) + ms

    def _load_times_dict(ws: Path) -> dict:
        # 1) capcut_map_all.json → 2) editing_cache.json → 3) SRT fallback
        times = {}
        man = ws / "capcut" / "capcut_map_all.json"
        if man.exists():
            try:
                data = json.loads(man.read_text(encoding="utf-8"))
                for it in data.get("items", []):
                    idx = it.get("row_index", it.get("index", None))
                    if idx is None: continue
                    s = it.get("start_ms"); e = it.get("end_ms")
                    if s is None and it.get("start"): s = _parse_srt_time_to_ms(it["start"])
                    if e is None and it.get("end"):   e = _parse_srt_time_to_ms(it["end"])
                    if s is not None and e is not None:
                        times[int(idx)] = (int(s), int(e))
            except Exception: pass
        if times: return times

        for cand in [ws/"editing_cache.json", ws/"capcut"/"editing_cache.json"]:
            if not cand.exists(): continue
            try:
                obj = json.loads(cand.read_text(encoding="utf-8"))
                arr = obj.get("rows") if isinstance(obj, dict) else obj
                for r in (arr or []):
                    try: idx = int(r.get("index", 0))
                    except Exception: continue
                    s = _parse_srt_time_to_ms(r.get("start",""))
                    e = _parse_srt_time_to_ms(r.get("end",""))
                    if idx and e >= s: times[idx] = (s, e)
                if times: return times
            except Exception: pass

        for name in ["source_video.srt", "translated_latest.srt", "translated.srt"]:
            f = ws / name
            if not f.exists(): continue
            content = f.read_text(encoding="utf-8", errors="ignore")
            blocks = _re.split(r"\r?\n\r?\n+", content.strip())
            for b in blocks:
                lines = [ln for ln in b.splitlines() if ln.strip()]
                if len(lines) < 2: continue
                try: idx = int(lines[0].strip())
                except Exception: continue
                tl = lines[1]
                m = _re.search(r"(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3}).*?(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3})", tl)
                if not m: continue
                s = _parse_srt_time_to_ms(m.group(1)); e = _parse_srt_time_to_ms(m.group(2))
                if e >= s: times[idx] = (s, e)
            if times: return times
        return times

    def _ffprobe_duration_ms(path: Path) -> int:
        try:
            p = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                capture_output=True, text=True
            )
            dur = float(p.stdout.strip() or "0")
            return int(dur * 1000)
        except Exception:
            return 0

    # ----- input & opsi -----
    ws = _ws_local(pm, session_id); ws.mkdir(parents=True, exist_ok=True)
    src_video = next((p for p in [ws/"video.mp4", ws/"video.mkv", ws/"video.mov"] if p.exists()), None)

    base_tempo   = float(body.get("base_tempo", 1.4))
    max_atempo   = float(body.get("max_atempo", 2.8))
    tts_vol      = float(body.get("tts_vol", 1.0))
    bg_vol       = float(body.get("bg_vol", 0.8))
    center_cut   = bool(body.get("center_cut", False))
    include_orig = bool(body.get("include_original", False))
    audio_br     = str(body.get("audio_bitrate", "128k"))
    guard_ms     = int(body.get("guard_ms", 15))
    safety_ms    = int(body.get("safety_ms", 80))
    min_seg_ms   = 200  # <<— durasi minimum agar tak 1ms

    # lokasi TTS
    tts_dir = None
    for d in [ws/"capcut"/"trim", ws/"capcut"/"trim"/"all", ws/"capcut"/"raw", ws/"capcut"/"raw"/"all"]:
        if d.exists(): tts_dir = d; break
    if not tts_dir: raise HTTPException(400, "Direktori TTS tidak ditemukan.")

    tts_files = sorted([p for p in tts_dir.glob("*.wav") if p.is_file()])
    if not tts_files: raise HTTPException(400, "Tidak ada file TTS (*.wav).")

    # waktu per index
    times = _load_times_dict(ws)

    # ----- hitung eff_start_map & eff_slot_map anti-overlap -----
    meta = []
    for f in tts_files:
        try: row_idx = int(_re.sub(r"\D","", f.stem) or "0")
        except Exception: row_idx = 0
        s, e = times.get(row_idx, (0,0))
        slot = max(1, e - s)
        meta.append({"row_idx": row_idx, "eff_start": s, "slot_ms": slot})

    meta.sort(key=lambda x: x["eff_start"])
    eff_start_map, eff_slot_map = {}, {}
    for i, cur in enumerate(meta):
        next_start = meta[i+1]["eff_start"] if i+1 < len(meta) else None
        hard_end = cur["eff_start"] + cur["slot_ms"]
        if next_start is not None:
            hard_end = min(hard_end, max(0, next_start - guard_ms))
        eff_start_map[cur["row_idx"]] = cur["eff_start"]
        eff_slot_map[cur["row_idx"]]  = max(1, hard_end - cur["eff_start"])

    # panjang timeline
    timeline_ms = 0
    for r in meta:
        timeline_ms = max(timeline_ms, eff_start_map.get(r["row_idx"], 0) + eff_slot_map.get(r["row_idx"], 0))

    # ----- build filter graph -----
    inputs, filter_parts, amixes = [], [], []

    # optional original bed
    if include_orig and src_video and src_video.exists():
        inputs.append(str(src_video))
        if center_cut:
            filter_parts.append(
                "[0:a]aformat=channel_layouts=stereo:sample_rates=48000,"
                "pan=stereo|c0=FL-FR|c1=FR-FL,alimiter=limit=0.95,"
                f"volume={bg_vol:.2f}[orig]"
            )
        else:
            filter_parts.append(
                "[0:a]aformat=channel_layouts=stereo:sample_rates=48000,"
                "alimiter=limit=0.95,"
                f"volume={bg_vol:.2f}[orig]"
            )
        amixes.append("[orig]")

    base_in = 1 if include_orig and src_video and src_video.exists() else 0

    # **silence bed** supaya selalu ada output (hindari "too small")
    sil_len = max(1000, timeline_ms + 1000)  # +1s ekor
    filter_parts.append(
        f"anullsrc=r=48000:cl=stereo,atrim=0:{sil_len/1000:.3f},asetpts=PTS-STARTPTS[sil]"
    )
    amixes.append("[sil]")

    seg_dir = ws / "preview_segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    tempo_map = {}
    plan = {"segments": [], "include_original": include_orig, "base_tempo": base_tempo,
            "max_atempo": max_atempo, "guard_ms": guard_ms, "safety_ms": safety_ms, "timeline_ms": timeline_ms}

    for k, f in enumerate(tts_files):
        inputs.append(str(f))
        in_idx = base_in + k
        in_label  = f"[{in_idx}:a]"
        out_label = f"[a{k+1}]"

        try: row_idx = int(_re.sub(r"\D","", f.stem) or "0")
        except Exception: row_idx = 0

        s_ms, e_ms = times.get(row_idx, (0,0))
        eff_start_ms = int(eff_start_map.get(row_idx, s_ms))
        slot_ms      = int(eff_slot_map.get(row_idx, max(1, e_ms - s_ms)))
        tts_ms       = _ffprobe_duration_ms(f)

        # tempo adaptif
        slot_eff = max(1, slot_ms - safety_ms)
        needed   = (tts_ms / float(slot_eff)) if tts_ms > 0 else base_tempo
        tempo    = max(base_tempo, min(max_atempo, needed))

        # pecah atempo agar 0.5..2.0 per langkah
        def _atempo_chain(v: float) -> str:
            ops, x = [], float(v)
            while x > 2.0 + 1e-6: ops.append("atempo=2.0"); x /= 2.0
            while x < 0.5 - 1e-6: ops.append("atempo=0.5"); x *= 2.0
            if abs(x-1.0) > 0.02: ops.append(f"atempo={x:.3f}")
            return ",".join(ops)
        tempo_ops = _atempo_chain(tempo)

        # chain utama (pakai min_seg_ms supaya tak 0.001)
        dur_s = max(min_seg_ms, slot_ms) / 1000.0
        ch = (
            f"{in_label}aformat=channel_layouts=stereo:sample_rates=48000,"
            f"{(tempo_ops+',') if tempo_ops else ''}"
            f"volume={tts_vol:.3f},apad,atrim=0:{dur_s:.6f},asetpts=PTS-STARTPTS,"
            f"adelay=delays={eff_start_ms}|{eff_start_ms}:all=1,aresample=48000{out_label}"
        )
        filter_parts.append(ch)
        amixes.append(out_label)

        # file segmen mandiri (untuk Play Segment)
        seg_out = seg_dir / f"row_{row_idx}.m4a"
        seg_ops = f"aformat=channel_layouts=stereo:sample_rates=48000,{(tempo_ops+',') if tempo_ops else ''}volume={tts_vol:.3f},apad,atrim=0:{dur_s:.6f},asetpts=PTS-STARTPTS"
        subprocess.run(["ffmpeg","-y","-hide_banner","-loglevel","error","-i", str(f),
                        "-af", seg_ops, "-vn","-c:a","aac","-b:a", audio_br,"-movflags","+faststart", str(seg_out)], check=False)

        tempo_map[str(row_idx)] = {
            "url": f"/api/session/{quote(session_id)}/wsfile?rel={quote(str(seg_out.relative_to(ws).as_posix()))}",
            "tempo": float(tempo), "slot_ms": int(slot_ms), "eff_start_ms": int(eff_start_ms), "mode": "sequential"
        }
        plan["segments"].append({"row_idx": row_idx, "file": f.name, "tts_ms": int(tts_ms),
                                 "slot_ms": int(slot_ms), "eff_start_ms": int(eff_start_ms),
                                 "tempo": float(tempo), "mode": "sequential"})

    # amix bertingkat
    if not amixes:
        raise HTTPException(400, "Tidak ada sumber audio untuk preview.")
    parts, _ = _hier_amix(amixes, normalize=0, group=24, final_label="[aout]")
    filter_parts += parts

    # jalankan ffmpeg
    out_path = ws / "preview_mix.m4a"
    fc_path  = ws / "_filter_complex_preview.txt"
    fc_text  = ";\n".join(filter_parts)
    fc_path.write_text(fc_text, encoding="utf-8", errors="ignore")

    cmd = ["ffmpeg","-y","-hide_banner","-loglevel","error"]
    for inp in inputs: cmd += ["-i", str(inp)]
    cmd += ["-filter_complex_script", str(fc_path), "-map","[aout]","-vn","-c:a","aac","-b:a", audio_br,"-movflags","+faststart", str(out_path)]
    (ws/"_last_ffmpeg_preview.txt").write_text(" ".join(cmd) + "\n\n# filter_complex:\n" + fc_text, encoding="utf-8", errors="ignore")

    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        (ws/"_ffmpeg_preview_err.txt").write_text(p.stderr or "", encoding="utf-8", errors="ignore")
        raise HTTPException(500, "ffmpeg preview gagal. Cek _ffmpeg_preview_err.txt.")

    # guard ukuran
    sz = out_path.stat().st_size if out_path.exists() else 0
    if sz < 4096:
        (ws/"_ffmpeg_preview_err.txt").write_text("[guard] output too small\n", encoding="utf-8", errors="ignore")
        raise HTTPException(500, "ffmpeg menghasilkan output terlalu kecil. Cek _ffmpeg_preview_err.txt.")

    # tulis tempo_map & plan
    tm_path = ws / "_preview_tempo_map.json"
    tm_path.write_text(json.dumps(tempo_map, ensure_ascii=False, indent=2), encoding="utf-8")
    plan_path = ws / "_preview_plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "ok": True,
        "file": f"/api/session/{quote(session_id)}/wsfile?rel={quote(out_path.relative_to(ws).as_posix())}",
        "tempo_map": f"/api/session/{quote(session_id)}/wsfile?rel={quote(tm_path.relative_to(ws).as_posix())}",
    }


from fastapi.responses import StreamingResponse

@router.post("/api/session/{session_id}/export/build_stream")
async def export_build_stream(
    session_id: str,
    body: dict = Body(...),
    pm = Depends(get_processing_manager),
):
    """
    Streaming export:
    - Kirim event JSON (NDJSON) berkala supaya FE bisa update progress bar realtime.
    - Tiap baris = 1 event, format {type:"progress", ...} / {type:"done", ...} / {type:"error", ...}

    NOTE:
    - PASS A: render _bus_part_xxx.wav per chunk (tanpa gabung antar-chunk)
    - PASS B: final assemble jadi MP4
    """

    import json, math, time, subprocess, re as _re, asyncio
    from pathlib import Path
    from urllib.parse import quote
    from typing import Optional, List, Tuple

    # ---------- helper lokal sama seperti di export_build 2-pass ----------
    def _escape_for_amovie(p: Path) -> str:
        full_posix = p.resolve().as_posix()
        if _re.match(r"^[A-Za-z]:", full_posix):
            full_posix = full_posix[0] + r"\:" + full_posix[2:]
        full_posix = full_posix.replace("'", r"'\''")
        return full_posix

    def _ws_local(pm, sid: str) -> Path:
        try:
            return pm.workspaces_dir / sid
        except Exception:
            return Path("workspaces") / sid

    def _parse_srt_time_to_ms(s: str) -> int:
        m = _re.match(r"(\d{1,2}):(\d{2}):(\d{2})[,\.](\d{1,3})", str(s).strip())
        if not m:
            return 0
        h, m_, s_, ms = map(int, m.groups())
        return ((h * 3600 + m_ * 60 + s_) * 1000) + ms

    def _load_times_dict(ws: Path) -> dict:
        times = {}
        man = ws / "capcut" / "capcut_map_all.json"
        if man.exists():
            try:
                data = json.loads(man.read_text(encoding="utf-8"))
                for it in data.get("items", []):
                    idx = it.get("row_index", it.get("index", None))
                    if idx is None:
                        continue
                    s = it.get("start_ms")
                    e = it.get("end_ms")
                    if s is None and it.get("start"):
                        s = _parse_srt_time_to_ms(it["start"])
                    if e is None and it.get("end"):
                        e = _parse_srt_time_to_ms(it["end"])
                    if s is not None and e is not None:
                        times[int(idx)] = (int(s), int(e))
            except Exception:
                pass
        if times:
            return times

        for cand in [ws / "editing_cache.json", ws / "capcut" / "editing_cache.json"]:
            if not cand.exists():
                continue
            try:
                obj = json.loads(cand.read_text(encoding="utf-8"))
                arr = obj.get("rows") if isinstance(obj, dict) else obj
                for r in (arr or []):
                    try:
                        idx = int(r.get("index", 0))
                    except Exception:
                        continue
                    s = _parse_srt_time_to_ms(r.get("start", ""))
                    e = _parse_srt_time_to_ms(r.get("end", ""))
                    if idx and e >= s:
                        times[idx] = (s, e)
                if times:
                    return times
            except Exception:
                pass

        for name in ["source_video.srt", "translated_latest.srt", "translated.srt"]:
            f = ws / name
            if not f.exists():
                continue
            content = f.read_text(encoding="utf-8", errors="ignore")
            blocks = _re.split(r"\r?\n\r?\n+", content.strip())
            for b in blocks:
                lines = [ln for ln in b.splitlines() if ln.strip()]
                if len(lines) < 2:
                    continue
                try:
                    idx = int(lines[0].strip())
                except Exception:
                    continue
                tl = lines[1]
                m = _re.match(
                    r".*?(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3}).*?(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3})",
                    tl,
                )
                if not m:
                    continue
                s = _parse_srt_time_to_ms(m.group(1))
                e = _parse_srt_time_to_ms(m.group(2))
                if e >= s:
                    times[idx] = (s, e)
            if times:
                return times

        return times

    def _ffprobe_duration_ms(p: Path) -> int:
        info = subprocess.check_output(
            [
                "ffprobe",
                "-v", "error",
                "-hide_banner",
                "-show_entries", "format=duration",
                "-of", "json",
                str(p),
            ],
            text=True,
            encoding="utf-8",
        )
        j = json.loads(info)
        sec = float(j.get("format", {}).get("duration", 0.0) or 0.0)
        return int(round(sec * 1000))

    # ---------- ambil params body / workspace ----------
    ws = _ws_local(pm, session_id)
    if not ws.exists():
        raise HTTPException(404, f"Workspace not found: {session_id}")

    src_video = ws / "source_video.mp4"
    if not src_video.exists():
        raise HTTPException(404, "source_video.mp4 tidak ditemukan")

    raw_center = body.get("center_cut", False)
    if isinstance(raw_center, str):
        center_cut = raw_center.strip().lower() in ("1", "true", "yes", "on")
    else:
        center_cut = bool(raw_center)

    tts_vol    = float(body.get("tts_vol", 1.0))
    bg_vol     = float(body.get("bg_vol", 0.8))
    audio_br   = str(body.get("audio_bitrate", "128k"))
    out_name   = (body.get("out_name") or "export.mp4").strip() or "export.mp4"
    bgm_path   = (body.get("bgm_path") or "").strip()
    max_atempo = float(body.get("max_atempo", 2.8))
    base_tempo = float(body.get("base_tempo", 1.4))
    guard_ms   = int(body.get("guard_ms", 15))

    chunk_size = int(body.get("chunk_size", 250) or 250)
    if chunk_size < 1:
        chunk_size = 250
    if chunk_size > 250:
        chunk_size = 250  # Windows fd limit

    shift = {}
    shiftf = ws / "capcut" / "shift_ms.json"
    if shiftf.exists():
        try:
            shift = json.loads(shiftf.read_text(encoding="utf-8"))
        except Exception:
            shift = {}

    mapf_main = ws / "capcut_map.json"
    mapf_alt  = ws / "capcut" / "capcut_map.json"
    capcut_map = {"global_offset_ms": 0, "global_nudge_ms": 0, "row_nudges": {}}
    for _cand in [mapf_main, mapf_alt]:
        if _cand.exists():
            try:
                capcut_map = json.loads(_cand.read_text(encoding="utf-8"))
                break
            except Exception:
                pass

    global_offset = int(capcut_map.get("global_offset_ms", 0) or 0)
    global_nudge  = int(capcut_map.get("global_nudge_ms", 0) or 0)
    row_nudges    = {
        int(k): int(v)
        for k, v in (capcut_map.get("row_nudges", {}) or {}).items()
    }

    times = _load_times_dict(ws)

    tts_dir = _pick_tts_dir(ws)
    if not tts_dir:
        raise HTTPException(400, "Tidak ada TTS di workspace (Import dulu).")

    tts_files = sorted(
        tts_dir.glob("*.wav"),
        key=lambda p: int(_re.sub(r"\D", "", p.stem) or "0"),
    )
    if not tts_files:
        raise HTTPException(400, "File TTS (*.wav) tidak ditemukan.")

    # ---------- precompute eff_start_map & eff_slot_map ----------
    safety_ms = 80
    meta = []
    for f in tts_files:
        try:
            row_idx = int(_re.sub(r"\D", "", f.stem) or "0") or 0
        except Exception:
            continue
        s_ms, e_ms = times.get(row_idx, (0, 0))
        if e_ms > s_ms:
            per_row = int(row_nudges.get(row_idx, 0) or 0)
            eff_start = max(
                0,
                int(s_ms)
                + global_offset
                + global_nudge
                + per_row
                + int(shift.get(str(row_idx), 0) or 0),
            )
            slot_ms = max(1, int(e_ms) - int(s_ms))
            meta.append({
                "row_idx": row_idx,
                "eff_start": eff_start,
                "slot_ms": slot_ms,
            })
    meta.sort(key=lambda x: x["eff_start"])

    eff_start_map = {}
    eff_slot_map  = {}
    for i, cur in enumerate(meta):
        nxt = meta[i + 1]["eff_start"] if i + 1 < len(meta) else None
        hard_end = cur["eff_start"] + cur["slot_ms"]
        if nxt is not None:
            hard_end = min(hard_end, max(0, nxt - guard_ms))
        eff_start_map[cur["row_idx"]] = cur["eff_start"]
        eff_slot_map[cur["row_idx"]]  = max(1, hard_end - cur["eff_start"])

    # ---------- PASS A helper: render chunk jadi _bus_part_xxx.wav ----------
    def _render_part(subset_files: List[Path], part_idx: int) -> Tuple[Path, int]:
        # cari base_ms = start global paling awal di batch
        base_candidates = []
        for f in subset_files:
            try:
                row_idx = int(_re.sub(r"\D", "", f.stem) or "0")
            except Exception:
                row_idx = None
            if row_idx is None:
                continue
            s_ms, e_ms = times.get(row_idx, (0, 0))
            if e_ms > s_ms:
                per_row = int(row_nudges.get(row_idx, 0) or 0)
                eff_start_ms = eff_start_map.get(
                    row_idx,
                    max(
                        0,
                        int(s_ms)
                        + global_offset
                        + global_nudge
                        + per_row
                        + int(shift.get(str(row_idx), 0) or 0),
                    ),
                )
                base_candidates.append(eff_start_ms)
        base_ms = min(base_candidates) if base_candidates else 0
        if base_ms < 0:
            base_ms = 0

        filter_parts_chunk = []
        amixes_chunk = []
        local_label_idx = 1

        for f in subset_files:
            try:
                row_idx = int(_re.sub(r"\D", "", f.stem) or "0")
            except Exception:
                row_idx = local_label_idx

            s_ms, e_ms = times.get(row_idx, (0, 0))
            s_ms = int(s_ms or 0)
            e_ms = int(e_ms or 0)

            per_row = int(row_nudges.get(row_idx, 0) or 0)
            eff_start_ms_global = eff_start_map.get(
                row_idx,
                max(
                    0,
                    s_ms
                    + global_offset
                    + global_nudge
                    + per_row
                    + int(shift.get(str(row_idx), 0) or 0),
                ),
            )
            eff_start_rel = max(0, eff_start_ms_global - base_ms)

            if e_ms > s_ms:
                slot_ms_orig = max(1, e_ms - s_ms)
                slot_ms = int(eff_slot_map.get(row_idx, slot_ms_orig))
                dur_ms  = _ffprobe_duration_ms(f) or 0

                slot_eff = max(1, slot_ms - safety_ms)
                needed   = (dur_ms / float(slot_eff)) if dur_ms > 0 else float(base_tempo or 1.2)
                tempo    = max(
                    float(base_tempo or 1.4),
                    min(float(max_atempo or 2.8), needed),
                )
                tempo_ops = _atempo_chain(tempo)

                src_esc = _escape_for_amovie(f)
                chain = (
                    f"amovie='{src_esc}',"
                    "aformat=channel_layouts=stereo:sample_rates=48000,"
                )
                if tempo_ops:
                    chain += tempo_ops + ","
                chain += (
                    f"volume={tts_vol:.3f},"
                    "apad,"
                    f"atrim=0:{slot_ms/1000:.6f},"
                    "asetpts=PTS-STARTPTS,"
                    f"adelay=delays={eff_start_rel}|{eff_start_rel}:all=1,"
                    "aresample=48000:async=1:first_pts=0"
                    f"[a{local_label_idx}]"
                )
            else:
                src_esc = _escape_for_amovie(f)
                chain = (
                    f"amovie='{src_esc}',"
                    "aformat=channel_layouts=stereo:sample_rates=48000,"
                    f"volume={tts_vol:.3f},"
                    f"adelay=delays={eff_start_rel}|{eff_start_rel}:all=1,"
                    "aresample=48000:async=1:first_pts=0"
                    f"[a{local_label_idx}]"
                )

            filter_parts_chunk.append(chain)
            amixes_chunk.append(f"[a{local_label_idx}]")
            local_label_idx += 1

        parts_mix, last_mix = _hier_amix(
            amixes_chunk,
            normalize=0,
            group=32,
            final_label="[mixbus]",
        )
        filter_parts_chunk += parts_mix

        filter_parts_chunk.append(
            f"{last_mix}"
            "alimiter=limit=0.95,"
            "aresample=48000:async=1:first_pts=0,"
            "aformat=channel_layouts=stereo:sample_rates=48000,"
            "asetpts=PTS-STARTPTS[partout]"
        )

        fc_text_chunk = ";\n".join(filter_parts_chunk)
        fc_path_chunk = ws / f"_fc_part_{part_idx:03d}.txt"
        fc_path_chunk.write_text(fc_text_chunk, encoding="utf-8", errors="ignore")

        out_part_path = ws / f"_bus_part_{part_idx:03d}.wav"
        cmd_chunk = [
            "ffmpeg", "-y",
            "-filter_complex_script", str(fc_path_chunk),
            "-map", "[partout]",
            "-c:a", "pcm_s16le",
            "-ar", "48000",
            "-ac", "2",
            str(out_part_path),
        ]
        (ws / f"_last_ffmpeg_part_{part_idx:03d}.txt").write_text(
            " ".join(cmd_chunk) + "\n\n# filter_complex:\n" + fc_text_chunk,
            encoding="utf-8",
            errors="ignore",
        )

        p_chunk = subprocess.run(cmd_chunk, capture_output=True, text=True)
        if p_chunk.returncode != 0:
            (ws / f"_ffmpeg_err_part_{part_idx:03d}.txt").write_text(
                p_chunk.stderr or "",
                encoding="utf-8",
                errors="ignore",
            )
            raise RuntimeError(
                f"ffmpeg gagal di PASS A chunk {part_idx}"
            )

        return out_part_path, base_ms

    # ---------- PASS B helper: final assemble semua part ----------
    def _final_assemble(parts_info: List[Tuple[Path, int]]) -> Tuple[Path, str]:
        # rakit input list: [video, (bgm optional), ...parts]
        inputs_final = [str(src_video)]
        bgm_input_index = None
        if bgm_path:
            p_bgm = Path(bgm_path)
            if not p_bgm.is_absolute():
                p_bgm = (ws / p_bgm).resolve()
            if not p_bgm.exists():
                raise RuntimeError(f"BGM tidak ditemukan: {p_bgm}")
            inputs_final.append(str(p_bgm))
            bgm_input_index = 1

        first_part_input_index = len(inputs_final)
        for (ppath, _bms) in parts_info:
            inputs_final.append(str(ppath))

        filter_final_parts = []

        # bed audio
        if bgm_input_index is not None:
            filter_final_parts.append(
                f"[{bgm_input_index}:a]"
                "aformat=channel_layouts=stereo:sample_rates=48000,"
                "alimiter=limit=0.95,"
                f"volume={bg_vol:.2f},"
                "aresample=48000:async=1:first_pts=0,"
                "asetpts=PTS-STARTPTS[bed]"
            )
        else:
            if center_cut:
                filter_final_parts.append(
                    "[0:a]"
                    "aformat=channel_layouts=stereo:sample_rates=48000,"
                    "pan=stereo|c0=FL-FR|c1=FR-FL,"
                    "alimiter=limit=0.95,"
                    f"volume={bg_vol:.2f},"
                    "aresample=48000:async=1:first_pts=0,"
                    "asetpts=PTS-STARTPTS[bed]"
                )
            else:
                filter_final_parts.append(
                    "[0:a]"
                    "aformat=channel_layouts=stereo:sample_rates=48000,"
                    "alimiter=limit=0.95,"
                    "volume=0.35,"
                    "aresample=48000:async=1:first_pts=0,"
                    "asetpts=PTS-STARTPTS[bed]"
                )

        # setiap part → delay sesuai base_ms global, lalu kumpulin jadi part_labels
        part_labels = []
        for idx_part, (ppath, base_ms_here) in enumerate(parts_info):
            inp_idx = first_part_input_index + idx_part
            lbl = f"[p{idx_part}]"
            chain = (
                f"[{inp_idx}:a]"
                "aformat=channel_layouts=stereo:sample_rates=48000,"
                "alimiter=limit=0.95,"
                "aresample=48000:async=1:first_pts=0,"
                "asetpts=PTS-STARTPTS,"
                f"adelay=delays={base_ms_here}|{base_ms_here}:all=1,"
                "aresample=48000:async=1:first_pts=0"
                f"{lbl}"
            )
            filter_final_parts.append(chain)
            part_labels.append(lbl)

        parts_mix, last_mix = _hier_amix(
            part_labels,
            normalize=0,
            group=32,
            final_label="[mix_tts]",
        )
        filter_final_parts += parts_mix

        filter_final_parts.append(
            f"{last_mix}"
            "alimiter=limit=0.95,"
            "aresample=48000:async=1:first_pts=0,"
            "aformat=channel_layouts=stereo:sample_rates=48000,"
            "asetpts=PTS-STARTPTS[tts_all]"
        )

        filter_final_parts.append(
            "[tts_all][bed]"
            "amix=inputs=2:normalize=0:dropout_transition=0:duration=longest[mix2]"
        )

        filter_final_parts.append(
            "[mix2]"
            "alimiter=limit=0.96,"
            "aresample=48000:async=1:first_pts=0,"
            "aformat=channel_layouts=stereo:sample_rates=48000,"
            "asetpts=PTS-STARTPTS[aout]"
        )

        fc_text_final = ";\n".join(filter_final_parts)
        fc_final_path = ws / "_filter_complex_final.txt"
        fc_final_path.write_text(fc_text_final, encoding="utf-8", errors="ignore")

        out_path = (ws / (out_name or "export.mp4")).resolve()

        cmd_final = ["ffmpeg", "-y"]
        for ip in inputs_final:
            cmd_final += ["-i", ip]
        cmd_final += [
            "-filter_complex_script", str(fc_final_path),
            "-map", "0:v:0",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", audio_br,
            "-movflags", "+faststart",
            "-max_interleave_delta", "0",
            "-shortest",
            str(out_path),
        ]

        (ws / "_last_ffmpeg.txt").write_text(
            " ".join(cmd_final) + "\n\n# filter_complex_final:\n" + fc_text_final,
            encoding="utf-8",
            errors="ignore",
        )

        p_final = subprocess.run(cmd_final, capture_output=True, text=True)
        if p_final.returncode != 0:
            (ws / "_ffmpeg_err.txt").write_text(
                p_final.stderr or "",
                encoding="utf-8",
                errors="ignore",
            )
            raise RuntimeError("ffmpeg gagal PASS B (final mux)")

        # rel path buat FE download
        rel = (
            str(out_path.resolve().relative_to(ws.resolve()))
            .replace("\\", "/")
            .lstrip("/")
        )
        return out_path, f"/api/session/{session_id}/wsfile?rel={quote(rel)}"

    # ---------- generator stream ----------
    async def _gen():
        start_t = time.time()
        parts_info: List[Tuple[Path,int]] = []

        total_parts = math.ceil(len(tts_files) / float(chunk_size))
        done_parts  = 0

        # PASS A: render semua chunk jadi _bus_part_xxx.wav
        for ci_start in range(0, len(tts_files), chunk_size):
            subset = tts_files[ci_start : ci_start + chunk_size]
            chunk_idx = ci_start // chunk_size

            t0 = time.time()
            try:
                out_wav, base_ms_here = _render_part(subset, chunk_idx)
                parts_info.append((out_wav, base_ms_here))
            except Exception as e:
                err_ev = {"type": "error", "message": str(e)}
                yield json.dumps(err_ev) + "\n"
                return

            done_parts += 1

            # kalkulasi ETA kasar (berdasarkan rata-rata durasi part sampai sekarang)
            elapsed_total = time.time() - start_t
            avg_per_part  = elapsed_total / max(1, done_parts)
            remain_parts  = max(0, total_parts - done_parts)
            eta_secs      = avg_per_part * remain_parts
            eta_epoch_ms  = int((time.time() + eta_secs) * 1000)

            percent_overall = (done_parts / float(total_parts)) * 80.0

            prog_ev = {
                "type": "progress",
                "phase": "parts",
                "done_parts": done_parts,
                "total_parts": total_parts,
                "percent": round(percent_overall, 2),
                "eta_epoch_ms": eta_epoch_ms,
                "msg": f"Render audio chunk {done_parts}/{total_parts}",
            }
            yield json.dumps(prog_ev) + "\n"
            await asyncio.sleep(0)

        # PASS B: final assemble
        try:
            mid_ev = {
                "type": "progress",
                "phase": "final",
                "percent": 90,
                "msg": "Mixing all parts + mux video"
            }
            yield json.dumps(mid_ev) + "\n"
            await asyncio.sleep(0)

            out_path, download_url = _final_assemble(parts_info)

        except Exception as e:
            err_ev = {"type": "error", "message": str(e)}
            yield json.dumps(err_ev) + "\n"
            return
        finally:
            # cleanup part wav
            for (ppath, _bms) in parts_info:
                try:
                    ppath.unlink()
                except Exception:
                    pass

        # DONE
        done_ev = {
            "type": "done",
            "percent": 100,
            "output_url": download_url,
            "msg": "Export selesai",
        }
        yield json.dumps(done_ev) + "\n"

    return StreamingResponse(_gen(), media_type="application/x-ndjson")

