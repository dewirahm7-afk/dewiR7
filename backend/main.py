# backend/main.py

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pathlib import Path
import sys, os

# ----- PATH SETUP (local only, no legacy root injection) -----
BACKEND_DIR = Path(__file__).parent.resolve()       # .../dracindub_web/backend
PROJECT_ROOT = BACKEND_DIR.parent.resolve()         # .../dracindub_web

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# router API (REST endpoints)
from api.endpoints import router as api_router

# ----- FASTAPI APP -----
app = FastAPI(title="Dewa Dracin", version="2.0")

# CORS (biar frontend lokal bisa fetch API tanpa error)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# daftar semua route REST (/api/...)
app.include_router(api_router)

# ----- STATIC FRONTEND -----
FRONTEND_DIR = PROJECT_ROOT / "frontend"

print(f"Backend directory: {BACKEND_DIR}")
print(f"Frontend directory: {FRONTEND_DIR}")
print(f"Frontend exists: {FRONTEND_DIR.exists()}")

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
    print("Static files mounted successfully")
    print("✅ Buka Browser dengan Url : 127.0.0.1:8000")
else:
    print(f"WARNING: Frontend directory not found at {FRONTEND_DIR}")

# ----- BASIC PAGES -----
@app.get("/")
async def root():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(index_file.read_text(encoding="utf-8"))
    else:
        return HTMLResponse(
            """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Dewa Dracin</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .error { color: red; }
                    .info { color: blue; }
                </style>
            </head>
            <body>
                <h1>Dewa Dracin - Auto Dubbing Studio</h1>
                <div class="error">Frontend files not found. Please check the frontend directory.</div>
                <div class="info">Backend is running correctly.</div>
                <div class="info">API Test: <a href="/api/test">/api/test</a></div>
                <div class="info">Health: <a href="/health">/health</a></div>
            </body>
            </html>
            """
        )

@app.get("/health")
async def health_check():
    return JSONResponse(
        {
            "status": "healthy",
            "version": "2.0.0",
            "frontend_path": str(FRONTEND_DIR),
            "backend": "running",
        }
    )

# ----- WEBSOCKET -----
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    # lazy import: websocket_manager baru di-load kalau websocket betulan dipakai
    from api.websockets import websocket_manager

    await websocket_manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                import json
                message = json.loads(data)
                if message.get("type") == "join_session":
                    session_id = message.get("data", {}).get("session_id")
                    if session_id:
                        websocket_manager.add_to_session(client_id, session_id)
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "type": "session_joined",
                                    "data": {"session_id": session_id},
                                }
                            )
                        )
            except Exception as e:
                print(f"WebSocket message error: {e}")
    except WebSocketDisconnect:
        websocket_manager.disconnect(client_id)

# ----- RUN SERVER DIRECTLY (tanpa reload watcher, start cepat) -----
if __name__ == "__main__":
    print("Starting Dewa Dracin Server...")
    print(f"Working directory: {os.getcwd()}")
    print(f"Python path: {os.environ.get('PYTHONPATH', 'Not set')}")

    workspaces_dir = PROJECT_ROOT / "workspaces"
    workspaces_dir.mkdir(exist_ok=True)
    print(f"Workspaces directory: {workspaces_dir.absolute()}")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,  # penting: gak spawn "Started reloader process ..."
    )
