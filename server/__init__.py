"""
ssook server 패키지 — FastAPI app 생성 + 라우터 등록.

사용법:
  from server import app   # 기존 server.py와 동일한 인터페이스
"""
import os
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

app = FastAPI(title="ssook", version="1.5.3")

# ── Static files ────────────────────────────────────────
WEB_DIR = ROOT / "web"
if WEB_DIR.exists():
    app.mount("/css", StaticFiles(directory=WEB_DIR / "css"), name="css")
    app.mount("/js", StaticFiles(directory=WEB_DIR / "js"), name="js")
    app.mount("/assets", StaticFiles(directory=ROOT / "assets"), name="assets")


@app.get("/")
async def index():
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    html = html.replace("{{VERSION}}", app.version)
    return HTMLResponse(html)


# ── Heartbeat & auto-shutdown ───────────────────────────
_last_heartbeat = time.time()
_HEARTBEAT_TIMEOUT = 120


@app.post("/api/heartbeat")
async def heartbeat():
    global _last_heartbeat
    _last_heartbeat = time.time()
    return {"ok": True}


@app.post("/api/shutdown")
async def shutdown():
    """Gracefully shut down the server and all child processes."""
    import threading, signal
    def _kill():
        try:
            import psutil
            parent = psutil.Process(os.getpid())
            for child in parent.children(recursive=True):
                child.kill()
        except Exception:
            pass
        os._exit(0)
    threading.Timer(0.5, _kill).start()
    return {"ok": True}


def _heartbeat_watchdog():
    global _last_heartbeat
    while True:
        time.sleep(5)
        if time.time() - _last_heartbeat > _HEARTBEAT_TIMEOUT:
            import logging
            logging.info("No client heartbeat — shutting down server")
            os._exit(0)


threading.Thread(target=_heartbeat_watchdog, daemon=True).start()

# ── Auto-load default model on startup ──────────────────
from server.state import executor
from server.model_manager import load_fresh


@app.on_event("startup")
async def _auto_load_default_model():
    def _load_bg():
        try:
            from core.app_config import AppConfig
            cfg = AppConfig()
            p = cfg.default_model_path
            if p and os.path.isfile(p):
                load_fresh(p, cfg.model_type, cfg)
                print(f"[Startup] Default model loaded: {os.path.basename(p)}")
        except Exception as e:
            print(f"[Startup] Default model load failed: {e}")
    executor.submit(_load_bg)


# ── Force-stop ──────────────────────────────────────────
from server.state import all_states


@app.post("/api/force-stop/{task_id}")
async def force_stop(task_id: str):
    if task_id == "all":
        for s in all_states.values():
            s["running"] = False
        return {"ok": True, "msg": "All tasks stopped"}
    state = all_states.get(task_id)
    if not state:
        return {"error": f"Unknown task: {task_id}"}
    state["running"] = False
    state["msg"] = "Stopped by user"
    return {"ok": True}


# ── Register routers ────────────────────────────────────
from server.config_routes import router as config_router
from server.model_routes import router as model_router
from server.viewer_routes import router as viewer_router
from server.eval_routes import router as eval_router
from server.analysis_routes import router as analysis_router
from server.benchmark_routes import router as benchmark_router
from server.data_routes import router as data_router
from server.quality_routes import router as quality_router
from server.system_routes import router as system_router
from server.extra_routes import router as extra_router
from server.optimization_routes import router as optimization_router

for r in [config_router, model_router, viewer_router, eval_router,
          analysis_router, benchmark_router, data_router, quality_router,
          system_router, extra_router, optimization_router]:
    app.include_router(r)
