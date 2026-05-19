"""
ssook server 패키지 — FastAPI app 생성 + 라우터 등록.

사용법:
  from server import app   # 기존 server.py와 동일한 인터페이스
"""
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load .env (no-op if absent). Must happen before logging_setup so
# SSOOK_LOG_LEVEL / SSOOK_LOG_DIR are honored.
from core.env import load_env, get_int, get_str
load_env()

# Boot-time logging (file + stderr).
from core.logging_setup import configure as _configure_logging
_configure_logging()

import logging
_boot_log = logging.getLogger("ssook.boot")


# ── Lifespan ────────────────────────────────────────────
# `lifespan` replaces `@on_event("startup")` and gives us a clean place
# to teardown the executor and signal background tasks to stop.
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: kick off default model load on the thread pool.
    from server.state import executor, all_states
    from server.model_manager import load_fresh

    def _load_default_model():
        try:
            from core.app_config import AppConfig
            cfg = AppConfig()
            p = cfg.default_model_path
            if p and os.path.isfile(p):
                load_fresh(p, cfg.model_type, cfg)
                _boot_log.info("Default model loaded: %s", os.path.basename(p))
        except Exception as e:
            _boot_log.warning("Default model load failed: %s", e)

    # Cleanup stale tmp from previous runs (non-fatal).
    try:
        from core.paths import cleanup_all
        cleanup_all()
    except Exception as e:
        _boot_log.debug("cleanup_all skipped: %s", e)

    executor.submit(_load_default_model)
    _boot_log.info("ssook server starting (pid=%d)", os.getpid())

    yield

    # Shutdown: signal background workers, drain executor.
    _boot_log.info("ssook server shutting down")
    for s in all_states.values():
        try:
            s["running"] = False
        except Exception:
            pass
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass


app = FastAPI(title="ssook", version="1.5.3", lifespan=lifespan)

# Install global exception handlers (envelope + trace_id).
from server.errors import install as _install_error_handlers
_install_error_handlers(app)

# Body-size cap (single-user local app, but defends against runaway payloads).
from server.middleware import BodySizeLimitMiddleware
app.add_middleware(BodySizeLimitMiddleware, max_bytes=1 * 1024 * 1024)

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
_HEARTBEAT_TIMEOUT = get_int("SSOOK_HEARTBEAT_TIMEOUT", 120)


@app.post("/api/heartbeat")
async def heartbeat():
    global _last_heartbeat
    _last_heartbeat = time.time()
    return {"ok": True}


# Uvicorn Server registry: run_web.py registers its Server here so
# `_request_shutdown` can flip `should_exit=True` instead of os._exit.
_uvicorn_servers: list = []


def register_uvicorn_server(server) -> None:
    """Called by run_web.py to register the live uvicorn.Server instance."""
    _uvicorn_servers.append(server)


def _request_shutdown(reason: str) -> None:
    """Graceful shutdown signal. The lifespan teardown drains the executor.

    Preferred path: flip `should_exit=True` on every registered uvicorn
    Server — uvicorn calls our lifespan teardown which cancels futures
    and clears running flags.

    Fallback: signal-based shutdown via os.kill(SIGINT), which uvicorn
    handles cleanly. Last resort: os._exit after a delay.
    """
    _boot_log.info("Shutdown requested: %s", reason)
    # Tell every background worker to wind down first so the lifespan
    # cleanup doesn't have to wait on them.
    try:
        from server.state import all_states
        for s in all_states.values():
            try:
                s["running"] = False
            except Exception:
                pass
    except Exception:
        pass

    if _uvicorn_servers:
        for srv in _uvicorn_servers:
            try:
                srv.should_exit = True
            except Exception:
                pass
        return

    # No server registered — try SIGINT (uvicorn handler triggers lifespan).
    try:
        import signal
        # On Windows, only SIGINT/SIGTERM/SIGBREAK can be sent to self.
        os.kill(os.getpid(), signal.SIGINT)
        return
    except Exception:
        pass

    # Final fallback: kill children and exit after a small delay so the
    # current HTTP response can return first.
    def _kill():
        try:
            import psutil
            parent = psutil.Process(os.getpid())
            for child in parent.children(recursive=True):
                try:
                    child.kill()
                except Exception:
                    pass
        except Exception:
            pass
        os._exit(0)
    threading.Timer(0.5, _kill).start()


@app.post("/api/shutdown")
async def shutdown():
    _request_shutdown("client requested /api/shutdown")
    return {"ok": True}


def _heartbeat_watchdog():
    while True:
        time.sleep(5)
        if time.time() - _last_heartbeat > _HEARTBEAT_TIMEOUT:
            _request_shutdown(f"no client heartbeat for {_HEARTBEAT_TIMEOUT}s")
            return


threading.Thread(target=_heartbeat_watchdog, daemon=True).start()


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


# ── Background task queue (Wave 5) ──────────────────────
@app.get("/api/tasks")
async def list_tasks():
    """Snapshot of every TaskState. The sidebar / task-queue panel
    polls this to show what's currently running.
    """
    out = []
    for name, s in all_states.items():
        snap = s.snapshot() if hasattr(s, "snapshot") else dict(s)
        out.append({
            "id": name,
            "running": bool(snap.get("running")),
            "progress": int(snap.get("progress") or 0),
            "total": int(snap.get("total") or 0),
            "msg": str(snap.get("msg") or ""),
        })
    return {"tasks": out}


# ── Log tail (Wave 5) ──────────────────────────────────
@app.get("/api/logs/tail")
async def logs_tail(lines: int = 200):
    """Return the last N log lines from settings/logs/ssook.log.

    The in-app log viewer can call this on demand; capped at 2000 to
    keep payload bounded.
    """
    from core.logging_setup import _current_log_path
    lines = max(1, min(int(lines or 200), 2000))
    p = _current_log_path()
    if not p.exists():
        return {"lines": [], "path": str(p)}
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            data = f.readlines()
    except OSError as e:
        return {"error": str(e), "lines": [], "path": str(p)}
    return {"lines": data[-lines:], "path": str(p)}


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
