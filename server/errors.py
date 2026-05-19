"""Common error envelope for ssook FastAPI routes.

Two goals:
1. Give the frontend a stable error shape (`{error, code, trace_id}`)
   so it can map field-level validation errors back to inputs.
2. Make backend exceptions traceable — every error response carries a
   trace_id that also lands in `settings/logs/ssook.log`.

Existing routes return `{"error": str(e)}` ad-hoc. The exception handler
registered here catches anything that escapes the handler (including
`SsookError` and Pydantic validation errors) and produces the envelope
while leaving already-formed `{"error": ...}` route returns untouched.

`route_errors(state=...)` decorator replaces the 30+ `except Exception:
traceback.print_exc(); state.update(msg=...)` blocks scattered across
extra_routes.py / quality_routes.py background workers. Wrap a sync
function and unhandled exceptions become a logged error + state update.
"""
from __future__ import annotations

import functools
import logging
import traceback
import uuid
from typing import Any, Callable, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from server.path_safety import UnsafePathError


log = logging.getLogger("ssook.errors")


def route_errors(state: Optional[dict] = None, scope: str = "background"):
    """Decorator for sync background workers.

    Usage:
        @route_errors(state=clip_state, scope="clip")
        def _run(): ...

    On exception: logs with traceback under `ssook.<scope>`, sets
    `state.update(running=False, msg=f"Error: ...")`, and swallows the
    exception (the worker is fire-and-forget on the executor).
    """
    logger = logging.getLogger(f"ssook.{scope}")

    def deco(fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                trace_id = uuid.uuid4().hex[:12]
                logger.exception("[%s] %s in %s", trace_id, type(e).__name__, fn.__name__)
                if state is not None:
                    try:
                        state.update(running=False, msg=f"Error: {e} (trace {trace_id})")
                    except Exception:
                        pass
                return None
        return wrapper
    return deco


class SsookError(Exception):
    """Application-level error with a stable code for the frontend.

    Raised by handlers when they want to short-circuit with a specific
    user-facing error message. Status defaults to 400 (bad request) since
    most user-supplied input errors land here.
    """

    def __init__(self, code: str, msg: str, status: int = 400):
        super().__init__(msg)
        self.code = code
        self.status = status


def _envelope(error: str, code: str, trace_id: str, **extra: Any) -> dict:
    return {"error": error, "code": code, "trace_id": trace_id, **extra}


def install(app: FastAPI) -> None:
    """Attach exception handlers to the app. Call once at boot."""

    @app.exception_handler(SsookError)
    async def _ssook_error(_request: Request, exc: SsookError):
        trace_id = uuid.uuid4().hex[:12]
        log.warning("[%s] SsookError %s: %s", trace_id, exc.code, exc)
        return JSONResponse(
            status_code=exc.status,
            content=_envelope(str(exc), exc.code, trace_id),
        )

    @app.exception_handler(UnsafePathError)
    async def _unsafe_path(_request: Request, exc: UnsafePathError):
        trace_id = uuid.uuid4().hex[:12]
        log.warning("[%s] UnsafePath %s: %s", trace_id, exc.code, exc)
        return JSONResponse(
            status_code=400,
            content=_envelope(str(exc), f"PATH_{exc.code}", trace_id),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(_request: Request, exc: RequestValidationError):
        trace_id = uuid.uuid4().hex[:12]
        # Keep Pydantic detail (frontend Form.showBackendErrors maps it).
        log.info("[%s] ValidationError: %s", trace_id, exc.errors())
        return JSONResponse(
            status_code=422,
            content=_envelope(
                "Validation error", "VALIDATION", trace_id, detail=exc.errors()
            ),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        trace_id = uuid.uuid4().hex[:12]
        log.error(
            "[%s] Unhandled %s on %s %s\n%s",
            trace_id,
            type(exc).__name__,
            request.method,
            request.url.path,
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )
        return JSONResponse(
            status_code=500,
            content=_envelope(
                f"{type(exc).__name__}: {exc}", "INTERNAL", trace_id
            ),
        )
