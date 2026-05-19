"""Custom middleware for ssook.

`BodySizeLimitMiddleware` rejects requests whose declared Content-Length
exceeds the configured cap. ssook is a single-user local app so we don't
need rate limits, but a misbehaving frontend (or future LAN exposure)
could otherwise pin memory by streaming a multi-GB JSON.

Implemented as a raw ASGI middleware (not BaseHTTPMiddleware) so it
doesn't interfere with FastAPI's global exception handlers — the
BaseHTTPMiddleware wrapper re-raises unhandled errors past the handler
chain in some starlette versions.
"""
from __future__ import annotations

import logging
from typing import Awaitable, Callable

from starlette.types import ASGIApp, Receive, Scope, Send

log = logging.getLogger("ssook.middleware")


class BodySizeLimitMiddleware:
    def __init__(self, app: ASGIApp, max_bytes: int = 1 * 1024 * 1024):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        headers = {k.decode("latin-1").lower(): v.decode("latin-1")
                   for k, v in scope.get("headers", [])}
        cl = headers.get("content-length")
        if cl:
            try:
                if int(cl) > self.max_bytes:
                    log.warning(
                        "Rejecting %s %s: Content-Length %s > cap %s",
                        scope.get("method"), scope.get("path"), cl, self.max_bytes,
                    )
                    body = (
                        b'{"error":"Request body too large","code":"BODY_TOO_LARGE",'
                        b'"trace_id":""}'
                    )
                    await send({
                        "type": "http.response.start",
                        "status": 413,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode()),
                        ],
                    })
                    await send({"type": "http.response.body", "body": body})
                    return
            except ValueError:
                pass

        await self.app(scope, receive, send)
