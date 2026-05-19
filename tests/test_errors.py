"""Tests for server.errors — the global exception handlers and
the @route_errors decorator for background workers.
"""


def test_ssook_error_envelope():
    from fastapi.testclient import TestClient
    from server import app
    from server.errors import SsookError

    @app.get("/__test_ssook_error")
    async def _h():
        raise SsookError("MY_CODE", "boom", status=418)

    client = TestClient(app)
    r = client.get("/__test_ssook_error")
    assert r.status_code == 418
    body = r.json()
    assert body["error"] == "boom"
    assert body["code"] == "MY_CODE"
    assert "trace_id" in body and body["trace_id"]


def test_unhandled_envelope():
    from fastapi.testclient import TestClient
    from server import app

    @app.get("/__test_unhandled")
    async def _h():
        raise RuntimeError("kaboom")

    # raise_server_exceptions=False matches uvicorn's prod behaviour —
    # otherwise starlette's TestClient re-raises unhandled exceptions
    # past our registered handler, which is a test-harness artefact, not
    # a real bug.
    client = TestClient(app, raise_server_exceptions=False)
    r = client.get("/__test_unhandled")
    assert r.status_code == 500
    body = r.json()
    assert body["code"] == "INTERNAL"
    assert "kaboom" in body["error"]


def test_route_errors_decorator_swallows_and_logs():
    from server.errors import route_errors
    from server.state import TaskState

    state = TaskState()
    state["running"] = True

    @route_errors(state=state, scope="test")
    def _worker():
        raise ValueError("bad")

    # Should not raise — the decorator swallows.
    _worker()
    assert state["running"] is False
    assert "bad" in state["msg"]
    assert "trace" in state["msg"]


def test_path_unsafe_maps_to_400():
    from fastapi.testclient import TestClient
    from server import app

    client = TestClient(app)
    r = client.post("/api/gt/classes", json={"label_dir": "/nowhere/at/all/xyz"})
    assert r.status_code == 400
    body = r.json()
    assert body["code"].startswith("PATH_")
    assert "trace_id" in body
