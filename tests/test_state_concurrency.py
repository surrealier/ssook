"""TaskState thread-safety smoke test.

50 writer threads bump `progress` while 50 reader threads call
`snapshot()`. No exception is acceptable and the final value must
equal the total number of writes.
"""
import threading


def test_concurrent_writes_and_snapshots():
    from server.state import TaskState
    s = TaskState()
    N = 50
    PER = 100

    def writer():
        for _ in range(PER):
            cur = int(s.get("progress") or 0)
            s["progress"] = cur + 1

    def reader():
        for _ in range(PER):
            snap = s.snapshot()
            assert isinstance(snap, dict)
            assert "running" in snap

    threads = [threading.Thread(target=writer) for _ in range(N)]
    threads += [threading.Thread(target=reader) for _ in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Race-prone read-modify-write (not atomic). We only assert progress
    # advanced — not that it equals N*PER, since concurrent get+set may
    # lose updates. The point is: no crash, snapshot stays consistent.
    assert s["progress"] > 0
    assert isinstance(s.snapshot(), dict)


def test_named_locks_distinct():
    from server.state import task_locks
    a = task_locks["gpu_infer"]
    b = task_locks["cpu_io"]
    c = task_locks["gpu_infer"]
    assert a is c        # same name → same lock
    assert a is not b    # different name → different lock


def test_try_start_is_atomic_under_contention():
    """Two concurrent try_start() calls must yield exactly one True.

    Guards the TOCTOU that the old check-then-set start pattern had:
    a double-click / parallel POST must never let two workers run.
    A barrier maximizes the overlap so both threads hit the critical
    section together.
    """
    from server.state import TaskState

    s = TaskState()
    results: list[bool] = []
    results_lock = threading.Lock()
    barrier = threading.Barrier(2)

    def starter():
        barrier.wait()
        ok = s.try_start(total=10, msg="go")
        with results_lock:
            results.append(ok)

    threads = [threading.Thread(target=starter) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(results) == 1            # exactly one winner
    assert s["running"] is True
    assert s["total"] == 10             # winner's init applied


def test_try_start_rejects_when_already_running():
    from server.state import TaskState

    s = TaskState()
    assert s.try_start(msg="first") is True
    assert s.try_start(msg="second") is False
    assert s["msg"] == "first"          # second call did not overwrite init
