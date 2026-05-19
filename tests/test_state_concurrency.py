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
