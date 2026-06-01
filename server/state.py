"""전역 비동기 작업 상태 관리.

TaskState is a dict subclass — chosen years ago for drop-in compatibility
with existing handlers that mutate it like a plain dict. We now also need
thread-safety (the background workers and the status pollers race on the
same instance) and a lock-isolated snapshot so status endpoints don't have
to hold the lock during JSON serialization.

Strategy:
- An RLock is attached to every instance.
- Mutating methods (__setitem__, update, setdefault, pop) acquire it.
- snapshot() returns a plain shallow-copied dict — safe to serialize.
- Named GPU/CPU locks live in `task_locks` for routes that must serialise
  access to the ORT session or disk I/O.
"""
import threading
from concurrent.futures import ThreadPoolExecutor


class TaskState(dict):
    """비동기 작업 상태 — dict 호환 + thread-safe."""

    def __init__(self, **extra):
        super().__init__(running=False, progress=0, total=0, msg="", results=[], **extra)
        # Object-level RLock attached via object.__setattr__ to avoid dict key collision.
        object.__setattr__(self, "_lock", threading.RLock())

    def __setitem__(self, key, value):
        with self._lock:
            super().__setitem__(key, value)

    def update(self, *args, **kwargs):  # type: ignore[override]
        with self._lock:
            super().update(*args, **kwargs)

    def setdefault(self, key, default=None):  # type: ignore[override]
        with self._lock:
            return super().setdefault(key, default)

    def pop(self, key, *args):  # type: ignore[override]
        with self._lock:
            return super().pop(key, *args)

    def snapshot(self) -> dict:
        """Lock-isolated shallow copy. Use this for status endpoints."""
        with self._lock:
            return dict(self)

    def try_start(self, **init) -> bool:
        """Atomic check-and-set start guard.

        Routes used to do `if state['running']: return error` then a
        separate `state.update(running=True)` — two distinct ops, so two
        concurrent POSTs (double-click / parallel tabs) could both pass
        the guard and race on the same results/tmp. This collapses the
        read+write into one locked critical section: returns False if a
        run is already in flight, otherwise marks running and applies the
        caller's initial state.
        """
        with self._lock:
            if self.get("running"):
                return False
            super().__setitem__("running", True)
            super().update(init)
            return True


# ── 스레드 풀 ──
executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ssook-bg")


# ── 명명된 락 (GPU/CPU 경합 직렬화용) ──
# Routes that compete for the ONNX Runtime GPU session should acquire
# task_locks["gpu_infer"] before running. CPU-heavy I/O serialises on
# task_locks["cpu_io"]. Lazy-created on first access.
class _NamedLocks:
    def __init__(self):
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def __getitem__(self, name: str) -> threading.Lock:
        with self._guard:
            lk = self._locks.get(name)
            if lk is None:
                lk = threading.Lock()
                self._locks[name] = lk
            return lk


task_locks = _NamedLocks()

# ── 작업별 상태 ──
eval_state = TaskState(model_name="")
bench_state = TaskState()
compare_state = TaskState(images=[])
error_analysis_state = TaskState()
conf_opt_state = TaskState()
embedding_state = TaskState(image=None)
clip_state = TaskState()
embedder_state = TaskState(detail=[])
seg_state = TaskState(detail=[])
explorer_state = TaskState(data=None)
splitter_state = TaskState()
converter_state = TaskState()
remapper_state = TaskState()
merger_state = TaskState()
sampler_state = TaskState()
anomaly_state = TaskState()
quality_state = TaskState()
dup_state = TaskState()
leaky_state = TaskState()
sim_state = TaskState(index=None)
quant_state = TaskState()
vlm_state = TaskState()

# ── 레지스트리 (강제 중지용) ──
all_states = {
    "eval": eval_state, "bench": bench_state, "compare": compare_state,
    "error_analysis": error_analysis_state, "conf_opt": conf_opt_state,
    "embedding": embedding_state, "clip": clip_state, "embedder": embedder_state,
    "seg": seg_state, "explorer": explorer_state, "splitter": splitter_state,
    "converter": converter_state, "remapper": remapper_state, "merger": merger_state,
    "sampler": sampler_state, "anomaly": anomaly_state, "quality": quality_state,
    "duplicate": dup_state, "leaky": leaky_state, "similarity": sim_state,
    "quantize": quant_state, "vlm": vlm_state,
}
