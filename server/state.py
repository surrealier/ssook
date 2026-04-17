"""전역 비동기 작업 상태 관리."""
from concurrent.futures import ThreadPoolExecutor


class TaskState(dict):
    """비동기 작업 상태 — dict 호환 (기존 코드 무수정 호환)."""

    def __init__(self, **extra):
        super().__init__(running=False, progress=0, total=0, msg="", results=[], **extra)


# ── 스레드 풀 ──
executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ssook-bg")

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

# ── 레지스트리 (강제 중지용) ──
all_states = {
    "eval": eval_state, "bench": bench_state, "compare": compare_state,
    "error_analysis": error_analysis_state, "conf_opt": conf_opt_state,
    "embedding": embedding_state, "clip": clip_state, "embedder": embedder_state,
    "seg": seg_state, "explorer": explorer_state, "splitter": splitter_state,
    "converter": converter_state, "remapper": remapper_state, "merger": merger_state,
    "sampler": sampler_state, "anomaly": anomaly_state, "quality": quality_state,
    "duplicate": dup_state, "leaky": leaky_state, "similarity": sim_state,
    "quantize": quant_state,
}
