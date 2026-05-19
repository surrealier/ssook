"""One-shot bulk replacement: `return dict(state)` -> `state.snapshot()`."""
import re
import os
import sys

ROOTS = [
    "server/quality_routes.py",
    "server/extra_routes.py",
    "server/data_routes.py",
    "server/analysis_routes.py",
]
STATES = (
    "anomaly_state|quality_state|dup_state|leaky_state|sim_state|seg_state|"
    "merger_state|splitter_state|converter_state|remapper_state|sampler_state|"
    "explorer_state|compare_state|error_analysis_state|conf_opt_state|"
    "embedding_state|bench_state|eval_state|clip_state|embedder_state|quant_state"
)


def main():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    for r in ROOTS:
        p = os.path.join(base, r)
        with open(p, "r", encoding="utf-8") as f:
            text = f.read()
        new = re.sub(
            r"return dict\((" + STATES + r")\)",
            r"return \1.snapshot() if hasattr(\1, 'snapshot') else dict(\1)",
            text,
        )
        if new != text:
            with open(p, "w", encoding="utf-8") as f:
                f.write(new)
            print("UPDATED", r)
        else:
            print("unchanged", r)


if __name__ == "__main__":
    main()
