"""Parse the v1.6.0 YOLOv8 benchmark run and emit a Markdown report.

Model-name encoding (per the operator):
  v3_<size>_960_[seq_]custom_best_opset<N>_<batch>_<imgsz>_<precision>
    size      : m | x          (model scale)
    seq        : present -> 3-frame sequential (9ch) variant, else standard
    opset      : 12 | 19
    batch      : b1 | b4 | b8 | dynamic
    imgsz      : 640 | 960 | 1280  (export input size; for `dynamic` the actual
                 inference size follows the UI input_size, NOT this nominal value)
    precision  : fp32 | fp16 | int8

Data captured from /api/benchmark/status at the moment the run stalled
(CUDAExecutionProvider, RTX-class 8 GB GPU). Codec was Raw for every row.
The x / b8 / 1280 / fp32 model is omitted: it exceeded 8 GB VRAM and hung
during warmup (OOM), so it has no measured result.
"""
import re

import pandas as pd

# name, fps, avg_ms, pre_ms, infer_ms, post_ms, p50, p95, p99, gpu%, gpu_peak%, gpu_mem_MiB
ROWS = [
    ("v3_m_960_custom_best_opset12_b1_640_fp32",       55.1,  18.16,  5.15,  12.91, 0.10, 17.79, 20.39, 21.75, 64,  68, 2125),
    ("v3_m_960_custom_best_opset12_b1_960_fp32",       25.3,  39.49, 12.41,  26.97, 0.12, 39.00, 44.35, 48.25, 63,  74, 2466),
    ("v3_m_960_custom_best_opset12_b1_1280_fp16",      15.5,  64.58, 20.50,  43.96, 0.12, 64.00, 70.14, 72.86, 64,  79, 2906),
    ("v3_m_960_custom_best_opset12_b1_1280_fp32",      13.9,  72.18, 26.05,  45.94, 0.18, 69.87, 88.94, 97.57, 61,  80, 3086),
    ("v3_m_960_custom_best_opset12_b1_1280_int8",      14.5,  68.76, 23.11,  45.48, 0.17, 67.56, 81.43, 91.34, 61,  77, 3220),
    ("v3_m_960_custom_best_opset12_b4_1280_fp32",      18.1, 220.72, 52.34, 168.07, 0.32, 218.12, 242.16, 267.33, 71, 100, 6359),
    ("v3_m_960_custom_best_opset12_b8_1280_fp32",      18.6, 429.20, 95.91, 332.78, 0.52, 427.42, 461.90, 496.85, 73, 100, 7605),
    ("v3_m_960_custom_best_opset12_dynamic_1280_fp32", 50.9,  19.65,  5.28,  14.22, 0.14, 19.25, 23.13, 25.76, 62,  66, 2302),
    ("v3_m_960_custom_best_opset19_b1_640_fp32",       38.0,  26.33,  6.35,  19.83, 0.15, 25.50, 32.91, 35.49, 54,  61, 2551),
    ("v3_m_960_custom_best_opset19_b1_960_fp32",       19.9,  50.18, 11.14,  38.91, 0.14, 49.68, 56.08, 60.82, 61,  69, 2542),
    ("v3_m_960_custom_best_opset19_b1_1280_fp16",      11.2,  89.10, 22.78,  66.16, 0.16, 88.24, 101.10, 107.21, 58, 69, 3058),
    ("v3_m_960_custom_best_opset19_b1_1280_fp32",      11.3,  88.28, 23.21,  64.89, 0.17, 87.51, 98.21, 103.64, 58, 70, 3054),
    ("v3_m_960_custom_best_opset19_b1_1280_int8",      11.7,  85.53, 20.94,  64.43, 0.15, 84.77, 94.44, 100.56, 60, 69, 3046),
    ("v3_m_960_custom_best_opset19_b4_1280_fp32",      13.7, 290.97, 51.11, 239.55, 0.32, 289.04, 309.98, 333.54, 66, 96, 6373),
    ("v3_m_960_custom_best_opset19_b8_1280_fp32",      12.7, 631.31, 81.67, 549.14, 0.50, 630.35, 687.17, 723.67, 69, 100, 7319),
    ("v3_m_960_custom_best_opset19_dynamic_1280_fp32", 44.5,  22.47,  4.28,  18.08, 0.11, 22.24, 25.23, 26.81, 62,  67, 1990),
    ("v3_m_960_seq_custom_best_opset12_b1_640_fp32",   56.9,  17.59,  4.49,  12.98, 0.12, 17.36, 19.79, 20.52, 65,  71, 1992),
    ("v3_m_960_seq_custom_best_opset12_b1_960_fp32",   27.1,  36.96,  9.74,  27.08, 0.13, 36.72, 40.00, 43.54, 69,  74, 2245),
    ("v3_m_960_seq_custom_best_opset12_b1_1280_fp16",  15.3,  65.53, 20.49,  44.89, 0.16, 64.81, 71.60, 76.06, 64,  79, 2771),
    ("v3_m_960_seq_custom_best_opset12_b1_1280_fp32",  15.5,  64.39, 19.60,  44.64, 0.15, 63.73, 70.19, 74.63, 67,  79, 2771),
    ("v3_m_960_seq_custom_best_opset12_b1_1280_int8",  15.7,  63.83, 19.37,  44.31, 0.15, 63.02, 70.61, 74.73, 65,  80, 2776),
    ("v3_m_960_seq_custom_best_opset12_b4_1280_fp32",  19.2, 207.96, 45.83, 161.84, 0.28, 206.96, 218.13, 226.63, 75, 100, 6099),
    ("v3_m_960_seq_custom_best_opset12_b8_1280_fp32",  19.5, 410.68, 82.21, 327.97, 0.50, 408.20, 436.38, 446.61, 71, 100, 7316),
    ("v3_m_960_seq_custom_best_opset12_dynamic_1280_fp32", 52.8, 18.95, 5.00, 13.81, 0.14, 18.49, 22.05, 24.75, 62, 68, 2006),
    ("v3_m_960_seq_custom_best_opset19_b1_640_fp32",   40.0,  25.00,  5.77,  19.09, 0.14, 23.96, 31.46, 36.87, 56,  62, 2260),
    ("v3_m_960_seq_custom_best_opset19_b1_960_fp32",   20.7,  48.34, 10.41,  37.79, 0.14, 47.47, 54.60, 64.49, 61,  69, 2260),
    ("v3_m_960_seq_custom_best_opset19_b1_1280_fp16",  12.2,  81.98, 18.34,  63.48, 0.15, 81.27, 88.85, 93.95, 61,  73, 2774),
    ("v3_m_960_seq_custom_best_opset19_b1_1280_fp32",  12.0,  83.48, 19.34,  63.98, 0.16, 82.75, 92.22, 99.23, 60,  71, 2780),
    ("v3_m_960_seq_custom_best_opset19_b1_1280_int8",  11.8,  84.45, 19.95,  64.34, 0.16, 83.68, 94.85, 98.97, 60,  73, 2773),
    ("v3_m_960_seq_custom_best_opset19_b4_1280_fp32",  14.0, 286.29, 46.95, 239.04, 0.29, 284.29, 305.54, 322.09, 66, 95, 6105),
    ("v3_m_960_seq_custom_best_opset19_b8_1280_fp32",  13.7, 583.03, 84.32, 498.19, 0.52, 581.87, 635.10, 670.71, 66, 100, 7324),
    ("v3_m_960_seq_custom_best_opset19_dynamic_1280_fp32", 43.9, 22.79, 4.64, 18.04, 0.11, 22.32, 26.42, 29.44, 60, 66, 2005),
    ("v3_x_960_custom_best_opset12_b1_640_fp32",       28.8,  34.75,  4.25,  30.38, 0.12, 34.50, 36.96, 37.86, 83,  87, 2775),
    ("v3_x_960_custom_best_opset12_b1_960_fp32",       13.8,  72.32, 11.07,  61.10, 0.14, 71.84, 76.83, 80.35, 83,  90, 2775),
    ("v3_x_960_custom_best_opset12_b1_1280_fp16",       7.9, 126.63, 20.52, 105.95, 0.16, 126.11, 133.10, 138.10, 81, 100, 2782),
    ("v3_x_960_custom_best_opset12_b1_1280_fp32",       7.9, 126.88, 20.86, 105.86, 0.16, 126.19, 133.74, 136.66, 79, 100, 2780),
    ("v3_x_960_custom_best_opset12_b1_1280_int8",       7.7, 129.47, 23.75, 105.57, 0.16, 128.45, 141.58, 147.33, 80, 100, 2779),
    ("v3_x_960_custom_best_opset12_b4_1280_fp32",       8.6, 466.00, 53.73, 411.95, 0.32, 464.09, 487.38, 504.37, 86, 100, 6875),
]

# Did not complete: x / b8 / 1280 / fp32 — exceeded 8 GB VRAM, hung in warmup (OOM).
OOM_ROWS = ["v3_x_960_custom_best_opset12_b8_1280_fp32"]

COLS = ["name", "fps", "avg_ms", "pre_ms", "infer_ms", "post_ms",
        "p50_ms", "p95_ms", "p99_ms", "gpu_pct", "gpu_peak_pct", "gpu_mem_mib"]


def parse_name(name: str) -> dict:
    size = re.search(r"^v3_([mx])_", name).group(1)
    variant = "seq" if "_seq_" in name else "std"
    opset = int(re.search(r"opset(\d+)", name).group(1))
    m = re.search(r"_(b\d+|dynamic)_(\d+)_(fp32|fp16|int8)$", name)
    batch_tok, imgsz, prec = m.group(1), int(m.group(2)), m.group(3)
    batch = "dyn" if batch_tok == "dynamic" else int(batch_tok[1:])
    return {"size": size, "variant": variant, "opset": opset,
            "batch": batch, "imgsz": imgsz, "prec": prec}


def to_md(df: pd.DataFrame) -> str:
    """Markdown table without requiring the optional `tabulate` dependency."""
    try:
        return df.to_markdown(index=False)
    except ImportError:
        cols = list(df.columns)
        out = ["| " + " | ".join(map(str, cols)) + " |",
               "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, r in df.iterrows():
            out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
        return "\n".join(out)


df = pd.DataFrame(ROWS, columns=COLS)
meta = df["name"].apply(parse_name).apply(pd.Series)
df = pd.concat([meta, df], axis=1)
# Per-image latency (avg_ms is per-batch). `dynamic` runs at batch 1, so treat as 1.
df["batch_n"] = df["batch"].apply(lambda b: 1 if b == "dyn" else int(b))
df["ms_per_img"] = (df["avg_ms"] / df["batch_n"]).round(2)

lines: list[str] = []
def w(s: str = ""):
    lines.append(s)

w("# YOLOv8 ONNX 벤치마크 결과 분석 / Benchmark Report")
w()
w("- 측정 환경: **CUDAExecutionProvider**, 8 GB GPU, codec=Raw, warmup 300 / iterations(측정) per run")
w("- 기준 모델: **YOLOv8** (`m`=medium, `x`=xlarge), `seq`=3-frame sequential(9ch) 변형")
w("- 모델명 규칙: `v3_<size>_960_[seq_]custom_best_opset<N>_<batch>_<imgsz>_<precision>`")
w(f"- 완료 {len(df)}건 + 미완료 {len(OOM_ROWS)}건(OOM)")
w()
w("> **FPS = batch × 1000 / 배치1회 평균(ms)** — 초당 처리 이미지 수(throughput). 배치 비교는 `avg_ms`(배치 1회 시간)가 아니라 **FPS**로 보아야 합니다.")
w("> **`dynamic` 주의**: 입력 크기가 동적이라 실제 추론은 UI input_size(이번 측정=640)로 수행됩니다. 모델명의 `1280`은 명목값일 뿐, dynamic 행의 FPS는 640 기준입니다.")
w()

w("## 1. 전체 결과 (FPS 내림차순)")
w()
full = df.sort_values("fps", ascending=False)[
    ["size", "variant", "opset", "batch", "imgsz", "prec",
     "fps", "avg_ms", "ms_per_img", "pre_ms", "infer_ms", "p50_ms", "p95_ms", "p99_ms", "gpu_pct", "gpu_mem_mib"]
]
w(to_md(full))
w()

# ── helper: controlled comparison ──
def pick(**kw):
    q = df.copy()
    for k, v in kw.items():
        q = q[q[k] == v]
    return q

w("## 2. 입력 크기(img size)별 — 640 / 960 / 1280")
w()
w("`m, std, opset12, b1, fp32` 기준 (변수: 입력 크기만):")
w()
t = pick(size="m", variant="std", opset=12, batch=1, prec="fp32")
t = t[t["imgsz"].isin([640, 960, 1280])].sort_values("imgsz")[["imgsz", "fps", "avg_ms", "infer_ms", "gpu_mem_mib"]]
w(to_md(t))
w()
w("- 입력 변(side)이 커지면 추론량은 **면적(H×W)**에 비례 → 640→960(2.25×)→1280(4×)로 갈수록 FPS가 크게 하락.")
w("- 640→1280에서 FPS 약 **4배** 감소(55.1→13.9), 면적비(4×)와 정확히 일치.")
w()

w("## 3. 정밀도(precision)별 — fp32 / fp16 / int8")
w()
w("`b1, 1280`에서 정밀도 3종이 모두 측정된 5개 그룹 비교 (FPS):")
w()
g3 = pick(batch=1, imgsz=1280).pivot_table(index=["size", "variant", "opset"], columns="prec", values="fps")
g3 = g3[["fp32", "fp16", "int8"]]
g3["fp16/int8"] = (g3["fp16"] / g3["int8"]).round(3)
g3["winner_16v8"] = g3.apply(lambda r: "fp16" if r["fp16"] > r["int8"] else ("int8" if r["int8"] > r["fp16"] else "tie"), axis=1)
w(to_md(g3.reset_index()))
w()
w("- **int8과 fp16은 사실상 동등** — fp16 3승 / int8 2승, 평균 차이 **3.9%**(최대 6.5%)로 측정 변동 수준. 방향이 일관되지 않아 \"int8이 fp16보다 느리다\"고 단정할 수 없다. (이전 버전에서 `m/std/opset12` 한 케이스만 보고 일반화했던 것을 정정한다.)")
w("- **양자화(fp16/int8)의 속도 이득 자체가 이 셋업에선 작거나 없음** — fp32와도 대부분 5% 이내. (`m/std/opset12`에서만 fp16이 fp32보다 11% 빠른 예외.)")
w()
w("**왜 GPU에서 양자화 이득이 작은가:**")
w()
w("1. **배치 1은 연산 바운드(compute-bound)가 아니다.** b1은 GPU 연산 유닛을 다 못 채워(peak ~60–80%) 레이어별 **커널 런치·메모리 이동 오버헤드가 지배적**이다. 정밀도를 낮춰도 이 오버헤드는 그대로라 fp16 Tensor Core의 연산 가속 이득이 거의 묻힌다. (배치를 키우면 연산 바운드로 진입해 fp16 이득이 커질 수 있다.)")
w("2. **INT8은 CUDAExecutionProvider에서 \"부분적으로만\" int8로 실행된다.** ONNX INT8은 보통 QDQ(Quantize-DeQuantize) 형식이라 노드마다 Q/DQ 쌍을 끼운다. ORT CUDA EP는 일부 노드만 int8 커널로 fuse하고, **나머지는 dequant→fp16/fp32 연산→quant로 되돌린다.** 즉 질문하신 대로 *내부 그래프 상당 부분이 실제로 fp로 돈다* — 거기에 Q/DQ 변환 오버헤드까지 더해져 int8의 연산 절감을 상쇄한다.")
w("3. 따라서 순수 **CUDA EP + 배치1**에서는 int8이 fp16 수준으로 수렴한다. INT8 본래의 속도 이득은 **TensorRT EP**(int8 전용 캘리브레이션·커널), **큰 배치**, 또는 **CPU/엣지 디바이스**에서 제대로 나타난다.")
w("4. 검증하려면 ORT profiling(또는 ssook **profiler 탭**)으로 노드별 EP/정밀도 배치를 보면, int8 모델에서 실제 int8로 실행되는 노드 비율을 확인할 수 있다.")
w()

w("## 4. opset 12 vs 19")
w()
w("`std, b1, fp32` 기준 (입력 크기별):")
w()
piv = pick(size="m", variant="std", batch=1, prec="fp32")
piv = piv[piv["imgsz"].isin([640, 960, 1280])].pivot_table(index="imgsz", columns="opset", values="fps")
piv.columns = [f"opset{c}_fps" for c in piv.columns]
piv = piv.reset_index()
piv["opset12/19_비율"] = (piv["opset12_fps"] / piv["opset19_fps"]).round(2)
w(to_md(piv))
w()
w("- **opset12가 opset19보다 일관되게 빠름**(약 1.25~1.45×). opset 버전이 높다고 빠른 게 아니며, 이 모델/ORT 조합에서는 opset12 그래프가 더 유리한 커널/최적화를 받음.")
w("- 배포 시 **opset12 export를 우선 검토**할 가치가 있음.")
w()

w("## 5. 배치(batch) 효과 — 1 / 4 / 8 / dynamic")
w()
w("`m, std, opset12, 1280, fp32` 기준:")
w()
t = pick(size="m", variant="std", opset=12, imgsz=1280, prec="fp32")
t = t.sort_values("fps")[["batch", "fps", "avg_ms", "ms_per_img", "infer_ms", "gpu_mem_mib", "gpu_peak_pct"]]
w(to_md(t))
w()
w("- **b1→b4: FPS 상승**(13.9→18.1, +30%) — 작은 배치에서 GPU가 덜 채워져(under-utilized) 있어 배치를 키우면 병렬성을 더 활용.")
w("- **b4→b8: 포화**(18.1→18.6, +3%) — GPU peak 100% + VRAM 7.6 GB(8 GB의 95%)로 한계 근접. 이득이 거의 사라짐.")
w("- **그 이상(x/b8)은 메모리 절벽** — VRAM 초과 시 cuDNN 느린 알고리즘 fallback·PCIe 스와핑으로 오히려 급락하거나 OOM. (`x_b8_1280_fp32`는 OOM으로 미완료.)")
w("- **`dynamic`(50.9)은 비교 대상 아님** — 실제 입력이 640이라 1280 행들과 직접 비교 불가. (640 b1=55.1과 비슷.)")
w("- 결론: 이 8 GB GPU + `m_1280`에서 **최적 배치는 4 근처**, 8은 이득 없음/위험.")
w()

w("## 6. 모델 크기 m vs x")
w()
w("`std, opset12, b1, fp32` 기준:")
w()
piv = pick(variant="std", opset=12, batch=1, prec="fp32")
piv = piv[piv["imgsz"].isin([640, 960, 1280])].pivot_table(index="imgsz", columns="size", values="fps")
piv.columns = [f"{c}_fps" for c in piv.columns]
piv = piv.reset_index()
if "m_fps" in piv and "x_fps" in piv:
    piv["m/x_배율"] = (piv["m_fps"] / piv["x_fps"]).round(2)
w(to_md(piv))
w()
w("- `x`는 `m`의 **약 1.7~1.9배 느림** (1280: 13.9 vs 7.9). xlarge 백본의 연산량이 그만큼 큼.")
w("- `x`는 b1·1280에서 이미 GPU peak 100% → 배치 이득 거의 없고(b4 8.6), b8은 OOM.")
w()

w("## 7. 표준 vs sequential(seq)")
w()
w("`opset12, b1, 1280, fp32` 기준:")
w()
t = pick(size="m", opset=12, batch=1, imgsz=1280, prec="fp32")
t = t.sort_values("variant")[["variant", "fps", "infer_ms", "gpu_mem_mib"]]
w(to_md(t))
w()
w("- 3-frame sequential(`seq`, 9채널 입력)이 표준과 **거의 동등하거나 약간 빠름**(15.5 vs 13.9). 입력 채널이 3배지만 첫 conv 외에는 연산 구조가 같아 오버헤드가 작음.")
w()

w("## 8. GPU 메모리 사용량")
w()
w("배치별 VRAM(`m, 1280, fp32`):")
w()
t = pick(size="m", variant="std", opset=12, imgsz=1280, prec="fp32").sort_values("gpu_mem_mib")[["batch", "gpu_mem_mib", "fps"]]
w(to_md(t))
w()
w("- b1≈3.1 GB → b4≈6.4 GB → b8≈7.6 GB. **b8은 8 GB의 95%** — 한계 직전.")
w("- `x_b4_1280`=6.9 GB, `x_b8_1280`은 8 GB 초과 → **OOM(미완료)**. 8 GB GPU에서 x 백본은 batch 4가 사실상 상한.")
w()

w("## 9. 핵심 인사이트 / 권장")
w()
w("1. **최고 throughput**: `m / seq / opset12 / dynamic(=640) / fp32` 52.8 FPS, 그리고 `m / opset12 / b1 / 640` 55.1 FPS. 속도가 최우선이면 **작은 입력(640) + opset12 + m**.")
w("2. **정밀도**: 이 CUDA EP + 배치1 셋업에서는 **fp16 ≈ int8 ≈ fp32**(차이 대부분 5% 이내) — 양자화 속도 이득이 작다. int8은 CUDA EP에서 부분 실행+QDQ 오버헤드, fp16은 b1이 연산 바운드가 아니라 Tensor Core 이득이 제한된다. INT8/fp16 이득을 보려면 **TensorRT EP / 큰 배치 / CPU**를 고려.")
w("3. **opset**: **12 > 19** 일관 — 같은 모델이면 opset12 export 우선.")
w("4. **배치**: 8 GB GPU + 1280에서 **m은 batch 4, x는 batch 4가 상한**. batch 8은 m에서 이득 없고 x에서 OOM.")
w("5. **입력 크기**가 성능을 가장 크게 좌우(면적 비례). 정확도가 허용하면 640/960이 1280보다 2~4배 빠름.")
w("6. **미완료**: `x_b8_1280_fp32`는 VRAM 초과 OOM. v1.6.0의 OOM-graceful-skip + gpu_mem_limit 적용 후 재측정하면 'OOM 실패' 행으로 기록되고 나머지는 완주함.")
w()

report = "\n".join(lines)
with open("BENCHMARK_REPORT.md", "w", encoding="utf-8") as f:
    f.write(report)
print(f"wrote BENCHMARK_REPORT.md ({len(report)} chars, {len(df)} rows)")
print(df.groupby(["size", "variant", "opset"])["fps"].count().to_string())
