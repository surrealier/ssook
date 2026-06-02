# Deployment / Export

Export a trained model — a PyTorch checkpoint (`.pt`) or an existing `.onnx` —
into a deployable artifact for a chosen runtime / execution provider, with
configurable **opset / batch / input-size / precision**, plus a self-describing
`deploy.json` manifest. Export is a long-running background job with progress
polling, mirroring the Optimization tab.

> Export backends are **optional** and lazy-imported. A target whose backend
> isn't installed on this host is shown greyed-out (e.g. OpenVINO needs the
> openvino env; CoreML is macOS-only). Install with
> `pip install -r requirements-train.txt`.

## Targets

| Target | Artifact | Provider | Notes |
|--------|----------|----------|-------|
| `onnxruntime_cpu` | `.onnx` | CPUExecutionProvider | most portable |
| `onnxruntime_cuda` | `.onnx` | CUDAExecutionProvider | onnxruntime-gpu; FP16 recommended |
| `onnxruntime_directml` | `.onnx` | DmlExecutionProvider | Windows (AMD/Intel/NVIDIA); session sets `enable_mem_pattern=False` |
| `tensorrt` | `.engine` | TensorrtExecutionProvider | NVIDIA GPU required to build |
| `openvino` | `.xml` + `.bin` | OpenVINOExecutionProvider | Intel CPU/iGPU |
| `coreml` | `.mlpackage` | CoreMLExecutionProvider | macOS build only |
| `torchscript` | `.torchscript` | — | libtorch deployment |

The three `onnxruntime_*` targets emit the **same** ONNX; they differ only in
the provider + provider-options recorded in the manifest (so one ONNX can be
deployed across CPU/CUDA/DirectML with the right session config).

## Source dispatch

- **`.pt` checkpoint** → Ultralytics' exporter, which natively honours
  `opset` / `imgsz` / `batch` / `dynamic` / `half` and emits any of the formats
  above.
- **`.onnx`** → precision conversion via the built-in quantizer
  (FP16/INT8, INT8 calibrated from an image dir); a TensorRT engine via the
  TensorRT Python API; an OpenVINO IR via `openvino.convert_model`.
  (Changing opset/batch on an existing ONNX is best-effort — re-export from the
  `.pt` for a clean change. If an opset re-spec can't be applied, the original
  opset is kept, a **warning** is returned, and the manifest records the opset
  the artifact *actually* carries — never the requested-but-unapplied value.)

CoreML and TorchScript require a `.pt` source; an incompatible source/target
pair is rejected up front with a clear message (and greyed-out in the UI). The
chosen `opset` (9–22), `batch` (≥1) and TensorRT `workspace` (>0) are validated
before the job starts.

## Configuration

- **Precision** — `fp32` / `fp16` / `int8`. INT8 uses the **Calibration images**
  directory when given (static quant), else dynamic quant. For `.pt` int8 a
  minimal calibration `data.yaml` is synthesized over a bare image dir
  automatically (an existing `data.yaml` in the dir is reused).
- **opset / batch / input-size** — input size accepts `640` or `640x480`.
- **Dynamic axes** — dynamic batch (TensorRT builds an optimization profile up
  to the chosen batch).
- Fixed-batch models are respected per the project's batch-aware rules; the
  chosen batch is recorded in the manifest.

After export the artifact is load-verified — ONNX is opened under the **target's**
execution provider when that provider is present in-process (CUDA/DirectML/…),
falling back to a CPU session with a warning otherwise — and a `deploy.json`
manifest is written next to it recording the provider, recommended
session/provider options, input/batch/opset/precision, class names and an
environment snapshot. Use **Benchmark this artifact** to validate ONNX output in
the Benchmark tab. Any soft caveats (opset kept, CPU-only validation) surface as
**warnings** in the result card without failing the export.

Only GPU-building targets (`tensorrt`, `onnxruntime_cuda`) serialize with
inference via the shared GPU lock; CPU-only exports (CPU ONNX, OpenVINO,
TorchScript, CoreML) run without it, so a long CPU export no longer blocks the
viewer/benchmark from running inference.

## API

- `GET /api/deploy/targets` — targets + availability + platform + `accepts`
  (source kinds) + `uses_gpu`
- `POST /api/deploy/run` — start (returns immediately)
- `GET /api/deploy/status` — `{running, progress, total, msg, results}`
- `GET /api/deploy/stop` — request a cooperative stop

## Execution-provider isolation

onnxruntime variants can't coexist in one Python process (see
[execution-providers.md](execution-providers.md)). For now export runs
in-process, so OpenVINO export needs the app launched from the openvino env.
TensorRT/CUDA export work from the CUDA env (e.g. `torch271`).

---

## 한국어 요약

학습된 모델(`.pt` 또는 기존 `.onnx`)을 선택한 런타임/실행 프로바이더용 산출물로
내보냅니다 — onnxruntime(CPU/CUDA/DirectML), TensorRT 엔진, OpenVINO IR, CoreML,
TorchScript. **opset/batch/입력 크기/정밀도(fp32·fp16·int8)** 를 설정할 수 있고,
실행 방법을 기록한 `deploy.json` 매니페스트를 함께 생성합니다. `.pt`는 ultralytics
익스포터로, `.onnx`는 quantizer/TensorRT/OpenVINO로 변환합니다. 백엔드는 선택
설치이며 미설치 타깃은 사유와 함께 비활성으로 표시됩니다. 소스↔타깃 비호환(예:
`.onnx`→CoreML)은 실행 전에 차단되고, opset/batch/workspace 값도 사전 검증합니다.
opset 재지정이 적용되지 않으면 원본 opset을 유지하고 경고를 표시하며 매니페스트엔
산출물의 실제 opset을 기록합니다. export 후 검증은 가능하면 **타깃 EP**로 수행하고,
GPU를 쓰는 타깃(tensorrt/onnxruntime_cuda)만 추론과 GPU 락을 공유하므로 CPU 전용
export는 추론을 막지 않습니다. ONNX 변종은 한 프로세스에 공존할 수 없어 현재 export는
인프로세스로 동작합니다(OpenVINO는 openvino env에서 실행).
