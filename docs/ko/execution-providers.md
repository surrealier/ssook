# Execution Providers

[← 목차로 돌아가기](../index.md) | 🌐 [English](../execution-providers.md) | **한국어** | [日本語](../ja/execution-providers.md) | [中文](../zh/execution-providers.md)

ssook은 시스템에 가장 적합한 하드웨어 가속기(Execution Provider)를 자동으로 선택합니다. 이 문서에서는 EP 시스템의 동작 원리와 설정 방법을 설명합니다.

---

## 목차

- [개요](#개요)
- [지원하는 Execution Providers](#지원하는-execution-providers)
- [자동 선택 동작 원리](#자동-선택-동작-원리)
- [venv 격리 아키텍처](#venv-격리-아키텍처)
- [설치](#설치)
- [문제 해결](#문제-해결)

---

## 개요

ONNX Runtime은 **Execution Providers (EP)**라고 불리는 여러 하드웨어 백엔드를 지원합니다. 각 EP는 서로 다른 하드웨어를 대상으로 합니다:

- **CUDA** → NVIDIA GPU
- **DirectML** → 모든 Windows GPU (NVIDIA, AMD, Intel)
- **OpenVINO** → Intel CPU 및 iGPU
- **CoreML** → Apple Silicon (M1/M2/M3)
- **CPU** → 모든 프로세서 (폴백)

문제는 각 EP가 `onnxruntime` 패키지의 **서로 다른 빌드**를 필요로 하며, 서로 충돌한다는 것입니다. 동일한 Python 환경에 `onnxruntime-gpu`와 `onnxruntime-directml`을 함께 설치할 수 없습니다.

ssook은 **venv 격리**로 이 문제를 해결합니다 — 각 EP가 자체 격리된 환경을 갖습니다.

---

## 지원하는 Execution Providers

| EP | 플랫폼 | 하드웨어 | 패키지 |
|----|--------|----------|--------|
| CUDA | Windows, Linux | NVIDIA GPU | `onnxruntime-gpu` |
| DirectML | Windows | 모든 GPU | `onnxruntime-directml` |
| OpenVINO | Windows, Linux | Intel CPU/iGPU | `onnxruntime-openvino` |
| CoreML | macOS | Apple Silicon | `onnxruntime-coreml` |
| CPU | 전체 | 모든 CPU | `onnxruntime` |

---

## 자동 선택 동작 원리

시작 시 `onnxruntime`을 임포트하기 **전에** ssook이 EP 선택기를 실행합니다:

```
┌─────────────────────────────┐
│   플랫폼(OS) 감지            │
│   Windows / Linux / macOS   │
└──────────┬──────────────────┘
           ▼
┌─────────────────────────────┐
│  우선순위 목록 가져오기       │
│  Win: cuda→directml→        │
│       openvino→cpu          │
│  Linux: cuda→openvino→cpu   │
│  macOS: coreml→cpu          │
└──────────┬──────────────────┘
           ▼
┌─────────────────────────────┐
│  각 EP에 대해 순서대로:      │
│  1. 런타임이 번들되어 있는가? │
│  2. 하드웨어가 존재하는가?    │
│  → 둘 다 예: 선택            │
│  → 아니오: 건너뛰고 이유 기록 │
└──────────┬──────────────────┘
           ▼
┌─────────────────────────────┐
│  선택된 EP의 site-packages를 │
│  sys.path[0]에 삽입          │
│  + DLL 디렉토리 추가         │
│  (Windows)                  │
└──────────┬──────────────────┘
           ▼
┌─────────────────────────────┐
│  import onnxruntime         │
│  (올바른 빌드가 로드됨)      │
└─────────────────────────────┘
```

**하드웨어 감지 방법:**

| EP | 감지 방법 |
|----|-----------|
| CUDA | `nvidia-smi --query-gpu=name` 실행 — GPU 이름이 반환되면 NVIDIA GPU 존재 |
| OpenVINO | Windows: `wmic path win32_videocontroller get name`에서 "intel" 확인. Linux: `/dev/dri` 존재 여부 확인 |
| DirectML | 모든 Windows GPU (추가 확인 불필요) |
| CoreML | macOS 플랫폼 확인 (Apple Silicon) |
| CPU | 항상 사용 가능 |

**폴백 동작:**
최우선 EP를 사용할 수 없는 경우 (하드웨어 없음, 런타임 미설치), ssook은 우선순위 목록의 다음 EP로 폴백합니다. 폴백 이유가 기록되어 Settings 탭에 표시되며, 수정 방법도 함께 안내됩니다.

---

## venv 격리 아키텍처

각 EP 런타임은 자체 격리된 디렉토리에 위치합니다:

```
ep_venvs/                          (소스 모드)
├── cuda/
│   └── Lib/site-packages/
│       └── onnxruntime/           (onnxruntime-gpu)
├── directml/
│   └── Lib/site-packages/
│       └── onnxruntime/           (onnxruntime-directml)
├── openvino/
│   └── Lib/site-packages/
│       └── onnxruntime/           (onnxruntime-openvino)
└── cpu/
    └── Lib/site-packages/
        └── onnxruntime/           (onnxruntime)

ep_runtimes/                       (프리징된 exe 모드)
├── cuda/
│   └── onnxruntime/
├── directml/
│   └── onnxruntime/
└── cpu/
    └── onnxruntime/
```

EP가 선택되면 해당 `site-packages` 디렉토리가 `sys.path`의 **맨 앞**에 삽입됩니다. 이를 통해 `import onnxruntime`이 선택된 EP에 맞는 올바른 빌드를 로드합니다.

Windows에서는 CUDA/TensorRT/DirectML 공유 라이브러리를 위한 DLL 검색 디렉토리도 등록됩니다.

---

## 설치

### 자동 설치 (권장)

```bash
# 플랫폼에 맞는 모든 EP 설치
python scripts/setup_ep.py

# 특정 EP만 설치
python scripts/setup_ep.py cuda cpu

# 설치 상태 확인
python scripts/setup_ep.py --status
```

설치 스크립트가 수행하는 작업:
1. 각 EP에 대한 가상 환경 생성
2. 올바른 `onnxruntime` 변종 설치
3. 설치 검증

### 수동 설치

CPU 추론만 필요한 경우 기본 패키지만 설치하면 됩니다:

```bash
pip install onnxruntime
```

GPU 가속이 필요한 경우:

```bash
# NVIDIA GPU
pip install onnxruntime-gpu

# Windows GPU (모든 제조사)
pip install onnxruntime-directml
```

> 참고: 수동 설치는 venv 격리를 사용하지 않습니다. 한 번에 하나의 EP 변종만 설치할 수 있습니다.

---

## 문제 해결

### "Falling back to CPU"

**원인**: 선호하는 EP의 하드웨어가 감지되지 않았습니다.

| EP | 확인 사항 | 해결 방법 |
|----|-----------|-----------|
| CUDA | `nvidia-smi` 실패 | NVIDIA 드라이버 설치, GPU 연결 확인 |
| OpenVINO | Intel GPU 미감지 | 이 EP는 Intel iGPU/dGPU가 필요합니다 |
| DirectML | — | 모든 Windows GPU에서 작동해야 합니다 |

### "EP runtime not bundled"

**원인**: EP의 onnxruntime 패키지가 `ep_venvs/`에 설치되어 있지 않습니다.

**해결 방법**: `python scripts/setup_ep.py <ep_name>`을 실행하여 설치하세요.

### 현재 EP 상태 확인

ssook에서 **Settings** 탭으로 이동하세요. EP 상태 섹션에서 다음을 확인할 수 있습니다:
- 현재 선택된 EP
- 시스템에서 사용 가능한 EP
- ep_venvs에 번들된 EP
- 건너뛴 EP와 이유 및 수정 방법 안내
- 폴백 발생 여부
