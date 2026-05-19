# ssook Backend Runbook

이 문서는 백엔드를 운영하면서 자주 마주치는 작업의 위치와 절차를 정리합니다. 대상 독자: 코드를 빌드/배포하는 개발자, 디버깅을 위한 사용자.

## 로그

- 위치: `settings/logs/ssook.log`
- 회전: 10 MiB × 5 (총 ~50 MiB)
- 레벨 조정: `.env`에 `SSOOK_LOG_LEVEL=DEBUG`
- 위치 변경: `SSOOK_LOG_DIR=/somewhere/else`
- 앱 실행 중에 마지막 N줄 조회: `GET /api/logs/tail?lines=200`

각 에러 응답에는 12자 `trace_id`가 포함됩니다. `ssook.log`에서 같은 ID를 grep 하면 전체 스택을 볼 수 있습니다.

```bash
# 예시
grep "abc123def456" settings/logs/ssook.log
```

## 환경 변수

`.env.example`를 `.env`로 복사 후 편집하세요. 모든 키는 선택사항이며, 기본값은 기존 동작과 동일합니다.

| 키 | 기본값 | 용도 |
|---|---|---|
| `SSOOK_PORT` | `8765` | uvicorn 포트 |
| `SSOOK_LOG_LEVEL` | `INFO` | 로깅 레벨 |
| `SSOOK_LOG_DIR` | `settings/logs` | 로그 디렉터리 |
| `SSOOK_HEARTBEAT_TIMEOUT` | `120` | 클라이언트 heartbeat 미수신 시 자동 종료까지 대기 (초) |
| `SSOOK_EXTRA_ROOTS` | (없음) | path_safety의 추가 허용 루트 (콤마 분리) |
| `SSOOK_CACHE_DIR` | `settings/cache` | 장기 캐시 (모델 메타 등) |
| `SSOOK_TMP_DIR` | `settings/tmp` | 단기 작업 디렉터리 |
| `SSOOK_BLIP_TOKENIZER` | (없음) | VLM v1.1 BLIP 토크나이저 경로 (현재 미사용) |

## 백그라운드 작업 / 강제 중지

- 현재 진행 중인 모든 작업 조회: `GET /api/tasks`
- 작업별 force-stop: `POST /api/force-stop/<task_id>`
- 전체 중지: `POST /api/force-stop/all`

`task_id`는 `server/state.py`의 `all_states` 키와 동일: `eval`, `bench`, `clip`, `embedder`, `seg`, `quality`, `duplicate`, `leaky`, `similarity`, `vlm`, `quantize` 등.

VLM 배치는 비동기로 돌고 force-stop이 한 루프 단위로 반응합니다(`vlm_state["running"]` 폴링).

## 경로 검증

`server/path_safety.py`가 user-supplied 경로를 정규화합니다.

- `safe_path(p)` — `..` 거부, NUL 거부, abspath
- `safe_model_file(p)` — `.onnx` 확장자 + 파일 존재
- `safe_image_file(p)` — 이미지 확장자 + 파일 존재
- `safe_image_dir(p)` / `safe_label_dir(p)` — 디렉터리 존재

실패 시 `UnsafePathError`가 발생하고 전역 핸들러가 400 + `code: "PATH_*"`로 반환합니다.

엄격한 화이트리스트가 필요한 라우트는 `safe_path(p, roots=[...])` 패턴으로 명시적으로 root를 전달하세요.

## 스모크 테스트 재실행

```bash
pip install -e .[dev]
pytest -q --maxfail=1
```

핵심 영역별 테스트 파일:

| 영역 | 파일 |
|---|---|
| 경로 보안 | `tests/test_path_safety.py` |
| VLM 라우트 | `tests/test_vlm_routes.py` |
| VLM 추론 | `tests/test_vlm_inference.py` |
| Env 로더 | `tests/test_env.py` |
| 경로 헬퍼 | `tests/test_paths.py` |
| dHash | `tests/test_hashing.py` |
| TaskState 동시성 | `tests/test_state_concurrency.py` |
| Export 헬퍼 | `tests/test_exports.py` |
| 에러 envelope | `tests/test_errors.py` |
| 라우트 스모크 | `tests/test_server_routes.py` (기존) |

## 셧다운

- `POST /api/shutdown` → 0.5초 후 자식 프로세스 종료 + lifespan 정리
- heartbeat watchdog: 120초 (env로 조정) 동안 클라이언트가 사라지면 자동 종료
- 두 경로 모두 `lifespan` async context의 teardown을 호출하므로 `ThreadPoolExecutor.shutdown(cancel_futures=True)` + 모든 `TaskState["running"] = False`가 실행됩니다.

## 임시 파일 청소

`core/paths.py::cleanup_stale(category, older_than_days=7)`. 부팅 시 자동으로 한 번 실행됩니다 (`compare`, `bench`, `vlm`, `embedding`). 수동 청소:

```python
from core.paths import cleanup_stale
cleanup_stale("compare", older_than_days=1)
```

## 모델 메타 캐시

`core/model_cache.py`는 `inspector` 라우트의 결과를 `settings/cache/model_meta/<sha>.json`에 저장합니다. 파일의 mtime/size가 바뀌면 자동 invalidate. 강제 초기화:

```python
from core.model_cache import clear
clear()
```

## 빌드 산출물

- 소스: `D:\bongkj\Projects\Visualizer\`
- PyInstaller 출력: `D:\bongkj\Projects\ssook\_internal\` (런타임 실행 시 사용)
- 새 모듈 추가 시 PyInstaller `spec` 파일에 hidden imports 갱신 필요 (`core/vlm_inference.py`, `core/hashing.py`, `core/env.py`, `core/paths.py`, `core/model_cache.py`, `core/exports.py`, `core/logging_setup.py`, `server/path_safety.py`, `server/middleware.py`, `server/errors.py`)
