# Project Rules — ssook

- 사용자가 한국어로 말하면 한국어로, 영어로 말하면 영어로 응답한다. 코드·식별자·로그·커밋 메시지는 영어.
- 작업 전 다음 파일들을 우선 확인한다 (모두 프로젝트 루트):
  - `prd.json` — 제품 요구사항 (있으면 따른다)
  - `FEATURE_AUDIT.md` — 30개 탭 기능별 알려진 리스크/병목/개선 작업의 backlog. 기능 개선 작업은 여기서 P0/P1/P2 항목을 찾아 시작.
  - `task.md`, `progress.txt` — 진행 중인 자동 반복(Ralph loop) 컨텍스트가 있을 수 있음.
  - `docs/backend-runbook.md`, `docs/general-features.md` — 운영·기능 문서.
- 본 문서를 수정할 때는 의미를 100% 보존하면서 가장 적은 단어로.
- 코드 변경(기능 추가/버그 수정/리팩토링/PR 머지) 후, `README.md`가 갱신될 필요가 있는지 확인. 설치·사용·구조·API가 바뀐 경우에만 최소 분량으로 업데이트.

## Repository Layout

- `server/` — FastAPI 라우터. 파일명은 `<feature>_routes.py`. 비즈니스 로직은 `core/`로 위임.
- `core/` — 추론·평가·최적화·해싱 등 도메인 로직. 라우터/IO와 분리.
- `web/` — 정적 프론트엔드 (vanilla JS + CSS, 빌드 단계 없음). 탭 정의는 `web/js/tabs.js`, `tabs-extra.js`.
- `tests/` — pytest. `conftest.py` 공통 픽스처. 파일명 `test_<module>.py`.
- `scripts/` — 빌드·EP 설치 보조 스크립트.
- `settings/app_config.yaml` — 영속 설정. `core/app_config.py`의 `AppConfig` 싱글톤이 RLock으로 보호.

## Testing

- 테스트 실행: `python -m pytest tests/ -v`.
- **conda env 주의**: `cv2`가 없는 base env에서는 import 실패. 검증된 환경은 `torch251` (`progress.txt` 참조). 사용자가 다른 env를 사용한다면 그 안에 `opencv-python`, `onnxruntime`, `pytest`가 모두 있는지 먼저 확인.
- 신규 기능과 버그 수정에는 가능한 테스트를 추가한다. 다만 단일 개발자 페이스를 깨뜨릴 만큼 엄격하지는 않음 — 영향 범위가 좁은 UI/문서/설정 변경은 테스트 없이 진행해도 됨.
- 테스트는 행위(observable behavior)를 검증한다. 구현 내부 디테일에 의존하지 않게.
- 동시성·상태 격리 테스트는 `test_state_concurrency.py` 패턴 참조 — `TaskState`의 RLock 거동을 검증.

## Logging

- 결정 지점·상태 전이·외부 호출 지점에 구조화 로그를 남긴다. 모든 줄에 남기지 않음.
- 컨텍스트 포함: 요청 ID, 입력 요약, 경과 시간, 결과(성공/실패 이유).
- 레벨: `ERROR`(조치 필요), `WARN`(복구 가능), `INFO`(비즈니스 이벤트), `DEBUG`(개발 진단).
- 자격증명·토큰·PII는 절대 기록하지 않음.
- Hot path(MJPEG 스트림 루프 등)에서는 로깅이 지연을 만들지 않도록 한다.

## Naming / Types / Comments

- 이름은 주변 코드 안 보고도 의미가 드러나야 함. `proc`/`mgr`/`tmp` 같은 약어 회피. `user_email > e`, `calculate_shipping_cost > calc`. 다만 과하게 길지도 않게.
- 불리언은 `is_*`/`has_*`/`should_*` 형식.
- Python 함수 시그니처에는 타입 어노테이션을 단다. 변수도 타입이 자명하지 않으면 명시.
- 키워드 인자가 가능한 언어에서는 사용 — 특히 다중 동일 타입 인자나 boolean 플래그.
- 코멘트는 **WHY**를 적는다. WHAT은 코드가 이미 말해줌. 비즈니스 규칙, 우회책, "이 접근을 택한 이유" 위주로.
- TODO/FIXME에는 항상 이유를 같이 적는다: `TODO(이유)`.
- 코드가 바뀌면 그 코드의 코멘트도 같이 갱신·삭제. 낡은 코멘트가 코멘트 없는 것보다 나쁘다.

## Architecture Guardrails

- **Router → Core 분리 유지**: HTTP 의존 코드(FastAPI Request, BackgroundTasks 등)는 `server/`에만. `core/`는 순수 Python 라이브러리처럼 호출 가능해야 함.
- **공유 상태는 락으로 보호**: `_video_sessions` (viewer), `compare_state`/`error_analysis_state`/`conf_opt_state` 등 전역 dict가 락 없이 공유되면 race. `TaskState`(RLock) 패턴 사용.
- **경로 입력 검증**: 사용자 디렉토리/파일 경로는 `server/path_safety.py`의 `safe_image_dir/label_dir/model_file`로 정규화. traversal 방지.
- **Optional dependency 처리**: `matplotlib`/`sklearn`/`umap-learn` 등은 옵션. import 실패 시 라우터·UI에서 명시적 안내.
- **고정 배치 모델**: ONNX 모델이 fixed batch (>1) 일 수 있음. `model_loader`의 자동 감지를 신뢰하되, FPS·처리량 계산에서 batch_size로 나누기 잊지 말 것.

## Formatting

- 현재 프로젝트에 통일된 포매터 설정이 없음(`pyproject.toml`에 `[tool.black]`/`[tool.ruff]` 없음).
- Python을 광범위하게 만지는 작업이라면 `black .`을 권장. 단, 기존 코드 스타일과 충돌하면 변경 범위 내에서만 일관성을 유지(전역 reformat 금지).
- JS/CSS는 현재 vanilla + 수기 정렬. `prettier`로 일괄 재포맷하면 diff가 거대해지므로, 새로 추가하는 파일·함수에만 적용.

## Git / Commit / Branch

- 커밋 author: 로컬 `git config user.name`/`user.email`을 그대로 사용. 현재 단일 메인테이너(`surrealier`).
- 커밋 메시지 컨벤션 (기존 커밋과 일치):
  - 형식: `<type>: <subject>` 또는 `<type>(<scope>): <subject>`.
  - type: `feat`, `fix`, `perf`, `refactor`, `ui`, `backend`, `tests`, `docs`, `chore`.
  - 예: `perf: add stride=2 + async preprocess for sequential models`.
  - Sign-off는 사용하지 않음. `git commit -s` 불필요.
- 브랜치: `<type>/<short-description>` (현재 작업 브랜치: `feat/sequential-detection`).
- PR이 가능한 환경이면 PR을 통해, 그렇지 않으면 feature 브랜치 → main 머지. 직접 main 커밋은 사소한 문서 변경 외에는 피함.
- worktree는 강제하지 않음 (단일 개발자 + 단일 브랜치 페이스). 병렬 작업이 필요한 경우에만 사용.
- 사용자가 명시적으로 요청하지 않은 한 다음은 절대 하지 않음: `git push --force`, `git reset --hard`, 브랜치 삭제, `git commit --no-verify`.

## Working with FEATURE_AUDIT.md

- 기능 개선 작업의 1차 backlog. 각 탭마다 P0/P1/P2 + 파일/함수/작업량(S/M/L)이 명시되어 있음.
- 개별 항목 작업 시:
  - 어느 [3] 항목(괴리/리스크)을 해결하는지 PR 본문에 인용.
  - 해결 후 `FEATURE_AUDIT.md`의 해당 라인을 갱신하거나, 해결 완료 항목을 정리한 별도 섹션을 본문 끝에 추가.
- 새로 발견한 리스크는 `FEATURE_AUDIT.md`의 해당 탭 [3]에 추가 — 진단 결과를 잃지 않게.

## UI / Design

- 디자인 가이드: `DESIGN.md` (Airtable 스타일 — 흰 캔버스, 차분한 type, signature 카드).
- 신규 UI는 `web/js/tabs.js` 또는 `tabs-extra.js`의 기존 패턴(`makeTab(opts)`)을 따른다.
- i18n: 모든 사용자-노출 텍스트는 `web/js/i18n.js`의 키 사용. 한국어/영어 양쪽 키를 같이 추가.
- 사이드바 탭은 `web/js/app.js`의 `_nav` 구조에 등록해야 사용자가 발견 가능 (`FEATURE_AUDIT.md` 7번 섹션 참조 — 7개 탭이 미등록 상태).

## Reference Material

- `docs/`에 영어·한국어·일본어·중국어 문서 트리. 새 문서는 영어 + 한국어를 우선.
- 외부 패턴이 필요할 때는 well-maintained OSS를 우선 참조하되, 본 프로젝트 코드 스타일·아키텍처를 우선시.
- 큰 변경 전 `FEATURE_AUDIT.md`, `prd.json`, `docs/backend-runbook.md`를 먼저 확인.

## Don'ts

- 무의미한 backwards-compat 셰임(`_x`/주석으로 남기는 deprecated 코드) 두지 않는다 — 단일 메인테이너 프로젝트이므로 깨끗이 삭제.
- 가설적 미래 요구를 위한 추상화·인터페이스를 미리 만들지 않는다 (YAGNI).
- 시스템 경계가 아닌 곳에 방어적 입력 검증을 추가하지 않는다.
- 외부 사용자 경로/입력은 항상 `path_safety`로 검증.
- "임시" 코드를 위한 feature flag/세팅을 도입하지 않는다 — 그 자리에서 코드를 바꾼다.
