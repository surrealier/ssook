# ssook QC 통합 리포트 / Consolidated QC Report

- 생성일 / Generated: **2026-05-29**
- 베이스라인 / Baseline: **181 tests passing**, 검증 환경 / verified env **`torch251`** (torch 2.5.1+cu124, CUDA True, transformers 4.57.3)
- 입력 / Input: 10개 섹션 감사 (Inference, Evaluation, Analysis, Tools, Data, Quality, Specialized, VLM deep-dive, Cross-cutting, Wiring)
- 산출 / Output: 교차 섹션 중복 제거 후 **49개 finding** (confirmed=false 중 feature 카테고리 VLM 설계 7건 유지, 나머지 stale 항목은 본문에서 별도 표기)

이 리포트의 `deduped_findings`(구조화 출력)가 구현 단계를 구동한다. 각 finding의 `fix_files`는 릴리스 계획의 소유권 트랙으로 라우팅된다.

---

## 1. 요약 대시보드 / Summary Dashboard

### 1.1 심각도별 / By Severity

| Severity | Count | 설명 |
|---|---|---|
| **P0** | 8 | 런타임 크래시 / 핵심 기능 완전 불능 / 헤드라인 메트릭 오류 |
| **P1** | 22 | 침묵 오작동 / 보안 경계 위반 / 취소 불능 / 발견 불가 |
| **P2** | 19 | UX·성능·dead-code·i18n 개선 |
| **합계** | **49** | (implement_now=true 32건 / deferred 17건) |

### 1.2 카테고리별 / By Category

| Category | Count |
|---|---|
| correctness | 17 |
| concurrency | 7 |
| security | 7 |
| feature | 9 |
| ux | 5 |
| performance | 6 |
| dead-code | 4 |
| contract | 3 |
| i18n | 2 |
| dependency | 1 |

(일부 finding은 카테고리가 통합되며 합계는 49를 약간 초과할 수 있음 — 대표 카테고리 기준)

### 1.3 탭별 핫스팟 / Hotspots by Tab

| 핫스팟 영역 | 대표 finding | 위험 요지 |
|---|---|---|
| **server/data_routes.py** | DATA-01, DATA-02 (P0×2) | `glob_module` 미정의 → Converter/Remapper 100% 크래시; TaskState 5개 plain-dict shadowing → force-stop/태스크큐 무력 |
| **server/system_routes.py** | CONFIG-01 (P0) | `_auto_cleanup_memory()` 4개 미정의 심볼 → 메모리 압박 시 `/api/system/hw` 500 |
| **navigation (web/js/app.js)** | WIRE-02 (P0) | 7개 탭(seg/clip/embedder/pose/instance-seg/tracking/vlm) `_nav` 미등록 → 사용자 도달 불가 |
| **server/extra_routes.py (segmentation)** | SPEC-02, SPEC-03 (P0×2) | `evaluate_segmentation` 시그니처 불일치 TypeError + 프론트 필드명 422 |
| **VLM (model_routes/tabs-extra)** | VLM-01, VLM-02 (P0×2) | 존재하지 않는 `/api/model/infer` 404 + 전역 `cfg.model_type` 라우팅 오류 |
| **core/evaluation.py** | EVAL-01 (P0) | mAP@50:95 argmax-once 매칭 → @50과 불일치, 체계적 저평가 |
| **server/eval_routes.py + tabs.js** | EVAL-02 (P0) | Stop 완전 무력 + 재실행 시 두 워커가 동일 state 경쟁 |
| **server/analysis_routes.py** | AN-01 (P0) | error-analyzer 클래스 무시 매칭 → FP/FN 통계 전부 오류 |
| **server/quality_routes.py + tabs-extra.js** | QUAL-01 (P0) | similarity `query` vs `query_path` 필드명 불일치 → 검색 영구 무동작 |
| **path safety (전역)** | INF/AN/EVAL/DATA 다수 (P1) | data/analysis/eval 라우트 다수 user path 미검증 (읽기+쓰기 traversal) |

---

## 2. 섹션별 상세 / Detailed Findings

표기: `[ID] 제목 | severity·category | file:lines | implement_now`

---

### 2.1 Inference / Viewer · Settings (track: viewer, config, shared, fe-main)

#### [VIEWER-01] `_auto_cleanup_memory()` — 4개 미정의 전역으로 `/api/system/hw` NameError 500
- **P0 · correctness** · `server/system_routes.py:58-88` · **implement_now: true**
- (병합: INF-01 + CC-02. 입력에서 P1으로 보고되었으나 cross-cutting 섹션이 P0로 격상 — 메모리 압박 시 HW 폴링 루프 전면 중단이므로 P0 채택.)
- **증거**: `system_hw()`가 RSS>50% RAM 시 try/except 없이 `_auto_cleanup_memory()` 호출(62). 본문(68-88)이 `_cleanup_stale_sessions`(viewer_routes 소속), `_compare_state`/`_embedding_state`(실제 `compare_state`/`embedding_state`, underscore 없음, server/state.py), `_palette_cache`(server/utils.py)를 참조 — 4개 모두 모듈 네임스페이스에 부재. grep으로 76·78·82·84행 확인.
- **영향**: RSS가 시스템 RAM 50% 초과(대형 ONNX 로드 시 흔함)하는 순간부터 `/api/system/hw`가 500, 뷰어 HW 패널 갱신 중단, 의도한 캐시 정리는 영영 실행 안 됨 — 가장 필요한 시점에 기능이 죽고 해롭게 동작.
- **제안 수정**: 실제 심볼 import(`from server.viewer_routes import _cleanup_stale_sessions`; `from server.state import compare_state, embedding_state`; `from server import utils`), `compare_state`/`embedding_state`는 TaskState 메서드 사용, `_palette_cache`는 utils 헬퍼로 trim. 전체를 try/except로 감싸 cleanup 버그가 HW 엔드포인트를 절대 500 못 시키게.
- **fix_files**: `server/system_routes.py`, `server/utils.py`, `server/viewer_routes.py`
- **effort**: S · **회귀위험**: low
- **테스트**: psutil RSS를 50% 초과로 monkeypatch → `system_hw()`가 예외 없이 dict 반환, cleanup side effect 실행 확인.

#### [VIEWER-02] `_video_sessions` 무락 전역 dict (RLock 가드레일 위반, 반복 중 변경 race)
- **P1 · concurrency** · `server/viewer_routes.py:21-43, 73-95, 305-329` · **implement_now: true**
- (병합: INF-02 + CC-07.)
- **증거**: line 21 `_video_sessions = {}` plain dict. `_cleanup_stale_sessions()`(29-31)가 `.items()` 순회 후 pop, `viewer_status()`(307)에서 ~300ms마다 호출. 동시에 `viewer_start()`(73) insert, `viewer_stop()`(325) pop, StreamingResponse generate() 스레드가 sess 필드 read + `cap.release()`. CLAUDE.md 가드레일이 명시적으로 `_video_sessions`를 TaskState/RLock 보호 대상으로 지정. 락 없음 → 'dict changed size during iteration' + 스트림 read 중 cap release race.
- **영향**: viewer_status에서 간헐 RuntimeError(폴러에 500), 세션 시작/중지 중 정리와 겹치면 torn read, generate() 스레드가 해제된 cap에서 read하여 크래시/garbage frame.
- **제안 수정**: 모듈 레벨 `threading.RLock` (`_sessions_lock`)으로 모든 mutation·정리 스캔·cap.release() 가드. stale id를 락 안에서 snapshot 후 락 밖에서 release. generate()는 자체 캡처한 `sess` local 유지 + 각 read 전 `playing` 재확인.
- **fix_files**: `server/viewer_routes.py`
- **effort**: M · **회귀위험**: low
- **테스트**: test_state_concurrency 패턴 — N 스레드가 insert/pop 난타하는 동안 `_cleanup_stale_sessions()` 루프, RuntimeError 없음 단언.

#### [VIEWER-03] seek/step 시 tracker 미리셋 — ID·궤적이 불연속 지점에서 점프
- **P1 · correctness** · `server/viewer_routes.py:116-131` · **implement_now: true**
- **증거**: generate()의 seek 핸들러(117-124)는 `_frame_buffer`/`_seq_tensor_buf`/`_seq_frame_counter`/`_seq_last_result`만 클리어, step 핸들러(127-131)는 cap 재배치만. 둘 다 `sess['tracker'].reset()` 미호출. SORT/ByteTracker 모두 reset()(core/tracking.py:64-67, 134-137) 존재.
- **영향**: seek/step 후 궤적선이 화면을 가로질러 streak, 사라진 객체의 ID 유지, ByteTrack/SORT 통계 손상.
- **제안 수정**: seek·step 두 분기 모두 재배치 후 `tracker = sess.get('tracker'); if tracker: tracker.reset()`.
- **fix_files**: `server/viewer_routes.py`
- **effort**: S · **회귀위험**: low
- **테스트**: ByteTracker로 세션 구성 → 검출 push로 track 성장 → seek 분기 시뮬 → `tracker.tracks == []` and `_next_id == 1` 단언.

#### [VIEWER-04] VLM 모델 viewer 차단 안 됨 — 스트림이 garbage 검출 박스 그림
- **P1 · correctness** · `server/viewer_routes.py:60-61, 154-271` · **implement_now: true**
- **증거**: viewer_start는 `task_type=='embedding'`(CLIP)만 차단(60-61). VLM은 task_type 'vlm'(model_loader.py:186-187)이라 미차단. generate()의 분기 체인에서 vlm_은 else(212)로 떨어져 run_inference→postprocess_v8/v5가 VLM logits에 적용되어 무의미 박스 그림(255-271). 이미지 경로(model_routes.py:248)는 task_type로 올바르게 처리 → viewer가 비일관.
- **영향**: VLM 선택+Play 시 의미 없는 사각형이 영상 위에 그려짐, 미지원 안내 없음.
- **제안 수정**: `if model.task_type in ('embedding', 'vlm'):` 로 가드 확장, 명확한 에러 반환. ko+en i18n 키 추가.
- **fix_files**: `server/viewer_routes.py`, `web/js/i18n.js`
- **effort**: S · **회귀위험**: low
- **테스트**: task_type='vlm' ModelInfo stub로 viewer_start 호출 → 응답에 'error' 단언.

#### [VIEWER-05] 배치-크기 UI 컨트롤이 no-op (cfg.batch_size를 추론 경로가 안 읽음)
- **P2 · ux** · `web/js/tabs.js:21-23, 278-280` · **implement_now: true**
- **증거**: viewer가 batch-size `<input id='v-batch-size'>`(tabs.js:21-23) 렌더, `_onBatchChange`(278-280)가 `/api/config`에 POST하여 cfg.batch_size 영속. 그러나 모든 추론 경로는 model_info.batch_size(ONNX dim0 자동감지, model_loader.py:313-314)만 읽음. cfg.batch_size는 core/benchmark_runner.py:158에서만 읽힘.
- **영향**: 사용자가 viewer 배치 필드 변경 → 효과 zero, 오해 유발 컨트롤.
- **제안 수정**: viewer 패널 렌더 + `_onBatchChange`에서 v-batch-size 컨트롤 제거(벤치마크 전용). 또는 dynamic-batch 모델 한정으로 run_inference가 cfg.batch_size 존중.
- **fix_files**: `web/js/tabs.js`
- **effort**: S · **회귀위험**: low

#### [VIEWER-06] MJPEG generate() 예외 시 에러 프레임 미생성 — 스트림 침묵 정지
- **P2 · ux** · `server/viewer_routes.py:114-302` · **implement_now: true**
- **증거**: except 블록(295-299)이 traceback print + `playing=False`만, 마지막 multipart 프레임 미전달. 브라우저는 마지막 디코드 프레임 표시 유지. `_pollStatus`가 결국 'playback_done'만 표시 — 정상 종료와 구분 불가.
- **영향**: 스트림 중 추론/postprocess 오류가 정상 종료처럼 보이고 frozen frame만 남음.
- **제안 수정**: except에서 에러 placeholder JPEG(cv2.putText) 인코딩 후 마지막 '--frame' 청크 yield. `sess['error']` 필드를 viewer_status가 노출, `_pollStatus`가 에러 vs 정상 구분(ko+en i18n 키).
- **fix_files**: `server/viewer_routes.py`, `web/js/tabs.js`, `web/js/i18n.js`
- **effort**: S · **회귀위험**: low

#### [VIEWER-07] 커스텀 모델 coord_format dead config; conf_class argmax가 비연속 클래스 인덱스 오매핑
- **P2 · correctness** · `core/inference.py:530-590` · **implement_now: true**
- **증거**: (a) `CustomModelType.coord_format`(app_config.py:32, 영속 92/116)이 postprocess_custom에서 미사용 — 좌표 처리는 attr_roles 존재 여부로만 결정(595-642). `CustomModelTypeRequest`(config_routes.py:87-96)가 coord_format를 받지도 않음 → 항상 'xyxy' dead state. (b) conf_indices가 'conf_class' 시작 role의 열 인덱스 수집(557-558), `class_ids = class_scores.argmax(axis=1)`(582)는 선택 열 중 위치(0..n-1) 반환 — trailing 클래스 번호 아님. 비연속(conf_class0, conf_class5) 매핑 시 column ordinal 보고.
- **영향**: (a) UI가 설정 못 하는 오해 유발 dead config. (b) sparse 매핑에서 검출이 잘못된 class id 라벨 → 클래스명/스타일 오염.
- **제안 수정**: (a) coord_format을 CustomModelType·load/save·YAML에서 삭제 OR 요청에 배선. (b) conf_class role의 숫자 suffix를 파싱해 `class_index_map` 구성, `class_ids = class_index_map[argmax]`로 실제 class id 변환.
- **fix_files**: `core/inference.py`, `core/app_config.py`, `server/config_routes.py`, `settings/app_config.yaml`
- **effort**: M · **회귀위험**: medium

#### [CONFIG-02] 네이티브 파일 다이얼로그 PowerShell 필터 문자열 인젝션
- **P2 · security** · `server/system_routes.py:153-191, 253-270` · **implement_now: true**
- **증거**: `_native_file_dialog`가 filter_str(req.filters 유래)를 `$d.Filter = '{filter_str}';`(178)에 직접 보간, `subprocess.run(['powershell','-NoProfile','-Command', script])`. req.filters는 POST `/api/fs/select`·`/api/fs/select-multi`로 완전 사용자 제어. single quote로 PS 문자열 리터럴 탈출 → 임의 명령 인젝션.
- **영향**: 악성 브라우저 탭/향후 LAN 노출 시 로컬 명령 실행. 단일 사용자 로컬이라 가능성 낮으나 실제 shell injection.
- **제안 수정**: PS 소스에 사용자 데이터 보간 금지. single quote doubling(`replace("'", "''")`) 모든 보간 값에, 또는 환경변수/`-EncodedCommand`로 전달. macOS/Linux 분기도 동일 escape.
- **fix_files**: `server/system_routes.py`
- **effort**: S · **회귀위험**: low

#### [VIEWER-08] viewer/image 경로 path_safety 우회; sess['conf'] dead; seq ThreadPool 누수; JPEG 품질 하드코딩 65
- **P2 · dead-code** · `server/viewer_routes.py:53-95, 105-302` · **implement_now: false**
- **증거**: (1) viewer_start가 req.video_path/req.model_path를 raw로 cv2.VideoCapture/ensure_model에 전달(58,63), path_safety 없음(infer_image model_routes.py:216-217도 동일). (2) `sess['conf']`(74) 설정되나 미읽힘 — generate()는 cfg.conf_threshold 사용. (3) `_start_async_preprocess`가 `model_info._seq_thread_pool`(inference.py:245-248) 생성하나 어디서도 shutdown 안 됨 → 모델 reload마다 스레드 누수. (4) JPEG 품질 6곳 imencode 모두 65 하드코딩(이미지 경로는 85).
- **제안 수정**: (1) safe_model_file/safe_video_file 래핑. (2) sess['conf']/VideoStartRequest.conf 제거 또는 run_inference에 전달. (3) viewer_stop/cleanup에서 `_seq_thread_pool.shutdown(wait=False, cancel_futures=True)`. (4) `stream_jpeg_quality` config 추가.
- **fix_files**: `server/viewer_routes.py`, `server/model_routes.py`, `core/inference.py`, `core/app_config.py`
- **effort**: M · **회귀위험**: medium

> **STALE (코드 변경 불필요)** — INF-10: FEATURE_AUDIT.md:84 "class styles not applied to viewer boxes"는 현재 코드(viewer_routes.py:232-270)가 `get_class_style`/`get_color`를 올바르게 적용하므로 **미확인**. FEATURE_AUDIT.md 라인 갱신만 권장.

---

### 2.2 Evaluation / benchmark · evaluation (track: eval, benchmark)

#### [EVAL-01] mAP@50:95 argmax-once 매칭이 단일 GT 이중부킹 — @50 greedy와 불일치, TP 저평가
- **P0 · correctness** · `core/evaluation.py:153-175 vs 80-91` · **implement_now: true**
- **증거**: `evaluate_dataset`는 매칭된 GT를 re-mask 후 argmax(84-86)로 올바른 per-image greedy. `evaluate_map50_95`는 전역 best_j를 한 번 argmax(155-164)하고 나중에 `best_j not in matched`(171)만 검사. 두 예측이 두 GT에 임계 초과지만 동일 전역 argmax 공유 시 저점 예측이 FP 강제. 재현: gt 2박스, pred 2박스 → @0.5에서 evaluate_dataset recall/precision=1.0이지만 evaluate_map50_95=0.505.
- **영향**: 헤드라인 mAP@50:95이 군집 객체 데이터셋에서 체계적 저평가, 옆 칸 mAP@50과 내부 비일관.
- **제안 수정**: 공유 헬퍼 `_match_greedy(iou_row, matched_mask, thr)` 추출, evaluate_map50_95도 동일 re-masked greedy 사용(임계마다 점수순 예측 순회, 미매칭 GT 열만 argmax).
- **fix_files**: `core/evaluation.py`
- **effort**: M · **회귀위험**: medium
- **테스트**: 군집 2박스 케이스 → evaluate_map50_95==1.0; 단일 완전매칭==1.0; map50:95 <= map50 불변식.

#### [EVAL-02] Stop 비기능: 프론트가 백엔드 미호출, 워커가 running 미검사, stop이 동시 두번째 워커 경쟁 유발
- **P0 · concurrency** · `web/js/tabs.js:1618-1623; server/eval_routes.py:138-281, 578-581` · **implement_now: true**
- (병합: EVAL-02 + WIRE-04.)
- **증거**: `Tabs.evaluation.stop()`(1618-1623)은 `_polling=false`만, `/api/eval/stop` 미호출(벤치마크 stop은 호출). 백엔드 `_run` 루프(188-247)·세부 루프가 `eval_state['running']` 미독. `/api/eval/stop`(578-581)이 running=False 설정해도 워커 무시. 게다가 stop 후 `run_evaluation_async` 가드(119) 통과 → 두 번째 _run이 동일 eval_state results/progress를 동시 변경(data race).
- **영향**: 긴 평가 취소 불능, Stop 후 Run 시 두 워커 interleaved 쓰기로 results 손상.
- **제안 수정**: (1) tabs.js stop()에서 `await API.get('/api/eval/stop')` 선행. (2) 각 _run 루프(detection per-image + classification/seg/clip/embedder) 상단에 `if not eval_state['running']: eval_state.update(running=False, msg='Stopped'); return`. (3) 워커가 자체 terminal 경로로만 running=False.
- **fix_files**: `web/js/tabs.js`, `server/eval_routes.py`
- **effort**: M · **회귀위험**: low

#### [EVAL-03] GT 클래스명 리매핑 dead: eval-classmap 요소 참조되나 미렌더 → 매핑 다이얼로그가 항상 class_N
- **P1 · dead-code** · `web/js/tabs.js:1201, 1270` · **implement_now: true**
- **증거**: run()이 `getElementById('eval-classmap')?.value`(1270), _onTaskChange가 `eval-classmap-group`(1201) 읽으나 render()(1117-1193)에 두 id 모두 없음. help-annotations-main.js:130이 문서화한 `#eval-classmap` textarea도 DOM에 부재. → classmapNames 항상 {}, gtItems 이름이 `class_${id}` fallback.
- **영향**: 광고된 'GT 클래스 id:name 매핑' 기능 비가시/비기능, 매핑 다이얼로그가 무의미 'class_0/1...' 표시.
- **제안 수정**: render()에 `<textarea id="eval-classmap">` + `eval-classmap-group` 컨테이너 복원, classmapNames 채우도록 배선. i18n 키 추가.
- **fix_files**: `web/js/tabs.js`, `web/js/i18n.js`, `web/js/help-annotations-main.js`
- **effort**: S · **회귀위험**: low

#### [EVAL-04] path traversal: `/api/eval/load/{filename}` + run-async dir가 path_safety 우회
- **P1 · security** · `server/eval_routes.py:774-782, 116-195` · **implement_now: true**
- (병합: EVAL-04 + CC-05.)
- **증거**: `eval_load`가 `os.path.join(_EVAL_HISTORY_DIR, filename)`(778)을 미검증 path param에서 구성, realpath-containment/확장자 검사 없이 json.load. Windows backslash로 escape 확인. run_evaluation_async가 req.img_dir/label_dir/model을 glob/open/_load_model(160,170,195)에 raw 사용 — safe_image_dir/safe_label_dir/safe_model_file 없음.
- **영향**: LAN/CSRF 시나리오에서 임의 파일 읽기, 임의 경로 모델 로드.
- **제안 수정**: eval_load는 `os.path.basename(filename) != filename` 거부 + `.json` 강제 + realpath containment. run-async/run은 safe_image_dir/safe_label_dir/safe_model_file로 래핑, benchmark도 동일.
- **fix_files**: `server/eval_routes.py`, `server/benchmark_routes.py`
- **effort**: S · **회귀위험**: low

#### [EVAL-05] 단일 손상 GT 라벨 라인이 전체 멀티모델 검출 평가 중단
- **P1 · correctness** · `server/eval_routes.py:172-178 (+645-651 legacy)` · **implement_now: true**
- **증거**: GT 로드가 `boxes.append((int(parts[0]), *map(float, parts[1:5])))`을 len>=5만 가드. class id '0.0'·헤더 라인·비숫자 토큰이 ValueError → 외곽 `except Exception`(280-281)이 전체 모델 run 중단.
- **영향**: 흔한 실데이터의 한 줄 오류로 전체 멀티모델 run이 불투명 에러로 실패.
- **제안 수정**: per-line parse를 try/except ValueError: continue로 감싸고 skip 카운트를 msg에 노출. run-async + legacy 양쪽.
- **fix_files**: `server/eval_routes.py`
- **effort**: S · **회귀위험**: low

#### [EVAL-06] 이미지 캐시가 고정 500 하드캡 + 무축출(LRU 아님) — 500장 초과 시 모델마다 디스크 재디코드, ~3GB 점유
- **P1 · performance** · `server/eval_routes.py:184-224` · **implement_now: true**
- **증거**: `_IMG_CACHE_LIMIT=500; _img_cache={}` 후 `if ... len(_img_cache)<_IMG_CACHE_LIMIT`(185,223). 500 채워지면 이후 이미지는 모델 반복마다 imread 재디코드(축출 없음). 캐시된 500 디코드 BGR이 run 내내 유지(277에서만 클리어) → 500×1080p×3 ≈ 3GB.
- **영향**: >500 이미지×N 모델에서 501..end가 N번 재디코드, 동시에 첫 500이 멀티-GB 낭비. 재사용 최적화가 가장 필요할 때 무력화.
- **제안 수정**: OrderedDict 기반 LRU(메모리 예산 기준) OR 루프 순서 반전(이미지당 1회 디코드, 모든 모델 실행). 루프 반전이 가장 단순.
- **fix_files**: `server/eval_routes.py`
- **effort**: M · **회귀위험**: medium

#### [EVAL-07] IoU 매칭이 모델당 3회 독립 계산 (evaluate_dataset + evaluate_map50_95 + _build_confusion_matrix)
- **P2 · performance** · `server/eval_routes.py:249-264; core/evaluation.py:49-194` · **implement_now: true**
- **증거**: 모델당 evaluate_dataset(0.5)(249), evaluate_map50_95(250), _build_confusion_matrix(264) 각각 독립적으로 class×image IoU 계산. 혼동행렬은 스칼라 _iou 별도 재실행.
- **영향**: 대형 데이터셋에서 매칭 비용 ~3배.
- **제안 수정**: 매칭 1회 계산 — evaluate_dataset(0.5)이 per-image 매칭 할당도 방출하여 혼동행렬 도출. evaluate_map50_95는 EVAL-01 공유 헬퍼+사전계산 IoU 재사용. 최소한 _build_confusion_matrix의 스칼라 _iou를 _compute_iou_matrix로 교체.
- **fix_files**: `core/evaluation.py`, `server/eval_routes.py`
- **effort**: M · **회귀위험**: medium

#### [BENCH-01] nvidia-smi를 벤치 루프 내부 동기 spawn; GPU/CPU 20반복마다만 샘플 → 거칠고 지연 주입
- **P2 · performance** · `core/benchmark_runner.py:42-66, 256-260` · **implement_now: true**
- **증거**: 루프 내 `if i % 20 == 0:`(256)가 `cpu_percent(interval=None)`+`_smi_query()`(subprocess.check_output, 42-66) 실행 — 반복 사이에 프로세스 spawn(수십~수백ms). 100반복 시 5샘플뿐.
- **영향**: GPU/CPU 수치 거칠고 노이즈, 측정 중 GPU 교란, run 연장.
- **제안 수정**: 전용 daemon 샘플러 스레드가 고정 wall-clock(예 250ms)로 폴링, 측정 루프 전 시작/후 정지하여 추론 cadence와 분리. mean/peak 집계.
- **fix_files**: `core/benchmark_runner.py`
- **effort**: M · **회귀위험**: low

#### [BENCH-02] decode_ms가 배치 FPS에 fold-in되어 throughput 과대; 비디오 decode_ms가 반복 단일프레임 file-open 측정
- **P2 · correctness** · `server/benchmark_routes.py:112-141` · **implement_now: true**
- **증거**: decode_ms는 한 프레임 디코드 평균(116-131)인데 batch 전체 시간 `r.mean_total_ms`에 더한 뒤 `fps = batch_size*1000/total`(134)로 곱함 → fixed-batch 과대. h264/h265는 매 rep마다 1-frame mp4에 VideoCapture open/read/release(126-129) — container-open 오버헤드 측정, 실제 스트림 디코드 아님.
- **영향**: fixed-batch FPS-with-decode 오류, 코덱 비교 오도.
- **제안 수정**: `total_with_decode = mean_total_ms + decode_ms * batch_size`. 비디오는 VideoCapture 1회 open 후 read만 루프 타이밍.
- **fix_files**: `server/benchmark_routes.py`
- **effort**: S · **회귀위험**: low

#### [EVAL-08] 혼동행렬 IoU 0.5 하드코딩 무UI; 대량 클래스가 (n+1)² DOM 테이블
- **P2 · ux** · `server/eval_routes.py:23, 264; web/js/tabs.js:1589-1617` · **implement_now: false**
- **제안 수정**: 혼동행렬 IoU를 detection 옵션 numeric input으로 노출. 렌더는 클래스 50+ 시 top-N+other 축약/가상화.
- **fix_files**: `server/eval_routes.py`, `web/js/tabs.js`, `web/js/i18n.js`
- **effort**: M · **회귀위험**: low

#### [EVAL-09] 클래스 매핑이 다대일(model→GT)만, 1 모델클래스→다중 GT 표현 불가
- **P2 · feature** · `server/eval_routes.py:203-243; web/js/tabs.js:1417-1492` · **implement_now: false**
- **제안 수정**: YAGNI 고려 — 현재 1:1 제약을 다이얼로그 힌트에 문서화 후 사용자 기대 검증; 필요 시 connections 값을 list로 확장.
- **fix_files**: `server/eval_routes.py`, `web/js/tabs.js`
- **effort**: L · **회귀위험**: medium

---

### 2.3 Analysis (analysis · model-compare · error-analyzer · conf-optimizer · embedding-viewer) (track: analysis, fe-extra)

#### [ANLY-01] FP/FN 매칭이 class_id 무시 + GT 클래스 라벨 미파싱
- **P0 · correctness** · `server/analysis_routes.py:188-213` · **implement_now: true**
- **증거**: GT parse가 `cx, cy, bw, bh = map(float, parts[1:5])`(189) — parts[0](class id) 미독. gt_boxes는 `[x1,y1,x2,y2,cy]`(192). 매칭 루프(198)가 IoU만 계산, `best_iou >= iou_threshold`(211)로 클래스 비교 없이 수락. dog 예측이 cat GT와 겹치면 TP로 카운트.
- **영향**: 멀티클래스 FP/FN-by-size-position 테이블 전부 오류, 오분류가 정답으로 채점.
- **제안 수정**: GT 로더에서 `cls=int(parts[0])` 읽어 `[x1,y1,x2,y2,cy,cls]` append. pred 루프를 `for pi,(pbox,_,pcid)`로 노출, 내부 GT 루프에 `if gt_matched[gi] or int(gb[5])!=int(pcid): continue`.
- **fix_files**: `server/analysis_routes.py`
- **effort**: S · **회귀위험**: low
- **테스트**: GT class=0, pred class=1 완전겹침 → 1 FP + 1 FN 단언; 동일클래스 겹침 → 0/0.

#### [ANLY-02] embedding-viewer 메서드 dropdown 값('t-SNE'/'UMAP'/'PCA')이 백엔드('tsne'/'umap'/'pca')와 불일치 — UMAP/PCA가 침묵 t-SNE 실행
- **P1 · correctness** · `web/js/tabs-extra.js:367, 391` · **implement_now: true**
- **증거**: `<select id="ev-method"><option>t-SNE</option>...`(367), value 속성 없어 `.value`가 텍스트 반환. 백엔드는 `=="pca"`/`=="umap"`/else t-SNE(433-443). 'PCA'!='pca' → 항상 else.
- **영향**: UMAP/PCA 선택이 항상 t-SNE 실행, 제목은 선택 메서드 표기(오라벨).
- **제안 수정**: `<option value="tsne">`/`value="umap"`/`value="pca"`. 백엔드도 `m = req.method.lower()` 방어.
- **fix_files**: `web/js/tabs-extra.js`
- **effort**: S · **회귀위험**: low

#### [ANLY-03] 5개 analysis 라우트 모두 img_dir/label_dir/model_path를 path_safety 미검증
- **P1 · security** · `server/analysis_routes.py:55, 153, 182-185, 271, 294-297, 387, 396, 508` · **implement_now: true**
- (병합: AN-03 + CC-04 의 analysis 부분.)
- **증거**: model-compare/error-analysis/conf-optimizer/embedding-viewer/inference-analysis 어느 것도 safe_image_dir/safe_label_dir/safe_model_file 미호출. label_dir이 open()으로, model_path가 onnxruntime로 직접 흐름.
- **영향**: cross-origin POST로 임의 디렉토리/파일 접근, 임의 .onnx 로드.
- **제안 수정**: 각 핸들러 상단(또는 Pydantic field_validator)에서 safe_image_dir/safe_label_dir/safe_model_file/safe_image_file 적용, UnsafePathError를 JSON 봉투로 반환.
- **fix_files**: `server/analysis_routes.py`
- **effort**: M · **회귀위험**: low

#### [ANLY-04] force-stop no-op: compare/error/conf/embedding 워커가 loop에서 running 미검사 + stop된 워커가 msg 덮어씀 + 새 run이 old와 경쟁
- **P1 · concurrency** · `server/analysis_routes.py:63-90, 173-235, 284-306, 384-476` · **implement_now: true**
- (병합: AN-04 + CC-06.)
- **증거**: force_stop이 `state['running']=False`+msg='Stopped'(server/__init__.py:213) 설정하나 compare 루프(63)·error/conf/embedding 루프(173,284,412)가 loop 내 running 미재검. 종료 시 `update(running=False, msg='Complete')`(88)로 사용자 'Stopped' 덮어씀. start 가드(34) 즉시 통과 → 두 번째 run이 동일 compare_state results/_tmp_dir 변경.
- **영향**: Stop 무력, 재실행 시 results 손상 + 삭제된 temp dir에 imwrite 크래시.
- **제안 수정**: 각 루프 본문 상단에 `if not <state>['running']: <state>['msg']='Stopped'; return`. 종료 시 'Complete'를 무조건 설정하지 말고 완주 시에만.
- **fix_files**: `server/analysis_routes.py`
- **effort**: S · **회귀위험**: low

#### [ANLY-05] model-compare conf 0.25 하드코딩, UI 컨트롤 없음
- **P2 · ux** · `web/js/tabs-extra.js:124` · **implement_now: true**
- **증거**: run()이 `conf: 0.25` 리터럴 POST, render() 폼에 conf input 없음.
- **제안 수정**: `<input id='cmp-conf' value='0.25' min/max/step>` 추가, i18n cmp.conf, run()에서 `conf: +value` 전송. 백엔드 무변경.
- **fix_files**: `web/js/tabs-extra.js`, `web/js/i18n.js`
- **effort**: S · **회귀위험**: low

#### [ANLY-06] conf-optimizer가 GT에 존재하는 클래스만 sweep, 모델이 예측하나 GT에 없는 클래스 비가시
- **P2 · correctness** · `server/analysis_routes.py:309, 314-316` · **implement_now: false**
- **제안 수정**: `class_ids = set(g[0] for g in all_gt) | set(p[0] for p in all_preds)`. 기존 F1 수식이 gt_cls=[] 처리.
- **fix_files**: `server/analysis_routes.py`
- **effort**: S · **회귀위험**: low

#### [ANLY-07] embedding plot이 고정 전역 temp 경로, 정리/run 격리 없음
- **P2 · correctness** · `server/analysis_routes.py:468-475, 481-489` · **implement_now: false**
- **제안 수정**: `f'ssook_embedding_{uuid4().hex}.png'` per-run 경로, status에서 read 후 os.remove. 또는 base64를 state에 유지.
- **fix_files**: `server/analysis_routes.py`
- **effort**: S · **회귀위험**: low

#### [ANLY-08] embedding 추출 루프 진행률 미보고 (total 미설정, 바 0% 고착)
- **P2 · ux** · `server/analysis_routes.py:411-423, 482-489` · **implement_now: false**
- **제안 수정**: 루프 전 `embedding_state['total']=len(files)`, 루프 내 `progress=i+1`. 프론트 무변경.
- **fix_files**: `server/analysis_routes.py`
- **effort**: S · **회귀위험**: low

#### [ANLY-09] inference-analysis가 요청당 추론 2회(낭비 forward pass)
- **P2 · performance** · `server/analysis_routes.py:515-523` · **implement_now: false**
- **제안 수정**: letterbox 결과 재사용, 중복 preprocess() 제거.
- **fix_files**: `server/analysis_routes.py`
- **effort**: S · **회귀위험**: low

---

### 2.4 Tools (inspector · profiler · calibration · diagnose) (track: extra, fe-extra)

#### [TOOLS-01] 채널 프루닝이 비자명 토폴로지(grouped conv, Concat/Add fan-in, 검출 헤드)에서 구조 손상 모델 생성, 롤백 없음
- **P0 · correctness** · `core/optimizers/channel_pruning.py:30-114` · **implement_now: true**
- **증거**: apply()가 Conv 출력 채널 프루닝(55)하고 단일 직결 Conv(BN/Relu 경유)만 입력 채널 복구(90-108). (a) grouped conv `group` attr 미변경 → ORT 거부; (b) Concat/Add/Split feed 시 sibling 차원 불일치; (c) 검출 헤드 출력 채널(num_classes) 손상. save(114) 후 검증/롤백 없음. RecommendationEngine이 'executable: True'로 자동 제안.
- **영향**: 사용자 한 번 클릭 → 로드 실패하거나 침묵 오검출 ONNX, UI는 성공 보고.
- **제안 수정**: (1) 프루닝 후 shape-inference + dummy InferenceSession 로드, 실패 시 원본 복사+에러(롤백). (2) `group!=1` Conv skip. (3) 단일-Conv 체인 외 consumer(Concat/Add/Mul/Split/Resize 또는 다중 consumer) skip. (4) graph 출력 feed하는 terminal Conv skip. skipped 레이어 보고.
- **fix_files**: `core/optimizers/channel_pruning.py`, `core/model_diagnosis.py`
- **effort**: M · **회귀위험**: low

#### [TOOLS-02] mixed-precision 제외가 node.name 키링 → 빈 이름 빈번한 ONNX에서 침묵 no-op
- **P1 · correctness** · `core/optimizers/mixed_precision.py:14-27, 50-70` · **implement_now: true**
- **증거**: `scores[node.name]`(25), `excluded = [name for name,_ in sorted_nodes[:n]]`을 `nodes_to_exclude`로 전달(50-70). node.name이 ''인 경우 다수 → excluded=['',...]가 아무것도 매칭 못함, sensitive 레이어가 그대로 양자화.
- **영향**: mixed-precision이 풀 INT8과 동일 동작('<1% 정확도 손실' 약속 위반).
- **제안 수정**: `node.name or node.output[0]` 키 사용. 양자화 전 빈 이름 제거 + 무명 노드에 결정적 이름(`f'{op_type}_{i}'`) 부여 후 양자화. 실제 제외 이름 반환.
- **fix_files**: `core/optimizers/mixed_precision.py`
- **effort**: S · **회귀위험**: low

#### [TOOLS-03] EP 호환성이 휴리스틱 테이블에 없는 EP를 supported_ratio=1.0(100%)로 허위 보고
- **P1 · correctness** · `core/model_inspector.py:121-145, 167-205` · **implement_now: true**
- **증거**: `_get_ep_supported_ops(ep)`가 테이블에 없는 EP에 None 반환, 이때 `{supported_ops:[], supported_ratio:1.0}`(140) — 0개 op 나열하며 100% 호환 주장. except 분기도 1.0(142).
- **영향**: CoreML/MIGraphX/ROCm 등에서 '100% 호환' 표시되나 실제 CPU fallback → GPU/NPU 가속 오판.
- **제안 수정**: None 반환 시 ratio=None + `{known:false}` 마크, UI는 'unknown' 렌더. 가능하면 단일 EP+CPU fallback 세션으로 실제 node placement 도출.
- **fix_files**: `core/model_inspector.py`, `web/js/tabs-extra.js`
- **effort**: M · **회귀위험**: low

#### [TOOLS-04] static-quant calibration reader가 letterbox 대신 cv2.resize(stretch) — 실추론 전처리 불일치, INT8 정확도 저하
- **P1 · correctness** · `core/quantizer.py:60-78` · **implement_now: true**
- **증거**: `_AutoCalibrationReader.get_next()`가 `cv2.resize(img,(w,h))`(67) 비종횡비 stretch. 프로젝트 검출 전처리는 letterbox(padded resize). calibration 입력 분포(stretched)가 배포 입력(letterboxed)과 달라 INT8 범위 오류. MixedPrecisionOptimizer도 동일 reader 재사용.
- **영향**: 모든 static INT8/mixed-precision이 불일치 전처리로 calibration → 불필요 정확도 하락(침묵).
- **제안 수정**: resize 블록을 letterbox로 교체. 비검출(classifier/embedder)용 플래그/파라미터로 plain resize fallback 유지, 정사각 검출 입력 기본 letterbox.
- **fix_files**: `core/quantizer.py`
- **effort**: S · **회귀위험**: medium

#### [TOOLS-05] quantize/optimize/diagnose 라우트가 path_safety 우회 (safe_model_file/safe_image_dir 없음)
- **P1 · security** · `server/extra_routes.py:341-382; server/optimization_routes.py:46, 112, 174` · **implement_now: true**
- **증거**: `/api/quantize`(345)가 `os.path.isfile`만, safe_model_file 미호출; calibration_dir도 isdir만. output_path 그대로 write. optimize/diagnose/apply-recommendation도 isfile만. inspector/profiler는 safe_model_file 호출 → 비일관.
- **영향**: 4개 TOOLS 엔드포인트에서 임의 파일 write(output_path) + 임의 모델 read.
- **제안 수정**: `path = safe_model_file(req.model_path)`로 교체, calibration_dir은 safe_image_dir, output_path는 `safe_path(out, allowed_exts={'.onnx'}, must_be_file=False)`. route_errors 데코레이터 패턴 사용.
- **fix_files**: `server/extra_routes.py`, `server/optimization_routes.py`
- **effort**: S · **회귀위험**: low

#### [TOOLS-06] opt_state/diag_state가 무락 plain dict, `dict(state)`가 백그라운드 writer와 경쟁 + 'Already running' 가드 비원자
- **P1 · concurrency** · `server/optimization_routes.py:12-16, 42-98, 108-163` · **implement_now: true**
- **증거**: opt_state/diag_state가 plain dict(12-13), status가 `dict(opt_state)`(98)/`dict(diag_state)`(163)을 락 없이 반환하는 동안 워커가 update(). dict() 순회 중 변경 → RuntimeError 또는 torn state. 가드(44)는 비원자 check-then-set.
- **영향**: status 폴링(500ms)이 크래시/비일관, 동시 run이 동일 output_path 중간 write 충돌 → 손상 .onnx.
- **제안 수정**: 두 dict를 TaskState로 교체, status는 `.snapshot()`. 원자 `try_start()` 헬퍼 추가.
- **fix_files**: `server/optimization_routes.py`, `server/state.py`
- **effort**: S · **회귀위험**: low

#### [TOOLS-07] profiler가 CPU 전용 + 기본 num_runs=20/warmup=3 + 배포 EP 매핑 불가 latency
- **P1 · performance** · `core/model_profiler.py:364-384, 93-98` · **implement_now: true**
- **증거**: profile_model()/_get_layer_profiles() 모두 `providers=['CPUExecutionProvider']` 하드코딩(365,98). 라우트 기본 num_runs=20, warmup=3. UI는 'FPS'/P95/P99/'Est INT8 speedup' 표시.
- **영향**: GPU 배포에 무의미한 CPU latency, 3 warmup/20 run으로 불안정 percentile.
- **제안 수정**: (1) provider/ep 파라미터(앱 auto-EP 기본) 수용+UI 노출. (2) warmup>=10, num_runs>=50. (3) UI에 provider 라벨+percentile 노트.
- **fix_files**: `core/model_profiler.py`, `server/extra_routes.py`, `web/js/tabs-extra.js`
- **effort**: M · **회귀위험**: medium

#### [TOOLS-08] FLOP 추정이 stride/dilation/padding 무시, 4개 op만 커버 → Conv 과대, 나머지 0
- **P2 · correctness** · `core/model_profiler.py:221-261` · **implement_now: false**
- **제안 수정**: stride/dilation 읽기, fallback 56×56 시 estimated_flops를 approximate/None 마크. MatMul 동적 처리. MACs를 실제 곱셈누적으로, FLOPs=2*MACs.
- **fix_files**: `core/model_profiler.py`, `web/js/tabs-extra.js`
- **effort**: M · **회귀위험**: low

#### [TOOLS-09] get_or_compute 모델-메타데이터 캐시 무락 — 동시 inspect가 이중계산 + 반쓰기 JSON 읽기
- **P2 · concurrency** · `core/model_cache.py:46-63` · **implement_now: true**
- **증거**: 파일 존재 검사 후 json.load(51-56), miss 시 compute()+json.dump 동일 경로(57-62), 파일락/atomic rename 없음. 동시 inspect 2건이 둘 다 miss·compute, 하나가 dump 중 다른 하나가 load → JSONDecodeError/truncated.
- **영향**: 중복 inspect CPU/GPU 낭비, EP 테스트 메모리 spike 2배, 캐시 thrash.
- **제안 수정**: 동일 dir temp 파일 write 후 os.replace(atomic). _key별 in-process lock(dict of Lock, 모듈락 가드)로 동일 inspect 1회 계산.
- **fix_files**: `core/model_cache.py`
- **effort**: S · **회귀위험**: low

#### [TOOLS-10] EP 호환성 테스트가 가용 EP마다 풀 InferenceSession 직렬 생성 — 메모리 spike + 느린 inspect
- **P2 · performance** · `core/model_inspector.py:115-145` · **implement_now: false**
- **제안 수정**: accelerator EP는 휴리스틱 테이블로 저비용 확인, 풀 세션 테스트는 'EP 호환성 테스트' 토글 뒤로 lazy. GPU EP 테스트는 task_locks['gpu_infer'] 래핑.
- **fix_files**: `core/model_inspector.py`
- **effort**: M · **회귀위험**: medium

#### [TOOLS-11] 아키텍처 감지가 op-set overlap만(카운트 미가중) → 오분류, 잘못된 추천
- **P2 · correctness** · `core/model_diagnosis.py:7-12, 103-119` · **implement_now: false**
- **제안 수정**: op 카운트 가중 + 판별 op 요구(transformer는 실제 Attention 노드 요구). 1·2위 score margin 요구.
- **fix_files**: `core/model_diagnosis.py`
- **effort**: S · **회귀위험**: low

#### [TOOLS-12] health score 가중 임의 + 0 clamp — warning 몇 개면 무정보
- **P2 · ux** · `core/model_diagnosis.py:81-84, 251-326` · **implement_now: false**
- **제안 수정**: 반복 카테고리 기여 cap(weight-outlier warning을 단일 집계 finding으로). 최적화 여력 기반 연속 스케일 재가중.
- **fix_files**: `core/model_diagnosis.py`
- **effort**: S · **회귀위험**: low

#### [TOOLS-13] diagnosis findings/recommendations 영어 하드코딩 — i18n 우회 (한국어 UI에 영문)
- **P2 · i18n** · `core/model_diagnosis.py:251-326, 332-458` · **implement_now: false**
- **제안 수정**: prose message/reason을 stable `code`+interpolation `params`로 교체, ko+en 템플릿(diag.finding.*/diag.rec.*) 추가, tabs-extra.js가 `t(code, params)` 렌더.
- **fix_files**: `core/model_diagnosis.py`, `web/js/i18n.js`, `web/js/tabs-extra.js`
- **effort**: M · **회귀위험**: low

---

### 2.5 Data (explorer · splitter · converter · remapper · merger · sampler · augmentation) (track: data, extra, fe-main, fe-extra)

#### [DATA-01] Converter/Remapper가 매 run 크래시: `glob_module` 미import
- **P0 · correctness** · `server/data_routes.py:296, 340, 445, 447` · **implement_now: true**
- (병합: DATA-01 + CC-01. grep으로 296/340/445/447 호출 + import 부재 확인.)
- **증거**: 296/340(converter), 445/447(remapper)가 `glob_module.glob(...)` 호출하나 import는 `csv,io,os,random,shutil`(2-6)+`glob_images`(16)만. server/utils.py가 `import glob as glob_module`하나 re-export 안 함. 두 워커가 NameError → state.msg에 'name glob_module is not defined' 노출.
- **영향**: Converter(YOLO↔COCO↔VOC)·Remapper 두 탭 100% 불능.
- **제안 수정**: data_routes.py 상단(line 6 이후)에 `import glob as glob_module` 추가.
- **fix_files**: `server/data_routes.py`
- **effort**: S · **회귀위험**: low

#### [DATA-02] TaskState singleton을 plain-dict로 5회 재선언 — force-stop·태스크큐가 orphan 객체 조작
- **P0 · concurrency** · `server/data_routes.py:15, 175, 284, 434, 492, 564` · **implement_now: true**
- (병합: DATA-02 + CC-03 + WIRE-03.)
- **증거**: line 15가 TaskState 6개 import하나 175/284/434/492/564가 splitter/converter/remapper/merger/sampler_state를 plain dict로 rebind(grep 확인). all_states는 원본 TaskState 가리킴 → `/api/force-stop/<task>`·`/api/tasks`가 워커 미독 객체 조작. explorer_state는 미재선언(영향 없음).
- **영향**: 5개 data 탭 force-stop 침묵 no-op, 태스크큐 패널/`/api/tasks`가 running/progress 미표시, 워커-poller race(무락).
- **제안 수정**: 5개 재선언(175,284,434,492,564) 삭제하여 import된 TaskState 사용. status 엔드포인트가 이미 `hasattr(...,'snapshot')` 분기하므로 자동 정상화.
- **fix_files**: `server/data_routes.py`
- **effort**: S · **회귀위험**: low

#### [DATA-03] 모든 data_routes 엔드포인트 path_safety 미검증 (traversal/임의 write)
- **P1 · security** · `server/data_routes.py:22-87, 129-162, 177-270, 286-419, 436-478, 494-546, 582-671` · **implement_now: true**
- (병합: DATA-03 + CC-04 의 data 부분.)
- **증거**: grep으로 safe_image_dir/safe_label_dir/safe_path/path_safety 부재 확인. explorer(27)·converter(295 makedirs+335/365/407 write)·merger(505/529/536 copy2)·sampler(648/654)·remapper(444/469)·preview(132-136 open) 모두 raw. augmentation(extra_routes.py:295)은 safe_image_dir 호출 → 비일관.
- **영향**: 악성/오류 경로로 임의 파일 read(explorer/preview) + 임의 위치 write(converter/remapper/merger/sampler output_dir).
- **제안 수정**: import 추가, 입력 dir은 safe_image_dir/safe_label_dir(must_exist), output_dir은 `safe_path(out, must_be_dir=False)`(생성 허용, traversal/NUL 거부), preview는 safe_image_file. UnsafePathError → `{'error': code}`.
- **fix_files**: `server/data_routes.py`
- **effort**: M · **회귀위험**: medium

#### [DATA-04] remapper auto_reindex 파라미터 UI 수집되나 미구현
- **P1 · feature** · `server/data_routes.py:431, 442-475` · **implement_now: true**
- **증거**: `auto_reindex: bool = True`(431) 선언, UI 전송(tabs-extra.js:827)하나 grep으로 데이터루트 내 사용 부재. _run(442-475)은 명시 mapping만 적용·미매핑 클래스 drop, contiguous 0..N-1 압축 안 함. 기본 UI 상태(reindex 체크, mapping 없음)에서 출력=입력 동일.
- **영향**: 사용자가 순차 class id 기대하나 그대로 유지, downstream `nc=` 오류.
- **제안 수정**: `if req.auto_reindex and not req.mapping:` 시 present id 스캔 → `{old: new for new, old in enumerate(sorted(present))}` 적용. 둘 다 설정 시 명시 매핑 후 reindex. reindex map을 results에 노출.
- **fix_files**: `server/data_routes.py`, `web/js/tabs-extra.js`
- **effort**: M · **회귀위험**: low

#### [DATA-05] Converter가 YOLO-seg 폴리곤 라벨을 garbage box로 손상 (항상 parts[1:5])
- **P1 · correctness** · `server/data_routes.py:322-329, 395-404` · **implement_now: true**
- **증거**: YOLO→COCO(322)·YOLO→VOC(395-397) 모두 `len(parts)>=5`면 5필드 bbox 취급, `parts[1:5]`. 폴리곤 라인은 많은 토큰 → parts[1:5]=첫 2 vertex를 cx,cy,bw,bh로 오해석.
- **영향**: seg 데이터셋 변환이 nonsense bbox 무경고 생성.
- **제안 수정**: `len(parts)>5 and (len(parts)-1)%2==0`이면 폴리곤 — vertex min/max로 enclosing bbox(COCO는 segmentation 필드). 정확히 5 토큰만 plain bbox.
- **fix_files**: `server/data_routes.py`
- **effort**: M · **회귀위험**: low

#### [DATA-06] Converter가 이미지 없을 때 640×640 침묵 fallback → 잘못된 COCO/VOC 좌표
- **P1 · correctness** · `server/data_routes.py:313-318, 381-389, 353-359` · **implement_now: true**
- **증거**: YOLO→COCO(313) `w,h=640,640` 매칭 이미지 없거나 imread 실패 시 fallback. 정규화 YOLO 좌표를 잘못된 640으로 변환 → 잘못된 절대 박스. 4개 확장자만 probe.
- **영향**: labels/images 분리 디렉토리(흔함) 변환이 640 스케일 박스 침묵 생성, .bmp/.webp/.tif skip.
- **제안 수정**: 차원 미정 시 skip+`results['skipped_no_size']` 카운트 OR 명시 image_dir 필드. 확장자 probe를 path_safety의 _IMAGE_EXTS로 확대, `results['missing_image']` 노출.
- **fix_files**: `server/data_routes.py`, `web/js/tabs-extra.js`
- **effort**: M · **회귀위험**: low

#### [DATA-07] augmentation이 라벨 무시: 박스 미변환 + label dir 미전송
- **P1 · correctness** · `server/extra_routes.py:287-324` · **implement_now: true**
- **증거**: BatchAugmentRequest(287-289)에 label_dir 없음, UI는 `lblDirInput('aug-lbl')`(1222) 렌더하나 run()(1238-1243)은 `{img_dir, aug_type}`만 전송. 백엔드(293-324)는 이미지만 flip/rotate/brighten/mosaic, 박스 미변환. 파라미터 하드코딩(15°, alpha=1.3, beta=30).
- **영향**: label-aware 암시하나 preview-only 이미지 변환, Flip/Rotate 결과가 유효 라벨 미대응.
- **제안 수정**: 최소(정직성): `lblDirInput('aug-lbl')` 제거, preview-only 명시(i18n 노트). 또는 실제 augmenter — label_dir 추가, 박스 좌표 변환, angle/alpha/beta 폼 필드. 우선 최소 수정 ship.
- **fix_files**: `server/extra_routes.py`, `web/js/tabs-extra.js`, `web/js/i18n.js`
- **effort**: M · **회귀위험**: low

#### [DATA-08] merger dedup O(N²) Hamming brute force + label-dir 휴리스틱이 rename 파일 mismatch
- **P2 · performance** · `server/data_routes.py:511-537` · **implement_now: false**
- **제안 수정**: BK-tree/VP-tree(core/hashing.py)로 dedup. label은 `results['labels_copied']`/`labels_missing'` 카운트.
- **fix_files**: `server/data_routes.py`, `core/hashing.py`, `web/js/tabs-extra.js`
- **effort**: L · **회귀위험**: medium

#### [DATA-09] balanced/stratified sampler가 target_count overshoot/undershoot + balanced가 box center FPS만
- **P2 · correctness** · `server/data_routes.py:617, 623-643` · **implement_now: false**
- **제안 수정**: selected 구성 후 target_count로 trim/top-up. total_assoc 명명 정정. box-less feature caveat.
- **fix_files**: `server/data_routes.py`
- **effort**: M · **회귀위험**: low

#### [DATA-10] stratified split이 희소 클래스 min-per-class 보장 없음 + ratio 합≠1 침묵 재정규화
- **P2 · correctness** · `server/data_routes.py:200-243` · **implement_now: false**
- **제안 수정**: per-class iterative stratification 또는 ≥2 이미지 그룹은 val/test에 최소 1. 정규화 ratio를 results에 echo. 'similarity' dead enum 제거 또는 구현.
- **fix_files**: `server/data_routes.py`, `web/js/tabs.js`
- **effort**: M · **회귀위험**: medium

#### [DATA-11] 긴 data 작업 취소 불능 (running 플래그만, 워커 루프 미검사)
- **P2 · ux** · `server/data_routes.py:33-86, 183-269, 514-539, 653-663` · **implement_now: false**
- **제안 수정**: DATA-02 수정 후 각 루프에 `if not <state>['running']: <state>.update(msg='Cancelled'); return`. 각 탭 Stop 버튼.
- **fix_files**: `server/data_routes.py`, `web/js/tabs.js`, `web/js/tabs-extra.js`
- **effort**: M · **회귀위험**: low

#### [DATA-12] explorer가 모든 이미지를 단일 in-memory 페이로드로 스캔 (가상 스크롤/페이지네이션 없음)
- **P2 · performance** · `server/data_routes.py:26, 44-81` · **implement_now: false**
- **제안 수정**: list 페이로드에서 box_details 제거(preview에서 lazy fetch), 갤러리 가상화/페이지네이션, 'Limit' 라벨을 'Scan limit'로.
- **fix_files**: `server/data_routes.py`, `web/js/tabs.js`, `web/js/i18n.js`
- **effort**: M · **회귀위험**: low

#### [DATA-13] 손상/누락 라벨·파싱 불가 라인 침묵 skip, 카운트 미노출
- **P2 · ux** · `server/data_routes.py:50-64, 207-212, 319-331, 457-468, 608-614` · **implement_now: false**
- **제안 수정**: per-line int/float를 try/except로 감싸 `bad_lines` 카운트. `missing_labels`/`corrupt_files` 추적·results 노출. explorer parse를 가드해 단일 bad 라인이 전체 스캔 중단 못하게.
- **fix_files**: `server/data_routes.py`, `web/js/tabs.js`
- **effort**: M · **회귀위험**: low

> **STALE** — DATA-14: augmentation 'empty-dir crash' + 'missing path validation' 미재현 (extra_routes.py:295가 safe_image_dir 먼저 호출, 297이 empty 체크 후 random.choice). FEATURE_AUDIT augmentation [3]에서 해당 항목 삭제 권장. 실제 남은 이슈는 DATA-07.

---

### 2.6 Quality (anomaly · quality · duplicate · leaky · similarity) (track: quality, fe-extra)

#### [QUAL-01] similarity query 침묵 drop: 프론트가 `query` 전송, 백엔드는 `query_path` 읽음
- **P0 · contract** · `server/quality_routes.py:34-39, 258-290; web/js/tabs-extra.js:1192` · **implement_now: true**
- **증거**: SimRequest `query_path: Optional[str]=""`(36) alias 없음, 라우트 `query = req.query_path or ""`(263). 프론트는 `query:` 키 POST(1192). Pydantic이 unknown `query` 무시 → query_path 항상 "". `if query and os.path.isfile(query)`(280) 미실행, line 286 첫 top_k를 distance 0으로 반환.
- **영향**: similarity search가 쿼리 대비 랭킹 영원히 안 함, 무의미한 첫 top_k 반환(오류 표시 없음).
- **제안 수정**: tabs-extra.js:1192의 `query:`를 `query_path:`로 변경. (또는 SimRequest에 `Field("", alias="query")`+populate_by_name.) QUAL-08과 함께 수정.
- **fix_files**: `web/js/tabs-extra.js`
- **effort**: S · **회귀위험**: low

#### [QUAL-02] dHash size 불일치: duplicate/leaky=8(64-bit), similarity=16(256-bit)
- **P1 · correctness** · `server/quality_routes.py:157-158, 180, 227, 278` · **implement_now: true**
- **증거**: `_dhash`가 duplicate(180)·leaky(227)는 size=8 기본 → 64-bit, similarity는 `_dhash(frame,16)`(278) → 256-bit. UI threshold input max=64 → similarity 256-bit 공간에서 1/4 민감도. core/hashing.py가 'single source of truth' 주장하나 size 인자가 분기 재도입.
- **영향**: 동일 이미지가 탭마다 다른 hash 폭, threshold 비일관.
- **제안 수정**: 프로젝트 전역 단일 폭. 최소: similarity의 `,16` 제거(278→`_dhash(frame)`)로 전부 64-bit. core/hashing.py에 폭 문서화.
- **fix_files**: `server/quality_routes.py`, `web/js/tabs-extra.js`
- **effort**: S · **회귀위험**: low

#### [QUAL-03] leaky가 이미지를 2회 read(filter imread + _dhash imread) + 진행률 미갱신
- **P1 · performance** · `server/quality_routes.py:221-246` · **implement_now: true**
- **증거**: line 227 `{... _dhash(imread(fp)) for fp in imgs if imread(fp) is not None}` — imread 2회 디코드. `total`이 hashing 후에만 설정(228), `progress` 미할당 → 프론트 바 0% 고착.
- **영향**: 2배 디스크 I/O+디코드, 전체 hashing 동안 0% 고착(hang과 구분 불가).
- **제안 수정**: 명시 루프로 파일당 1회 디코드, 루프 전 `total` 설정, 루프 내 `progress += 1`.
- **fix_files**: `server/quality_routes.py`
- **effort**: S · **회귀위험**: low

#### [QUAL-04] sim_state['index'] 선언되나 미사용 — 인덱스 캐싱 없음, 매 쿼리 전체 rehash
- **P2 · performance** · `server/quality_routes.py:258-290` · **implement_now: true**
- **증거**: state.py:94 `sim_state = TaskState(index=None)`. quality_routes가 index 미read/write(grep으로 'Building index...' 리터럴만). 매 POST가 dir 전체 re-glob+rehash(269-279).
- **영향**: 다른 쿼리로 재실행(흔한 워크플로) 시 dir 전체 rehash, 10k 이미지면 쿼리당 10k 재디코드.
- **제안 수정**: (resolved img_dir, recursive, size) 키로 hash list를 `sim_state['index']` 캐싱. 키 일치 시 재사용, dir 변경 시 무효화.
- **fix_files**: `server/quality_routes.py`
- **effort**: M · **회귀위험**: low

#### [QUAL-05] duplicate가 O(N²) inline 재구현 + `group` 카운터가 클러스터링 안 함(쌍당 1 group)
- **P2 · correctness** · `server/quality_routes.py:183-196` · **implement_now: true**
- **증거**: 중첩 루프(185-194)가 find_near_duplicates 대신 inline Hamming 재계산. `group`이 1부터 매 매칭 쌍마다 증가(190) → 행 인덱스일 뿐, A↔B↔C 클러스터 안 함. 500 cap(191-194) 침묵 truncate.
- **영향**: 'Group' 열 무의미, 상호 유사 클러스터 비가시(near-dup 헤드라인 기능).
- **제안 수정**: find_near_duplicates(hashes, threshold) 위 union-find로 연결 컴포넌트 group id. cap 도달 시 msg에 truncation 표기. 장기: BK/VP-tree를 core/hashing.py에.
- **fix_files**: `server/quality_routes.py`, `core/hashing.py`
- **effort**: M · **회귀위험**: medium

#### [QUAL-06] DOM-XSS: 파일명이 5개 탭 전부 innerHTML에 unescaped 삽입
- **P1 · security** · `web/js/tabs-extra.js:1018, 1059-1060, 1102, 1162, 1205-1206` · **implement_now: true**
- **증거**: 5개 결과 렌더러가 서버 값(os.path.basename 유래)을 template literal로 innerHTML에 escape 없이 보간. `<img src=x onerror=alert(1)>.jpg` 파일명이 스크립트 실행. `_esc` 헬퍼(notify.js:123) 미사용.
- **영향**: 조작된 파일명이 앱 컨텍스트에서 임의 JS 실행, 서드파티 데이터셋 흔함.
- **제안 수정**: 공유 `escHtml(s)` 유틸(notify.js _esc 패턴) 추가, 5개 렌더러의 모든 보간 서버 문자열 래핑. 숫자는 안전.
- **fix_files**: `web/js/tabs-extra.js`
- **effort**: S · **회귀위험**: low

#### [QUAL-07] anomaly label_dir path_safety 우회 — 미검증 경로를 open()에 사용
- **P2 · security** · `server/quality_routes.py:45-92` · **implement_now: true**
- **증거**: img_dir은 검증(49)되나 `label_dir = req.label_dir`(50) raw → `os.path.join(label_dir, stem+".txt")`(64) open(66). leaky는 검증(214-216).
- **영향**: 제약된 임의 .txt read(image_stem.txt만), path-safety 경계 위반.
- **제안 수정**: line 50 후 `label_dir = safe_label_dir(req.label_dir) if req.label_dir else ""`.
- **fix_files**: `server/quality_routes.py`
- **effort**: S · **회귀위험**: low

#### [QUAL-08] similarity query_path path_safety 우회 — raw os.path.isfile+imread
- **P2 · security** · `server/quality_routes.py:262-282` · **implement_now: true**
- **증거**: query를 raw 사용(263), `os.path.isfile(query)`(280)+`imread(query)`(281), safe_image_file 없음. QUAL-01 수정이 이 경로를 노출하므로 함께 수정 필수.
- **제안 수정**: `query = safe_image_file(req.query_path) if req.query_path else ""`. safe_image_file이 must_exist+image ext 강제하므로 isfile 재검사 제거.
- **fix_files**: `server/quality_routes.py`
- **effort**: S · **회귀위험**: low

#### [QUAL-09] 침묵 결과 truncation + 오도하는 카운트 (results[:1000]이나 msg는 full length)
- **P2 · ux** · `server/quality_routes.py:88-89, 140-141, 191-196` · **implement_now: true**
- **증거**: anomaly가 `results[:1000]`(88) 저장하나 msg는 `{len(results)}`(89). quality 동일. duplicate 500 cap 표기 없음. 페이지네이션 없음.
- **영향**: 사용자가 전체로 오인, cap 초과 항목 침묵 손실, 카운트가 가시 행과 모순.
- **제안 수정**: `truncated = len(results)>CAP` 저장, msg를 '... (showing first 1000)'. CAP 상수화, 선택적 client 페이지네이션.
- **fix_files**: `server/quality_routes.py`, `web/js/tabs-extra.js`
- **effort**: M · **회귀위험**: low

#### [QUAL-10] 손상/읽기불가 이미지 침묵 skip, skip 카운트 미보고, total은 포함
- **P2 · ux** · `server/quality_routes.py:118-121, 178-181, 276-279, 282` · **implement_now: true**
- **증거**: quality `if frame is None: progress++; continue`(118-121) 침묵. duplicate/similarity 동일. similarity는 query 실패 시 `q_hash=0`(282) fallback → 모든 이미지를 distance-to-zero로 랭킹(garbage).
- **영향**: 손상 파일 데이터셋이 무경고 불완전 결과, similarity q_hash=0이 유효해 보이는 무의미 랭킹.
- **제안 수정**: 워커별 `skipped` 카운트, msg에 추가. similarity 쿼리 디코드 실패 시 에러 반환(q_hash=0 fallback 제거).
- **fix_files**: `server/quality_routes.py`
- **effort**: S · **회귀위험**: low

#### [QUAL-11] entropy 계산(히스토그램 2회)되나 이슈 감지 미사용
- **P2 · dead-code** · `server/quality_routes.py:125-136` · **implement_now: false**
- **제안 수정**: 히스토그램 1회 계산, `if entropy < 3.0: issues.append('Low detail')`. ko/en i18n 키.
- **fix_files**: `server/quality_routes.py`, `web/js/i18n.js`
- **effort**: S · **회귀위험**: low

#### [QUAL-12] 하드코딩 절대 임계값(blur<50, brightness 40/220, aspect 4/0.25) 도메인 취약
- **P2 · correctness** · `server/quality_routes.py:129-136, 73-82` · **implement_now: false**
- **제안 수정**: blur 임계를 기존 `req.threshold`로 노출, brightness/aspect knob 추가. 미사용 threshold 필드 배선. 기본값 유지.
- **fix_files**: `server/quality_routes.py`, `web/js/tabs-extra.js`, `web/js/i18n.js`
- **effort**: M · **회귀위험**: low

---

### 2.7 Specialized (segmentation · clip · embedder · pose · instance-seg · tracking) (track: extra, eval, fe-extra, fe-infra)

#### [SPEC-01] 7개 specialized 탭 + vlm `_nav` 미등록 — 발견 불가
- **P0 · feature** · `web/js/app.js:6-13` · **implement_now: true**
- (병합: SPEC-01 + VLM-04 + WIRE-02.)
- **증거**: App._nav(6-13)에 6 섹션 25 탭만. segmentation/clip/embedder/pose/instance-seg/tracking/vlm 부재. switchTab/renderSidebar/Ctrl+1..9가 _nav만 순회. 탭은 tabs-extra.js에 정의되고 i18n nav 키(i18n.js:19-43, 541-560) 존재 → 등록만이 gap.
- **영향**: 7개 완성 기능(seg eval, CLIP, embedder, pose, instance-seg, tracking, VLM) UI에서 도달 불가, README가 광고.
- **제안 수정**: _nav에 새 섹션 추가, 예: `['sec.specialized', [['segmentation','segmentation'],['clip','clip'],['embedder','embedder'],['pose','pose'],['instance-seg','instance-seg'],['tracking','tracking'],['vlm','vlm']]]`. `sec.specialized` i18n 키(ko/en). VLM은 VLM-01/02 수정과 조율.
- **fix_files**: `web/js/app.js`, `web/js/i18n.js`, `web/js/icons.js`
- **effort**: S · **회귀위험**: low

#### [SPEC-02] segmentation eval 라우트가 evaluate_segmentation을 잘못된 시그니처로 호출 (TypeError 확정)
- **P0 · correctness** · `server/extra_routes.py:602-608` · **implement_now: true**
- **증거**: extra_routes.py:602-607이 `evaluate_segmentation(mi, images, gt_mask_dir, num_classes, conf, progress_cb=, stop_flag=)` 호출하나 core/evaluation.py:199 정의는 `(pred_mask, gt_mask, num_classes)`. per-image 루프·모델 실행·progress_cb 없음 → TypeError. 올바른 구현은 eval_routes.py:348-429 `_run_segmentation_eval`에 존재.
- **영향**: 독립 Segmentation 탭이 결과 생성 불가, 매 run 에러.
- **제안 수정**: extra_routes.py seg_state 워커를 _run_segmentation_eval 미러로 재작성(per-image imread, GT mask 로드, run_segmentation, INTER_NEAREST resize, per-image evaluate_segmentation(pm,gm,nc), per-class IoU/Dice 평균, stop_flag/progress_cb). 공유 루프를 core/ 헬퍼로 추출 권장.
- **fix_files**: `server/extra_routes.py`, `core/evaluation.py`
- **effort**: M · **회귀위험**: medium

#### [SPEC-03] segmentation 프론트가 label_dir 전송하나 요청 모델은 gt_mask_dir 필수 (로직 전 422)
- **P0 · contract** · `web/js/tabs-extra.js:448; server/extra_routes.py:575-580` · **implement_now: true**
- **증거**: tabs-extra.js:448이 `{model_path, img_dir, label_dir}` POST. SegmentationRunRequest(575-580)가 `gt_mask_dir: str` 기본값/alias 없음 → 422 missing-field, label_dir은 unexpected.
- **영향**: SPEC-02와 별개로 HTTP 경계에서 실패, 사용자 선택 GT dir drop.
- **제안 수정**: 필드명 정렬 — 프론트가 `gt_mask_dir: label_dir` 전송 또는 백엔드가 `req.gt_mask_dir or req.label_dir` 허용. num_classes/conf도 UI 노출 시 전달.
- **fix_files**: `web/js/tabs-extra.js`, `server/extra_routes.py`
- **effort**: S · **회귀위험**: low

#### [SPEC-04] Pose/Instance-Seg 탭이 디렉토리의 첫 이미지만 처리
- **P1 · feature** · `web/js/tabs-extra.js:1286-1292, 1342-1348` · **implement_now: true**
- **증거**: pose run()(1286-1287)이 `imgs.files[0]`만 단일 이미지로 `/api/infer/pose` POST. instance-seg 동일(1342-1343). 백엔드는 단일 이미지 설계(InferImageRequest). 탭은 이미지-디렉토리 input(imgDirInput)으로 batch 암시.
- **영향**: 폴더 가리켜도 알파벳 첫 이미지만 추론·표시, 오해 유발.
- **제안 수정**: (a) single-image picker로 변경(pickFile/_showFileBrowser) — S, 정직. 또는 (b) 결과 갤러리: 디렉토리 유지, 전체 파일 순차 실행+진행바+썸네일. (a) 우선 + next/prev 컨트롤.
- **fix_files**: `web/js/tabs-extra.js`
- **effort**: S · **회귀위험**: low

#### [SPEC-05] Tracking 탭 비기능(update/infer 엔드포인트 없음) + _trackers 무한 증가
- **P1 · feature** · `server/extra_routes.py:501-526` · **implement_now: true**
- **증거**: `/api/tracking/create`(501-518)·`/api/tracking/reset`(521-526)만, detection/frame을 tracker에 먹이는 엔드포인트 없음. core/tracking.py update()가 어느 라우트에서도 호출 안 됨. 탭(1359-1400)은 Create/Reset 버튼만. `_trackers`(502) 모듈 전역 dict가 create마다 uuid 추가, 삭제 안 됨(무락).
- **영향**: Tracking 탭 가치 zero, _trackers 메모리 누수, stub로 ship.
- **제안 수정**: 제거(YAGNI, 가장 깨끗) 또는 완성 — `/api/tracking/update {tracker_id, boxes/scores/class_ids OR model+image}`가 detection→tracker.update→track id/box/trajectory 반환, UI 비디오/시퀀스 driver. 어느 쪽이든 _trackers를 LRU/TTL bound + Lock. 공격적 릴리스 고려 시 제거 우선.
- **fix_files**: `server/extra_routes.py`, `core/tracking.py`, `web/js/tabs-extra.js`, `web/js/app.js`, `web/js/i18n.js`
- **effort**: M · **회귀위험**: low

#### [SPEC-06] CLIP/Embedder `/run` 라우트가 path_safety 우회 (raw path를 InferenceSession/os.listdir로)
- **P1 · security** · `server/extra_routes.py:33-104, 120-240` · **implement_now: true**
- **증거**: run_clip(33-104)이 req.image_encoder/text_encoder를 CLIPModel→InferenceSession에, req.img_dir을 glob_images에 raw 전달. run_embedder(120-240)가 req.model_path를 _load_model에, req.img_dir을 os.listdir에 raw. 반면 seg/pose/instance-seg/embedder_compare는 safe_* 호출 → 비일관.
- **영향**: traversal 경로가 정규화 없이 FS/ORT 도달, 비존재 dir이 혼란스러운 downstream 에러.
- **제안 수정**: 워커 submit 전 동기 검증 — run_clip: safe_model_file(image_encoder/text_encoder)+safe_image_dir(img_dir); run_embedder: safe_model_file(model_path)+safe_image_dir(img_dir). local 사용.
- **fix_files**: `server/extra_routes.py`
- **effort**: S · **회귀위험**: low

#### [SPEC-07] CLIP zero-shot에 프롬프트 템플릿('a photo of {}') 없음 — 정확도 저하
- **P2 · correctness** · `server/extra_routes.py:47-50` · **implement_now: true**
- **증거**: run_clip(48-50)이 bare 라벨 토큰화. OpenAI CLIP은 'a photo of a {label}.' 프롬프트 calibration, temperature*100 softmax도 템플릿 가정.
- **영향**: 보고 zero-shot 정확도가 체계적 저하/노이즈.
- **제안 수정**: 토큰화 전 `prompt = f"a photo of a {label}."`. 선택적 프롬프트-템플릿 input. 최소: run_clip 하드코딩.
- **fix_files**: `server/extra_routes.py`, `web/js/tabs-extra.js`
- **effort**: S · **회귀위험**: low

#### [SPEC-08] segmentation GT mask lookup이 확장자 엄격(.png만), 침묵 skip
- **P2 · correctness** · `server/eval_routes.py:382-388` · **implement_now: true**
- **증거**: eval_routes.py:384가 `stem + '.png'`만, 385-386 `if not isfile: done++; continue`. .bmp/.tif/.jpg mask 침묵 skip, skip 카운트 없음.
- **영향**: 비-.png mask 데이터셋이 0 평가, 'Complete'에 zero 메트릭.
- **제안 수정**: stem+(.png,.bmp,.tif,.tiff,.jpg)+원본 ext 후보 검색, 첫 존재 선택. skip 카운트 msg 노출. SPEC-02 통합 루프에 적용.
- **fix_files**: `server/eval_routes.py`, `server/extra_routes.py`
- **effort**: S · **회귀위험**: low

#### [SPEC-09] keypoint 가시성 임계 0.5 하드코딩 (conf input 무관)
- **P2 · ux** · `server/extra_routes.py:445-450` · **implement_now: false**
- **제안 수정**: InferImageRequest에 `kpt_conf: float = 0.5` 추가, 445/448에 사용, pose 탭에 'Keypoint conf' input, 프론트 visible-count(1299) 동기.
- **fix_files**: `server/extra_routes.py`, `web/js/tabs-extra.js`
- **effort**: S · **회귀위험**: low

#### [SPEC-10] embedder self-similarity 제외가 np.array_equal per pair (O(N²·D))
- **P2 · performance** · `server/extra_routes.py:190-204` · **implement_now: false**
- **제안 수정**: embeddings 구성 시 stable global index 추적, sims=all_embs@emb, `sims[self_idx]=-inf`. all_embs@all_embs.T로 O(N²) 벡터화.
- **fix_files**: `server/extra_routes.py`
- **effort**: M · **회귀위험**: low

#### [SPEC-11] CLIP/embedder/seg 손상 이미지 침묵 skip, 분모 inflate
- **P2 · correctness** · `server/extra_routes.py:64-67, 154-157` · **implement_now: false**
- **제안 수정**: clip_state/embedder_state/seg에 `skipped` 카운터, None read 시 증가, 완료 msg에 추가.
- **fix_files**: `server/extra_routes.py`, `web/js/tabs-extra.js`
- **effort**: S · **회귀위험**: low

---

### 2.8 Cross-cutting (state isolation · path safety · cancellation · caps · i18n · errors · logging) (track: shared, fe-infra)

> 주: cross-cutting 섹션의 P0/P1 다수는 위 섹션 finding으로 병합됨 — CC-01→DATA-01, CC-02→VIEWER-01, CC-03→DATA-02, CC-04→ANLY-03/DATA-03/EVAL-04, CC-05→EVAL-04, CC-06→ANLY-04, CC-07→VIEWER-02. 아래는 cross-cutting 고유 잔여 finding.

#### [XCUT-01] 모든 async 라우트의 running-flag start 가드 TOCTOU
- **P2 · concurrency** · `server/state.py:28-47` · **implement_now: false**
- **증거**: 모든 라우트가 `if state['running']: return {'error':...}` 후 `state.update(running=True)`. TaskState가 각 op 원자지만 read+write 2회 분리 op → 동시 2 POST가 둘 다 가드 통과, 두 워커가 동일 results/tmp 경쟁.
- **영향**: 빠른 더블클릭/동시 탭에서 동일 작업 이중 제출, results/tmp 손상(model-compare tmp wipe/recreate).
- **제안 수정**: TaskState에 원자 `try_start(**init) -> bool` 추가(`with self._lock: running 검사+설정`). 각 라우트의 check-then-set를 `if not state.try_start(...): return {'error':...}`로 교체.
- **fix_files**: `server/state.py`, `server/analysis_routes.py`, `server/eval_routes.py`, `server/quality_routes.py`, `server/extra_routes.py`, `server/data_routes.py`, `server/optimization_routes.py`, `server/benchmark_routes.py`
- **effort**: M · **회귀위험**: low

#### [XCUT-02] 결과 cap 하드코딩(1000/500/10/200) 페이지네이션/truncation 알림 없음
- **P2 · ux** · `server/quality_routes.py:88, 140, 191-194, 242, 284; server/extra_routes.py:206, 100` · **implement_now: false**
- (QUAL-09와 부분 중첩 — QUAL-09는 quality 탭 한정 즉시 수정, XCUT-02는 embedder/clip detail cap 포함 전역.)
- **제안 수정**: full count를 state에 저장+'showing N of M' 노출, duplicate는 scan break 제거(카운트만 계속, 반환 list만 truncate), status에 offset/limit. i18n 'results_truncated'.
- **fix_files**: `server/quality_routes.py`, `server/extra_routes.py`, `web/js/tabs-extra.js`, `web/js/i18n.js`
- **effort**: M · **회귀위험**: low

#### [XCUT-03] 백엔드 findings/recommendations + anomaly/quality issue 문자열 영어 전용
- **P2 · i18n** · `core/model_diagnosis.py:251-301; server/quality_routes.py:73-82, 129-135` · **implement_now: false**
- (TOOLS-13의 model_diagnosis 부분과 중첩 + quality 라우트 이슈 문자열 추가.)
- **제안 수정**: 백엔드가 prose message 대신 stable code+params 방출, ko+en 키 추가, 프론트 `t(code, params)` 렌더. anomaly/quality issue type을 머신 코드로.
- **fix_files**: `core/model_diagnosis.py`, `server/quality_routes.py`, `web/js/i18n.js`, `web/js/tabs-extra.js`
- **effort**: M · **회귀위험**: low

#### [XCUT-04] 침묵 corrupt-file skip(카운트 없음) + worker error를 traceback 없이 삼키는 broad except
- **P2 · correctness** · `server/analysis_routes.py:64-66, 89-90; server/eval_routes.py; server/data_routes.py:71` · **implement_now: false**
- **제안 수정**: 워커별 `skipped` 카운터+msg 노출. analysis/data/eval _run의 inline broad except를 `@route_errors(state=, scope=)` 데코레이터로 교체하여 traceback+trace_id 로깅.
- **fix_files**: `server/analysis_routes.py`, `server/eval_routes.py`, `server/data_routes.py`, `server/errors.py`
- **effort**: M · **회귀위험**: low

---

### 2.9 Wiring / contracts (track: fe-extra, fe-infra, vlm)

> WIRE-01→VLM-01, WIRE-02→SPEC-01, WIRE-03→DATA-02, WIRE-04→EVAL-02 로 병합. 아래는 wiring 고유 잔여.

#### [WIRE-01] VLM/specialized 탭 전용 라우트가 미도달 탭으로만 배선 (실질 dead until SPEC-01)
- **P2 · dead-code** · `server/extra_routes.py:33, 120, 251, 428, 465, 513, 584` · **implement_now: false**
- **제안 수정**: SPEC-01에 종속. 7 탭 등록 후 각 엔드포인트 e2e smoke. 의도적 deprecate면 탭+라우트 동시 삭제.
- **fix_files**: `web/js/app.js`, `server/extra_routes.py`
- **effort**: S · **회귀위험**: low

#### [WIRE-02] 프론트 caller 전혀 없는 백엔드 라우트 (eval/save, fs/browse, fs/select-multi, system/full-info, optimize/methods, config/custom-model-types)
- **P2 · dead-code** · `server/system_routes.py:263, 274, 343; server/eval_routes.py:785; server/optimization_routes.py:30; server/config_routes.py:143` · **implement_now: false**
- **제안 수정**: 라우트별 결정 — 삭제(eval/save는 auto-save 중복, fs/browse) 또는 UI gap 배선(optimize/methods로 calibration 드롭다운, fs/select-multi로 멀티파일 picker).
- **fix_files**: `server/eval_routes.py`, `server/system_routes.py`, `server/optimization_routes.py`, `server/config_routes.py`
- **effort**: S · **회귀위험**: low

#### [WIRE-03] api.js dead wrapper (classifyModel/findPartner/classCatalogs/classCatalog/suggestCatalog/selectFile)
- **P2 · dead-code** · `web/js/api.js:77, 82, 83, 88, 89, 90` · **implement_now: false**
- **제안 수정**: VLM/embedder text-encoder auto-pick에 findPartner 배선 또는 6개 dead wrapper+고아 백엔드 라우트 삭제.
- **fix_files**: `web/js/api.js`, `server/model_routes.py`
- **effort**: S · **회귀위험**: low

#### [WIRE-04] dead sim_state['index'] 필드 + tabs-extra _poll 루프의 silent catch
- **P2 · dead-code** · `server/state.py:94; web/js/tabs-extra.js:1166` · **implement_now: false**
- (QUAL-04가 index를 활용하는 방향이면 본 항목 무효 — index를 캐시로 쓰므로 QUAL-04 우선. silent catch만 잔여.)
- **제안 수정**: QUAL-04 미채택 시 `sim_state = TaskState()`. silent catch는 US-008 패턴(App.toast/setStatus) 적용.
- **fix_files**: `server/state.py`, `web/js/tabs-extra.js`
- **effort**: S · **회귀위험**: low

#### [WIRE-05] README drift: 버전 배지 1.5.0 vs 앱 1.5.3 + 도달 불가 7 기능 광고
- **P2 · contract** · `README.md:13, 35-37, 62-63` · **implement_now: true**
- **증거**: README:13 'version-1.5.0' vs `__init__.py:79` version="1.5.3". 'Do You Need' 표(35-37)·Supported Models(62-63)가 도달 불가 탭(SPEC-01) 기능 광고. changelog top이 v1.4.0.
- **영향**: README 따르는 사용자가 CLIP/Embedder/Segmentation/Tracking/VLM 못 찾음, 버전/changelog stale.
- **제안 수정**: 배지 1.5.3, 1.5.x changelog 추가. SPEC-01과 동일 릴리스로 ship하여 기능 주장 정합. 최소 분량.
- **fix_files**: `README.md`
- **effort**: S · **회귀위험**: low

> **STALE/INFO** — WIRE-10: prd US-002/003/004/010 회귀 spot-check 결과 4개 모두 정상(미회귀). 단 동일 shadowing 버그가 5개 data state에 잔존(DATA-02로 추적). 코드 변경 불필요.

---

## 3. VLM 고도화 설계 (Pluggable Backends) / VLM Upgrade Design

현 백엔드는 CLIP 전용(180-prompt best-of 캡셔닝, candidate-list VQA, grounding은 caption fallback)이며 실제 생성형 VLM 실행 불가. 아래는 구현 준비된 설계.

### 3.1 현 한계 (baseline) — [VLM-05, P1, feature]
`core/vlm_inference.py`: `_OBJECT_VOCAB` 60 × `_TEMPLATES` 3 = 180 prompt(88-91). `get_backend`(125-141)이 CLIPCaptioner만 반환, text_encoder 없으면 NotImplementedError. grounding은 caption fallback(model_routes.py:269). batch max_images=50 하드코딩(tabs-extra.js:2190). 단일 run(VLM_BUSY). text_encoder를 사용자가 수동 ONNX 제공.

### 3.2 플러그블 백엔드 인터페이스 — [VLM-06, P1, feature]
`core/vlm_inference.py:56-141`. **VLMBackend를 Protocol→ABC로 전환:**

```python
class VLMBackend(ABC):
    @abstractmethod
    def describe(self, frame, prompt: str, *, max_new_tokens: int, temperature: float) -> str: ...
    @abstractmethod
    def answer(self, frame, question: str, *, candidates=None, max_new_tokens: int, temperature: float) -> str: ...
    @classmethod
    @abstractmethod
    def capabilities(cls) -> dict: ...  # {name, tasks, requires_text_encoder, generative, deps}
    @staticmethod
    def is_available() -> bool: ...
```

- **CLIPBackend**: 기존 CLIPCaptioner 래핑. describe()→caption(), answer()→vqa(). capabilities: generative False, requires_text_encoder True, 추가 deps 없음.
- **TransformersBackend(model_id_or_path, device=auto)**: `__init__`에서 torch+transformers lazy-import(부재 시 pip 안내 RuntimeError). `AutoProcessor.from_pretrained` + `AutoModelForImageTextToText.from_pretrained(..., torch_dtype=auto, device_map='cuda' if cuda else 'cpu')`. chat messages[{role:user, content:[{type:image},{type:text,text:prompt}]}] → `apply_chat_template` → `generate(max_new_tokens, do_sample=temperature>0, temperature)` → decode. BGR np.ndarray → `PIL.Image.fromarray(cv2.cvtColor(frame, BGR2RGB))`.
- **OpenAICompatBackend(base_url, api_key=None, model_id)**: httpx lazy-import. `POST {base_url}/chat/completions`, content image_url = `data:image/jpeg;base64,<cv2.imencode JPEG>` + text, `choices[0].message.content` 파싱, max_new_tokens→max_tokens/temperature 존중, HTTP 오류→RuntimeError.

env 검증: torch251에 torch 2.5.1+cu124(CUDA True), transformers 4.57.3(AutoModelForImageTextToText, Qwen2_5_VLForConditionalGeneration, AutoProcessor 존재), httpx+accelerate 존재. qwen_vl_utils 부재.

### 3.3 백엔드 레지스트리 + 능력 엔드포인트 — [VLM-07, P1, feature]
`core/vlm_inference.py`에 `BACKENDS = {name: class}` 레지스트리 + `list_backends() -> list[dict]`(백엔드별 `is_available()` via `importlib.util.find_spec`, `{name, available, tasks, requires_text_encoder, generative, missing_deps}` 반환) + `make_backend(spec)`(spec.backend dispatch). `server/model_routes.py`에 `@router.get('/api/vlm/backends')` → `{backends: list_backends(), cuda: <torch.cuda guarded>}`. `web/js/api.js`에 `API.vlmBackends`.

### 3.4 요청 스키마 — [VLM-08, P1, feature]
`server/model_routes.py`. InferRequest + VLMBatchRequest에 추가: `backend: str = 'clip'`, `model_id: Optional[str]=None`, `endpoint_url: Optional[str]=None`, `api_key: Optional[str]=None`, `max_new_tokens: int=128`, `temperature: float=0.0`. text_encoder는 VLMBatchRequest에서 Optional. 두 VLM call site(262-273 단일, 360-396 batch)를 `make_backend(spec)` + describe()/answer()로 교체. 백엔드별 검증(clip→text_encoder+model_path 파일, transformers→model_id|model_path, openai→endpoint_url). **api_key는 절대 로깅 금지** — RunRecorder inputs(365-368)에서 제외.

### 3.5 프론트 UI — [VLM-09, P1, feature]
`web/js/tabs-extra.js:2063-2235`. render()에 backend `<select id='vlm-backend' onchange='Tabs.vlm._onBackendChange()'>`(API.vlmBackends로 채움, 미가용은 missing_deps 주석). 조건부 그룹: `#vlm-model-id`(Transformers HF id), `#vlm-endpoint`+`#vlm-api-key`(OpenAI), `#vlm-text-encoder`는 backend==='clip'만. `#vlm-maxtok`/`#vlm-temp` 생성형용. `_onBackendChange()`(_onTaskChange 미러). _readInputs/_payload 확장. _validate에서 text-encoder는 clip만 필수. run() 엔드포인트 `/api/infer/image`로 수정. ko+en i18n 키.

### 3.6 배치 크기 + 단일-run — [VLM-10, P2, feature]
`#vlm-max-images` input(기본 50, 1-500) 추가, runBatch()가 리터럴 50 대신 전달. 서버 clamp 이미 존재. VLM_BUSY 단일-run 플래그 유지(공유 세션 가드).

### 3.7 테스트 전략 — [VLM-11, P1, feature]
새 `tests/test_vlm_backends.py`: (1) list_backends()가 항상 'clip' available=True + transformers/openai 가용성 booleans가 find_spec과 일치. (2) make_backend 오설정 거부(clip without text_encoder, transformers without model_id, openai without endpoint_url). (3) TestClient 라우트: GET /api/vlm/backends 200; POST /api/infer/image backend='openai' bogus endpoint → inline error(422/404 아님); transformers without model_id → 검증 봉투. (4) live transformers는 `pytest.importorskip` + `SSOOK_VLM_LIVE=1` env flag 가드(weights 자동 다운로드 금지), 모델 로드 stub monkeypatch로 wiring 테스트. qwen_vl_utils 미요구.

### 3.8 Optional Deps — [VLM-12, P2, dependency]
새 `requirements-vlm.txt`: `transformers>=4.50,<5` ; `torch>=2.2`(GPU는 PyTorch index의 CUDA build) ; `accelerate>=0.30` ; `qwen-vl-utils>=0.0.8`(선택) ; `httpx>=0.27`. requirements-web.txt에 포인터 노트, README 섹션. CLIP 백엔드는 의존성 free 유지.

### 3.9 선결 버그 (설계 전 반드시 수정) — VLM-01, VLM-02, VLM-03
- **[VLM-01, P0]** `web/js/tabs-extra.js:2152` POST 대상을 `/api/model/infer`(404) → `/api/infer/image`. 그리고 `model_type:'vlm'` bogus 키 제거. (병합: WIRE-01.)
- **[VLM-02, P0]** `server/model_routes.py:213-216, 248, 35-44`. InferRequest에 `model_type: Optional[str]=None` 추가, `model_type = req.model_type or cfg.model_type`로 해석, 'vlm'을 vlm_ task로 매핑(또는 req.vlm_task+req.vlm_text_encoder로 직접 분기). 현재 전역 cfg.model_type(기본 'yolo')라 VLM이 detection으로 오라우팅.
- **[VLM-03, P1]** `server/model_routes.py:217, 252-260`. `imread(req.image_path)`(217)가 검증 전 raw path 사용, VLM 분기 path-safety 블록(252-260)이 반환값 폐기 → dead check. 검증을 함수 상단으로 이동하고 반환된 safe path 사용(`safe_img = safe_image_file(...); frame = imread(safe_img)`).
- **fix_files (VLM-01)**: `web/js/tabs-extra.js` · **(VLM-02)**: `server/model_routes.py`, `web/js/tabs-extra.js` · **(VLM-03)**: `server/model_routes.py`

---

## 4. 릴리스 계획 (implement_now 트랙별) / Release Plan by Ownership Track

각 파일은 정확히 한 트랙이 소유 → 병렬 편집 무충돌. P0/P1 우선.

| 트랙 | 소유 파일 | 포함 finding ID (implement_now=true) | effort | 위험 |
|---|---|---|---|---|
| **viewer** | server/viewer_routes.py | VIEWER-02, VIEWER-03, VIEWER-04, VIEWER-06 | M+S+S+S | low |
| **config** | server/system_routes.py, core/app_config.py, server/config_routes.py | VIEWER-01(+shared/viewer 공유 파일), CONFIG-02, VIEWER-07(공유 core/inference.py) | S+S+M | low~med |
| **eval** | server/eval_routes.py, core/evaluation.py | EVAL-01, EVAL-02(+fe-main), EVAL-04(+benchmark), EVAL-05, EVAL-06, EVAL-07, SPEC-08(+extra) | M·M·S·S·M·M·S | low~med |
| **benchmark** | server/benchmark_routes.py, core/benchmark_runner.py | BENCH-01, BENCH-02, EVAL-04(benchmark_routes 부분) | M+S+S | low |
| **analysis** | server/analysis_routes.py | ANLY-01, ANLY-03, ANLY-04 | S+M+S | low |
| **data** | server/data_routes.py | DATA-01, DATA-02, DATA-03, DATA-04(+fe-extra), DATA-05, DATA-06(+fe-extra) | S·S·M·M·M·M | low~med |
| **quality** | server/quality_routes.py, core/hashing.py | QUAL-02(+fe-extra), QUAL-03, QUAL-04, QUAL-05, QUAL-07, QUAL-08, QUAL-09(+fe-extra), QUAL-10 | S·S·M·M·S·S·M·S | low~med |
| **extra** | server/extra_routes.py, server/optimization_routes.py, core/model_inspector.py, core/model_profiler.py, core/quantizer.py, core/optimizers/*, core/model_diagnosis.py, core/model_cache.py | TOOLS-01, TOOLS-02, TOOLS-03(+fe-extra), TOOLS-04, TOOLS-05, TOOLS-06(+shared), TOOLS-07(+fe-extra), TOOLS-09, SPEC-02(+eval core/evaluation.py), SPEC-03(+fe-extra), SPEC-05(+fe), SPEC-06, SPEC-07(+fe-extra), DATA-07(+fe-extra) | M·S·M·S·S·S·M·S·M·S·M·S·S·M | low~med |
| **vlm** | core/vlm_inference.py, core/clip_inference.py, server/model_routes.py, core/hf_downloader.py | VLM-01(+fe-extra), VLM-02(+fe-extra), VLM-03, VLM-06, VLM-07(+fe-infra api.js), VLM-08, VLM-09(+fe-extra/infra), VLM-10(+fe-extra), VLM-11(tests), VLM-12(requirements/README) | S·S·S·L·M·M·M·S·M·S | low~med |
| **shared** | server/state.py, server/utils.py, server/path_safety.py | (VIEWER-01 utils/viewer_routes 공유), (TOOLS-06 state.py try_start) | S | low |
| **fe-main** | web/js/tabs.js | VIEWER-05, EVAL-02(stop 호출), EVAL-03(+fe-infra) | S·M·S | low |
| **fe-extra** | web/js/tabs-extra.js | ANLY-02, ANLY-05(+fe-infra), QUAL-01, QUAL-06, SPEC-04, (다수 트랙의 fe 부분) | S·S·S·S·S | low |
| **fe-infra** | web/js/app.js, web/js/i18n.js, web/js/icons.js, web/js/api.js, web/js/help-annotations-main.js | SPEC-01(app.js _nav + i18n + icons), WIRE-05(README는 docs로 별도), i18n 키 추가(다수) | S | low |

> 권장 머지 순서: (1) P0 크래시·라우팅(VIEWER-01, DATA-01, DATA-02, SPEC-01, SPEC-02/03, VLM-01/02, QUAL-01, EVAL-01/02, ANLY-01) → (2) P1 보안·취소·정확도 → (3) VLM 고도화(VLM-06~12) → (4) README/문서(WIRE-05).

---

## 5. 보류 (Deferred) / Lower-Priority Backlog

implement_now=false (17건) — 후속 릴리스 추적:

| ID | 탭 | 요지 | severity | effort |
|---|---|---|---|---|
| VIEWER-08 | viewer | path_safety/seq ThreadPool 누수/JPEG 품질 config | P2 | M |
| EVAL-08 | evaluation | 혼동행렬 IoU UI + 대량클래스 DOM | P2 | M |
| EVAL-09 | evaluation | 클래스 매핑 1:N fan-out | P2 | L |
| ANLY-06 | conf-optimizer | GT 부재 클래스 비가시 | P2 | S |
| ANLY-07 | embedding-viewer | 고정 temp 경로/정리 | P2 | S |
| ANLY-08 | embedding-viewer | 추출 진행률 미보고 | P2 | S |
| ANLY-09 | analysis | 추론 2회 낭비 | P2 | S |
| TOOLS-08 | profiler | FLOP stride/dilation 무시 | P2 | M |
| TOOLS-10 | inspector | EP 테스트 세션 spike | P2 | M |
| TOOLS-11 | diagnose | 아키텍처 감지 오분류 | P2 | S |
| TOOLS-12 | diagnose | health score clamp | P2 | S |
| TOOLS-13 | diagnose | findings 영어 전용 i18n | P2 | M |
| DATA-08 | merger | dedup O(N²)/label 휴리스틱 | P2 | L |
| DATA-09 | sampler | target_count over/undershoot | P2 | M |
| DATA-10 | splitter | stratified min-per-class | P2 | M |
| DATA-11 | data | 취소 불능 | P2 | M |
| DATA-12 | explorer | 가상스크롤/페이지네이션 | P2 | M |
| DATA-13 | data | 손상 라벨 침묵 skip | P2 | M |
| QUAL-11 | quality | entropy dead | P2 | S |
| QUAL-12 | quality | 하드코딩 임계 | P2 | M |
| SPEC-09 | pose | keypoint conf 하드코딩 | P2 | S |
| SPEC-10 | embedder | self-sim O(N²·D) | P2 | M |
| SPEC-11 | clip/embedder | 손상 이미지 침묵 skip | P2 | S |
| XCUT-01 | cross-cutting | start 가드 TOCTOU | P2 | M |
| XCUT-02 | cross-cutting | 결과 cap 페이지네이션 | P2 | M |
| XCUT-03 | cross-cutting | 백엔드 메시지 i18n | P2 | M |
| XCUT-04 | cross-cutting | 침묵 skip/broad except 로깅 | P2 | M |
| WIRE-01~04 | wiring | dead route/wrapper/field 정리 | P2 | S |
| VLM-05 | vlm | 현 한계 문서(baseline) | P1(info) | S |

(STALE 항목: INF-10, DATA-14, WIRE-10 — 코드 변경 불필요, FEATURE_AUDIT 갱신만.)
