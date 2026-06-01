# ssook 기능별 감사 (Feature Audit)

> 7개 섹션 30개 탭에 대해 각각 다음 4 가지를 정리한 문서입니다.
> **[1]** 기능 / 최종 목표 — **[2]** 현재 구현 구성 — **[3]** 1과 2의 괴리(리스크·병목·반례) — **[4]** 개선 작업 항목
>
> 우선순위: **P0** Critical · **P1** Important · **P2** Nice-to-have | 작업량: **S** ≤0.5d · **M** ≤2d · **L** >2d

## 목차

1. [Inference](#1-inference) — viewer, settings
2. [Evaluation](#2-evaluation) — benchmark, evaluation
3. [Analysis](#3-analysis) — analysis, model-compare, error-analyzer, conf-optimizer, embedding-viewer
4. [Tools](#4-tools) — inspector, profiler, calibration, diagnose
5. [Data](#5-data) — explorer, splitter, converter, remapper, merger, sampler, augmentation
6. [Quality](#6-quality) — anomaly, quality, duplicate, leaky, similarity
7. [Specialized / 사이드바 미노출](#7-specialized--사이드바-미노출) — segmentation, clip, embedder, pose, instance-seg, tracking, vlm
8. [공통 / 횡단 이슈](#8-공통--횡단-이슈)

---

## 1. Inference

### `viewer` — 실시간 추론 뷰어

**[1] 기능 / 최종 목표**

영상/이미지에서 실시간으로 모델 추론을 수행하고 검출 결과를 화면에 표시. 모델과 입력을 선택하면 MJPEG 스트림으로 시각화. 영상 재생/일시정지/정지, 프레임 스킵·속도, 신뢰도 필터, 추적기(ByteTrack/SORT), 스냅샷/크롭 저장, 다중 작업 타입(검출/분류/분할/포즈/인스턴스분할/CLIP/VLM) 지원.

입력: 모델 + 영상/이미지 경로 → 출력: 박스 오버레이된 MJPEG 스트림 + 통계(FPS, 추론시간, 검출 수).

**[2] 현재 구현 구성**

- 프론트엔드 `web/js/tabs.js:6-550`: 모델 드롭다운, 배치사이즈, 신뢰도 슬라이더, 트래커 옵션, 파일 목록, 재생 컨트롤, 우측 통계 패널
- 상태 `G.model`, `G.videoPath` (`web/js/global.js:2-7`)
- API: `/api/viewer/start`, `/stream/{sid}`, `/status/{sid}`, `/pause`, `/seek`, `/step`, `/speed`, `/snapshot`, `/save-crops`, `/stop`
- 폴링: 상태 300ms (tabs.js:493), 하드웨어 2초 (tabs.js:222)
- 백엔드 `server/viewer_routes.py`: 7개 엔드포인트, `_video_sessions` 전역 dict로 세션 관리. 비활동 5/10분 후 자동 정리(line 22-43)
- MJPEG 루프(line 115-272): 시크/스텝/일시정지 처리 → 프레임 읽기 + 스킵 → 작업 타입별 추론 → 박스 그리기 → cv2.imencode(JPEG quality=65)
- core `core/inference.py`: letterbox 전처리(line 82-110), PreprocessBuffer로 메모리 재사용(line 49-79), 클래스별 offset NMS(line 254-274)
- 자료구조: 전역 dict + 추적기 인스턴스 + sequential 모델용 3프레임 버퍼. **락 없음.**

**[3] 1과 2의 괴리**

- **(a) 정확성** _video_sessions에 락 없음 — 동일 세션 다중 클라이언트 접근 시 상태 손상. 세션 정리 시 cap.release()만, model/tracker/버퍼 참조 잔존 → 메모리 누적. VLM 모델도 검출류로 취급되어 박스 그리기 시도.
- **(b) 스케일** 단일 스레드 read→infer→encode 순차. 4K 영상에서 60-120ms/frame → 8-17 FPS(<30 미달). 스킵으로 프레임 손실. 배치사이즈 UI만 있고 추론에서 무시. JPEG quality=65 하드코딩(line 166/174/194/211/272).
- **(c) 동시성** _video_sessions[sid] 동시 접근 시 일관성 위반. 폴링 300ms — 60FPS 영상에서 18프레임 손실. 추적기 누적 ID/궤적이 정지 없이 메모리에 남음.
- **(d) 에러** 영상 end 시 명시적 신호 없이 status로만 감지. 추론 중 ORT 에러(provider fallback, OOM) 미처리 → 스트림 조용히 중단. snapshot/save-crops 디스크 풀 에러 무시. seek/step 후 추적기 reset 안 함 → ID 점프.
- **(e) UX** 폴더 변경 시 이미지 목록 유실. CLIP/VLM text input이 초기 숨김(line 260-270) — 발견 어려움. speed=x4가 재생 가속인지 프레임 버림인지 불명(코드상 버림). 일부 에러 메시지 i18n 미적용.
- **(f) 보안** symlink traversal 미체크. snapshots/ 폴더 보안 정책 미적용.
- **(g) 플랫폼** cv2.VideoCapture가 OS/빌드별 코덱 다름. 세션 생성 후 EP 변경되어도 이전 세션은 낡은 provider 유지. 모델 메모리 미해제 → 재로드 시 누적.

**[4] 개선 작업 항목**

- **P0** `_video_sessions` 스레드 안전성(RLock + snapshot 메서드). `server/viewer_routes.py:21`. **M**
- **P0** seek/step/stop 후 tracker.reset() 호출. `server/viewer_routes.py:121-124, 332-337`. **S**
- **P0** generate() 예외 처리 세밀화 + 에러 프레임 전송. `server/viewer_routes.py:295-299`. **M**
- **P1** 세션 초기화 완전성(model/tracker/버퍼 정리). `:26-42`. **M**
- **P1** 배치 처리 실제 적용. `:105-300`, `core/inference.py`. **L**
- **P1** 고해상도 최적화: read/infer/encode 분리 파이프라인 + ring buffer + 입력 다운스케일 + JPEG 품질 동적. **L**
- **P1** CLIP/VLM 추론 통합 (`/api/infer/image` 로직을 MJPEG 루프에). `:212-271`. **M**
- **P1** 이미지 폴더 UX 개선(자동재생→리스트 클릭). `web/js/tabs.js:384-407`. **M**
- **P1** 폴링 적응형 또는 WebSocket. `tabs.js:484-495`. **M**
- **P2** JPEG quality UI 노출. **S** / 프레임 드롭 UI 표시. **S** / 추적기 선택 UI를 모델 타입별 제약. **S** / 앙상블 다중 모델. **L**

---

### `settings` — 애플리케이션 설정

**[1] 기능 / 최종 목표**

전역 설정(박스 두께, 레이블 크기 등), 클래스 스타일(색상/두께/활성화), 커스텀 모델 타입 등록, 테스트 모델/데이터 다운로드 링크. → app_config.yaml에 영속화.

**[2] 현재 구현 구성**

- 프론트엔드 `web/js/tabs.js:553-990`: 설정 폼 + 클래스 테이블 + 다운로드 링크 + **커스텀 모델 타입 다이얼로그(710-990)**: 모델 선택→추론→출력 텐서 인덱스/dim_roles/attr_roles 매핑→테스트 추론→class_names CSV→저장
- 백엔드 `server/config_routes.py` 4 엔드포인트 + `system_routes.py`(파일 다이얼로그)
  - `GET/POST /api/config`, `POST /api/config/class-style`, `POST /api/config/custom-model-type[/test]`
- core `core/app_config.py`: AppConfig 싱글톤, threading.Lock(), atomic YAML write(replace)
- Dimension/Attribute Mapping UI(818-881, 857-876) — 사용자가 슬롯별 의미 수동 선택
- Windows 파일 다이얼로그는 PowerShell, 타 OS는 AppleScript/zenity, fallback은 tkinter

**[3] 1과 2의 괴리**

- **(a)** 클래스 스타일 UI BGR 색상이 viewer MJPEG 박스에 미반영(viewer는 팔레트 자동 생성 사용). 단일 output_index만 지원 → 다중 출력 모델 처리 불가. attr_roles 빈 슬롯이 빈 문자열로 저장 → postprocess_custom() index error 위험. coord_format "xyxy" 고정.
- **(b)** 클래스 80+에서 320+ DOM 요소 생성 → 수백 ms 레이아웃. "Select All"이 클래스 수만큼 POST → 100×100ms = 10초.
- **(c)** YAML atomic write 있지만 다른 탭 GET이 부분 쓰기 사이에 부분 데이터 읽을 수 있음. 같은 class_id 동시 업데이트는 last-write-wins.
- **(d)** 커스텀 모델 테스트 실패 메시지 부실. 파일 선택 취소 검증 없음. YAML 쓰기 실패 시 500.
- **(e)** dimension/attribute 매핑 가이드 부족. class_names CSV 형식 명시 없음. 다운로드 링크 하드코딩.
- **(f)** 커스텀 모델 타입 name에 특수문자/path separator 체크 없음. 파일 선택 경로 재검증 부재.
- **(g)** PowerShell 보안 정책 제한 시 fallback UX 차이.

**[4] 개선 작업 항목**

- **P0** 클래스 스타일을 viewer에 적용(또는 viewer 로드 시 강제 재로드). `web/js/tabs.js:625-631`, `server/viewer_routes.py:228-271`. **S**
- **P0** `attr_roles` 빈 슬롯 처리(skip + offset 조정). `config_routes.py:130-136`, `core/inference.py:postprocess_custom`. **M**
- **P0** 클래스 스타일 배치 엔드포인트(list). `web/js/tabs.js:675-690`, `config_routes.py:69-84`. **M**
- **P1** 커스텀 모델 입력 검증(dim_roles 합/빈 attr_role/coord_format). `config_routes.py:99-114`. **M**
- **P1** 다중 출력 텐서 지원(output_indices list). `core/app_config.py:24-37`. **L**
- **P1** 클래스 테이블 가상 스크롤. `web/js/tabs.js:632-659`. **M**
- **P1** 좌표 형식 선택 UI(xyxy/xywh/cxcywh). `web/js/tabs.js:734-740`, `core/inference.py`. **M**
- **P1** 커스텀 모델 다이얼로그 가이드 + 프리셋 템플릿. `web/js/tabs.js:710-990`. **L**
- **P1** 다운로드 링크 동적화(GitHub API). `web/js/tabs.js:564-585`. **M**
- **P2** class_names UI 개선(테이블 형식). **S** / 클래스 스타일 프리셋. **M** / 커스텀 모델 변경 이력. **L** / YAML 미리보기. **S**

---

## 2. Evaluation

### `benchmark` — 벤치마크

**[1] 기능 / 최종 목표**

다중 모델·코덱 조합의 추론 성능 측정(FPS, P50/P95/P99, CPU/GPU%, decode_ms). 입력: 모델 목록 + 반복 횟수 + 입력 크기 + 코덱(none/JPEG/PNG/WebP/H.264/H.265). 출력: BenchmarkResult JSON + CSV 익스포트.

**[2] 현재 구현 구성**

- 프론트엔드 `web/js/tabs.js:994-1110` — run/poll(500ms)/stop. API `/api/benchmark/{run,status,stop,export-csv}`
- 백엔드 `server/benchmark_routes.py` → core `core/benchmark_runner.py`(413 lines)
- warmup(300회 고정) + 메인 루프(iterations). 단계별 타이밍(전처리/추론/후처리). 기본 소스 1080×1920. CPU: 매 20회 `cpu_percent(interval=None)`. GPU: `nvidia-smi` (타임아웃 2초). 통계: numpy percentile.
- `bench_state = TaskState()` (RLock). 코덱은 더미 프레임 인코딩 재사용. H.264/H.265는 임시 MP4.
- 모델 고정 배치 자동 감지 조정(line 160-164).

**[3] 1과 2의 괴리**

- **(a) 정확성** GPU 샘플링 매 20회 → 가벼운 모델의 스파이크 놓침. `cpu_percent(interval=None)`는 마지막 호출 이후 경과시간 의존 → 루프 시간 변화 시 값 변동. warmup 타이밍이 total_steps에 포함되어 초기 오버헤드를 정확히 분리 못함. decode_ms 100회 평균이 캐시 효과로 초회와 다름. H.264/H.265 단회 open 비용 미포함.
- **(b) 스케일** 100 모델 × 5 코덱 × 500 반복 = 3시간+. `encoded_bufs` 메모리 누적(4K → 수 GB). 고정 배치=4 모델은 단일 이미지를 4배 패딩 → FPS 왜곡.
- **(c) 동시성** `is_stopped()` 체크가 느슨해 다음 반복 시작 후 종료. 폴링 중 재실행 거부만 함.
- **(d) 에러** 모델 로드 실패 시 `done += warmup + iterations` 가산 → 진행률 왜곡. 코덱 인코딩 실패는 "—"만 표시.
- **(e) UX** 500ms 폴링이 빠른 백엔드 진행을 못 따라감. 결과 테이블 15열 → 수평 스크롤. GPU% N/A 원인 설명 없음.
- **(f) 보안** 임시 MP4 finally 정리는 있으나 예외 시 잔존 가능. 모델 경로 검증 없음.
- **(g) EP/메모리** GPU 벤치마크 중 ORT 미정리 → 연속 시 CUDA OOM. 같은 모델을 여러 EP로 비교 불가(요청 1개당 1 세션).

반례: 고정 배치=4 + iter=100 → 400 이미지 처리, P50이 배치 단위. 추론 50ms 모델의 GPU 90% 스파이크가 20회 평균 10%로 보임.

**[4] 개선 작업 항목**

- **P0** GPU 샘플링 별도 스레드 + 주기 단축(매 20→2-5). `core/benchmark_runner.py:256-260`. **M**
- **P0** 고정 배치 FPS 정정(fps/batch). `:294`. **S**
- **P1** decode_ms 정확도(첫 회 + 평균 분리, file open 포함). `server/benchmark_routes.py:114-131`. **M**
- **P1** CPU 샘플링 `interval=0.05` 고정. **S**
- **P2** 임시 파일 정리 강화 + 로깅. **S** / warmup 후 상태 로깅. **S** / 모델 경로 검증. **S**

---

### `evaluation` — 평가 (mAP/P/R/F1 etc.)

**[1] 기능 / 최종 목표**

다중 모델 정확도 평가. Detection(mAP@50, mAP@50:95, P/R/F1, 혼동행렬), Classification(Acc, P/R/F1), Segmentation(mIoU/mDice), CLIP(Top-1/Top-5), Embedder(Retrieval@K). 입력: 모델+타입, 이미지 디렉토리, GT 라벨, 클래스 매핑. 출력: 모델별/per-class 메트릭.

**[2] 현재 구현 구성**

- 프론트엔드 `web/js/tabs.js:1114-1663`: 태스크 선택, 모델 슬롯, **클래스 매핑 다이얼로그**(SVG 시각화), 신뢰도(기본 0.25), 폴링 결과 렌더.
- API `/api/evaluation/{run-async,status,export-csv}`, `/api/eval/stop`
- 백엔드 `server/eval_routes.py`(797 lines):
  - GT 로드 → 모델별: 이미지 캐시(500장 고정) → 추론 → 클래스 리매핑 → mapped_only 필터 → `evaluate_dataset(iou=0.5)` + `evaluate_map50_95()` + `_build_confusion_matrix()`
  - 결과 자동저장 `eval_history/eval_YYYYMMDD_HHMMSS.json`
- core `core/evaluation.py`(284 lines):
  - greedy matching(confidence sort), 101-point AP interpolation
  - `_compute_iou_matrix` vectorized
- `eval_state = TaskState()` (RLock). 매핑은 메모리 dict.

**[3] 1과 2의 괴리**

- **(a) 정확성**
  1. greedy matching이 COCO 표준과 다름(전역 confidence sort vs per-GT best-pred) → mAP ±2~5% 차이.
  2. 클래스 매핑이 일대일만 가능(다대일 불가).
  3. `mapped_only=True` 시 unmapped GT 무시 → FN 누락 → Recall 부풀려짐.
  4. IoU 0.5 고정.
  5. macro/micro 구분 없음 → 작은 클래스 과소평가.
- **(b) 스케일** 이미지 캐시가 LRU 아닌 고정 500장 → 초과 시 매번 재로드(10k×10 모델에서 큰 I/O). 혼동 행렬 IoU 중복 계산.
- **(c) 동시성** 모델 순차 처리, 중단 신호는 현재 이미지 추론 끝나야 반영.
- **(d) 에러** GT 부재 시 `boxes=[]`로 조용히 진행 → 모두 FP. 모델 로드 실패는 결과에 error 기록(OK). 클래스 ID 부동소수 시 int() truncate.
- **(e) UX** 100×100 매핑 다이얼로그가 SVG 1만 선 → 수초 지연. 매핑 영속화 없음(재시작 시 손실). 수천 클래스 결과 JSON 10MB+ → 폴링 지연. 80×80 혼동 행렬 6400 셀 DOM.
- **(f) 보안** label_dir/model_path 경로 traversal 검증 없음.
- **(g) EP/메모리** 평가 후 모델 객체 잔존 → 장시간 반복 시 메모리 증가.

반례:
1. GT(class1=car) + Model(class2=vehicle, 2→1 매핑) + 모델이 class0도 출력 → class0은 GT에 없어 모두 FP → 사용자는 매핑 오류로 착각.
2. 고정 배치=4 + 99장 → 마지막 배치 패딩 미검증 → 혼동 행렬에 쓰레기.
3. GT 포맷이 (x,y,x2,y2,cls)이면 파싱 오류.

**[4] 개선 작업 항목**

- **P0** GT-pred matching COCO 호환. `core/evaluation.py:49-119`. **M**
- **P0** 클래스 매핑 다대일 지원. `web/js/tabs.js:1314-1541`, `server/eval_routes.py:203-214`. **M**
- **P0** mapped_only Recall 의미 명확화(옵션 이름 변경 + 결과 표시). **S**
- **P1** 이미지 캐시 LRU. `OrderedDict`. **S**
- **P1** 혼동 행렬 IoU 중복 제거(`evaluate_dataset` 결과 재사용). **M**
- **P1** 매핑 다이얼로그 100+ 클래스 시 테이블 모드. **M**
- **P1** `/api/eval/save-mappings` 영속화. **M**
- **P2** GT 부재 경고(`missing_gt_files`). **S** / macro/micro 선택. **M** / 경로 검증. **S** / Segmentation per-image 통계. **M**

---

## 3. Analysis

### `analysis` — 단일 이미지 추론 검사

**[1] 기능 / 최종 목표**

1장 이미지로 letterbox + 추론 결과 + 텐서 통계 시각화. 입력: 모델·타입·이미지+conf. 출력: 원본/letterbox/detection 이미지 3종 + timing(pre/infer/post) + 클래스별 신뢰도.

**[2] 현재 구현 구성**

- 프론트엔드 `web/js/tabs.js:1666-1725`: `POST /api/analysis/inference-analysis` → base64 이미지 3개 + timing + detections
- 백엔드 `server/analysis_routes.py:493-583`: `_load(model_path, model_type)` → `run_inference()` → DetectionResult, letterbox 시각화는 빨간 overlay로 패딩 영역 표시, `draw_detections()` 박스 그리기, 텐서 min/max/mean/std
- core `core/inference.py`: letterbox + `_padded_to_tensor()` NCHW float32 [0,1], postprocess_v8/v5/darknet/detr, NMS

**[3] 1과 2의 괴리**

- **(a) 정확성** Letterbox 패딩 시각화에서 float→int 강제 변환 시 1-2px rounding error 가능(415×415 등 비율 안 떨어지는 size).
- **(b) 스케일** 단일 이미지만 → 배치 성능(batch_size>1) 확인 불가.
- **(c) 동시성** GPU ORT session에 락 없이 동시 호출 → 고부하 시 race.
- **(d) 에러** try/except 1개로 단계 구분 안 됨(이미지 로드 vs 추론 vs OOM).
- **(e) UX** "No detections" vs "N detections" 표현 불일치. 신뢰도 분포 히스토그램 없음.
- **(f) 보안** image_path 경로 sandbox 없음.
- **(g) 의존성** cv2 필수.

**[4] 개선 작업 항목**

- **P1** Letterbox float→int 반올림 보정. `core/inference.py:letterbox`, `server/analysis_routes.py`. **S**
- **P1** try/except 세분화(load/infer/encode). **S**
- **P1** GPU 추론 동시성 락(task_locks["gpu_infer"]). **S**
- **P2** 배치 이미지 입력 옵션. **M**
- **P2** 경로 sandbox 검증. **S**
- **P2** 신뢰도 분포 차트. **M**

---

### `model-compare` — A/B 비교(슬라이더)

**[1] 기능 / 최종 목표**

두 모델을 동일 이미지 디렉토리에서 실행, 이미지별 슬라이더로 비교. 입력: model_a/b + types + img_dir + conf. 출력: 이미지별 count_a/b + ms_a/b + 슬라이더 뷰.

**[2] 현재 구현 구성**

- 프론트엔드 `web/js/tabs-extra.js:71-165`: 배경 작업 + 슬라이더 + `/api/analysis/model-compare/image/{i}/{side}` 로 lazy 이미지 로드
- 백엔드 `server/analysis_routes.py:32-127`: 두 모델 순차 호출, 결과를 임시 JPG로 저장(메모리 절감). `compare_state` 전역, GET /status, GET /image/{i}/{side}.

**[3] 1과 2의 괴리**

- **(a)** count는 conf≥0.25 고정. 사용자 조정 불가.
- **(b)** N=10k+ 이미지에서 2N 임시 JPG → 디스크 I/O 병목.
- **(c)** `compare_state` 전역 → 동시 실행 시 results 덮어쓰기 race.
- **(d)** 모델 A/B 로드 실패 구분 안 됨.
- **(e)** 박스 차이를 시각적으로 강조(공통/A-only/B-only) 안 함.
- **(f)** img_dir 경로 검증 없음. /tmp 다른 프로세스 접근 가능.
- **(g)** cv2 필수.
- **(h)** A와 B의 input_size가 다르면(640 vs 416) 작은 객체 검출률이 다를 수 있어 count 비교가 부당.

**[4] 개선 작업 항목**

- **P0** `compare_state` 동시성 격리(task별 state). `server/analysis_routes.py:run_model_compare` + `state.py`. **S**
- **P1** A/B 예외 구분. **S** / conf 파라미터화. **S** / 임시 파일 finally 정리. **S**
- **P2** 입력 사이즈 정규화 옵션. **M** / 박스 차이 시각화(공통/Δ). **M**
- **P3** 스트리밍/in-memory 버퍼. **M** / A/B 병렬 실행. **M**

---

### `error-analyzer` — FP/FN 에러 분류

**[1] 기능 / 최종 목표**

GT 대비 FP/FN을 크기(S/M/L) × 위치(Top/Center/Bottom)로 분류. 입력: 모델 + img_dir + label_dir + IoU(0.5) + conf(0.25). 출력: 8행 테이블(FP/FN × 7 카테고리).

**[2] 현재 구현 구성**

- 프론트엔드 `web/js/tabs-extra.js:168-238`: 입력 + 폴링 + 결과 테이블
- 백엔드 `server/analysis_routes.py:129-245`: 이미지별 순회 → `run_inference()` → txt 파싱 → denormalize → 단순 for IoU(vectorize 아님) → greedy matching → size(<32², <96², ≥96² COCO 기준), pos(cy<0.33/0.67) 분류

**[3] 1과 2의 괴리**

- **(a) 정확성**
  - IoU 0.5, 크기 32²/96², position 0.33/0.67 모두 hardcoded → 도메인 변경 불가
  - **클래스 일치 검사 누락**: IoU만 매칭하므로 dog↔cat가 IoU 높으면 TP 처리. **버그**.
- **(b) 스케일** O(N_pred × N_gt) 단순 이중 루프 — 이미지당 GT 1000개면 1M 연산.
- **(c) 동시성** `error_analysis_state` 단일 전역.
- **(d) 에러** GT 부재 silent skip → 사용자는 모름.
- **(e) UX** 텍스트 수치만, 시각화 없음.
- **(f) 보안** label_dir 검증 없음.

**[4] 개선 작업 항목**

- **P0** IoU 매칭에 class_id 일치 추가. `server/analysis_routes.py:_run()` matching 루프. **S**
- **P1** GT 부재 경고 메시지. **S** / 상태 격리. **S** / IoU·크기 경계 파라미터화. **S**
- **P2** IoU 매칭 vectorize(`_compute_iou_matrix` 재사용). **M** / FP/FN 분포 차트. **M**
- **P3** 오분류 분석(혼동 행렬과 통합). **M** / 단일 이미지 드릴다운. **M**

---

### `conf-optimizer` — 신뢰도 threshold 최적화

**[1] 기능 / 최종 목표**

per-class confidence threshold 0.05~0.95 grid sweep → F1 최대화 threshold + PR curve. 입력: 모델 + img_dir + label_dir + step + conf_range. 출력: per-class best_threshold/F1/P/R + PR curve.

**[2] 현재 구현 구성**

- 프론트엔드 `web/js/tabs-extra.js:241-351`: 결과 테이블 + SVG PR curve(_showPR)
- 백엔드 `server/analysis_routes.py:248-368`:
  - Phase1: low conf(0.01)로 전부 추론 → all_preds, all_gt 수집
  - Phase2: 클래스별 threshold 그리드 sweep, greedy matching → P/R/F1
  - 결과: [{class_id, best_threshold, best_f1, P, R, pr_curve[]}]

**[3] 1과 2의 괴리**

- **(a) 정확성** Grid step 0.05 고정 → 최적값 사이 손실. IoU 0.5 hardcoded. GT에 없는 클래스는 빠짐(문서 없음). 크기별 threshold(small vs large) 미지원.
- **(b) 스케일** Phase1이 전체 dataset에 O(N) 저신뢰도 추론 → 50k+ 이미지에서 수십 분.
- **(c) 동시성** `conf_opt_state` 전역.
- **(d) 에러** GT 클래스 0개 시 빈 결과 반환만 함.
- **(e) UX** SVG PR curve 축 폰트 9px → 가독성 낮음.
- **(f) 보안** 경로 검증 없음.
- **(h) 클래스 불균형** GT 20개 vs 5000개도 같은 sweep → 소수 클래스 P/R 매우 불안정.

**[4] 개선 작업 항목**

- **P1** Grid step / IoU threshold 파라미터화. **S** / 상태 격리. **S** / GT 부재 경고. **S**
- **P2** 2-pass refine(coarse 0.05 → fine 0.01). **M** / per-size threshold. **M** / 서버사이드 PNG PR curve. **M**
- **P3** 클래스 불균형 confidence interval. **M** / 다중 IoU 동시 sweep. **L**

---

### `embedding-viewer` — t-SNE/UMAP/PCA 시각화

**[1] 기능 / 최종 목표**

임베딩 모델 → 모든 이미지 vector → 2D 감소(t-SNE/UMAP/PCA) → 클래스별 색상 산점도. 입력: 모델 + img_dir(폴더=클래스) + method.

**[2] 현재 구현 구성**

- 프론트엔드 `web/js/tabs-extra.js:355-415`: POST + 폴링 + PNG 표시
- 백엔드 `server/analysis_routes.py:371-489`:
  - 224×224 resize + RGB norm /255 + ONNX forward
  - method=pca/umap/tsne → matplotlib scatter + legend → 임시 PNG

**[3] 1과 2의 괴리**

- **(a)** 입력 resize 224×224 고정 → 모델 input_size 무시. t-SNE perplexity min(30, N-1), UMAP n_neighbors min(15, N-1) → N<30에서 비정상. L2 norm 미적용.
- **(b)** **t-SNE O(N²) 시간·메모리** — N=10k에서 100MB+ pairwise affinity 행렬, 수십 분. UMAP은 더 빠르지만 큰 dataset에서 부족. embeddings 배열 자체가 N×D, D=1024일 때 N=10k → 40MB.
- **(c)** `embedding_state` 전역.
- **(d) 의존성** UMAP optional이지만 frontend dropdown에 항상 노출 → 미설치 시 사용자가 선택 후 실패.
- **(e)** `tab10` cmap 10색만 — 클래스 >10에서 색 부족.
- **(f)** img_dir traversal 검증 없음.
- **(h)** 색 정규화(ImageNet stats) 미적용 — 모델별 전처리 불일치.
- **(i)** 레이블이 폴더명만 — 다른 라벨 구조 미지원.

**[4] 개선 작업 항목**

- **P0** UMAP optional 확인 후 frontend 동적 옵션. `server/analysis_routes.py:_run()` + frontend init. **S**
- **P1** ModelInfo.input_size 기반 유연 resize. **S** / t-SNE perplexity 경고(N<30). **S** / 상태 격리. **S**
- **P2** 색상 palette 확장(tableau20/xkcd). **S** / L2 norm 옵션. **S** / N>5k 시 subsample 또는 approximate t-SNE. **M** / filename/manifest 라벨링. **M**
- **P3** Interactive plot(plotly/canvas). **M** / 배치 임베딩 제너레이터. **M**

---

## 4. Tools

### `inspector` — 모델 그래프 검사

**[1] 기능 / 최종 목표**

ONNX 모델 구조 파악: I/O shape, op 분포, 노드/파라미터, EP 호환성.

**[2] 현재 구현 구성**

- 프론트엔드 `web/js/tabs-extra.js:1403-1474`: 카드 그리드 렌더링
- 백엔드 `server/extra_routes.py:406-415`: `/api/inspector/inspect` + `get_or_compute` 캐시
- core `core/model_inspector.py`: ORT InferenceSession + ONNX 로드 → 각 EP 테스트 세션 생성 → `_get_ep_supported_ops` 휴리스틱

**[3] 1과 2의 괴리**

- **(a)** ORT가 per-node EP 할당 미노출 → CPU fallback 감지 불가. `_get_ep_supported_ops`가 None 반환 시 호환성 1.0(전부 OK)으로 fallback.
- **(b)** 1000+ 노드 모델에서 EP마다 세션 생성 시 메모리/시간 스파이크. 4GB+ 모델에서 EP 4-5개 테스트하면 수십 초.
- **(c)** `get_or_compute` 캐시 dict 락 명시 없음.
- **(d)** EP 실패 원인(CUDA/TRT 미설치) 미보고.
- **(e)** 진행률 없음.
- **(g)** 휴리스틱이 ORT 버전에 따라 outdated 가능.

**[4] 개선 작업 항목**

- **P0** `_get_ep_supported_ops` 구현/개선. `core/model_inspector.py`. **M**
- **P0** EP 테스트 병렬화(concurrent.futures). **M**
- **P1** 캐시 동시성 락. **S** / EP별 실패 이유 캡처. **S**
- **P2** 진행률 API. **M** / 디스크 사이드카 캐시. **M**

---

### `profiler` — 모델 프로파일링

**[1] 기능 / 최종 목표**

지연시간(avg/min/max/P50/P95/P99), 레이어별 병목, 메모리 추정, 양자화 가능도, 최적화 제안. 입력: 모델 + 반복(20).

**[2] 현재 구현 구성**

- 프론트엔드 `web/js/tabs-extra.js:1477-1623`: 격자 결과 + severity 색상
- 백엔드 `server/extra_routes.py:419-424` → `profile_model(path, num_runs)`
- core `core/model_profiler.py`:
  - Dummy feed 640×640, warmup 1회, ORT enable_profiling → JSON 파싱
  - FLOP 추정(padding/stride 무시), activation 고정 추정
  - hardcoded 휴리스틱 제안

**[3] 1과 2의 괴리**

- **(a)** Dummy feed → 실제 입력 분포/캐시 미반영. FLOP padding/stride 무시. Activation 고정 → 큰 배치 과소 계산. ORT profiling 자체가 ~5-10% 오버헤드.
- **(b)** 1000+ 노드 × 20 runs → 수십 초.
- **(c)** sync API — 다른 inference 탭 응답성 저하.
- **(d)** Dummy shape 불일치 시 silent skip → "0개 연산 분석됨".
- **(e)** Warmup 1회 부족(JIT 미완료). high/medium/low 절대 기준 없음.
- **(g)** CPUExecutionProvider 고정 → GPU 모델 프로파일 부정확.

**[4] 개선 작업 항목**

- **P0** EP 선택 지원(드롭다운 + `profile_model(..., ep='cuda')`). **M**
- **P0** Activation peak 정확도(ORT profiling 또는 TensorProto shape). **M**
- **P1** Warmup 횟수 UI. **S** / FLOP padding·stride 정확 계산. **M** / 병목 절대 기준(>10ms→high). **S**
- **P2** 진행률 스트림. **M** / Calibration 데이터로 실제 입력 profiling. **L**

---

### `calibration` — 양자화 / 최적화 파이프라인

**[1] 기능 / 최종 목표**

Dynamic INT8, Static INT8(calibration), FP16, Pruning(Weight/Channel), Graph Opt, Pipeline Builder.

**[2] 현재 구현 구성**

- 프론트엔드 `web/js/tabs-extra.js:1626-1902`: 조건부 UI + Pipeline Builder + 폴링
- 백엔드: `server/extra_routes.py:327-387`(quantize) + `server/optimization_routes.py`(optimize/methods)
- core `core/quantizer.py`(`_AutoCalibrationReader`), `core/optimization_pipeline.py`, `core/optimizer_registry.py`

**[3] 1과 2의 괴리**

- **(a)** `_AutoCalibrationReader`가 "일반 전처리"만 가정 → YOLO letterbox/custom norm 모델에서는 calibration 부정확 → 정확도 큰 하락. Per-channel always-enabled → Transformer attention에서 부정확. Pruning 정확도 미검증 → 누적 손실.
- **(b)** Static calibration max_images=100 → ResNet-101 10분+. Pipeline 5단계 = 5× read-write-parse.
- **(c)** quant_state/opt_state가 단일 전역 dict → 동시 작업 충돌.
- **(d)** Calibration 폴더 비었을 때 안내 부족. WEBP/TIFF 미지원 시 메시지 없음. Pruning 정확도 급락 시 롤백 없음.
- **(e)** Pipeline 단계별 파라미터 hardcoded(weight pruning 30% 고정 등).
- **(f)** QDQ vs QOperator 선택 가이드 없음.

**[4] 개선 작업 항목**

- **P0** Calibration 전처리 자동 감지 또는 선택. **L**
- **P0** 상태 격리(unique job id). **M**
- **P1** Pruning 정확도 검증(선택적 calibration). **L** / Pipeline 메모리 처리(임시 파일 제거). **L** / 단계별 파라미터 UI. **M**
- **P2** Calibration 샘플링 추천. **S** / EP 기반 quant_format 자동 선택. **S** / Calibration 분포 시각화. **M**

---

### `diagnose` — 모델 진단 + 권장

**[1] 기능 / 최종 목표**

종합 분석(아키텍처/가중치/양자화 가능도/pruning 기회/graph efficiency), Health score, Findings, Recommendations(executable + pipeline_config), 차트.

**[2] 현재 구현 구성**

- 프론트엔드 `web/js/tabs-extra.js:1906-2060`: 폴링 4단계, 차트, Apply 버튼
- 백엔드 `server/optimization_routes.py:108-202`: `ModelDiagnosisEngine` → `RecommendationEngine` → 차트 → state. Apply는 `/api/optimize/run`으로 위임.
- core `core/model_diagnosis.py`(op overlap 휴리스틱, weight 통계, sensitive_nodes, prunable channels, fusable patterns), `core/diagnosis_charts.py`(matplotlib 5개 차트)

**[3] 1과 2의 괴리**

- **(a)** 아키텍처 감지 op overlap만 → ResNet/RegNet/EfficientNet 구분 불가. Quant 민감도 = weight range × outlier_ratio → 실제 정확도 손실과 상관 약함. Sparsity threshold 1e-7 hardcoded — FP16 모델 noise와 구분 어려움. Health score가 finding 개수만 반영(심각도 가중 부재).
- **(b)** 10000+ 노드 모델 분석 수십 초. matplotlib 5개 직렬 → 3-5초.
- **(c)** `diag_state` 단일 dict → 동시 진단 불가.
- **(d)** 일부 recommendation `executable=False`(KD/Fine-tuning) → 사용자가 실행 불가. matplotlib 부재 시 `charts:{error:...}`.
- **(e)** Findings/Recommendations 영어 only. expected_impact 정성적("5-15%"). Apply 후 calibration 탭으로 자동 전환되는데 사용자가 놓침.

**[4] 개선 작업 항목**

- **P0** 아키텍처 감지 휴리스틱 + 파일명/HF 모델 ID. **M**
- **P0** 상태 격리(job id). **M**
- **P1** Quant 민감도 정확도(activation distribution, SNR). **L** / Health score 가중. **M** / 차트 병렬 생성. **S**
- **P2** Findings i18n. **M** / Expected impact 정량화(micro-benchmark). **L** / 대형 모델 sampling. **M** / Recommendation 정렬(executable 우선). **S**

---

## 5. Data

### `explorer` — 데이터셋 탐색기

**[1] 기능 / 최종 목표**

데이터셋 갤러리 + 필터(class/box count) + 분포 차트(class/size/aspect). 입력: img_dir + label_dir + limit. 출력: files[], class_counts, box_sizes, aspect_ratios + CSV.

**[2] 현재 구현 구성**

- 프론트엔드 `web/js/tabs.js:1728-2013`: list/chart_box/chart_image/size_dist/aspect_dist/box_aspect_dist 뷰, 클래스 다중선택 + box count 연산자
- 백엔드 `server/data_routes.py:22-162`: explorer_state + 스레드풀, glob_images → limit까지 → txt 파싱 → 통계 누적
- 자료구조: TaskState(data={total, shown, files[], class_counts{}, img_class_counts{}, box_sizes[], aspect_ratios[], box_aspect_ratios[]})

**[3] 1과 2의 괴리**

- **(a)** limit이 "전체 스캔"과 "갤러리 표시" 두 의미로 혼용 → 10만장 중 1000장만 통계에 잡혀 차트 편향.
- **(b)** 50k 갤러리 DOM 노드 50000개 → 브라우저 버벅거림(virtual scroll 미구현).
- **(c)** 로드 중 취소 불가. running 플래그만 중복 차단.
- **(d)** 라벨 포맷 오류/이미지 읽기 실패 silent skip → 부분 데이터로 차트 왜곡.
- **(e)** "Total/Shown" 통계가 필터 적용 전 차트 리렌더 시 오래된 상태 반영 가능.
- **(f)** `safe_image_dir()` 미사용 → ../../, symlink traversal 가능성.
- **(g)** 클래스 ID가 비숫자일 때 sort 깨짐.

**[4] 개선 작업 항목**

- **P0** `safe_image_dir()` 적용. `data_routes.py:24, 129`. **S**
- **P0** limit 의미 분리(전체 스캔 vs 갤러리 표시). **M**
- **P1** 가상 스크롤. `tabs.js:1789-1899`. **L**
- **P1** 취소 메커니즘 + UI 취소 버튼. **M**
- **P1** 부분 오류 로깅(`explorer_state.errors[]`). **M**
- **P2** 필터 적용 시점 명확화. **S** / 비숫자 class ID 검증. **M**

---

### `splitter` — 데이터셋 분할

**[1] 기능 / 최종 목표**

train/val/test 분할(random/stratified). 입력: img_dir/label_dir/output_dir/비율/전략. 출력: split별 images/labels 디렉토리 + 카운트.

**[2] 현재 구현 구성**

- 프론트엔드 `web/js/tabs.js:2017-2110`: 비율 3개 + 전략 select + 폴링
- 백엔드 `server/data_routes.py:177-274`:
  - Stratified: 이미지의 "클래스 집합"으로 그룹화 → 그룹 내 fractional accumulation(`frac_train += gn * norm_train; n_train = round(frac_train)`)
  - Random: 셔플 후 비율 절단
  - 복사: `copy2()`

**[3] 1과 2의 괴리**

- **(a) Stratified 의미** "이미지 단위 비율 보존"만 보장 — 모든 클래스가 모든 split에 나타나는 보장 없음. 클래스당 이미지 1장이면 train에만 → val/test에 누락.
- **(b) 비율 정규화** 비율 합 ≠ 1일 때(예: 0.5/0.3/0.3) round() 누적 오차 → 100장이 110장으로 split.
- **(c)** 10만장 분할 시 라벨 파일 6M 회 읽기/닫기 — I/O 병목.
- **(d)** long-running 중 탭 전환 → UI/백엔드 진행도 불일치.
- **(e)** 라벨 부재 silent skip(이미지만 복사).
- **(f)** `safe_path()` 미사용. output_dir 권한 검증 없음 — 기존 파일 덮어쓰기.
- **(g)** Box-level stratification 미구현 — 클래스별 박스 수 불균형 시 불안정.

**[4] 개선 작업 항목**

- **P0** 경로 검증. **S**
- **P0** 분할 비율 정규화 + 최종 조정(합 100 보장). **M**
- **P1** Stratified 최소 클래스 보장(per-class quota). **M**
- **P1** Box-level Stratification 옵션. **L**
- **P1** I/O 성능(배치 읽기/캐싱). **M**
- **P2** 부분 실패 시 롤백. **M** / 취소 메커니즘. **M**

---

### `converter` — 포맷 변환

**[1] 기능 / 최종 목표**

YOLO ↔ COCO JSON ↔ Pascal VOC XML. 입력: input/output dir + from/to. 좌표 변환(정규화↔픽셀, xyxy↔xywh↔cxcywh).

**[2] 현재 구현 구성**

- 프론트엔드 `web/js/tabs-extra.js:743-789`
- 백엔드 `server/data_routes.py:287-423`: YOLO→COCO(이미지 크기 필요), COCO→YOLO(cx=x+w/2 / img_w), YOLO→VOC(bndbox xyxy)

**[3] 1과 2의 괴리**

- **(a)** 좌표 반올림 형식 불일치(`round(x,1)` vs `.6f`) → 왕복 변환 시 누적 오차. 이미지 크기 찾기 실패 시 **기본값 640×640 사용** → 박스 좌표 왜곡. Segmentation/polygon은 bbox로만 변환(정보 손실).
- **(b)** 대용량 COCO JSON(100MB+) 전체 in-memory 로드.
- **(d)** bbox 필드 누락/오류 silent skip.
- **(f)** `safe_path()` 미사용.
- **(g)** YOLO(0-base) ↔ COCO(1-base) category_id 자동 매핑 없음.

**[4] 개선 작업 항목**

- **P0** 경로 검증. **S**
- **P0** 좌표 반올림 형식 통일. `data_routes.py:325, 357-360`. **S**
- **P1** 이미지 크기 부재 시 경고(silent default 금지). **M**
- **P1** Segmentation polygon 보존 옵션. **L**
- **P2** category_id 재매핑 옵션. **M** / JSON 스트리밍 로드. **M**

---

### `remapper` — 클래스 ID 재매핑

**[1] 기능 / 최종 목표**

클래스 ID 일괄 변경/병합/삭제. 입력: label_dir + mapping("5:0, 3:2") + auto_reindex + recursive.

**[2] 현재 구현 구성**

- 프론트엔드 `web/js/tabs-extra.js:793-844`
- 백엔드 `server/data_routes.py:436-482`: 매핑 dict 파싱, 각 line의 cid를 dict로 치환, 매핑에 없는 클래스는 삭제

**[3] 1과 2의 괴리**

- **(a)** 순환 매핑(5→3, 3→5) 검증 없음. **auto_reindex 미구현**(파라미터만 수집, 로직 없음). "매핑에 없는 클래스 삭제"가 의도된 동작인지 모호.
- **(d)** 매핑 형식 오류("5-0") silent skip → empty mapping.
- **(f)** `safe_path()` 미사용.
- **(d) 동시성** 500/1000 진행 후 디스크 부족 → 롤백 없음.
- **(e)** 매핑 형식 직관성 낮음("5:0" CSV).

**[4] 개선 작업 항목**

- **P0** `safe_label_dir()` 검증. **S**
- **P0** auto_reindex 실제 구현. **M**
- **P1** 매핑 유효성 검증(순환/구문). **M** / 부분 실패 롤백/리포트. **M**
- **P2** 매핑 형식 개선(JSON/DSL). **M** / Dry-run 옵션. **M**

---

### `merger` — 데이터셋 병합 + 중복 제거

**[1] 기능 / 최종 목표**

여러 데이터셋 통합 + dHash로 중복 스킵. 입력: 이미지 디렉토리 N개 + threshold(0-64 bit) + recursive.

**[2] 현재 구현 구성**

- 프론트엔드 `web/js/tabs-extra.js:848-910`: Add Dataset + threshold 슬라이더(기본 10)
- 백엔드 `server/data_routes.py:494-550`: 평탄화 → dHash 계산(`core/hashing.py`) → seen_hashes O(N) 검색 → copy2 + 라벨 자동 탐색(["labels","../labels","."])

**[3] 1과 2의 괴리**

- **(a)** threshold=10 권장이지만 의미 불명확(0/5/15에서 결과 큰 변동).
- **(b) O(N²) brute-force** — 10000장 → 50M 해밍 비교(수십 초). 100000장 → 5B 비교(수분). 인덱싱(BK-tree/LSH) 부재.
- **(c)** 라벨 위치 탐색이 휴리스틱 — 비표준 구조에서 실패.
- **(d)** 이미지 읽기 실패 silent → 손실됨.
- **(f)** `safe_path()` 미사용, 동적 파일명 충돌 처리.
- **(g)** Dataset A,B에 동일 이미지 다른 라벨 → B의 라벨 손실.

**[4] 개선 작업 항목**

- **P0** 경로 검증. **S**
- **P0** O(N²) → BK-tree/VP-tree 인덱싱. `core/hashing.py`. **L**
- **P1** threshold UI 의미 표기(Strict/Normal/Relaxed). **S** / 라벨 매칭 명시화. **M** / 손상 이미지 로깅. **M**
- **P2** 원본 경로 추적(중복 시 어느 dataset 우선인지). **M** / 병렬 해싱. **L**

---

### `sampler` — 스마트 샘플러

**[1] 기능 / 최종 목표**

Random / Balanced / Stratified 표본 추출 + before/after 분포. 입력: img/label_dir + 전략 + 목표 N + seed.

**[2] 현재 구현 구성**

- 프론트엔드 `web/js/tabs-extra.js:914-980`
- 백엔드 `server/data_routes.py:582-675`:
  - Random: random.sample
  - Stratified: class_portion×target + 부족분 보충
  - Balanced: per_class=target/num_classes, farthest-point sampling(feature=박스 중심 평균 [cx,cy])

**[3] 1과 2의 괴리**

- **(a)** Balanced에서 클래스 풀이 per_class보다 작으면 전부 선택 → 총 sample이 target 미만(샘플링이 아님). Stratified의 `total_assoc` 변수명/의도 불일치(라인 624).
- **(b)** Balanced farthest-point가 그리디 O(per_class × pool_size) — KD-tree 미사용.
- **(d)** 박스 없는 이미지의 feature=[0.5,0.5] 기본값 → 모든 unlabeled 이미지가 같은 위치에 cluster → farthest-point 중복.
- **(f)** `safe_path()` 미사용.
- **(g)** `class_images.items()` 순회 순서가 dict 삽입 순서에 의존 → 재현성 미흡.

**[4] 개선 작업 항목**

- **P0** 경로 검증. **S**
- **P0** Stratified total_assoc 명확화(변수명/주석). **S**
- **P1** per_class < pool_size 경고/조정. **M** / 박스 없는 이미지 feature 처리. **M** / Balanced KD-tree. **L**
- **P2** 클래스 순회 정렬. **S** / Seed 재현성 보장 명시. **S**

---

### `augmentation` — 증강 미리보기

**[1] 기능 / 최종 목표**

Mosaic, Flip, Rotate, Brightness 등 효과 비교. 입력: img/label_dir + 증강 타입.

**[2] 현재 구현 구성**

- 프론트엔드 `web/js/tabs-extra.js:1214-1250`
- 백엔드 `server/extra_routes.py:293-324`: 동기 처리, base64 JPEG 응답
  - Flip: cv2.flip(_,1)
  - Rotate: 15도 고정
  - Brightness: alpha=1.3, beta=30 고정
  - Mosaic: 4장 2×2(실패 시 zeros 검은색 블록)

**[3] 1과 2의 괴리**

- **(a)** Mosaic 일부 이미지 읽기 실패 → 검은 블록 → 실제 mosaic과 다름. Rotate 15도/Brightness 1.3/30 hardcoded — 파라미터 조정 불가.
- **(d) 라벨 무시** — 라벨 디렉토리 받지만 증강 후 박스 좌표 변환 안 함. 사용자가 기대한 "라벨 함께 변환"이 없음.
- **(b)** 4K 이미지 → base64 → 네트워크 전송 수 초 지연.
- **(d)** 빈 디렉토리 시 `random.choice([])` AttributeError.
- **(g) Albumentations 미지원** — README에 언급되나 UI/구현 없음.

**[4] 개선 작업 항목**

- **P0** 입력 검증(빈 디렉토리). **S** / Mosaic 실패 처리 개선. **S**
- **P1** Rotate 각도/Brightness 파라미터 UI. **M** / Albumentations optional 지원. **M** / **라벨 변환(박스 좌표 업데이트)**. `core/augmentation.py` 신규. **L**
- **P2** 비동기 처리(state). **M** / Shear/Cutout/MixUp 추가. **L**

---

## 6. Quality

### `anomaly` — 라벨 이상 탐지

**[1] 기능 / 최종 목표**

YOLO 라벨의 OOB, Tiny(area<0.0001), Huge(area>0.9), Extreme aspect(>20 or <0.05) 탐지. 결과 테이블 최대 1000개.

**[2] 현재 구현 구성**

- 프론트엔드 `web/js/tabs-extra.js:984-1024`
- 백엔드 `server/quality_routes.py:45-96`: glob_images → txt 파싱 → 4규칙 평가. core 위임 없음(라우터에서 직접).

**[3] 1과 2의 괴리**

- **(a)** OOB가 정규화 좌표만 보므로 letterbox 처리 전·후 동일하게 판정(이미지 해상도 정보 없음, 문서 부족).
- **(b)** 크기 임계값 0.0001/0.9가 절대값 → 도메인별 부적합(객체 검출용 vs 분류용).
- **(c)** Extreme aspect 20/0.05가 도로 차선·텍스트 등 정상 케이스 잘못 탐지.
- **(d)** 라벨 부재/형식 오류 silent skip → 사용자는 모름.
- **(e)** 결과 1000개 hardcoded + 정렬/필터 없음.
- **(f)** running 플래그 체크가 이미지 단위만 — 이미지당 라벨 수천 개 시 취소 반응 지연.

**[4] 개선 작업 항목**

- **P0** 라벨-이미지 mismatch 오류 처리 + 손상 파일 수 보고. **S**
- **P1** OOB 판정에 이미지 해상도 정보 활용. **M** / 임계값 UI 입력. **M** / severity/type 정렬·필터. **M**
- **P2** 라인 단위 취소 신호. **S** / 1000개 상한 제거 + pagination. **M**

---

### `quality` — 이미지 품질 검사

**[1] 기능 / 최종 목표**

Laplacian variance(blur), brightness, entropy, aspect ratio. 임계값: blur<50, brightness<40 or >220, aspect>4 or <0.25.

**[2] 현재 구현 구성**

- 프론트엔드 `web/js/tabs-extra.js:1027-1065`
- 백엔드 `server/quality_routes.py:99-148`: 그레이스케일 → Laplacian.var() → brightness=mean → 히스토그램 entropy

**[3] 1과 2의 괴리**

- **(a)** **Blur threshold 50 절대값** — Laplacian variance가 카메라·해상도·콘텐츠 의존적이라 4K vs 보안카메라에서 큰 편차. Brightness 40/220 절대값도 야외/실내, 의도적 암흑/명도와 충돌.
- **(c)** Aspect 4/0.25는 문서 스캔/초상화에서 오판.
- **(d)** Entropy 결과 표시되지만 issues 판정에 미사용(중복 계산).
- **(d)** imread 실패 silent → 손상 파일 개수 미보고.
- **(e)** 결과 수치만, 임계값 설명/안내 텍스트 없음.

**[4] 개선 작업 항목**

- **P0** Blur/Brightness 임계값 사용자 입력 UI(또는 데이터셋 calibration). **M**
- **P1** Entropy 미사용 제거 또는 판정 기준 추가. **S** / imread 실패 보고. **S** / 임계값 안내 텍스트. **S**
- **P2** 히스토그램 계산 최적화. **S**

---

### `duplicate` — 근접 중복 (dHash)

**[1] 기능 / 최종 목표**

dHash 8×8(64bit) + Hamming ≤ threshold(기본 10) 쌍 검출. 결과 최대 500개.

**[2] 현재 구현 구성**

- 프론트엔드 `web/js/tabs-extra.js:1068-1108`
- 백엔드 `server/quality_routes.py:160-203`: O(N²) brute force(이중 루프), 500개 도달 시 조기 종료

**[3] 1과 2의 괴리**

- **(a) O(N²) brute-force** — N=1000 → 500k, N=5000 → 12.5M, N=10000 → 50M. 인덱싱(BK-tree/LSH) 부재 → 대규모에서 수십 초+.
- **(b)** 500개 도달 시 두 루프 break → 나머지 쌍 미탐지(사용자는 정확한 중복 수 모름).
- **(c) dHash 8×8의 변형 민감도** — 압축/약한 crop은 잡지만 90도 회전·큰 색보정은 놓침.
- **(d)** 해시 실패 silent skip.
- **(e)** Hamming threshold 가이드 없음(슬라이더 0-64만).
- **(f)** 텍스트/단색 배경 이미지에서 해시 충돌 증가.

**[4] 개선 작업 항목**

- **P0** BK-tree/LSH 인덱싱. `core/hashing.py` + `quality_routes.py:185-194`. **L**
- **P0** 500 상한 제거 + pagination. **M**
- **P1** Hamming 가이드 + dHash 크기 옵션(8 vs 16). **S+M**
- **P1** 손상 이미지 개수 보고. **S**
- **P2** phash 추가(회전 강건성). **L**

---

### `leaky` — 누수 분할 탐지

**[1] 기능 / 최종 목표**

Train/Val/Test 간 dHash 중복 쌍 검출(쌍별 통계 + 샘플 최대 10개).

**[2] 현재 구현 구성**

- 프론트엔드 `web/js/tabs-extra.js:1111-1168`
- 백엔드 `server/quality_routes.py:208-253`: 각 split별 hash dict → 쌍별 O(M×N) 비교(train×val + train×test + val×test). recursive=False.

**[3] 1과 2의 괴리**

- **(a) O(M×N) 폭발** — train 7k × val 1k × test 1k = 약 15M 비교. 인덱싱 없음.
- **(b)** N=10k 균등분할(3.3k×3) → 약 40M 비교. recursive 미지원으로 하위 디렉토리 무시.
- **(c)** 파일명 충돌 vs 내용 충돌 구분 부족.
- **(d)** None hash 처리 silent.
- **(e)** 샘플 10개 상한 → 중복 100개여도 10개만 표시.
- **(f)** 분할별 해싱 순차 처리(병렬 기회 손실).

**[4] 개선 작업 항목**

- **P0** 재귀 옵션 추가. **M**
- **P0** O(M×N) 인덱싱 최적화(BK-tree). **L**
- **P1** 샘플 상한 제거 + pagination. **M** / None hash 보고. **S** / 분할 쌍 명확화. **S**
- **P2** 병렬 해시 계산. **M**

---

### `similarity` — 유사 이미지 검색

**[1] 기능 / 최종 목표**

dHash 16×16(256bit) 쿼리 → Top-K(1-100). 출력: rank/image/distance.

**[2] 현재 구현 구성**

- 프론트엔드 `web/js/tabs-extra.js:1171-1211`
- 백엔드 `server/quality_routes.py:258-294`: 모든 이미지 해싱 → 쿼리 해싱 → `sorted(key=hamming(x,q))` → top_k

**[3] 1과 2의 괴리**

- **(a)** 매 쿼리마다 O(N log N) 재정렬(캐싱 미사용 — sim_state["index"] 선언만 있고 사용 안 함).
- **(b)** 쿼리 파일 검증 부재 — 읽기 실패 시 q_hash=0 → 의도와 다른 결과.
- **(c)** duplicate(8×8)와 similarity(16×16) 해시 크기 불일치 → 같은 쌍의 거리 다름.
- **(d)** 쿼리 미선택 시 fallback이 "첫 K개 반환"이라 의미 없음(필수 표시 부재).
- **(e)** Top-K 1-100 상한(N=100k에서 부족).
- **(f)** 정렬 단계 진행률 미보고 → "멈춘 건가?" 사용자 혼란.
- **(g)** Hamming 거리만 — 정규화/점수 안 함.

**[4] 개선 작업 항목**

- **P0** LSH/VP-tree 인덱싱 + 캐싱(sim_state["index"] 활용). **L**
- **P0** 쿼리 검증 + 에러 처리. **M**
- **P1** Top-K 상한 확대. **S** / 유사도 정규화(0-100). **S** / 정렬 진행률. **S** / 8 vs 16 dHash 일관성. **M**
- **P2** UX 개선(쿼리 필수 표시). **S**

---

## 7. Specialized / 사이드바 미노출

> **공통 가장 큰 문제**: 아래 7개 탭이 `web/js/app.js`의 `_nav` 구조에 등록되지 않아 일반 사용자가 발견 불가. JS Console에서 `App.switchTab('vlm')` 같이 호출해야 접근됨. **사이드바 노출 정책 결정이 P0**.

### `segmentation` — 세그멘테이션 평가

**[1] 기능 / 최종 목표** — Seg 모델의 mIoU/mDice + per-image 오버레이. **접근경로: 미노출.**

**[2] 현재 구현** — 프론트엔드 `tabs-extra.js:418-506`, 백엔드 `extra_routes.py:584-617` → `core.evaluation.evaluate_segmentation`. core `core/inference.py:984+ run_segmentation()` (YOLO-seg 프로토타입 마스크 + detection head). 입력 안전성: `safe_model_file()`, `safe_image_dir()` 적용됨.

**[3] 괴리**
- (a) 발견 불가(사이드바 미등록)
- (b) GT 마스크 매칭이 확장자 strict — `image.jpg` vs `image.png` 자동 매칭 안 됨
- (c) 배치 처리 부재, 고해상도 다수 → OOM 위험
- (d) running 플래그만, 타임아웃 처리 없음
- (e) 부분 실패(마스크 부재) silent skip → 평가 대상 수 불명확

**[4] 개선**
- **P0** 사이드바 노출 정책. `web/js/app.js _nav`. **M**
- **P1** GT 매칭 fuzzy(확장자 무시). `server/extra_routes.py:_run()`, `core/evaluation.py`. **S**
- **P2** 배치 처리/메모리 최적화. **L** / 타임아웃 + 부분 실패 재개. **M** / Per-image 실패 로그. **S**

---

### `clip` — CLIP 제로샷 분류

**[1]** 이미지 인코더+텍스트 인코더로 zero-shot 분류. 입력: 두 ONNX + img_dir + 레이블 CSV. 출력: per-label accuracy + Top-3 후보.

**[2]** 프론트엔드 `tabs-extra.js:509-606`, 백엔드 `extra_routes.py:22-108`, core `core/clip_inference.py`(center crop 224×224 + ImageNet norm + `simple_tokenize` + cosine softmax × 100). 텍스트 임베딩 캐싱.

**[3] 괴리**
- (a) 발견 불가
- (b) 프롬프트 템플릿("a photo of X") 미지원 — 레이블 그대로 사용해 정확도 편차
- (c) per-image 순차 — 10k+ 메모리 부하
- (e) 이미지 로드 실패 silent skip → 평가 대상 수 불명확
- (g) CLIP 모델 자동 다운로드 미지원

**[4] 개선**
- **P0** 사이드바 노출 정책. **M**
- **P1** 프롬프트 템플릿 선택 UI. **S**
- **P2** CLIP 모델 자동 다운로드(`core/hf_downloader.py` 확장). **M** / 배치 추론. **M** / 로드 실패 로깅. **S**

---

### `embedder` — 임베더 평가

**[1]** Retrieval@1/@K, 평균 cosine similarity. 이미지 비교(pairwise) 부가 기능. 입력: ONNX + img_dir(class/image.jpg) + Top-K(5).

**[2]** 프론트엔드 `tabs-extra.js:609-740`, 백엔드 `extra_routes.py:111-284`. 폴더 구조 자동 탐지 → 임베딩 추출 → L2 norm → 전체 Q×G cosine 행렬 → per-class 메트릭. 자기 자신 제외는 `sims[self_indices] = -1`.

**[3] 괴리**
- (a) 발견 불가 (문서에는 "Evaluation tab → Embedder"로 적혀 있으나 실제 사이드바에 없음 — 문서 불일치)
- (b) 자기 유사도 제외 로직이 `np.array_equal()` 사용으로 비효율 (대규모/중복 이미지에서 느림)
- (c) 배치 추론 없음
- (e) 이미지 로드 실패 silent

**[4] 개선**
- **P0** 사이드바 노출 정책. **M**
- **P1** 자기 유사도 제외 로직 hash/path 기반으로 변경. `extra_routes.py:190-204`. **S**
- **P2** 배치 추론. **M** / Top-K heapq 최적화. **S**

---

### `pose` — 포즈 추정

**[1]** 인물 박스 + 17 COCO keypoint. **접근경로 + 한계: 디렉토리 입력이지만 첫 이미지만 처리.**

**[2]** 프론트엔드 `tabs-extra.js:1254-1308`, 백엔드 `extra_routes.py:428-461` → `core/inference.py:1174-1205 run_pose()`. COCO_SKELETON 골격 시각화.

**[3] 괴리**
- (a) 발견 불가 + 단일 이미지만(디렉토리 입력 받지만 첫 장만 처리)
- (b) 키포인트 신뢰도 0.5 hardcoded
- (c) 배치 미지원
- (d) 동기 추론(블로킹)

**[4] 개선**
- **P0** 사이드바 노출 정책. **M**
- **P1** 배치 추론(디렉토리 전체). 새 엔드포인트. **L** / 신뢰도 임계값 UI. **S**
- **P2** PCK/OKS 평가 메트릭. **M**

---

### `instance-seg` — 인스턴스 분할

**[1]** 인스턴스별 마스크 + 박스. 단일 이미지 처리(pose와 동일 구조).

**[2]** 프론트엔드 `tabs-extra.js:1311-1356`, 백엔드 `extra_routes.py:465-498` → `core/inference.py:1206+ run_instance_seg()`. 인스턴스별 색상 오버레이(0.4 alpha).

**[3] 괴리** — pose와 동일 + 마스크 계산 무거움.

**[4] 개선**
- **P0** 사이드바 노출 정책. **M**
- **P1** 배치 추론. **L**
- **P2** Mask IoU / Panoptic 메트릭. **M**

---

### `tracking` — 다중 객체 추적

**[1]** ByteTrack/SORT 추적기 생성/리셋(viewer에서 사용). 독립 UI는 추적기 관리만.

**[2]** 프론트엔드 `tabs-extra.js:1359-1400`, 백엔드 `extra_routes.py:501-526`(`/api/tracking/{create,reset}`). core `core/tracking.py`: SORT(IoU≥0.3, max_age=30, min_hits=3), ByteTrack(2-stage association). Track 객체에 trajectory(최근 60 프레임).

**[3] 괴리**
- (a) 독립 탭 의미 약함 — 추적기 생성/리셋만 가능. 실제 추적은 viewer가 수행. UI 발견 가치 낮음.
- (b) IoU 매칭이 부분 폐색 시 ID 스위칭
- (c) trajectory는 60프레임 제한이나 track list 자체는 무제한 → 매우 긴 영상 + 잦은 진출입에서 메모리 누수
- (d) 같은 추적기 ID로 여러 비디오 추론 시 상태 혼돈
- (f) **사이드바 미노출 + 독립 탭으로서의 가치 자체가 낮음**

**[4] 개선**
- **P0** **사이드바 노출 vs 폐기 결정** — 단순 viewer 옵션으로 흡수가 합리적일 수도 있음. **M**
- **P1** MOTA/IDF1 메트릭 산출 기능 추가(평가 탭과 연계). **M**
- **P2** Dead track 정리(메모리 누수 방지). **S** / 동시 추적기 멀티 영상 테스트. **S**

---

### `vlm` — Vision-Language Model

**[1]** CLIP 기반 captioning / VQA / grounding(v1는 caption로 fallback). 입력: 이미지·텍스트 인코더 + img_dir + task + 프롬프트/후보 답변. **최근 추가**(commit `8f7f6ae`).

**[2]** 프론트엔드 `tabs-extra.js:2063-2235`, 백엔드 단일 `model_routes.py:248-281`, 배치 `model_routes.py:339-404`(`/api/vlm/batch`, `/status`, 최대 50장). core `core/vlm_inference.py`:
- CLIPCaptioner: 사전정의 어휘 60개 × 템플릿 3개 = 180 프롬프트, 사용자 hint를 추가 후보로 → Top-1
- CLIPVQA: "Question: ...? Answer: ..." 형식 후보들 → 유사도 → Top-1

**[3] 괴리**
- (a) 발견 불가. Grounding은 v1에서 사실상 Caption fallback
- (b) 어휘 180개 → 도메인 외(의료/산업) 표현 제한
- (c) 배치 50장 hardcoded — 큰 셋 처리 시 여러 번 실행 필요
- (d) `vlm_state.running` 플래그 → 동시 다중 배치 불가
- (g) CLIP 양 인코더 수동 준비 필요(자동 다운로드 미지원)

**[4] 개선**
- **P0** 사이드바 노출 정책. **M**
- **P1** 사용자 정의 객체 어휘 입력. **S**
- **P2** Grounding v1.1 (localization). **L** / CLIP 모델 자동 다운로드. **M** / 배치 크기 동적 조정. **S**

---

## 8. 공통 / 횡단 이슈

| 우선순위 | 작업 | 영향 범위 | 작업량 |
|---|---|---|---|
| **P0** | **사이드바 노출 정책 결정** — segmentation/clip/embedder/pose/instance-seg/tracking/vlm 7개 탭이 사이드바 미등록. 옵션: (1) 섹션 추가(Analysis/Evaluation 확장), (2) 모델 타입 감지 자동 라우팅, (3) Settings에서 수동 토글, (4) 폐기. | `web/js/app.js _nav` | **L** |
| **P0** | **상태 격리 통일** — 다수 탭(`compare_state`, `error_analysis_state`, `conf_opt_state`, `embedding_state`, `quant_state`, `opt_state`, `diag_state`)이 단일 전역 dict로 동시 요청 race 발생. job ID 기반 격리 필요. | `server/state.py` + 각 라우터 | M |
| **P0** | **경로 검증 통일** — 다수 탭(`explorer`, `splitter`, `converter`, `remapper`, `merger`, `sampler`, `analysis`)이 `safe_image_dir/label_dir/model_file` 미사용. traversal 가능성. | `server/path_safety.py`, 라우터 전반 | M |
| **P1** | **GT 부재/손상 silent skip 일관 처리** — anomaly/quality/duplicate/leaky/similarity/explorer/splitter/sampler/eval 모두 동일 패턴. 공통 helper로 missed/corrupted 카운트 + 결과에 경고. | 공통 모듈 + 각 라우터 | M |
| **P1** | **O(N²) 또는 O(N·M) 비교 인덱싱** — `merger`, `duplicate`, `leaky`, `similarity` 모두 brute-force. dHash용 BK-tree/VP-tree 공통 모듈 추가. | `core/hashing.py` 신규 인덱싱 | L |
| **P1** | **long-running task 취소 일관화** — explorer/splitter/converter/merger/sampler/eval/benchmark/diagnose/calibration 모두 running 플래그만으로 중복 차단, 사용자 취소 미지원. TaskState에 `cancel_requested` 플래그 + UI Cancel 버튼 표준화. | `server/state.py`, 모든 탭 | M |
| **P1** | **결과 상한 hardcoded 제거** — anomaly 1000, duplicate 500, leaky 샘플 10, embedder detail 200 등. pagination 또는 lazy load 도입. | 각 라우터 + 프론트엔드 | M |
| **P1** | **이미지 캐시 LRU 통일** — `evaluation` 고정 500장 등 LRU 미적용. 공통 LRU 캐시 + 메모리 한계 설정. | core 공통 유틸 | M |
| **P2** | **모델 자동 다운로드 확장** — CLIP/VLM/Embedder 모델은 사용자가 수동 다운로드. `core/hf_downloader.py` 확장. | core | L |
| **P2** | **dHash 크기 일관성** — `duplicate`(8×8) vs `similarity`(16×16) 불일치. 통일 또는 사용자 선택. | `core/hashing.py` | M |
| **P2** | **차트/시각화 표준화** — error-analyzer/conf-optimizer/quality/anomaly 모두 수치만, 차트 없거나 SVG 직접 그리기. matplotlib PNG 백엔드 + 일관 컴포넌트. | `core/diagnosis_charts.py` 재사용 | M |
| **P2** | **i18n 완성도** — findings/recommendations/일부 에러 메시지 영어 only. | i18n 전반 | M |

---

## 분석 메타데이터

- 분석 시점: 2026-05-27
- 브랜치: `feat/sequential-detection`
- 분석 범위: 30개 탭 (사이드바 등록 23 + 미등록 7)
- 방법: 섹션별 병렬 sub-agent 7개 (Explore subagent_type) — 각 에이전트가 프론트엔드 + 라우터 + core + 문서를 모두 읽고 [1]-[4] 4-section 템플릿 적용 → 본 문서에서 정합
- 우선순위는 "기능적 정확성 > 동시성·상태 > 스케일/병목 > UX > 보안 > 의존성" 순으로 부여됨
