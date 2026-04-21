# ssook — Future Works: 전체 기능 로드맵

> **v1.6.1 기준** — 현재 구현 상태를 정밀 분석하고, 모든 기능의 완성도 향상 및 신규 기능 확장 계획을 정리한 문서입니다.

---

## 1. 현재 기능 현황 총정리

### 1.1 기능 상태 매트릭스

| 섹션 | 기능 | 상태 | 완성도 | 핵심 이슈 |
|------|------|------|--------|-----------|
| **Inference** | Viewer (실시간 추론) | ✅ 정상 | 85% | 웹캠/RTSP 미지원, 영상 녹화 미지원 |
| | Settings (설정) | ✅ 정상 | 90% | 커스텀 모델 타입 UX 복잡 |
| **Evaluation** | Multi-Model Evaluation | ✅ 정상 | 75% | 중단 불가, CSV 내보내기 Detection만 대응, 혼동 행렬 없음 |
| | Benchmark | ✅ 정상 | 80% | 서버 측 중단 불가, CSV 내보내기 엔드포인트 없음 |
| **Analysis** | Inference Analysis | ✅ 정상 | 70% | 텐서 히트맵 미구현 (README에 명시됨), Detection만 지원 |
| | Model A/B Compare | ✅ 정상 | 65% | 메트릭 수준 비교 없음, 클래스별 diff 없음 |
| | FP/FN Error Analysis | ✅ 정상 | 60% | 클래스 무관 매칭, 이미지별 상세 없음, 시각적 예시 없음 |
| | Confidence Optimizer | ✅ 정상 | 70% | 최적 임계값 적용 기능 없음, 전체 클래스 최적화 없음 |
| | Embedding Visualization | ✅ 정상 | 60% | 정적 PNG만, 인터랙티브 플롯 없음, 파라미터 조정 불가 |
| **Tools** | Calibration/Quantization | ✅ 정상 | 80% | 양자화 후 정확도 검증 없음, FP16에 추가 패키지 필요 |
| | Model Inspector | ✅ 정상 | 85% | 그래프 시각화 없음, EP 호환성이 휴리스틱 기반 |
| | Model Profiler | ✅ 정상 | 80% | CPU 전용 프로파일링, GPU 프로파일링 미지원 |
| **Data** | Dataset Explorer | ✅ 정상 | 80% | 5,000장 제한, 클래스명 매핑 없음, 통계 내보내기 없음 |
| | Dataset Splitter | ✅ 정상 | 85% | 분할 전 미리보기 없음, similarity 전략 미구현 |
| | Format Converter | ✅ 정상 | 65% | VOC→YOLO, VOC→COCO 미지원, 세그멘테이션 포맷 미지원 |
| | Class Remapper | ✅ 정상 | 75% | auto_reindex 미구현, 전후 분포 미리보기 없음 |
| | Dataset Merger | ✅ 정상 | 60% | dHash 미구현 (MD5 사용), 중복 리포트 없음 |
| | Smart Sampler | ✅ 정상 | 80% | 라벨 없는 이미지 제외됨, 샘플링 미리보기 없음 |
| | Augmentation Preview | ✅ 정상 | 40% | 4종만 지원, 파라미터 조정 불가, 배치 적용 불가, 라벨 변환 없음 |
| **Quality** | Label Anomaly Detector | ✅ 정상 | 70% | 과도한 겹침 미검출, 임계값 고정, 1,000건 제한 |
| | Image Quality Checker | ✅ 정상 | 70% | 임계값 고정, 해상도 검사 없음, 손상 파일 미보고 |
| | Near-Duplicate Detector | ✅ 정상 | 65% | O(n²) 성능, 실제 그룹 클러스터링 없음, 500쌍 제한 |
| | Leaky Split Detector | ✅ 정상 | 70% | 해시 캐싱 없음, 자동 제거 없음 |
| | Similarity Search | ✅ 정상 | 60% | 해시 기반만 (임베딩 미지원), 인덱스 미캐싱 |
| **Hidden** | Pose Estimation | 🟡 beta | 50% | 단일 이미지만, 비디오/배치 미지원, 평가 메트릭 없음 |
| | Instance Segmentation | 🟡 beta | 50% | 단일 이미지만, 마스크 내보내기 없음, 평가 메트릭 없음 |
| | Object Tracking | 🟡 beta | 20% | create/reset만 가능, update 엔드포인트 없음, 실질적 미작동 |
| | VLM | 🔴 WIP | 5% | UI 플레이스홀더만, 백엔드 완전 미구현 |
| | Segmentation (독립 탭) | ⚠️ 숨김 | 70% | Evaluation 탭에 통합됨, 독립 탭은 결과 테이블 미표시 |
| | CLIP Zero-Shot (독립 탭) | ⚠️ 숨김 | 80% | Detail 버튼 중복 생성 버그 |
| | Embedder Eval (독립 탭) | ⚠️ 숨김 | 75% | 실행 전 Detail/Compare 버튼 노출 |

### 1.2 크리티컬 버그 (즉시 수정 필요)

| 파일 | 버그 | 영향 |
|------|------|------|
| `analysis_routes.py` | `import base64` 누락 | Model Compare 이미지 로드, Embedding Viewer 결과 반환 시 크래시 |
| `analysis_routes.py` | `from pathlib import Path` 누락 | Embedding Viewer 실행 시 크래시 |
| `eval_routes.py` | `eval_state` 변수 섀도잉 — import 후 재선언 | force-stop 미작동, `all_states` 레지스트리 무효화 |
| `data_routes.py` | Merger가 dHash 대신 MD5 사용 | README와 불일치, `dhash_threshold` 파라미터 무시됨 |
| `tabs-extra.js` | Leaky Split의 threshold input에 `id` 속성 없음 | 사용자 입력 무시, 항상 기본값 10 사용 |

### 1.3 크로스커팅 UX 이슈

| 이슈 | 영향 범위 | 심각도 |
|------|-----------|--------|
| 폴링 에러 무시 (`catch(e) {}`) | 13개 탭 | 🔴 높음 |
| 실행 중 Run 버튼 미비활성화 (이중 실행 위험) | 18개 탭 | 🔴 높음 |
| 서버 측 중단 미지원 (polling만 중단) | Benchmark, Evaluation, Embedding Viewer | 🟠 중간 |
| ARIA 접근성 속성 전무 | 전체 UI | 🟠 중간 |
| 파괴적 작업 확인 없음 (Split, Merge, Remap, Convert) | Data 섹션 | 🟡 낮음 |

---

## 2. 기존 기능 완성도 향상

### 2.1 Viewer (실시간 추론)

**현재 상태**: 비디오 파일 기반 MJPEG 스트리밍, Detection/Classification/Segmentation 오버레이, ByteTrack/SORT 트래커 통합, 스냅샷/크롭 저장, 프레임 스킵/시크/속도 조절.

| 개선 항목 | 설명 | 우선순위 |
|-----------|------|----------|
| 웹캠/RTSP 스트림 입력 | `cv2.VideoCapture`에 URL/디바이스 인덱스 전달, 재연결 로직 | 🔴 높음 |
| 단일 이미지 뷰어 | 이미지 로드 → 추론 → 어노테이션 결과 표시 (현재 Analysis 탭에서만 가능) | 🔴 높음 |
| 어노테이션 영상 녹화/내보내기 | 추론 결과 오버레이된 영상을 MP4로 저장 | 🔴 높음 |
| 세그멘테이션 클래스 범례 | 컬러 오버레이에 어떤 색이 어떤 클래스인지 범례 표시 | 🟠 중간 |
| Classification 시각화 강화 | 상위 K개 클래스 confidence bar 표시 (현재 텍스트만) | 🟠 중간 |
| 스트림 품질 조절 | JPEG 품질 슬라이더 (현재 65 고정) | 🟡 낮음 |
| ROI/줌 기능 | 특정 영역 확대 보기 | 🟡 낮음 |
| 동시 세션 제한 | 메모리 보호를 위한 최대 세션 수 설정 | 🟡 낮음 |
| 키보드 단축키 안내 | Space/화살표/S/+/- 등 단축키 발견 가능성 향상 (툴팁, 도움말) | 🟡 낮음 |

### 2.2 Settings (설정)

**현재 상태**: 모델 타입 관리, 테스트 모델/데이터 다운로드, 디스플레이 설정, 클래스별 스타일 관리.

| 개선 항목 | 설명 | 우선순위 |
|-----------|------|----------|
| 설정 프로필 저장/로드 | 프로젝트별 설정 프리셋 (모델 경로, 클래스 매핑 등) | 🟠 중간 |
| 클래스명 파일 임포트 | `.yaml`/`.names` 파일에서 클래스명 자동 로드 | 🟠 중간 |
| 커스텀 모델 타입 UX 개선 | 위저드 형태의 단계별 설정 (현재 한 화면에 모든 옵션) | 🟡 낮음 |

### 2.3 Multi-Model Evaluation

**현재 상태**: Detection/Classification/Segmentation/CLIP/Embedder 5개 태스크 비동기 평가, 멀티모델 비교, 클래스 매핑, CSV 내보내기.

| 개선 항목 | 설명 | 우선순위 |
|-----------|------|----------|
| 평가 중단 기능 | `eval_state` 섀도잉 버그 수정 + force-stop 연동 | 🔴 높음 |
| 혼동 행렬 (Confusion Matrix) | 클래스 간 오분류 패턴 시각화 (히트맵) | 🔴 높음 |
| 이미지별 결과 드릴다운 | 특정 이미지의 예측 vs GT 상세 보기 | 🔴 높음 |
| CSV 내보내기 태스크 적응 | Classification(Accuracy), Segmentation(mIoU), CLIP(Top-1/5), Embedder(Retrieval@K) 컬럼 | 🔴 높음 |
| COCO JSON GT 지원 | YOLO txt 외에 COCO JSON 형식 GT 직접 로드 | 🟠 중간 |
| 재귀 디렉토리 스캔 | 하위 폴더 포함 이미지 탐색 | 🟠 중간 |
| Excel 내보내기 | openpyxl 활용 (이미 선택적 의존성) | 🟠 중간 |
| 이미지별 Detection 시각화 | 평가 결과에서 특정 이미지 클릭 → 예측 박스 + GT 박스 오버레이 | 🟠 중간 |
| COCO 스타일 크기별 AP | Small/Medium/Large 객체별 AP 분리 | 🟡 낮음 |
| 멀티라벨 Classification | 파일당 복수 클래스 GT 지원 | 🟡 낮음 |

### 2.4 Benchmark

**현재 상태**: 멀티모델 FPS/레이턴시(P50/P95/P99), 전처리/추론/후처리 분리 측정, 코덱 디코드 벤치마크, CPU/RAM/GPU 모니터링.

| 개선 항목 | 설명 | 우선순위 |
|-----------|------|----------|
| 서버 측 중단 | force-stop 연동 (현재 클라이언트 폴링만 중단) | 🔴 높음 |
| CSV/Excel 내보내기 엔드포인트 | `/api/benchmark/export-csv` 구현 (현재 미존재) | 🔴 높음 |
| 멀티 해상도 벤치마크 | 320/640/1280 등 여러 입력 크기 자동 스윕 | 🟠 중간 |
| 배치 사이즈 스윕 | 최적 배치 사이즈 자동 탐색 | 🟠 중간 |
| 이전 결과와 비교 | 벤치마크 히스토리 저장 + diff 뷰 | 🟡 낮음 |
| 실제 비디오 코덱 디코드 | 더미 프레임 대신 실제 비디오 콘텐츠 사용 | 🟡 낮음 |
| 써멀 스로틀링 감지 | 장시간 벤치마크 시 성능 저하 감지 | 🟡 낮음 |

### 2.5 Analysis — Inference Analysis

**현재 상태**: 레터박스 시각화, 텐서 통계, 타이밍 분석, Detection 결과 리스트.

| 개선 항목 | 설명 | 우선순위 |
|-----------|------|----------|
| `import base64`, `from pathlib import Path` 추가 | 크래시 버그 수정 | 🔴 즉시 |
| 텐서 히트맵 시각화 | README에 명시된 기능 — 입력 텐서의 채널별 활성화 히트맵 | 🔴 높음 |
| Classification/Segmentation 분석 | Detection 외 태스크 지원 | 🟠 중간 |
| Grad-CAM 시각화 | 모델이 주목하는 영역 히트맵 (ONNX 중간 레이어 출력 활용) | 🟠 중간 |

### 2.6 Analysis — Model A/B Compare

**현재 상태**: 동일 이미지셋에 두 모델 추론, 이미지별 슬라이더 탐색, 박스 수/추론 시간 표시.

| 개선 항목 | 설명 | 우선순위 |
|-----------|------|----------|
| 메트릭 수준 비교 | mAP/Precision/Recall diff 요약 테이블 | 🔴 높음 |
| 클래스별 diff | 클래스별 성능 차이 하이라이트 | 🟠 중간 |
| 임시 파일 정리 | `ssook_compare/` 서버 종료 시 자동 삭제 | 🟠 중간 |
| 비교 결과 내보내기 | 이미지 + 메트릭 리포트 | 🟡 낮음 |

### 2.7 Analysis — FP/FN Error Analysis

**현재 상태**: FP/FN을 크기(S/M/L)와 위치(상/중/하)로 분류, IoU 기반 매칭.

| 개선 항목 | 설명 | 우선순위 |
|-----------|------|----------|
| 클래스 인식 매칭 | 현재 클래스 무관 매칭 → 동일 클래스만 매칭하도록 수정 | 🔴 높음 |
| 이미지별 FP/FN 상세 | 특정 이미지의 FP/FN 박스 시각화 (썸네일 + 오버레이) | 🔴 높음 |
| 클래스별 FP/FN 분석 | 어떤 클래스에서 FP/FN이 많은지 분석 | 🟠 중간 |
| 결과 내보내기 | FP/FN 이미지 목록 CSV, 시각화 이미지 저장 | 🟠 중간 |

### 2.8 Analysis — Confidence Optimizer

**현재 상태**: 클래스별 임계값 스윕(0.05~0.95), PR 곡선, 최적 F1/임계값 도출.

| 개선 항목 | 설명 | 우선순위 |
|-----------|------|----------|
| 최적 임계값 적용 | "Apply" 버튼으로 설정에 반영 | 🔴 높음 |
| 전체 클래스 최적화 | 개별 클래스 외에 전체 통합 최적 임계값 | 🟠 중간 |
| 결과 내보내기 | 클래스별 최적 임계값 JSON/YAML 저장 | 🟡 낮음 |

### 2.9 Analysis — Embedding Visualization

**현재 상태**: t-SNE/UMAP/PCA 2D 산점도, 서브디렉토리명 기반 라벨, matplotlib PNG.

| 개선 항목 | 설명 | 우선순위 |
|-----------|------|----------|
| 인터랙티브 플롯 | 호버 시 이미지 미리보기, 줌/패닝 (D3.js 또는 Plotly) | 🔴 높음 |
| 파라미터 조정 | t-SNE perplexity, UMAP n_neighbors 등 사용자 설정 | 🟠 중간 |
| 라벨 파일 지원 | 폴더 구조 외에 CSV/JSON 라벨 파일 | 🟡 낮음 |
| 임베딩 내보내기 | numpy/CSV 형태로 임베딩 벡터 저장 | 🟡 낮음 |

### 2.10 Tools — Calibration/Quantization

**현재 상태**: Dynamic INT8, Static INT8 (캘리브레이션), FP16 변환. 압축률 리포트, 진행률 추적.

| 개선 항목 | 설명 | 우선순위 |
|-----------|------|----------|
| 양자화 후 정확도 검증 | 원본 vs 양자화 모델 자동 비교 (mAP diff) | 🔴 높음 |
| A/B 벤치마크 연동 | "Compare in Benchmark" 버튼 → 원본+양자화 모델 자동 등록 | 🟠 중간 |
| 혼합 정밀도 양자화 | 레이어별 선택적 양자화 (민감 레이어 제외) | 🟡 낮음 |
| INT4/GPTQ/AWQ | 최신 양자화 기법 지원 | 🟡 낮음 |
| `onnxconverter-common` 자동 설치 | FP16 변환 시 패키지 없으면 안내 메시지 | 🟡 낮음 |

### 2.11 Tools — Model Inspector

**현재 상태**: 파일 정보, opset/IR 버전, I/O 텐서, 메타데이터, 노드 수, op 분포, 파라미터 수, EP 호환성.

| 개선 항목 | 설명 | 우선순위 |
|-----------|------|----------|
| 그래프 시각화 | Netron 스타일 노드 그래프 (D3.js 기반) | 🟠 중간 |
| 가중치 분포 분석 | 레이어별 가중치 히스토그램, 희소성(sparsity) 분석 | 🟠 중간 |
| EP 호환성 정확도 향상 | 휴리스틱 → 실제 EP 로드 테스트 기반 | 🟡 낮음 |
| 동적 shape 분석 | 심볼릭 차원의 유효 범위 표시 | 🟡 낮음 |

### 2.12 Tools — Model Profiler

**현재 상태**: 레이어별 타이밍, 레이턴시 통계, FLOPs/MACs 추정, 메모리 추정, 양자화 준비도, 병목 진단, 최적화 제안.

| 개선 항목 | 설명 | 우선순위 |
|-----------|------|----------|
| GPU 프로파일링 | CUDA 이벤트 기반 레이어별 GPU 시간 측정 | 🔴 높음 |
| FLOPs 정확도 향상 | Attention, LayerNorm, element-wise ops 포함 | 🟠 중간 |
| 프로파일링 비교 | 최적화 전후 비교 뷰 | 🟠 중간 |
| 표준 포맷 내보내기 | Chrome Trace, TensorBoard 호환 포맷 | 🟡 낮음 |

### 2.13 Data — Dataset Explorer

**현재 상태**: 이미지+라벨 스캔, 클래스/크기/종횡비 분포 차트, 멀티클래스 필터, 박스 수 필터, 이미지 미리보기.

| 개선 항목 | 설명 | 우선순위 |
|-----------|------|----------|
| 5,000장 제한 해제 | 페이지네이션 또는 가상 스크롤로 대규모 데이터셋 지원 | 🔴 높음 |
| 클래스명 매핑 | 숫자 ID → 이름 매핑 (`.yaml`/`.names` 파일 로드) | 🟠 중간 |
| 통계 내보내기 | 분포 차트 PNG/CSV 저장 | 🟠 중간 |
| 썸네일 캐싱 | 반복 로드 시 성능 향상 | 🟡 낮음 |
| 서버 측 정렬/필터링 | 클라이언트 부하 감소 | 🟡 낮음 |

### 2.14 Data — Dataset Splitter

**현재 상태**: Random/Stratified 분할, 커스텀 비율, 진행률 추적.

| 개선 항목 | 설명 | 우선순위 |
|-----------|------|----------|
| 분할 전 미리보기 | 예상 분포 표시 후 확인 | 🟠 중간 |
| similarity 전략 구현 | 모델 정의에 있으나 미구현 (random으로 폴스루) | 🟠 중간 |
| 출력 디렉토리 검증 | 입력과 동일 경로 방지, 기존 파일 덮어쓰기 경고 | 🟠 중간 |
| Undo/롤백 | 분할 결과 되돌리기 | 🟡 낮음 |

### 2.15 Data — Format Converter

**현재 상태**: YOLO→COCO JSON, COCO JSON→YOLO, YOLO→Pascal VOC XML.

| 개선 항목 | 설명 | 우선순위 |
|-----------|------|----------|
| VOC→YOLO 변환 | 누락된 변환 방향 | 🔴 높음 |
| VOC→COCO, COCO→VOC 변환 | 모든 방향 완전 지원 | 🔴 높음 |
| 클래스명 파일 처리 | 변환 시 `.names`/`.yaml` 파일 자동 생성/참조 | 🟠 중간 |
| 변환 결과 검증 | 변환 후 샘플 시각화로 정확성 확인 | 🟠 중간 |
| 세그멘테이션 포맷 변환 | COCO Seg ↔ 마스크 이미지 ↔ Polygon JSON | 🟡 낮음 |

### 2.16 Data — Class Remapper

**현재 상태**: 클래스 ID 리매핑, 재귀 스캔, 미매핑 클래스 제거.

| 개선 항목 | 설명 | 우선순위 |
|-----------|------|----------|
| auto_reindex 구현 | 파라미터 존재하나 로직 미구현 | 🟠 중간 |
| 전후 분포 미리보기 | 리매핑 전/후 클래스 분포 비교 | 🟠 중간 |
| 클래스명 파일 업데이트 | `.yaml`/`.names` 파일 자동 수정 | 🟡 낮음 |

### 2.17 Data — Dataset Merger

**현재 상태**: 다중 데이터셋 병합, MD5 기반 중복 제거, 파일명 충돌 처리.

| 개선 항목 | 설명 | 우선순위 |
|-----------|------|----------|
| dHash 지각 해싱 구현 | README/파라미터와 일치하도록 실제 dHash 적용 | 🔴 높음 |
| 중복 리포트 | 어떤 파일이 중복으로 제외되었는지 상세 보고 | 🟠 중간 |
| 드라이런 모드 | 실제 복사 없이 병합 결과 미리보기 | 🟠 중간 |
| 라벨 경로 탐색 강화 | 비표준 디렉토리 구조 지원 | 🟡 낮음 |

### 2.18 Data — Smart Sampler

**현재 상태**: Random/Stratified/Balanced 전략, Balanced에서 farthest-point 다양성 샘플링.

| 개선 항목 | 설명 | 우선순위 |
|-----------|------|----------|
| 임베딩 기반 다양성 | bbox 중심점 대신 이미지 임베딩 활용 | 🟠 중간 |
| 라벨 없는 이미지 처리 | 현재 무시됨 → 별도 카테고리로 포함 옵션 | 🟠 중간 |
| 샘플링 미리보기 | 예상 클래스 분포 표시 | 🟡 낮음 |

### 2.19 Data — Augmentation Preview

**현재 상태**: Flip/Rotate(15°)/Brightness/Mosaic 4종 미리보기, 랜덤 이미지 선택.

| 개선 항목 | 설명 | 우선순위 |
|-----------|------|----------|
| 파라미터 조정 | 회전 각도, 밝기 범위, 모자이크 그리드 크기 등 | 🔴 높음 |
| Albumentations 통합 | README에 명시된 기능 — 다양한 증강 기법 | 🔴 높음 |
| 배치 증강 적용 | 미리보기 → 전체 데이터셋에 적용 (이미지 + 라벨 저장) | 🔴 높음 |
| 라벨 변환 | 증강 시 bbox/polygon 좌표 자동 변환 | 🔴 높음 |
| 증강 파이프라인 정의 | 여러 증강을 순차 적용하는 파이프라인 | 🟠 중간 |
| 태스크별 프리셋 | Detection/Classification/Segmentation용 추천 증강 조합 | 🟡 낮음 |

### 2.20 Quality — Label Anomaly Detector

| 개선 항목 | 설명 | 우선순위 |
|-----------|------|----------|
| 과도한 겹침 검출 | README에 명시된 기능 — IoU 기반 과다 겹침 탐지 | 🔴 높음 |
| 임계값 사용자 설정 | 현재 하드코딩된 임계값을 UI에서 조정 | 🟠 중간 |
| 결과 제한 해제 | 1,000건 → 페이지네이션 또는 전체 결과 | 🟠 중간 |
| 빈 라벨 파일 검출 | 이미지는 있으나 라벨이 비어있는 경우 | 🟡 낮음 |
| 클래스별 이상 분석 | 특정 클래스에 이상이 집중되는지 분석 | 🟡 낮음 |

### 2.21 Quality — Image Quality Checker

| 개선 항목 | 설명 | 우선순위 |
|-----------|------|----------|
| 임계값 사용자 설정 | blur/brightness/exposure 임계값 UI 조정 | 🟠 중간 |
| 해상도 검사 | 최소/최대 해상도 기준 필터링 | 🟠 중간 |
| 손상 파일 보고 | imread 실패 시 별도 보고 (현재 무시) | 🟠 중간 |
| 결과 제한 해제 | 1,000건 → 전체 결과 | 🟡 낮음 |

### 2.22 Quality — Near-Duplicate Detector

| 개선 항목 | 설명 | 우선순위 |
|-----------|------|----------|
| 성능 최적화 | O(n²) → LSH(Locality-Sensitive Hashing) 기반 근사 탐색 | 🔴 높음 |
| 실제 그룹 클러스터링 | 현재 쌍별 → Union-Find로 그룹화 | 🟠 중간 |
| 썸네일 미리보기 | 중복 쌍의 이미지 나란히 표시 | 🟠 중간 |
| 자동 제거/이동 | 중복 파일 삭제 또는 별도 폴더 이동 | 🟡 낮음 |
| 결과 제한 해제 | 500쌍 → 전체 결과 | 🟡 낮음 |

### 2.23 Quality — Leaky Split Detector

| 개선 항목 | 설명 | 우선순위 |
|-----------|------|----------|
| threshold input id 버그 수정 | HTML `id` 속성 누락으로 사용자 입력 무시됨 | 🔴 즉시 |
| 해시 캐싱 | 반복 실행 시 재계산 방지 | 🟠 중간 |
| 자동 제거 | 누출된 중복 파일 자동 처리 옵션 | 🟡 낮음 |

### 2.24 Quality — Similarity Search

| 개선 항목 | 설명 | 우선순위 |
|-----------|------|----------|
| 임베딩 기반 검색 | dHash 외에 모델 임베딩 활용 시맨틱 유사도 | 🔴 높음 |
| 인덱스 캐싱 | 빌드된 인덱스 재사용 (현재 매번 재구축) | 🟠 중간 |
| 결과 썸네일 | 유사 이미지 미리보기 표시 | 🟠 중간 |

### 2.25 Beta — Pose Estimation 완성

**현재 상태**: 단일 이미지 YOLO-Pose 추론, COCO 17 키포인트 + 스켈레톤 시각화. 백엔드 버그 수정 완료(v1.6.1).

| 개선 항목 | 설명 | 우선순위 |
|-----------|------|----------|
| 비디오 포즈 추정 | Viewer 통합 — 실시간 포즈 오버레이 | 🔴 높음 |
| 배치 처리 | 디렉토리 내 전체 이미지 일괄 처리 | 🔴 높음 |
| 평가 메트릭 | OKS(Object Keypoint Similarity) 기반 AP, PCK | 🔴 높음 |
| 키포인트별 정확도 히트맵 | 어떤 키포인트가 부정확한지 시각화 | 🟠 중간 |
| HRNet/ViTPose 후처리 | MODEL_TYPES에 정의되어 있으나 후처리 미구현 | 🟠 중간 |
| 포즈 A/B 비교 | 두 모델의 포즈 추정 결과 비교 | 🟡 낮음 |
| 사이드바 노출 | 현재 숨김 탭 → 사이드바에 추가 | 🟠 중간 |

### 2.26 Beta — Instance Segmentation 완성

**현재 상태**: 단일 이미지 YOLO-Seg 인스턴스 마스크 오버레이. 백엔드 버그 수정 완료(v1.6.1).

| 개선 항목 | 설명 | 우선순위 |
|-----------|------|----------|
| 비디오 인스턴스 세그멘테이션 | Viewer 통합 — 실시간 마스크 오버레이 | 🔴 높음 |
| 배치 처리 | 디렉토리 내 전체 이미지 일괄 처리 | 🔴 높음 |
| 평가 메트릭 | Mask AP, Mask AR, 인스턴스별 IoU | 🔴 높음 |
| 마스크 내보내기 | RLE/Polygon/PNG 마스크 저장 | 🟠 중간 |
| Mask R-CNN 후처리 | MODEL_TYPES에 정의되어 있으나 후처리 미구현 | 🟠 중간 |
| 사이드바 노출 | 현재 숨김 탭 → 사이드바에 추가 | 🟠 중간 |

### 2.27 Beta — Object Tracking 완성

**현재 상태**: ByteTrack/SORT 트래커 생성/리셋 API만 존재. core/tracking.py에 전체 트래커 로직 구현됨. 실질적으로 미작동.

| 개선 항목 | 설명 | 우선순위 |
|-----------|------|----------|
| `/api/tracking/update` 엔드포인트 | Detection 결과를 트래커에 전달하고 트래킹 결과 반환 | 🔴 즉시 |
| Viewer 트래킹 통합 강화 | 트래킹 ID 표시, 궤적 시각화 (이미 Viewer에 부분 구현) | 🔴 높음 |
| 트래킹 평가 메트릭 | MOTA, MOTP, IDF1, HOTA, ID Switch 수 | 🔴 높음 |
| MOTChallenge 포맷 내보내기 | 트래킹 결과를 표준 포맷으로 저장 | 🟠 중간 |
| 궤적 시각화 | 전체 비디오에 걸친 객체 이동 경로 표시 | 🟠 중간 |
| Kalman Filter 추가 | 현재 순수 IoU 매칭 → 모션 예측 추가 | 🟠 중간 |
| Hungarian Algorithm | 현재 그리디 매칭 → 최적 매칭 | 🟠 중간 |
| DeepSORT/OC-SORT/BoT-SORT | 외형 특징 기반 Re-ID 트래커 | 🟡 낮음 |
| 사이드바 노출 | 현재 숨김 탭 → 사이드바에 추가 | 🟠 중간 |

### 2.28 WIP — VLM (Vision-Language Model) 구현

**현재 상태**: UI 플레이스홀더만 존재 (render만 있고 init/run 없음). 백엔드 API 없음. MODEL_TYPES에 vlm_vqa/vlm_caption/vlm_grounding 정의됨.

| 구현 항목 | 설명 | 우선순위 |
|-----------|------|----------|
| VQA (Visual Question Answering) | 이미지 + 질문 → 답변 생성 (ViLT, BLIP 등 ONNX) | 🔴 높음 |
| Image Captioning | 이미지 → 설명 텍스트 생성 (BLIP, GIT 등 ONNX) | 🔴 높음 |
| `/api/infer/vlm` 엔드포인트 | VQA/Captioning/Grounding 통합 추론 API | 🔴 높음 |
| VLM 탭 UI 완성 | Run 버튼, 결과 표시, 히스토리 | 🔴 높음 |
| Visual Grounding | 텍스트 설명 → 이미지 내 영역 지정 | 🟠 중간 |
| Open-Vocabulary Detection | Grounding DINO 등 텍스트 기반 객체 검출 | 🟡 낮음 |
| 멀티턴 대화형 분석 | 이미지에 대한 연속 질의응답 | 🟡 낮음 |
| 사이드바 노출 | 현재 숨김 탭 → 사이드바에 추가 | 🔴 높음 |

---

## 3. 신규 기능 추가

### 3.1 신규 모델 태스크

#### 3.1.1 OCR (텍스트 검출 + 인식)
- Text Detection 모델 (CRAFT, DBNet 등 ONNX)
- Text Recognition 모델 (CRNN, TrOCR 등 ONNX)
- Detection + Recognition 파이프라인 연결
- 검출된 텍스트 영역 시각화 + 인식 결과 오버레이
- 평가 메트릭: Detection (IoU, P/R), Recognition (CER, WER, Accuracy)

#### 3.1.2 Depth Estimation (단안 깊이 추정)
- MiDaS, Depth Anything 등 ONNX 모델 지원
- 깊이맵 컬러맵 시각화 (viridis, plasma 등)
- 원본 이미지와 깊이맵 나란히 비교
- 상대/절대 깊이 스케일 조정

#### 3.1.3 Anomaly Detection (시각적 이상 탐지)
- 비지도 학습 기반 시각적 이상 탐지 (PatchCore, PaDiM 등)
- 정상 이미지 학습 → 이상 영역 히트맵 시각화
- 평가 메트릭: AUROC, AUPRO, F1-max
- 산업용 결함 검출 시나리오 지원

#### 3.1.4 Super Resolution
- ESRGAN, Real-ESRGAN 등 ONNX 모델 지원
- 원본 vs 업스케일 비교 뷰어
- 배율 선택 (2x, 4x, 8x)
- 품질 메트릭: PSNR, SSIM, LPIPS

#### 3.1.5 Face Detection & Recognition
- Face Detection (RetinaFace, SCRFD 등)
- Face Landmark Detection (5점, 68점)
- Face Recognition / Verification (ArcFace 등)
- 평가 메트릭: TAR@FAR, ROC 곡선

#### 3.1.6 Image Generation / Diffusion
- Stable Diffusion ONNX 파이프라인 지원
- Text-to-Image, Image-to-Image, Inpainting
- 생성 결과 갤러리 + 프롬프트 히스토리
- 품질 메트릭: FID, CLIP Score

#### 3.1.7 Video Understanding
- Action Recognition (행동 인식) — 비디오 클립 분류
- Temporal Action Detection — 시간 구간별 행동 검출
- 비디오 요약 (key frame extraction)

#### 3.1.8 Optical Flow
- RAFT, FlowNet 등 ONNX 모델 지원
- 플로우 필드 시각화 (화살표, 컬러 코딩)
- 비디오 프레임 간 모션 분석

#### 3.1.9 3D Vision
- Point Cloud 시각화 (3D 뷰어)
- 3D Object Detection (PointPillars 등)
- Stereo Depth Estimation
- 3D BEV (Bird's Eye View) 시각화

### 3.2 미지원 Detection 모델 후처리 구현

**현재 상태**: `MODEL_TYPES`에 정의되어 있으나 전용 후처리가 없는 모델들 (v5/v8 제네릭 경로로 폴스루).

| 모델 | 필요한 후처리 | 우선순위 |
|------|--------------|----------|
| YOLOv10 | NMS-free 후처리 (내장 NMS) | 🔴 높음 |
| YOLOX | 디코딩 + NMS (그리드 기반 출력) | 🟠 중간 |
| DAMO-YOLO | 전용 디코딩 | 🟡 낮음 |
| Gold-YOLO | 전용 디코딩 | 🟡 낮음 |
| EfficientDet | 앵커 기반 디코딩 + NMS | 🟡 낮음 |

### 3.3 데이터 관리 확장

#### 3.3.1 어노테이션 도구 (라벨링)
- Bounding Box 어노테이션 (드래그 & 드롭)
- Polygon / Freehand 세그멘테이션 마스크 어노테이션
- 키포인트 어노테이션
- 클래스 라벨 관리 (추가/삭제/색상 지정)
- 단축키 기반 빠른 라벨링 워크플로우
- 어노테이션 히스토리 (Undo/Redo)

#### 3.3.2 Auto-Labeling (자동 라벨링)
- 로드된 모델을 사용한 자동 라벨 생성
- 자동 라벨 → 수동 검수 워크플로우
- Confidence 기반 자동 라벨 필터링
- SAM (Segment Anything Model) 통합 — 클릭/박스 기반 자동 세그멘테이션

#### 3.3.3 Active Learning
- 불확실성 기반 샘플 추천 (라벨링 우선순위)
- 모델 예측 엔트로피 시각화
- 라벨링 → 재학습 → 재평가 루프 지원
- 데이터 효율성 곡선 (라벨 수 vs 성능)

#### 3.3.4 키포인트 데이터 관리
- COCO Keypoint 포맷 지원
- 키포인트 시각화 + 편집
- 스켈레톤 정의 커스터마이징

#### 3.3.5 비디오 어노테이션
- 프레임 단위 어노테이션
- 트래킹 ID 할당 + 보간 (interpolation)
- 시간 구간 라벨링 (Action Detection용)
- 비디오 → 프레임 추출 + 자동 라벨링

#### 3.3.6 데이터셋 버전 관리
- 데이터셋 스냅샷 / 버전 태깅
- 버전 간 diff (추가/삭제/변경된 이미지 및 라벨)
- 버전별 평가 결과 비교

#### 3.3.7 클래스 불균형 분석 & 리밸런싱
- 클래스별 샘플 수 / 박스 수 불균형 시각화 (Explorer에서 부분 지원)
- 자동 오버샘플링 / 언더샘플링 제안
- 클래스 가중치 계산 (학습용 weight 파일 생성)
- 불균형 해소 전후 분포 비교

#### 3.3.8 멀티모달 데이터 지원
- 이미지 + 텍스트 페어 데이터 관리 (VLM/CLIP 학습용)
- 이미지 + 캡션 데이터셋 탐색기
- Point Cloud 데이터 로더 (3D Vision용)

### 3.4 평가 & 분석 확장

#### 3.4.1 Cross-Domain 평가
- 도메인 시프트 분석 (학습 데이터 vs 테스트 데이터 분포 비교)
- 환경별 성능 분석 (주간/야간, 실내/실외, 날씨 등)
- 데이터 서브셋별 성능 브레이크다운

#### 3.4.2 Robustness 테스트
- 이미지 corruption 테스트 (노이즈, 블러, 날씨 효과 등)
- Corruption 수준별 성능 저하 곡선
- 입력 해상도별 성능 변화 분석

#### 3.4.3 Fairness & Bias 분석
- 속성별 성능 분석 (객체 크기, 종횡비, 밀도 등)
- 클래스 간 혼동 행렬 시각화
- 오분류 패턴 분석

#### 3.4.4 모델 해석 (Explainability)
- Grad-CAM / Grad-CAM++ 시각화
- Attention Map 시각화 (Transformer 기반 모델)
- 클래스별 활성화 영역 비교

#### 3.4.5 리포트 생성
- PDF/HTML 평가 리포트 자동 생성
- 차트, 테이블, 샘플 이미지 포함
- 커스텀 리포트 템플릿
- 모델 카드 (Model Card) 생성

---

## 4. UX & 인프라 개선

### 4.1 프론트엔드 안정성

| 개선 항목 | 설명 | 우선순위 |
|-----------|------|----------|
| 폴링 에러 핸들링 | 13개 탭의 빈 `catch(e) {}` → 에러 로깅 + 재시도 + 사용자 알림 | 🔴 즉시 |
| Run 버튼 비활성화 | 18개 탭에서 실행 중 이중 클릭 방지 | 🔴 즉시 |
| 서버 측 중단 연동 | Benchmark, Evaluation, Embedding Viewer에 force-stop 호출 추가 | 🔴 높음 |
| force-stop 협력 | 백그라운드 태스크에서 `running` 플래그 주기적 확인 | 🔴 높음 |
| CLIP Detail 버튼 중복 | 반복 실행 시 버튼 누적 → 기존 버튼 제거 후 추가 | 🟠 중간 |
| Embedder 버튼 사전 노출 | 실행 전 Detail/Compare 버튼 숨김 처리 | 🟠 중간 |
| Converter 완료 메시지 | `JSON.stringify(s.results)` → 사용자 친화적 메시지 | 🟡 낮음 |

### 4.2 접근성 (Accessibility)

| 개선 항목 | 설명 | 우선순위 |
|-----------|------|----------|
| ARIA 속성 추가 | `role`, `aria-label`, `aria-live`, `aria-valuenow` 전체 적용 | 🟠 중간 |
| 프로그레스 바 접근성 | `role="progressbar"` + `aria-valuenow/min/max` | 🟠 중간 |
| 모달 접근성 | `role="dialog"`, `aria-modal`, 포커스 트랩, 포커스 복원 | 🟠 중간 |
| 키보드 네비게이션 | 사이드바 `<div>` → `<button>` 또는 `tabindex` + 키 핸들러 | 🟠 중간 |
| 스킵 네비게이션 링크 | 메인 콘텐츠로 바로 이동 | 🟡 낮음 |
| 고대비 모드 | 시각 장애 사용자를 위한 고대비 테마 | 🟡 낮음 |

### 4.3 사용자 경험 향상

| 개선 항목 | 설명 | 우선순위 |
|-----------|------|----------|
| 파괴적 작업 확인 | Split/Merge/Remap/Convert 실행 전 확인 다이얼로그 | 🔴 높음 |
| 작업 히스토리 | 완료된 작업 결과 유지 (현재 진행률 바 사라짐) | 🟠 중간 |
| 파일 브라우저 개선 | 브레드크럼 네비게이션, 최근 경로, 드래그 앤 드롭 | 🟠 중간 |
| 키보드 단축키 문서화 | 도움말 오버레이에 단축키 목록 | 🟠 중간 |
| 클래스 매핑 프리셋 | 자주 사용하는 매핑 저장/로드 | 🟡 낮음 |
| 다크/라이트 테마 전환 안정화 | 사이드바 전체 리렌더 → 부분 업데이트 (깜빡임 방지) | 🟡 낮음 |
| 알림 히스토리 | 토스트 알림 로그 보기 | 🟡 낮음 |

### 4.4 다국어 지원 강화

| 개선 항목 | 설명 | 우선순위 |
|-----------|------|----------|
| 일본어 추가 | 일본 시장 대응 | 🟡 낮음 |
| 중국어 (간체/번체) 추가 | 중국 시장 대응 | 🟡 낮음 |
| 커뮤니티 번역 기여 시스템 | JSON 기반 번역 파일 분리 + 기여 가이드 | 🟡 낮음 |

### 4.5 모델 허브 & 변환

| 개선 항목 | 설명 | 우선순위 |
|-----------|------|----------|
| HF 다운로드 진행률 | 현재 진행률 미표시 → 프로그레스 바 | 🟠 중간 |
| 모델 카드/README 표시 | 다운로드 전 모델 정보 확인 | 🟠 중간 |
| 다운로드 후 자동 타입 감지 | 모델 메타데이터에서 태스크 타입 추론 | 🟡 낮음 |
| ONNX Model Zoo 브라우저 | HF 외에 ONNX 공식 모델 저장소 탐색 | 🟡 낮음 |
| PyTorch → ONNX 변환 | 앱 내 `.pt` → `.onnx` 변환 (ultralytics 연동) | 🟡 낮음 |
| ONNX → TensorRT 변환 | TensorRT 엔진 빌드 | 🟡 낮음 |

### 4.6 실험 추적 & 자동화

| 개선 항목 | 설명 | 우선순위 |
|-----------|------|----------|
| 평가 히스토리 자동 저장 | 실행 결과를 로컬 DB/JSON에 누적 | 🟠 중간 |
| 실험 간 비교 대시보드 | 여러 평가 결과를 한 화면에서 비교 | 🟠 중간 |
| CLI 배치 평가 | `python -m ssook evaluate --config eval.yaml` (headless) | 🟠 중간 |
| CI/CD 통합 | GitHub Actions에서 모델 평가 자동 실행 | 🟡 낮음 |
| MLflow/W&B 연동 | 외부 실험 추적 도구 연동 (선택적) | 🟡 낮음 |

### 4.7 시스템 & 인프라

| 개선 항목 | 설명 | 우선순위 |
|-----------|------|----------|
| AMD/Intel GPU 모니터링 | 현재 NVIDIA만 지원 (nvidia-smi) | 🟠 중간 |
| 디렉토리 선택 다이얼로그 | 현재 파일 선택만 가능 (`/api/fs/select`) | 🟠 중간 |
| `/api/fs/browse` 구현 | 프론트엔드에서 호출하나 백엔드 미존재 | 🟠 중간 |
| `/api/fs/list` 구현 | Viewer에서 호출하나 백엔드 미존재 | 🟠 중간 |
| 디스크 공간 정보 | 대용량 데이터셋 작업 시 공간 부족 사전 경고 | 🟡 낮음 |
| 리소스 사용량 히스토리 | CPU/GPU/RAM 시계열 그래프 | 🟡 낮음 |
| 플러그인 시스템 | 커스텀 전처리/후처리/메트릭/시각화 플러그인 | 🟡 낮음 |
| 원격 추론 | SSH/REST API로 원격 서버 연결, Triton 연동 | 🟡 낮음 |

---

## 5. 우선순위 로드맵

### Phase 0 — 즉시 수정 (버그 & 안정성)

> 기존 기능의 정상 동작을 보장하는 긴급 수정 사항.

| 항목 | 파일 | 작업 |
|------|------|------|
| `import base64` 추가 | `analysis_routes.py` | Model Compare/Embedding Viewer 크래시 수정 |
| `from pathlib import Path` 추가 | `analysis_routes.py` | Embedding Viewer 크래시 수정 |
| `eval_state` 섀도잉 제거 | `eval_routes.py` | force-stop 연동 복구 |
| Merger dHash 구현 | `data_routes.py` | MD5 → dHash 교체, `dhash_threshold` 파라미터 활성화 |
| Leaky threshold input id 추가 | `tabs-extra.js` | 사용자 입력 반영 |
| 폴링 에러 핸들링 | `tabs.js`, `tabs-extra.js` | 13개 빈 catch 블록에 에러 처리 추가 |
| Run 버튼 비활성화 | `tabs.js`, `tabs-extra.js` | 18개 탭에 실행 중 비활성화 로직 |

### Phase 1 — 핵심 완성 (기존 기능 100% 달성)

> README에 명시된 모든 기능이 실제로 동작하도록 완성.

| 기능 | 핵심 작업 | 난이도 |
|------|-----------|--------|
| Tracking 완성 | `/api/tracking/update` 구현, Viewer 통합 | ★★☆ |
| VLM 구현 | VQA/Captioning 백엔드 + UI 완성 | ★★★ |
| Augmentation 강화 | 파라미터 조정, 배치 적용, 라벨 변환 | ★★☆ |
| Format Converter 완성 | VOC↔YOLO, VOC↔COCO 모든 방향 | ★★☆ |
| 텐서 히트맵 | Inference Analysis에 히트맵 시각화 추가 | ★★☆ |
| 과도한 겹침 검출 | Label Anomaly Detector에 IoU 기반 겹침 탐지 | ★☆☆ |
| Pose/Instance-Seg 배치 | 디렉토리 일괄 처리 + 비디오 지원 | ★★☆ |
| 혼동 행렬 | Evaluation에 Confusion Matrix 추가 | ★★☆ |
| 평가 중단 | eval_state 수정 + force-stop 연동 | ★☆☆ |
| Benchmark CSV 내보내기 | `/api/benchmark/export-csv` 구현 | ★☆☆ |

### Phase 2 — 사용성 증대 (UX & 분석 강화)

> 사용자 경험을 크게 향상시키는 기능.

| 기능 | 핵심 작업 | 난이도 |
|------|-----------|--------|
| 이미지별 결과 드릴다운 | Evaluation에서 이미지 클릭 → 상세 보기 | ★★☆ |
| 인터랙티브 임베딩 플롯 | D3.js/Plotly 기반 호버/줌 | ★★☆ |
| 웹캠/RTSP 입력 | Viewer에 스트림 소스 추가 | ★★☆ |
| 영상 녹화/내보내기 | 어노테이션 오버레이 MP4 저장 | ★★☆ |
| 파괴적 작업 확인 | Split/Merge/Remap/Convert 확인 다이얼로그 | ★☆☆ |
| 접근성 (ARIA) | 전체 UI에 ARIA 속성 추가 | ★★☆ |
| 에러 분석 시각화 | FP/FN 이미지별 썸네일 + 오버레이 | ★★☆ |
| 최적 임계값 적용 | Conf Optimizer → Settings 연동 | ★☆☆ |
| 양자화 후 정확도 검증 | 원본 vs 양자화 자동 비교 | ★★☆ |
| 임베딩 기반 유사도 검색 | dHash → 모델 임베딩 활용 | ★★☆ |

### Phase 3 — 태스크 확장 (신규 모델 지원)

| 기능 | 핵심 작업 | 난이도 |
|------|-----------|--------|
| Pose 평가 메트릭 | OKS, PCK, AP 구현 | ★★★ |
| Instance-Seg 평가 메트릭 | Mask AP, Mask AR 구현 | ★★★ |
| Tracking 평가 메트릭 | MOTA, MOTP, IDF1, HOTA 구현 | ★★★ |
| YOLOv10 후처리 | NMS-free 디코딩 | ★★☆ |
| YOLOX 후처리 | 그리드 기반 디코딩 | ★★☆ |
| HRNet/ViTPose 후처리 | 히트맵 기반 키포인트 디코딩 | ★★★ |
| Mask R-CNN 후처리 | ROI 기반 마스크 디코딩 | ★★★ |
| OCR 파이프라인 | Detection + Recognition 통합 | ★★★ |
| Depth Estimation | 깊이맵 추론 + 시각화 | ★★☆ |

### Phase 4 — 데이터 워크플로우

| 기능 | 핵심 작업 | 난이도 |
|------|-----------|--------|
| 어노테이션 도구 | 웹 기반 bbox/polygon/keypoint 라벨링 | ★★★ |
| Auto-Labeling | 모델 추론 → 라벨 생성 → 검수 워크플로우 | ★★★ |
| 세그멘테이션 포맷 변환 | COCO Seg ↔ 마스크 이미지 ↔ Polygon | ★★☆ |
| 데이터셋 버전 관리 | 스냅샷, diff, 버전별 평가 비교 | ★★★ |
| 리포트 생성 | PDF/HTML 자동 생성 | ★★☆ |

### Phase 5 — 고급 분석 & 생태계

| 기능 | 핵심 작업 | 난이도 |
|------|-----------|--------|
| Grad-CAM | ONNX 중간 레이어 출력 기반 히트맵 | ★★★ |
| Robustness 테스트 | 이미지 corruption 자동 적용 + 성능 곡선 | ★★★ |
| GPU 프로파일링 | CUDA 이벤트 기반 레이어별 GPU 시간 | ★★★ |
| 실험 추적 | 평가 히스토리 + 비교 대시보드 | ★★☆ |
| CLI 배치 평가 | headless 모드 + YAML 설정 | ★★☆ |
| 플러그인 시스템 | 커스텀 전처리/후처리/메트릭 확장 | ★★★ |
| 원격 추론 | SSH/REST/Triton 연동 | ★★★ |

---

## 6. 현재 지원 현황 요약

### 6.1 태스크별 지원 현황

| 태스크 | 추론 | 평가 | 비디오 | 배치 | 상태 |
|--------|------|------|--------|------|------|
| Detection (YOLO v5/v8/v9/v11) | ✅ | ✅ mAP/P/R/F1 | ✅ | ✅ | 정상 |
| Detection (DETR/RT-DETR) | ✅ | ✅ | ✅ | ❌ | 정상 |
| Detection (CenterNet/Darknet) | ✅ | ✅ | ✅ | ✅ | 정상 |
| Detection (YOLOv10) | ⚠️ 제네릭 | ✅ | ✅ | ❌ | 후처리 필요 |
| Detection (YOLOX) | ⚠️ 제네릭 | ✅ | ✅ | ❌ | 후처리 필요 |
| Classification | ✅ | ✅ Acc/P/R/F1 | ✅ | ❌ | 정상 |
| Semantic Segmentation | ✅ | ✅ mIoU/mDice | ✅ | ❌ | 정상 |
| Instance Segmentation | ✅ YOLO만 | ❌ | ❌ | ❌ | 🟡 beta |
| Pose Estimation | ✅ YOLO만 | ❌ | ❌ | ❌ | 🟡 beta |
| Object Tracking | ⚠️ API만 | ❌ | ❌ | ❌ | 🟡 beta |
| CLIP Zero-Shot | ✅ | ✅ Top-1/5 | ❌ | ✅ | 정상 |
| Embedder | ✅ | ✅ Ret@K/Cosine | ❌ | ✅ | 정상 |
| VLM (VQA/Caption) | ❌ | ❌ | ❌ | ❌ | 🔴 WIP |
| OCR | ❌ | ❌ | ❌ | ❌ | 미계획 |
| Depth Estimation | ❌ | ❌ | ❌ | ❌ | 미계획 |

### 6.2 도구별 지원 현황

| 도구 | 상태 | 완성도 |
|------|------|--------|
| Model Inspector | ✅ | 85% |
| Model Profiler | ✅ | 80% |
| Quantization (Dynamic/Static/FP16) | ✅ | 80% |
| HuggingFace Hub 연동 | ✅ | 75% |
| Dataset Explorer | ✅ | 80% |
| Dataset Splitter | ✅ | 85% |
| Format Converter | ✅ | 65% |
| Class Remapper | ✅ | 75% |
| Dataset Merger | ✅ | 60% |
| Smart Sampler | ✅ | 80% |
| Augmentation Preview | ✅ | 40% |
| Label Anomaly Detector | ✅ | 70% |
| Image Quality Checker | ✅ | 70% |
| Near-Duplicate Detector | ✅ | 65% |
| Leaky Split Detector | ✅ | 70% |
| Similarity Search | ✅ | 60% |

### 6.3 인프라 지원 현황

| 항목 | 상태 |
|------|------|
| ONNX Runtime (CPU) | ✅ |
| ONNX Runtime (CUDA) | ✅ |
| ONNX Runtime (TensorRT) | ✅ |
| ONNX Runtime (DirectML) | ✅ |
| ONNX Runtime (OpenVINO) | ✅ |
| ONNX Runtime (CoreML) | ✅ |
| EP venv 격리 | ✅ |
| 다국어 (한국어/영어) | ✅ |
| 다크 모드 | ✅ |
| pywebview 네이티브 윈도우 | ✅ |
| PyInstaller 빌드 (Windows/macOS) | ✅ |
| GitHub Actions CI/CD | ✅ |
| 접근성 (ARIA) | ❌ |
| 키보드 네비게이션 | ⚠️ 부분 |
| CLI 배치 모드 | ❌ |
| 플러그인 시스템 | ❌ |
| 원격 추론 | ❌ |

---

---

## 7. TASKS — 실행 가능한 작업 목록

> 개발자가 바로 착수할 수 있는 구체적 작업 단위. 각 TASK는 하나의 PR로 완결 가능한 범위.

### 🔴 CRITICAL (즉시)

| ID | TASK | 파일 | 예상 시간 |
|----|------|------|-----------|
| T-001 | `import base64` 추가 | `server/analysis_routes.py` | 5분 |
| T-002 | `from pathlib import Path` 추가 | `server/analysis_routes.py` | 5분 |
| T-003 | `eval_state` 섀도잉 제거 (import 후 재선언 삭제) | `server/eval_routes.py` | 15분 |
| T-004 | Merger: MD5 → dHash 교체, `dhash_threshold` 파라미터 활성화 | `server/data_routes.py` | 2시간 |
| T-005 | Leaky Split: threshold input에 `id="leaky-threshold"` 추가 | `web/js/tabs-extra.js` | 5분 |
| T-006 | 13개 탭 폴링 빈 `catch(e) {}` → 에러 로깅 + 재시도 로직 | `web/js/tabs.js`, `tabs-extra.js` | 2시간 |
| T-007 | 18개 탭 Run 버튼 실행 중 `disabled` 처리 | `web/js/tabs.js`, `tabs-extra.js` | 2시간 |

### 🟠 HIGH (Phase 0~1)

| ID | TASK | 파일 | 예상 시간 |
|----|------|------|-----------|
| T-010 | `/api/tracking/update` 엔드포인트 구현 | `server/extra_routes.py` | 3시간 |
| T-011 | Evaluation force-stop 연동 (서버 측 중단) | `server/eval_routes.py`, `tabs.js` | 2시간 |
| T-012 | Benchmark force-stop + `/api/benchmark/export-csv` 구현 | `server/benchmark_routes.py`, `tabs.js` | 3시간 |
| T-013 | Error Analysis: 클래스 인식 매칭으로 수정 | `server/analysis_routes.py` | 2시간 |
| T-014 | Evaluation: Confusion Matrix 데이터 생성 + UI 히트맵 | `server/eval_routes.py`, `tabs.js` | 4시간 |
| T-015 | Evaluation: CSV 내보내기 태스크별 컬럼 적응 | `server/eval_routes.py` | 2시간 |
| T-016 | Inference Analysis: 텐서 히트맵 시각화 구현 | `server/analysis_routes.py` | 4시간 |
| T-017 | Model Compare: 메트릭 수준 비교 (mAP diff 요약) | `server/analysis_routes.py`, `tabs-extra.js` | 4시간 |
| T-018 | Conf Optimizer: "Apply" 버튼 → Settings 반영 | `tabs-extra.js`, `server/config_routes.py` | 2시간 |
| T-019 | Format Converter: VOC→YOLO, VOC→COCO, COCO→VOC 추가 | `server/data_routes.py` | 4시간 |
| T-020 | Augmentation: 파라미터 조정 UI + 배치 적용 + 라벨 변환 | `server/extra_routes.py`, `tabs-extra.js` | 8시간 |
| T-021 | Label Anomaly: 과도한 겹침(IoU) 검출 추가 | `server/quality_routes.py` | 2시간 |
| T-022 | Explorer: 5,000장 제한 → 페이지네이션 | `server/data_routes.py`, `tabs.js` | 4시간 |
| T-023 | Near-Duplicate: O(n²) → LSH 기반 근사 탐색 | `server/quality_routes.py` | 4시간 |
| T-024 | Embedding Viewer: 인터랙티브 플롯 (D3.js/Plotly) | `tabs-extra.js`, `analysis_routes.py` | 6시간 |
| T-025 | Similarity Search: 임베딩 기반 시맨틱 검색 추가 | `server/quality_routes.py` | 4시간 |

### 🟡 MEDIUM (Phase 1~2)

| ID | TASK | 파일 | 예상 시간 |
|----|------|------|-----------|
| T-030 | VLM: `/api/infer/vlm` 엔드포인트 + VQA/Captioning 구현 | `server/extra_routes.py`, `core/` | 16시간 |
| T-031 | VLM: 탭 UI 완성 (Run, 결과 표시, 히스토리) | `tabs-extra.js` | 4시간 |
| T-032 | Viewer: 웹캠/RTSP 스트림 입력 지원 | `server/viewer_routes.py` | 4시간 |
| T-033 | Viewer: 어노테이션 영상 MP4 녹화/내보내기 | `server/viewer_routes.py` | 6시간 |
| T-034 | Pose/Instance-Seg: 배치 처리 + 비디오 지원 | `server/extra_routes.py` | 6시간 |
| T-035 | Evaluation: 이미지별 결과 드릴다운 (예측 vs GT 오버레이) | `server/eval_routes.py`, `tabs.js` | 6시간 |
| T-036 | Error Analysis: 이미지별 FP/FN 썸네일 시각화 | `server/analysis_routes.py`, `tabs-extra.js` | 4시간 |
| T-037 | Quantization: 양자화 후 정확도 자동 검증 | `server/extra_routes.py` | 4시간 |
| T-038 | 파괴적 작업 확인 다이얼로그 (Split/Merge/Remap/Convert) | `tabs.js`, `tabs-extra.js` | 2시간 |
| T-039 | ARIA 접근성 속성 전체 적용 | `web/js/*.js` | 8시간 |
| T-040 | `/api/fs/browse`, `/api/fs/list` 엔드포인트 구현 | `server/system_routes.py` | 3시간 |
| T-041 | Pose/Instance-Seg/Tracking/VLM 사이드바 노출 | `web/js/app.js` | 1시간 |
| T-042 | CLIP Detail 버튼 중복 생성 수정 | `tabs-extra.js` | 30분 |
| T-043 | Embedder Detail/Compare 버튼 실행 전 숨김 | `tabs-extra.js` | 30분 |
| T-044 | YOLOv10 NMS-free 후처리 구현 | `core/inference.py` | 3시간 |
| T-045 | Profiler: GPU 프로파일링 (CUDA 이벤트) | `core/model_profiler.py` | 8시간 |

---

## 8. TODOS — 완료 추적 체크리스트

> 체크박스 형태로 진행 상황을 추적. `[x]`로 완료 표시.

### 버그 수정
- [ ] `analysis_routes.py` — `import base64` 추가 (T-001)
- [ ] `analysis_routes.py` — `from pathlib import Path` 추가 (T-002)
- [ ] `eval_routes.py` — `eval_state` 섀도잉 제거 (T-003)
- [ ] `data_routes.py` — Merger dHash 실제 구현 (T-004)
- [ ] `tabs-extra.js` — Leaky threshold input id 추가 (T-005)
- [x] `extra_routes.py` — `_load` → `_load_model` 수정 (v1.6.1)
- [x] `extra_routes.py` — `import uuid` 추가 (v1.6.1)
- [x] `extra_routes.py` — `_generate_palette` → `generate_palette` 수정 (v1.6.1)
- [x] `extra_routes.py` — `import asyncio` 추가 (v1.6.1)
- [x] `extra_routes.py` — `run_inexecutor` → `run_in_executor` 수정 (v1.6.1)

### 프론트엔드 안정성
- [ ] 13개 탭 폴링 에러 핸들링 (T-006)
- [ ] 18개 탭 Run 버튼 비활성화 (T-007)
- [ ] CLIP Detail 버튼 중복 생성 수정 (T-042)
- [ ] Embedder 버튼 사전 노출 수정 (T-043)
- [ ] Converter 완료 메시지 사용자 친화적으로 변경
- [ ] Benchmark/Evaluation/Embedding Viewer 서버 측 중단 연동

### 기존 기능 완성
- [ ] `/api/tracking/update` 엔드포인트 구현 (T-010)
- [ ] Evaluation Confusion Matrix (T-014)
- [ ] Evaluation CSV 태스크별 적응 (T-015)
- [ ] Inference Analysis 텐서 히트맵 (T-016)
- [ ] Model Compare 메트릭 비교 (T-017)
- [ ] Conf Optimizer "Apply" 버튼 (T-018)
- [ ] Format Converter 모든 방향 지원 (T-019)
- [ ] Augmentation 파라미터/배치/라벨 (T-020)
- [ ] Label Anomaly 과도한 겹침 검출 (T-021)
- [ ] Explorer 페이지네이션 (T-022)
- [ ] Benchmark CSV 내보내기 (T-012)
- [ ] Error Analysis 클래스 인식 매칭 (T-013)

### Beta → 정식
- [ ] Pose: 비디오 + 배치 + 평가 메트릭 (T-034)
- [ ] Instance-Seg: 비디오 + 배치 + 평가 메트릭 (T-034)
- [ ] Tracking: update API + Viewer 통합 + 평가 메트릭 (T-010)
- [ ] Pose/Instance-Seg/Tracking 사이드바 노출 (T-041)

### WIP → Beta
- [ ] VLM: 백엔드 API 구현 (T-030)
- [ ] VLM: 탭 UI 완성 (T-031)
- [ ] VLM: 사이드바 노출 (T-041)

### UX 향상
- [ ] 파괴적 작업 확인 다이얼로그 (T-038)
- [ ] ARIA 접근성 전체 적용 (T-039)
- [ ] `/api/fs/browse`, `/api/fs/list` 구현 (T-040)
- [ ] 인터랙티브 임베딩 플롯 (T-024)
- [ ] 임베딩 기반 유사도 검색 (T-025)
- [ ] 이미지별 결과 드릴다운 (T-035)
- [ ] 에러 분석 이미지별 시각화 (T-036)

### 신규 기능
- [ ] 웹캠/RTSP 입력 (T-032)
- [ ] 영상 녹화/내보내기 (T-033)
- [ ] 양자화 후 정확도 검증 (T-037)
- [ ] YOLOv10 후처리 (T-044)
- [ ] GPU 프로파일링 (T-045)

---

## 9. FEATURES — 사용자 관점 기능 명세

> 사용자가 체감하는 기능 단위로 정리. 각 FEATURE는 "사용자가 ~할 수 있다" 형태로 기술.

### 9.1 추론 (Inference)

| ID | FEATURE | 현재 | 목표 |
|----|---------|------|------|
| F-001 | 비디오 파일에 실시간 Detection/Classification/Segmentation 추론 | ✅ 구현 | — |
| F-002 | 단일 이미지에 추론 결과 오버레이 보기 | ⚠️ Analysis 탭에서만 | Viewer 탭에서 직접 지원 |
| F-003 | 웹캠/RTSP 스트림에 실시간 추론 | ❌ | 스트림 URL/디바이스 입력 지원 |
| F-004 | 추론 결과가 오버레이된 영상을 MP4로 저장 | ❌ | 녹화 시작/중지 버튼 |
| F-005 | 비디오에서 객체 트래킹 (ID + 궤적) | ⚠️ Viewer에 부분 통합 | 전용 트래킹 모드 + 궤적 시각화 |
| F-006 | 이미지에 포즈 추정 (키포인트 + 스켈레톤) | 🟡 beta 단일 이미지 | 비디오 + 배치 지원 |
| F-007 | 이미지에 인스턴스 세그멘테이션 (마스크 오버레이) | 🟡 beta 단일 이미지 | 비디오 + 배치 + 마스크 내보내기 |
| F-008 | 이미지에 VQA/Captioning (질문 → 답변, 설명 생성) | 🔴 WIP | VLM 탭에서 완전 지원 |
| F-009 | HuggingFace Hub에서 모델 검색/다운로드 | ✅ 구현 | 다운로드 진행률 표시 추가 |
| F-010 | 커스텀 ONNX 모델 타입 등록 및 추론 | ✅ 구현 | 위저드 형태 UX 개선 |

### 9.2 평가 (Evaluation)

| ID | FEATURE | 현재 | 목표 |
|----|---------|------|------|
| F-020 | 여러 모델을 동시에 평가하고 메트릭 비교 | ✅ 구현 | — |
| F-021 | 평가 중 중단하기 | ❌ | force-stop 연동 |
| F-022 | 혼동 행렬로 클래스 간 오분류 패턴 확인 | ❌ | 히트맵 시각화 |
| F-023 | 특정 이미지의 예측 vs GT 상세 보기 | ❌ | 이미지 클릭 → 드릴다운 |
| F-024 | 평가 결과를 CSV/Excel로 내보내기 (모든 태스크) | ⚠️ Detection만 | 태스크별 컬럼 적응 |
| F-025 | COCO JSON 형식 GT 직접 로드 | ❌ | YOLO txt 외 포맷 지원 |
| F-026 | Pose 평가 (OKS, PCK, AP) | ❌ | 키포인트 메트릭 구현 |
| F-027 | Instance-Seg 평가 (Mask AP, Mask AR) | ❌ | 마스크 메트릭 구현 |
| F-028 | Tracking 평가 (MOTA, IDF1, HOTA) | ❌ | MOT 메트릭 구현 |
| F-029 | FPS/레이턴시 벤치마크 + CSV 내보내기 | ⚠️ 내보내기 없음 | export-csv 엔드포인트 |

### 9.3 분석 (Analysis)

| ID | FEATURE | 현재 | 목표 |
|----|---------|------|------|
| F-030 | 단일 이미지 추론 분석 (레터박스, 타이밍) | ✅ 구현 | — |
| F-031 | 텐서 히트맵으로 모델 입력 시각화 | ❌ (README에 명시) | 채널별 활성화 히트맵 |
| F-032 | 두 모델의 추론 결과를 이미지별로 비교 | ✅ 구현 | 메트릭 수준 diff 추가 |
| F-033 | FP/FN을 크기/위치/클래스별로 분석 | ⚠️ 클래스 무관 | 클래스 인식 매칭 + 이미지별 시각화 |
| F-034 | 클래스별 최적 confidence 임계값 자동 탐색 | ✅ 구현 | "Apply" 버튼으로 설정 반영 |
| F-035 | 임베딩 공간을 2D 산점도로 시각화 | ✅ 정적 PNG | 인터랙티브 플롯 (호버/줌) |
| F-036 | Grad-CAM으로 모델 주목 영역 시각화 | ❌ | ONNX 중간 레이어 기반 히트맵 |

### 9.4 도구 (Tools)

| ID | FEATURE | 현재 | 목표 |
|----|---------|------|------|
| F-040 | ONNX 모델 구조/메타데이터 검사 | ✅ 구현 | 그래프 시각화 추가 |
| F-041 | 모델 레이어별 성능 프로파일링 | ✅ CPU만 | GPU 프로파일링 추가 |
| F-042 | 모델 양자화 (Dynamic/Static/FP16) | ✅ 구현 | 양자화 후 정확도 자동 검증 |
| F-043 | 양자화 전후 벤치마크 비교 | ⚠️ 수동 | "Compare in Benchmark" 자동 연동 |

### 9.5 데이터 관리 (Data)

| ID | FEATURE | 현재 | 목표 |
|----|---------|------|------|
| F-050 | 데이터셋 탐색 (클래스/크기/종횡비 분포) | ✅ 5,000장 제한 | 페이지네이션으로 무제한 |
| F-051 | 데이터셋 Train/Val/Test 분할 | ✅ 구현 | 분할 전 미리보기 추가 |
| F-052 | 라벨 포맷 변환 (YOLO↔COCO↔VOC) | ⚠️ 3방향만 | 6방향 완전 지원 |
| F-053 | 클래스 ID 리매핑/병합/삭제 | ✅ 구현 | auto_reindex 구현 + 전후 미리보기 |
| F-054 | 여러 데이터셋 병합 (중복 제거) | ⚠️ MD5만 | dHash 지각 해싱 + 중복 리포트 |
| F-055 | 데이터셋 샘플링 (Random/Stratified/Balanced) | ✅ 구현 | 임베딩 기반 다양성 샘플링 |
| F-056 | 증강 미리보기 및 배치 적용 | ⚠️ 미리보기만 | 파라미터 조정 + 배치 적용 + 라벨 변환 |
| F-057 | 웹 기반 어노테이션 (bbox/polygon/keypoint) | ❌ | 내장 라벨링 도구 |
| F-058 | 모델 기반 자동 라벨링 + 검수 워크플로우 | ❌ | Auto-Label + SAM 통합 |

### 9.6 품질 검사 (Quality)

| ID | FEATURE | 현재 | 목표 |
|----|---------|------|------|
| F-060 | 라벨 이상 탐지 (OOB, 크기, 종횡비) | ✅ 구현 | 과도한 겹침 검출 추가 |
| F-061 | 이미지 품질 검사 (블러, 밝기, 노출) | ✅ 임계값 고정 | 사용자 임계값 조정 + 해상도 검사 |
| F-062 | 근접 중복 이미지 탐지 | ✅ O(n²) | LSH 기반 고속 탐색 + 그룹 클러스터링 |
| F-063 | Train/Val/Test 간 중복 누출 탐지 | ✅ 구현 | 해시 캐싱 + 자동 제거 옵션 |
| F-064 | 쿼리 이미지로 유사 이미지 검색 | ✅ dHash만 | 임베딩 기반 시맨틱 검색 추가 |

### 9.7 인프라 & UX

| ID | FEATURE | 현재 | 목표 |
|----|---------|------|------|
| F-070 | 다크/라이트 테마 전환 | ✅ 구현 | 깜빡임 없는 부분 업데이트 |
| F-071 | 한국어/영어 다국어 지원 | ✅ 구현 | 일본어/중국어 추가 |
| F-072 | 키보드 접근성 (ARIA, 포커스 관리) | ❌ | 전체 UI ARIA 적용 |
| F-073 | 파일/디렉토리 브라우저 | ⚠️ 파일만 | 디렉토리 선택 + 브레드크럼 |
| F-074 | 평가 히스토리 저장 + 실험 비교 | ❌ | 로컬 DB + 비교 대시보드 |
| F-075 | CLI 배치 평가 (headless) | ❌ | `python -m ssook evaluate --config` |
| F-076 | 플러그인 시스템 (커스텀 메트릭/시각화) | ❌ | 플러그인 API + 마켓플레이스 |

> **이 문서는 v1.6.1 기준으로 작성되었으며, 코드 레벨의 정밀 분석을 기반으로 합니다.**
> **각 항목의 우선순위는 사용자 영향도와 구현 난이도를 종합하여 결정되었습니다.**
