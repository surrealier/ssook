# ssook AI-스택 정합 계획 (RAG·VectorDB·GraphDB·vLLM·LangChain·LangGraph·Multi-Agent·Orchestration·Hybrid RAG·A2A)

## Context

ssook는 **로컬-퍼스트 ONNX CV 모델 툴킷**(FastAPI + vanilla JS, 빌드 단계 없음, 옵션 의존성 lazy-import,
단일 메인테이너)다. 추론/평가/분석/데이터/학습/배포 30+ 탭이 풍부한 **구조화 산출물**을 만든다:
`run.yaml`(run_record, sha256 fingerprint+env), `eval_history/*.json`(per-model/class 메트릭),
`benchmark_results.csv`+`BENCHMARK_REPORT.md`, `*.deploy.json`(ssook.deploy/1 매니페스트, source_model
fingerprint+runtime config), model inspector/diagnosis/profiler JSON, dataset explorer 통계, ~400KB
다국어 docs(en/ko/ja/zh) + FEATURE_AUDIT, `class_catalog`(COCO/VOC/ImageNet), `app_config.yaml`.

**핵심 통합 seam (이미 존재)**:
- LLM/VLM: `core/vlm_inference.py`의 `OpenAICompatBackend`(`/chat/completions`, base64 image_url) +
  `server/model_routes.py:validate_endpoint_url`(SSRF 가드) + `SSOOK_VLM_ALLOW_INTERNAL` →
  **로컬 vLLM(OpenAI 호환 API)이 코드 변경 없이 연결**.
- Job 기판: `server/state.py`의 `TaskState`/`try_start`/`executor`/`task_locks`/`all_states` +
  `/api/tasks`·`/api/{task}/status`·`/api/force-stop/{key}`.
- 플러그인: registry 패턴(`BaseOptimizer`/`OptimizerRegistry`, deployment/training registry),
  `optimization_pipeline.py`(add_step/run = 기존 체이닝), `hf_downloader.py`(HF 모델 fetch).
- 부재: LLM·VectorDB·GraphDB·임베딩 영속화(현재 dHash+BKTree, t-SNE 2D 좌표 캐시만).

**목표**: ssook에 "로컬 Copilot/Agent 레이어"를 정합 — 모델·데이터셋·실험 산출물을 검색·추론·자동화한다.
아래는 키워드별 **3개 이상**의 프로젝트 맞춤 아이디어(연결 seam·접근·가치·effort)와 통합 아키텍처/로드맵.
**원칙: 오프라인 우선(전부 embedded/local 모드 + 옵션 의존성 gating), SSRF 가드 재사용, vanilla 탭 UI, registry+TaskState 패턴 재사용, 파괴적 액션은 human-in-the-loop.**

---

## 1. RAG — 산출물·문서 위 질의응답 Copilot
- **R1. "내 실험에게 묻기"**: `run.yaml`+`eval_history/*.json`+benchmark CSV+`deploy.json`을 인덱싱 →
  "데이터셋 X에서 mAP@50 최고 모델은?", "이 .engine은 어느 데이터셋에서 왔나?". 답변에 trace_id/파일 인용.
  *연결*: run_record, eval_history, manifest. *접근*: JSON/YAML 로더→청크→임베딩→retrieve→vLLM 생성. *effort*: M.
- **R2. 인앱 문서 어시스턴트**: `docs/**`(다국어)+FEATURE_AUDIT+README 인덱싱 → "TensorRT export 방법?",
  "mIoU 정의?"를 프로젝트 자체 문서로 답. *연결*: docs 코퍼스. *가치*: 온보딩·도움말. *effort*: S(가장 먼저).
- **R3. 진단 기반 권고 NL화**: `model_diagnosis`+`profiler`+`inspector` 출력 + 최적화 문서를 RAG로 묶어
  "이 모델은 quantizable 40% + Conv+ReLU fusable → 권장 파이프라인은…"을 자연어+실행가능 step으로.
  *연결*: model_diagnosis RecommendationEngine, optimization_pipeline. *effort*: M.
- **R4. 로그/에러 RAG**: `settings/logs/ssook.log` + 에러 envelope(trace_id) 인덱싱 → "run abc123 왜 실패?" 디버깅. *effort*: S.

## 2. VectorDB — 임베딩 영속화 & 의미 검색 (embedded/offline)
- **V1. 의미 기반 중복/유사 탐지(dHash 보강)**: CLIP/embedder 이미지 임베딩을 embedded 벡터스토어
  (LanceDB/sqlite-vec/FAISS)에 영속화 → duplicate/leaky/similarity 탭이 crop·recolor·rotate까지 잡는
  **의미적 ANN**으로 확장(현 dHash+BKTree 한계 보완). *연결*: extra_routes embedder/clip, quality_routes. *effort*: M.
- **V2. "유사 찾기 / 예시 찾기" 검색**: 이미지 또는 **CLIP 텍스트 프롬프트**로 데이터셋 질의("야간 트럭 같은 이미지").
  active-learning 샘플 선별(sampler 탭)·실패 케이스 수집에 직결. *연결*: CLIP 텍스트+이미지 인코더(이미 존재). *effort*: M.
- **V3. RAG 멀티모달 retriever 기판**: 동일 스토어에 doc 청크+산출물+이미지 임베딩을 함께 보관 → R1–R3와
  Hybrid RAG의 공통 retrieval 레이어. *연결*: RAG 레이어. *effort*: M.
- **V4. 임베딩 캐시 영속화**: 현재 매 세션 재계산(2D 좌표만 캐시) → `model_sha256+image` 키로 전체 벡터 캐시
  (`paths.cache_dir`) → 재질의 즉시. *연결*: run_record fingerprint, embedding-viewer. *effort*: S.
- *권장*: LanceDB 또는 sqlite-vec(서버리스·단일파일, 로컬 ethos 부합).

## 3. GraphDB — 계보·온톨로지 (embedded)
- **G1. 모델·실험 계보 그래프**: 노드={dataset, model(sha256), train run, eval run, deploy artifact, benchmark},
  엣지={trained-from, exported-to, evaluated-on, derived-from(quant/prune)}. sha256 fingerprint가 자연 join 키.
  "이 .engine의 원본 데이터셋?", "모델 X 파생 산출물 전부". *연결*: run_record/deploy.json fingerprint. *effort*: M.
- **G2. 클래스/라벨 온톨로지 그래프**: class_catalog+custom_model_types+remapper 매핑을 그래프로(계층·동의어·
  데이터셋 간 클래스 정렬) → converter/remapper 자동 매핑 제안, eval 클래스매핑 다이얼로그 보조. *연결*: class_catalog, remapper. *effort*: M.
- **G3. GraphRAG (멀티홉)**: G1 그래프 + 벡터 검색 → "leaky-split 데이터셋으로 학습된 모델 중 배포된 것?"을
  그래프 순회→LLM 요약. *연결*: GraphDB+VectorDB+RAG. *effort*: L.
- **G4. 혼동(confusion) 그래프**: eval confusion matrix → 클래스 간 체계적 혼동 그래프 → "어느 클래스쌍이 자주 혼동?". *effort*: S.
- *권장*: Kùzu(embedded, Cypher, 오프라인) 또는 NetworkX(소규모 in-proc).

## 4. vLLM — 로컬 고처리량 LLM/VLM 서빙 (seam 이미 존재)
- **vL1. 기존 VLM 백엔드에 즉시 연결(코드 0)**: `vllm serve Qwen2.5-VL` → openai 백엔드 endpoint를 가리키게 +
  "로컬 vLLM 사용" 프리셋(endpoint `http://localhost:8000/v1`, `SSOOK_VLM_ALLOW_INTERNAL=1`). CLIP-기반
  caption/VQA를 진짜 VLM으로 업그레이드. *연결*: OpenAICompatBackend, validate_endpoint_url. *effort*: S(최우선·최대 즉효).
- **vL2. RAG/Agent의 생성 엔진**: 동일 vLLM 서버(텍스트 LLM)가 RAG(R1–R3)·agent 레이어의 생성기. 1개 로컬 서버가
  VLM+LLM 역할 겸직, OpenAI 호환·오프라인. *연결*: RAG/agent. *effort*: S(인프라 공유).
- **vL3. 데이터셋 자동 캡션/라벨링**: `/api/vlm/batch`를 vLLM로 확장해 수천 장 캡션/태그/coarse-label →
  데이터 도구(explorer/sampler)·VectorDB(캡션 텍스트 임베딩) 공급. vLLM 처리량이 dataset-scale을 가능케. *effort*: M.
- **vL4. Guided decoding(구조화 출력)**: vLLM JSON-schema 제약 출력으로 LLM이 신뢰성 있게 tool-call/구조화 요약 생성. *연결*: agent tool-calling. *effort*: S.

## 5. LangChain — RAG·툴 배선
- **LC1. 산출물 RAG 체인**: 로더(JSON/YAML/MD)+splitter+VectorDB retriever+ChatOpenAI(base_url=vLLM)로 R1–R3 구성. *effort*: M.
- **LC2. ssook API를 LangChain Tool로 래핑**: run_eval/run_benchmark/inspect/diagnose/export/similarity-search를
  Tool로 → LLM이 "모델 X를 데이터셋 Y로 평가하고 mAP 알려줘"를 실제 실행. *연결*: registry 패턴(BaseOptimizer→Tool 미러),
  TaskState job. 각 Tool=기존 route/core 얇은 래퍼. *effort*: M.
- **LC3. 구조화 출력 파서**: output parser+Pydantic로 LLM 분석→프로젝트 결과 스키마(예: 권장 최적화 파이프라인을
  `optimization_pipeline` step으로 파싱→원클릭 적용). *연결*: optimization_pipeline, deploy config. *effort*: M.
- **LC4. 세션 메모리**: 현재 모델/데이터셋 맥락(`G.model`/`G.imgDir`)을 대화 메모리로 유지. *effort*: S.

## 6. LangGraph — 상태형 멀티스텝 워크플로
- **LG1. "내 모델 최적화" 그래프**: inspect→diagnose→파이프라인 제안→(승인)→quant/prune→**정확도 검증
  (Wave-8 do_validate)**→degraded면 분기·재시도(완화)→export→benchmark. 조건 엣지+체크포인트가 재시도/분기에 적합.
  *연결*: inspector/diagnosis, optimization_pipeline, Wave-8 검증, deploy/benchmark. *effort*: L.
- **LG2. 데이터셋 큐레이션 그래프**: explore→품질 이슈(anomaly/quality/duplicate/leaky)→VectorDB 클러스터링→
  수정 제안(Wave-7 delete-files·remap·sampler 리밸런스)→(승인)→적용→재explore. interrupt=human-in-the-loop. *effort*: L.
- **LG3. 평가 트리아지 그래프**: eval→mAP 낮으면 분기(error-analyzer FP/FN, confusion 최악 클래스, VectorDB로 실패 예시
  수집, LLM 원인 요약)→권고(데이터 추가/conf-optimizer 튜닝/재학습). *연결*: eval, error-analyzer, conf-optimizer, VectorDB. *effort*: L.
- **LG4. 체크포인트 장기실행**: LangGraph 영속화↔run_record/trace_id 내구성 매핑(재시작 생존). *effort*: M.

## 7. Multi-Agent — 도메인 전문 에이전트
- **MA1. 전문 에이전트 크루**: DataAgent/TrainAgent/OptimizeAgent/DeployAgent/EvalAgent(각자 툴셋) + Supervisor가
  목표("엣지용 빠른 모델 출하")를 라우팅. 탭 도메인이 에이전트와 1:1, registry 패턴 재사용. *effort*: L.
- **MA2. 비평/검증 에이전트(adversarial)**: 파괴적 액션(delete-files, degraded 양자화 수용) 적용 전 다른 에이전트의
  제안을 검증·게이팅 — Wave-7 삭제 안전성·Wave-8 do_validate와 결합, 자율 파괴 위험 감소. *effort*: M.
- **MA3. 자동 라벨링 팀**: VLM 에이전트(vLLM 캡션)+embedder 에이전트(VectorDB 클러스터)+labeling 에이전트(YOLO 라벨 제안)가
  미라벨 데이터셋 부트스트랩. *연결*: vlm batch, embedder, Wave-4 augmentation 라벨변환. *effort*: L.
- **MA4. 벤치/EP 에이전트**: 타깃(엣지/서버)별 EP·정밀도 조합을 Wave-8 EP-격리 서브프로세스로 시도·벤치→최적 deploy config 추천.
  *연결*: ep_worker, benchmark, deployment. *effort*: M.

## 8. Orchestration — 탭을 엔드투엔드 파이프라인으로
- **O1. E2E MLOps 파이프라인 러너**: 선언적 파이프라인(data→split→train→eval→optimize→validate→deploy→benchmark)을
  기존 TaskState job 위에서 실행 — 현 train→deploy→benchmark 핸드오프를 그래프로 일반화. *연결*: _loadModel/_openInBenchmark,
  optimization_pipeline, all_states. 신규 orchestration_routes+orchestration_state. *effort*: M.
- **O2. TaskState 기반 오케스트레이션 기판**: 부모 orchestration_state가 자식 job(eval/deploy/…)을 executor로 spawn·모니터,
  `/api/orchestration/run`+force-stop. 신규 인프라 0(기존 패턴 재사용). *effort*: M.
- **O3. 비주얼 파이프라인 빌더(프론트)**: 기존 최적화 "Pipeline Builder" UI를 전체 MLOps DAG 빌더로 확장(노드=탭/도구). *effort*: M.
- **O4. 스케줄/트리거 실행**: 신규 ORT 버전 시 재벤치, 데이터셋 갱신 시 재eval 등 간단 스케줄러. *effort*: S.

## 9. Hybrid RAG — 다중 검색 모드 융합
- **HR1. Vector + Keyword(BM25)**: dense(임베딩)+sparse(키워드) reciprocal-rank-fusion — 모델명·trace_id·메트릭명 등
  정확 토큰은 BM25, 의미는 dense. *연결*: docs+산출물 코퍼스. *effort*: M.
- **HR2. 멀티모달 Hybrid(텍스트+이미지)**: 텍스트 코퍼스 AND 이미지 임베딩(CLIP 공유공간)을 함께 검색 — 문서+예시 이미지를
  동시에 요구하는 질의. *연결*: CLIP, VectorDB 이미지 임베딩. *effort*: M.
- **HR3. Graph + Vector(GraphRAG) — 핵심**: 계보/온톨로지 GraphDB(G1/G2)로 구조적 사실(멀티홉) 순회 후 벡터로 근거 보강·융합.
  *연결*: GraphDB+VectorDB+RAG. *effort*: L.
- **HR4. 메트릭 인지 라우팅**: 수치 질의("최고 mAP")는 eval_history/benchmark 구조적 쿼리(SQL/pandas), 의미 질의는 벡터 —
  query router(LangChain/LangGraph)가 모드 선택. *effort*: M.

## 10. A2A — 에이전트 노출 & 연합
- **A2A-1. ssook을 A2A 서버로(Agent Card)**: 역량(eval/benchmark/inspect/optimize/export/quality/similarity)을 A2A
  Agent Card로 노출 → 외부 오케스트레이터/IDE/타 에이전트가 ssook에 태스킹. TaskState job 생명주기(running/progress/complete+
  force-stop)가 A2A task 생명주기에 직접 매핑. *연결*: REST routers+registry→A2A skill manifest. *effort*: L.
- **A2A-2. ssook을 A2A 클라이언트로**: 크루(MA1)가 외부 전문 에이전트(원격 라벨링/클라우드 학습)에 위임 — SSRF 가드·endpoint
  검증이 아웃바운드 A2A를 보안. *연결*: validate_endpoint_url, openai-backend seam. *effort*: M.
- **A2A-3. 내부 크루 프로토콜**: ssook 내부 에이전트(Data↔Optimize↔Deploy) 간 메시지 버스를 A2A로 표준화(독립 배포·교체 가능). *effort*: M.
- **A2A-4. MCP 보완**: ssook 툴을 MCP 서버로 노출 → MCP 클라이언트(Claude/IDE)가 ssook 구동. MCP=LLM용 툴, A2A=에이전트 간
  태스킹으로 상보. *연결*: LC2 Tool 래퍼. *effort*: M.

---

## 통합 아키텍처 (How it composes — "ssook Copilot")
- **Compute**: 로컬 vLLM(OpenAI 호환)이 LLM+VLM 겸직(기존 seam 재사용, 오프라인).
- **Memory/Retrieval**: embedded VectorDB(LanceDB/sqlite-vec) = doc+산출물+이미지 임베딩; embedded GraphDB(Kùzu) = 계보·온톨로지.
- **Reasoning**: LangChain(체인/툴/파서) + LangGraph(상태형 워크플로); Hybrid RAG(vector+BM25+graph)가 검색 전략.
- **Agency**: 도메인 Multi-Agent 크루 + Orchestration(TaskState job 위) ; A2A/MCP로 노출·연합.
- 전부 옵션 의존성(`requirements-ai.txt`, lazy import, availability-gating — torch/ultralytics/openvino와 동일 패턴),
  신규 router는 router+TaskState 패턴, 신규 탭은 vanilla `Tabs.*`, 아웃바운드는 SSRF 가드.

## 단계별 로드맵 (각 단계 독립 가치, 점진 채택)
- **Phase 0 (최소·즉효)**: vL1(vLLM 프리셋) + R2(docs-RAG copilot) + 최소 embedded VectorDB → 스택 E2E 입증. 신규 탭 "Copilot".
- **Phase 1**: R1/R3 산출물 RAG + V1/V2/V4 임베딩 영속화·의미검색 + LC1/LC2 체인·툴.
- **Phase 2**: G1 계보 GraphDB + HR1–HR3 Hybrid/GraphRAG + LC3 구조화 출력.
- **Phase 3**: LG1–LG3 LangGraph 워크플로 + O1–O3 Orchestration(파이프라인 빌더).
- **Phase 4**: MA1–MA2 멀티에이전트 + A2A-1/A2A-4 노출(A2A/MCP).

## 제약 / 리스크 (ssook 특이)
- **오프라인 우선**: 모든 구성요소 embedded/local 모드 필수(클라우드 강제 금지). 옵션 의존성 gating으로 base(ONNX-only) env 보존.
- **단일 메인테이너·vanilla 프론트**: Copilot UI는 vanilla 탭, 백엔드는 기존 router/TaskState/registry 패턴 — 학습비용 최소.
- **보안**: 모든 아웃바운드(vLLM·A2A·원격 DB)에 `validate_endpoint_url` 재사용; 파괴적 에이전트 액션은 human-in-the-loop +
  Wave-7(안전 삭제)·Wave-8(do_validate) 게이팅.
- **자원**: GPU 경합(`task_locks["gpu_infer"]`) — vLLM은 별도 GPU/프로세스 권장(동일 소형 GPU에서 CV 추론과 동시 구동 지양). 문서화.
- **YAGNI/범위**: 표면이 넓다 — 로드맵으로 점진 채택, Phase 0만으로도 단독 가치.

## Verification (단계별)
- **Phase 0**: vLLM 기동 → Copilot에 문서 질의 → 인용 포함 답변 확인; 작은 코퍼스로 retriever 오프라인 단위테스트.
- **공통**: 신규 기능은 TaskState job 패턴 → 기존 `/api/tasks` 폴링·force-stop·`@route_errors`·run_record 그대로 적용;
  retrieval/그래프/라우팅은 순수 로직 단위테스트(무거운 LLM/임베딩 없이), 실제 LLM 경로는 옵션-의존성 gating 뒤 통합테스트.
- **회귀 0**: AI 레이어는 전부 additive·옵션 — 미설치 시 기존 동작 불변(`torch271` 전체 pytest green 유지).

> 본 문서는 **계획(아이디어 제안)**이며 구현을 포함하지 않는다. 승인 시 `docs/ai-stack-integration-plan.md`로
> 영속화하고 Phase 0부터 착수할 수 있다.
