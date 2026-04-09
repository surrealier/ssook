/* help-annotations-main.js
   Main-tab paginated help annotations (대분류 단위 페이지)
   Structure: tabName → [ { page, items:[{sel,text}] }, … ]
*/
var _ANNOTATIONS_MAIN = {

  /* ───────── 1. viewer ───────── */
  viewer: [
    {
      page: { en: '📋 Tab Summary', ko: '📋 탭 요약' },
      items: [
        { sel: '#page-title', text: { en: '🎬 Viewer Tab\nLoad an ONNX model and view real-time\ninference results on video/images.\nSupports Detection, Classification,\nSegmentation and more.', ko: '🎬 뷰어 탭\nONNX 모델을 로드하고 비디오/이미지에서\n실시간 추론 결과를 확인하는 탭입니다.\nDetection, Classification, Segmentation 등\n다양한 모델 타입을 지원합니다.' } }
      ]
    },
    {
      page: { en: '🎬 Model Settings', ko: '🎬 모델 설정' },
      items: [
        { sel: '#v-model-type', text: { en: 'Select the model type for inference.\nyolo, classification, segmentation, etc.\nMust match the model architecture\nfor correct post-processing.', ko: '추론에 사용할 모델 타입을 선택합니다.\nyolo, classification, segmentation 등\n모델 구조에 맞는 타입을 정확히 지정해야\n올바른 후처리가 적용됩니다.' } },
        { sel: '#v-batch-size', text: { en: 'Number of images per batch (1–16).\nFixed-batch models are auto-detected.\nGenerally 1 is recommended.\nHigher values use more GPU memory.', ko: '한 번에 처리할 이미지 수 (1~16).\n고정 배치 모델은 자동 감지되며,\n일반적으로 1을 권장합니다.\n값이 클수록 GPU 메모리를 많이 사용합니다.' } },
        { sel: '#v-conf-slider', text: { en: 'Detection confidence threshold (0.01–0.99).\nLower → more detections,\nHigher → only confident results.\nTypically 0.25–0.5 is recommended.', ko: '검출 신뢰도 임계값 (0.01~0.99).\n값을 낮추면 더 많은 객체가 검출되고,\n높이면 확실한 결과만 표시됩니다.\n일반적으로 0.25~0.5 사이를 권장합니다.' } },
        { sel: '#v-model-list', text: { en: 'List of loaded .onnx model files.\nClick a file to activate that model.\nYou can register multiple models\nand switch between them.', ko: '로드된 .onnx 모델 파일 목록입니다.\n파일을 클릭하면 해당 모델이 활성화됩니다.\n여러 모델을 등록해두고 전환할 수 있습니다.' } }
      ]
    },
    {
      page: { en: '📹 Media & Playback', ko: '📹 미디어 & 재생' },
      items: [
        { sel: '#v-video-list', text: { en: 'Video or image file list.\nSupported: mp4, avi, mov, jpg, png, etc.\nClick a file to load it in the viewer.', ko: '비디오 또는 이미지 파일 목록입니다.\n지원 형식: mp4, avi, mov, jpg, png 등\n파일을 클릭하면 뷰어에 로드됩니다.' } },
        { sel: '#viewer-canvas', text: { en: 'Main canvas showing real-time inference.\nBounding boxes, class names, and confidence\nare overlaid. Shows a still frame in image mode.', ko: '추론 결과가 실시간으로 표시되는 메인 캔버스입니다.\n바운딩 박스, 클래스명, 신뢰도가 오버레이됩니다.\n이미지 모드에서는 정지 화면으로 표시됩니다.' } },
        { sel: '#v-seek', text: { en: 'Video seek slider.\nDrag to jump to any position.\nDisabled in image mode.', ko: '비디오 탐색 슬라이더입니다.\n드래그하여 원하는 위치로 이동할 수 있습니다.\n이미지 모드에서는 비활성화됩니다.' } },
        { sel: '#btn-play', text: { en: 'Play / Pause button.\nClick to toggle between\nplaying and paused states.', ko: '비디오 재생/일시정지 버튼입니다.\n재생 중 클릭하면 일시정지,\n정지 상태에서 클릭하면 재생을 시작합니다.' } },
        { sel: '#btn-stop', text: { en: 'Stop playback completely\nand return to the first frame.', ko: '재생을 완전히 중지하고\n처음 프레임으로 되돌아갑니다.' } },
        { sel: '#btn-snapshot', text: { en: 'Capture the current frame as an image.\nSaved with detection results overlaid.', ko: '현재 프레임을 이미지로 캡처합니다.\n검출 결과가 포함된 상태로 저장됩니다.' } },
        { sel: '#v-speed', text: { en: 'Frame-skip speed multiplier.\nHigher values skip more frames\nfor faster playback.', ko: '프레임 스킵 배속 설정입니다.\n값이 클수록 프레임을 건너뛰어\n빠르게 재생됩니다.' } },
        { sel: '#v-frame-counter', text: { en: 'Displays current frame number\n/ total frame count.', ko: '현재 프레임 번호 / 전체 프레임 수를\n표시합니다.' } }
      ]
    },
    {
      page: { en: '📊 Info Panel', ko: '📊 정보 패널' },
      items: [
        { sel: '#v-fps', text: { en: 'Current frames per second (FPS).\nHigher values expected with GPU acceleration.', ko: '현재 초당 처리 프레임 수(FPS)입니다.\nGPU 사용 시 더 높은 값을 기대할 수 있습니다.' } },
        { sel: '#v-model-info', text: { en: 'Loaded model details.\nInput size, output shape, class count\nand other model metadata.', ko: '로드된 모델의 상세 정보입니다.\n입력 크기, 출력 형태, 클래스 수 등\n모델 메타데이터를 확인할 수 있습니다.' } },
        { sel: '#v-video-info', text: { en: 'Current media information.\nResolution, frame count, codec\nand other video/image properties.', ko: '현재 미디어의 정보입니다.\n해상도, 프레임 수, 코덱 등\n비디오/이미지 속성을 표시합니다.' } },
        { sel: '#v-infer-stats', text: { en: 'Inference performance stats.\nPre / Infer / Post stage latency (ms)\nupdated in real time.', ko: '추론 성능 통계입니다.\n전처리/추론/후처리 각 단계의\n소요 시간(ms)을 실시간으로 보여줍니다.' } },
        { sel: '#viewer-results', text: { en: 'Detected object list.\nClass name, confidence, and bounding box\ncoordinates — refreshed every frame.', ko: '검출된 객체 목록입니다.\n클래스명, 신뢰도, 바운딩 박스 좌표가\n표시되며 프레임마다 갱신됩니다.' } },
        { sel: '#v-hw-stats', text: { en: 'Real-time hardware resource monitor:\nCPU/GPU utilization, memory usage, etc.', ko: 'CPU/GPU 사용률, 메모리 점유율 등\n하드웨어 리소스 현황을 실시간 모니터링합니다.' } },
        { sel: '#v-sys-info', text: { en: 'System hardware specs:\nOS, CPU model, GPU model, RAM capacity, etc.', ko: 'OS, CPU 모델, GPU 모델, RAM 용량 등\n시스템 하드웨어 사양 정보입니다.' } }
      ]
    }
  ],

  /* ───────── 2. settings ───────── */
  settings: [
    {
      page: { en: '📋 Tab Summary', ko: '📋 탭 요약' },
      items: [
        { sel: '#page-title', text: { en: '⚙️ Settings Tab\nManage model display options, per-class\ncolors/activation, custom model type\nregistration, and test model downloads.', ko: '⚙️ 설정 탭\n모델 표시 옵션, 클래스별 색상/활성화,\n커스텀 모델 타입 등록, 테스트 모델 다운로드를\n관리하는 탭입니다.' } }
      ]
    },
    {
      page: { en: '🤖 Model Type & Download', ko: '🤖 모델 타입 & 다운로드' },
      items: [
        { sel: '#settings-model-section', text: { en: 'Register custom model types or download\nsample test models.\nTry the test models to explore features.\nDetection, Classification, Segmentation,\nCLIP, and Embedder models are available.', ko: '커스텀 모델 타입을 등록하거나\n테스트용 샘플 모델을 다운로드할 수 있습니다.\n처음 사용 시 테스트 모델로 기능을 체험해 보세요.\nDetection, Classification, Segmentation,\nCLIP, Embedder 모델을 제공합니다.' } }
      ]
    },
    {
      page: { en: '🎨 Display Settings', ko: '🎨 표시 설정' },
      items: [
        { sel: '#box-thickness', text: { en: 'Bounding box line thickness (px).\n2–3 for high-res images,\n1 for low-res.', ko: '바운딩 박스 선 두께 (px)입니다.\n고해상도 이미지에서는 2~3,\n저해상도에서는 1을 권장합니다.' } },
        { sel: '#label-size', text: { en: 'Label text size multiplier.\nDefault is 0.55.\nAdjust within 0.3–1.0.', ko: '라벨 텍스트 크기 배율입니다.\n기본값 0.55 기준으로\n0.3~1.0 범위에서 조절하세요.' } },
        { sel: '#show-labels', text: { en: 'When checked, class names\nare shown above detection boxes.', ko: '체크하면 검출 박스 위에\n클래스 이름이 표시됩니다.' } },
        { sel: '#show-conf', text: { en: 'When checked, confidence scores (%)\nare shown next to class names.', ko: '체크하면 클래스 이름 옆에\n신뢰도 수치(%)가 함께 표시됩니다.' } },
        { sel: '#show-label-bg', text: { en: 'When checked, a background color\nis added behind label text\nfor better readability.', ko: '체크하면 라벨 텍스트 뒤에\n배경색이 추가되어 가독성이 높아집니다.' } }
      ]
    },
    {
      page: { en: '🏷️ Per-Class Settings', ko: '🏷️ 클래스별 설정' },
      items: [
        { sel: '#class-table-container', text: { en: 'Per-class settings table.\nToggle enable/disable, display color,\nand box thickness for each class.\nDisable unneeded classes for a cleaner view.', ko: '클래스별 세부 설정 테이블입니다.\n각 클래스의 활성화/비활성화, 표시 색상,\n박스 두께를 개별적으로 지정할 수 있습니다.\n불필요한 클래스를 끄면 화면이 깔끔해집니다.' } }
      ]
    }
  ],

  /* ───────── 3. benchmark ───────── */
  benchmark: [
    {
      page: { en: '📋 Tab Summary', ko: '📋 탭 요약' },
      items: [
        { sel: '#page-title', text: { en: '⚡ Benchmark Tab\nMeasure inference speed of multiple ONNX models\nand compare FPS, latency (P50/P95/P99),\nand CPU/GPU utilization.', ko: '⚡ 벤치마크 탭\n여러 ONNX 모델의 추론 속도를 측정하고\nFPS, 지연시간(P50/P95/P99), CPU/GPU 사용률을\n비교하는 탭입니다.' } }
      ]
    },
    {
      page: { en: '⚡ Model & Settings', ko: '⚡ 모델 & 설정' },
      items: [
        { sel: '#bench-slots', text: { en: 'Add .onnx models to benchmark.\nRegister multiple models to compare\nperformance side by side.\nCPU/GPU providers are auto-detected.', ko: '벤치마크할 .onnx 모델을 추가합니다.\n여러 모델을 동시에 등록하여\n성능을 비교할 수 있습니다.\nCPU/GPU 프로바이더가 자동 감지됩니다.' } },
        { sel: '#bench-iters', text: { en: 'Number of inference iterations.\nMore iterations → more accurate stats\nbut longer runtime.\nAt least 50 is recommended.', ko: '반복 추론 횟수입니다.\n값이 클수록 정확한 통계를 얻지만\n시간이 오래 걸립니다.\n최소 50회 이상을 권장합니다.' } },
        { sel: '#bench-size', text: { en: 'Model input resolution.\n320: quick test\n640: general use (recommended)\n1280: high-res precision detection', ko: '모델 입력 해상도를 선택합니다.\n320: 빠른 테스트용\n640: 일반적 사용 (권장)\n1280: 고해상도 정밀 검출용' } }
      ]
    },
    {
      page: { en: '📈 Run & Results', ko: '📈 실행 & 결과' },
      items: [
        { sel: '#bench-run', text: { en: 'Start the benchmark.\nAfter warm-up, runs the specified\nnumber of iterations and collects stats.', ko: '벤치마크를 시작합니다.\n워밍업 후 지정 횟수만큼 반복 추론하여\n통계를 수집합니다.' } },
        { sel: '#bench-stop', text: { en: 'Stop the running benchmark.\nResults up to the stop point are shown.', ko: '진행 중인 벤치마크를 중단합니다.\n중단 시점까지의 결과가 표시됩니다.' } },
        { sel: '#bench-progress', text: { en: 'Shows current benchmark progress.', ko: '현재 벤치마크 진행률을 표시합니다.' } },
        { sel: '#bench-results', text: { en: 'Benchmark results table.\nFPS: throughput / Avg: mean latency\nP50/P95/P99: latency percentiles\nCPU%/RAM/GPU%: resource usage\nIf P95·P99 are much higher than Avg,\nsporadic latency spikes are occurring.', ko: '벤치마크 결과 테이블입니다.\nFPS: 초당 처리량 / Avg: 평균 지연시간\nP50/P95/P99: 지연시간 백분위수\nCPU%/RAM/GPU%: 리소스 사용률\nP95·P99가 Avg보다 크게 높으면\n간헐적 지연이 발생하는 것입니다.' } }
      ]
    }
  ],

  /* ───────── 4. evaluation ───────── */
  evaluation: [
    {
      page: { en: '📋 Tab Summary', ko: '📋 탭 요약' },
      items: [
        { sel: '#page-title', text: { en: '📊 Evaluation Tab\nEvaluate multiple models on the same dataset\nand compare mAP, Precision, Recall, F1.\nClass mapping lets you align\nmodel outputs with GT classes.', ko: '📊 평가 탭\n여러 모델을 동일 데이터셋에서 평가하여\nmAP, Precision, Recall, F1 등 성능 지표를\n비교하는 탭입니다. 클래스 매핑 기능으로\n모델-GT 간 클래스를 정확히 연결할 수 있습니다.' } }
      ]
    },
    {
      page: { en: '📂 Model & Data', ko: '📂 모델 & 데이터' },
      items: [
        { sel: '#eval-model-slots', text: { en: 'Add .onnx models to evaluate.\nRegister multiple models to compare\nperformance on the same dataset.\nSpecify the type (yolo, etc.) per model.', ko: '평가할 .onnx 모델을 추가합니다.\n여러 모델을 등록하면 동일 데이터셋으로\n성능을 비교할 수 있습니다.\n모델별로 타입(yolo 등)을 지정하세요.' } },
        { sel: '#eval-img', text: { en: 'Specify the image folder for evaluation.\nSupports jpg, png formats.\nFilenames must match label files.\n(e.g. img001.jpg ↔ img001.txt)', ko: '평가용 이미지 폴더를 지정합니다.\njpg, png 형식을 지원합니다.\n라벨 파일과 동일한 파일명이어야 합니다.\n(예: img001.jpg ↔ img001.txt)' } },
        { sel: '#eval-lbl', text: { en: 'Specify the ground-truth label folder.\nRequires YOLO-format .txt files.\nEach line: class_id cx cy w h (normalized)\nMust have 1:1 correspondence with images.', ko: '정답(GT) 라벨 폴더를 지정합니다.\nYOLO 형식 .txt 파일이 필요합니다.\n각 줄: class_id cx cy w h (정규화 좌표)\n이미지와 1:1 대응되어야 합니다.' } },
        { sel: '#eval-conf', text: { en: 'Confidence threshold for evaluation.\nDetections below this value are ignored.\nTypically 0.25 is used.', ko: '평가 시 적용할 신뢰도 임계값입니다.\n이 값 미만의 검출은 무시됩니다.\n일반적으로 0.25를 사용합니다.' } }
      ]
    },
    {
      page: { en: '🔗 Class Mapping', ko: '🔗 클래스 매핑' },
      items: [
        { sel: '#eval-classmap', text: { en: 'Define GT class mapping.\nFormat: id: name (one per line)\nExample:\n0: person\n1: car\n2: bicycle\nModel output classes must be correctly\nmapped to GT classes for valid evaluation.', ko: 'GT 클래스 매핑을 정의합니다.\n형식: id: name (한 줄에 하나)\n예)\n0: person\n1: car\n2: bicycle\n모델 출력 클래스와 GT 클래스를\n정확히 연결해야 올바른 평가가 됩니다.' } }
      ]
    },
    {
      page: { en: '📊 Interpreting Results', ko: '📊 결과 해석' },
      items: [
        { sel: '#eval-run-btn', text: { en: 'Start evaluation.\nRuns inference on all images then\ncompares with GT to compute metrics.', ko: '평가를 시작합니다.\n모든 이미지에 대해 추론 후\nGT와 비교하여 지표를 산출합니다.' } },
        { sel: '#eval-stop-btn', text: { en: 'Stop the running evaluation.', ko: '진행 중인 평가를 중단합니다.' } },
        { sel: '#eval-prog', text: { en: 'Shows evaluation progress.', ko: '평가 진행률을 표시합니다.' } },
        { sel: '#eval-results', text: { en: 'Evaluation results table.\nmAP@50: mean AP at IoU 0.5\nmAP@50:95: mean AP at IoU 0.5–0.95 (COCO)\nPrecision: correct detections / all detections\nRecall: detected / all ground truths\nF1: harmonic mean of Precision & Recall\nCloser to 1 is better.', ko: '평가 결과 테이블입니다.\nmAP@50: IoU 0.5 기준 평균 정밀도\nmAP@50:95: IoU 0.5~0.95 평균 (COCO 표준)\nPrecision: 검출 중 정답 비율\nRecall: 정답 중 검출된 비율\nF1: Precision과 Recall의 조화 평균\n값이 1에 가까울수록 우수합니다.' } }
      ]
    }
  ],

  /* ───────── 5. analysis ───────── */
  analysis: [
    {
      page: { en: '📋 Tab Summary', ko: '📋 탭 요약' },
      items: [
        { sel: '#page-title', text: { en: '🔬 Analysis Tab\nVisualize the inference pipeline\nstep by step on a single image.\nPreprocess (Letterbox) → Inference → Post-process\nall shown as images.', ko: '🔬 분석 탭\n단일 이미지에 대한 추론 과정을\n단계별로 시각화합니다.\n전처리(Letterbox) → 추론 → 후처리 과정을\n이미지로 확인할 수 있습니다.' } }
      ]
    },
    {
      page: { en: '🔬 Input Settings', ko: '🔬 입력 설정' },
      items: [
        { sel: '#ana-model', text: { en: '.onnx model file for analysis.\nVisually inspect the detailed inference\npipeline on a single image.', ko: '분석에 사용할 .onnx 모델 파일입니다.\n단일 이미지에 대한 상세 추론 과정을\n시각적으로 확인할 수 있습니다.' } },
        { sel: '#ana-type', text: { en: 'Select the model type.\nyolo, classification, segmentation, etc.\nMust match the model architecture.', ko: '모델 타입을 선택합니다.\nyolo, classification, segmentation 등\n모델 구조에 맞게 지정하세요.' } },
        { sel: '#ana-img', text: { en: 'Select a single image to analyze.\nSupports jpg, png formats.\nOne image is analyzed in depth.', ko: '분석할 단일 이미지를 선택합니다.\njpg, png 형식을 지원합니다.\n한 장의 이미지를 깊이 있게 분석합니다.' } }
      ]
    },
    {
      page: { en: '📋 Analysis Results', ko: '📋 분석 결과' },
      items: [
        { sel: '#ana-panels', text: { en: 'Analysis visualization panels.\nOriginal: raw input image\nLetterbox: resized to model input dimensions\nDetection: image with bounding box overlay\nCompare each stage to understand preprocessing.', ko: '분석 결과 시각화 패널입니다.\n원본 이미지: 입력 원본 그대로\nLetterbox: 모델 입력 크기로 변환된 모습\n검출 결과: 바운딩 박스가 오버레이된 이미지\n각 단계를 비교하여 전처리 과정을 이해할 수 있습니다.' } },
        { sel: '#ana-stats', text: { en: 'Detailed inference timing.\nPre: preprocessing (resize, normalize)\nInfer: model inference\nPost: post-processing (NMS, etc.)\nTotal: end-to-end latency (ms)', ko: '추론 타이밍 상세 분석입니다.\nPre: 전처리 시간 (리사이즈, 정규화)\nInfer: 모델 추론 시간\nPost: 후처리 시간 (NMS 등)\nTotal: 전체 소요 시간 (ms)' } },
        { sel: '#ana-dets', text: { en: 'Detailed list of detected objects.\nClass name, confidence, and coordinates\nfor each individual detection.', ko: '검출된 객체의 상세 목록입니다.\n클래스명, 신뢰도, 좌표가 표시되며\n각 검출 결과를 개별적으로 확인할 수 있습니다.' } }
      ]
    }
  ],

  /* ───────── 6. explorer ───────── */
  explorer: [
    {
      page: { en: '📋 Tab Summary', ko: '📋 탭 요약' },
      items: [
        { sel: '#page-title', text: { en: '📁 Dataset Explorer Tab\nLoad an image folder to browse as a\nthumbnail gallery and view class\ndistribution and statistics.', ko: '📁 데이터셋 탐색기 탭\n이미지 폴더를 로드하여 썸네일 갤러리로\n탐색하고, 클래스 분포와 통계를 확인하는\n탭입니다.' } }
      ]
    },
    {
      page: { en: '📁 Load Data', ko: '📁 데이터 로드' },
      items: [
        { sel: '#exp-img', text: { en: 'Specify the image folder to explore.\nSelect a directory containing\njpg, png, or other image files.', ko: '탐색할 이미지 폴더를 지정합니다.\njpg, png 등 이미지 파일이 포함된\n디렉토리를 선택하세요.' } },
        { sel: '#exp-lbl', text: { en: 'Specify the label folder (optional).\nIf YOLO-format .txt files are present,\nbounding boxes are shown on thumbnails.\nFilenames must match the images.', ko: '라벨 폴더를 지정합니다 (선택사항).\nYOLO 형식 .txt 파일이 있으면\n바운딩 박스가 썸네일에 표시됩니다.\n이미지와 동일한 파일명이어야 합니다.' } }
      ]
    },
    {
      page: { en: '🔍 Filter & Gallery', ko: '🔍 필터 & 갤러리' },
      items: [
        { sel: '#exp-class-filter', text: { en: 'Filter images containing\nspecific classes.\nMultiple classes can be selected.', ko: '특정 클래스가 포함된 이미지만\n필터링하여 볼 수 있습니다.\n여러 클래스를 선택할 수 있습니다.' } },
        { sel: '#exp-min-boxes', text: { en: 'Minimum box count filter.\nOnly shows images with at least\nthe specified number of objects.', ko: '최소 박스 수 필터입니다.\n지정한 수 이상의 객체가 있는\n이미지만 표시합니다.' } },
        { sel: '#exp-stats', text: { en: 'Dataset statistics summary.\nTotal image count, per-class distribution,\nbox size distribution, and more.', ko: '데이터셋 통계 요약입니다.\n전체 이미지 수, 클래스별 분포,\n박스 크기 분포 등을 확인할 수 있습니다.' } },
        { sel: '#exp-gallery', text: { en: 'FiftyOne-style thumbnail gallery.\nClick an image to enlarge.\nLabels are overlaid as boxes if available.', ko: 'FiftyOne 스타일 썸네일 갤러리입니다.\n이미지를 클릭하면 확대 보기가 가능하며\n라벨이 있으면 박스가 오버레이됩니다.' } }
      ]
    }
  ],

  /* ───────── 7. splitter ───────── */
  splitter: [
    {
      page: { en: '📋 Tab Summary', ko: '📋 탭 요약' },
      items: [
        { sel: '#page-title', text: { en: '✂️ Dataset Splitter Tab\nSplit images + labels into Train/Val/Test\nby ratio. Use the Stratified option\nto maintain balanced class proportions.', ko: '✂️ 데이터셋 분할 탭\n이미지+라벨을 Train/Val/Test로\n비율에 맞게 분할하는 탭입니다.\nStratified 옵션으로 클래스 비율을\n균등하게 유지할 수 있습니다.' } }
      ]
    },
    {
      page: { en: '📂 Input Directories', ko: '📂 입력 디렉토리' },
      items: [
        { sel: '#split-img', text: { en: 'Specify the image folder to split.\nSelect a directory containing\njpg, png format images.', ko: '분할할 이미지 폴더를 지정합니다.\njpg, png 형식의 이미지가\n포함된 디렉토리를 선택하세요.' } },
        { sel: '#split-lbl', text: { en: 'Specify the label folder.\nRequires YOLO-format .txt files.\nFilenames must match the images.\n(e.g. img001.jpg ↔ img001.txt)', ko: '라벨 폴더를 지정합니다.\nYOLO 형식 .txt 파일이 필요합니다.\n이미지와 동일한 파일명이어야 합니다.\n(예: img001.jpg ↔ img001.txt)' } },
        { sel: '#split-out', text: { en: 'Specify the output directory.\nSplit results are created as:\noutput/\n  train/images/ + labels/\n  val/images/ + labels/\n  test/images/ + labels/', ko: '출력 디렉토리를 지정합니다.\n분할 결과가 아래 구조로 생성됩니다:\noutput/\n  train/images/ + labels/\n  val/images/ + labels/\n  test/images/ + labels/' } }
      ]
    },
    {
      page: { en: '⚙️ Split Settings', ko: '⚙️ 분할 설정' },
      items: [
        { sel: '#split-ratio-section', text: { en: 'Set Train / Val / Test ratios.\nMust sum to 100%.\n70/20/10 or 80/10/10\nis generally recommended.', ko: 'Train / Val / Test 비율을 설정합니다.\n합계가 100%가 되어야 합니다.\n일반적으로 70/20/10 또는 80/10/10을\n권장합니다.' } },
        { sel: '#split-stratified', text: { en: 'Enable Stratified splitting to maintain\nbalanced class proportions in each split.\nHighly recommended for datasets\nwith class imbalance.', ko: 'Stratified(계층적) 옵션을 활성화하면\n각 분할에 클래스 비율이 균등하게 유지됩니다.\n클래스 불균형이 있는 데이터셋에서\n반드시 사용을 권장합니다.' } }
      ]
    }
  ]

};