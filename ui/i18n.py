"""Minimal i18n module. Default language: English."""

_LANG = "en"

_STRINGS = {
    # main_window tabs
    "viewer": {"en": "Viewer", "ko": "뷰어", "zh": "查看器"},
    "settings": {"en": "Settings", "ko": "설정", "zh": "设置"},
    "evaluation": {"en": "Evaluation", "ko": "평가", "zh": "评估"},
    "analysis": {"en": "Analysis", "ko": "분석", "zh": "分析"},
    "data": {"en": "Data", "ko": "데이터", "zh": "数据"},
    # eval sub-tabs
    "segmentation": {"en": "Segmentation", "ko": "세그멘테이션", "zh": "分割"},
    "benchmark": {"en": "Benchmark", "ko": "벤치마크", "zh": "基准测试"},
    # analysis sub-tabs
    "inference_analysis": {"en": "Inference", "ko": "추론 분석", "zh": "推理分析"},
    "model_compare": {"en": "Compare", "ko": "모델 비교", "zh": "模型对比"},
    "fp_fn": {"en": "FP/FN", "ko": "오탐/미탐", "zh": "误检/漏检"},
    "conf_opt": {"en": "Conf Optimizer", "ko": "Conf 최적화", "zh": "置信度优化"},
    # data sub-tabs
    "explorer": {"en": "Explorer", "ko": "탐색기", "zh": "浏览器"},
    "splitter": {"en": "Splitter", "ko": "분할", "zh": "拆分"},
    "quality": {"en": "Quality", "ko": "품질검사", "zh": "质量检查"},
    "duplicates": {"en": "Duplicates", "ko": "중복탐지", "zh": "重复检测"},
    "label_anomaly": {"en": "Label Anomaly", "ko": "라벨이상", "zh": "标签异常"},
    "format_conv": {"en": "Format Conv.", "ko": "포맷변환", "zh": "格式转换"},
    "augmentation": {"en": "Augmentation", "ko": "증강", "zh": "增强"},
    "class_remap": {"en": "Class Remap", "ko": "클래스매핑", "zh": "类别映射"},
    "similarity": {"en": "Similarity", "ko": "유사검색", "zh": "相似搜索"},
    "sampler": {"en": "Sampler", "ko": "샘플링", "zh": "采样"},
    "merger": {"en": "Merger", "ko": "병합", "zh": "合并"},
    "leaky_split": {"en": "Leak Detect", "ko": "누수탐지", "zh": "泄漏检测"},
    # menu
    "view": {"en": "View", "ko": "보기", "zh": "视图"},
    "dark_mode": {"en": "Dark Mode", "ko": "다크 모드", "zh": "深色模式"},
    "language": {"en": "Language", "ko": "언어", "zh": "语言"},
    # status
    "ready": {"en": "Ready", "ko": "준비", "zh": "就绪"},
    "detection": {"en": "Det", "ko": "탐지", "zh": "检测"},
    "csv_record_start": {"en": "CSV recording started", "ko": "CSV 기록 시작", "zh": "CSV记录开始"},
    "csv_record_stop": {"en": "CSV recording stopped ({n} rows)", "ko": "CSV 기록 중지 ({n}행 누적)", "zh": "CSV记录停止 ({n}行)"},
    "csv_empty": {"en": "No detection data to save.\nEnable 'CSV Record' first and play.", "ko": "저장할 탐지 데이터가 없습니다.\n먼저 'CSV 기록'을 활성화하고 재생하세요.", "zh": "没有检测数据可保存。\n请先启用CSV记录并播放。"},
    "csv_saved": {"en": "CSV saved: {path} ({n} rows)", "ko": "CSV 저장 완료: {path} ({n}행)", "zh": "CSV已保存: {path} ({n}行)"},
    "notice": {"en": "Notice", "ko": "알림", "zh": "提示"},
    "playback_done": {"en": "Playback finished", "ko": "재생 완료", "zh": "播放完成"},
}


def set_language(lang: str):
    global _LANG
    if lang in ("en", "ko", "zh"):
        _LANG = lang


def get_language() -> str:
    return _LANG


def t(key: str, **kwargs) -> str:
    entry = _STRINGS.get(key)
    if entry is None:
        return key
    text = entry.get(_LANG, entry.get("en", key))
    if kwargs:
        text = text.format(**kwargs)
    return text
