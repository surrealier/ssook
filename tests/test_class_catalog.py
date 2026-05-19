"""Tests for the class label fallback catalogue."""


def test_coco80_size():
    from core.class_catalog import get
    labels = get("coco80")
    assert len(labels) == 80
    assert labels[0] == "person"


def test_voc20_size():
    from core.class_catalog import get
    labels = get("voc20")
    assert len(labels) == 20
    assert "aeroplane" in labels


def test_voc21_seg_has_background():
    from core.class_catalog import get
    labels = get("voc21_seg")
    assert labels[0] == "background"
    assert len(labels) == 21


def test_suggest_by_count():
    from core.class_catalog import suggest
    assert suggest(80) == "coco80"
    assert suggest(20) == "voc20"
    assert suggest(21) == "voc21_seg"
    assert suggest(1000) == "imagenet1k"
    assert suggest(42) is None


def test_as_class_names_dict():
    from core.class_catalog import as_class_names
    d = as_class_names("coco80")
    assert isinstance(d, dict)
    assert d[0] == "person"
    assert len(d) == 80


def test_unknown_returns_none():
    from core.class_catalog import get
    assert get("totally-bogus") is None


def test_list_catalogs_includes_imagenet():
    from core.class_catalog import list_catalogs
    names = [c["name"] for c in list_catalogs()]
    assert "coco80" in names
    assert "imagenet1k" in names


def test_imagenet1k_load_returns_1000():
    from core.class_catalog import get
    labels = get("imagenet1k")
    assert len(labels) == 1000
