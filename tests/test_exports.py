"""Tests for core.exports — unified CSV/JSON output helpers."""


def test_rows_to_csv_has_bom_and_header():
    from core.exports import rows_to_csv
    rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    data = rows_to_csv(rows)
    text = data.decode("utf-8")
    assert text.startswith("﻿")  # BOM for Excel
    assert "a,b" in text
    assert "1,x" in text


def test_rows_to_csv_custom_columns():
    from core.exports import rows_to_csv
    rows = [{"a": 1, "b": "x", "c": "skip"}]
    data = rows_to_csv(rows, columns=["b", "a"]).decode("utf-8")
    first_data_line = data.splitlines()[1]
    assert first_data_line == "x,1"
    assert "skip" not in data


def test_rows_to_json_unicode():
    from core.exports import rows_to_json
    rows = [{"label": "사람", "n": 3}]
    out = rows_to_json(rows).decode("utf-8")
    assert "사람" in out


def test_export_bytes_csv_default():
    from core.exports import export_bytes
    rows = [{"a": 1}]
    data, mime, ext = export_bytes(rows, "csv")
    assert "csv" in mime
    assert ext == "csv"
    assert b"a" in data


def test_export_bytes_json():
    from core.exports import export_bytes
    rows = [{"a": 1}]
    data, mime, ext = export_bytes(rows, "json")
    assert mime == "application/json"
    assert ext == "json"
