"""Result export helpers.

Different tabs historically emit results as raw dict-of-dict CSV, JSON,
or Excel. Each writer was reimplemented in place, producing slightly
different column orders and quoting. This module centralises the
formats so callers can pick one knob (`format="csv"|"json"|"xlsx"`)
and the rest works.

The implementations stay framework-free (csv module, json module,
optional openpyxl). The FastAPI routes just stream the resulting
bytes via `StreamingResponse`.
"""
from __future__ import annotations

import csv
import io
import json
from typing import Any, Iterable, Sequence


def rows_to_csv(rows: Sequence[dict], columns: Iterable[str] | None = None) -> bytes:
    """Serialize a list of dicts to UTF-8 CSV bytes (with BOM for Excel).

    `columns` controls order; when None, the union of keys (preserving
    first-seen order) is used.
    """
    if columns is None:
        seen: list[str] = []
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.append(k)
        columns = seen
    columns = list(columns)
    buf = io.StringIO()
    buf.write("﻿")  # BOM so Excel detects UTF-8
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow({c: r.get(c, "") for c in columns})
    return buf.getvalue().encode("utf-8")


def rows_to_json(rows: Sequence[dict]) -> bytes:
    return json.dumps(rows, ensure_ascii=False, indent=2).encode("utf-8")


def rows_to_xlsx(rows: Sequence[dict], sheet_name: str = "Results") -> bytes:
    """Serialize rows to xlsx bytes. Requires openpyxl. Raises ImportError
    if the optional dep is missing — callers should catch and fall back
    to CSV.
    """
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name
    if not rows:
        wb.save_virtual_workbook = None
        out = io.BytesIO()
        wb.save(out)
        return out.getvalue()
    columns = list(rows[0].keys())
    ws.append(columns)
    for r in rows:
        ws.append([r.get(c, "") for c in columns])
    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()


def export_bytes(rows: Sequence[dict], fmt: str, columns: Iterable[str] | None = None) -> tuple[bytes, str, str]:
    """One-stop export. Returns (bytes, mime_type, suggested_filename_ext)."""
    fmt = (fmt or "csv").lower()
    if fmt == "json":
        return rows_to_json(rows), "application/json", "json"
    if fmt == "xlsx":
        try:
            return rows_to_xlsx(rows), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "xlsx"
        except ImportError:
            # Fall back to CSV when openpyxl is missing.
            pass
    return rows_to_csv(rows, columns=columns), "text/csv; charset=utf-8", "csv"
