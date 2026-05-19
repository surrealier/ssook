"""Tests for core.env — the minimal .env loader."""
import os

import pytest


def test_get_int_default():
    from core.env import get_int
    assert get_int("SSOOK_TEST_NOT_SET_XYZ", 42) == 42


def test_get_int_parses(monkeypatch):
    from core.env import get_int
    monkeypatch.setenv("SSOOK_TEST_NUM", "7")
    assert get_int("SSOOK_TEST_NUM", 0) == 7


def test_get_int_invalid_falls_back(monkeypatch):
    from core.env import get_int
    monkeypatch.setenv("SSOOK_TEST_NUM", "not-a-number")
    assert get_int("SSOOK_TEST_NUM", 99) == 99


def test_get_bool(monkeypatch):
    from core.env import get_bool
    monkeypatch.setenv("SSOOK_TEST_FLAG", "yes")
    assert get_bool("SSOOK_TEST_FLAG") is True
    monkeypatch.setenv("SSOOK_TEST_FLAG", "0")
    assert get_bool("SSOOK_TEST_FLAG") is False


def test_load_env_does_not_override(monkeypatch, tmp_path):
    from core.env import load_env
    monkeypatch.setenv("SSOOK_TEST_KEY", "from-env")
    p = tmp_path / ".env"
    p.write_text("SSOOK_TEST_KEY=from-file\n")
    load_env(paths=[p], override=False)
    assert os.environ["SSOOK_TEST_KEY"] == "from-env"


def test_load_env_override(monkeypatch, tmp_path):
    from core.env import load_env
    monkeypatch.setenv("SSOOK_TEST_KEY2", "from-env")
    p = tmp_path / ".env"
    p.write_text("SSOOK_TEST_KEY2=from-file\n")
    load_env(paths=[p], override=True)
    assert os.environ["SSOOK_TEST_KEY2"] == "from-file"


def test_load_env_comments_and_quotes(tmp_path):
    from core.env import load_env
    p = tmp_path / ".env"
    p.write_text(
        "# a comment\n"
        "\n"
        'KEY1="quoted value"\n'
        "KEY2=simple\n"
    )
    applied = load_env(paths=[p], override=True)
    assert applied["KEY1"] == "quoted value"
    assert applied["KEY2"] == "simple"
