"""Shared pytest fixtures for ssook tests.

Keeps each test file's `sys.path.insert` lean by doing it once here.
"""
import os
import sys

# Ensure the project root is importable for `from server import app` etc.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


@pytest.fixture(scope="session")
def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def isolated_cache(monkeypatch, tmp_path):
    """Redirect SSOOK_CACHE_DIR + SSOOK_TMP_DIR to a tmp dir for the test."""
    monkeypatch.setenv("SSOOK_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("SSOOK_TMP_DIR", str(tmp_path / "tmp"))
    yield tmp_path
