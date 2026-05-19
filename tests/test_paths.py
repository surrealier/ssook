"""Tests for core.paths — temp/cache directory layout."""
import os


def test_cache_dir_creates(isolated_cache):
    from core import paths
    p = paths.cache_dir("test_cat")
    assert p.exists() and p.is_dir()
    assert p.name == "test_cat"


def test_tmp_dir_creates(isolated_cache):
    from core import paths
    p = paths.tmp_dir("test_tmp")
    assert p.exists() and p.is_dir()


def test_cleanup_stale_removes_old(isolated_cache):
    from core import paths
    p = paths.tmp_dir("stale_test")
    f = p / "old.txt"
    f.write_text("x")
    # Set mtime to 30 days ago.
    old = int(__import__("time").time()) - 30 * 86400
    os.utime(f, (old, old))
    removed = paths.cleanup_stale("stale_test", older_than_days=7)
    assert removed >= 1
    assert not f.exists()


def test_cleanup_stale_keeps_fresh(isolated_cache):
    from core import paths
    p = paths.tmp_dir("fresh_test")
    f = p / "new.txt"
    f.write_text("x")
    paths.cleanup_stale("fresh_test", older_than_days=7)
    assert f.exists()
