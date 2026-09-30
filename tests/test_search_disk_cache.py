import os
import sys
import time

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import pipeline_smart as ps  # noqa: E402


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SEARCH_CACHE_DIR", str(tmp_path / "sc"))
    return tmp_path / "sc"


def test_second_render_does_not_spend_the_quota_again(cache_dir):
    calls = []

    def fetch():
        calls.append(1)
        return {"photos": [{"id": 1}]}
    assert ps.cached_search_json("pexels_photo", "knight|80", fetch) == {"photos": [{"id": 1}]}
    assert ps.cached_search_json("pexels_photo", "knight|80", fetch) == {"photos": [{"id": 1}]}
    assert len(calls) == 1


def test_empty_answer_is_an_answer(cache_dir):
    calls = []
    for _ in range(2):
        ps.cached_search_json("pixabay_video", "arrow hits armor", lambda: calls.append(1) or {"hits": []})
    assert len(calls) == 1


def test_failure_is_not_cached(cache_dir):
    def boom():
        raise OSError("429")
    with pytest.raises(OSError):
        ps.cached_search_json("pexels_video", "q", boom)
    assert ps.cached_search_json("pexels_video", "q", lambda: {"videos": []}) == {"videos": []}


def test_stale_entry_is_fetched_again(cache_dir):
    ps.cached_search_json("pexels_photo", "q", lambda: {"photos": []})
    (f,) = os.listdir(cache_dir)
    old = time.time() - ps.SEARCH_DISK_CACHE_TTL_SEC - 10
    os.utime(cache_dir / f, (old, old))
    assert ps.cached_search_json("pexels_photo", "q", lambda: {"photos": [{"id": 2}]}) == {"photos": [{"id": 2}]}


def test_sources_and_parameters_do_not_collide(cache_dir):
    ps.cached_search_json("pexels_photo", "q|80", lambda: {"a": 1})
    assert ps.cached_search_json("pexels_video", "q|80", lambda: {"b": 2}) == {"b": 2}
    assert ps.cached_search_json("pexels_photo", "q|40", lambda: {"c": 3}) == {"c": 3}


def test_access_key_never_enters_the_cache_key():
    src = open(os.path.join(REPO, "scripts", "pipeline_smart.py"), encoding="utf-8").read()
    for line in src.splitlines():
        if "cached_search_json(" in line and "def " not in line:
            assert "KEY" not in line, line
