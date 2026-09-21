"""Сбой Pexels стоит вклада Pexels, а не всего слота.

Измерено вживую 18.09: на марсианском эпизоде квота Pexels кончилась
посреди прогона, и 39 слотов из 51 (76%) стали карточками-фолбэками — при
том что Викисклад предложил в том же прогоне 414 кандидатов, а Openverse
200. Причина — один общий try/except на всё тело pexels_photo(): сетевая
ошибка поиска Pexels уносила с собой музей, Викисклад, Openverse, Pixabay
и Unsplash. 13.09 этот же класс уже чинили для ОТСУТСТВУЮЩЕГО ключа
(`return []`), а сетевая ошибка осталась.
"""
import os
import sys
import urllib.error

import pytest

sys.argv = ["pipeline_smart.py", "/tmp"]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import pipeline_smart as ps  # noqa: E402


def _quota_error(*a, **k):
    raise urllib.error.HTTPError("https://api.pexels.com/", 429, "Too Many Requests", {}, None)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(ps, "PEXELS_API_KEY", "x", raising=False)
    monkeypatch.setattr(ps, "PEXELS_BROKEN", False, raising=False)
    monkeypatch.setattr(ps, "PEXELS_FAIL_STREAK", 0, raising=False)
    ps._PEXELS_SEARCH_CACHE.clear()
    ps._PEXELS_VIDEO_SEARCH_CACHE.clear()


@pytest.mark.parametrize("fn,cache", [
    ("_pexels_search_photos", "_PEXELS_SEARCH_CACHE"),
    ("_pexels_search_videos", "_PEXELS_VIDEO_SEARCH_CACHE"),
])
def test_quota_error_returns_empty_instead_of_raising(monkeypatch, fn, cache):
    monkeypatch.setattr(ps.urllib.request, "urlopen", _quota_error)
    assert getattr(ps, fn)("mars rover wheel tracks") == []


@pytest.mark.parametrize("fn,cache", [
    ("_pexels_search_photos", "_PEXELS_SEARCH_CACHE"),
    ("_pexels_search_videos", "_PEXELS_VIDEO_SEARCH_CACHE"),
])
def test_failure_is_not_cached(monkeypatch, fn, cache):
    """429 — состояние минуты, а не свойство запроса. Запомнить пустую
    выдачу на прогон значило бы выключить Pexels до конца эпизода тем же
    молчанием, от которого этот фикс и защищает."""
    monkeypatch.setattr(ps.urllib.request, "urlopen", _quota_error)
    getattr(ps, fn)("mars rover wheel tracks")
    assert "mars rover wheel tracks" not in getattr(ps, cache)


def test_quota_error_does_not_disable_stock_for_the_episode(monkeypatch):
    """PEXELS_BROKEN — только 401/403 или серия сбоев подряд. Одиночная
    квотная ошибка не имеет права выключить сток на весь эпизод."""
    monkeypatch.setattr(ps.urllib.request, "urlopen", _quota_error)
    ps._pexels_search_photos("mars dust storm")
    assert ps.PEXELS_BROKEN is False
