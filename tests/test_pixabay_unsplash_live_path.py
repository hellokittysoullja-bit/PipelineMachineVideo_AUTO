"""Pixabay и Unsplash подключены к ЖИВОМУ пути отбора эпизода.

РЕАЛЬНЫЙ пробел, найденный прямой проверкой: ровно тот же класс, что уже
ловили у Openverse 07.09 («код существовал, был включён, и не давал ролику
ничего»), просто про два других источника.

    grep -n "pixabay\\|unsplash" scripts/pipeline_smart.py   ->   ПУСТО

При этом `stock_fetch_multisource.py` реализует Pixabay (фото и видео) и
Unsplash (фото) полностью, оба ключа стоят в `config.example.env`, а
ЧАСТЬ 14 CLAUDE.md описывает мультисток как ДЕФОЛТ заполнения. То есть
`pipeline_smart.py` — реальный путь отбора — не вызывал их ни разу, и вклад
обоих источников в собранный ролик был ровно нулевым.

Почему это бьёт в узкое место: ограничение пайплайна — голод пула (259
слотов на 39 авторских запросов = 6.6 уникальных медиа на запрос), а
расширение упирается в квоту Pexels 200/час. У Pixabay и Unsplash квоты
свои — пул растёт, не отнимая у Pexels ни одного вызова.
"""
import json
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]

import pipeline_smart as ps  # noqa: E402
import stock_fetch_multisource as ms  # noqa: E402


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return json.dumps(self._payload).encode()


@pytest.fixture(autouse=True)
def _clear_caches():
    ps._PIXABAY_PHOTO_CACHE.clear()
    ps._PIXABAY_VIDEO_CACHE.clear()
    ps._UNSPLASH_PHOTO_CACHE.clear()
    ps._UNSPLASH_CALLS_THIS_RUN[0] = 0
    yield
    ps._PIXABAY_PHOTO_CACHE.clear()
    ps._PIXABAY_VIDEO_CACHE.clear()
    ps._UNSPLASH_PHOTO_CACHE.clear()
    ps._UNSPLASH_CALLS_THIS_RUN[0] = 0


PIXABAY_PHOTOS = {"hits": [
    {"id": 111, "tags": "knight, armor, medieval",
     "pageURL": "https://pixabay.com/photos/knight-armor-medieval-111/",
     "largeImageURL": "https://cdn.pixabay.com/111_1280.jpg"},
    {"id": 222, "tags": "reenactment festival",
     "pageURL": "https://pixabay.com/photos/reenactment-festival-222/",
     "largeImageURL": "https://cdn.pixabay.com/222_1280.jpg"},
]}

PIXABAY_VIDEOS = {"hits": [
    {"id": 333, "tags": "castle, siege", "duration": 22,
     "pageURL": "https://pixabay.com/videos/castle-siege-333/",
     "videos": {"large": {"url": "https://cdn.pixabay.com/333_l.mp4", "width": 1920},
                "small": {"url": "https://cdn.pixabay.com/333_s.mp4", "width": 640}}},
]}

UNSPLASH_PHOTOS = {"results": [
    {"id": "aBc123", "alt_description": "medieval stone fortress wall",
     "links": {"html": "https://unsplash.com/photos/medieval-fortress-aBc123"},
     "urls": {"regular": "https://images.unsplash.com/aBc123?w=1080"}},
]}


def _patch(monkeypatch, payload, key_attr="PIXABAY_API_KEY", key="k"):
    monkeypatch.setattr(ms, key_attr, key, raising=False)
    monkeypatch.setattr(ps.urllib.request, "urlopen",
                        lambda *a, **kw: _FakeResponse(payload))


# ---------- флаг реально выключает источник ----------

@pytest.mark.parametrize("flag,fn", [
    ("PIXABAY_ENABLED", "_pixabay_search_photos"),
    ("PIXABAY_ENABLED", "_pixabay_search_videos"),
    ("UNSPLASH_ENABLED", "_unsplash_search_photos"),
])
def test_disabled_flag_never_touches_network(monkeypatch, flag, fn):
    monkeypatch.setenv(flag, "0")

    def _boom(*a, **kw):
        raise AssertionError("сеть не должна дёргаться при выключенном флаге")

    monkeypatch.setattr(ps.urllib.request, "urlopen", _boom)
    assert getattr(ps, fn)("medieval knight") == []


@pytest.mark.parametrize("flag,fn,attr", [
    ("PIXABAY_ENABLED", "_pixabay_search_photos", "PIXABAY_API_KEY"),
    ("PIXABAY_ENABLED", "_pixabay_search_videos", "PIXABAY_API_KEY"),
    ("UNSPLASH_ENABLED", "_unsplash_search_photos", "UNSPLASH_ACCESS_KEY"),
])
def test_missing_key_is_a_silent_noop(monkeypatch, flag, fn, attr):
    """Без ключа источник ничего не меняет — ровно прежнее поведение."""
    monkeypatch.setenv(flag, "1")
    monkeypatch.setattr(ms, attr, "", raising=False)

    def _boom(*a, **kw):
        raise AssertionError("без ключа запроса быть не должно")

    monkeypatch.setattr(ps.urllib.request, "urlopen", _boom)
    assert getattr(ps, fn)("medieval knight") == []


# ---------- нормализация в форму Pexels-кандидата ----------

def test_pixabay_photo_is_shaped_like_a_pexels_candidate(monkeypatch):
    monkeypatch.setenv("PIXABAY_ENABLED", "1")
    _patch(monkeypatch, PIXABAY_PHOTOS)
    got = ps._pixabay_search_photos("medieval knight")
    assert len(got) == 2
    first = got[0]
    assert first["id"] == "pixabay:111"
    assert first["src"]["large2x"] == "https://cdn.pixabay.com/111_1280.jpg"
    assert first["tags"] == ["knight", "armor", "medieval"]


def test_candidate_id_is_a_string_so_dedup_cannot_collide_with_pexels(monkeypatch):
    """ID Pexels — числа. Строковый префикс гарантирует, что общий used_ids
    не спутает pixabay:111 с фотографией Pexels №111 (тот же приём, что у
    openverse:/met:)."""
    monkeypatch.setenv("PIXABAY_ENABLED", "1")
    monkeypatch.setenv("UNSPLASH_ENABLED", "1")
    _patch(monkeypatch, PIXABAY_PHOTOS)
    pix = ps._pixabay_search_photos("q")
    _patch(monkeypatch, UNSPLASH_PHOTOS, "UNSPLASH_ACCESS_KEY")
    uns = ps._unsplash_search_photos("q")
    for c in pix + uns:
        assert isinstance(c["id"], str)
        assert c["id"].split(":")[0] in ("pixabay", "unsplash")


def test_genre_blocklist_can_fire_on_the_new_sources(monkeypatch):
    """ГЛАВНОЕ требование «в один пул под те же гейты»: жанровый фильтр по
    тексту кандидата обязан читать новые источники так же, как Pexels."""
    monkeypatch.setenv("PIXABAY_ENABLED", "1")
    _patch(monkeypatch, PIXABAY_PHOTOS)
    got = ps._pixabay_search_photos("medieval knight")
    texts = [ps.pexels_candidate_text(c) for c in got]
    assert "reenactment" in texts[1], texts[1]
    kept = ps.filter_alt_blocklist(got)
    assert [c["id"] for c in kept] == ["pixabay:111"], \
        "костюмированный фестиваль должен отсеиваться тем же блоклистом"


def test_pixabay_video_carries_duration_and_video_files(monkeypatch):
    """duration — то самое поле, которого не хватало отбору видео (у Pexels
    оно было и не читалось; здесь оно есть и читается сразу)."""
    monkeypatch.setenv("PIXABAY_ENABLED", "1")
    _patch(monkeypatch, PIXABAY_VIDEOS)
    got = ps._pixabay_search_videos("castle siege")
    assert len(got) == 1
    v = got[0]
    assert v["duration"] == 22
    widths = sorted(f["width"] for f in v["video_files"])
    assert widths == [640, 1920]
    assert all(f["file_type"] == "video/mp4" for f in v["video_files"])
    assert not ps._video_candidate_too_short(v, 12.0)
    assert ps._video_candidate_too_short({"duration": 2}, 12.0)


# ---------- квота и устойчивость ----------

def test_unsplash_stops_at_its_own_hourly_cap(monkeypatch):
    monkeypatch.setenv("UNSPLASH_ENABLED", "1")
    _patch(monkeypatch, UNSPLASH_PHOTOS, "UNSPLASH_ACCESS_KEY")
    ps._UNSPLASH_CALLS_THIS_RUN[0] = ms.UNSPLASH_HOURLY_CAP
    assert ps._unsplash_search_photos("medieval") == []


def test_unsplash_counts_its_calls(monkeypatch):
    monkeypatch.setenv("UNSPLASH_ENABLED", "1")
    _patch(monkeypatch, UNSPLASH_PHOTOS, "UNSPLASH_ACCESS_KEY")
    ps._unsplash_search_photos("medieval a")
    ps._unsplash_search_photos("medieval b")
    assert ps._UNSPLASH_CALLS_THIS_RUN[0] == 2


@pytest.mark.parametrize("fn,attr", [
    ("_pixabay_search_photos", "PIXABAY_API_KEY"),
    ("_pixabay_search_videos", "PIXABAY_API_KEY"),
    ("_unsplash_search_photos", "UNSPLASH_ACCESS_KEY"),
])
def test_source_failure_is_fail_open_not_a_crash(monkeypatch, fn, attr):
    """Упавший источник не роняет слот — у него есть рабочий путь Pexels."""
    monkeypatch.setenv("PIXABAY_ENABLED", "1")
    monkeypatch.setenv("UNSPLASH_ENABLED", "1")
    monkeypatch.setattr(ms, attr, "k", raising=False)

    def _boom(*a, **kw):
        raise OSError("сеть отвалилась")

    monkeypatch.setattr(ps.urllib.request, "urlopen", _boom)
    assert getattr(ps, fn)("medieval") == []


def test_result_is_cached_per_process(monkeypatch):
    """Квота тратится один раз на уникальный запрос, а не на каждый слот."""
    monkeypatch.setenv("PIXABAY_ENABLED", "1")
    calls = []
    monkeypatch.setattr(ms, "PIXABAY_API_KEY", "k", raising=False)

    def _count(*a, **kw):
        calls.append(1)
        return _FakeResponse(PIXABAY_PHOTOS)

    monkeypatch.setattr(ps.urllib.request, "urlopen", _count)
    ps._pixabay_search_photos("same query")
    ps._pixabay_search_photos("same query")
    assert len(calls) == 1


# ---------- источники реально вкручены в пул ----------

def test_sources_are_actually_called_from_the_pools():
    """Тот самый пробел, ради которого написан модуль: функция может
    существовать и не вызываться ни разу."""
    import inspect
    photo = inspect.getsource(ps.pexels_photo)
    video = inspect.getsource(ps.pexels_video)
    assert "_pixabay_search_photos(api_q)" in photo
    assert "_unsplash_search_photos(api_q)" in photo
    assert "_pixabay_search_videos(api_q)" in video


def test_flags_participate_in_the_selection_signature():
    """Включение источника меняет состав пула и победителя — без подписи
    это не дошло бы до экрана на прогретом temp_smart/."""
    import inspect
    sig = inspect.getsource(ps._selection_stack_signature)
    assert "PIXABAY_ENABLED" in sig
    assert "UNSPLASH_ENABLED" in sig


def test_new_sources_are_appended_after_pexels(monkeypatch):
    """Порядок «в конец» оставляет взаимный порядок уже существовавших
    кандидатов прежним: новый источник выигрывает слот только по скорингу,
    а не потому что оказался раньше в списке при равенстве."""
    import inspect
    photo = inspect.getsource(ps.pexels_photo)
    assert photo.index("_pexels_search_photos(api_q)") < photo.index("_pixabay_search_photos(api_q)")
    assert photo.index("_pixabay_search_photos(api_q)") < photo.index("_unsplash_search_photos(api_q)")
