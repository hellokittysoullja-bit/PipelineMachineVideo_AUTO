"""Машинный сток Шага 4 больше не обходит отбор: он кандидат общей кучи.

РЕАЛЬНЫЙ пробел, найденный прямой проверкой цепочки. В пайплайне два
независимых механизма подбора, и приоритет у того, у которого нет ни одной
проверки:

    Шаг 4  stock_fetch_multisource.py — round-robin по чётности слота,
           запрос из словаря themes.json, и НИ ОДНОГО упоминания
           relevance/CLIP/домен-гварда/негативного вето во всём файле;
    Шаг 7  pipeline_smart.py — смысловое назначение запроса + полный стек.

`use_local` взводится от наличия media/{index+1:03d}_*, а
`photo = local_photo(i) if use_local else None` проверяется РАНЬШЕ Pexels.
То есть выполнение документированного Шага 4 отключало гейты для каждого
закрытого им слота.

САМОЕ ВАЖНОЕ в этих тестах — не то, что гейт работает, а то, что он НЕ
ТРОГАЕТ курируемые человеком картинки: по ЧАСТИ 14 AI-картинки
оцениваются, но не заменяются никогда.
"""
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]

import pipeline_smart as ps  # noqa: E402


@pytest.mark.parametrize("name", [
    "007_stock.jpg", "001_stock.jpeg", "042_stock.png",
    "/abs/path/media/013_stock.jpg",
])
def test_machine_stock_is_recognised(name):
    assert ps.local_file_is_machine_stock(name)


@pytest.mark.parametrize("name", [
    "001_flow.jpg", "002_grok.jpg", "003_fastgen.jpg", "004_ai.jpg",
])
def test_curated_ai_images_are_never_gated(name):
    """ЧАСТЬ 14: AI-картинки курирует человек, они не заменяются никогда."""
    assert not ps.local_file_is_machine_stock(name)


@pytest.mark.parametrize("name", [
    "my_photo.jpg", "001.jpg", "knight_closeup.png", "hook_frame_2.jpg",
])
def test_unknown_names_are_treated_as_curated(name):
    """Fail-open в сторону человека: выбросить кадр, который кто-то выбрал
    сам, хуже, чем пропустить один машинный."""
    assert not ps.local_file_is_machine_stock(name)


def test_curated_suffix_wins_over_stock_marker():
    """Пограничный случай: если в имени есть и то и другое, побеждает
    курируемость — ошибка в эту сторону дешевле."""
    assert not ps.local_file_is_machine_stock("005_stock_flow.jpg")


def test_machine_stock_is_not_placed_directly():
    """Файл Шага 4 больше не встаёт в слот мимо судьи: main() обнуляет
    photo и отдаёт слот обычному отбору, где файл — кандидат кучи."""
    import inspect
    body = inspect.getsource(ps.main)
    idx = body.index("local_file_is_machine_stock(photo)")
    tail = body[idx:idx + 400]
    assert "LOCAL_STOCK_POOLED.append" in tail
    assert "photo = None" in tail
    assert "is_relevant_candidate(photo" not in tail


def _media(monkeypatch, tmp_path, names):
    from PIL import Image
    media = tmp_path / "media"
    media.mkdir()
    for n in names:
        Image.new("RGB", (64, 48), (120, 90, 60)).save(media / n)
    monkeypatch.setattr(ps, "MEDIA_FOLDER", str(media))
    monkeypatch.setattr(ps, "_LOCAL_PHOTOS_CACHE", None)
    return media


def test_local_stock_candidate_is_a_readable_pool_candidate(monkeypatch, tmp_path):
    _media(monkeypatch, tmp_path, ["007_stock.jpg"])
    monkeypatch.setenv("LOCAL_STOCK_GATE", "1")
    cand = ps.local_stock_candidate(6)
    assert cand["id"] == "local:007_stock.jpg"
    assert ps.candidate_source(cand) == "local"
    assert cand["url"] == ""   # слова пути не идут в жанровый фильтр
    dest = str(tmp_path / "probe.jpg")
    import urllib.request
    ps.atomic_url_download(urllib.request.Request(ps.candidate_probe_url(cand)), dest, timeout=5)
    assert os.path.getsize(dest) == os.path.getsize(tmp_path / "media" / "007_stock.jpg")
    assert ps.local_stock_candidate(5) is None


def test_curated_and_disabled_are_not_candidates(monkeypatch, tmp_path):
    _media(monkeypatch, tmp_path, ["001_flow.jpg", "002_stock.jpg"])
    monkeypatch.setenv("LOCAL_STOCK_GATE", "1")
    assert ps.local_stock_candidate(0) is None      # AI-картинку ставит человек
    monkeypatch.setenv("LOCAL_STOCK_GATE", "0")
    assert ps.local_stock_candidate(1) is None


def _request(index):
    import selection_engine
    return selection_engine.SlotRequest(
        index=index, query="medieval dagger", extra_queries=(), text_key=None,
        shot_brief=None, shot_spec=None, block_text="Вот кинжал.", arbiter_text=None,
        is_opening=False, slot_dur=4.0, action_qualifier=None, target_luma=None,
        director_score_fn=None, director_assist=False, director_report=None,
        video_score_fn=None, used_photo_ids=set(), used_video_ids=set(),
        used_hashes=[], recent_sizes=[])


def test_stock_file_enters_the_photo_pool(monkeypatch, tmp_path):
    _media(monkeypatch, tmp_path, ["003_stock.jpg"])
    monkeypatch.setenv("LOCAL_STOCK_GATE", "1")
    for f in ("_shelf_search_photos", "_museum_search_photos", "_commons_search_photos",
              "_openverse_search_photos", "_pexels_search_photos",
              "_pixabay_search_photos", "_unsplash_search_photos"):
        monkeypatch.setattr(ps, f, lambda *a, **k: [{"id": 111, "alt": "a dagger",
                                                      "url": "", "src": {"medium": "x"}}])
    per_source = ps.PHOTO_ADAPTER.sources(_request(2), "medieval dagger")
    assert per_source[0][0]["id"] == "local:003_stock.jpg"
    assert per_source[0][0]["_origin_query"] == "medieval dagger"
    # слот без файла Шага 4 — куча прежняя
    assert all(c["id"] != "local:003_stock.jpg"
               for lst in ps.PHOTO_ADAPTER.sources(_request(0), "medieval dagger") for c in lst)


def test_cache_key_changes_only_when_stock_file_exists(monkeypatch, tmp_path):
    monkeypatch.setattr(ps, "TEMP_FOLDER", str(tmp_path / "temp"))
    monkeypatch.setenv("LOCAL_STOCK_GATE", "1")
    monkeypatch.setattr(ps, "MEDIA_FOLDER", str(tmp_path / "nomedia"))
    monkeypatch.setattr(ps, "_LOCAL_PHOTOS_CACHE", None)
    before = ps.PHOTO_ADAPTER.cache_path(_request(2))
    _media(monkeypatch, tmp_path, ["003_stock.jpg"])
    with_file = ps.PHOTO_ADAPTER.cache_path(_request(2))
    assert with_file != before
    assert ps.PHOTO_ADAPTER.cache_path(_request(0)) != with_file


def test_flag_is_registered_with_default_on():
    import feature_flags as ff
    assert ff.FLAGS["LOCAL_STOCK_GATE"].default == "1"
