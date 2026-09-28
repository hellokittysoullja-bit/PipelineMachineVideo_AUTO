#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Судья не смог — решение слота временное (сбой провайдера 27.09).

Прерванный прогон эпизода 94: Qwen и DeepSeek отвечали 502/503 на 78%
попыток. Слот платной зоны, где судья не ответил, решался без него — и
кэшировался под той же подписью отбора, что решение С судьёй: следующий
рендер брал этот кадр готовым, навсегда. Теперь причина пишется рядом с
кадром, и следующий рендер решает такой слот заново."""
import dataclasses
import json
import os
import sys

import pytest
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.argv = ["pipeline_smart.py", REPO]
import pipeline_smart as ps  # noqa: E402
import selection_engine  # noqa: E402


@pytest.fixture
def judge_on(monkeypatch):
    monkeypatch.setenv("SHOT_JUDGE", "1")
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", "k")
    ps.JUDGE_GAPS.clear()
    ps.POOL_GAPS.clear()
    yield
    ps.JUDGE_GAPS.clear()
    ps.POOL_GAPS.clear()


def _req(index):
    fields = {f.name: None for f in dataclasses.fields(selection_engine.SlotRequest)}
    fields.update(index=index, query="medieval dagger", extra_queries=(), block_text="Вот кинжал.",
                  is_opening=False, director_assist=False)
    return selection_engine.SlotRequest(**fields)


def _cached(path, **side):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (64, 36), (90, 60, 30)).save(path)
    with open(ps.media_sidecar_path(path), "w", encoding="utf-8") as f:
        json.dump(dict({"pexels_id": "met:1", "kind": "photo"}, **side), f)
    return path


def test_gap_is_noted_only_where_the_judge_was_due(judge_on):
    ps.note_judge_gap(0, "502")
    ps.note_judge_gap(0, "второй сбой не перезаписывает первый")
    ps.note_judge_gap(ps.SHOT_JUDGE_PAID_SLOTS, "бесплатная зона — судьи там нет")
    assert ps.JUDGE_GAPS == {0: "502"}


def test_an_unusable_answer_is_not_a_gap(judge_on):
    """Модель ответила, но ответ пуст или не разобран — повтор того же вопроса
    скорее всего даст то же: слот не перерешался бы никогда."""
    for reason in ("отсев по подписи: EmptyAnswer: ds/deepseek-v4-flash: пустой ответ",
                   "проверка кадра: неразобранный ответ: ...", "сетка судьи: неполный ответ: 3 из 9"):
        ps.note_judge_gap(1, reason)
    assert ps.JUDGE_GAPS == {}


def test_an_image_that_never_reached_the_model_is_a_gap(judge_on):
    """«The image failed to upload» — сбой доставки: шлюз повторил вызов и
    сдался; кадр проверен не был, слот решается заново следующим рендером
    (раньше тот же ответ шёл «неразобранным» и решение становилось
    окончательным)."""
    ps.note_judge_gap(2, "проверка кадра: ImageNotReceived: qwen/qwen3.7-plus: картинка не дошла до модели")
    assert list(ps.JUDGE_GAPS) == [2]


def test_no_gap_without_the_judge(monkeypatch):
    monkeypatch.setenv("SHOT_JUDGE", "0")
    ps.JUDGE_GAPS.clear()
    ps.note_judge_gap(0, "502")
    assert ps.JUDGE_GAPS == {}


def test_gap_marks_the_shown_frame_and_the_cache_rejects_it(judge_on):
    cf = _cached(os.path.join(ps.TEMP_FOLDER, "pexels_cache", "0000_ab_cd.jpg"))
    ps.note_judge_gap(0, "сетка судьи: GatewayError 502")
    assert ps.mark_provisional_media(0, cf) == "сетка судьи: GatewayError 502"
    assert ps.read_media_sidecar(cf)["judge_missing"].startswith("сетка судьи")
    assert ps.read_media_sidecar(cf)["pexels_id"] == "met:1", "остальные поля не тронуты"
    req = _req(0)
    assert ps.PhotoAdapter().stale_cache(req, cf), "кадр без судьи — промах кэша"
    assert ps.VideoAdapter().stale_cache(req, cf)


def test_the_engine_does_not_take_a_stale_cache(monkeypatch, tmp_path):
    """Общее ядро отбора: временный кадр в кэше не отдаётся, а отбирается
    заново — одно место для фото и видео."""
    final = tmp_path / "cached.jpg"
    final.write_bytes(b"x")

    class A(selection_engine.MediaAdapter):
        kind = "photo"
        hits = 0

        def cache_path(self, request):
            return str(final)

        def stale_cache(self, request, path):
            return "судья не ответил"

        def cache_hit(self, request, path):
            A.hits += 1
            return path
    monkeypatch.setattr(selection_engine, "build_pool", lambda request, adapter: [])
    import selection_attempt
    att = selection_attempt.Attempt(0, "photo", str(tmp_path / "stage"))
    with selection_attempt.activate(att):
        assert selection_engine.select(_req(0), A()) is None
    assert A.hits == 0


def test_without_the_judge_a_provisional_frame_is_a_normal_hit(monkeypatch):
    """Судьи нет в этом прогоне (нет ключа) — перерешать нечем: кэш годен."""
    monkeypatch.setenv("SHOT_JUDGE", "1")
    monkeypatch.delenv("LLM_GATEWAY_API_KEY", raising=False)
    cf = _cached(os.path.join(ps.TEMP_FOLDER, "pexels_cache", "0001_ab_cd.jpg"),
                 judge_missing="сбой")
    req = _req(1)
    assert ps.provisional_reason(1, cf) is None
    assert not ps.PhotoAdapter().stale_cache(req, cf)


def test_a_human_file_is_never_marked(judge_on, tmp_path):
    own = _cached(str(tmp_path / "media" / "001_flow.jpg"))
    ps.note_judge_gap(0, "сбой")
    assert ps.mark_provisional_media(0, own) is None
    assert "judge_missing" not in ps.read_media_sidecar(own)


def test_clean_slot_is_not_marked(judge_on):
    cf = _cached(os.path.join(ps.TEMP_FOLDER, "pexels_cache", "0002_ab_cd.jpg"))
    assert ps.mark_provisional_media(2, cf) is None
    assert "judge_missing" not in ps.read_media_sidecar(cf)


def test_a_provisional_clip_is_reselected(judge_on):
    """Кэш-хит клипа идёт ДО отбора: без этой проверки временное решение
    доживало бы до следующего рендера в готовом клипе."""
    cf = _cached(os.path.join(ps.TEMP_FOLDER, "pexels_cache", "0003_ab_cd.jpg"),
                 judge_missing="проверка кадра: 502")
    prev = {"shots": [{"index": 3, "text": "Вот кинжал.", "kind": "photo",
                       "file": os.path.relpath(cf, ps.VIDEO_FOLDER)}]}
    assert ps.provisional_clip_reason(3, prev, {"text": "Вот кинжал."}) == "проверка кадра: 502"
    assert ps.provisional_clip_reason(3, prev, {"text": "Другая фраза."}) is None
    prev["shots"][0]["judge_missing"] = "из шотлиста"
    assert ps.provisional_clip_reason(3, prev, {"text": "Вот кинжал."}) == "из шотлиста"


def test_judge_without_gateway_leaves_a_gap(judge_on, monkeypatch, tmp_path):
    p = tmp_path / "a.jpg"
    Image.new("RGB", (32, 32), (60, 0, 0)).save(p)
    c = {"path": str(p), "p": {"id": "a"}, "is_dup_free": 1}
    monkeypatch.setattr(ps, "_shot_judge_gateway", lambda: None)
    monkeypatch.setitem(ps._SHOT_JUDGE_STATE, "refused", "модель x не видит картинок")
    monkeypatch.setitem(ps._SHOT_JUDGE_STATE, "recheck_at", None)
    assert ps.judge_candidates(0, "photo", "Вот кинжал.", "a dagger", [c]) is False
    assert ps.JUDGE_GAPS == {}, "модель не видит картинок — постоянно, слот не временный"
    monkeypatch.setitem(ps._SHOT_JUDGE_STATE, "refused", "проверка зрения не состоялась: 502")
    monkeypatch.setitem(ps._SHOT_JUDGE_STATE, "recheck_at", 1e12)
    assert ps.judge_candidates(0, "photo", "Вот кинжал.", "a dagger", [c]) is False
    assert ps.JUDGE_GAPS[0].startswith("проверка зрения")


def test_grid_failure_leaves_a_gap(judge_on, monkeypatch, tmp_path):
    p = tmp_path / "a.jpg"
    Image.new("RGB", (32, 32), (60, 0, 0)).save(p)
    c = {"path": str(p), "p": {"id": "a"}, "is_dup_free": 1}
    import shot_judge

    def failed(gw, model, *, report=None, **k):
        report["refused"] = "GatewayError: повторы исчерпаны: 502"
        return None
    monkeypatch.setattr(shot_judge, "judge", failed)
    monkeypatch.setattr(shot_judge, "verify_claims", lambda *a, **k: (None, {"refused": "502"}))
    monkeypatch.setattr(ps, "_shot_judge_gateway", lambda: object())
    monkeypatch.setattr(ps, "episode_world_card", lambda video_dir=None: None)
    assert ps.judge_candidates(0, "photo", "Вот кинжал.", "a dagger", [c]) is False
    assert ps.JUDGE_GAPS[0].startswith("сетка судьи") and "502" in ps.JUDGE_GAPS[0]


def test_one_unanswered_finalist_leaves_a_gap(judge_on, monkeypatch, tmp_path):
    """Сетка ответила, проверка одного финалиста — нет: он мог быть лучшим."""
    cs = []
    for k in range(2):
        p = tmp_path / f"{k}.jpg"
        Image.new("RGB", (32, 32), (60 * k, 0, 0)).save(p)
        cs.append({"path": str(p), "p": {"id": str(k)}, "is_dup_free": 1})
    import shot_judge
    monkeypatch.setattr(shot_judge, "judge", lambda gw, model, **k: {"0": 3, "1": 2})

    def verify(gw, model, *, path, **k):
        if path.endswith("1.jpg"):
            return None, {"refused": "GatewayError: 503"}
        return {"c1": "yes"}, {"call": True}
    monkeypatch.setattr(shot_judge, "verify_claims", verify)
    monkeypatch.setattr(ps, "_shot_judge_gateway", lambda: object())
    monkeypatch.setattr(ps, "episode_world_card", lambda video_dir=None: None)
    ps.judge_candidates(0, "photo", "Вот кинжал.", "a dagger", cs)
    assert ps.JUDGE_GAPS[0] == "проверка кадра: GatewayError: 503"


def test_vision_check_broken_by_the_gateway_is_asked_again(judge_on, monkeypatch):
    """Раньше один сбой на старте выключал судью на весь прогон."""
    import shot_judge
    answers = iter([(False, shot_judge.VISION_CHECK_GATEWAY_FAILURE + ": GatewayError 502"),
                    (True, "")])
    calls = []

    def check(gw, model):
        calls.append(1)
        return next(answers)
    monkeypatch.setattr(shot_judge, "vision_check", check)
    now = [1000.0]
    monkeypatch.setattr(ps.time, "monotonic", lambda: now[0])
    ps._SHOT_JUDGE_STATE.update(gateway=None, made=False, refused=None, candidate=None, recheck_at=None)
    try:
        assert ps._shot_judge_gateway() is None
        now[0] += ps.JUDGE_RECHECK_SEC - 1
        assert ps._shot_judge_gateway() is None and len(calls) == 1, "до срока — без вопроса"
        now[0] += 1
        assert ps._shot_judge_gateway() is not None and len(calls) == 2
    finally:
        ps._SHOT_JUDGE_STATE.update(gateway=None, made=False, refused=None, candidate=None,
                                    recheck_at=None)


def test_a_model_that_cannot_see_is_not_asked_again(judge_on, monkeypatch):
    import shot_judge
    calls = []

    def check(gw, model):
        calls.append(1)
        return False, "модель x не видит картинок"
    monkeypatch.setattr(shot_judge, "vision_check", check)
    now = [1000.0]
    monkeypatch.setattr(ps.time, "monotonic", lambda: now[0])
    ps._SHOT_JUDGE_STATE.update(gateway=None, made=False, refused=None, candidate=None, recheck_at=None)
    try:
        assert ps._shot_judge_gateway() is None
        now[0] += 10 * ps.JUDGE_RECHECK_SEC
        assert ps._shot_judge_gateway() is None and len(calls) == 1
    finally:
        ps._SHOT_JUDGE_STATE.update(gateway=None, made=False, refused=None, candidate=None,
                                    recheck_at=None)


# ---------- неполная куча: источник не ответил из-за временного сбоя ----------

def _http(code):
    import io
    import urllib.error
    return urllib.error.HTTPError("u", code, "x", {}, io.BytesIO(b""))


def test_transient_errors_are_told_from_permanent():
    import source_health
    import urllib.error
    assert source_health.transient_error(_http(429))
    assert source_health.transient_error(_http(503))
    assert source_health.transient_error(urllib.error.URLError("reset"))
    assert source_health.transient_error(TimeoutError())
    assert source_health.transient_error(ConnectionResetError())
    assert source_health.transient_error(OSError("Network is unreachable"))
    for permanent in (_http(404), _http(401), _http(403), ValueError("bad json"),
                      FileNotFoundError("x"), KeyError("hits")):
        assert not source_health.transient_error(permanent), permanent


def _in_attempt(tmp_path, index, fn):
    import selection_attempt
    att = selection_attempt.Attempt(index, "photo", str(tmp_path / "stage"))
    with selection_attempt.activate(att):
        return fn()


def test_a_source_that_did_not_answer_makes_the_slot_provisional(tmp_path):
    ps.POOL_GAPS.clear()
    _in_attempt(tmp_path, 40, lambda: ps._note_source_search_error("commons", _http(429), "q"))
    _in_attempt(tmp_path, 41, lambda: ps._note_source_search_error("pixabay", _http(404), "q"))
    assert ps.POOL_GAPS == {40: "commons: HTTP 429"}, "404 повтором не лечится"
    ps.POOL_GAPS.clear()


def test_background_preparation_does_not_mark_slots(tmp_path):
    import slot_prefetch
    ps.POOL_GAPS.clear()
    token = slot_prefetch.TARGET.set(7)
    try:
        _in_attempt(tmp_path, 7, lambda: ps._note_source_search_error("commons", _http(503), "q"))
    finally:
        slot_prefetch.TARGET.reset(token)
    ps._note_source_search_error("commons", _http(503), "q")    # вне попытки
    assert ps.POOL_GAPS == {}


def test_pexels_failure_marks_its_slot(tmp_path):
    ps.POOL_GAPS.clear()
    ps.PhotoAdapter().on_source_failure(_req(42), "pexels", _http(502))
    ps.VideoAdapter().on_source_failure(_req(43), "pexels", _http(401))
    assert ps.POOL_GAPS == {42: "pexels: HTTP 502"}
    ps.POOL_GAPS.clear()


def test_incomplete_museum_answer_marks_the_slot(tmp_path, monkeypatch):
    import museum_sources as ms
    ps.POOL_GAPS.clear()
    ps._MUSEUM_SEARCH_CACHE.clear()
    ms._SEARCH_CACHE.clear()
    monkeypatch.setattr(ms.feature_flags, "enabled", lambda name, *a, **k: name != "MET_CATALOG")
    monkeypatch.setattr(ms, "MUSEUM_CACHE_DIR", str(tmp_path / "museum"))

    def met(q, limit=None, department=None):
        return [{"id": "met:1"}]

    def broken(q, limit=None):
        raise TimeoutError()
    monkeypatch.setattr(ms, "_sources", lambda department=None: [("met", met), ("chicago", broken)])
    _in_attempt(tmp_path, 44, lambda: ps._museum_search_photos("medieval dagger"))
    assert ps.POOL_GAPS == {44: "museum: неполный ответ музеев"}
    ps.POOL_GAPS.clear()


def test_a_permanently_broken_museum_does_not_keep_every_answer_incomplete(tmp_path, monkeypatch):
    """Постоянный отказ (битый ответ) повтором не лечится: неполным ответ с
    ним был бы всегда, и решения слотов не становились бы окончательными."""
    import museum_sources as ms
    ms._SEARCH_CACHE.clear()
    monkeypatch.setattr(ms.feature_flags, "enabled", lambda name, *a, **k: name != "MET_CATALOG")
    monkeypatch.setattr(ms, "MUSEUM_CACHE_DIR", str(tmp_path / "museum"))
    monkeypatch.setattr(ms, "_sources", lambda department=None: [
        ("met", lambda q, limit=None, department=None: [{"id": "met:1"}]),
        ("chicago", lambda q, limit=None: (_ for _ in ()).throw(ValueError("bad json")))])
    report = {}
    assert [c["id"] for c in ms.search_museums("medieval dagger", report=report)] == ["met:1"]
    assert report["complete"] is True


def test_pool_gap_is_provisional_in_any_zone(monkeypatch):
    """Неполная куча — повод перерешить слот и без судьи (бесплатная зона)."""
    monkeypatch.setenv("SHOT_JUDGE", "0")
    ps.POOL_GAPS.clear()
    ps.JUDGE_GAPS.clear()
    idx = ps.SHOT_JUDGE_PAID_SLOTS + 3
    cf = _cached(os.path.join(ps.TEMP_FOLDER, "pexels_cache", f"{idx:04d}_ab_cd.jpg"))
    ps.POOL_GAPS[idx] = "commons: HTTP 429"
    assert ps.mark_provisional_media(idx, cf) == "commons: HTTP 429"
    assert ps.read_media_sidecar(cf)["pool_missing"] == "commons: HTTP 429"
    assert "judge_missing" not in ps.read_media_sidecar(cf)
    assert ps.provisional_reason(idx, cf) == "commons: HTTP 429"
    prev = {"shots": [{"index": idx, "text": "Фраза.", "kind": "photo",
                       "file": os.path.relpath(cf, ps.VIDEO_FOLDER)}]}
    assert ps.provisional_clip_reason(idx, prev, {"text": "Фраза."}) == "commons: HTTP 429"
    ps.POOL_GAPS.clear()
