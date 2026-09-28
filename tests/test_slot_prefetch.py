#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Подготовка следующих слотов в фоне: расписание, общий запрос в полёте,
тишина процессора, одинаковые числа каскада — без сети и без моделей."""
import os
import sys
import threading
import time

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import slot_prefetch  # noqa: E402


def _wait(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


# ---------------------------------------------------------------- расписание

def test_prefetch_stays_inside_the_window_and_follows_the_selection():
    searched, cascaded = [], []
    p = slot_prefetch.Prefetcher(10, searched.append, lambda j: cascaded.append(j) or True,
                                 search_ahead=lambda i: 2, cascade_ahead=lambda i: 1)
    try:
        assert not _wait(lambda: searched, 0.3), "до первого слота отбора подготовка не начинается"
        p.advance(0)
        # Поиск идёт в два потока: порядок внутри окна не гарантирован, окно — да.
        assert _wait(lambda: sorted(searched) == [0, 1, 2])
        assert _wait(lambda: cascaded == [0, 1])
        time.sleep(0.2)
        assert sorted(searched) == [0, 1, 2] and cascaded == [0, 1], "за окно не выходит"
        p.advance(5)
        # Слоты, которые отбор уже прошёл, не готовятся: поздно.
        assert _wait(lambda: 7 in searched)
        assert 3 not in searched and 4 not in searched
        assert _wait(lambda: cascaded[-1] == 6)
    finally:
        p.close()


def test_a_slot_stuck_in_search_does_not_hold_the_next_one():
    """27.09, эп.94: поиск слота 0 ждал Commons 134 с, и поиск слотов 1-3
    стоял за ним в очереди."""
    gate, searched = threading.Event(), []

    def search(j):
        if j == 0:
            gate.wait(5)
        searched.append(j)
    p = slot_prefetch.Prefetcher(4, search, lambda j: True, search_ahead=lambda i: 2,
                                 cascade_ahead=lambda i: 0)
    try:
        p.advance(0)
        assert _wait(lambda: 1 in searched), "слот 1 ждёт застрявший слот 0"
        assert 0 not in searched
        gate.set()
        assert _wait(lambda: sorted(searched) == [0, 1, 2])
    finally:
        p.close()


def test_cascade_waits_for_the_search_of_its_slot():
    gate = threading.Event()
    cascaded = []

    def search(j):
        if j == 1:
            gate.wait(5)
    p = slot_prefetch.Prefetcher(3, search, lambda j: cascaded.append(j) or True,
                                 search_ahead=lambda i: 2, cascade_ahead=lambda i: 2)
    try:
        p.advance(0)
        assert _wait(lambda: cascaded == [0])
        time.sleep(0.2)
        assert cascaded == [0], "куча слота 1 ещё не найдена — оценивать нечего"
        gate.set()
        assert _wait(lambda: cascaded == [0, 1, 2])
    finally:
        p.close()


def test_errors_of_prefetch_never_escape():
    def boom(j):
        raise RuntimeError("источник упал")
    p = slot_prefetch.Prefetcher(2, boom, boom, search_ahead=lambda i: 1, cascade_ahead=lambda i: 1)
    try:
        p.advance(0)
        assert _wait(lambda: p.stats["errors"] >= 3)
    finally:
        p.close()


# ---------------------------------------------------------------- один запрос в полёте

def test_single_flight_shares_the_answer_in_flight():
    sf = slot_prefetch.SingleFlight()
    started, release = threading.Event(), threading.Event()
    calls = []

    def slow():
        calls.append(1)
        started.set()
        release.wait(5)
        return ["кандидат"]
    out = []
    t = threading.Thread(target=lambda: out.append(sf.call("k", slow)))
    t.start()
    started.wait(5)
    t2 = threading.Thread(target=lambda: out.append(sf.call("k", slow)))
    t2.start()
    time.sleep(0.1)
    release.set()
    t.join(5)
    t2.join(5)
    assert out == [["кандидат"], ["кандидат"]] and len(calls) == 1 and sf.shared == 1


def test_single_flight_does_not_pass_a_failure_on():
    """Сбой подготовки не становится сбоем отбора: второй спрашивает сам."""
    sf = slot_prefetch.SingleFlight()
    started, release = threading.Event(), threading.Event()

    def failing():
        started.set()
        release.wait(5)
        raise RuntimeError("429")
    errors = []

    def first():
        try:
            sf.call("k", failing)
        except RuntimeError as e:
            errors.append(str(e))
    t = threading.Thread(target=first)
    t.start()
    started.wait(5)
    got = []
    t2 = threading.Thread(target=lambda: got.append(sf.call("k", lambda: "свой ответ")))
    t2.start()
    time.sleep(0.1)
    release.set()
    t.join(5)
    t2.join(5)
    assert errors == ["429"] and got == ["свой ответ"]


# ---------------------------------------------------------------- один вычислитель моделей

def test_background_turn_waits_for_quiet_after_selection_work():
    clock = slot_prefetch.ForegroundClock(0.3)
    with clock.model():
        pass
    t0 = time.monotonic()
    with clock.background_turn() as turn:
        assert turn
    assert time.monotonic() - t0 >= 0.25


def test_selection_and_background_never_compute_at_the_same_time():
    """Замер 27.09: две пачки модели одновременно — 61 с, по очереди — 4.5 с.
    Пока фон держит модель, отбор ждёт; отбор, вставший в очередь, идёт
    раньше следующей фоновой пачки."""
    clock = slot_prefetch.ForegroundClock(0.05)
    inside = []
    order = []

    def bg():
        with clock.background_turn() as turn:
            assert turn
            inside.append("bg")
            time.sleep(0.3)
            assert inside == ["bg"], "отбор вошёл в модель, пока считал фон"
            inside.remove("bg")
            order.append("bg1")
    t = threading.Thread(target=bg)
    t.start()
    assert _wait(lambda: inside == ["bg"])

    def fg():
        with clock.model():
            assert not inside
            order.append("fg")
    f = threading.Thread(target=fg)
    f.start()
    t.join(5)
    f.join(5)
    assert order == ["bg1", "fg"]


def test_background_calls_inside_the_turn_are_not_selection_work():
    clock = slot_prefetch.ForegroundClock(5.0)
    clock._last = 0.0
    with clock.background_turn() as turn:
        assert turn
        with clock.model():      # вызов модели изнутри фоновой пачки
            pass
    assert clock._last == 0.0, "фоновая работа не должна задерживать сама себя"


def test_stop_interrupts_the_wait():
    clock = slot_prefetch.ForegroundClock(60.0)
    with clock.model():
        pass
    with clock.background_turn(stopped=lambda: True) as turn:
        assert turn is False


# ---------------------------------------------------------------- числа каскада

@pytest.fixture
def ps(monkeypatch, tmp_path):
    pytest.importorskip("PIL")
    import importlib
    sys.argv = ["pipeline_smart.py", str(tmp_path)]
    import pipeline_smart
    monkeypatch.setattr(pipeline_smart, "TEMP_FOLDER", str(tmp_path / "temp_smart"))
    monkeypatch.setattr(pipeline_smart, "_CASCADE_EMB", {})
    monkeypatch.setattr(pipeline_smart, "_GATE_IMG_EMB_CACHE", {})
    return importlib.import_module("pipeline_smart")


def test_every_model_batch_has_the_same_size(ps, monkeypatch):
    """Неполная пачка добивается копиями последнего кадра: число кадра не
    должно зависеть от того, в какой пачке он оказался (замер: без добивки
    расходится до 4e-7)."""
    import numpy as np
    sizes = []

    def fake(images=None, text=None):
        sizes.append(len(images))
        return np.arange(len(images), dtype="float32")[:, None] * np.ones((1, 4), dtype="float32")
    monkeypatch.setattr(ps, "_gate_embed", fake)
    out = ps.gate_embed_batch(["a", "b", "c"])
    assert sizes == [ps.CASCADE_BATCH] and len(out) == 3


def _fake_candidates(n):
    return [{"id": f"c{k}", "src": {"large": f"http://img/{k}.jpg"}} for k in range(n)]


def test_prefetched_embeddings_are_reused_by_the_selection(ps, monkeypatch, tmp_path):
    """Подготовка посчитала эмбеддинги кучи — отбор берёт их из кэша и ни
    одного кадра не считает второй раз; числа — те же, что посчитал бы он."""
    import numpy as np
    from PIL import Image
    calls = []

    def fake_embed(images=None, text=None):
        calls.append(len(images) if images is not None else "text")
        if images is None:
            return np.ones((1, 4), dtype="float32")
        return np.stack([np.asarray(im.resize((2, 2))).astype("float32").ravel()[:4] + 1
                         for im in images])
    monkeypatch.setattr(ps, "_gate_embed", fake_embed)

    def probe(p, dest):
        k = int(p["id"][1:])
        Image.new("RGB", (8, 8), (k * 10 % 256, 20, 30)).save(dest)
    cands = _fake_candidates(20)
    url_of = lambda p: p["src"]["large"]  # noqa: E731
    warm, fresh = ps.cascade_embed(cands, url_of, probe, str(tmp_path / "w_"),
                                   background=True)
    assert fresh == 20 and calls == [16, 16]
    calls.clear()
    got, fresh2 = ps.cascade_embed(cands, url_of, probe, str(tmp_path / "m_"))
    assert fresh2 == 0 and calls == [], "отбор не должен считать заново"
    for p in cands:
        assert np.array_equal(got[id(p)], warm[id(p)])
    # Тот же кадр, посчитанный отбором сам (без подготовки), — те же числа.
    ps._CASCADE_EMB.clear()
    import shutil
    shutil.rmtree(os.path.join(ps.TEMP_FOLDER, "cascade_embed_cache"))
    fresh_self = ps.cascade_embed(cands, url_of, probe, str(tmp_path / "s_"))[0]
    for p in cands:
        assert np.array_equal(fresh_self[id(p)], warm[id(p)])


def test_prefetch_pool_never_spends_pexels_quota(ps, monkeypatch):
    """Pexels и Unsplash подготовка не спрашивает: остаток их квоты решает
    состав кучи. Уже спрошенное отбором — берёт из кэша."""
    asked = []

    def spy(name):
        def f(api_query, *a, **k):
            asked.append((name, api_query))
            return [{"id": f"{name}:{api_query}", "src": {"large": "u"}, "alt": ""}]
        return f
    for name in ("_pexels_search_photos", "_unsplash_search_photos", "_pixabay_search_photos",
                 "_openverse_search_photos", "_commons_search_photos", "_museum_search_photos",
                 "_shelf_search_photos"):
        monkeypatch.setattr(ps, name, spy(name))
    monkeypatch.setattr(ps, "_PEXELS_SEARCH_CACHE", {})
    monkeypatch.setattr(ps, "local_stock_candidate", lambda i: None)
    monkeypatch.setattr(ps, "art_museums_fit_episode", lambda: True)
    req = ps.build_slot_request(
        index=0, query="medieval dagger", extra_queries=("knight armour",), text_key=None,
        shot_brief=None, block_text="Вот кинжал.", shot_spec=None, arbiter_text=None,
        is_opening=False, slot_dur=5.0, action_qualifier=None, target_luma=None,
        director_score_fn=None, director_assist=False, director_report=None,
        video_score_fn=None, used_photo_ids=None, used_video_ids=None, used_hashes=None,
        recent_sizes=None)
    pool = ps.prefetch_pool(req, "photo")
    names = {n for n, _q in asked}
    assert "_pexels_search_photos" not in names and "_unsplash_search_photos" not in names
    assert "_museum_search_photos" in names and pool
    # Отбор уже спросил Pexels этим запросом — подготовка берёт его выдачу.
    api_q = ps.stock_api_query(req, "medieval dagger")
    ps._PEXELS_SEARCH_CACHE[api_q] = [{"id": 777, "src": {"large": "p"}, "alt": ""}]
    asked.clear()
    pool = ps.prefetch_pool(req, "photo")
    assert any(c.get("id") == 777 for c in pool)
    assert "_pexels_search_photos" not in {n for n, _q in asked}


def test_prefetch_source_failure_is_not_recorded_as_a_selection_failure(ps, monkeypatch):
    for name in ("_pixabay_search_photos", "_openverse_search_photos", "_commons_search_photos",
                 "_shelf_search_photos", "_pexels_search_photos", "_unsplash_search_photos"):
        monkeypatch.setattr(ps, name, lambda *a, **k: [])

    def boom(*a, **k):
        raise RuntimeError("503")
    monkeypatch.setattr(ps, "_museum_search_photos", boom)
    monkeypatch.setattr(ps, "local_stock_candidate", lambda i: None)
    monkeypatch.setattr(ps, "art_museums_fit_episode", lambda: True)
    recorded = []
    monkeypatch.setattr(ps, "_note_source_search_error", lambda *a, **k: recorded.append(a))
    before = ps.PREFETCH_STATS["source_errors"]
    req = ps.build_slot_request(
        index=0, query="medieval dagger", extra_queries=(), text_key=None,
        shot_brief=None, block_text="x", shot_spec=None, arbiter_text=None,
        is_opening=False, slot_dur=5.0, action_qualifier=None, target_luma=None,
        director_score_fn=None, director_assist=False, director_report=None,
        video_score_fn=None, used_photo_ids=None, used_video_ids=None, used_hashes=None,
        recent_sizes=None)
    assert ps.prefetch_pool(req, "photo") == []
    assert recorded == [] and ps.PREFETCH_STATS["source_errors"] == before + 1


def test_in_the_paid_zone_prefetch_asks_pexels_only_the_slot_own_query(ps, monkeypatch):
    """Собственный запрос слота отбор делает при любом остатке квоты — его
    подготовка делает раньше; дополнительные запросы (зависят от остатка) —
    нет."""
    asked = []

    def pexels(api_query, *a, **k):
        asked.append(api_query)
        return [{"id": 1, "src": {"large": "u"}, "alt": ""}]
    monkeypatch.setattr(ps, "_pexels_search_photos", pexels)
    for name in ("_unsplash_search_photos", "_pixabay_search_photos", "_openverse_search_photos",
                 "_commons_search_photos", "_museum_search_photos", "_shelf_search_photos"):
        monkeypatch.setattr(ps, name, lambda *a, **k: [])
    monkeypatch.setattr(ps, "_PEXELS_SEARCH_CACHE", {})
    monkeypatch.setattr(ps, "local_stock_candidate", lambda i: None)
    monkeypatch.setattr(ps, "art_museums_fit_episode", lambda: True)
    monkeypatch.setattr(ps, "shot_judge_active", lambda index=None: True)
    req = ps.build_slot_request(
        index=0, query="medieval dagger", extra_queries=("knight armour",), text_key=None,
        shot_brief=None, block_text="Вот кинжал.", shot_spec=None, arbiter_text=None,
        is_opening=False, slot_dur=5.0, action_qualifier=None, target_luma=None,
        director_score_fn=None, director_assist=False, director_report=None,
        video_score_fn=None, used_photo_ids=None, used_video_ids=None, used_hashes=None,
        recent_sizes=None)
    ps.prefetch_pool(req, "photo")
    assert asked == [ps.stock_api_query(req, "medieval dagger")]


# ---------------------------------------------------------------- по первым ответам

def test_eager_takes_the_nearest_slot_first():
    """Работа текущего слота не ждёт в очереди за слотами подготовки."""
    gate, seen = threading.Event(), []
    e = slot_prefetch.Eager("t-eager-prio")
    try:
        e.submit(0, gate.wait, 5)             # занять поток, пока очередь набирается
        for target in (5, 3, 1, 3):
            e.submit(target, lambda t=target: seen.append(t))
        gate.set()
        assert _wait(lambda: len(seen) == 4)
        assert seen == [1, 3, 3, 5]
    finally:
        e.close()


def test_eager_runs_jobs_in_order_with_their_slot_and_survives_errors():
    seen = []
    e = slot_prefetch.Eager("t-eager")
    try:
        e.submit(3, lambda: seen.append(("a", slot_prefetch.TARGET.get())))
        e.submit(4, lambda: 1 / 0)
        e.submit(5, lambda: seen.append(("b", slot_prefetch.TARGET.get())))
        assert _wait(lambda: len(seen) == 2)
        assert seen == [("a", 3), ("b", 5)] and e.stats == {"jobs": 2, "errors": 1}
        assert slot_prefetch.TARGET.get() is None, "метка слота не протекает наружу"
    finally:
        e.close()
    e.submit(6, lambda: seen.append("after close"))
    time.sleep(0.1)
    assert "after close" not in seen


def test_answer_of_a_fast_source_is_handed_over_before_the_slow_one_ends():
    """27.09, эп.94, слот 0: музеи и стоки ответили за секунды, Commons —
    через 134 с из-за лимита, и всё это время процессор стоял."""
    import dataclasses
    import selection_engine
    fields = {f.name: None for f in dataclasses.fields(selection_engine.SlotRequest)}
    fields.update(index=0, query="q", extra_queries=())
    request = selection_engine.SlotRequest(**fields)
    handed = threading.Event()
    got = []

    class A:
        def source_jobs(self, request, pq):
            def fast():
                return ["m1", "m2"]

            def slow():
                assert handed.wait(5), "ответ быстрого источника не передан до конца медленного"
                return ["c1"]
            return [("museum", fast), ("commons", slow)]

        def on_source_result(self, request, pq, candidates):
            got.append((pq, list(candidates)))
            if candidates == ["m1", "m2"]:
                handed.set()
            raise RuntimeError("сбой подсказки не трогает кучу")

        def on_source_failure(self, request, name, exc):
            raise AssertionError(f"{name}: {exc}")
    out = selection_engine.fetch_sources(request, A(), ["q"])
    assert out == {"q": [["m1", "m2"], ["c1"]]}
    assert got == [("q", ["m1", "m2"]), ("q", ["c1"])]


def test_eager_cascade_only_in_the_paid_zone_and_only_with_preparation(ps, monkeypatch):
    calls = []
    monkeypatch.setenv("SHOT_JUDGE", "1")
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", "k")
    monkeypatch.setattr(ps, "cascade_embed",
                        lambda head, url_of, probe, prefix, **k: calls.append(
                            (len(head), url_of, probe, k.get("index"), k.get("background"))) or ({}, 0))
    ps.eager_cascade(0, "photo", _fake_candidates(3))
    assert calls == [], "без подготовки слотов — ничего"
    e = slot_prefetch.Eager("t-eager2")
    monkeypatch.setattr(ps, "EAGER_CASCADE", [e])
    try:
        ps.eager_cascade(0, "photo", _fake_candidates(3))
        ps.eager_cascade(1, "video", [{"id": "v1"}])
        ps.eager_cascade(ps.SHOT_JUDGE_PAID_SLOTS, "photo", _fake_candidates(3))
        assert _wait(lambda: len(calls) == 2)
        time.sleep(0.1)
    finally:
        e.close()
    assert calls[0] == (3, ps.candidate_probe_url, ps.photo_probe_download, 0, True)
    assert calls[1] == (1, ps.video_middle_url, ps.video_middle_probe, 1, True)
    assert len(calls) == 2, "вне платной зоны каскада нет"


def test_eager_skips_work_of_slots_the_selection_has_passed():
    """Каскад прошедшего слота отбор уже посчитал сам — его работа из очереди
    не делается (иначе она шла бы первой, впереди текущего слота)."""
    now, seen = {"slot": 5}, []
    e = slot_prefetch.Eager("t-eager-horizon", horizon=lambda: now["slot"])
    try:
        e.submit(3, lambda: seen.append(3))
        e.submit(5, lambda: seen.append(5))
        e.submit(6, lambda: seen.append(6))
        assert _wait(lambda: seen == [5, 6])
        assert e.stats["skipped"] == 1
    finally:
        e.close()
