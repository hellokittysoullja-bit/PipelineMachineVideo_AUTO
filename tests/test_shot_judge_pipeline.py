#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Судья кадров внутри отбора: где стоит его оценка и что он НЕ меняет."""
import os
import sys

import pytest
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.argv = ["pipeline_smart.py", REPO]
import pipeline_smart as ps  # noqa: E402
import selection_attempt as sa  # noqa: E402


def _c(cid, *, judge=None, relevant=1, size_ok=1, aesthetic=5.0, dup_free=1, path="x"):
    return {"path": path, "p": {"id": cid}, "is_dup_free": dup_free, "size_ok": size_ok,
            "is_relevant": relevant, "sharp_ok": 1, "aesthetic_val": aesthetic,
            "luma_score": 0.0, "min_d": 10, "relevance": 0.1, "judge": judge}


def test_judge_score_outranks_embedding_gates():
    """Живой случай эпизода 94: эмбеддинг пропустил современный нож и
    отверг подлинный кинжал ловушкой. Судья сильнее — его оценка решает."""
    knife = _c("knife", judge=0, relevant=1, aesthetic=9.0)
    dagger = _c("dagger", judge=3, relevant=0, aesthetic=4.0)
    assert ps._score_and_pick([knife, dagger])[0] is dagger


def test_duplicate_stays_first_key():
    assert ps._score_and_pick([_c("dup", judge=3, dup_free=0), _c("ok", judge=1)])[0]["p"]["id"] == "ok"


def test_without_judge_the_order_is_exactly_as_before():
    """Судьи не было — у всех judge=None; победитель тот же, что без ключа."""
    a = [_c("a", relevant=0, aesthetic=9.0), _c("b", relevant=1, aesthetic=1.0)]
    b = [dict(c, judge=None) for c in a]
    for c in a:
        del c["judge"]
    assert ps._score_and_pick(a)[0]["p"]["id"] == ps._score_and_pick(b)[0]["p"]["id"] == "b"


def test_inactive_judge_leaves_no_trace_in_the_selection_signature(monkeypatch):
    monkeypatch.setenv("SHOT_JUDGE", "1")
    monkeypatch.delenv("LLM_GATEWAY_API_KEY", raising=False)
    without_key = ps._selection_stack_signature()
    monkeypatch.setenv("SHOT_JUDGE", "0")
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", "k")
    flag_off = ps._selection_stack_signature()
    assert without_key == flag_off and "judge" not in without_key
    monkeypatch.setenv("SHOT_JUDGE", "1")
    active = ps._selection_stack_signature()
    assert active != without_key and "judge" in active
    monkeypatch.setenv("SHOT_JUDGE_MODEL", "other/model")
    assert ps._selection_stack_signature() != active, "смена модели меняет победителя"


def test_judge_candidates_scores_only_non_duplicates(tmp_path, monkeypatch):
    paths = []
    for k in range(3):
        p = tmp_path / f"{k}.jpg"
        Image.new("RGB", (32, 32), (k * 60, 0, 0)).save(p)
        paths.append(str(p))
    info = [_c("a", path=paths[0]), _c("b", path=paths[1]), _c("dup", path=paths[2], dup_free=0)]
    seen = {}

    def fake_judge(gw, model, *, phrase, brief, candidates, cache_dir=None, report=None, kind, setting=None):
        seen["ids"] = [cid for cid, _p in candidates]
        seen["brief"], seen["phrase"], seen["kind"] = brief, phrase, kind
        return {"a": 3, "b": 1}
    import shot_judge
    monkeypatch.setattr(shot_judge, "judge", fake_judge)
    monkeypatch.setattr(ps, "_shot_judge_gateway", lambda: object())
    assert ps.judge_candidates(0, "photo", "Вот кинжал.", "a dagger", info)
    assert seen["ids"] == ["a", "b"] and seen["brief"] == "a dagger" and seen["kind"] == "photo"
    assert [c["judge"] for c in info] == [3, 1, None]


def test_episode_world_reaches_the_judge_as_one_line(tmp_path, monkeypatch):
    """Строка мира берётся ТОЛЬКО из паспорта эпизода; паспорта нет — None,
    вопрос прежний."""
    p = tmp_path / "a.jpg"
    Image.new("RGB", (32, 32), (60, 0, 0)).save(p)
    seen = {}

    def fake_judge(gw, model, *, phrase, brief, candidates, cache_dir=None, report=None, kind,
                   setting=None):
        seen["setting"] = setting
        return {"a": 2}
    import shot_judge
    monkeypatch.setattr(shot_judge, "judge", fake_judge)
    monkeypatch.setattr(ps, "_shot_judge_gateway", lambda: object())
    card = {"schema_version": 1, "register": "historical", "era": {"from": 1300, "to": 1500},
            "culture": {"include": [], "exclude": ["asian"]}, "must_not_show": ["firearm"]}
    monkeypatch.setattr(ps, "episode_world_card", lambda video_dir=None: card)
    ps.judge_candidates(0, "photo", "Вот кинжал.", "a dagger", [_c("a", path=str(p))])
    assert seen["setting"] == "historical, 1300 AD-1500 AD", "запреты и чужие культуры не входят"
    monkeypatch.setattr(ps, "episode_world_card", lambda video_dir=None: None)
    ps.judge_candidates(0, "photo", "Вот кинжал.", "a dagger", [_c("a", path=str(p))])
    assert seen["setting"] is None


def test_video_candidates_are_judged_by_their_strip(tmp_path, monkeypatch):
    """Видео судья видит лентой кадров превью (judge_path), а не одним
    средним кадром, по которому считаются гейты: движение — это смена
    кадров, одним кадром его не показать."""
    strip, mid = tmp_path / "strip.jpg", tmp_path / "mid.jpg"
    for p in (strip, mid):
        Image.new("RGB", (32, 32), (9, 9, 9)).save(p)
    info = [dict(_c("v", path=str(mid)), judge_path=str(strip))]
    seen = {}

    def fake_judge(gw, model, *, phrase, brief, candidates, cache_dir=None, report=None, kind, setting=None):
        seen["paths"], seen["kind"] = [p for _cid, p in candidates], kind
        return {"v": 2}
    import shot_judge
    monkeypatch.setattr(shot_judge, "judge", fake_judge)
    monkeypatch.setattr(ps, "_shot_judge_gateway", lambda: object())
    assert ps.judge_candidates(0, "video", "Стрела летит.", "an arrow in flight", info)
    assert seen == {"paths": [str(strip)], "kind": "video"} and info[0]["judge"] == 2


def test_failed_judge_clears_every_score(monkeypatch):
    import shot_judge
    monkeypatch.setattr(shot_judge, "judge", lambda *a, **k: None)
    monkeypatch.setattr(ps, "_shot_judge_gateway", lambda: object())
    info = [_c("a", judge=3, path=__file__), _c("b", judge=0, path=__file__)]
    assert not ps.judge_candidates(0, "photo", "p", "b", info)
    assert all(c["judge"] is None for c in info), "без смешивания оценённых с неоценёнными"


def test_judge_rejection_is_the_strongest_known_bad_reason():
    assert ps.known_bad_reason([("smart_veto", {}), ("judge", {})]) == "shot_judge_rejected"
    assert ps.VERDICT_REPORT_LISTS["judge"] == "SHOT_JUDGE_MISSES"


def test_approved_frame_is_not_vetoed_by_the_weaker_check():
    assert ps.judge_approved(_c("a", judge=2)) and not ps.judge_approved(_c("a", judge=1))
    assert not ps.judge_approved(None) and not ps.judge_approved(_c("a", judge=None))
    src = open(os.path.join(REPO, "scripts", "pipeline_smart.py"), encoding="utf-8").read()
    assert "judge_approved(winner) or not smart_relevance_veto(cf, query)" in src


def test_tests_never_reach_the_paid_gateway():
    """conftest гасит судью: дефолт реестра 1, ключ мог быть в окружении."""
    assert os.environ.get("SHOT_JUDGE") == "0" and not os.environ.get("LLM_GATEWAY_API_KEY")
    assert not ps.shot_judge_active()


def test_transparent_download_is_flattened_for_gates_and_render(tmp_path):
    """Гейты и рендер видят тот же кадр, что и судья: прозрачное — на
    светлом фоне, а не произвольный цвет прозрачных пикселей."""
    import numpy as np
    from PIL import Image
    arr = np.zeros((40, 40, 4), dtype=np.uint8)
    arr[..., 0] = 255                      # «мусор» под прозрачностью — красный
    arr[10:30, 10:30] = (20, 20, 20, 255)  # сам предмет
    p = str(tmp_path / "cand.jpg")
    Image.fromarray(arr, "RGBA").save(p, "PNG")
    assert ps.flatten_transparency(p) is True
    im = Image.open(p)
    assert im.mode == "RGB"
    assert im.getpixel((0, 0))[0] < 245 and abs(im.getpixel((0, 0))[1] - 235) < 12
    assert max(im.getpixel((20, 20))) < 40
    opaque = str(tmp_path / "opaque.jpg")
    Image.new("RGB", (8, 8), (1, 2, 3)).save(opaque, "JPEG")
    before = open(opaque, "rb").read()
    assert ps.flatten_transparency(opaque) is False
    assert open(opaque, "rb").read() == before


def _tie_setup(tmp_path, monkeypatch, first, second):
    paths = []
    for k in range(3):
        p = tmp_path / f"t{k}.jpg"
        Image.new("RGB", (32, 32), (k * 60, 0, 0)).save(p)
        paths.append(str(p))
    info = [_c("sword", path=paths[0]), _c("dagger", path=paths[1]), _c("other", path=paths[2])]
    calls = []

    def fake_judge(gw, model, *, phrase, brief, candidates, cache_dir=None, report=None, kind, setting=None):
        calls.append([cid for cid, _p in candidates])
        return first if len(calls) == 1 else second
    import shot_judge
    monkeypatch.setattr(shot_judge, "judge", fake_judge)
    monkeypatch.setattr(ps, "_shot_judge_gateway", lambda: object())
    return info, calls


def test_top_tie_is_asked_again_in_one_grid_and_decides(tmp_path, monkeypatch):
    """Живой случай «Вот кинжал»: кинжал и меч оба с 3; ничью решал
    эмбеддинг — в пользу меча. Переспрос разделивших высшую оценку одной
    сеткой (в обратном порядке) решает её раньше эмбеддинга."""
    info, calls = _tie_setup(tmp_path, monkeypatch,
                             {"sword": 3, "dagger": 3, "other": 1}, {"dagger": 3, "sword": 2})
    info[0]["relevance"], info[0]["aesthetic_val"] = 0.3, 9.0   # эмбеддинг за меч
    assert ps.judge_candidates(0, "photo", "Вот кинжал.", "a rondel dagger", info)
    assert calls[1] == ["dagger", "sword"], "переспрашиваются только разделившие высшую, в обратном порядке"
    base, _d = ps._score_and_pick(info)
    assert base["p"]["id"] == "dagger"


def test_no_tie_or_low_top_asks_once(tmp_path, monkeypatch):
    info, calls = _tie_setup(tmp_path, monkeypatch, {"sword": 3, "dagger": 2, "other": 1}, None)
    assert ps.judge_candidates(0, "photo", "x", "y", info) and len(calls) == 1
    info, calls = _tie_setup(tmp_path, monkeypatch, {"sword": 1, "dagger": 1, "other": 0}, None)
    assert ps.judge_candidates(0, "photo", "x", "y", info) and len(calls) == 1


def test_failed_tie_question_keeps_first_scores(tmp_path, monkeypatch):
    info, calls = _tie_setup(tmp_path, monkeypatch, {"sword": 3, "dagger": 3, "other": 1}, None)
    assert ps.judge_candidates(0, "photo", "x", "y", info)
    assert [c["judge"] for c in info] == [3, 3, 1]
    assert all(ps.judge_tie_rank(c) == -1 for c in info)


def test_budget_forecast_warns_once_early(monkeypatch, capsys):
    """Замер эпизода 94: ~2 300 на слот, 250 слотов при потолке 300 тыс. —
    судья выключился бы посреди ролика молча. Прогноз называет это один раз."""
    class GW:
        spend_cap, spent = 300000, 0
    gw = GW()
    monkeypatch.setattr(ps, "SHOT_JUDGE_EPISODE_SLOTS", 250)
    monkeypatch.setitem(ps._SHOT_JUDGE_STATE, "slots", set())
    monkeypatch.setitem(ps._SHOT_JUDGE_STATE, "warned", False)
    monkeypatch.setattr(ps, "SHOT_JUDGE_LOG", [])
    for i in range(8):
        gw.spent += 2300
        ps._judge_budget_forecast(i, gw)
    out = capsys.readouterr().out
    assert out.count("прогноз") == 1 and "SHOT_JUDGE_MAX_SPEND" in out
    assert ps.SHOT_JUDGE_LOG[0]["cutoff_slot"] == 130


def test_budget_forecast_silent_when_it_fits(monkeypatch, capsys):
    class GW:
        spend_cap, spent = 300000, 0
    gw = GW()
    monkeypatch.setattr(ps, "SHOT_JUDGE_EPISODE_SLOTS", 50)
    monkeypatch.setitem(ps._SHOT_JUDGE_STATE, "slots", set())
    monkeypatch.setitem(ps._SHOT_JUDGE_STATE, "warned", False)
    for i in range(10):
        gw.spent += 2300
        ps._judge_budget_forecast(i, gw)
    assert "прогноз" not in capsys.readouterr().out
