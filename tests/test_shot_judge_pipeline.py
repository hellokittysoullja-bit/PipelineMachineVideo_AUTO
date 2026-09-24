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
    without_key = ps.candidate_gate_signature(0)
    monkeypatch.setenv("SHOT_JUDGE", "0")
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", "k")
    flag_off = ps.candidate_gate_signature(0)
    assert without_key == flag_off and ps.shot_judge_signature(0) == ""
    monkeypatch.setenv("SHOT_JUDGE", "1")
    active = ps.candidate_gate_signature(0)
    assert active != without_key and "judge" in ps.shot_judge_signature(0)
    monkeypatch.setenv("SHOT_JUDGE_MODEL", "other/model")
    assert ps.candidate_gate_signature(0) != active, "смена модели меняет победителя"


def test_paid_judge_only_in_the_hook_slots(monkeypatch):
    """Решение владельца 24.09: платная проверка — первые SHOT_JUDGE_PAID_SLOTS
    слотов. Дальше судья не зовётся, и его подпись в ключ кэша слота не
    входит: смена модели судьи не перекачивает слоты, которых он не видит."""
    monkeypatch.setenv("SHOT_JUDGE", "1")
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", "k")
    last_paid, first_free = ps.SHOT_JUDGE_PAID_SLOTS - 1, ps.SHOT_JUDGE_PAID_SLOTS
    assert ps.shot_judge_active(last_paid) and not ps.shot_judge_active(first_free)
    assert ps.shot_judge_signature(first_free) == ""
    free = ps.candidate_gate_signature(first_free)
    monkeypatch.setenv("SHOT_JUDGE_MODEL", "other/model")
    assert ps.candidate_gate_signature(first_free) == free
    monkeypatch.setenv("SHOT_JUDGE", "0")
    assert ps.candidate_gate_signature(last_paid) == free, "вне судьи подпись общая"
    c = {"p": {"id": 1}, "path": "x", "is_dup_free": 1}
    monkeypatch.setenv("SHOT_JUDGE", "1")
    monkeypatch.setattr(ps, "_shot_judge_gateway",
                        lambda: (_ for _ in ()).throw(AssertionError("платный вызов вне хука")))
    assert ps.judge_candidates(first_free, "photo", "фраза", "brief", [c]) is False


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
    assert "claims_checked(winner) or not smart_relevance_veto(cf, query)" in src
    assert ps.claims_checked({"p": {}, "verify": (1.0, 0.0, 1.0), "verify_focus": False}), \
        "проверенный по утверждениям кадр вторую проверку эмбеддингом не проходит"


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


def _fake_verify(answers_by_path):
    def verify_claims(gw, model, *, path, **_k):
        return answers_by_path[path], {"call": True}
    return verify_claims


def _ans(claims, medium="photo", world=None, why=""):
    out = {"claims": claims, "medium": medium, "why": why}
    if world is not None:
        out["main_in_world"], out["background_foreign"] = world
    return out


def _verify_setup(tmp_path, monkeypatch, n=3):
    paths = []
    for k in range(n):
        p = str(tmp_path / f"v{k}.jpg")
        Image.new("RGB", (32, 32), (k * 60, 0, 0)).save(p)
        paths.append(p)
    info = [{"path": p, "p": {"id": f"c{k}"}, "is_dup_free": 1, "is_relevant": 1, "size_ok": 1,
             "sharp_ok": 1, "aesthetic_val": 0.0, "luma_score": 0.0, "min_d": 99}
            for k, p in enumerate(paths)]
    monkeypatch.setattr(ps, "_shot_judge_gateway", lambda: object())
    monkeypatch.setattr(ps, "episode_world_card", lambda: None)
    import shot_judge
    monkeypatch.setattr(shot_judge, "judge",
                        lambda *a, **k: {c["p"]["id"]: 3 for c in info})
    return info, paths, shot_judge


ARROW_SPEC = {"focus": "an arrow glancing off a breastplate", "claims": [
    {"id": "c1", "text": "an arrow is visible", "tier": "must"},
    {"id": "c2", "text": "the arrow glances off armour", "tier": "must", "motion": True},
    {"id": "c3", "text": "a steel breastplate", "tier": "should"}]}


def test_frame_with_the_focus_beats_the_frame_with_only_the_surface(tmp_path, monkeypatch):
    """Фраза «Стрела скользит по нагруднику»: нагрудник без стрелы — это
    невыполненный фокус, и он проигрывает картине, где стрела есть."""
    info, paths, sj = _verify_setup(tmp_path, monkeypatch, n=2)
    monkeypatch.setattr(sj, "verify_claims", _fake_verify({
        paths[0]: _ans({"c1": "no", "c3": "yes"}),
        paths[1]: _ans({"c1": "yes", "c3": "no"}, medium="artwork")}))
    assert ps.judge_candidates(0, "photo", "x", "y", info, ARROW_SPEC)
    winner = ps._score_and_pick(info)[0]
    assert winner["p"]["id"] == "c1" and ps.judge_approved(winner)
    assert not ps.judge_approved(info[0]), "нагрудник без стрелы не «одобрен»"


def test_motion_claim_is_not_asked_of_a_photo_and_counts_as_unmet():
    import shot_judge as sj
    assert [c["id"] for c in sj.asked_claims(ARROW_SPEC, "photo")] == ["c1", "c3"]
    assert [c["id"] for c in sj.asked_claims(ARROW_SPEC, "video")] == ["c1", "c2", "c3"]
    assert [c["id"] for c in sj.asked_claims(ARROW_SPEC, "video", frames=1)] == ["c1", "c3"], \
        "одно превью движения не показывает"
    photo = sj.claims_vector(ARROW_SPEC, _ans({"c1": "yes", "c3": "yes"}))
    video = sj.claims_vector(ARROW_SPEC, _ans({"c1": "yes", "c2": "yes", "c3": "no"}))
    assert video > photo, "видео, где стрела видна в движении, выше фото той же стрелы"
    assert ps.pick_kind_by_judge("photo", (photo, 3, False), (video, 3, False), False) == "video"
    no_arrow_video = sj.claims_vector(ARROW_SPEC, _ans({"c1": "no", "c2": "no", "c3": "yes"}))
    assert ps.pick_kind_by_judge("video", (no_arrow_video, 3, False), (photo, 3, False), True) == "photo", \
        "фото со стрелой выше видео без неё"


def test_single_preview_video_is_not_told_it_has_three_frames():
    import shot_judge as sj
    q1 = sj.claims_question("x", ARROW_SPEC, kind="video", frames=1)
    assert "frames" not in q1 and "c2:" not in q1
    assert "2 frames" in sj.claims_question("x", ARROW_SPEC, kind="video", frames=2)


def test_spectators_on_background_are_a_penalty_not_a_rejection(tmp_path, monkeypatch):
    """Упавший рыцарь на турнире со зрителями на фоне (эп.94, слот 4) — точный
    кадр; бинарная проверка мира заменяла его рыцарем в лесу."""
    info, paths, sj = _verify_setup(tmp_path, monkeypatch, n=2)
    monkeypatch.setattr(ps, "episode_world_card", lambda: {"register": "historical"})
    import world_card
    monkeypatch.setattr(world_card, "judge_setting", lambda card: "historical, 1400 AD")
    spec = {"focus": "a knight fallen in mud", "claims": [
        {"id": "c1", "text": "a knight in armour", "tier": "must"},
        {"id": "c2", "text": "the knight lies on the ground", "tier": "must"}]}
    monkeypatch.setattr(sj, "verify_claims", _fake_verify({
        paths[0]: _ans({"c1": "yes", "c2": "no"}, world=(True, False)),
        paths[1]: _ans({"c1": "yes", "c2": "yes"}, world=(True, True))}))
    assert ps.judge_candidates(0, "photo", "x", "y", info, spec)
    winner = ps._score_and_pick(info)[0]
    assert winner["p"]["id"] == "c1" and ps.judge_approved(winner)


def test_main_subject_out_of_world_is_rejected(tmp_path, monkeypatch):
    info, paths, sj = _verify_setup(tmp_path, monkeypatch, n=2)
    monkeypatch.setattr(ps, "episode_world_card", lambda: {"register": "historical"})
    import world_card
    monkeypatch.setattr(world_card, "judge_setting", lambda card: "historical, 1400 AD")
    monkeypatch.setattr(sj, "verify_claims", _fake_verify({
        paths[0]: _ans({"c1": "yes"}, world=(False, False), why="modern tactical knife"),
        paths[1]: _ans({"c1": "unsure"}, medium="object", world=(True, False))}))
    assert ps.judge_candidates(0, "photo", "x", "y", info)
    assert info[0]["verify"] == "veto" and not ps.judge_approved(info[0])
    assert ps._score_and_pick(info)[0]["p"]["id"] == "c1"


def test_unsure_is_half_and_asked_once(tmp_path, monkeypatch):
    info, paths, sj = _verify_setup(tmp_path, monkeypatch, n=2)
    asked = []

    def verify_claims(gw, model, *, path, **_k):
        asked.append(path)
        return {paths[0]: _ans({"c1": "unsure"}), paths[1]: _ans({"c1": "yes"})}[path], {"call": True}
    monkeypatch.setattr(sj, "verify_claims", verify_claims)
    assert ps.judge_candidates(0, "photo", "x", "y", info)
    assert sorted(asked) == sorted(paths), "один вопрос на кадр — второго голоса нет"
    assert info[0]["verify"][1] == 0.5 and not info[0]["verify_focus"]
    assert ps._score_and_pick(info)[0]["p"]["id"] == "c1"


def test_all_finalists_vetoed_checks_the_next_batch_not_an_unchecked_one(tmp_path, monkeypatch):
    n = ps.VERIFY_FINALISTS + 3
    info, paths, sj = _verify_setup(tmp_path, monkeypatch, n=n)
    monkeypatch.setattr(ps, "episode_world_card", lambda: {"register": "historical",
                                                           "era": {"from": 1300, "to": 1500}})
    monkeypatch.setitem(ps._SHOT_JUDGE_STATE, "world_votes", None)
    foreign = _ans({"c1": "yes"}, world=(False, False))
    good = _ans({"c1": "yes"}, world=(True, False))
    answers = {p: foreign for p in paths[:ps.VERIFY_FINALISTS]}
    answers.update({p: good for p in paths[ps.VERIFY_FINALISTS:]})
    monkeypatch.setattr(sj, "verify_claims", _fake_verify(answers))
    assert ps.judge_candidates(0, "photo", "x", "y", info)
    win = ps._score_and_pick(info)[0]
    assert win["p"]["id"] == f"c{ps.VERIFY_FINALISTS}" and isinstance(win["verify"], tuple), \
        "победил проверенный из следующей порции"


def test_world_breaker_needs_several_slots_not_one(monkeypatch, capsys):
    """Один слот с десятком современных ножей («Вот кинжал») — ровно тот
    случай, ради которого отказ по миру заведён; он не имеет права
    выключить отказ для всего ролика."""
    monkeypatch.setitem(ps._SHOT_JUDGE_STATE, "world_votes", None)
    monkeypatch.setattr(ps, "episode_world_card", lambda: None)
    monkeypatch.setattr(ps, "SHOT_JUDGE_LOG", [])
    ps._record_world_vote(0, 10, 10)
    assert ps.world_veto_active()
    for k in range(1, ps.WORLD_BREAKER_MIN_SLOTS - 1):
        ps._record_world_vote(k, 3, 2)
    assert ps.world_veto_active(), "голосов меньше порога — отказ в силе"
    ps._record_world_vote(ps.WORLD_BREAKER_MIN_SLOTS - 1, 3, 2)
    assert not ps.world_veto_active() and "штрафуется" in capsys.readouterr().out
    import shot_judge as sj
    v = sj.claims_vector({"claims": [{"id": "c1", "tier": "must"}]},
                         _ans({"c1": "yes"}, world=(False, False)), world_veto=False)
    assert v == (0.0, 1.0, 1.0), "чужой мир — штраф первым элементом, а не отказ"


def test_world_votes_survive_a_new_render_and_die_with_the_passport(monkeypatch):
    """Голоса лежат в media_plan под отпечатком паспорта: слоты из кэша не
    теряют голос, а смена паспорта обнуляет чужую статистику."""
    card = {"register": "historical", "era": {"from": 1300, "to": 1500}}
    monkeypatch.setattr(ps, "episode_world_card", lambda: card)
    monkeypatch.setitem(ps._SHOT_JUDGE_STATE, "world_votes", None)
    for k in range(ps.WORLD_BREAKER_MIN_SLOTS):
        ps._record_world_vote(k, 2, 2)
    assert not ps.world_veto_active()
    ps._SHOT_JUDGE_STATE["world_votes"] = None          # новый рендер
    assert not ps.world_veto_active()
    card2 = dict(card, era={"from": 1400, "to": 1500})
    monkeypatch.setattr(ps, "episode_world_card", lambda: card2)
    ps._SHOT_JUDGE_STATE["world_votes"] = None
    assert ps.world_veto_active()


def test_slot_without_found_focus_does_not_vote(monkeypatch):
    monkeypatch.setitem(ps._SHOT_JUDGE_STATE, "world_votes", None)
    monkeypatch.setattr(ps, "episode_world_card", lambda: None)
    for k in range(ps.WORLD_BREAKER_MIN_SLOTS + 2):
        ps._record_world_vote(k, 0, 0)
    assert ps._world_votes() == {} and ps.world_veto_active()


def test_world_decision_is_fixed_for_the_whole_slot(tmp_path, monkeypatch):
    """Решение берётся в начале слота: кадры одного слота не делятся на
    «отказ» и «штраф» из-за того, что порог перешли посреди проверки."""
    info, paths, sj = _verify_setup(tmp_path, monkeypatch, n=4)
    monkeypatch.setattr(ps, "episode_world_card", lambda: {"register": "historical",
                                                           "era": {"from": 1300, "to": 1500}})
    monkeypatch.setitem(ps._SHOT_JUDGE_STATE, "world_votes",
                        {k: True for k in range(1, ps.WORLD_BREAKER_MIN_SLOTS)})
    monkeypatch.setattr(sj, "verify_claims", _fake_verify(
        {p: _ans({"c1": "yes"}, world=(False, False)) for p in paths}))
    assert ps.judge_candidates(0, "photo", "x", "y", info)
    assert {c["verify"] for c in info} == {"veto"}, "все кадры слота судятся одним решением"
    assert not ps.world_veto_active(), "голос слота учтён уже после проверки"


def test_nothing_met_everywhere_checks_the_next_batch(tmp_path, monkeypatch):
    n = ps.VERIFY_FINALISTS + 2
    info, paths, sj = _verify_setup(tmp_path, monkeypatch, n=n)
    monkeypatch.setitem(ps._SHOT_JUDGE_STATE, "world_votes", None)
    answers = {p: _ans({"c1": "no"}) for p in paths[:ps.VERIFY_FINALISTS]}
    answers.update({p: _ans({"c1": "yes"}) for p in paths[ps.VERIFY_FINALISTS:]})
    monkeypatch.setattr(sj, "verify_claims", _fake_verify(answers))
    assert ps.judge_candidates(0, "photo", "x", "y", info)
    win = ps._score_and_pick(info)[0]
    assert win["p"]["id"] == f"c{ps.VERIFY_FINALISTS}" and isinstance(win["verify"], tuple) \
        and not win.get("verify_nothing"), "следующая порция проверена, а не взят непроверенный"


def test_blocklist_marks_only_when_the_check_can_clear(monkeypatch):
    """Помеченный словарём кандидат может очистить только проверка мира.
    Нет мира в паспорте или судья выключен — словарь выбрасывает, как раньше."""
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", "k")
    monkeypatch.setenv("SHOT_JUDGE", "1")
    monkeypatch.setattr(ps, "CONTENT_ALT_BLOCKLIST", ("fencing",))
    items = [{"id": 1, "alt": "sport fencing match"}, {"id": 2, "alt": "medieval knight"}]
    monkeypatch.setattr(ps, "_shot_judge_gateway", lambda: object())
    monkeypatch.setattr(ps, "episode_world_card", lambda: {"register": "historical",
                                                           "era": {"from": 1300, "to": 1500}})
    assert [p.get("_blocklisted") for p in ps.filter_pool_by_text(items, 0)] == [True, None]
    monkeypatch.setattr(ps, "episode_world_card", lambda: None)
    assert [p["id"] for p in ps.filter_pool_by_text(items, 0)] == [2], "мира нет — выбрасывает"
    monkeypatch.setattr(ps, "episode_world_card", lambda: {"register": "historical",
                                                           "era": {"from": 1300, "to": 1500}})
    monkeypatch.setattr(ps, "_shot_judge_gateway", lambda: None)
    assert [p["id"] for p in ps.filter_pool_by_text(items, 0)] == [2], "судья выключен — выбрасывает"


def test_checked_reject_ranks_below_unchecked_candidate():
    bad = dict(_c("bad"), verify=(1.0, 0.0, 1.0), verify_nothing=True)
    unchecked = dict(_c("un", judge=3), verify=None)
    assert ps.verify_key(bad) < ps.verify_key(unchecked)
    assert ps._score_and_pick([bad, unchecked])[0]["p"]["id"] == "un"


def test_blocklisted_candidate_needs_a_clean_world_check(tmp_path, monkeypatch):
    info, paths, sj = _verify_setup(tmp_path, monkeypatch, n=2)
    info[1]["p"]["_blocklisted"] = True
    monkeypatch.setattr(sj, "verify_claims", _fake_verify({paths[0]: _ans({"c1": "unsure"}),
                                                          paths[1]: _ans({"c1": "yes"})}))
    assert ps.judge_candidates(0, "photo", "x", "y", info)
    assert ps._score_and_pick(info)[0]["p"]["id"] == "c0", "без паспорта мир не проверен — не на экран"
    monkeypatch.setattr(ps, "episode_world_card", lambda: {"register": "historical",
                                                           "era": {"from": 1300, "to": 1500}})
    monkeypatch.setattr(sj, "verify_claims", _fake_verify({
        paths[0]: _ans({"c1": "unsure"}, world=(True, False)),
        paths[1]: _ans({"c1": "yes"}, world=(True, False))}))
    assert ps.judge_candidates(0, "photo", "x", "y", info)
    assert ps._score_and_pick(info)[0]["p"]["id"] == "c1", "мир проверен и чист — реконструкция годна"


def test_failed_verification_keeps_grid_order(tmp_path, monkeypatch):
    info, paths, sj = _verify_setup(tmp_path, monkeypatch, n=2)
    monkeypatch.setattr(sj, "verify_claims", lambda *a, **k: (None, {"refused": "сбой"}))
    before = ps._score_and_pick([dict(c) for c in info])[0]["p"]["id"]
    assert ps.judge_candidates(0, "photo", "x", "y", info)
    assert all(ps.verify_key(c) == (-1,) for c in info)
    assert ps._score_and_pick(info)[0]["p"]["id"] == before


def test_only_the_grid_finalists_are_verified(tmp_path, monkeypatch):
    info, paths, sj = _verify_setup(tmp_path, monkeypatch, n=ps.VERIFY_FINALISTS + 2)
    asked = []

    def verify_claims(gw, model, *, path, **_k):
        asked.append(path)
        return None, {}
    monkeypatch.setattr(sj, "verify_claims", verify_claims)
    ps.judge_candidates(0, "photo", "x", "y", info)
    assert len(asked) == ps.VERIFY_FINALISTS, "сетка всем поставила поровну — лучшие и первые совпали"


def test_finalists_are_grid_best_cascade_first_and_blocklisted():
    judged = [{"judge": g, "p": {}} for g in (1, 1, 1, 1, 1, 1, 3, 3, 1)]
    judged[8]["p"]["_blocklisted"] = True
    got = ps.verify_finalists_of(judged)
    assert judged[6] in got and judged[7] in got, "лучшие по сетке"
    assert all(judged[k] in got for k in range(ps.VERIFY_FINALISTS)), "первые по каскаду"
    assert judged[8] in got, "помеченный словарём проверяется сверх"
    assert len(got) == len({id(c) for c in got})


def test_without_a_plan_the_brief_is_the_single_must_claim():
    import shot_judge as sj
    spec = sj.spec_from_brief("фраза", "a dagger on a table")
    assert spec["claims"] == [{"id": "c1", "text": "a dagger on a table", "tier": "must"}]
    q = sj.claims_question("фраза", spec)
    assert "c1: a dagger on a table" in q and "close" not in q


def test_parse_claims_answer_is_strict():
    import shot_judge as sj
    ok = '{"claims": {"c1": "Yes", "c2": "no"}, "medium": "photo", "why": "x"}'
    assert sj.parse_claims_answer(ok, ["c1", "c2"], False)["claims"] == {"c1": "yes", "c2": "no"}
    assert sj.parse_claims_answer(ok, ["c1", "c2", "c3"], False) is None, "не ответил на пункт"
    assert sj.parse_claims_answer(ok, ["c1"], True) is None, "мир спрошен — нужен ответ"
    assert sj.parse_claims_answer('{"claims": {"c1": "maybe"}, "medium": "photo"}', ["c1"], False) is None


def test_binary_world_question_is_gone_from_the_pipeline():
    """Бинарная проверка мира заменена проверкой по пунктам: по замеру эп.94
    она отклоняла 6 годных из 33 и пропускала 46 брачных из 71."""
    src = open(os.path.join(REPO, "scripts", "pipeline_smart.py"), encoding="utf-8").read()
    assert "judge_world_violation" not in src and "_judge_top_tie" not in src
    assert "shot_substitutes" not in src and "verify_rank" not in src


def test_budget_forecast_warns_once_early(monkeypatch, capsys):
    """~2 300 на слот, 25 платных слотов при потолке 30 тыс. — судья
    выключился бы посреди хука молча. Прогноз называет это один раз."""
    class GW:
        spend_cap, spent = 30000, 0
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
    assert ps.SHOT_JUDGE_LOG[0]["cutoff_slot"] == 13


def test_budget_forecast_counts_only_paid_slots(monkeypatch, capsys):
    """Эпизод в 250 слотов при прежних ~2 300 на слот: судятся 25, это ~58
    тыс. при потолке 300 тыс. — предупреждать не о чем."""
    class GW:
        spend_cap, spent = 300000, 0
    gw = GW()
    monkeypatch.setattr(ps, "SHOT_JUDGE_EPISODE_SLOTS", 250)
    monkeypatch.setitem(ps._SHOT_JUDGE_STATE, "slots", set())
    monkeypatch.setitem(ps._SHOT_JUDGE_STATE, "warned", False)
    for i in range(8):
        gw.spent += 2300
        ps._judge_budget_forecast(i, gw)
    assert "прогноз" not in capsys.readouterr().out


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


def test_candidate_caption_carries_source_text_and_museum_passport():
    p = {"alt": "Roundel dagger", "url": "https://www.metmuseum.org/art/collection/search/1",
         "_museum_meta": {"begin": 1400, "end": 1450, "culture": "French"}}
    cap = ps.candidate_caption(p)
    assert "roundel dagger" in cap and "dated 1400-1450" in cap and "French" in cap
    assert ps.candidate_caption({"alt": "", "url": ""}) == ""


def test_caption_reaches_the_verification_question():
    import shot_judge
    spec = shot_judge.spec_from_brief("фраза", "brief")
    q = shot_judge.claims_question("фраза", spec, caption="moroccan horsemen perform a tbourida")
    assert "moroccan horsemen" in q and "may be incomplete or wrong" in q
    assert "caption" not in shot_judge.claims_question("фраза", spec).lower()


def test_text_blocklist_off_where_the_world_is_checked_on_the_frame(monkeypatch):
    """Реконструкция в слоте с проверкой мира не выбрасывается словарём —
    её судит проверка по кадру; явный id брака выбрасывается везде; вне
    платной зоны словарь прежний."""
    pool = [{"id": 1, "alt": "knights in a battle reenactment"}, {"id": 2, "alt": "arrow"}]
    monkeypatch.setattr(ps, "CONTENT_BLOCKED_CANDIDATE_IDS", {"pexels:2"})
    monkeypatch.setattr(ps, "content_blocklist_effective", lambda: ("reenactment",))
    monkeypatch.setattr(ps, "shot_judge_active", lambda index=None: index is not None and index < 25)
    monkeypatch.setattr(ps, "_shot_judge_gateway", lambda: object())
    monkeypatch.setattr(ps, "episode_world_card", lambda: {"register": "historical",
                                                           "era": {"from": 1300, "to": 1500}})
    assert [p["id"] for p in ps.filter_pool_by_text(pool, 3)] == [1]
    assert [p["id"] for p in ps.filter_pool_by_text(pool, 30)] == [], "без проверки — словарь и id"


def test_vision_check_and_grid_never_leave_reasoning_to_the_provider_default():
    """24.09 провайдер включил рассуждение Qwen по умолчанию: проверка зрения
    (20 токенов) стала пустой, и судья выключился на весь прогон. Рассуждение
    выключается явно — там, где на нём всё замерено."""
    import shot_judge
    seen = []

    class GW:
        def chat(self, model, content, max_tokens, est, **kw):
            seen.append(kw.get("reasoning"))
            return "red blue", {}, 1

    shot_judge.vision_check(GW(), "m")
    assert seen and all(r is False for r in seen)
    assert shot_judge.GRID_REASONING is False


def test_render_asks_the_world_separately_with_the_excluded_cultures(tmp_path, monkeypatch):
    """Замер 24.09: мир отдельным вопросом + чужие культуры паспорта — лучший
    кадр слота 8 из 9 против 6-7; каждая половина по отдельности не даёт."""
    info, paths, sj = _verify_setup(tmp_path, monkeypatch, n=2)
    card = {"register": "historical", "era": {"from": 1300, "to": 1500},
            "culture": {"include": [], "exclude": ["japanese"]}}
    monkeypatch.setattr(ps, "episode_world_card", lambda: card)
    seen = []

    def verify_claims(gw, model, *, path, setting=None, world_separate=False, **_k):
        seen.append((setting, world_separate))
        return _ans({"c1": "yes"}, world=(True, False)), {"call": True}
    monkeypatch.setattr(sj, "verify_claims", verify_claims)
    spec = {"focus": "a dagger", "claims": [{"id": "c1", "text": "a dagger", "tier": "must"}]}
    assert ps.judge_candidates(0, "photo", "x", "y", info, spec)
    assert seen and all(ws for _s, ws in seen)
    assert all(s and "not: japanese" in s for s, _ws in seen)


def test_grid_failure_still_verifies_the_first_by_cascade(tmp_path, monkeypatch):
    """Живой случай judge12, слот 0: сетка не ответила (502 четыре раза) — и
    слот шёл вообще без проверки. Проверка по утверждениям — отдельные вызовы:
    она спрашивается у первых по каскаду, и брак не встаёт на экран."""
    info, paths, sj = _verify_setup(tmp_path, monkeypatch, n=3)
    monkeypatch.setattr(sj, "judge", lambda *a, **k: None)
    monkeypatch.setattr(sj, "verify_claims", _fake_verify({
        paths[0]: _ans({"c1": "no"}), paths[1]: _ans({"c1": "yes"}), paths[2]: _ans({"c1": "no"})}))
    spec = {"focus": "a dagger", "claims": [{"id": "c1", "text": "a dagger", "tier": "must"}]}
    assert ps.judge_candidates(0, "photo", "x", "y", info, spec)
    winner = ps._score_and_pick(info)[0]
    assert winner["p"]["id"] == "c1" and ps.judge_approved(winner)


def test_grid_and_verification_both_down_means_no_judge(tmp_path, monkeypatch):
    info, paths, sj = _verify_setup(tmp_path, monkeypatch, n=2)
    monkeypatch.setattr(sj, "judge", lambda *a, **k: None)
    monkeypatch.setattr(sj, "verify_claims", lambda *a, **k: (None, {"refused": "502"}))
    assert not ps.judge_candidates(0, "photo", "x", "y", info, None)
