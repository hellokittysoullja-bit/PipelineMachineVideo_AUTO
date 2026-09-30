# -*- coding: utf-8 -*-
"""Кадр, выполнивший ВСЕ обязательные утверждения, отказом по миру и фону не
выбрасывается — только штрафуется. Живой случай videos/99_mify (30.09):
бородатый мужчина в рогатом шлеме получил «да» на все пункты фразы
«А викинги носили шлемы с рогами», и одно шумное «не из мира» его выкинуло;
девушку в костюме дьявола тот же вопрос на повторе то отклонял, то
пропускал."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import shot_judge as sj  # noqa: E402

SPEC = {"focus": "a viking in a horned helmet",
        "claims": [{"id": "core", "text": "a man wears a metal helmet with horns", "tier": "must"},
                   {"id": "c1", "text": "the man has a beard", "tier": "must"},
                   {"id": "c2", "text": "the scene looks old", "tier": "should"}]}


def _ans(core="yes", c1="yes", c2="yes", main=True, bg=False, medium="photo"):
    return {"claims": {"core": core, "c1": c1, "c2": c2}, "medium": medium,
            "main_in_world": main, "background_foreign": bg, "why": "x"}


def test_all_musts_yes_foreign_world_is_penalty_not_veto():
    v = sj.claims_vector(SPEC, _ans(main=False))
    assert v is not None, "все пункты «да» — одно «не из мира» не выбрасывает кадр"
    assert v[0] == 0.0, "штраф — первым элементом"
    assert sj.claims_vector(SPEC, _ans(main=False, c1="no")) is None, \
        "не все обязательные «да» — прежний отказ"
    assert sj.claims_vector(SPEC, _ans(main=False, c1="unsure")) is None


def test_all_musts_yes_foreign_background_is_penalty_not_veto():
    v = sj.claims_vector(SPEC, _ans(bg=True))
    assert v is not None and v[3] == 0.0, "чистота фона — 0, кадр остаётся"
    assert sj.claims_vector(SPEC, _ans(bg=True, core="no")) is None


def test_cg_veto_untouched():
    assert sj.claims_vector(SPEC, _ans(medium="cg")) is None
    assert sj.claims_vector(SPEC, _ans(medium="cg", main=False)) is None


def test_own_world_frame_beats_doubted_frame():
    doubted = sj.claims_vector(SPEC, _ans(main=False))
    partial_own = sj.claims_vector(SPEC, _ans(c1="no", c2="no"))
    assert partial_own > doubted, "кадр своего мира обгоняет штрафованный"


def test_world_doubted_flag():
    assert sj.world_doubted(SPEC, _ans(main=False))
    assert sj.world_doubted(SPEC, _ans(bg=True))
    assert not sj.world_doubted(SPEC, _ans())
    assert not sj.world_doubted(SPEC, _ans(main=False, c1="no"))
    assert not sj.world_doubted(SPEC, _ans(main=False), world_veto=False), \
        "под предохранителем мира это обычный штраф"
    assert not sj.world_doubted(SPEC, _ans(bg=True), cg_veto=False)


def test_verify_finalists_keeps_doubted_frame_and_checks_next_portion(monkeypatch, tmp_path):
    """Первая порция: все пункты «да», мир под сомнением — кадр не «veto», а
    следующая порция всё равно проверяется (кадр своего мира может найтись)."""
    sys.argv = ["pipeline_smart.py", str(tmp_path)]
    import pipeline_smart as ps
    monkeypatch.setattr(ps, "TEMP_FOLDER", str(tmp_path))
    monkeypatch.setitem(ps._SHOT_JUDGE_STATE, "world_votes", None)
    monkeypatch.setattr(ps, "episode_world_card", lambda: None)
    monkeypatch.setattr(ps, "SHOT_JUDGE_LOG", [])
    first = {"p": {"id": 1}, "path": "a.jpg", "judge": 3, "cascade_rank": 0}
    second = {"p": {"id": 2}, "path": "b.jpg", "judge": 2, "cascade_rank": 1}
    answers = {"a.jpg": _ans(main=False), "b.jpg": _ans(c1="no")}
    monkeypatch.setattr(sj, "verify_claims", lambda *a, **k: (dict(answers[k["path"]]), {}))
    portions = []

    def of(judged, more=False):
        portions.append(more)
        return [second] if more else [first]
    monkeypatch.setattr(ps, "verify_finalists_of", of)
    ps._verify_finalists(4, "photo", "А викинги носили шлемы с рогами", None,
                         [first, second], None, "m", None, spec=SPEC)
    assert isinstance(first["verify"], tuple) and first["verify"][0] == 0.0
    assert first["verify_world_doubt"] is True and not first["world_clear"]
    assert not first["verify_perfect"]
    assert portions == [False, True], "следующая порция проверена"
    assert ps.verify_key(second) > ps.verify_key(first), "кадр своего мира выше"


def test_doubted_winner_triggers_second_round():
    """Победитель со штрафом за мир — повод для второго круга («weak»): кадр
    своего мира встанет вместо него, запасной останется, если не найдётся."""
    import types
    import pipeline_smart as ps
    att = types.SimpleNamespace(verdicts=[], notes={"focus_met": True, "world_doubt": True})
    assert ps.research_trigger(att) == "weak"
    att.notes["world_doubt"] = False
    assert ps.research_trigger(att) is None
    src = open(ps.__file__, encoding="utf-8").read()
    assert src.count('record_note("world_doubt"') == 2, "фото и видео пишут одну отметку"
