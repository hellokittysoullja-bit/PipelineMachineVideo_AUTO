#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Фото или видео — по оценке судьи, а не по хэшу текста и порядку попыток.

Живой случай (эпизод 94, слот 7): фото встало только потому, что все видео
провалили проверки; наоборот, видео ставилось первым по хэшу фразы, и фото
с лучшей оценкой не добывалось вовсе."""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = ["pipeline_smart.py", REPO]
import pipeline_smart as ps  # noqa: E402
import pytest  # noqa: E402

from _video_world import infra, video  # noqa: E402,F401
from test_repick_exclusion import _judge, _select_video  # noqa: E402

pick = ps.pick_kind_by_judge


def test_higher_score_wins_across_kinds():
    assert pick("video", 2, 3, prefer_video=True) == "photo"
    assert pick("photo", 1, 3, prefer_video=False) == "video"
    assert pick("photo", 3, 2, prefer_video=True) == "photo"


def test_tie_is_decided_by_the_kind_rule_not_by_attempt_order():
    assert pick("photo", 2, 2, prefer_video=True) == "video"
    assert pick("video", 2, 2, prefer_video=False) == "photo"


def test_unjudged_other_kind_replaces_only_a_rejected_first():
    """Неизвестное лучше известного брака, известное годное — нет."""
    assert pick("video", 1, None, prefer_video=True) == "photo"
    assert pick("video", 2, None, prefer_video=True) == "video"
    assert pick("photo", None, 3, prefer_video=False) == "photo"


def test_video_winner_carries_its_score_for_the_slot(infra, monkeypatch):
    infra["videos"] = [video(1), video(2)]
    infra["relevant"] = {1, 2}
    _judge(monkeypatch, {1: 1, 2: 2})
    res, att = _select_video()
    assert res is not None
    assert att.notes.get("judge_score") == 2


def test_note_outside_an_attempt_is_an_error():
    with pytest.raises(ps.selection_attempt.AttemptStateError):
        ps.selection_attempt.record_note("judge_score", 3)


def test_main_compares_kinds_and_does_not_refetch_a_losing_photo():
    """Сторож проводки: решение по оценке стоит в main() до лестницы
    фолбэков, а спасение фотографией не добывает фото повторно."""
    src = open(os.path.join(REPO, "scripts", "pipeline_smart.py"), encoding="utf-8").read()
    body = src[src.index("\ndef main("):]
    assert "pick_kind_by_judge(first_kind" in body
    assert body.index("pick_kind_by_judge(first_kind") < body.index('"VIDEO_PHOTO_RESCUE"')
    assert 'a.kind == "photo" and a.media' in body
    assert "other_tried = any(a.kind == other_kind for a in slot_attempts)" in body
    assert "and not other_tried" in body


def test_verification_quality_decides_between_kinds():
    """Вектор проверки сильнее оценки сетки: видео, выполнившее главное,
    бьёт фото без главного, даже если сетка поставила фото выше."""
    photo_no_focus = ((1.0, 0.0, 1.0), 3, False)
    video_focus = ((1.0, 1.0, 0.0), 2, False)
    assert pick("photo", photo_no_focus, video_focus, prefer_video=False) == "video"
    assert pick("video", ((-9,), 3, False), None, prefer_video=True) == "photo", "отказ — берём неизвестное"
    assert pick("photo", ((1.0, 1.0, 0.0), 1, False), None, prefer_video=False) == "photo", \
        "главное найдено — остаётся"
    assert pick("photo", ((1.0, 0.0, 1.0), 3, False), None, prefer_video=False) == "video", \
        "главное не найдено — берём неизвестное"
    assert pick("photo", ((0.0, 1.0, 1.0), 3, False), None, prefer_video=False) == "video", \
        "чужой мир (штраф предохранителя) — не одобрено"


def test_perfect_first_kind_needs_no_second_search():
    assert ps.quality_perfect(ps._as_quality(((1.0, 1.0, 1.0), 1, True)))
    assert not ps.quality_perfect(ps._as_quality(((1.0, 1.0, 0.0), 3, False)))
    assert ps.quality_perfect(ps._as_quality(3)) and not ps.quality_perfect(ps._as_quality(2))


def test_winner_quality_is_none_without_a_judge():
    assert ps.winner_quality({"judge": None, "verify": None}) is None
    assert ps.winner_quality({"judge": 2, "verify": (1.0, 0.5), "verify_perfect": True}) == ((1.0, 0.5), 2, True)


def test_spec_motion_decides_the_first_kind_only_when_it_is_must():
    import stock_query_planner as sqp
    must = {"claims": [{"id": "core", "tier": "must"}, {"id": "c1", "tier": "must", "motion": True}]}
    should = {"claims": [{"id": "core", "tier": "must"}, {"id": "c1", "tier": "should", "motion": True}]}
    assert sqp.has_motion(must, must=True) and not sqp.has_motion(should, must=True)
    src = open(os.path.join(REPO, "scripts", "pipeline_smart.py"), encoding="utf-8").read()
    body = src[src.index("\ndef main("):]
    i = body.index('spec = b.get("shot_spec")')
    block = body[i:i + 1600]
    spec_branch = block[:block.index("            else:\n                h_text")]
    assert "has_motion(spec, must=True)" in spec_branch
    assert "has_action_word" not in spec_branch and "md5" not in spec_branch, \
        "со спецификацией ритм без словаря и хэша"
    assert 'shot_spec=b.get("shot_spec")' in body


def test_cached_winner_keeps_its_verified_quality(tmp_path, monkeypatch):
    """Повторный рендер берёт кадр из кэша без проверки; оценка прошлой
    проверки восстанавливается из sidecar, иначе выбор вида сравнивал бы
    свежую оценку одного вида с пустотой у другого."""
    import selection_attempt
    cf = str(tmp_path / "0001_x.mp4")
    open(cf, "wb").write(b"x")
    q = ((1.0, 1.0, 1.0, 1.0), 3, True)
    ps.write_media_sidecar(cf, pexels_id="pexels:1", kind="video", quality=q)
    notes = {}
    monkeypatch.setattr(selection_attempt, "record_note", lambda k, v: notes.__setitem__(k, v))
    ps._restore_cached_quality(cf)
    assert notes["quality"] == q
    notes.clear()
    ps.write_media_sidecar(cf, pexels_id="pexels:1", kind="video")
    ps._restore_cached_quality(cf)
    assert notes == {}, "без записанной оценки — как раньше"
