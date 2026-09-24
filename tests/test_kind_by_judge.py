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
    """Проверка по пунктам сильнее оценки сетки: точное видео (предмет и
    действие) бьёт фото-замену, даже если сетка поставила фото выше."""
    photo_sub = ((1, 0, 1), 3)
    video_exact = ((2, 1, 1), 2)
    assert pick("photo", photo_sub, video_exact, prefer_video=False) == "video"
    assert pick("video", ((-9,), 3), None, prefer_video=True) == "photo", "отказ — берём неизвестное"
    assert pick("photo", ((1, 0, 1), 1), None, prefer_video=False) == "photo", "одобренная замена остаётся"


def test_perfect_first_kind_needs_no_second_search():
    assert ps.quality_perfect(ps._as_quality(((2, 1, 1), 1)))
    assert not ps.quality_perfect(ps._as_quality(((2, 1, 0), 3))), "чужой фон — ищем второй вид"
    assert ps.quality_perfect(ps._as_quality(3)) and not ps.quality_perfect(ps._as_quality(2))


def test_winner_quality_is_none_without_a_judge():
    assert ps.winner_quality({"judge": None, "verify": None}) is None
    assert ps.winner_quality({"judge": 2, "verify": (1, 1, 1)}) == ((1, 1, 1), 2)


def test_plan_kind_preference_overrides_the_word_rule_in_main():
    src = open(os.path.join(REPO, "scripts", "pipeline_smart.py"), encoding="utf-8").read()
    body = src[src.index("\ndef main("):]
    i = body.index('want_video = (has_action_word(b["text"])')
    assert 'b.get("kind_pref") in ("photo", "video") and not stat' in body[i:i + 800]
    assert "shot_substitutes=tuple(b.get(\"shot_rungs\") or ())" in body
