#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Повторный выбор после отказа (скачивание, резкость, вторая проверка).

Отказник больше не возвращается в выбор. Раньше он только понижался по
одному ключу кортежа ранжирования, а оценка судьи стоит в кортеже ВЫШЕ
этих ключей: понижение переставало понижать. Замер на эпизоде 94 (прогон с
судьёй): два равных по оценке ролика, оба не прошедшие резкость, качались
по кругу, пока не кончался лимит повторов, а отказ по второй проверке
сдавался, хотя следующий кандидат лежал рядом."""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = ["pipeline_smart.py", REPO]
import pipeline_smart as ps  # noqa: E402

from _video_world import QUERY, infra, video  # noqa: E402,F401


def _judge(monkeypatch, scores):
    def fake(index, kind, phrase, brief, info):
        for c in info:
            c["judge"] = scores.get(c["p"]["id"])
        return True
    monkeypatch.setattr(ps, "judge_candidates", fake)


def _select_video(index=0):
    import dataclasses
    import selection_engine
    f = {x.name: None for x in dataclasses.fields(selection_engine.SlotRequest)}
    f.update(index=index, query=QUERY, extra_queries=(), is_opening=False, director_assist=False)
    att = ps.new_attempt(index, "video")
    with ps.selection_attempt.activate(att):
        res = ps.select_media(ps.build_slot_request(**f), "video")
    return res, att


def test_rejected_video_is_never_downloaded_again(infra, monkeypatch):
    """Два ролика с оценкой 2 не прошли резкость (живой случай: 3D-мультфильм
    с рыцарем в слоте «доспех держит человека, как капкан»). Прежний цикл
    скачивал 1, 2, 1 и сдавался по лимиту; теперь — ровно по разу."""
    infra["videos"] = [video(1), video(2), video(3)]
    infra["relevant"] = {1, 2, 3}
    infra["sharp_bad"] = {1, 2}
    _judge(monkeypatch, {1: 2, 2: 2, 3: 1})
    res, att = _select_video()
    assert infra["downloads"] == [1, 2]
    assert res is None, "технический отказ не заменяется кадром хуже по смыслу"
    rec = [m for k, m in att.verdicts if k == "smart_veto"]
    assert [r["reason"] for r in rec[0]["rejected"]] == ["sharpness", "sharpness"], \
        "вердикт называет настоящую причину, а не «вторая проверка»"


def test_technical_rejection_takes_the_next_of_the_same_meaning(infra, monkeypatch):
    infra["videos"] = [video(1), video(2), video(3)]
    infra["relevant"] = {1, 2, 3}
    infra["sharp_bad"] = {1}
    _judge(monkeypatch, {1: 3, 2: 1, 3: 3})
    res, _att = _select_video()
    assert res is not None
    assert infra["downloads"] == [1, 3], "резкий того же смысла, а не резкий хуже по смыслу"


def test_veto_rejection_moves_on_even_across_judge_scores(infra, monkeypatch):
    """Вето — отказ по смыслу: следующий по ранжированию, как и задумано у
    итеративного вето. Раньше с судьёй отказник с оценкой 1 оставался
    первым над кандидатом с оценкой 0, и слот сдавался после первого отказа."""
    infra["videos"] = [video(1), video(2)]
    infra["relevant"] = {1, 2}
    infra["veto"] = {1}
    _judge(monkeypatch, {1: 1, 2: 0})
    _select_video()
    assert infra["downloads"] == [1, 2]


def test_download_failure_moves_on(infra, monkeypatch):
    infra["videos"] = [video(1), video(2)]
    infra["relevant"] = {1, 2}
    infra["fail_dl"] = {1}
    _judge(monkeypatch, {1: 3, 2: 2})
    res, att = _select_video()
    assert res is not None and infra["downloads"] == [1, 2]


def _c(cid, judge, sharp=1, relevant=1):
    return {"p": {"id": cid}, "path": cid, "is_dup_free": 1, "size_ok": 1, "is_relevant": relevant,
            "sharp_ok": sharp, "aesthetic_val": 5.0, "luma_score": 0.0, "min_d": 10,
            "relevance": 0.1, "judge": judge}


def test_repick_never_returns_an_excluded_candidate():
    a, b = _c("a", 3), _c("b", 3)
    assert ps._repick([a, b], a, None, False, {id(a)}, same_meaning=True) is b
    assert ps._repick([a, b], b, None, False, {id(a), id(b)}, same_meaning=False) is None


def test_same_meaning_refuses_a_worse_meaning_replacement():
    a, b = _c("a", 3), _c("b", 2)
    assert ps._repick([a, b], a, None, False, {id(a)}, same_meaning=True) is None
    assert ps._repick([a, b], a, None, False, {id(a)}, same_meaning=False) is b
