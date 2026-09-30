#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Проверка мира победителя: узкий вопрос по одному кадру. Сетка одобряла
кадры чужой эпохи стабильно (марокканское шоу, видео со стрелами в табличках
«HOMEWORK»); вопрос по одному кадру их видит (замер в shot_judge.py)."""
import os
import sys

from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.argv = ["pipeline_smart.py", REPO]
import pipeline_smart as ps  # noqa: E402
import shot_judge  # noqa: E402


class GW:
    def __init__(self, answer):
        self.answer, self.calls, self.texts = answer, 0, []

    def chat(self, model, content, max_tokens, est):
        self.calls += 1
        self.texts.append(content[0]["text"])
        return self.answer, {}, 200


def _img(tmp_path):
    p = tmp_path / "f.jpg"
    Image.new("RGB", (32, 32), (120, 90, 60)).save(p)
    return str(p)


def test_violation_is_read_and_cached(tmp_path):
    gw = GW('{"world_ok": false, "subject_ok": true, "why": "printed signs HOMEWORK"}')
    kw = dict(phrase="Стрела", brief="an arrow", setting="historical, 1300 AD-1500 AD",
              path=_img(tmp_path), kind="video", cache_dir=str(tmp_path / "c"))
    ok, why, info = shot_judge.world_check(gw, "m", **kw)
    assert ok is False and "HOMEWORK" in why and info["call"]
    assert "three frames" in gw.texts[0], "видео спрашивается как лента кадров одного ролика"
    ok2, _w, info2 = shot_judge.world_check(gw, "m", **kw)
    assert ok2 is False and info2.get("cache_hit") and gw.calls == 1


def test_no_world_no_question(tmp_path):
    """Без паспорта мира «чужая эпоха» не определена: не спрашивать, не
    бракуя кадр чужой нише по средневековым меркам."""
    gw = GW('{"world_ok": false}')
    ok, _w, info = shot_judge.world_check(gw, "m", phrase="x", brief="y", setting=None,
                                          path=_img(tmp_path))
    assert ok is None and info == {} and gw.calls == 0


def test_unparsed_answer_is_not_a_rejection(tmp_path):
    gw = GW("не знаю")
    ok, _w, _i = shot_judge.world_check(gw, "m", phrase="x", brief="y", setting="modern",
                                        path=_img(tmp_path), cache_dir=str(tmp_path / "c"))
    assert ok is None and not os.listdir(tmp_path / "c") if os.path.isdir(tmp_path / "c") else ok is None


