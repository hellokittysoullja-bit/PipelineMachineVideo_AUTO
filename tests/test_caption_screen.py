# -*- coding: utf-8 -*-
"""Отсев чужого мира по подписи (caption_screen): выбрасывается только то,
в чём согласны ОБА вопроса; без чужих культур в паспорте, при сбое и при
выключенном судье — ни одного вызова и ни одной потери."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import caption_screen as cs  # noqa: E402

CARD = {"register": "medieval European warfare", "era": {"from": 1300, "to": 1500},
        "culture": {"include": [], "exclude": ["ottoman", "islamic", "japanese"]},
        "must_not_show": ["firearm"]}
CARD_SCIENCE = {"register": "deep sea biology", "era": None, "culture": None}

ROWS = [("1", "pexels", "moroccan fantasia horsemen in rabat"),
        ("2", "pexels", "a knight on horseback in plate armour"),
        ("3", "met", "armor for man and horse, german 1480"),
        ("4", "pexels", "a historical reenactment featuring a warrior on horseback in konya")]


class FakeGateway:
    def __init__(self, x=None, b=None, fail=False):
        self.x, self.b, self.fail = x or {}, b or [], fail
        self.calls = []

    def chat(self, model, content, max_tokens, est, **kw):
        text = content[0]["text"]
        self.calls.append(text)
        if self.fail:
            raise RuntimeError("шлюз лежит")
        if '"mark"' in text:
            return json.dumps({"mark": self.x}), {}, 10
        return json.dumps({"drop": self.b}), {}, 12


def test_drops_only_when_both_questions_agree():
    # X: 1 и 4 чужой мир, 2 — «J» (не используется); B: 1, 2, 3.
    gw = FakeGateway(x={"1": "X", "2": "J", "4": "X"}, b=[1, 2, 3])
    drop, info = cs.screen(gw, "Конница мчится", "cavalry charge", CARD, ROWS)
    assert drop == {"1"}
    assert len(gw.calls) == 2 and info["price"] == 22
    assert [d["id"] for d in info["dropped"]] == ["1"]


def test_j_marks_never_drop():
    gw = FakeGateway(x={"1": "J", "2": "J"}, b=[1, 2])
    drop, _ = cs.screen(gw, "фраза", "focus", CARD, ROWS)
    assert drop == set()


def test_no_excluded_cultures_no_calls():
    gw = FakeGateway(x={"1": "X"}, b=[1])
    drop, _ = cs.screen(gw, "фраза", "focus", CARD_SCIENCE, ROWS)
    assert drop == set() and gw.calls == []


def test_gateway_failure_keeps_everything():
    drop, info = cs.screen(FakeGateway(fail=True), "фраза", "focus", CARD, ROWS)
    assert drop == set() and info["error"]


def test_garbage_answer_keeps_everything():
    class Bad(FakeGateway):
        def chat(self, *a, **k):
            return "не json", {}, 1
    drop, info = cs.screen(Bad(), "фраза", "focus", CARD, ROWS)
    assert drop == set() and info["error"]


def test_out_of_range_numbers_ignored():
    gw = FakeGateway(x={"1": "X", "99": "X", "0": "X"}, b=[1, 99, 0])
    drop, _ = cs.screen(gw, "фраза", "focus", CARD, ROWS)
    assert drop == {"1"}


def test_disk_cache_second_call_is_free(tmp_path):
    gw = FakeGateway(x={"1": "X"}, b=[1])
    cs.screen(gw, "фраза", "focus", CARD, ROWS, cache_dir=str(tmp_path))
    gw2 = FakeGateway(x={}, b=[])
    drop, info = cs.screen(gw2, "фраза", "focus", CARD, ROWS, cache_dir=str(tmp_path))
    assert gw2.calls == [] and drop == {"1"} and info["cached"] == 2 and info["price"] == 0


def test_prompts_carry_excluded_cultures_and_phrase():
    px, pb = cs.build_prompts("Конница мчится", "cavalry", CARD, ROWS)
    assert "ottoman, islamic, japanese" in px and "Конница мчится" in px and "Конница мчится" in pb
    assert "another culture or another era" in pb     # исторический мир
    assert "moroccan fantasia horsemen in rabat" in px


# ---------------- встройка в отбор ----------------

@pytest.fixture
def ps(monkeypatch):
    sys.argv = ["pipeline_smart.py", "/tmp"]
    import pipeline_smart
    return pipeline_smart


class _Req:
    block_text = "Конница мчится через поле"
    query = "medieval cavalry charge"
    shot_brief = None
    shot_spec = {"focus": "cavalry charging across a field"}


POOL = [{"id": 1, "alt": "moroccan fantasia horsemen in rabat"},
        {"id": 2, "alt": "a knight on horseback in plate armour"},
        {"id": 3, "alt": "horses in a field"}]


def _arm(ps, monkeypatch, gw, card=CARD):
    monkeypatch.setenv("CAPTION_SCREEN", "1")
    monkeypatch.setattr(ps, "shot_judge_active", lambda index=None: True)
    monkeypatch.setattr(ps, "episode_world_card", lambda *a: card)
    monkeypatch.setattr(ps, "_caption_screen_gateway", lambda: gw)
    ps.CAPTION_SCREEN_LOG.clear()


def test_pool_keeps_order_and_removes_only_agreed(ps, monkeypatch, tmp_path):
    monkeypatch.setattr(ps, "TEMP_FOLDER", str(tmp_path))
    _arm(ps, monkeypatch, FakeGateway(x={"1": "X"}, b=[1, 3]))
    out = ps.caption_screen_pool(list(POOL), _Req(), "photo", 0)
    assert [c["id"] for c in out] == [2, 3]
    assert ps.CAPTION_SCREEN_LOG and ps.CAPTION_SCREEN_LOG[0]["dropped"][0]["id"] == "1"


def test_pool_untouched_without_judge_or_flag(ps, monkeypatch, tmp_path):
    monkeypatch.setattr(ps, "TEMP_FOLDER", str(tmp_path))
    gw = FakeGateway(x={"1": "X"}, b=[1])
    _arm(ps, monkeypatch, gw)
    monkeypatch.setattr(ps, "shot_judge_active", lambda index=None: False)
    assert ps.caption_screen_pool(list(POOL), _Req(), "photo", 30) == POOL
    monkeypatch.setattr(ps, "shot_judge_active", lambda index=None: True)
    monkeypatch.setenv("CAPTION_SCREEN", "0")
    assert ps.caption_screen_pool(list(POOL), _Req(), "photo", 0) == POOL
    assert gw.calls == []


def test_pool_untouched_in_world_without_foreign_cultures(ps, monkeypatch, tmp_path):
    monkeypatch.setattr(ps, "TEMP_FOLDER", str(tmp_path))
    gw = FakeGateway(x={"1": "X"}, b=[1])
    _arm(ps, monkeypatch, gw, card=CARD_SCIENCE)
    assert ps.caption_screen_pool(list(POOL), _Req(), "video", 0) == POOL
    assert gw.calls == []


def test_screen_enters_judge_signature(ps, monkeypatch):
    monkeypatch.setattr(ps, "shot_judge_active", lambda index=None: True)
    monkeypatch.setenv("CAPTION_SCREEN", "1")
    on = ps.shot_judge_signature(0)
    monkeypatch.setenv("CAPTION_SCREEN", "0")
    assert ps.shot_judge_signature(0) != on
