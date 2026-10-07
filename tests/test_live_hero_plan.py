"""Планировщик: герой живьём (MASCOT_LIVE_PLAN) — только по флагу, только простые действия,
доля героя считается по всем его кадрам, без переписанного описания кадр возвращается рисунку."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import frame_planner as fp  # noqa: E402


def test_rule_and_field_only_with_flag(monkeypatch):
    monkeypatch.delenv("MASCOT_LIVE_PLAN", raising=False)
    assert "hero_action" not in fp.hero_rule_text("the black cat")
    obj = {"hero": True, "hero_action": "look"}
    assert "hero_action" not in fp.extras(obj, "фраза")[0]
    monkeypatch.setenv("MASCOT_LIVE_PLAN", "1")
    assert "hero_action" in fp.hero_rule_text("the black cat")
    assert fp.extras(obj, "фраза")[0]["hero_action"] == "look"
    assert "hero_action" not in fp.extras({"hero": True, "hero_action": "other"}, "фраза")[0]
    assert "hero_action" not in fp.extras({"hero": True, "hero_action": "point"}, "фраза")[0]   # жесты лап — не кукле
    assert "hero_action" not in fp.extras({"hero": False, "hero_action": "look"}, "фраза")[0]


def test_live_pass_after_limit_and_revert_without_rewrite():
    frames = [{"hero": True, "hero_action": "look", "picture": "the main character sits beside one envelope on the floor"},
              {"hero": True, "hero_action": "paw_on_chest", "picture": "one envelope lies on the floor, a worn path around it"},
              {"hero": True, "picture": "the main character holds its head"},
              {"hero": False, "picture": "a desk"}]
    assert fp.live_hero_pass(frames) == 2
    assert [f.get("hero") for f in frames] == [False, False, True, False]
    assert any(c["id"] == fp.LIVE_NOFIG_ID and c["tier"] == "must" for c in frames[0]["spec"]["claims"])
    assert fp.live_hero_pass(frames) == 0 and sum(c["id"] == fp.LIVE_NOFIG_ID for c in frames[0]["spec"]["claims"]) == 1
    assert frames[0]["hero_live"] and frames[1]["hero_live"]
    # предпроверки не было: кадр 0 всё ещё называет персонажа — возвращается рисунку; кадр 1 чист — остаётся живым
    assert fp.live_hero_revert(frames, "the black cat") == 1
    assert frames[0]["hero"] is True and not frames[0]["hero_live"]
    assert frames[1]["hero"] is False and frames[1]["hero_live"]
