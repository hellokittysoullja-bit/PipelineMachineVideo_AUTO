import json

import frame_planner as fp
import stock_query_planner as sqp

PACKET = {"units": [{"n": 1, "text": "Утро начиналось с костра."},
                    {"n": 2, "text": "Жив. Полностью."},
                    {"n": 3, "text": "Три потребности."}]}


def line(n, frame, core="a campfire is visible"):
    return json.dumps({"n": n, "focus": "a campfire in the morning", "subject": "a campfire",
                       "core": core, "claims": [{"id": "c1", "text": "children sit around it", "tier": "must"}],
                       "frame": frame}, ensure_ascii=False)


def test_spec_rules_are_the_old_planner_text_verbatim():
    # правила спецификации — дословно из старого планировщика, второй копии нет
    rules = fp.spec_rules()
    assert rules in sqp.SPEC_PROMPT and rules.startswith("focus —") and "queries —" not in rules


def test_parse_takes_spec_and_frame_and_loses_only_broken_line():
    raw = "\n".join([
        line(1, {"kind": "scene", "picture": "kids around a campfire in a cave at dawn"}),
        "{broken",
        line(2, {"kind": "caption", "labels": ["жив. полностью."], "picture": "a stick figure glowing in the sun"}),
        line(3, {"kind": "diagram", "labels": ["EAT"], "picture": "a pyramid with three tiers and labels"}),
    ])
    got, errors = fp.parse_answer(raw, PACKET)
    assert set(got) == {1, 2}
    assert got[1]["spec"]["claims"][0]["id"] == sqp.CORE_ID
    assert got[2]["frame"]["labels"] == ["ЖИВ. ПОЛНОСТЬЮ."]
    assert any("latin_in_label" in e for e in errors)


def test_scene_never_carries_labels_and_motion_is_dropped():
    raw = json.dumps({"n": 1, "focus": "a ball flying", "core": "a ball is visible",
                      "claims": [{"id": "c1", "text": "the ball flies", "tier": "must", "motion": True}],
                      "frame": {"kind": "scene", "labels": ["МЯЧ"], "picture": "a ball flying over grass"}})
    got, _ = fp.parse_answer(raw, PACKET)
    assert got[1]["frame"]["labels"] == []
    assert all("motion" not in c for c in got[1]["spec"]["claims"])


def test_label_limits():
    assert fp.validate_frame({"kind": "diagram", "picture": "a b c d e", "labels": []})[1] == "no_labels"
    assert fp.validate_frame({"kind": "caption", "picture": "a b c d e",
                              "labels": ["ОДИН ДВА ТРИ ЧЕТЫРЕ ПЯТЬ ШЕСТЬ"]})[1].startswith("label_too_long")
    assert fp.validate_frame({"kind": "scene", "picture": "картинка по-русски тут нельзя"})[1] == "bad_picture"


def test_fallback_is_textless_scene_with_judgeable_spec():
    fb = fp.fallback({"text": "Утро.", "shot_brief": "kids around a fire"})
    assert fb["frame"]["kind"] == "scene" and fb["frame"]["labels"] == [] and fb["fallback"]
    assert fb["spec"]["claims"][0]["tier"] == "must"


def test_prompt_has_setting_mascot_and_author_brief():
    import channel
    p = fp.render_prompt({"episode_title": "T", "units": [{"n": 1, "text": "x", "author_brief": "kids at a fire"}]},
                         "historical, 10000 BC-3000 BC", channel.load_profile())
    assert "historical, 10000 BC-3000 BC" in p and "kids at a fire" in p
    assert channel.load_profile()["mascot"]["description"] in p
