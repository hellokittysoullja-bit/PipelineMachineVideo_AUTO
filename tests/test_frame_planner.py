import json

import frame_planner as fp


def test_validate_scene_drops_labels():
    fr, err = fp.validate({"kind": "scene", "picture": "kids around a fire in a cave", "labels": ["ОГОНЬ"]})
    assert err is None and fr["labels"] == []


def test_validate_rejects_latin_and_long_labels():
    assert fp.validate({"kind": "caption", "picture": "a b c d e", "labels": ["ALIVE"]})[1].startswith("latin")
    assert fp.validate({"kind": "caption", "picture": "a b c d e",
                        "labels": ["ОДИН ДВА ТРИ ЧЕТЫРЕ ПЯТЬ ШЕСТЬ"]})[1].startswith("label_too_long")
    assert fp.validate({"kind": "diagram", "picture": "a b c d e", "labels": []})[1] == "no_labels"


def test_labels_are_uppercased():
    fr, _ = fp.validate({"kind": "caption", "picture": "a stick figure in the desert", "labels": ["жив. полностью."]})
    assert fr["labels"] == ["ЖИВ. ПОЛНОСТЬЮ."]


def test_parse_answer_loses_only_broken_line():
    text = "\n".join([
        json.dumps({"n": 1, "kind": "scene", "picture": "kids carry water from a river"}),
        "{broken",
        json.dumps({"n": 3, "kind": "caption", "picture": "a stick figure smiling in the sun", "labels": ["ЖИВ"]}),
        json.dumps({"n": 9, "kind": "scene", "picture": "out of range line here"}),
    ])
    frames, errors = fp.parse_answer(text, 3)
    assert set(frames) == {1, 3}
    assert "bad_json" in errors and any(e.startswith("bad_n") for e in errors)


def test_author_brief_goes_into_prompt():
    units = [(0, {"text": "Утро начиналось с костра.", "shot_brief": "kids around a fire"})]
    p = fp.build_prompt("BLOCK 1", units, title="T", niche="n", mascot="m", prev_tail="")
    assert "author brief: kids around a fire" in p


def test_fallback_is_scene_without_text():
    fr = fp.fallback_frame({"text": "Утро.", "shot_brief": None})
    assert fr["kind"] == "scene" and fr["labels"] == [] and fr["fallback"]
