import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import frame_generator as fg  # noqa: E402
import look  # noqa: E402


def test_person_replaced_everywhere():
    spec = {"focus": "a person circling a letter", "subject": "a person",
            "claims": [{"id": "core", "text": "a person taking one step forward is visible", "tier": "must"},
                       {"id": "c1", "text": "the person looks back with relief", "tier": "should"}]}
    out = fg.hero_spec(spec, "a black cartoon cat")
    assert out["subject"] == "a black cartoon cat"
    assert "person" not in out["focus"]
    assert all("person" not in c["text"] for c in out["claims"])
    assert out["claims"][0]["tier"] == "must"
    assert spec["subject"] == "a person"          # исходник не тронут


def test_hero_text_loaded(tmp_path):
    (tmp_path / "style").mkdir()
    from PIL import Image
    Image.new("RGB", (8, 8)).save(tmp_path / "style" / "1.png")
    Image.new("RGB", (8, 8)).save(tmp_path / "hero.png")
    (tmp_path / "hero.txt").write_text("  a black\ncartoon cat \n", encoding="utf-8")
    assert look.load(str(tmp_path)).hero_text == "a black cartoon cat"


def test_judge_spec_only_for_hero_frames():
    g = fg.Generator.__new__(fg.Generator)
    g.look = look.Look(["s"], "h", "a black cartoon cat")
    spec = {"focus": "a person", "subject": "a person", "claims": []}
    assert g._judge_spec({"hero": True, "spec": spec})["subject"] == "a black cartoon cat"
    assert g._judge_spec({"hero": False, "spec": spec})["subject"] == "a person"


def test_hero_states_load_and_validate(tmp_path):
    import json
    import pytest
    import look
    from PIL import Image
    (tmp_path / "style").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "style" / "a.png")
    Image.new("RGB", (8, 8)).save(tmp_path / "hero.png")
    (tmp_path / "hero_states.json").write_text(json.dumps({"ember": {"when": "stuck", "draw": "dim  ember"}}))
    assert look.load(str(tmp_path)).hero_states == {"ember": {"when": "stuck", "draw": "dim ember"}}
    (tmp_path / "hero_states.json").write_text(json.dumps({"ember": {"when": ""}}))
    with pytest.raises(look.LookError):
        look.load(str(tmp_path))
