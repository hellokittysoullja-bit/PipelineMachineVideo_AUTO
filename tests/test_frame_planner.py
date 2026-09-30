import json

import channel
import frame_planner as fp

PACKET = {"units": [{"n": 1, "text": "Утро начиналось с костра."},
                    {"n": 2, "text": "Жив. Полностью."},
                    {"n": 3, "text": "Три потребности."}]}


def line(n, frame):
    return json.dumps({"n": n, "focus": "a campfire in the morning", "subject": "a campfire",
                       "core": "a campfire is visible",
                       "claims": [{"id": "c1", "text": "children sit around it", "tier": "must"}],
                       "frame": frame}, ensure_ascii=False)


def test_parse_takes_spec_and_frame_and_loses_only_broken_line():
    raw = "\n".join([
        line(1, {"kind": "scene", "picture": "kids around a campfire in a cave at dawn"}),
        "{broken",
        line(2, {"kind": "caption", "labels": ["жив. полностью."], "picture": "a stick figure glowing in the sun"}),
        line(3, {"kind": "diagram", "labels": ["EAT"], "picture": "a pyramid with three tiers and labels"}),
    ])
    got, errors = fp.parse_answer(raw, PACKET)
    assert set(got) == {1, 2}
    assert got[1]["spec"]["claims"][0]["id"] == fp.CORE_ID
    assert got[2]["frame"]["labels"] == ["ЖИВ. ПОЛНОСТЬЮ."]
    assert any("latin_in_label" in e for e in errors)


def test_scene_never_carries_labels_and_motion_is_dropped():
    raw = json.dumps({"n": 1, "focus": "a ball flying", "core": "a ball is visible",
                      "claims": [{"id": "c1", "text": "the ball flies", "tier": "must", "motion": True}],
                      "frame": {"kind": "scene", "labels": ["МЯЧ"], "picture": "a ball flying over grass"}})
    got, _ = fp.parse_answer(raw, PACKET)
    assert got[1]["frame"]["labels"] == [] and all("motion" not in c for c in got[1]["spec"]["claims"])


def test_label_limits():
    assert fp.validate_frame({"kind": "diagram", "picture": "a b c d e", "labels": []})[1] == "no_labels"
    assert fp.validate_frame({"kind": "caption", "picture": "a b c d e",
                              "labels": ["ОДИН ДВА ТРИ ЧЕТЫРЕ ПЯТЬ ШЕСТЬ"]})[1].startswith("label_too_long")
    assert fp.validate_frame({"kind": "scene", "picture": "картинка по-русски тут нельзя"})[1] == "bad_picture"


def test_prompt_has_spec_rules_mascot_brief_and_prev_chapter():
    p = fp.render_prompt({"episode_title": "T", "prev_tail": "конец прошлой главы",
                          "units": [{"n": 1, "text": "x", "author_brief": "kids at a fire"}]},
                         channel.load_profile())
    assert "core — WHO or WHAT must be visible" in p       # правило v3 дословно
    assert "kids at a fire" in p and "конец прошлой главы" in p
    assert channel.load_profile()["mascot"]["description"] in p


def test_packets_group_by_section_with_tail():
    blocks = [{"text": "А.", "section": "HOOK"}, {"text": "Б.", "section": "BLOCK 1"},
              {"text": "В.", "section": "BLOCK 1"}]
    pk = fp.packets(blocks, "T")
    assert [p["section"] for p in pk] == ["HOOK", "BLOCK 1"]
    assert pk[1]["prev_tail"] == "А." and [u["block_index"] for u in pk[1]["units"]] == [1, 2]


def test_whole_episode_with_fake_model(tmp_path):
    (tmp_path / "script.txt").write_text("=== HOOK ===\nРаз. [pause] Два.\n=== FINAL ===\nТри.\n", encoding="utf-8")

    class GW:
        def chat(self, model, content, max_tokens, est, **kw):
            import re
            nums = re.findall(r"^(\d+)\. «", content[0]["text"], re.M)
            return "\n".join(line(int(n), {"kind": "scene", "picture": "kids around a small campfire"})
                             for n in nums), {}, 1

    plan = fp.plan_episode(str(tmp_path), GW(), model="fake", verbose=False)
    assert plan["stats"] == {"planned": 3, "fallback": 0, "cached_chapters": 0}
    plan2 = fp.plan_episode(str(tmp_path), None, model="fake", verbose=False)   # из кэша, без модели
    assert plan2["stats"]["cached_chapters"] == 2
