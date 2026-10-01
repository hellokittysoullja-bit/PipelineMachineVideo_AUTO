"""Спецификация кадра — на каждый слот, а не на фразу до нарезки (01.10).

Живой брак эп.03: фраза хука в 68 слов несла ДВА брифа [shot:] («две армии
конных рыцарей» и «монах пишет хронику»); парсер оставлял последний,
спецификация планировалась на всю фразу, и 7 слотов, нарезанных из неё,
наследовали одно «монах с пером» — на экране семь раз подряд монах.
Тесты держат каждое звено этой цепочки по отдельности.
"""
import inspect
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import script_parser  # noqa: E402
import stock_query_planner as sqp  # noqa: E402


def _ps():
    import pipeline_smart
    return pipeline_smart


SCRIPT = """=== HOOK ===
[shot:two armies of mounted knights charging]Звучит как выдумка, но сошлись девятьсот рыцарей и бились весь день.
[shot:a monk writing a chronicle]Монах записал, что погибло всего трое.
"""


def test_parser_keeps_every_brief_with_its_position(tmp_path):
    p = tmp_path / "script.txt"
    p.write_text(SCRIPT, encoding="utf-8")
    blocks = script_parser.parse_blocks(str(p))
    assert len(blocks) == 1
    b = blocks[0]
    assert [x["brief"] for x in b["shot_briefs"]] == [
        "two armies of mounted knights charging", "a monk writing a chronicle"]
    assert b["shot_briefs"][0]["word_pos"] == 0
    assert b["shot_briefs"][1]["word_pos"] == len(
        "Звучит как выдумка, но сошлись девятьсот рыцарей и бились весь день.".split())
    # Бриф блока — ПЕРВЫЙ (описывает начало фразы), а не последний.
    assert b["shot_brief"] == "two armies of mounted knights charging"


def _block(words, **kw):
    b = {"text": " ".join(words), "words": len(words), "section": "HOOK", "stat": None,
         "stat_word_pos": None, "pause_after": 0.5, "is_climax": False, "sfx": []}
    b.update(kw)
    return b


def test_subcuts_get_the_brief_written_inside_them_and_no_inherited_spec():
    ps = _ps()
    words = [f"w{i}" for i in range(40)]
    b = _block(words, shot_brief="A", shot_briefs=[{"word_pos": 0, "brief": "A"},
                                                    {"word_pos": 28, "brief": "B"}],
               shot_spec={"focus": "parent"}, phrase_queries=["parent query"])
    out, w = [], []
    ps._emit_subcut_blocks(b, 10.0, words, [(0, 10), (10, 30), (30, 40)], out, w)
    assert [x["shot_brief"] for x in out] == ["A", "B", None]
    assert [x["subcut_k"] for x in out] == [0, 1, 2]
    assert out[1]["shot_briefs"] == [{"word_pos": 18, "brief": "B"}]
    for x in out:
        assert "shot_spec" not in x and "phrase_queries" not in x


def test_director_brief_without_position_stays_on_first_piece_only():
    ps = _ps()
    words = [f"w{i}" for i in range(20)]
    b = _block(words, shot_brief="director", shot_type_hint="scene")
    out, w = [], []
    ps._emit_subcut_blocks(b, None, words, [(0, 10), (10, 20)], out, w)
    assert [x["shot_brief"] for x in out] == ["director", None]
    assert out[0].get("shot_type_hint") == "scene" and "shot_type_hint" not in out[1]


def test_merge_keeps_briefs_of_both_pieces():
    ps = _ps()
    pb = _block(["a", "b"], shot_brief="A", shot_briefs=[{"word_pos": 0, "brief": "A"}])
    cb = _block(["c"], shot_brief="B", shot_briefs=[{"word_pos": 0, "brief": "B"}])
    nb = ps._merge_two_blocks(pb, cb, 2)
    assert nb["shot_briefs"] == [{"word_pos": 0, "brief": "A"}, {"word_pos": 2, "brief": "B"}]
    assert nb["shot_brief"] == "A"


def test_fallback_rotates_parent_queries_and_gives_spec_to_first_piece_only():
    ps = _ps()
    import shot_planner_llm
    parent = "одна длинная фраза"
    key = shot_planner_llm.unit_key(parent)
    blocks = [{"text": f"кусок {k}", "parent_text": parent, "subcut_k": k} for k in range(3)]
    blocks.append({"text": "своя", "parent_text": "своя", "phrase_queries": ["own"]})
    n = ps.subcut_plan_fallback(blocks, {key: ["q1", "q2", "q3"]}, {key: {"focus": "f"}})
    assert n == 3
    assert [b["phrase_queries"][0] for b in blocks[:3]] == ["q1", "q2", "q3"]
    assert blocks[0]["shot_spec"] == {"focus": "f"}
    assert "shot_spec" not in blocks[1] and "shot_spec" not in blocks[2]
    assert blocks[3]["phrase_queries"] == ["own"]


def test_plan_is_attached_after_the_split_in_main():
    """Порядок в main(): нарезка -> план -> attach. Откат к «план до
    нарезки» роняет этот тест."""
    ps = _ps()
    src = inspect.getsource(ps.main)
    split = src.index("prepare_slot_blocks(")
    assert src.index("auto_plan_episode(") > split
    assert src.index("stock_query_planner.attach(") > split
    assert src.index("subcut_plan_fallback(") > split


def test_cli_planner_builds_slots_the_same_way_as_render():
    src = inspect.getsource(sqp.main)
    assert "prepare_slot_blocks(" in src


def test_long_chapter_is_asked_in_portions_with_context():
    units = [{"n": i + 1, "text": f"line {i}"} for i in range(32)]
    parts = list(sqp._portions([{"section": "S", "units": units, "prev_tail": "before"}], size=15))
    assert [len(p["units"]) for p in parts] == [15, 15, 2]
    assert [u["n"] for u in parts[2]["units"]] == [31, 32]     # исходные номера главы
    assert parts[0]["prev_tail"] == "before" and parts[1]["prev_tail"] == "before"
    assert parts[1]["chapter_context"][14] == "15. line 14" and parts[1]["portion"] == (16, 30)


def test_prompt_demands_different_shots_for_neighbouring_lines():
    assert "Neighbouring lines must show DIFFERENT things" in sqp.SPEC_PROMPT


def test_same_shoot_means_same_author_and_close_upload_number():
    ps = _ps()
    ps.SLOT_SHOOT.clear()
    ps.SLOT_SHOOT[9] = "pexels:42:7977776"
    same_shoot = {"id": 7977676, "user": {"id": 42}}           # та самая пара эп.03
    same_author_far = {"id": 31474665, "user": {"id": 42}}     # агентство, другая съёмка
    other = {"id": 7977700, "user": {"id": 7}}
    try:
        assert ps.shoot_is_recent(same_shoot, 10)
        assert ps.shoot_is_recent(same_shoot, 12)
        assert not ps.shoot_is_recent(same_shoot, 13)          # за пределом окна
        assert not ps.shoot_is_recent(same_author_far, 10)     # автор = агентство, не съёмка
        assert not ps.shoot_is_recent(other, 10)
        assert not ps.shoot_is_recent({"id": 5}, 10)           # автор неизвестен — не судим
        # аккаунт-переливка «Pixabay» на Pexels — не съёмка
        assert ps.candidate_shoot({"id": 1, "photographer_id": 2659}) is None
    finally:
        ps.SLOT_SHOOT.clear()


def test_cached_candidate_of_the_same_shoot_is_not_taken(tmp_path):
    ps = _ps()
    f = tmp_path / "x.jpg"
    f.write_bytes(b"x")
    ps.write_media_sidecar(str(f), pexels_id=7977676, shoot="pexels:42:7977676")
    ps.SLOT_SHOOT.clear()
    ps.SLOT_SHOOT[9] = "pexels:42:7977776"
    try:
        assert ps.cached_shoot_is_recent(str(f), 10)
        assert not ps.cached_shoot_is_recent(str(f), 20)
    finally:
        ps.SLOT_SHOOT.clear()


def test_old_plan_version_still_gives_specs_without_a_gateway_key():
    assert sqp.PLAN_VERSION == 3


def test_portion_sees_the_whole_chapter_with_the_same_numbers():
    units = [{"n": i + 1, "text": f"line {i}"} for i in range(20)]
    parts = list(sqp._portions([{"section": "S", "units": units, "episode_title": "T"}], size=15))
    assert [u["n"] for u in parts[1]["units"]] == [16, 17, 18, 19, 20]
    prompt = sqp.render_spec_prompt(parts[1], "medieval")
    assert "1. line 0" in prompt and "20. line 19" in prompt and "lines 16 to 20" in prompt
    assert "\n16. «line 15»" in prompt          # строка порции — под исходным номером


def test_portion_answer_with_chapter_numbers_is_parsed(tmp_path):
    """Двойник модели отвечает исходными номерами главы — спецификации
    второй порции не теряются."""
    units = [{"n": i + 1, "text": f"line {i}"} for i in range(20)]
    part = list(sqp._portions([{"section": "S", "units": units}], size=15))[1]
    ans = "\n".join('{"n": %d, "focus": "a sword on a table", "core": "a sword is visible", '
                     '"subject": "a sword", "claims": [], "queries": [{"q": "medieval sword", '
                     '"for": ["core"], "type": "object"}]}' % u["n"] for u in part["units"])
    got = sqp.parse_spec(ans, part)
    assert got is not None and sorted(got) == [16, 17, 18, 19, 20]


def test_all_briefs_of_a_slot_reach_the_planner():
    prompt = sqp.render_spec_prompt({"units": [{"n": 1, "text": "t", "author_brief": "A",
                                                "author_briefs": ["A", "B"]}]}, "x")
    assert "shot: A; also: B" in prompt


def test_brief_at_the_very_end_goes_to_the_last_piece():
    ps = _ps()
    words = [f"w{i}" for i in range(20)]
    b = _block(words, shot_brief="E", shot_briefs=[{"word_pos": 20, "brief": "E"}])
    out, w = [], []
    ps._emit_subcut_blocks(b, None, words, [(0, 10), (10, 20)], out, w)
    assert [x["shot_brief"] for x in out] == [None, "E"]


def test_merge_keeps_the_director_brief_of_the_first_piece():
    ps = _ps()
    pb = _block(["a"], shot_brief="director")
    cb = _block(["b"], shot_brief="B", shot_briefs=[{"word_pos": 0, "brief": "B"}])
    nb = ps._merge_two_blocks(pb, cb, 1)
    assert nb["shot_brief"] == "director" and nb["shot_briefs"][0]["brief"] == "B"


def test_orphan_keys_of_a_fallback_model_do_not_force_replanning(tmp_path, monkeypatch):
    import json
    import shot_planner_llm
    import world_card
    monkeypatch.setattr(world_card, "load", lambda *a, **k: None)
    monkeypatch.setattr(world_card, "judge_setting", lambda c: "s")
    d = tmp_path / "media_plan"
    d.mkdir()
    cur = shot_planner_llm.unit_key("текущий слот")
    plan = {"version": sqp.PLAN_VERSION, "sig": sqp.plan_signature(sqp.DEFAULT_MODEL, "s"),
            "units": {cur: {"model": sqp.DEFAULT_MODEL},
                      "orphan": {"model": "some/fallback"}}}
    (d / sqp.PLAN_NAME).write_text(json.dumps(plan), encoding="utf-8")
    assert not sqp.needs_planning(str(tmp_path), [{"text": "текущий слот"}])




def test_changed_brief_with_the_same_text_forces_replanning(tmp_path, monkeypatch):
    import json
    import shot_planner_llm
    import world_card
    monkeypatch.setattr(world_card, "load", lambda *a, **k: None)
    monkeypatch.setattr(world_card, "judge_setting", lambda c: "s")
    d = tmp_path / "media_plan"
    d.mkdir()
    key = shot_planner_llm.unit_key("слот")
    plan = {"version": sqp.PLAN_VERSION, "sig": sqp.plan_signature(sqp.DEFAULT_MODEL, "s"),
            "units": {key: {"model": sqp.DEFAULT_MODEL, "bsig": sqp.briefs_signature(["a monk"])}}}
    (d / sqp.PLAN_NAME).write_text(json.dumps(plan), encoding="utf-8")
    assert not sqp.needs_planning(str(tmp_path), [{"text": "слот", "shot_brief": "a monk"}])
    assert sqp.needs_planning(str(tmp_path), [{"text": "слот", "shot_brief": "two armies"}])


def test_spec_planned_for_other_briefs_is_not_attached():
    import shot_planner_llm
    key = shot_planner_llm.unit_key("слот")
    spec = {"focus": "monk", "bsig": sqp.briefs_signature(["a monk"])}
    b = {"text": "слот", "shot_brief": "two armies"}
    sqp.attach([b], {key: ["monk writing"]}, {key: spec})
    assert "shot_spec" not in b
    assert b["phrase_queries"] == ["monk writing"]   # запросы фразы остаются
    b2 = {"text": "слот", "shot_brief": "a monk"}
    sqp.attach([b2], {key: ["monk writing"]}, {key: spec})
    assert b2["shot_spec"] == {"focus": "monk"}       # без служебного bsig
    b3 = {"text": "слот", "shot_brief": "a monk"}      # план до отпечатков — принимается
    sqp.attach([b3], {key: ["monk writing"]}, {key: {"focus": "monk"}})
    assert b3["shot_spec"] == {"focus": "monk"}


def test_full_round_plan_load_attach_keeps_spec_of_a_slot_with_brief(tmp_path, monkeypatch):
    """plan_episode -> load_specs -> attach на слоте с брифом автора: спецификация
    доезжает до слота. Четвёртый круг критика: load_specs терял bsig, attach его
    сверял — 128 слотов из 180 эп.03 оставались без спецификации."""
    import json
    import world_card
    monkeypatch.setattr(world_card, "load", lambda *a, **k: None)
    monkeypatch.setattr(world_card, "judge_setting", lambda c: "s")
    import shot_brief_director as sbd
    blocks = [{"text": "Вот кинжал.", "section": "HOOK", "shot_brief": "a dagger in a palm",
               "shot_briefs": [{"word_pos": 0, "brief": "a dagger in a palm"}]}]
    monkeypatch.setattr(sbd, "packets", lambda d, bl, *a, **k: [{
        "section": "HOOK", "units": [{"n": 1, "text": "Вот кинжал.", "author_brief": "a dagger in a palm",
                                      "author_briefs": ["a dagger in a palm"]}]}])
    ans = ('{"n": 1, "focus": "a dagger in a palm", "core": "a dagger is visible", "subject": "a dagger",'
           ' "claims": [], "queries": [{"q": "dagger palm", "for": ["core"], "type": "object"}]}')
    monkeypatch.setattr(sqp, "ask_chapter", lambda *a, **k: (sqp.parse_spec(ans, a[2]), False))
    (tmp_path / "media_plan").mkdir()
    assert sqp.plan_episode(str(tmp_path), blocks, object(), verbose=False) == 1
    specs = sqp.load_specs(str(tmp_path))
    assert [v.get("bsig") for v in specs.values()] == [sqp.briefs_signature(["a dagger in a palm"])]
    b = dict(blocks[0])
    sqp.attach([b], sqp.load(str(tmp_path)), specs)
    assert b["shot_spec"]["focus"] == "a dagger in a palm" and "bsig" not in b["shot_spec"]
    assert b["phrase_queries"]
    assert not sqp.needs_planning(str(tmp_path), blocks)


def test_neighbour_repeats_are_named():
    ps = _ps()
    bl = [{"section": "H", "shot_spec": {"subject": "a scribe"}, "phrase_queries": ["q1"]},
          {"section": "H", "shot_spec": {"subject": "a scribe"}, "phrase_queries": ["q2"]},
          {"section": "H", "shot_spec": {"subject": "a knight"}, "phrase_queries": ["q2"]},
          {"section": "B", "shot_spec": {"subject": "a knight"}, "phrase_queries": ["q2"]}]
    got = ps.neighbour_plan_repeats(bl)
    assert got == ["слоты 0 и 1 — один предмет «a scribe»", "слоты 1 и 2 — один первый запрос «q2»"]
