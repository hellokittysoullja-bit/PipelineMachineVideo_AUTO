import json

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


def test_prompt_has_spec_rules_brief_prev_chapter_and_hero_rule():
    pk = {"episode_title": "T", "prev_tail": "конец прошлой главы",
          "units": [{"n": 1, "text": "x", "author_brief": "kids at a fire"}]}
    p = fp.render_prompt(pk, True)
    assert "core — WHO or WHAT must be visible" in p       # правило v3
    assert "kids at a fire" in p and "конец прошлой главы" in p
    assert "one recurring main character" in p and "never on 3 lines in a row" in p
    assert "never describe the drawing style" in p
    assert '"zoom"' in p and '"key"' in p and "paw prints" in p
    assert "always false" in fp.render_prompt(pk, False)


def test_hero_is_only_explicit_true():
    def ok(h):
        return fp.validate_frame({"kind": "scene", "picture": "a b c d e", "hero": h})[0]["hero"]
    assert ok(True) is True and ok("yes") is False and ok(None) is False


def test_hero_never_three_in_a_row_and_at_most_a_third():
    frames = [{"hero": h, "picture": "the main character waves"} for h in
              (True, True, True, True, False, True, True, False, False, False, False, False)]
    trimmed = fp.limit_hero(frames)
    heroes = [f["hero"] for f in frames]
    assert sum(heroes) <= int(fp.MAX_HERO_SHARE * len(frames))
    assert not any(heroes[i] and heroes[i + 1] and heroes[i + 2] for i in range(len(heroes) - 2))
    assert trimmed == 6 - sum(heroes)
    # снятый герой не превращается в «другого главного героя»
    assert all(f["picture"] == "a person waves" for f in frames if not f["hero"] and "person" in f["picture"])
    assert all("main character" in f["picture"] for f in frames if f["hero"])


def test_no_hero_file_means_no_hero_anywhere(tmp_path):
    (tmp_path / "script.txt").write_text("=== HOOK ===\nРаз. [pause] Два.\n", encoding="utf-8")

    class GW:
        def chat(self, model, content, max_tokens, est, **kw):
            import re
            nums = re.findall(r"^(\d+)\. «", content[0]["text"], re.M)
            return "\n".join(line(int(n), {"kind": "scene", "hero": True,
                                            "picture": "the main character sits by a small campfire"})
                             for n in nums), {}, 1

    plan = fp.plan_episode(str(tmp_path), GW(), model="fake", verbose=False, has_hero=False)
    assert not any(f["hero"] for f in plan["frames"])
    assert all("main character" not in f["picture"] for f in plan["frames"])


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
    assert plan["stats"] == {"planned": 3, "fallback": 0, "cached_chapters": 0, "hero_trimmed": 0}
    plan2 = fp.plan_episode(str(tmp_path), None, model="fake", verbose=False)   # из кэша, без модели
    assert plan2["stats"]["cached_chapters"] == 2


def test_prompt_keeps_the_measured_rules_of_the_old_generator():
    """Правила, замеренные в старом генераторе (shot_brief_director, shot_generator),
    и то, что было в v4: тихо пропасть при правке промпта они не должны."""
    p = fp.render_prompt({"episode_title": "T", "prev_tail": "", "units": [{"n": 1, "text": "x"}]}, True)
    for rule in ("show what is NEW in this line",                 # «новое, а не главное»
                 "a comparison that only flashes by",             # мимолётное сравнение
                 "ONE picture with both side by side",            # противопоставление — один кадр
                 "stock symbols are not",                         # без штампов-символов
                 "described by its look, or replaced by a familiar object",  # незнакомый предмет
                 "A period is named in words, never as years",    # годы рисуются надписью
                 "no pronouns without a clear owner",
                 "keep the mood of the chapter",
                 "must show the core and every must claim",       # связь рисунка со спецификацией
                 "a path of footprints",                          # набор схем канала
                 "the core is the ball, not the wall",            # урок примера v3
                 "named the same way both times",                 # 01.10: «экран» нарисован планшетом
                 "name as many patches as there are labels"):     # 01.10: 4 подписи на 3 места
        assert rule in p, rule



def test_hero_share_and_run_come_from_env(monkeypatch):
    """Маскот — лицо канала: доля героя задаётся в .env и попадает и в правило
    кода, и в просьбу к модели (иначе модель просит треть, а код режет до трети)."""
    def frames(n):
        return [{"hero": True, "picture": "the main character waves", "kind": "scene", "labels": []} for _ in range(n)]
    monkeypatch.setenv("HERO_MAX_SHARE", "0.6")
    monkeypatch.setenv("HERO_MAX_RUN", "2")
    f = frames(5)
    fp.limit_hero(f)
    assert sum(x["hero"] for x in f) == 3
    assert "at most 60% of the pictures" in fp.hero_rule_text() and "never on 3 lines in a row" in fp.hero_rule_text()
    monkeypatch.delenv("HERO_MAX_SHARE")
    f = frames(5)
    fp.limit_hero(f)
    assert sum(x["hero"] for x in f) == 1                      # по умолчанию — герой-гость, треть
    assert "at most 35% of the pictures" in fp.hero_rule_text()


PK2 = {"units": [{"n": 1, "text": "Ответить на одно письмо — это пять минут."},
                 {"n": 2, "text": "Поэтому договорись с собой: только открыть письмо."}]}


def _l(n, frame):
    return json.dumps({"n": n, "focus": "a letter", "core": "a letter is visible", "claims": [],
                       "frame": frame}, ensure_ascii=False)


def test_zoom_and_key_must_be_words_of_the_line():
    raw = "\n".join([
        _l(1, {"kind": "scene", "picture": "a cat circles a desk with a letter", "hero": True,
               "zoom": {"object": "the letter", "word": "письмо"}, "key": "Пять минут", "hero_state": "ember"}),
        _l(2, {"kind": "scene", "picture": "a cat opens a letter on a desk", "hero": True,
               "zoom": {"object": "the letter", "word": "конверт"}, "key": "только открыть сейчас же",
               "hero_state": "flying"}),
    ])
    got, errors = fp.parse_answer(raw, PK2, ("ember", "bright"))
    f1, f2 = got[1]["frame"], got[2]["frame"]
    assert f1["zoom"] == {"object": "the letter", "word": "письмо"}   # «письмо» есть в «письмо»
    assert f1["key_thought"] == "пять минут" and f1["hero_state"] == "ember"
    assert "zoom" not in f2 and "key_thought" not in f2 and "hero_state" not in f2
    assert "2:zoom_dropped" in errors and "2:key_dropped" in errors


def test_hero_state_needs_hero_and_is_dropped_with_hero():
    raw = _l(1, {"kind": "scene", "picture": "a desk with a letter on it", "hero": False, "hero_state": "ember"})
    got, _ = fp.parse_answer(raw, PK2, ("ember",))
    assert "hero_state" not in got[1]["frame"]
    f = {"hero": True, "hero_state": "ember", "picture": "the main character sits"}
    fp._drop_hero(f)
    assert "hero_state" not in f and f["picture"] == "a person sits"


def test_hero_rule_names_the_hero_and_its_states():
    pk = {"episode_title": "T", "prev_tail": "", "units": [{"n": 1, "text": "x"}]}
    p = fp.render_prompt(pk, True, {"text": "a black cartoon cat",
                                    "states": {"ember": {"when": "stuck", "draw": "dim ember"}}})
    assert "(a black cartoon cat)" in p and '"hero_state"' in p and '"ember" when stuck' in p


def test_key_near_kept_only_with_key_and_in_english():
    raw = _l(1, {"kind": "scene", "picture": "a desk with a letter on it", "key": "пять минут",
                 "key_near": "the envelope"})
    got, _ = fp.parse_answer(raw, PK2)
    assert got[1]["frame"]["key_near"] == "the envelope"
    raw = _l(1, {"kind": "scene", "picture": "a desk with a letter on it", "key": "пять минут", "key_near": "конверт"})
    assert "key_near" not in fp.parse_answer(raw, PK2)[0][1]["frame"]


def test_failed_chapter_never_overwrites_an_existing_plan(tmp_path):
    import os
    (tmp_path / "media_plan").mkdir()
    (tmp_path / "script.txt").write_text("=== HOOK ===\nОдна фраза здесь.\n", encoding="utf-8")
    old = tmp_path / "media_plan" / "frame_plan.json"
    old.write_text('{"frames": "old"}', encoding="utf-8")

    class Down:
        def chat(self, *a, **k):
            raise RuntimeError("429")
    plan = fp.plan_episode(str(tmp_path), Down(), verbose=False)
    assert plan["not_written"] == ["HOOK"]
    assert old.read_text(encoding="utf-8") == '{"frames": "old"}'
    assert os.path.exists(old)


def test_partial_answer_does_not_overwrite_either(tmp_path):
    (tmp_path / "media_plan").mkdir()
    (tmp_path / "script.txt").write_text("=== HOOK ===\nОдна фраза. [pause] Вторая фраза.\n", encoding="utf-8")
    old = tmp_path / "media_plan" / "frame_plan.json"
    old.write_text('{"frames": "old"}', encoding="utf-8")

    class Half:
        def chat(self, *a, **k):
            return line(1, {"kind": "scene", "picture": "kids around a campfire in a cave"}), {}, 0
    plan = fp.plan_episode(str(tmp_path), Half(), verbose=False)
    assert plan["not_written"] == ["HOOK"] and old.read_text(encoding="utf-8") == '{"frames": "old"}'


def test_trim_keeps_the_hero_where_it_carries_meaning():
    frames = [{"hero": True, "picture": "the main character waves"} for _ in range(5)]
    frames[3]["key_thought"] = "только открыть"
    frames[3]["hero_state"] = "bright"
    frames[0]["zoom"] = {"object": "x", "word": "y"}
    fp.limit_hero(frames, max_run=4, max_share=0.6)
    assert frames[3]["hero"] and frames[0]["hero"]
    assert sum(f["hero"] for f in frames) == 3


def test_pictures_that_invite_writing_are_rejected():
    bad = ["blobs labeled by shape as shame and anxiety on an envelope", "a cat walks from a starting line to a desk",
           "a calendar with three crossed-off days on the wall", "a road sign next to the cat"]
    for t in bad:
        assert fp.validate_frame({"kind": "scene", "picture": t})[1].startswith("writing_in_picture"), t
    ok = ["a phone with a blank glowing screen on a desk", "a clock face without numerals on the wall",
          "the main character holds one envelope with both paws"]
    for t in ok:
        assert fp.validate_frame({"kind": "scene", "picture": t})[1] is None, t
    d = {"kind": "diagram", "labels": ["А"], "picture": "a pyramid, an empty patch next to each labelled tier"}
    assert fp.validate_frame(d)[1] is None


def test_a_letter_is_mail_not_writing():
    assert fp.validate_frame({"kind": "scene", "picture": "one unopened letter lies on the desk"})[1] is None


def test_accent_is_words_of_the_line_and_never_with_key():
    raw = _l(1, {"kind": "scene", "picture": "a desk with a letter on it", "accent": "пять минут"})
    assert fp.parse_answer(raw, PK2)[0][1]["frame"]["accent"] == "пять минут"
    raw = _l(1, {"kind": "scene", "picture": "a desk with a letter on it", "accent": "пять минут", "key": "пять минут"})
    assert "accent" not in fp.parse_answer(raw, PK2)[0][1]["frame"]
    raw = _l(1, {"kind": "scene", "picture": "a desk with a letter on it", "accent": "десять часов"})
    assert "accent" not in fp.parse_answer(raw, PK2)[0][1]["frame"]


def test_details_are_kept_english_unique_and_at_most_three():
    import frame_planner as fp
    out, _n = fp.extras({"details": ["the chain and the boulder", "цепь", "the chain and the boulder",
                                     "the smoking tail", "a", "b"]}, "текст")
    assert out["details"] == ["the chain and the boulder"]       # только первые три, без кириллицы и повторов
    out, _n = fp.extras({"details": ["the chain", "the tail", "the paw"]}, "текст")
    assert out["details"] == ["the chain", "the tail", "the paw"]
    assert "details" not in fp.extras({"details": "the chain"}, "текст")[0]
