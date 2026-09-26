# -*- coding: utf-8 -*-
"""Режиссёрский слой планировщика (версия 4): задание ролика, смысл фразы,
тип чтения, ловушки; задания по финальным блокам (подкадры не делят одно)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import stock_query_planner as sqp  # noqa: E402

RICH = json.dumps({
    "topic": "How the ADHD brain starts only when a deadline turns into now.",
    "genre": "Popular-science psychology explainer",
    "viewer": "Adults who recognise themselves; relief, not shame.",
    "look": "Two registers. " + "Real contemporary life, faces and hands, practical light. " * 12,
    "literal": "Figurative lines are the spine: «мозг не заводится» — never a literal engine.",
    "motifs": ["One recurring adult in three spaces"],
    "terms": [{"term": "дофамин", "show": "a hand hovering over an opened file", "avoid": "cartoon blobs"}],
    "never": ["medieval imagery", "fantasy creatures"]}, ensure_ascii=False)


def test_rich_answer_with_russian_terms_is_parsed():
    """Первая версия разбора выбросила именно такой ответ (длинный облик,
    русские термины) — главы шли без задания."""
    d = sqp.parse_direction(RICH)
    assert d and d["terms"][0]["term"] == "дофамин"
    assert len(d["look"]) <= 900 and d["look"].startswith("Two registers.")
    assert d["never"] == ["medieval imagery", "fantasy creatures"]


def test_direction_needs_topic_and_look():
    assert sqp.parse_direction('{"genre": "x"}') is None
    assert sqp.parse_direction("не json") is None


def test_direction_block_goes_into_chapter_prompt():
    d = sqp.parse_direction(RICH)
    packet = {"episode_title": "T", "units": [{"n": 1, "text": "Раз"}], "direction": d}
    prompt = sqp.render_spec_prompt(packet, "modern life")
    assert "Film direction" in prompt and "«дофамин»: show a hand hovering" in prompt
    assert "never show: medieval imagery; fantasy creatures" in prompt
    packet["direction"] = None
    assert "Film direction" not in sqp.render_spec_prompt(packet, "modern life")


def test_spec_extras_parsed_and_optional():
    packet = {"units": [{"n": 1, "text": "a"}, {"n": 2, "text": "b"}]}
    base = {"focus": "a ball bouncing off a wall", "core": "a ball is visible",
            "queries": [{"q": "ball bouncing wall", "for": ["core"], "type": "scene"}]}
    one = dict(base, n=1, meaning="the ball bounced off the wall", reading="literal",
               traps=["a ball lying still on a shelf", "x"])
    two = dict(base, n=2, reading="nonsense")
    got = sqp.parse_spec(json.dumps(one) + "\n" + json.dumps(two), packet)
    assert got[1]["meaning"] == "the ball bounced off the wall" and got[1]["reading"] == "literal"
    assert got[1]["traps"] == ["a ball lying still on a shelf"]     # «x» короче двух слов
    assert "reading" not in got[2] and "traps" not in got[2]        # сорванные поля не отнимают кадр


def test_plan_signature_depends_on_direction():
    d = sqp.parse_direction(RICH)
    assert sqp.plan_signature("m", "s", d) != sqp.plan_signature("m", "s", None)


def test_subcuts_get_their_own_specs():
    """Привязка по тексту финального блока: два подкадра одной фразы — два
    разных задания."""
    specs = {}
    import shot_planner_llm
    for t, f in (("Его часто называют гормоном удовольствия, но это упрощение.", "pleasure"),
                 ("Учёные всё чаще описывают дофамин как вещество ожидания.", "anticipation")):
        specs[shot_planner_llm.unit_key(t)] = {"focus": f, "claims": [{"id": "core"}], "queries": []}
    blocks = [{"text": "Его часто называют гормоном удовольствия, но это упрощение."},
              {"text": "Учёные всё чаще описывают дофамин как вещество ожидания."}]
    sqp.attach(blocks, {k: ["q"] for k in specs}, specs)
    assert [b["shot_spec"]["focus"] for b in blocks] == ["pleasure", "anticipation"]


def test_render_plans_after_the_split():
    """Порядок в main(): паспорт — до нарезки, задания и привязка — после."""
    src = open(os.path.join(os.path.dirname(__file__), "..", "scripts", "pipeline_smart.py"),
               encoding="utf-8").read()
    main = src[src.index("def main():"):]
    split = main.index("split_long_blocks(blocks, real_weights)")
    merge = main.index("merge_short_phrase_locked_blocks(blocks, real_weights, total)")
    assert main.index("auto_plan_episode(blocks, specs=False)") < split
    assert merge < main.index("auto_plan_episode(blocks, world=False)")
    assert merge < main.index("stock_query_planner.attach(blocks")


def test_direction_is_asked_once_and_reused(tmp_path):
    d = tmp_path / "ep"
    (d / "media_plan").mkdir(parents=True)
    (d / "script.txt").write_text("=== HOOK ===\nВот кинжал.\n", encoding="utf-8")

    class GW:
        calls = 0

        def chat(self, model, content, max_tokens, est, **kw):
            GW.calls += 1
            return RICH, {}, 1

    first = sqp.make_direction(str(d), GW(), model="m", setting="s", title="T")
    again = sqp.make_direction(str(d), GW(), model="m", setting="s", title="T")
    assert first == again and first["terms"][0]["term"] == "дофамин"
    assert GW.calls == 1, "второй раз — с диска, без вызова"
    assert sqp.load_direction(str(d)) == first


def test_failed_direction_keeps_chapters_going(tmp_path):
    import llm_gateway
    d = tmp_path / "ep"
    (d / "media_plan").mkdir(parents=True)
    (d / "script.txt").write_text("=== HOOK ===\nВот кинжал.\n", encoding="utf-8")

    class GW:
        def chat(self, *a, **k):
            raise llm_gateway.GatewayError("пустой ответ")

    assert sqp.make_direction(str(d), GW(), model="m", setting="s") is None
