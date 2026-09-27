#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Запросы к стокам на каждую фразу: план по главам, проверка ответа
модели, подключение к слоту и бюджет квоты Pexels. Без сети."""
import json
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.argv = ["pipeline_smart.py", REPO]
import pipeline_smart as ps  # noqa: E402
import stock_query_planner as sqp  # noqa: E402

SCRIPT = """=== METADATA ===
TITLE: Тест
=== HOOK ===
[shot:a medieval rondel dagger, studio shot]Вот кинжал.[pause]
[shot:an arrow glancing off a dented steel breastplate]Стрела скользит по нагруднику.[pause]
=== FINAL ===
Итог.[pause]
"""


CARD = {"schema_version": 1, "register": "historical", "era": {"from": 1300, "to": 1500}, "culture": {"include": [], "exclude": ["asian", "japanese", "chinese", "ottoman", "islamic"]}, "must_not_show": ["modern tactical knife", "modern military uniform", "napoleonic uniform", "firearm", "gunpowder era"], "expected_subjects": ["rondel dagger", "knight", "plate armour", "cavalry", "arrow", "muddy battlefield"], "era_anchor_terms": ["medieval", "knight", "14th century", "15th century"]}


class FakeGateway:
    """Отвечает на окна главы хука; на остальное (библия, другие главы)
    молчит (пустой ответ)."""

    def __init__(self, answer):
        self.answer, self.prompts = answer, []

    def chat(self, model, content, max_tokens, est, **kw):
        import llm_gateway
        text = content[0]["text"]
        self.prompts.append(text)
        answer = self.answer if "1. «Вот кинжал.»" in text and "Plan shots" in text else ""
        if not answer:
            # Настоящий шлюз на пустой ответ поднимает EmptyAnswer, а не
            # отдаёт пустую строку (llm_gateway, «пустой ответ — ошибка»).
            raise llm_gateway.EmptyAnswer("пустой ответ")
        return answer, {}, 1


def _episode(tmp_path, card=None):
    d = tmp_path / "ep"
    (d / "media_plan").mkdir(parents=True)
    (d / "script.txt").write_text(SCRIPT, encoding="utf-8")
    if card:
        (d / "media_plan" / "world_card.json").write_text(json.dumps(card), encoding="utf-8")
    import script_parser
    return str(d), script_parser.parse_blocks(str(d / "script.txt"))


def test_clean_query_keeps_only_short_latin_queries():
    assert sqp.clean_query(' "Medieval Battle Reenactment". ') == "medieval battle reenactment"
    assert sqp.clean_query("1. knight armour mud") == "knight armour mud"
    assert sqp.clean_query("рыцарь в грязи") is None
    # Запрос-знание называет работу архивным названием: до 7 слов, с годом.
    assert sqp.clean_query("Battle of Poitiers 1356 miniature") == "battle of poitiers 1356 miniature"
    assert sqp.clean_query("a b c d e f g") == "a b c d e f g"
    assert sqp.clean_query("a b c d e f g h") is None
    assert sqp.clean_query("") is None


def test_prompt_carries_no_niche_words_and_takes_the_world_from_the_card(tmp_path):
    card = CARD
    d, blocks = _episode(tmp_path, card=card)
    gw = FakeGateway("")
    sqp.plan_episode(d, blocks, gw, model="m", verbose=False)
    prompt = [p for p in gw.prompts if "Plan shots" in p][0]
    assert "register: historical" in prompt and "era: 1300 AD to 1500 AD" in prompt
    assert "never show: modern tactical knife" in prompt, "планировщик видит паспорт целиком"
    for template in (sqp.SPEC_PROMPT, sqp.BIBLE_PROMPT, sqp.BRIEF_RULE):
        t = template.lower()
        for word in ("medieval", "knight", "sword", "armour", "europe", "dagger"):
            assert word not in t, f"слово ниши «{word}» в шаблоне вопроса"


def _shot(n, core, queries, meaning="a thing in this film", shot="a thing on a plain table in daylight",
          claims="[]", reading="literal", vehicle="[]", traps='["a toy version of it"]', extra=""):
    return ('{"n": %d, "meaning": "%s", "reading": "%s", "vehicle": %s, "shot": "%s", "core": "%s",'
            ' "claims": %s, "traps": %s, "queries": %s%s}\n'
            % (n, meaning, reading, vehicle, shot, core, claims, traps, queries, extra))


DAGGER = _shot(1, "a rondel dagger is visible",
               '[{"q": "museum dagger", "for": ["c2"]}, {"q": "rondel dagger closeup", "for": ["core"]},'
               ' {"q": "medieval dagger", "for": ["core", "c2"]}]',
               meaning="here is the rondel dagger", shot="a rondel dagger lying on grey cloth",
               claims='[{"id": "c2", "text": "a plain background", "tier": "should"}]')
ARROW = _shot(2, "an arrow is visible",
              '[{"q": "arrow hitting armor", "for": ["core", "c1"]}, {"q": "archer shooting", "for": ["core"]}]',
              meaning="an arrow glances off the breastplate", shot="an arrow glancing off plate armour",
              claims='[{"id": "c1", "text": "the arrow glances off armour", "tier": "must", "motion": true}]')


def test_plan_roundtrip_attaches_queries_and_spec_by_phrase_text(tmp_path):
    d, blocks = _episode(tmp_path)
    gw = FakeGateway(DAGGER + ARROW)
    assert sqp.plan_episode(d, blocks, gw, model="m", verbose=False) == 2
    assert sqp.attach(blocks, sqp.load(d), sqp.load_specs(d)) == 2
    by_text = {b["text"]: b for b in blocks}
    assert by_text["Вот кинжал."]["phrase_queries"] == ["rondel dagger closeup", "medieval dagger",
                                                        "museum dagger"], "сначала запросы главного"
    arrow = by_text["Стрела скользит по нагруднику."]["shot_spec"]
    assert arrow["focus"] == "an arrow glancing off plate armour", "судье уходит КАДР, а не пересказ фразы"
    assert arrow["meaning"] == "an arrow glances off the breastplate" and arrow["reading"] == "literal"
    assert sqp.has_motion(arrow) and not sqp.has_motion(by_text["Вот кинжал."]["shot_spec"])
    assert "phrase_queries" not in by_text["Итог."] and "shot_spec" not in by_text["Итог."], \
        "фраза без ответа идёт прежним путём"
    readable = open(os.path.join(d, "media_plan", sqp.READABLE_NAME), encoding="utf-8").read()
    assert "кадр: an arrow glancing off plate armour" in readable and "Итог." in readable


def test_second_run_is_served_from_cache(tmp_path):
    d, blocks = _episode(tmp_path)
    gw = FakeGateway(DAGGER)
    sqp.plan_episode(d, blocks, gw, model="m", verbose=False)

    def hook_asks():
        return [p for p in gw.prompts if "1. «Вот кинжал.»" in p and "Plan shots" in p]
    hook = hook_asks()
    final = sum(1 for p in gw.prompts if "1. «Итог.»" in p)
    sqp.plan_episode(d, blocks, gw, model="m", verbose=False)
    assert hook_asks() == hook, "отвеченная глава из кэша; фраза, не давшая годного задания, не перепокупается"
    assert sum(1 for p in gw.prompts if "1. «Итог.»" in p) > final, \
        "сбой шлюза (пустой ответ) не записывается отказом — глава спрашивается снова"


def test_broken_plan_file_is_empty_plan_not_a_crash(tmp_path):
    d, _blocks = _episode(tmp_path)
    open(os.path.join(d, "media_plan", sqp.PLAN_NAME), "w").write("{not json")
    assert sqp.load(d) == {}


def test_slot_takes_its_phrase_query_first_and_keeps_the_section_pool():
    blocks = [{"text": "a", "phrase_queries": ["q1", "q2", "q3"]}, {"text": "b"}]
    assert ps.apply_phrase_queries(blocks, ["sec a", "sec b"]) == ["q1", "sec b"]
    assert ps.slot_extra_queries(blocks[0], ["s1", "q3"]) == ["q2", "q3", "s1"]
    assert ps.slot_extra_queries(blocks[1], ["s1"]) == ["s1"], "без плана — прежний пул секции"


def test_pexels_budget_spares_the_slot_own_queries(monkeypatch):
    monkeypatch.setattr(ps, "PEXELS_QUOTA_LEFT", 10)
    monkeypatch.setattr(ps, "PEXELS_QUOTA_RESERVE", 20)
    monkeypatch.setattr(ps, "PEXELS_LOW_PRIORITY_SKIPPED", 0)
    assert ps.pexels_query_allowed("own", {}, low_priority=False)
    assert ps.pexels_query_allowed("cached", {"cached": []}, low_priority=True), "кэш бесплатен"
    assert not ps.pexels_query_allowed("extra", {}, low_priority=True)
    assert ps.PEXELS_LOW_PRIORITY_SKIPPED == 1
    monkeypatch.setattr(ps, "PEXELS_QUOTA_LEFT", 50)
    assert ps.pexels_query_allowed("extra", {}, low_priority=True)
    monkeypatch.setattr(ps, "PEXELS_QUOTA_LEFT", None)
    assert ps.pexels_query_allowed("extra", {}, low_priority=True), "остаток неизвестен — не режем"


def test_quota_is_read_from_the_response_header():
    class R:
        headers = {"X-Ratelimit-Remaining": "137"}
    ps.PEXELS_QUOTA_LEFT = None
    ps._note_pexels_quota(R())
    assert ps.PEXELS_QUOTA_LEFT == 137
    ps.PEXELS_QUOTA_LEFT = None


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _video_world import QUERY, infra, video  # noqa: E402,F401


def test_video_path_spends_pexels_quota_on_own_query_only(infra, monkeypatch):
    asked = []

    def search(q):
        asked.append(q)
        return [video(1)]
    monkeypatch.setattr(ps, "_pexels_search_videos", search)
    monkeypatch.setattr(ps, "PEXELS_QUOTA_LEFT", 5)
    monkeypatch.setattr(ps, "PEXELS_QUOTA_RESERVE", 50)
    import dataclasses
    import selection_engine
    f = {x.name: None for x in dataclasses.fields(selection_engine.SlotRequest)}
    f.update(index=0, query=QUERY, extra_queries=("reenactment knight fall",), is_opening=False,
             director_assist=False)
    req = ps.build_slot_request(**f)
    ps.VIDEO_ADAPTER.sources(req, QUERY)
    ps.VIDEO_ADAPTER.sources(req, "reenactment knight fall")
    assert len(asked) == 1 and "reenactment" not in asked[0]


def _window(*ns):
    return [{"n": n, "text": f"фраза {n}", "group": ("id", n), "author_brief": None} for n in ns]


def test_spec_line_is_parsed_alone_and_bad_specs_drop():
    core = "a knight kneeling in armour"
    ok_q = '[{"q": "knight", "for": ["core"]}]'
    raw = (
        _shot(1, core, '[{"q": "knight kneeling", "for": ["core", "zz"]}, {"q": "Knight \\"1\\"", "for": ["core"]},'
                       ' {"q": "field grass", "for": ["zz"]}]',
              claims='[{"id": "c1", "text": "grass field", "tier": "should"}, {"id": "c1", "text": "a duplicate id",'
                     ' "tier": "must"}, {"id": "c3", "text": "sky above", "tier": "maybe"}]')
        + 'garbage line\n'
        + _shot(2, core, ok_q, shot="рыцарь на коленях в поле")
        + _shot(3, "", ok_q)
        + _shot(4, core, '[{"q": "green grass", "for": ["c2"]}]',
                claims='[{"id": "c2", "text": "grass field", "tier": "should"}]')
        + _shot(5, core, ok_q, claims='[{"id": "c2", "text": "he moves forward", "tier": "must", "motion": true},'
                                      ' {"id": "c3", "text": "he falls down", "tier": "must", "motion": true}]')
        + _shot(6, core, ok_q, shot="he is kneeling in the grass field")
        + _shot(9, core, ok_q))
    got, why = sqp.parse_window(raw, _window(1, 2, 3, 4, 5, 6, 9))
    assert sorted(got) == [1, 5, 9], "кириллица, нет главного, ни одного запроса главного, кадр-пересказ — выпадают"
    assert "pronoun" in why[6] and "no core" in why[3] and "no query" in why[4]
    assert [c.get("motion", False) for c in got[5]["claims"]] == [False, True, False], \
        "лишний флаг движения снимается, фраза остаётся"
    assert [c["id"] for c in got[1]["claims"]] == ["core", "c1"], "главное первым; повтор id и чужой tier отброшены"
    assert got[1]["claims"][0] == {"id": "core", "text": core, "tier": "must"}
    assert got[1]["queries"] == [{"q": "knight kneeling", "for": ["core"]}], "цель без утверждения — не цель"


def test_metaphor_image_never_reaches_the_shot_the_core_or_the_queries():
    """Живой промах v3: «доспех держит, как капкан» -> фокус «like a trap»
    судье требованием, а запросы «screwdriver against metal door» в сток."""
    q = ('[{"q": "knight fallen mud", "for": ["core"]}, {"q": "steel trap jaws", "for": ["core"]},'
         ' {"q": "armour on ground", "for": ["core"]}]')
    good = _shot(1, "a fallen knight is visible", q, reading="figurative", vehicle='["trap", "jaws"]',
                 shot="a knight lying in mud unable to rise")
    got, _why = sqp.parse_window(good, _window(1))
    assert [x["q"] for x in got[1]["queries"]] == ["knight fallen mud", "armour on ground"], \
        "запрос с образом метафоры выброшен"
    assert "trap" in got[1]["traps"] and got[1]["vehicle"] == ["trap", "jaws"], "образ — ещё и ловушка для судьи"
    bad_shot = _shot(1, "a fallen knight is visible", q, reading="figurative", vehicle='["trap"]',
                     shot="a knight caught in a steel trap in the mud")
    got, why = sqp.parse_window(bad_shot, _window(1))
    assert got == {} and "vehicle" in why[1], "кадр с образом метафоры не проходит — переспрашивается"
    bad_must = _shot(1, "a fallen knight is visible", q, reading="figurative", vehicle='["trap"]',
                     shot="a knight lying in mud",
                     claims='[{"id": "c1", "text": "a trap holds his leg", "tier": "must"}]')
    assert sqp.parse_window(bad_must, _window(1))[0] == {}
    grip = _shot(1, "a hand gripping a dagger point down is visible",
                 '[{"q": "dagger reverse grip", "for": ["core"]}, {"q": "ice axe climbing", "for": ["core"]}]',
                 reading="literal", vehicle='["ice axe"]', shot="a gloved hand holding a dagger point down")
    got, _why = sqp.parse_window(grip, _window(1))
    assert [x["q"] for x in got[1]["queries"]] == ["dagger reverse grip"], \
        "сравнение внутри буквальной фразы — тоже образ: в запросы не идёт (замер 26.09, «как ледоруб»)"
    milk = _shot(1, "a milk carton is visible", '[{"q": "milk carton", "for": ["core"]}]',
                 reading="literal", vehicle="[]", shot="a milk carton on a kitchen table")
    assert sqp.parse_window(milk, _window(1))[0][1]["vehicle"] == [], \
        "сравнение, которое фраза просит представить, — не образ: модель его не называет, кадр его показывает"


def test_model_cannot_smuggle_its_own_core_claim():
    raw = _shot(1, "a ball is visible", '[{"q": "ball", "for": ["core"]}]',
                claims='[{"id": "core", "text": "a wall is visible", "tier": "must"}]')
    got, _ = sqp.parse_window(raw, _window(1))
    assert got[1]["claims"] == [{"id": "core", "text": "a ball is visible", "tier": "must"}]


def test_query_target_given_as_a_string_is_accepted():
    got, _ = sqp.parse_window(_shot(1, "a b", '[{"q": "a b", "for": "core"}]'), _window(1))
    assert got[1]["queries"] == [{"q": "a b", "for": ["core"]}]


def test_unknown_unit_number_is_ignored():
    got, why = sqp.parse_window(_shot(7, "a b", '[{"q": "a b", "for": ["core"]}]'), _window(1))
    assert got == {} and why == {1: "missing from the answer"}


def test_queries_ordered_by_importance_of_what_they_look_for():
    spec = {"claims": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
            "queries": [{"q": "x", "for": ["c"]}, {"q": "y", "for": ["b", "c"]}, {"q": "z", "for": ["a"]}]}
    assert [x["q"] for x in sqp.order_queries(spec)] == ["z", "y", "x"]


def test_old_plan_version_gives_no_specs(tmp_path, capsys):
    d, _blocks = _episode(tmp_path)
    json.dump({"version": 3, "units": {"k": {"text": "t", "queries": ["a b"], "focus": "f",
                                             "claims": [{"id": "c1"}]}}},
              open(os.path.join(d, "media_plan", sqp.PLAN_NAME), "w"))
    assert sqp.load_specs(d) == {}
    assert "версия 3" in capsys.readouterr().out
    assert sqp.load(d) == {"k": ["a b"]}, "запросы старого плана по-прежнему читаются"


def test_prompt_never_tells_the_model_to_drop_the_subject():
    t = sqp.SPEC_PROMPT.lower()
    assert "without the action" not in t and "substitute" not in t
    assert "no words, captions or signs" in t, "текст в кадре запрещён — правило терялось при переписывании"
    assert "never search for the vehicle" in t
    # Замер 26.09 (эп.94, «Но именно он решал исход поединка»): смысл
    # разрешён верно — кинжал, а кадр и главное стали мечом, потому что
    # соседние кадры уже показали кинжал, а правило разнообразия было без
    # оговорки.
    assert "never swap what the line is about" in t
    # Замер 26.09: «будто на ладонь поставили слона» и «глаз размером с
    # тарелку» прочитаны как сравнения, которые фраза просит представить, —
    # в запросы ушли слоник и обеденная тарелка.
    assert "«будто»" in t and "«размером с»" in t
    # То же правило обязано стоять и в вопросе библии: 26.09 библия фильма
    # про глубоководье сама разрешила «слона на ладонях» и «тарелку рядом с
    # глазом кальмара» по прежней оговорке, и планировщик пошёл за ней.
    b = sqp.BIBLE_PROMPT.lower()
    assert "unless the line itself asks the viewer to picture" not in b
    assert "not even as a size comparison" in b


def test_a_composite_made_in_editing_is_not_a_shot():
    """Кадр, собранный монтажом (кольцо поверх съёмки, два снимка рядом в
    одном кадре), в библиотеках не лежит — его не найти, а запрос за ним
    приносит мусор. Такое задание переспрашивается."""
    q = '[{"q": "phone scrolling dark room", "for": ["core"]}]'
    raw = _shot(1, "a thumb scrolling a phone is visible", q,
                shot="a thin ring overlay closing over a dark room where a thumb scrolls a phone")
    got, why = sqp.parse_window(raw, _window(1))
    assert got == {} and "composite" in why[1]
    ok = _shot(1, "a thumb scrolling a phone is visible", q, shot="a thumb scrolling a phone in a dark room")
    assert sqp.parse_window(ok, _window(1))[0][1]["focus"] == "a thumb scrolling a phone in a dark room"


def test_code_or_network_failure_is_not_recorded_as_a_failed_phrase(tmp_path, monkeypatch):
    """Сбой кода или связи — не ответ модели о фразе: фраза не попадает в
    «не удалось» (там она не спрашивалась бы больше никогда при этой
    подписи), следующий рендер спросит её снова."""
    import script_parser
    d = tmp_path / "ep"
    (d / "media_plan").mkdir(parents=True)
    (d / "script.txt").write_text("=== HOOK ===\nВот кинжал.[pause]\nСтрела летит.[pause]\n", encoding="utf-8")
    blocks = script_parser.parse_blocks(str(d / "script.txt"))

    def boom(*a, **k):
        raise RuntimeError("неожиданная ошибка")
    monkeypatch.setattr(sqp, "plan_chapter", boom)
    assert sqp.plan_episode(str(d), blocks, FakeGateway({}), model="m", verbose=False, bible=None) == 0
    plan = json.load(open(d / "media_plan" / sqp.PLAN_NAME, encoding="utf-8"))
    assert plan["failed"] == {}
    assert len(plan["retry"]) == 2 and all(v["why"].startswith("error: RuntimeError")
                                           for v in plan["retry"].values()), "причина сбоя видна в плане"
    assert "error: RuntimeError" in open(d / "media_plan" / sqp.READABLE_NAME, encoding="utf-8").read()
    assert sqp.needs_planning(str(d), blocks, model="m")


def test_bible_reaches_every_chapter_question_including_the_far_away_list():
    """Всё, что библия знает о фильме, доходит до вопроса по главе — в том
    числе «никогда». 26.09 этот список составлялся и сохранялся, но в
    вопрос не попадал: слой без читателя."""
    bible = sqp.parse_bible(json.dumps({
        "topic": "how knights really died", "look": "museum armour and period miniatures",
        "viewer": "a French man-at-arms at Agincourt",
        "cast": [{"name": "the rondel dagger", "refs": ["он"], "show": "a narrow steel dagger"}],
        "reveals": [{"hidden": "the real killer", "is": "the mud of the field"}],
        "never": ["samurai armour", "modern sport fencing"]}))
    block = sqp.bible_block(bible)
    for part in ("how knights really died", "a French man-at-arms at Agincourt", "(called: он)",
                 "the mud of the field", "samurai armour; modern sport fencing"):
        assert part in block, part
    assert sqp.bible_block(None) == ""


def test_broken_bible_answer_is_asked_once_more(tmp_path):
    """Живой случай 26.09: у эпизода про крах 1929 ответ библии не
    разобрался, и эпизод спланирован без неё — все главы без облика фильма
    и сквозных предметов. Сорванный ответ спрашивается ещё раз."""
    d, _blocks = _episode(tmp_path)
    good = json.dumps({"topic": "a film", "look": "real photos"})

    class Gw:
        def __init__(self, answers):
            self.answers, self.calls = list(answers), 0

        def chat(self, *a, **kw):
            self.calls += 1
            return self.answers.pop(0), {"finish_reason": "stop"}, 1
    gw = Gw(['{"topic": "a film", "look": "real "photos"}', good])
    bible, what = sqp.make_bible(d, gw, None, model="m")
    assert what == "made" and bible["topic"] == "a film" and gw.calls == 2
    gw = Gw(["not json", "still not json"])
    shutil.rmtree(os.path.join(d, "media_plan"))
    bible, what = sqp.make_bible(d, gw, None, model="m")
    assert bible is None and what == "failed: ответ не разобран" and gw.calls == sqp.BIBLE_ATTEMPTS


def test_empty_paid_answer_is_asked_again_in_smaller_windows(tmp_path):
    """Пустой оплаченный ответ — сорванное окно, а не лежащий шлюз: окно
    переспрашивается в этом же прогоне окнами вдвое меньше. Замер 26.09:
    9 пустых ответов на 22 вызова, 45 фраз эп.02 из 142 остались без
    задания, потому что такие окна не переспрашивались вовсе."""
    import llm_gateway
    d, blocks = _episode(tmp_path)

    class FlakyOnce(FakeGateway):
        def __init__(self, answer):
            super().__init__(answer)
            self.empty_left = 1

        def chat(self, model, content, max_tokens, est, **kw):
            text = content[0]["text"]
            if "1. «Вот кинжал.»" in text and "Plan shots" in text and self.empty_left:
                self.empty_left -= 1
                self.prompts.append(text)
                raise llm_gateway.EmptyAnswer("пустой ответ (finish_reason=length)")
            return super().chat(model, content, max_tokens, est, **kw)
    gw = FlakyOnce(DAGGER + ARROW)
    assert sqp.plan_episode(d, blocks, gw, model="m", verbose=False) == 2
    plan = json.load(open(os.path.join(d, "media_plan", sqp.PLAN_NAME), encoding="utf-8"))
    assert gw.empty_left == 0 and len(plan["units"]) == 2, "окно хука, сорвавшееся пустым ответом, переспрошено"
    assert plan["failed"] == {} and list(plan["retry"].values())[0]["text"] == "Итог.", \
        "глава, где пустой ответ повторялся, ждёт следующего рендера, а не записана отказом"
    # Сводка объясняет переспросы: пустые ответы — по причине конца ответа,
    # отказы — по виду (без неё 47 переспросов на 142 фразы нечем объяснить).
    assert plan["stats"]["empty_finish"].get("length") == 1


def test_plan_stats_count_rejections_by_kind(tmp_path):
    d, blocks = _episode(tmp_path)
    pronoun = ARROW.replace('"shot": "an arrow glancing off plate armour"', '"shot": "it glances off plate armour"')
    gw = FakeGateway(DAGGER + pronoun)
    sqp.plan_episode(d, blocks, gw, model="m", verbose=False)
    plan = json.load(open(os.path.join(d, "media_plan", sqp.PLAN_NAME), encoding="utf-8"))
    assert plan["stats"]["rejected"]["pronoun"] >= 1


def test_reason_kinds_cover_every_rejection_the_parser_writes():
    assert sqp.reason_kind("missing from the answer") == "missing"
    assert sqp.reason_kind("answer cut by the output limit") == "cut"
    assert sqp.reason_kind("the core must show what the line is about — «x»") == "about"
    assert sqp.reason_kind("the shot starts with a pronoun — name what is visible instead") == "pronoun"
    assert sqp.reason_kind("the shot is a composite made in editing («overlay»)") == "composite"
    assert sqp.reason_kind("the core shows the vehicle «trap»") == "vehicle"
    assert sqp.reason_kind("a must claim asks for the vehicle «trap»") == "vehicle"
    assert sqp.reason_kind("no query searches for the core (queries with the vehicle are dropped)") == \
        "no_core_query"
    assert sqp.reason_kind("no shot of 4-32 English words") == "form"


def test_persistent_empty_answer_is_retried_next_render_not_marked_failed(tmp_path):
    import llm_gateway
    d, blocks = _episode(tmp_path)

    class AlwaysEmpty(FakeGateway):
        def chat(self, model, content, max_tokens, est, **kw):
            self.prompts.append(content[0]["text"])
            raise llm_gateway.EmptyAnswer("пустой ответ")
    sqp.plan_episode(d, blocks, AlwaysEmpty(""), model="m", verbose=False, bible=None)
    plan = json.load(open(os.path.join(d, "media_plan", sqp.PLAN_NAME), encoding="utf-8"))
    assert plan["failed"] == {} and plan["retry"], "пустой ответ не приговор фразе"
    assert sqp.needs_planning(d, blocks, model="m")


def test_head_noun_of_an_english_noun_phrase():
    assert sqp.head_noun("the rondel dagger") == "dagger"
    assert sqp.head_noun("a pile of coins") == "pile"
    assert sqp.head_noun("the arrows") == "arrows" and "arrow" in sqp.noun_forms("arrows")
    assert sqp.head_noun("") == "" and sqp.noun_forms("") == []


def test_the_core_shows_what_the_line_is_about():
    """Живой промах 26.09 (эп.94): «Но именно он решал исход поединка, когда
    меч уже бесполезен» — смысл «кинжал, а не меч», а главное — меч в грязи.
    Модель называет, о чём фраза; главное без этого предмета не проходит."""
    q = '[{"q": "dagger close combat", "for": ["core"]}]'
    wrong = _shot(1, "a sword lying useless in mud is visible", q, shot="a sword lying useless in battlefield mud",
                  extra=', "about": "the rondel dagger"')
    got, why = sqp.parse_window(wrong, _window(1))
    assert got == {} and "the rondel dagger" in why[1]
    got, _ = sqp.parse_window(wrong, _window(1), reasked=True)
    assert got[1]["about_off"] is True, \
        "сверка по словам не знает синонимов: ответ переспроса принимается, фраза не теряется"
    compound = _shot(1, "a gauntleted hand gripping a longsword blade is visible", q,
                     shot="a gauntleted hand gripping a longsword blade halfway", extra=', "about": "the sword"')
    assert "about_off" not in sqp.parse_window(compound, _window(1))[0][1], "longsword содержит sword"
    right = _shot(1, "a rondel dagger thrust at close quarters is visible", q,
                  shot="a rondel dagger thrust between two grappling armoured men", extra=', "about": "the rondel dagger"')
    got, _ = sqp.parse_window(right, _window(1))
    assert got[1]["about"] == "the rondel dagger"
    plural = _shot(1, "an arrow striking a packed line of men is visible", q,
                   shot="arrows falling on a packed line of armoured men", extra=', "about": "the arrows"')
    assert sqp.parse_window(plural, _window(1))[0], "the arrows ~ an arrow"
    idea = _shot(1, "a person turning away from a laptop is visible", q,
                 shot="a person turning away from a glowing laptop", extra=', "about": ""')
    assert sqp.parse_window(idea, _window(1))[0], "идея без предмета — проверки нет"
    # «Ты открыл ноутбук...» — о тебе, а не о ноутбуке: «you» словом в
    # главном не найти, сверять нечего (первая версия переспрашивала такие
    # фразы каждый раз).
    you = _shot(1, "a young adult sitting before an open laptop is visible", q,
                shot="a young adult sitting still before an open laptop with an empty document",
                extra=', "about": "you"')
    got, _ = sqp.parse_window(you, _window(1))
    assert got and "about_off" not in got[1]
    assert sqp.shares_a_word("an exhausted person", "you") is True
    assert sqp.shares_a_word("an exhausted person", "you at your desk") is False, "desk — значимое слово"
    # Имя собственное не сверяется: человека по лицу не узнать, и сверка не
    # должна толкать имя в главное (замер 26.09: 4 прогона из 4 с именем).
    assert sqp.shares_a_word("a bearded craftsman in a leather apron", "Johann Gutenberg") is True
    assert sqp.shares_a_word("a woman in a laboratory coat", "Marie Curie") is True
    assert sqp.shares_a_word("a dark print workshop", "Gutenberg's dark workshop") is True
    assert sqp.shares_a_word("a white blood cell", "Macrophages") is False, "первое слово фразы — не имя"


def test_about_is_the_subject_not_the_event_or_the_other_noun():
    """Замер 26.09: «Но именно он решал исход поединка, когда меч уже
    бесполезен» — «он» это кинжал, а в восьми прогонах главным был то меч
    (второе существительное фразы), то «поединок» (событие), и только в двух
    — кинжал. Правило в вопросе: о чём фраза — обычно её подлежащее с
    раскрытыми местоимениями; событие никогда не «о чём», о чём — тот, кто
    действует. Пример — не из ниши канала."""
    t = sqp.SPEC_PROMPT
    assert "usually the subject of its sentence" in t
    assert "An event or an action is never the about: the one who acts is." in t
    assert 'not "the expedition", not "the map"' in t
    # Человек описывается тем, что видно: с правилом «о чём — подлежащее»
    # главным вставало имя («Johann Gutenberg ... is visible», 4 из 4), а
    # судья по лицу не узнаёт никого.
    assert "nobody can tell who a person is by looking" in t
    # Замер судьи 26.09 (эп.94): главное подвидом («a rondel dagger») отклоняло
    # годные кинжалы другого вида — сетка ставила им 2, проверка «нет»; главное
    # с действием («a real arrow glancing off curved steel») отклоняло стрелы
    # без этого действия. Главное — общее название с эпохой, подвид и действие —
    # утверждения. Тот же вывод дал замер 25.09 (shot_judge.nothing_met).
    assert "EVERY correct picture of it satisfies" in t
    assert "A subtype, a feature, a material and anything it does are claims" in t
    # Подвид в главное приносила библия: её сквозной предмет назывался точным
    # типом («the rondel dagger»), хотя в тексте — просто «кинжал».
    assert "the narration's own word for it, even when the film bible knows its exact type" in t


def test_queries_name_the_thing_and_the_kind_of_picture_not_the_framing():
    """Живой пул бесплатной зоны (26.09, 26 фраз четырёх ниш, руки в одном
    процессе, разметка вслепую): у задания, чьи запросы описывали композицию
    кадра («hand holding dagger two fingers», «dagger blade between armour
    plates»), сумма меток победителей 17 против 27 у прежнего задания с
    запросами «предмет + вид снимка» («rondel dagger museum», «lymphocyte cell
    microscope»). Поиск находит по названию снимка, композицию проверяет судья."""
    t = sqp.SPEC_PROMPT
    assert "not how this shot is framed" in t
    assert "checked by the judge on the picture, not searched" in t
    assert "so it is the plainest: the core and the kind of picture" in t
    assert "not the same words with an extra word" in t


def test_the_head_of_a_multi_word_vehicle_is_banned_in_queries():
    q = ('[{"q": "arrow rain battle", "for": ["core"]}, {"q": "arrows packed formation", "for": ["core"]}]')
    raw = _shot(1, "an arrow striking a packed line of men is visible", q, reading="figurative",
                vehicle='["iron rain"]', shot="arrows falling on a packed line of armoured men")
    got, _ = sqp.parse_window(raw, _window(1))
    assert [x["q"] for x in got[1]["queries"]] == ["arrows packed formation"]


def test_at_most_two_must_claims_besides_the_core():
    """Каждое лишнее «must» (руки, грязь, фон) отдаёт выбор судьи кадру с
    обстановкой, но без предмета фразы (замер 26.09 на снимке эп.94)."""
    claims = ('[{"id": "c1", "text": "a longsword lies in the mud", "tier": "must"},'
              ' {"id": "c2", "text": "the blade is bare", "tier": "must"},'
              ' {"id": "c3", "text": "the ground is churned mud", "tier": "must"}]')
    raw = _shot(1, "a dagger is visible", '[{"q": "dagger", "for": ["core"]}]', claims=claims)
    got, _ = sqp.parse_window(raw, _window(1))
    assert [c["tier"] for c in got[1]["claims"]] == ["must", "must", "must", "should"]
    assert "a plain photo or museum object of that thing must satisfy the core" in sqp.SPEC_PROMPT.lower()
