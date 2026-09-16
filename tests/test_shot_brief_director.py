# -*- coding: utf-8 -*-
"""Режиссёрская разработка главы и ключ плана.

Каждый тест здесь проверен КОНТРОЛЬНЫМ ПРОГОНОМ со снятой правкой: без
неё он падает. Тест, который зелен и с правкой, и без неё, не защищает
ничего — этот урок в репозитории уже оплачен трижды.
"""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))


def test_plan_key_survives_missing_model_env(monkeypatch):
    """План, посчитанный с моделью в окружении, ОБЯЗАН находиться при
    рендере без неё.

    Реальный отказ, найденный прогоном: планируют один раз с
    LLAMA_MODEL_GGUF, собирают ролик потом и без него. Ключ включал имя
    файла модели -> при рендере не совпадал -> fill_briefs() проставлял
    НОЛЬ брифов молча, при включённом флаге и готовом плане на диске.
    """
    import importlib
    monkeypatch.setenv("LLAMA_MODEL_GGUF", "/models/qwen-x.gguf")
    import shot_planner_llm as p
    importlib.reload(p)
    phrase = "Тебе нужно всего лишь встать."
    key_when_planned = p.unit_key(phrase)

    monkeypatch.delenv("LLAMA_MODEL_GGUF")
    importlib.reload(p)
    assert p.unit_key(phrase) == key_when_planned

    blocks = [{"text": phrase, "shot_brief": None}]
    plan = {key_when_planned: {"shot_en": "a steel sabaton on bare earth",
                               "function": "object"}}
    assert p.fill_briefs(blocks, plan) == 1
    assert blocks[0]["shot_brief"] == "a steel sabaton on bare earth"


def test_cache_key_still_separates_models(monkeypatch):
    """А вот КЭШ ОТВЕТА обязан зависеть от модели и версии промпта:
    иначе новая модель молча отдавала бы ответ старой."""
    import importlib
    import shot_planner_llm as p
    phrase = "Тебе нужно всего лишь встать."
    monkeypatch.setenv("LLAMA_MODEL_GGUF", "/models/a.gguf")
    importlib.reload(p)
    a = p.cache_key(phrase)
    monkeypatch.setenv("LLAMA_MODEL_GGUF", "/models/b.gguf")
    importlib.reload(p)
    assert p.cache_key(phrase) != a


def _packet(units, section="BLOCK 4: ОТВЕТ"):
    return {"section": section, "section_query": "", "episode_title": "",
            "niche": "", "prev_tail": "",
            "units": [{"n": i + 1, "block_index": i, "text": t,
                       "arc_stage": None, "stat": None, "is_climax": False,
                       "author_brief": None} for i, t in enumerate(units)]}


def test_chapter_prompt_carries_neighbours():
    """Главная причина существования модуля: фраза с местоимением видна
    режиссёру ВМЕСТЕ с той, которая местоимение раскрывает."""
    import shot_brief_director as d
    p = d.render_prompt(_packet(["Рыцарей убивала земля.",
                                 "При этом он был под ногами у каждого."]))
    assert "Рыцарей убивала земля." in p
    assert "он был под ногами" in p
    assert p.index("Рыцарей убивала") < p.index("он был под ногами")


def test_broken_row_loses_only_itself():
    """Построчный разбор против одного JSON на главу: сорванная строка
    не должна уносить с собой всю главу."""
    import shot_brief_director as d
    pkt = _packet(["а", "б", "в"])
    got = d.parse_answer(
        "1 | object | a dented steel breastplate, close up\n"
        "2 | ??? совершенно сломанная строка\n"
        "3 | scene | a churned muddy field under grey sky\n", pkt)
    assert set(got) == {1, 3}
    assert got[1]["shot_en"] == "a dented steel breastplate, close up"
    assert got[3]["function"] == "scene"


@pytest.mark.parametrize("row", ["4 | object | a sword", "0 | object | a sword"])
def test_answer_about_a_phrase_that_is_not_there_is_dropped(row):
    """Модель назвала номер, которого в главе нет. Принять его значило бы
    поставить кадр под чужую фразу — ровно то, от чего инлайновый
    `[shot:]` защищает автора."""
    import shot_brief_director as d
    assert d.parse_answer(row, _packet(["а", "б", "в"])) == {}


def test_empty_shot_is_not_a_brief():
    """«Показывать нечего» — законный ответ, и он обязан оставить юнит на
    прежнем пути, а не проставить прочерк как описание кадра."""
    import shot_brief_director as d
    got = d.parse_answer("1 | - | -\n2 | object | a rondel dagger blade\n",
                         _packet(["а", "б"]))
    assert set(got) == {2}


def test_validation_is_the_same_one_as_per_phrase_mode():
    """Второго свода правил не заводится: заявка, отклонённая пофразовым
    режиссёром, обязана отклоняться и главным."""
    import shot_brief_director as d
    import shot_planner_llm as p
    bad = "a man in full protective gear"
    assert p.brief_is_safe(bad, "фраза")[0] is False
    assert d.shot_planner_llm.brief_is_safe is p.brief_is_safe


# --- ПРОВЕРКА БРИФА: ложные отказы, найденные на брифах автора ---------------
#
# Валидатор устроен так, что отказ ничего не стоит: слот идёт прежним
# путём. Это верно ровно до тех пор, пока отказы не бьют по ХОРОШИМ
# брифам систематически. Замер на 142 брифах автора эпизода 02 нашёл
# четыре ложных отказа, три из них — весь блок про братскую могилу.

@pytest.mark.parametrize("shot", [
    "a human skull from an archaeological excavation",
    "human bones laid out from an excavation",
    "a human skull with wounds on it",
    "an English archer's simple clothing and equipment",
])
def test_author_briefs_are_not_rejected(shot):
    import shot_planner_llm as p
    ok, why = p.brief_is_safe(shot, "фраза сценария", blocklist=())
    assert ok, why


@pytest.mark.parametrize("shot,part", [
    # Причина переименована 15.09 вместе с переездом словаря в профиль:
    # правило теперь про МИР КАНАЛА, а не про эпоху — эпоха была частным
    # случаем одной ниши.
    ("A man stepping onto a battlefield", "мир"),
    ("A person standing up", "мир"),
    ("A warrior in full protective gear", "снаряжение"),
    ("He was at their feet", "пересказ"),
])
def test_measured_misses_are_still_rejected(shot, part):
    """Послабление не должно открыть дорогу тем промахам, ради которых
    проверка написана. Все четыре — из реального замера планировщика."""
    import shot_planner_llm as p
    ok, why = p.brief_is_safe(shot, "фраза сценария", blocklist=())
    assert not ok and part in why


# --- ПАСПОРТ КУЛЬТУРЫ: третья дыра того же класса -------------------------
#
# Список чужих культур дополнялся 14.09 (Afghan/Javanese/Mongol/Iraq) и
# 16.09 (доколумбова Америка). Оба раза причина одна: НАМЕРЕНИЕ списка
# выполнялось не на тех именах, которыми музей на самом деле подписывает
# предметы. Здесь то же самое для индонезийского архипелага: java и
# javanese в списке были, а Бали, Суматра, Мадура и Ачех — нет.

@pytest.mark.parametrize("culture", [
    "Balinese", "Sumatran", "Sumatran, Acheen", "Madurese", "Acheen",
    "Philippines",
])
def test_indonesian_and_philippine_cultures_are_foreign(culture):
    import museum_sources as ms
    assert ms.culture_is_foreign(culture, None, None, None)


@pytest.mark.parametrize("culture", [
    "Italian, Venice", "French", "Western European", "Swiss",
    "Flemish, possibly Antwerp", "Spanish, possibly Granada",
])
def test_european_cultures_survive_the_addition(culture):
    """Ужесточение не имеет права начать есть подлинники ниши."""
    import museum_sources as ms
    assert not ms.culture_is_foreign(culture, None, None, None)


@pytest.mark.parametrize("term", ["bali", "moro"])
def test_known_traps_stay_out_of_the_list(term):
    """Проверено негативным контролем на реальном корпусе: «bali»
    совпадает внутри «Kabbalism» на европейской гравюре, «moro» — внутри
    «Morose», «Nemorosus», «Amorosi». Тот же класс, что «зал» внутри
    «ЗАЛП» в словаре атмосферы."""
    import museum_sources as ms
    assert term not in ms.DEFAULT_FOREIGN_CULTURE_TERMS


def test_arc_stage_really_reaches_the_prompt(tmp_path):
    """Драматургическая стадия обязана ДОЕХАТЬ до режиссёра, а не просто
    быть прочитанной.

    Ветка читает speech_plan.json по СХЕМЕ speech_planner (plan["units"],
    у юнита "text" и "arc_stage"). Разойдись схема — ветка стала бы тихим
    no-op: стадии нет ни у одного юнита, промпт прежний, и снаружи это
    неотличимо от «эпизод без speech_plan». Ровно тот класс, который
    закрывает tests/test_no_dead_layers.py на уровне вызовов.
    """
    import json
    import shot_brief_director as d

    mp = tmp_path / "media_plan"
    mp.mkdir()
    (mp / "speech_plan.json").write_text(json.dumps({"units": [
        {"unit_id": 1, "section": "BLOCK 4", "text": "Рыцарей убивала земля.",
         "arc_stage": "слом"}]}), encoding="utf-8")

    stages = d.arc_stages(str(tmp_path))
    assert stages == {"Рыцарей убивала земля.": "слом"}

    packet = _packet(["Рыцарей убивала земля."])
    packet["units"][0]["arc_stage"] = stages["Рыцарей убивала земля."]
    assert "стадия: слом" in d.render_prompt(packet)


def test_museum_vocabulary_is_optional_and_silent_without_index():
    """Заземление на словарь музея — additive: индекса нет, промпт
    возвращается к прежнему виду байт-в-байт."""
    import shot_brief_director as d
    plain = d.render_prompt(_packet(["а", "б"]))
    off = d.render_prompt(dict(_packet(["а", "б"]), use_vocabulary=False))
    assert plain == off
    assert "СЛОВАРЬ МУЗЕЯ" not in plain


def test_channel_blocklist_reaches_the_prompt_from_one_source():
    """Мозгу называют слова, которые проверка всё равно отклонит.

    Найдено замером, а не рассуждением: бриф «a page from a medieval
    fencing manual» отклонялся блоклистом канала, где `fencing` стоит
    против СОВРЕМЕННОГО спортивного фехтования (45 из 655 живых
    кандидатов Pexels). А глава 2 эпизода 02 буквально про Тальхоффера и
    Фиоре — то есть канал сам просит фехтбух. Ослаблять гвард по двум
    случаям нельзя; правильный ход — сказать слова заранее, чтобы мозг
    выбрал другую формулировку, а не потерял слот.

    Список берётся из ЕДИНСТВЕННОГО источника (channel_profile.json через
    pipeline_smart.CONTENT_ALT_BLOCKLIST). Второй копии здесь нет — иначе
    промпт запрещал бы одно, а проверка отклоняла другое.
    """
    import shot_brief_director as d
    import shot_planner_llm as p
    banned = p.channel_blocklist()
    if not banned:
        pytest.skip("блоклист канала недоступен в этом окружении")
    prompt = d.render_prompt(_packet(["Открой любой фехтбух."]))
    assert "ЭТИХ СЛОВ" in prompt
    for term in list(banned)[:5]:
        assert term in prompt


def test_inline_anchor_survives_a_tag_inside_the_phrase():
    """Фраза с внутренним тегом в сыром файле буквально не встречается.

    Парсер СКЛЕИВАЕТ куски вокруг `[stat:...]`, поэтому поиск полного
    текста давал «встречается 0 раз». Живой прогон: три юнита из 107
    пропускались молча по этой причине. Ищется самый длинный префикс,
    встречающийся ровно один раз.
    """
    import shot_brief_director as d
    body = ("=== BLOCK 10 ===\n"
            "Генрих Пятый умер в тысяча четыреста двадцать втором году в "
            "Венсене.[stat:ГЕНРИХ V, 1422] Не в бою.[pause]")
    text = ("Генрих Пятый умер в тысяча четыреста двадцать втором году в "
            "Венсене. Не в бою.")
    assert body.count(text) == 0          # именно из-за этого и ломалось
    at = d._unique_anchor(body, text)
    assert at is not None and body[at:].startswith("Генрих Пятый умер")


def test_inline_anchor_refuses_when_the_place_is_ambiguous():
    """Две одинаковые фразы — тег поставить некуда, и угадывать нельзя:
    он молча описал бы чужой слот."""
    import shot_brief_director as d
    text = "Он не висит в музее под стеклом и не имеет клейма мастера."
    assert d._unique_anchor(text + " " + text, text) is None


def test_inline_never_overwrites_the_author(tmp_path):
    """Бриф автора сильнее заявки модели — правило всего модуля."""
    import shot_brief_director as d
    script = tmp_path / "script.txt"
    script.write_text("[shot:a rondel dagger, the whole dagger]"
                      "Рондельный кинжал появляется здесь впервые.",
                      encoding="utf-8")
    blocks = [{"text": "Рондельный кинжал появляется здесь впервые.",
               "shot_brief": "a rondel dagger, the whole dagger"}]
    placed, skipped = d.write_inline(str(tmp_path), blocks,
                                     {0: {"shot_en": "something else"}},
                                     dry_run=True)
    assert placed == 0 and skipped[0][1] == "у автора уже есть бриф"


# --- УТЕЧКА ПРИМЕРА ИЗ ПРОМПТА ---------------------------------------------
#
# Пример в промпте стоит на ЧУЖОЙ теме (Рим, дороги) намеренно: пример из
# этого же эпизода подсказал бы готовые ответы ровно на тех фразах, на
# которых модель потом меряют. Побочный эффект измерен на живых прогонах:
# 1 заявка из ~80 у 7B копирует пример дословно.

@pytest.mark.parametrize("shot", [
    "a battlefield with a straight roman stone road in the background",
    "a roman stone road in a hilly landscape, 1461 AD",
    "A medieval stone road stretching across a landscape",
])
def test_example_leak_is_caught(shot):
    """Все три — реальные заявки Qwen2.5-7B на фразы про вес доспеха, про
    хроники и про кузницу. Вторая ушла бы в сток как есть и принесла бы
    римскую дорогу в эпизод про Войну Роз; ни один существующий гейт её
    не ловит — слова эпохи там формально нет."""
    import shot_brief_director as d
    assert d.copies_the_example(shot)


@pytest.mark.parametrize("shot", [
    "a dented steel breastplate, close up",
    "a manuscript illumination of armoured men advancing on foot",
    "a rondel dagger with both round discs, the whole weapon",
    "an archaeological excavation of a mass grave with human bones",
    "a steel gorget and bevor covering the throat and the neck",
    "a straight european sword with a plain steel blade, the whole sword",
])
def test_example_guard_touches_nothing_real(shot):
    """Негативный контроль, без которого проверка ничего не стоит: на 142
    брифах автора, 107 брифах Claude и 111 заявках пофразовой 7B гвард
    ловит НОЛЬ."""
    import shot_brief_director as d
    assert not d.copies_the_example(shot)


def test_guard_is_about_copying_not_about_rome():
    """Проверка обязана пережить замену примера: она про совпадение с
    ТЕКУЩИМ FEWSHOT, а не про список слов про Рим."""
    import shot_brief_director as d
    assert d._FEWSHOT_WORDS, "пример разобран пустым — гвард стал no-op"
    for words in d._FEWSHOT_WORDS:
        assert d.copies_the_example(" ".join(sorted(words)))


# --- РАБОТА В ЛЮБОЙ НИШЕ ----------------------------------------------------
#
# ЧАСТЬ 24 CLAUDE.md прямо требует, чтобы клон репозитория не тащил
# творческие характеристики старой ниши. Замер 15.09 нашёл два места, где
# он их тащил, и оба молча.

PSYCH_BRIEFS = [
    "a man sitting alone in an empty waiting room",
    "a person's hands clenched on a kitchen table",
    "a phone lying face down on a bedside table at night",
    "a woman looking out of a rain-streaked window",
    "an empty chair opposite a made bed",
    "a human figure small against a huge office corridor",
]
MEDIEVAL_WORDS = ("medieval", "knight", "warrior", "armour", "armor",
                  "sword", "helmet", "castle", "manuscript")


@pytest.mark.parametrize("shot", PSYCH_BRIEFS)
def test_other_niche_briefs_pass_when_channel_declares_no_world(shot, monkeypatch):
    """Канал без объявленного мира кадра не должен получать чужой.

    Со средневековым словарём, зашитым в код до 15.09, три из этих шести
    законных психологических брифов отклонялись правилом «человек без
    привязки к эпохе» — то есть система запрещала показывать человека
    каналу, у которого человек и есть предмет разговора.
    """
    import pipeline_smart as ps
    import shot_planner_llm as p
    monkeypatch.setattr(ps, "CHANNEL_PROFILE", {}, raising=False)
    ok, why = p.brief_is_safe(shot, "фраза", blocklist=())
    assert ok, why


def test_the_old_hardcoded_list_really_did_reject_them():
    """Негативный контроль: без него предыдущий тест зелен по построению
    и ничего не доказывает."""
    import shot_planner_llm as p
    rejected = [s for s in PSYCH_BRIEFS
                if not p.brief_is_safe(s, "ф", blocklist=(),
                                       era_words=MEDIEVAL_WORDS)[0]]
    assert len(rejected) == 3, rejected


def test_this_channel_keeps_its_world(monkeypatch):
    """А исторический канал обязан остаться строгим: правило не отменено,
    оно переехало в channel_profile.json."""
    import shot_planner_llm as p
    assert p.domain_anchor_words(), "канал объявил мир кадра — список не пуст"
    assert not p.brief_is_safe("A man stepping onto a battlefield", "ф",
                               blocklist=())[0]


def test_prompt_has_no_hardcoded_niche():
    """В промпте не должно быть ниши, которой канал не объявлял."""
    import shot_brief_director as d
    pkt = _packet(["Ему стало нечем дышать."])
    prompt = d.render_prompt(pkt)
    assert "историческ" not in prompt.lower()
    assert "СИТУАЦИЕЙ" in prompt      # заземление абстракции
    assert "MOOD |" in prompt         # настроение


def test_stock_query_gets_no_era_anchor_without_a_profile(monkeypatch):
    """Клон под психологию не должен просить у стока «medieval phone».

    Реальный замер до правки: бриф «a phone lying face down on a bedside
    table at night» превращался в запрос `medieval phone lying face down`
    — и так С КАЖДЫМ запросом, потому что якорь подставлялся из
    ЗАШИТОГО В КОД средневекового списка.
    """
    import pipeline_smart as ps
    monkeypatch.setattr(ps, "OPENVERSE_ERA_ANCHORS", ())
    monkeypatch.setattr(ps, "OPENVERSE_DOMAIN_NOUNS", ())
    q = ps.brief_to_stock_query("a phone lying face down on a bedside table")
    assert "medieval" not in q and "phone" in q


def test_code_default_carries_no_niche():
    """Сам дефолт в коде обязан быть пустым: иначе клон получает нишу
    репозитория молча, даже не зная, что она где-то объявлена."""
    import pipeline_smart as ps
    assert ps._OPENVERSE_ERA_ANCHORS_DEFAULT == ()
    assert ps._OPENVERSE_DOMAIN_NOUNS_DEFAULT == ()


@pytest.mark.parametrize("line,tone,tension", [
    ("MOOD | -2 | 3 | безысходность", -2.0, 3.0),
    ("MOOD | 1 | 0 | спокойно", 1.0, 0.0),
    ("MOOD | -5 | 9 | край", -2.0, 3.0),          # зажим в диапазон
])
def test_mood_is_parsed_and_clamped(line, tone, tension):
    import shot_brief_director as d
    got = d.parse_mood(line + "\n1 | object | a sword")
    assert got["tone"] == tone and got["tension"] == tension


def test_mood_is_optional():
    """Строки настроения нет — глава считается как раньше, ни один кадр
    из-за этого не теряется."""
    import shot_brief_director as d
    assert d.parse_mood("1 | object | a sword") is None
    assert d.parse_answer("1 | object | a rondel dagger blade",
                          _packet(["а"])) != {}


# --- СКВОЗНАЯ ПРОВЕРКА НА ЧУЖОЙ НИШЕ ----------------------------------------
#
# Фикстура — настоящий сценарий из другой ниши (психология избегания), не
# синтетика: она нужна именно потому, что абстракций в ней много, а
# предметов мало — то есть ровно тот случай, на котором исторические
# правила и словари ломаются.

FIXTURE = os.path.join(REPO, "tests", "fixtures", "other_niche")


def _clean_channel(monkeypatch):
    """Свежий канал: ни мира кадра, ни блоклиста, ни якорей эпохи."""
    import pipeline_smart as ps
    monkeypatch.setattr(ps, "CHANNEL_PROFILE", {}, raising=False)
    monkeypatch.setattr(ps, "CONTENT_ALT_BLOCKLIST", (), raising=False)
    monkeypatch.setattr(ps, "OPENVERSE_ERA_ANCHORS", ())
    monkeypatch.setattr(ps, "OPENVERSE_DOMAIN_NOUNS", ())


def test_other_niche_prompt_carries_nothing_medieval(monkeypatch, tmp_path):
    import shutil
    import script_parser
    import shot_brief_director as d
    _clean_channel(monkeypatch)
    shutil.copy(os.path.join(FIXTURE, "script_psychology.txt"),
                tmp_path / "script.txt")
    blocks = script_parser.parse_blocks(str(tmp_path / "script.txt"))
    assert len(blocks) >= 15
    for packet in d.packets(str(tmp_path), blocks):
        prompt = d.render_prompt(packet).lower()
        for w in ("medieval", "knight", "armour", "рыцар", "доспех", "эпоха"):
            assert w not in prompt, f"{w!r} просочилось в чужую нишу"
        assert "ситуацией" in prompt      # заземление абстракции на месте
        assert "mood |" in prompt


def test_other_niche_runs_end_to_end(monkeypatch, tmp_path):
    """Заявки принимаются, настроение разбирается, теги встают в сценарий."""
    import shutil
    import script_parser
    import shot_brief_director as d
    _clean_channel(monkeypatch)
    monkeypatch.setattr(d, "MOODS", {})
    shutil.copy(os.path.join(FIXTURE, "script_psychology.txt"),
                tmp_path / "script.txt")
    blocks = script_parser.parse_blocks(str(tmp_path / "script.txt"))
    found = d.run(str(tmp_path), blocks,
                  d.FileBrain(os.path.join(FIXTURE, "answers")),
                  cache_dir=None, verbose=False)
    assert len(found) >= 14, f"принято всего {len(found)}"
    assert len(d.MOODS) == 2, d.MOODS
    assert d.MOODS["HOOK"]["tone"] == -1.0
    placed, skipped = d.write_inline(str(tmp_path), blocks, found)
    assert placed == len(found) and not skipped
    again = script_parser.parse_blocks(str(tmp_path / "script.txt"))
    assert sum(1 for x in again if (x.get("shot_brief") or "").strip()) == placed


def test_other_niche_stock_queries_stay_clean(monkeypatch):
    """Ни один запрос чужой ниши не должен уехать в сток со средневековым
    якорем. Реальный замер до правки: `medieval phone lying face down`."""
    import pipeline_smart as ps
    _clean_channel(monkeypatch)
    briefs = ["an open laptop with a blank document on a dark desk",
              "a mug of cold coffee with a skin on the surface, close up",
              "a thumb scrolling a phone feed, close up"]
    for b in briefs:
        q = ps.brief_to_stock_query(b)
        assert "medieval" not in q and "knight" not in q, q
        assert q.split()[0] in b.lower()


# --- СЛЕПОЙ ЛИСТ РАЗМЕТКИ ---------------------------------------------------
#
# Единственный способ узнать, чей бриф лучше, — глаза владельца. Всё
# остальное в этом замере считано против эталона, который написал Claude
# в прошлой сессии, то есть рука «Claude» отчасти мерит саму себя.

def test_marking_sheet_letters_are_deterministic():
    """Один и тот же лист обязан собираться одинаково: иначе разметку
    нельзя расшифровать ключом, снятым при прошлом запуске."""
    import brief_marking_sheet as ms
    a = ms._order("Рыцарей убивала земля.", 5)
    b = ms._order("Рыцарей убивала земля.", 5)
    assert a == b and sorted(a) == list(range(5))


def test_marking_sheet_has_no_positional_bias():
    """Порядок букв обязан быть СВОЙ у каждой фразы.

    Фиксированный порядок означал бы, что «А» — всегда одна и та же
    система, и привычка руки заменила бы суждение. Проверяется на
    реальных фразах эпизода, а не на выдуманных строках.
    """
    import collections
    import brief_marking_sheet as ms
    import script_parser
    blocks = script_parser.parse_blocks(
        os.path.join(REPO, "videos", "02_ne-mechom", "script.txt"))
    first = collections.Counter()
    for b in blocks:
        text = (b.get("text") or "").strip()
        if text:
            first[ms._order(text, 5)[0]] += 1
    # ни одна система не должна стоять первой чаще, чем в половине случаев
    assert max(first.values()) < len(list(first.elements())) * 0.5, first
    assert len(first) == 5, "не все позиции встречаются первыми"


def test_marking_round_trip(tmp_path):
    """Лист -> разметка -> вердикт. Круг обязан сходиться.

    Проверяется на ЗАРАНЕЕ ИЗВЕСТНОМ ответе: во всех юнитах отмечается
    буква, под которой стоит система «победитель», и счёт обязан отдать
    ей все голоса. Без этого скорер мог бы молча сдвинуть расшифровку на
    одну позицию — а буква у каждого юнита своя, и такую ошибку по
    итоговой таблице не увидеть.
    """
    import json
    import brief_marking_sheet as ms
    import brief_marking_score as sc

    names = ["победитель", "второй", "третий"]
    units, lines = [], []
    for n, text in enumerate(["Первая фраза.", "Вторая фраза тут.",
                              "Третья фраза здесь.", "Четвёртая фраза."], 1):
        order = ms._order(text, len(names))
        mapping = {ms.LETTERS[pos]: names[ai] for pos, ai in enumerate(order)}
        units.append({"unit": n, "phrase": text, "map": mapping})
        letter = next(l for l, nm in mapping.items() if nm == "победитель")
        lines += [f"### {n}. SECTION", "", f"> {text}", "",
                  f"лучший: {letter}    почему: ____", ""]

    sheet = tmp_path / "s.md"
    sheet.write_text("\n".join(lines), encoding="utf-8")
    key = tmp_path / "k.json"
    key.write_text(json.dumps({"arms": names, "units": units}),
                   encoding="utf-8")

    marks = sc.read_sheet(str(sheet))
    assert len(marks) == 4
    decoded = [units[i - 1]["map"][marks[i][0]] for i in sorted(marks)]
    assert decoded == ["победитель"] * 4
    # буквы обязаны быть РАЗНЫМИ хотя бы у части юнитов, иначе тест
    # проходил бы и при фиксированном порядке
    assert len({marks[i][0] for i in marks}) > 1


def test_marking_reads_none_and_ties():
    """«нет» не приписывается никому, а несколько букв — это несколько
    голосов, а не ошибка разбора."""
    import brief_marking_score as sc
    import tempfile
    import os as _os
    body = ("### 1. S\n\n> ф\n\nлучший: нет    почему: ____\n\n"
            "### 2. S\n\n> ф\n\nлучший: А, В    почему: ____\n\n"
            "### 3. S\n\n> ф\n\nлучший: ____    почему: ____\n")
    fd, path = tempfile.mkstemp(suffix=".md")
    with _os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(body)
    try:
        marks = sc.read_sheet(path)
    finally:
        _os.remove(path)
    assert marks[1] is None
    assert marks[2] == ["А", "В"]
    assert 3 not in marks


# --- РАЗДЕЛИТЕЛЬ В СТРОКЕ ОБЯЗАТЕЛЕН ----------------------------------------
#
# Найдено живым прогоном «думающей» Qwen3.6-35B-A3B: она рассуждает вслух
# НУМЕРОВАННЫМИ пунктами, и в план ушли семнадцать «заявок» вида
# «1. Analyze User Input» и «2. Resolve pronouns to actual subjects» —
# пересказ собственных правил промпта. Ни один гейт их не ловил:
# латиница, длина в норме, местоимений нет, снаряжение не современное.

@pytest.mark.parametrize("junk", [
    "1. Analyze User Input:",
    "2. Resolve pronouns to actual subjects from context",
    "3. Adjacent frames must be different",
    "4) Show abstraction as situation or bodily sign",
])
def test_reasoning_lines_are_not_briefs(junk):
    import shot_brief_director as d
    assert d.parse_answer(junk, _packet(["а", "б", "в", "г"])) == {}


def test_proper_rows_still_pass():
    """Ужесточение не имеет права съесть настоящий ответ."""
    import shot_brief_director as d
    got = d.parse_answer("1 | object | a rondel dagger blade, close up\n"
                         "2 | scene | a churned muddy field under grey sky\n",
                         _packet(["а", "б"]))
    assert set(got) == {1, 2}
    assert got[1]["shot_en"] == "a rondel dagger blade, close up"


def test_hardening_changes_nothing_on_real_measured_runs():
    """Контроль, без которого ужесточение отменяло бы прежние числа.

    Проверено на всех снятых замерах (7B, 7B+словарь, 30B, Qwen3-4B,
    30B+словарь): строк без разделителя там НОЛЬ. Тест держит сам
    инвариант на образцах реальных ответов этих моделей.
    """
    import shot_brief_director as d
    real = ("MOOD | -1 | 2 | тяжело\n"
            " 1 | object | a narrow rondel dagger with two round guards\n"
            " 2 | scene | a fallen knight lying in mud\n"
            " 3 | - | -\n")
    got = d.parse_answer(real, _packet(["а", "б", "в"]))
    assert set(got) == {1, 2}
    assert d.parse_mood(real)["tone"] == -1.0


def test_foreign_profile_is_loud(capsys, monkeypatch):
    """Чужой мир канала обязан СКАЗАТЬ о себе, а не молча съесть кадры.

    Измерено живым прогоном: психологический сценарий в репозитории с
    объявленным средневековым миром потерял четыре заявки из двенадцати,
    и все четыре были правильными кадрами («a person slumped over a desk,
    head in hands», «a human hand recoiling sharply from a hot stove
    burner»). Заявки просто не появлялись — снаружи неотличимо от «модель
    не справилась».
    """
    import shot_brief_director as d
    monkeypatch.setattr(d, "STATS", dict(d.STATS, rows=8, rejected=4))
    monkeypatch.setattr(d, "REJECTED",
                        [{"reason": "человек без привязки к миру канала"}] * 4)
    d._warn_if_world_rule_eats_everything()
    assert "ВНИМАНИЕ" in capsys.readouterr().out


def test_occasional_world_rejection_stays_quiet(capsys, monkeypatch):
    """Одиночный отказ — норма работы гварда, а не повод кричать."""
    import shot_brief_director as d
    monkeypatch.setattr(d, "STATS", dict(d.STATS, rows=100, rejected=1))
    monkeypatch.setattr(d, "REJECTED",
                        [{"reason": "человек без привязки к миру канала"}])
    d._warn_if_world_rule_eats_everything()
    assert capsys.readouterr().out == ""


# --- СИМВОЛ ВМЕСТО ВЕЩИ -----------------------------------------------------
#
# Найдено живым прогоном Qwen3.6-35B-A3B на психологическом сценарии: она
# дважды выдала штамп из фотобанка вместо ситуации. На ИСТОРИЧЕСКОМ
# эпизоде такого нет ни у одного из семи мозгов (0 из 766) — проблема
# именно нишевая и вылезает там, где абстракций много.

@pytest.mark.parametrize("shot", [
    "a heavy stone weight resting on a wooden desk, symbolizing mental burden",
    "a broken chain representing the loss of control",
    "a conceptual image of burnout at work",
    "an empty road as a metaphor for the journey ahead",
])
def test_symbol_talk_is_rejected(shot):
    """Бриф, объясняющий свой ЗАМЫСЕЛ, описывает намерение, а не вещь.
    Сфотографировать намерение нельзя."""
    import shot_planner_llm as p
    ok, why = p.brief_is_safe(shot, "фраза", blocklist=())
    assert not ok and "символ" in why


@pytest.mark.parametrize("shot", [
    "a rondel dagger with both round discs, the whole weapon",
    "a phone lying face down on a bedside table at night",
    "a manuscript illumination of armoured men advancing on foot",
    "an hourglass on a wooden table beside a candle",
    "a human skull from an archaeological excavation",
])
def test_real_briefs_survive_the_symbol_guard(shot):
    """Негативный контроль. Класс узкий НАМЕРЕННО: список штампов по
    предметам (песочные часы, клубок ниток) сюда не вносится — песочные
    часы в историческом ролике законны, и запрет по предмету отклонял бы
    годные кадры. Проверено на 888 реально измеренных брифах: ноль
    ложных срабатываний."""
    import shot_planner_llm as p
    ok, why = p.brief_is_safe(shot, "фраза", blocklist=())
    assert ok, why


def test_symbol_rule_is_in_the_prompt():
    import shot_brief_director as d
    assert "НИКАКИХ СИМВОЛОВ" in d.render_prompt(_packet(["Ему тяжело."]))


# --- ВЫКЛЮЧАТЕЛЬ ДОБАВЛЕННЫХ ПРАВИЛ -----------------------------------------
#
# Правила поверх ядра (заземление абстракции, настроение, «показывай
# новое», противопоставление) измерены и оказались НЕ БЕСПЛАТНЫМИ: на
# Qwen3-30B-A3B точность та же (58.9% -> 59.1%), а ответов стало 88
# вместо 107 — от длинного свода модель осторожничает. Выключатель делает
# этот обмен выбором, а не побочным эффектом.

def test_extra_rules_off_keeps_the_core(monkeypatch):
    import importlib
    import shot_brief_director as d
    monkeypatch.setenv("SHOT_BRIEF_EXTRA_RULES", "0")
    importlib.reload(d)
    try:
        prompt = d.render_prompt(_packet(["Ему тяжело."]))
        # ядро на месте
        assert "1. Кадр" in prompt and "5. Показывать нечего" in prompt
        # добавленные — нет
        for head in ("АБСТРАКЦИЮ", "ПОКАЗЫВАЙ НОВОЕ",
                     "ПРОТИВОПОСТАВЛЕНИЕ", "НАСТРОЕНИЕ"):
            assert head not in prompt, head
    finally:
        monkeypatch.delenv("SHOT_BRIEF_EXTRA_RULES")
        importlib.reload(d)


def test_extra_rules_on_by_default():
    import shot_brief_director as d
    prompt = d.render_prompt(_packet(["Ему тяжело."]))
    assert "АБСТРАКЦИЮ" in prompt and "НАСТРОЕНИЕ" in prompt


def test_world_rule_survives_switching_extras_off(monkeypatch):
    """Мир канала — не «добавленное правило», а паспорт ниши: он обязан
    остаться в обоих режимах."""
    import importlib
    import shot_brief_director as d
    monkeypatch.setenv("SHOT_BRIEF_EXTRA_RULES", "0")
    importlib.reload(d)
    try:
        assert "МИР КАДРА" in d.render_prompt(_packet(["ф"]))
    finally:
        monkeypatch.delenv("SHOT_BRIEF_EXTRA_RULES")
        importlib.reload(d)


# --- ЧУЖОЙ МИР КАНАЛА НЕ ТОЛЬКО ФИЛЬТРУЕТ, НО И РУЛИТ -----------------------
#
# Самая опасная находка всей работы. Предохранитель по доле отказов
# (test_foreign_profile_is_loud) ловит случай, когда чужой мир РЕЖЕТ
# хорошие кадры. Но живой прогон психологического сценария в этом
# репозитории показал другое: модель ПОСЛУШАЛАСЬ объявленного
# средневекового мира и выдала «a knight in armor standing beside a closed
# chest» и «a monk's hand touching a cracked mirror» — на текст про пустой
# файл в ноутбуке. Отклонять было нечего, предохранитель промолчал.
#
# Автоматически отличить «модель послушалась чужого мира» от «модель
# права» нечем: обе выдачи формально безупречны. Поэтому защита здесь не
# автоматическая, а громкая — объявленный мир печатается в начале КАЖДОГО
# прогона, и его можно выключить одной переменной.

def test_declared_world_is_printed_loudly(capsys):
    import script_parser
    import shot_brief_director as d
    blocks = script_parser.parse_blocks(
        os.path.join(REPO, "videos", "02_ne-mechom", "script.txt"))[:3]

    class Silent:
        name = "silent"

        def ask(self, prompt, chapter_no):
            return ""

    d.run(os.path.join(REPO, "videos", "02_ne-mechom"), blocks, Silent(),
          cache_dir=None, verbose=True)
    out = capsys.readouterr().out
    assert "МИР КАДРА ЭТОГО КАНАЛА" in out
    assert "SHOT_BRIEF_WORLD=off" in out


def test_world_can_be_switched_off_for_a_foreign_episode(monkeypatch):
    """Эпизод из чужой ниши в этом репозитории обязан иметь способ НЕ
    получать средневековый мир."""
    import shot_brief_director as d
    assert d.domain_contract(), "у этого канала мир объявлен"
    monkeypatch.setenv("SHOT_BRIEF_WORLD", "off")
    assert d.domain_contract() == ""
    prompt = d.render_prompt(_packet(["Ему стало нечем дышать."]))
    assert "МИР КАДРА" not in prompt
    for w in ("knight", "warrior", "armour"):
        assert w not in prompt.lower()
