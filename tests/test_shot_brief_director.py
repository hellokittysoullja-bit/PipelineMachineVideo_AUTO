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
    ("A man stepping onto a battlefield", "эпох"),
    ("A person standing up", "эпох"),
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
