# -*- coding: utf-8 -*-
"""Режиссёрская разработка главы и ключ плана.

Каждый тест здесь проверен КОНТРОЛЬНЫМ ПРОГОНОМ со снятой правкой: без
неё он падает. Тест, который зелен и с правкой, и без неё, не защищает
ничего — этот урок в репозитории уже оплачен трижды.
"""
import re
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


def test_cache_key_still_separates_brain_and_prompt():
    """А вот КЭШ ОТВЕТА обязан зависеть и от мозга, и от версии пакета:
    иначе новая модель молча отдавала бы ответ старой, а переписанный
    промпт наследовал бы ответы, снятые по прежним правилам.

    Гарантия переехала с пофразового автомата на главу вместе с мозгом;
    сам автомат удалён (16.09), а требование к ключу осталось прежним."""
    import shot_brief_director as d
    packet = _packet([{"n": 1, "text": "Тебе нужно всего лишь встать."}])
    a = d._cache_key(packet, "qwen3-30b")
    assert d._cache_key(packet, "qwen3-4b") != a, "мозг не входит в ключ"

    other = _packet([{"n": 1, "text": "Другая фраза целиком."}])
    assert d._cache_key(other, "qwen3-30b") != a, "текст главы не входит в ключ"


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

    Со средневековым словарём, зашитым в код до 15.09, ЧЕТЫРЕ из этих
    шести законных психологических брифов отклонялись правилом «человек
    без привязки к эпохе» — то есть система запрещала показывать человека
    каналу, у которого человек и есть предмет разговора.

    Было «три», стало «четыре» 16.09, и это не переписанное задним числом
    число: совпадение стало считаться по ГРАНИЦЕ СЛОВА вместо пробелов с
    обеих сторон, и `a person's hands` перестал прятаться за апострофом.
    То есть чужой мир кусался ещё сильнее, чем показывал прежний замер.
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
    assert len(rejected) == 4, rejected
    # Поимённо, а не числом: иначе «четыре» удержится и в случае, когда
    # отклоняются совсем другие четыре брифа.
    assert "a person's hands clenched on a kitchen table" in rejected


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


def test_world_switch_reaches_the_validator_too(monkeypatch):
    """Выключатель мира обязан действовать И в проверке, не только в задании.

    Найдено собственным предохранителем на живом прогоне:
    SHOT_BRIEF_WORLD=off снял доменное правило из задания, но проверка
    по-прежнему читала словарь профиля и зарубила 7 годных заявок из 17
    («hand resting on desk», «empty chair beside desk»). Половинчатый
    выключатель хуже отсутствующего: задание уже не диктует чужой мир, а
    проверка всё ещё требует его слов — и кадры пропадают молча.
    """
    import shot_planner_llm as p
    psych = "a person sitting rigidly at a desk, shoulders tense"
    assert not p.brief_is_safe(psych, "ф", blocklist=())[0]
    monkeypatch.setenv("SHOT_BRIEF_WORLD", "off")
    assert p.domain_anchor_words() == ()
    assert p.brief_is_safe(psych, "ф", blocklist=())[0]


def test_world_switch_off_does_not_weaken_this_channel(monkeypatch):
    """Снятие выключателя возвращает строгость полностью."""
    import shot_planner_llm as p
    monkeypatch.setenv("SHOT_BRIEF_WORLD", "off")
    monkeypatch.delenv("SHOT_BRIEF_WORLD")
    assert p.domain_anchor_words()
    assert not p.brief_is_safe("A man stepping onto a battlefield", "ф",
                               blocklist=())[0]


# --- УСТАНОВЩИК ЛОКАЛЬНОГО РЕЖИССЁРА ----------------------------------------

@pytest.mark.parametrize("ram", [64.0, 32.0, 16.0, 15.0, 14.9, 8.0, 4.0, None])
def test_setup_always_picks_the_small_model(ram):
    """Выбор БОЛЬШЕ НЕ ЗАВИСИТ ОТ ПАМЯТИ (решение владельца 16.09).

    Прежнее правило «15+ ГБ -> крупная» стояло на её 69 попаданиях против
    63 у маленькой, и это число оказалось НЕСРАВНИМЫМ: снято пакетом
    версии 4, тогда как маленькая мерилась на другой. Та же ловушка, что
    в тот же день поймана на квантовке. Преимущество крупной не
    установлено, а цена установлена: 180 с на главу против 35 и 11 ГБ ОЗУ
    под работой.

    Памяти много — это НЕ повод тащить лишние десять гигабайт.
    """
    import setup_local_director as s
    assert s.pick(ram) == "4b"


def test_setup_respects_manual_choice():
    import setup_local_director as s
    assert s.pick(4.0, forced="30b") == "30b"


def test_setup_numbers_match_the_measurement():
    """Числа в установщике — те же, что в отчёте. Разойдись они, человек
    выбирал бы модель по устаревшему обещанию."""
    import setup_local_director as s
    assert s.MODELS["30b"]["score"] == 69
    assert s.MODELS["4b"]["score"] == 63
    assert s.MODELS["4b"]["gb"] < s.MODELS["30b"]["gb"] / 4


class TestSamplingIsDeterministic:
    """Гарантия переехала сюда вместе с мозгом (16.09).

    Найдено внешней оценкой и подтверждено проверкой: у llama.cpp `--seed`
    по умолчанию -1 (случайный), а температура у прежнего пофразового
    автомата стояла 0.2 — не ноль. Значит сравнение промптов v2 и v3 было
    НЕВОСПРОИЗВОДИМЫМ, и разница могла оказаться шумом выборки, а не
    эффектом правки.

    Планирование — не творческая задача: на один и тот же вопрос нужен
    один и тот же ответ, иначе теряет смысл и сравнение версий, и план,
    который эпизод переиспользует между прогонами.

    Проверяется ИСХОДНИКОМ, а не живым вызовом: llama_cpp — опциональная
    зависимость, и тест, молча пропускающийся без неё, не сторожил бы
    ничего именно в том окружении, где мозг работает.
    """

    def _local_brain_source(self):
        import inspect
        import shot_brief_director as d
        return inspect.getsource(d.LocalBrain)

    def test_seed_default_is_fixed_never_random(self):
        import inspect
        import shot_brief_director as d
        sig = inspect.signature(d.LocalBrain.__init__)
        seed = sig.parameters["seed"].default
        assert isinstance(seed, int)
        assert seed >= 0, "seed=-1 у llama.cpp означает случайный"

    def test_every_generation_call_pins_temperature_and_seed(self):
        src = self._local_brain_source()
        calls = [m for m in re.finditer(r"self\.llm\.create_\w+\(", src)]
        assert calls, "не нашлось ни одного вызова генерации — тест ослеп"
        for m in calls:
            tail = src[m.start():m.start() + 400]
            assert "temperature=0.0" in tail, tail[:200]
            assert "seed=self.seed" in tail, tail[:200]


class TestModelIsFoundWithoutFlags:
    """«Полностью в коде навсегда»: обычный запуск — одна команда без
    флагов. Поиск модели обязан быть ДЕТЕРМИНИРОВАННЫМ: два прогона на
    одной машине, взявшие разные файлы, дали бы план от разных мозгов, и
    сравнить их было бы нечем.
    """

    def _dir(self, tmp_path, monkeypatch, names):
        import shot_brief_director as d
        monkeypatch.delenv("LLAMA_MODEL_GGUF", raising=False)
        models = tmp_path / d.MODELS_DIR_NAME
        models.mkdir()
        for n in names:
            (models / n).write_bytes(b"x")
        monkeypatch.setattr(d, "REPO", str(tmp_path))
        return d

    def test_explicit_path_wins_over_everything(self, tmp_path, monkeypatch):
        d = self._dir(tmp_path, monkeypatch, d_names := ["Qwen3-4B-Instruct-2507-Q4_K_M.gguf"])
        mine = tmp_path / "mine.gguf"
        mine.write_bytes(b"x")
        monkeypatch.setenv("LLAMA_MODEL_GGUF", str(tmp_path / d_names[0]))
        assert d.find_model(str(mine)) == str(mine)

    def test_env_wins_over_directory(self, tmp_path, monkeypatch):
        d = self._dir(tmp_path, monkeypatch, ["Qwen3-4B-Instruct-2507-Q4_K_M.gguf"])
        env = tmp_path / "env.gguf"
        env.write_bytes(b"x")
        monkeypatch.setenv("LLAMA_MODEL_GGUF", str(env))
        assert d.find_model(None) == str(env)

    def test_missing_explicit_path_falls_through_not_crashes(self, tmp_path, monkeypatch):
        """Опечатка в --model не должна ронять прогон молчаливым исключением
        и не должна выдавать несуществующий путь за найденную модель."""
        d = self._dir(tmp_path, monkeypatch, ["Qwen3-4B-Instruct-2507-Q4_K_M.gguf"])
        got = d.find_model("/нет/такого.gguf")
        assert got and os.path.exists(got)

    def test_measured_preference_order(self, tmp_path, monkeypatch):
        """30B замерена выше 4B (69 против 63) — при обеих на диске берётся
        она, а не алфавит (по алфавиту первой была бы 30B случайно, поэтому
        проверяется и обратный порядок имён)."""
        d = self._dir(tmp_path, monkeypatch, [
            "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
            "Qwen3-30B-A3B-Instruct-2507-Q3_K_S.gguf",
            "aaa-first-in-alphabet.gguf",
        ])
        assert os.path.basename(d.find_model(None)) == d.PREFERRED_MODELS[0]

    def test_unknown_model_is_taken_deterministically(self, tmp_path, monkeypatch):
        """Чужой .gguf тоже берётся — список предпочтений это не белый
        список. Но при нескольких выбор обязан быть воспроизводимым."""
        d = self._dir(tmp_path, monkeypatch, ["zeta.gguf", "alpha.gguf"])
        first = d.find_model(None)
        assert os.path.basename(first) == "alpha.gguf"
        assert d.find_model(None) == first

    def test_no_directory_is_none_not_exception(self, tmp_path, monkeypatch):
        import shot_brief_director as d
        monkeypatch.delenv("LLAMA_MODEL_GGUF", raising=False)
        monkeypatch.setattr(d, "REPO", str(tmp_path / "нет"))
        assert d.find_model(None) is None

    def test_non_gguf_files_are_ignored(self, tmp_path, monkeypatch):
        d = self._dir(tmp_path, monkeypatch, [])
        (tmp_path / d.MODELS_DIR_NAME / "README.md").write_text("не модель")
        assert d.find_model(None) is None


class TestInlineWriteIsTheDefault:
    """План на диске, который никто не применил, — это ровно тот класс
    «слой есть, и его никто не зовёт», которым репозиторий горел шесть раз
    (Openverse, Pixabay, Unsplash, reveal-акценты, filter_alt_blocklist,
    DEFLICKER). Поэтому запись включена по умолчанию, а выключается явно.
    """

    def _parser_defaults(self):
        import argparse
        import shot_brief_director as d
        parsed = {}
        real = argparse.ArgumentParser.parse_args

        def spy(self, args=None, namespace=None):
            ns = real(self, args, namespace)
            parsed["ns"] = ns
            raise SystemExit(0)
        argparse.ArgumentParser.parse_args = spy
        try:
            try:
                d.main(["x", "somewhere"])
            except SystemExit:
                pass
        finally:
            argparse.ArgumentParser.parse_args = real
        return parsed["ns"]

    def test_write_inline_defaults_on(self):
        assert self._parser_defaults().write_inline is True

    def test_brain_defaults_to_local(self):
        assert self._parser_defaults().brain == "local"


class TestChapterFitsInTheContextWindow:
    """Молчаливая обрезка входа по лимиту модели в этом репозитории УЖЕ
    случалась: SIGLIP2_MAX_TEXT_LENGTH=64 резала 25% фраз, и увидеть это
    было негде, пока не измерили. У режиссёра тот же риск и он опаснее:
    обрежется ХВОСТ главы, то есть последние фразы просто не дойдут до
    модели, а ответ на них она всё равно обязана дать.

    Замер 16.09 настоящим токенизатором Qwen3-4B на эпизоде 02: худшая
    глава 3133 токена входа из 8192, ответ 364 из 900 — двукратный запас
    по обеим осям. Тест сторожит не сам замер (для него нужна модель, а
    её в CI нет), а то, ЧТО ЕГО ЛОМАЕТ: раздувшийся промпт.

    Оценка сверху нарочно грубая и завышенная — символы делятся на 2, а
    не на 3.5-4, как реально даёт токенизатор на смеси русского с
    английским. Тест обязан падать РАНЬШЕ настоящей обрезки, а не после.
    """

    CHARS_PER_TOKEN = 2.0          # заведомо пессимистично
    TOKENS_PER_ANSWER_LINE = 22    # замер: 364 токена на 16 строк + шапка

    @property
    def n_ctx(self):
        """Читается ИЗ КОДА, а не держится константой рядом.

        Первая версия теста писала 8192 литералом — и контрольный прогон
        показал, что при n_ctx=2048 она остаётся ЗЕЛЁНОЙ, то есть не
        сторожит ровно тот случай, ради которого написана. Тот же класс,
        что этот файл уже дважды ловил на рассинхроне двух копий одного
        значения."""
        import inspect
        import shot_brief_director as d
        return inspect.signature(d.LocalBrain.__init__).parameters["n_ctx"].default

    def _packets(self):
        import script_parser
        import shot_brief_director as d
        path = os.path.join(REPO, "videos", "02_ne-mechom", "script.txt")
        if not os.path.exists(path):
            pytest.skip("эпизод 02 недоступен")
        blocks = script_parser.parse_blocks(path)
        return d, list(d.packets(os.path.join(REPO, "videos", "02_ne-mechom"),
                                 blocks))

    def test_prompt_plus_answer_leaves_room(self):
        d, packets = self._packets()
        assert packets, "пакеты глав не собрались — тест ослеп"
        worst = []
        for p in packets:
            est = len(d.render_prompt(p)) / self.CHARS_PER_TOKEN
            est += len(p["units"]) * self.TOKENS_PER_ANSWER_LINE
            worst.append((est, p["section"]))
        est, section = max(worst)
        assert est < self.n_ctx, (
            f"глава {section!r} по грубой оценке просит {est:.0f} токенов "
            f"при n_ctx={self.n_ctx}. Хвост главы обрежется МОЛЧА: модель "
            f"не увидит последние фразы, но ответ на них всё равно обязана "
            f"дать. Поднять n_ctx у LocalBrain или резать главу на части.")

    def test_answer_cap_covers_the_longest_chapter(self):
        """Потолок генерации обязан покрывать САМУЮ длинную главу: обрыв
        на потолке теряет последние заявки главы, и снаружи это
        неотличимо от «модель про них промолчала»."""
        import shot_brief_director as d
        _, packets = self._packets()
        longest = max(len(p["units"]) for p in packets)
        need = longest * self.TOKENS_PER_ANSWER_LINE
        assert d.LocalBrain.DEFAULT_MAX_TOKENS >= need, (
            f"самая длинная глава — {longest} фраз (~{need} токенов ответа), "
            f"а потолок {d.LocalBrain.DEFAULT_MAX_TOKENS}")


# --- Самопроверка режиссёра (SHOT_BRIEF_CRITIQUE) ---------------------------
#
# Контроль: снятая правка (вернуть merge_critique к `return dict(draft_rows,
# **critique_rows)` или удалить проверку `if critique_on and rows`) валит
# test_merge_never_adds_a_unit_the_draft_skipped и
# test_critique_off_by_default_means_one_call_per_chapter.

class _FakeBrain:
    """Считает вызовы и различает черновик от критики по тексту промпта —
    так же, как это делает render_critique_prompt (маркер «Ты только что
    описал»), а не по порядковому номеру вызова."""

    def __init__(self, draft_text, critique_text=None):
        self.name = "fake"
        self.calls = []
        self.draft_text = draft_text
        self.critique_text = critique_text

    def ask(self, prompt, chapter_no):
        is_critique = "Ты только что описал" in prompt
        self.calls.append(("critique" if is_critique else "draft", chapter_no))
        if is_critique:
            return self.critique_text if self.critique_text is not None else self.draft_text
        return self.draft_text


def _one_packet(monkeypatch, tmp_path):
    import shutil
    import script_parser
    import shot_brief_director as d
    _clean_channel(monkeypatch)
    shutil.copy(os.path.join(FIXTURE, "script_psychology.txt"),
                tmp_path / "script.txt")
    blocks = script_parser.parse_blocks(str(tmp_path / "script.txt"))
    packet = next(iter(d.packets(str(tmp_path), blocks)))
    return d, blocks, packet


def test_merge_never_adds_a_unit_the_draft_skipped():
    """Критика ответила на юнит 3, которого не было в черновике — merge
    его выбрасывает. Молчание черновика уже прошло свою проверку;
    ответ второй попытки той же модели на нём не надёжнее первой."""
    import shot_brief_director as d
    draft = {1: {"shot_en": "a", "function": "object"},
             2: {"shot_en": "b", "function": "object"}}
    critique = {2: {"shot_en": "b-fixed", "function": "object"},
                3: {"shot_en": "c-new", "function": "object"}}
    merged = d.merge_critique(draft, critique)
    assert set(merged) == {1, 2}, "критика добавила юнит, которого не было в черновике"
    assert merged[2]["shot_en"] == "b-fixed"
    assert merged[1]["shot_en"] == "a"


def test_critique_prompt_carries_the_draft_and_the_phrases(monkeypatch, tmp_path):
    d, _, packet = _one_packet(monkeypatch, tmp_path)
    draft_raw = "1 | object | a closed door\n2 | scene | an empty room"
    prompt = d.render_critique_prompt(packet, draft_raw)
    assert draft_raw in prompt
    assert packet["units"][0]["text"] in prompt
    # Оба задокументированных класса промахов названы явно, не общим
    # «сделай лучше» — расплывчатая инструкция моделям этого размера
    # ничего не чинит (тот же урок, что уже записан про молчание модели).
    assert "отсыл" in prompt.lower()
    assert "предмет" in prompt.lower()


def test_critique_cache_key_differs_from_draft(monkeypatch, tmp_path):
    """Одна и та же функция ключа (`_cache_key_text`) на РАЗНОМ тексте —
    черновик и критика не должны читать/писать один файл кэша."""
    d, _, packet = _one_packet(monkeypatch, tmp_path)
    draft_prompt = d.render_prompt(packet)
    draft_raw = "1 | object | a closed door"
    crit_prompt = d.render_critique_prompt(packet, draft_raw)
    assert (d._cache_key_text(draft_prompt, "fake")
            != d._cache_key_text(crit_prompt, "fake"))


def test_critique_off_by_default_means_one_call_per_chapter(monkeypatch, tmp_path):
    """Флаг не выставлен — второго вызова модели нет вообще, ноль лишней
    цены для всех, кто ничего не менял в .env."""
    monkeypatch.delenv("SHOT_BRIEF_CRITIQUE", raising=False)
    d, blocks, _ = _one_packet(monkeypatch, tmp_path)
    monkeypatch.setattr(d, "MOODS", {})
    brain = _FakeBrain("1 | object | a closed door\n2 | scene | an empty room")
    d.run(str(tmp_path), blocks, brain, cache_dir=None, verbose=False)
    assert all(kind == "draft" for kind, _ in brain.calls), brain.calls


def test_critique_on_rewrites_via_second_call(monkeypatch, tmp_path):
    """Флаг включён — второй, отличимый по промпту вызов реально
    происходит, и его ответ доходит до итоговой заявки."""
    import shot_planner_llm
    monkeypatch.setenv("SHOT_BRIEF_CRITIQUE", "1")
    d, blocks, packet = _one_packet(monkeypatch, tmp_path)
    monkeypatch.setattr(d, "MOODS", {})
    first_unit_text = packet["units"][0]["text"]
    n1 = packet["units"][0]["n"]
    draft = f"{n1} | object | a generic wrong picture that is safe text"
    fixed = f"{n1} | object | a specific correct picture that is safe text"
    # brief_is_safe должна принимать обе — тест про перенос критики,
    # не про сам гейт безопасности.
    monkeypatch.setattr(shot_planner_llm, "brief_is_safe",
                        lambda shot, text: (True, None))
    brain = _FakeBrain(draft, critique_text=fixed)
    found = d.run(str(tmp_path), blocks, brain, cache_dir=None, verbose=False,
                  only_sections={d._section_key(packet["section"])})
    assert any(k == "critique" for k, _ in brain.calls), (
        "SHOT_BRIEF_CRITIQUE=1, а второго вызова не было")
    block_index = packet["units"][0]["block_index"]
    assert found[block_index]["shot_en"] == "a specific correct picture that is safe text"


# --- Гвард против сдвига нумерации в критике --------------------------------
#
# Фикстура — РЕАЛЬНЫЙ ответ модели (эпизод 02, BLOCK 5 «ДВЕСТИ МЕТРОВ»,
# живой прогон 16.09), не синтетика: синтетический пример не воспроизвёл
# бы то, что случилось на самом деле — критика молча пропустила строку 5
# и заново пронумеровала хвост, каждая следующая строка стала формально
# безупречным ответом на ЧУЖОЙ юнит. Живой A/B на полном эпизоде (142
# юнита) дал 65 -> 62 попаданий БЕЗ этого гварда — чистый регресс на трёх
# юнитах из четырёх задетых сдвигом (пятый совпал с черновиком случайно).
#
# Контроль: закомментировать вызов `_looks_like_neighbor_shift` в
# merge_critique — test_shift_guard_recovers_the_real_regression падает,
# юниты 5-8 остаются испорченными сдвигом.

_SHIFT_FIXTURE = os.path.join(REPO, "tests", "fixtures", "critique_shift")


def _shift_fixture_packet():
    """Реальный пакет главы BLOCK 5 эпизода 02 — тот же самый, на котором
    снят живой A/B 16.09. Профиль канала (мир кадра) не подчищаем — он
    не влияет на разбор ответа, только на текст промпта, которого здесь
    не строим."""
    import script_parser
    import shot_brief_director as d
    blocks = script_parser.parse_blocks(
        os.path.join(REPO, "videos", "02_ne-mechom", "script.txt"))
    for p in d.packets(os.path.join(REPO, "videos", "02_ne-mechom"), blocks):
        if p["section"].startswith("BLOCK 5"):
            return d, p
    raise AssertionError("BLOCK 5 не нашёлся — фикстура эпизода изменилась")


def test_shift_guard_recovers_the_real_regression():
    d, packet = _shift_fixture_packet()
    with open(os.path.join(_SHIFT_FIXTURE, "draft_block5.txt"), encoding="utf-8") as f:
        draft_raw = f.read()
    with open(os.path.join(_SHIFT_FIXTURE, "critique_block5.txt"), encoding="utf-8") as f:
        crit_raw = f.read()
    draft_rows = d.parse_answer(draft_raw, packet)
    crit_rows = d.parse_answer(crit_raw, packet)
    d.STATS["critique_shift_caught"] = 0
    merged = d.merge_critique(draft_rows, crit_rows)
    for n in (5, 6, 7, 8):
        assert merged[n]["shot_en"] == draft_rows[n]["shot_en"], (
            f"юнит {n}: гвард не остановил сдвиг, заявка испорчена")
    assert d.STATS["critique_shift_caught"] == 4, d.STATS["critique_shift_caught"]


def test_shift_guard_does_not_block_a_real_unrelated_fix():
    """Гвард обязан ловить ТОЛЬКО буквальное совпадение с соседом, а не
    любую замену вообще — иначе критика становится no-op."""
    import shot_brief_director as d
    draft = {1: {"shot_en": "a closed wooden door", "function": "object"},
             2: {"shot_en": "an empty chair by the window", "function": "object"}}
    critique = {1: {"shot_en": "a rusted iron gate, chain wrapped around it",
                    "function": "object"}}
    d.STATS["critique_shift_caught"] = 0
    merged = d.merge_critique(draft, critique)
    assert merged[1]["shot_en"] == "a rusted iron gate, chain wrapped around it"
    assert d.STATS["critique_shift_caught"] == 0


# --- Рассуждение внутри строки (SHOT_BRIEF_REASONING) -----------------------
#
# Контроль: убрать `_REF_BRACKET_RE.match(shot)` из parse_answer (не
# снимать скобку) — test_reference_bracket_is_stripped_before_validation
# падает, потому что "[ref: он=земля] a churned..." не проходит проверку
# длины слов (скобка считается словами) или остаётся в shot_en как есть.

def test_reference_bracket_is_stripped_before_validation():
    """Заметка о ссылке уходит из текста ДО всех проверок — включая
    подсчёт слов, иначе сама скобка исказила бы длину описания."""
    import shot_brief_director as d
    pkt = _packet(["Земля тянет.", "При этом он был под ногами у каждого."])
    got = d.parse_answer(
        "1 | object | a churned muddy field, boot prints everywhere\n"
        "2 | object | [ref: он=земля] a churned field seen underfoot, "
        "trampled soil\n", pkt)
    assert got[2]["shot_en"] == "a churned field seen underfoot, trampled soil"
    assert got[2]["referent"] == "он=земля"
    assert got[1]["referent"] is None


def test_bracket_only_matches_at_the_start():
    """Скобка ПОСЕРЕДИНЕ описания — не заметка о ссылке, а часть кадра.
    Снимать её значило бы терять текст без всякой причины."""
    import shot_brief_director as d
    pkt = _packet(["Дверь со стеклянной вставкой."])
    got = d.parse_answer(
        "1 | object | a wooden door [glass panel] standing ajar\n", pkt)
    assert got[1]["shot_en"] == "a wooden door [glass panel] standing ajar"
    assert got[1]["referent"] is None


def test_reasoning_instruction_is_off_by_default(monkeypatch):
    monkeypatch.delenv("SHOT_BRIEF_REASONING", raising=False)
    import shot_brief_director as d
    prompt = d.render_prompt(_packet(["а", "б"]))
    assert "[ref:" not in prompt


def test_reasoning_instruction_appears_when_enabled(monkeypatch):
    monkeypatch.setenv("SHOT_BRIEF_REASONING", "1")
    import shot_brief_director as d
    prompt = d.render_prompt(_packet(["а", "б"]))
    assert "[ref:" in prompt


# --- Снятие рассуждения перед разбором ---------------------------------
#
# Контроль: убрать вызовы `_THINK_RE.sub`/`_UNCLOSED_THINK_RE.sub` из
# shot_planner_llm._clean_stream — оба теста ниже падают, строки
# рассуждения снова попадают на разбор `_ROW_RE`.

def test_closed_think_block_is_removed_before_parsing():
    """Реальный, уже случившийся класс отказа (Qwen3.6-35B-A3B, пересказ
    правил нумерованными пунктами) — блок рассуждения не должен доехать
    до построчного разбора вообще."""
    import shot_brief_director as d
    pkt = _packet(["а", "б"])
    raw = ("<think>\n1 | object | это рассуждение, а не ответ, но с "
           "разделителем\n2 | object | и тут тоже\n</think>\n"
           "1 | object | a dented steel breastplate, close up\n"
           "2 | scene | a churned muddy field under grey sky\n")
    got = d.parse_answer(raw, pkt)
    assert got[1]["shot_en"] == "a dented steel breastplate, close up"
    assert got[2]["shot_en"] == "a churned muddy field under grey sky"


def test_unclosed_think_block_leaves_nothing_to_salvage():
    """Модель упёрлась в потолок токенов посреди рассуждения — до ответа
    не дошла. Честный исход — пустая глава, а не заявки из рассуждения."""
    import shot_brief_director as d
    pkt = _packet(["а", "б"])
    raw = ("<think>\n1 | object | долгое рассуждение без конца, "
           "которое никогда не закрывается тегом\n")
    assert d.parse_answer(raw, pkt) == {}


def test_a_non_thinking_model_response_is_untouched():
    """Без тега рассуждения — новый шаг чистый no-op."""
    import shot_planner_llm as p
    raw = "1 | object | a dented steel breastplate, close up"
    assert p._clean_stream(raw) == raw


class _FakeModel:
    """Минимальный дубль llama.cpp-модели: только то, что читает `_chat_prompt`."""

    def __init__(self, eos="<|im_end|>", bos_id=-1):
        self._eos, self._bos_id = eos, bos_id

    def token_get_text(self, tok):
        return self._eos


class _FakeLlama:
    def __init__(self, template, eos="<|im_end|>"):
        self.metadata = {"tokenizer.chat_template": template} if template else {}
        self._model = _FakeModel(eos)

    def token_eos(self):
        return 1

    def token_bos(self):
        return -1


_QWEN_TEMPLATE = (
    "{% for m in messages %}"
    "<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def _brain_with(llm):
    """Объект LocalBrain без загрузки весов: нужен только `self.llm`."""
    import shot_brief_director as d
    b = d.LocalBrain.__new__(d.LocalBrain)
    b.llm, b.name, b.seed, b.max_tokens = llm, "fake.gguf", 1, 64
    return b


def test_chat_prompt_uses_model_template_and_matches_old_hardcode():
    """Разметку даёт САМА модель, и для семейства Qwen это БАЙТ-В-БАЙТ то,
    что было зашито раньше.

    Оба утверждения нужны вместе: первое — что зашитого допущения про
    семейство больше нет, второе — что у канала на модели по умолчанию
    ничего не поехало. Проверено и живой моделью: рендер шаблона
    Qwen3-4B-Instruct-2507 совпал с прежней строкой символ в символ.
    """
    pytest.importorskip("llama_cpp")
    import shot_brief_director as d
    b = _brain_with(_FakeLlama(_QWEN_TEMPLATE))
    got = b._chat_prompt("ТЕКСТ")
    assert got == d.LocalBrain.FALLBACK_CHAT_FMT.format(prompt="ТЕКСТ")


def test_chat_prompt_is_not_chatml_for_a_foreign_template():
    """Модель с ЧУЖОЙ разметкой получает СВОЮ, а не ChatML.

    Ради этого правка и сделана: MiniCPM5-2B — думающая модель, её
    `<think>` не закрывался в пределах потолка и все 13 глав вернулись
    пустыми, а затравить закрытым блоком было нечем — ChatML не её.
    """
    pytest.importorskip("llama_cpp")
    b = _brain_with(_FakeLlama("{% for m in messages %}<用户>{{ m['content'] }}<AI>{% endfor %}"))
    captured = []
    b.llm.create_completion = lambda **kw: (
        captured.append(kw) or {"choices": [{"text": "ok"}]})
    b._ask_without_thinking("ТЕКСТ")
    # Проверяется ПРОД-путь целиком, а не только рендер: до правки сюда
    # уходил ChatML независимо от модели, и тест обязан падать именно на
    # этом, а не на отсутствии новой функции.
    sent = captured[0]["prompt"]
    assert "<|im_start|>" not in sent and "<用户>" in sent


def test_missing_template_falls_back_loudly(capsys):
    """Нет шаблона — откат на ChatML, и он НАЗВАН вслух.

    Молчаливый откат здесь означал бы чужую разметку в затравке и пустую
    главу без единой строки о причине — ровно тот класс, которым этот
    репозиторий уже горел.
    """
    b = _brain_with(_FakeLlama(None))
    assert b._chat_prompt("ТЕКСТ") is None
    captured = []
    b.llm.create_completion = lambda **kw: (
        captured.append(kw) or {"choices": [{"text": "ok"}]})
    b._ask_without_thinking("ТЕКСТ")
    out = capsys.readouterr().out
    assert "чат-шаблон" in out
    assert captured[0]["prompt"].startswith("<|im_start|>user\nТЕКСТ")


def test_closed_think_block_is_seeded_after_the_model_template():
    """Затравка закрытым блоком идёт ПОСЛЕ разметки модели, а не вместо неё."""
    pytest.importorskip("llama_cpp")
    import shot_brief_director as d
    b = _brain_with(_FakeLlama(_QWEN_TEMPLATE))
    captured = []
    b.llm.create_completion = lambda **kw: (
        captured.append(kw) or {"choices": [{"text": "ok"}]})
    b._ask_without_thinking("ТЕКСТ")
    p = captured[0]["prompt"]
    assert p.endswith(f"{d.LocalBrain.THINK_OPEN}\n\n{d.LocalBrain.THINK_CLOSE}\n\n")
    assert p.index("<|im_start|>assistant") < p.index(d.LocalBrain.THINK_OPEN)
