# -*- coding: utf-8 -*-
"""Бриф -> короткий запрос для СТОКА.

Полка принимает описание целиком; у стоков текстовый API, где каждое лишнее
слово сужает выдачу. Перевод между этими двумя режимами — то место, где
брифы впервые начинают влиять на слоты, у которых нет музейного пути.
"""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import pipeline_smart as ps  # noqa: E402


def _q(brief, fallback="medieval knight"):
    return ps.brief_to_stock_query(brief, fallback=fallback)


def test_no_brief_is_byte_identical_to_today():
    """Нет брифа — слот обязан получить ровно то, что получал раньше."""
    assert _q(None) == "medieval knight"
    assert _q("") == "medieval knight"
    assert _q("   ") == "medieval knight"


def test_brief_of_pure_framing_words_falls_back():
    """Бриф из одних слов ракурса не несёт предмета — откат, а не мусор."""
    assert _q("close up, seen from the front, whole figure") == "medieval knight"


def test_era_anchor_survives_the_word_cap():
    """Реальный найденный промах: единственный якорь эпохи стоял шестым
    словом и обрезался, оставляя запрос без эпохи вообще. Одиночное
    `plate armour` первым результатом даёт танк — см. каскад.

    Сравнение — через query_mentions_term (составные слова/множественное
    число/британское-американское написание), не точным `w in era`: сам
    брифа несёт "armoured" (форма якоря "armour"), и правильный, ФАКТИЧЕСКИ
    авторский якорь обязан пережить обрезку — а не абстрактное слово из
    fallback, найденное только потому, что старая проверка не узнавала
    словоформу."""
    q = _q("a manuscript illumination of a battle between armoured knights")
    era_terms = [a.lower() for a in ps.OPENVERSE_ERA_ANCHORS]
    assert any(ps.query_mentions_term(w, term) for w in q.split() for term in era_terms), q


def test_author_word_order_is_preserved():
    """Перестановка ломает термин: `rondel dagger` — название предмета,
    `dagger rondel` — нет. Измерено на реальных брифах эпизода."""
    q = _q("a rondel dagger with a narrow stiff blade and a disc guard")
    assert "rondel dagger" in q


def test_subject_of_a_scene_is_not_stripped():
    """Словарь ракурса НЕ ТОТ ЖЕ, что OPENVERSE_QUERY_MODIFIERS: тот режет
    `mud`/`field`/`battlefield`, потому что расширяет архивный запрос. Для
    брифа сцены это и есть предмет кадра."""
    q = _q("thick wet clay mud with deep boot prints")
    assert "mud" in q
    q2 = _q("an open muddy field seen low from the ground")
    assert "field" in q2 or "muddy" in q2


def test_the_two_vocabularies_are_intentionally_different():
    """Если списки когда-нибудь сольют в один, сцены начнут терять предмет."""
    framing = {w.lower() for w in ps.BRIEF_FRAMING_WORDS}
    modifiers = {w.lower() for w in ps.OPENVERSE_QUERY_MODIFIERS}
    assert "mud" in modifiers and "mud" not in framing
    assert "battlefield" in modifiers and "battlefield" not in framing


def test_query_stays_short_enough_for_a_text_api():
    q = _q("a complete articulated suit of medieval plate armour standing, "
           "whole figure, frontal, dark background, polished steel")
    assert ps.OPENVERSE_QUERY_MIN_WORDS <= len(q.split()) <= ps.BRIEF_STOCK_QUERY_MAX_WORDS


def test_brief_query_is_added_not_substituted():
    """Additive по построению: авторский запрос и запросы секции остаются
    в пуле, бриф только добавляет свой.

    Тест проверяет УСТРОЙСТВО, а не написание. Первая версия сверяла
    буквальную строку `_bq = brief_to_stock_query(shot_brief` — и упала от
    переименования переменной, не найдя ни одного дефекта: ровно тот класс
    пустой проверки, который в этом репозитории уже ломал три теста рядом
    с `fix_pauses` («считать надо то, ради чего тест написан»). Теперь имя
    переменной свободно, а инвариант — нет: пул обязан НАЧИНАТЬСЯ с
    авторского запроса и запросов секции, добавка обязана быть именно
    добавкой (`pool_queries = [X] + pool_queries`), и X обязан выводиться
    из брифа, а не из чего-нибудь ещё."""
    import re
    src = open(os.path.join(REPO, "scripts", "pipeline_smart.py"),
               encoding="utf-8").read()
    start = src.index("pool_queries = [query]")
    block = src[start:start + 1500]
    # 1. Авторский запрос и запросы секции остаются основой пула.
    assert re.search(r"pool_queries = \[query\] \+ \[q for q in \(extra_queries",
                     block)
    # 2. Бриф ДОБАВЛЯЕТСЯ в начало, а не заменяет собой пул.
    m = re.search(r"pool_queries = \[(\w+)\] \+ pool_queries", block)
    assert m, "запрос из брифа обязан именно ДОБАВЛЯТЬСЯ к пулу"
    # 3. И добавляется именно запрос из брифа, а не что-нибудь ещё.
    #    ПРОВЕРКА ПО ПОВЕДЕНИЮ, а не по написанию — см.
    #    test_photo_path_asks_the_stock_with_a_plain_query ниже. Прежняя
    #    редакция искала здесь строку `<var> = brief_to_stock_query(shot_brief`
    #    и была ЗЕЛЁНОЙ ПО ПОСТРОЕНИЮ: единственное совпадение в файле — цитата
    #    старой формулы внутри ДОКСТРИНГА candidate_brief_key(), где она
    #    приведена как пример уже исправленного бага. Тест не мог упасть и
    #    честно проспал настоящий дефект (в пул уходило имя кэш-файла).
    var = m.group(1)
    assert re.search(re.escape(var) + r"\s*(,\s*\w+\s*)?=\s*"
                     r"(brief_to_stock_query|candidate_brief_keys)\(shot_brief", src), (
        "запрос из брифа обязан выводиться из самого брифа")


def test_photo_path_asks_the_stock_with_a_plain_query(tmp_path, monkeypatch):
    """У фото-пути этой проверки не было, и ровно поэтому дефект дожил до
    разбора 17.09: в `pool_queries` клался КЛЮЧ КЭША, а он с 15.09 несёт
    хвост `|shelf:<md5>` — и эта строка уходила в поиск ВСЕХ словесных
    источников. Живой прогон до правки:

        ('pexels',  'medieval dented steel breastplate|shelf:953c23ac')
        ('pixabay', 'medieval dented steel breastplate|shelf:953c23ac')

    У Pixabay И-логика, у остальных свободный текст — токен-хэш обнуляет
    выдачу. Тест смотрит на то, ЧТО реально ушло в источник, а не на то,
    как называется переменная."""
    import pipeline_smart as ps

    seen = []
    # Хвост `|shelf:` появляется в ключе ТОЛЬКО когда полка реально отвечает,
    # а `tests/conftest.py` гасит SHELF_INDEX (иначе каждый тест тянул бы
    # модель на 4.3 ГБ). Без этой строки тест зелёный по построению — первая
    # его редакция такой и была, и контрольный прогон это поймал: дефект
    # вернули, тест прошёл. Проверяется именно тот режим, в котором дефект
    # живёт, — полка на диске есть.
    monkeypatch.setattr(ps, "_shelf_question_active", lambda: True)
    monkeypatch.setattr(ps, "TEMP_FOLDER", str(tmp_path))
    for name in ("_pexels_search_photos", "_pixabay_search_photos",
                 "_unsplash_search_photos", "_openverse_search_photos"):
        monkeypatch.setattr(ps, name, lambda q, _n=name, **k: seen.append(q) or [])
    monkeypatch.setattr(ps, "_museum_search_photos", lambda q, **k: seen.append(q) or [])
    # Полка отвечает на бриф ЦЕЛИКОМ и к словесным источникам отношения не
    # имеет — её вопрос в этой проверке не участвует.
    monkeypatch.setattr(ps, "_shelf_search_photos", lambda q, **k: [])

    brief = "a dented steel breastplate, close up"
    ps.pexels_photo("medieval breastplate", 3, used_ids=set(), used_hashes=[],
                    shot_brief=brief, block_text="Стрела скользнула по нагруднику.")

    assert seen, "фото-путь вообще не спросил ни один источник — тест не про то"
    bad = [q for q in seen if "|" in q or "shelf:" in q]
    assert not bad, f"в поиск ушёл ключ кэша, а не запрос: {bad[:3]}"
    brief_q = ps.brief_to_stock_query(brief, fallback=None)
    assert brief_q and any(brief_q in q for q in seen), (
        f"запрос из брифа не дошёл до источников: {seen[:3]}")


def test_brief_is_in_the_candidate_cache_key():
    """Бриф меняет состав пула — без него прогретый temp_smart/ отдал бы
    кандидата, выбранного до появления брифа.

    Ключ считает candidate_brief_key() (15.09): прежняя формула клала в
    ключ только СТОКОВЫЙ ПЕРЕВОД брифа, и два брифа, отличающиеся ракурсом,
    давали одно имя файла при разных вопросах к полке (замер: 3 совпадения
    из 3). Инвариант этого теста прежний и стал строже — проверяется он
    теперь по резолверу, а не по букве старой строки."""
    assert ps.candidate_brief_keys("a dented steel breastplate, close up")[1] != ""
    src = open(os.path.join(REPO, "scripts", "pipeline_smart.py"),
               encoding="utf-8").read()
    start = src.index("_brief_key = candidate_brief_key")
    assert "[_brief_key] if _brief_key else []" in src[start:start + 600]


def test_change_is_in_the_selection_signature():
    src = open(os.path.join(REPO, "scripts", "pipeline_smart.py"),
               encoding="utf-8").read()
    start = src.index("def _selection_stack_signature")
    block = src[start:src.index("\ndef candidate_gate_signature", start)]
    assert "BRIEF_STOCK_QUERY_VERSION" in block
    assert isinstance(ps.BRIEF_STOCK_QUERY_VERSION, int)


@pytest.mark.parametrize("brief,must", [
    ("an armoured foot in a steel sabaton standing on bare earth", "sabaton"),
    ("a war hammer head of blunt steel", "hammer"),
    ("a human skull from an archaeological excavation", "skull"),
    ("a medieval military camp of tents", "camp"),
])
def test_the_subject_of_the_shot_reaches_the_query(brief, must):
    assert must in _q(brief)


# ------------------------------------------- бриф доходит и до видео-пути

def test_video_path_asks_the_stock_with_the_brief_too(tmp_path, monkeypatch):
    """Асимметрия из того же класса, что уже дважды стоила этому
    репозиторию половины эпизода: `filter_alt_blocklist()` жила только в
    `pexels_photo()` и на видео не вызывалась НИ РАЗУ, `director_score_fn`
    поднимал только фото. Здесь было то же — 142 брифа эпизода влияли на
    фото-слоты и не влияли на видео, при том что видео это примерно
    половина слотов."""
    import pipeline_smart as ps

    seen = []
    monkeypatch.setattr(ps, "TEMP_FOLDER", str(tmp_path))
    monkeypatch.setattr(ps, "_pexels_search_videos",
                        lambda q, **k: seen.append(q) or [])
    monkeypatch.setattr(ps, "_pixabay_search_videos", lambda q, **k: [])

    ps.pexels_video("medieval battle", 7, used_ids=set(), used_hashes=[],
                    extra_queries=["medieval camp"],
                    shot_brief="a dented steel breastplate, close up")

    assert seen, "видео-путь вообще не спросил сток — тест не про то"
    brief_q = ps.brief_to_stock_query("a dented steel breastplate, close up",
                                      fallback=None)
    assert brief_q and brief_q in seen
    # И именно ПЕРВЫМ: кандидаты чередуются между запросами, перебирается
    # лишь VIDEO_RELEVANCE_MAX_TRIES штук, поэтому позиция решает при
    # равенстве.
    assert seen[0] == ps.apply_action_qualifier(
        ps.disambiguate_search_query(brief_q), None)


def test_video_cache_key_moves_with_the_brief(tmp_path, monkeypatch):
    """На прогретом temp_smart/ кэш-хит делает continue ДО переподбора —
    без брифа в ключе правка брифа не дошла бы до экрана вообще."""
    import pipeline_smart as ps

    paths = []
    monkeypatch.setattr(ps, "TEMP_FOLDER", str(tmp_path))
    monkeypatch.setattr(ps, "_pexels_search_videos", lambda q, **k: [])
    monkeypatch.setattr(ps, "_pixabay_search_videos", lambda q, **k: [])
    real_join = os.path.join

    def spy(*parts):
        p = real_join(*parts)
        if p.endswith(".mp4"):
            paths.append(os.path.basename(p))
        return p
    monkeypatch.setattr(ps.os.path, "join", spy)
    for brief in ("a dented steel breastplate, close up",
                  "a manuscript illumination of a battle"):
        ps.pexels_video("medieval battle", 7, used_ids=set(), used_hashes=[],
                        shot_brief=brief)
    assert len(set(paths)) == len(paths) >= 2, paths


def test_wrong_word_is_not_injected_when_the_real_anchor_is_just_a_variant(monkeypatch):
    """РЕАЛЬНЫЙ найденный баг (19.09, живая жалоба на готовом рендере):
    «Конница мчится через поле прямо на пехоту» получала запрос к стоку
    `dagger armored knights warhorses galloping» — кинжал в начале запроса
    сцены, где о нём ни слова.

    Причина — `w in era` сравнивал ТОКЕН БРИФА со словом якоря БУКВА В
    БУКВУ: «knights» (множественное число) не равно «knight», «armored»
    (американское написание автора) не равно «armour» (якорь канала).
    Хотя нужное слово физически стояло в брифе, код решал, что якоря нет
    вообще, и подставлял ПЕРВОЕ слово списка якорей («dagger») — только
    потому, что оно первое, а не потому, что относится к сцене.

    Якоря монкипатчены на РЕАЛЬНЫЙ список эпизода, где баг найден —
    `content_world.json` этого конкретного эпизода объявлял только
    единственную (не множественную) и только одну (не обе) форму
    написания, ровно как содержательно и пишет модель, заполняющая
    авто-нишу. Продовый `channel_profile.json` этого репозитория к тому
    моменту уже перечислял "knight"/"knights"/"armor"/"armour" отдельными
    строками — обходной путь, замаскировавший бы регрессию именно для
    этого канала, но не для нового эпизода с чистым авто-профилем."""
    monkeypatch.setattr(ps, "OPENVERSE_ERA_ANCHORS",
                         ("dagger", "sword", "armour", "knight", "cavalry",
                          "battlefield"))
    q = ps.brief_to_stock_query(
        "armored knights on warhorses galloping across a muddy field "
        "towards pikemen",
        fallback="medieval cavalry charge field",
    )
    assert "dagger" not in q.split(), q
    assert "knights" in q.split(), q

    q2 = ps.brief_to_stock_query(
        "a steel gauntlet stuck in mud, unable to lift heavy armor",
        fallback="medieval breastplate arrow",
    )
    assert "dagger" not in q2.split(), q2
    # Якорь физически ЕСТЬ в брифе (armor), просто дальше окна усечения —
    # честнее поднять его словоформу, чем вставить чужое слово.
    assert "armor" in q2.split(), q2


def test_era_anchor_lookup_is_bidirectional_on_spelling():
    """`QUERY_TERM_SPELLINGS` ключуется американским написанием
    ("armor" -> ("armor", "armour")); спросить его британским ("armour",
    ровно то написание, которым этот канал объявляет якорь эпохи) `.get()`
    раньше находил только `("armour",)` и не видел «armor» в брифе автора.
    `_spelling_forms()` обязана отвечать одинаково с любой стороны."""
    assert ps._spelling_forms("armour") == ps._spelling_forms("armor")
    assert ps.query_mentions_term("a suit of plate armor", "armour")
    assert ps.query_mentions_term("a suit of plate armour", "armor")


def test_unrelated_word_is_not_forced_when_the_brief_has_its_own_era_word(monkeypatch):
    """РЕАЛЬНЫЙ живой баг (20.09, слот 8 медиевал-теста, жалоба владельца
    на контактном листе): «Стрела скользит по нагруднику» получила бриф
    `a steel arrowhead scraping and deflecting off a curved steel
    breastplate» — ни слова про кинжал. Но узкий словарь якорей ЭТОГО
    эпизода (`content_world.json`, объявлен confidence 0.95) — всего шесть
    предметных слов («dagger», «sword», «armour», «knight», «cavalry»,
    «battlefield»), и ни одно из них не встречается в брифе про стрелу.
    Код решал «якоря нет вообще» и слепо форсировал ПЕРВОЕ слово списка —
    «dagger» — в начало запроса: `dagger steel arrowhead scraping
    deflecting`. Сток честно нашёл нож вместо стрелы.

    `arrowhead` и `breastplate` сами по себе однозначно средневековые
    термины — просто не входят в куцый список ЭТОГО эпизода. Детекция
    якоря теперь смотрит и в широкий исторический словарь
    (`_QUERY_ERA_ANCHORS_DEFAULT`), где оба слова есть, поэтому кинжал не
    подставляется вообще: своих предметных слов в брифе достаточно."""
    monkeypatch.setattr(ps, "OPENVERSE_ERA_ANCHORS",
                         ("dagger", "sword", "armour", "knight", "cavalry",
                          "battlefield"))
    q = ps.brief_to_stock_query(
        "a steel arrowhead scraping and deflecting off a curved steel "
        "breastplate",
        fallback="dagger steel arrowhead scraping deflecting",
    )
    assert "dagger" not in q.split(), q
    assert "arrowhead" in q.split(), q


def test_present_anchor_is_not_dropped_by_the_word_cap(monkeypatch):
    """РЕАЛЬНЫЙ живой баг (20.09, слот 3 медиевал-теста): «Но именно он
    решал исход поединка, когда меч уже бесполезен» получила бриф `a
    broken longsword lying in the mud next to a drawn dagger» — «dagger»
    физически стоит в брифе, это самое важное слово фразы. Но слово
    «longsword» само содержит «sword» и засчитывалось якорем уже на
    второй позиции — проверка «есть ли якорь в начале» успокаивалась и
    дальше не смотрела, а лимит 5 слов срезал «dagger» (шестое слово) без
    следа. Итоговый запрос `broken longsword lying mud drawn» не содержал
    ни одного слова про кинжал, и сток честно нашёл лежащего в траве
    зубра по слову «lying».

    Теперь собираются ВСЕ слова-якоря брифа, и обрубка обязана оставить их
    все — при нехватке места вытесняются не-якорные слова хвоста, а не
    единственный найденный якорь."""
    monkeypatch.setattr(ps, "OPENVERSE_ERA_ANCHORS",
                         ("dagger", "sword", "armour", "knight", "cavalry",
                          "battlefield"))
    q = ps.brief_to_stock_query(
        "a broken longsword lying in the mud next to a drawn dagger",
        fallback="broken longsword lying mud drawn",
    )
    assert "dagger" in q.split(), q
    assert len(q.split()) <= ps.BRIEF_STOCK_QUERY_MAX_WORDS, q
