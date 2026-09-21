#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Паспорт мира реально доходит до запроса — иначе он был бы файлом,
который никто не читает (шестой случай класса «слой есть, его никто не
зовёт» в этом репозитории, см. tests/test_no_dead_layers.py).

Проверяется ТРИ вещи, и каждая проверена контрольным прогоном со снятой
правкой (тесты падают):
  1. без паспорта поведение байт-в-байт прежнее;
  2. с паспортом чужой ниши средневековый якорь НЕ подставляется;
  3. паспорт входит в подпись отбора — иначе на прогретом temp_smart/
     смена мира не дошла бы до экрана.
"""
import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

import pipeline_smart as ps  # noqa: E402
import world_card as wc  # noqa: E402

PSY_BRIEF = "a phone lying face down on a bedside table at night"


@pytest.fixture
def episode(tmp_path):
    """Пустая папка эпизода + сброшенный кэш паспорта на каждый тест."""
    ps.reset_world_card_cache()
    d = tmp_path / "ep"
    (d / "media_plan").mkdir(parents=True)
    yield d
    ps.reset_world_card_cache()


def _psy_card():
    return {
        "schema_version": 1,
        "register": "modern",
        "era": None,
        "culture": {"include": [], "exclude": []},
        "must_not_show": ["plate armour", "castle"],
        "expected_subjects": ["phone face down"],
        "era_anchor_terms": ["contemporary", "everyday"],
    }


def _write(episode, card):
    with open(wc.path(str(episode)), "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False)


def test_without_card_behaviour_is_unchanged(episode, monkeypatch):
    """Эпизод без паспорта обязан получить РОВНО тот запрос, что и до
    правки: якорь канала, а не паспорта."""
    monkeypatch.setattr(ps, "VIDEO_FOLDER", str(episode))
    monkeypatch.setattr(ps, "OPENVERSE_ERA_ANCHORS", ("medieval", "knight"))
    assert ps.episode_world_card() is None
    got = ps.brief_to_stock_query(PSY_BRIEF, fallback="x y")
    assert got.split()[0] == "medieval", got


def test_card_of_another_niche_kills_the_medieval_anchor(episode, monkeypatch):
    """Измеренный в CLAUDE.md случай: бриф про телефон на тумбочке уезжал в
    сток как `medieval phone lying face down`. С КАЖДЫМ запросом."""
    monkeypatch.setattr(ps, "VIDEO_FOLDER", str(episode))
    monkeypatch.setattr(ps, "OPENVERSE_ERA_ANCHORS", ("medieval", "knight"))
    _write(episode, _psy_card())
    got = ps.brief_to_stock_query(PSY_BRIEF, fallback="x y")
    assert "medieval" not in got, got
    assert "knight" not in got, got
    assert got.split()[0] == "contemporary", got


def test_prehistoric_card_anchors_to_its_own_era(episode, monkeypatch):
    """Требование владельца дословно: ниша может быть любой, в том числе
    доисторической. Якорь берётся из эпизода, а не из кода."""
    monkeypatch.setattr(ps, "VIDEO_FOLDER", str(episode))
    monkeypatch.setattr(ps, "OPENVERSE_ERA_ANCHORS", ("medieval",))
    _write(episode, {
        "schema_version": 1, "register": "historical",
        "era": {"from": -300000, "to": -10000},
        "culture": {"include": ["neanderthal"], "exclude": []},
        "must_not_show": ["metal tools"], "expected_subjects": ["flint scraper"],
        "era_anchor_terms": ["paleolithic", "stone age"],
    })
    got = ps.brief_to_stock_query("a flint scraper held in a hand", fallback="x y")
    assert got.split()[0] == "paleolithic", got
    assert "medieval" not in got


def test_lint_uses_episode_anchors_not_the_hardcoded_medieval_list(episode, monkeypatch, capsys):
    """Линт авторских запросов обязан сверяться с миром ЭПИЗОДА.

    Проверяем РАЗНИЦУ поведения, а не факт печати: запрос
    `medieval knight armour` в психологическом эпизоде якоря этого мира не
    несёт и обязан попасть в список, тогда как по зашитому в код
    средневековому словарю он выглядел бы образцовым. Без паспорта тот же
    запрос не должен попадать в список вовсе.
    """
    monkeypatch.setattr(ps, "VIDEO_FOLDER", str(episode))
    queries = {"HOOK": ["medieval knight armour"]}

    ps.lint_authored_queries(queries)
    without_card = capsys.readouterr().out
    assert "без якоря" not in without_card, (
        "по словарю канала средневековый запрос якорь имеет — "
        "значит до правки линт молчал, и это правильно")

    _write(episode, _psy_card())
    ps.reset_world_card_cache()
    ps.lint_authored_queries(queries)
    with_card = capsys.readouterr().out
    assert "без якоря" in with_card and "medieval knight armour" in with_card, (
        "линт не заметил, что запрос чужд миру этого эпизода")


def test_card_enters_selection_signature(episode, monkeypatch):
    """Без этого смена паспорта не дошла бы до экрана на прогретом
    temp_smart/: кэш-хит клипа делает continue ДО переподбора кандидата."""
    monkeypatch.setattr(ps, "VIDEO_FOLDER", str(episode))
    monkeypatch.setattr(ps, "OPENVERSE_ERA_ANCHORS", ("medieval",))
    before = ps._selection_stack_signature()
    _write(episode, _psy_card())
    ps.reset_world_card_cache()
    after = ps._selection_stack_signature()
    assert before != after, "паспорт мира не влияет на подпись отбора"


def test_broken_card_stops_work_instead_of_silently_disabling_the_world(episode, monkeypatch):
    """Сломанный паспорт — крик, а не тишина: на паспорте держится приёмка
    кадра, и «молча считать, что мира нет» значит «молча выключить защиту»."""
    monkeypatch.setattr(ps, "VIDEO_FOLDER", str(episode))
    with open(wc.path(str(episode)), "w", encoding="utf-8") as f:
        f.write('{"schema_version": 1, "register": "нет такого"}')
    with pytest.raises(wc.WorldCardError):
        ps.episode_world_card()


def test_real_episode_card_on_disk_is_valid():
    """Паспорт эпизода 02 лежит в репозитории и обязан оставаться годным:
    это первый настоящий паспорт, и на нём стоят замеры этой правки.

    `videos/` целиком в .gitignore, поэтому паспорт добавлен force-add,
    как и сам script.txt. Если папки эпизода нет (чужая машина, чистый
    клон без медиа) — тест пропускается, а не падает: проверять нечего.
    """
    d = os.path.join(REPO_ROOT, "videos", "02_ne-mechom")
    if not os.path.exists(wc.path(d)):
        pytest.skip("эпизод 02 не выложен в этом клоне")
    card = wc.load(d)
    assert card is not None, "паспорт эпизода 02 пропал"
    assert wc.era_window(card) == (1290, 1480)
    anchors = wc.era_anchors(card)
    assert anchors[0] == "medieval"
    # Эпизод сознательно смешанный: в нём есть и современная археология
    # (раскопки Таутона 1996, судебная медицина), и бытовые сравнения мифа.
    assert card["modern_props_allowed"] is True
    # Запреты названы КОНКРЕТНЫМИ классами, а не «современностью вообще» —
    # общий запрет убил бы законный кадр раскопок.
    forb = " ".join(wc.forbidden_classes(card))
    assert "reenactment" in forb and "sport fencing" in forb
    assert "modern people" not in wc.forbidden_classes(card)


def test_episode_culture_exclude_catches_the_roman_legionaries(episode, monkeypatch):
    """Найдено ГЛАЗАМИ на готовом ролике 02 (17.09): в эпизоде про Азенкур
    на экране стоят римские легионеры. Правильный континент, не то
    тысячелетие — домен-гвард спрашивает «европейский клинок или
    азиатский», и по этой оси Рим европейский.

    Слаг взят настоящий, из CLAUDE.md (живая выдача Pexels по 30 запросам
    эпизода): `roman-soldiers-historical-reenactment-event-38103939`.
    Синтетический слаг не воспроизвёл бы ни отсутствие `alt` у видео, ни
    дефисы вместо пробелов.
    """
    monkeypatch.setattr(ps, "VIDEO_FOLDER", str(episode))
    monkeypatch.setattr(ps, "CONTENT_ALT_BLOCKLIST", ("katana", "samurai"))
    items = [
        {"id": 1, "alt": None, "url":
         "https://www.pexels.com/video/roman-soldiers-historical-reenactment-event-38103939/"},
        {"id": 2, "alt": "medieval plate armour on a stand", "url":
         "https://www.pexels.com/photo/plate-armour-1/"},
    ]
    # Без паспорта слово `roman` не запрещено никем — и это ПРАВИЛЬНО:
    # для эпизода про Римскую империю запрет был бы запретом на предмет
    # разговора. Чужая культура — свойство эпизода, не канала.
    assert len(ps.filter_alt_blocklist(items)) == 2

    _write(episode, {
        "schema_version": 1, "register": "historical",
        "era": {"from": 1290, "to": 1480},
        "culture": {"include": ["english", "french"],
                    "exclude": ["roman", "viking"]},
        "must_not_show": ["sport fencing"],
        "expected_subjects": ["plate armour"],
        "era_anchor_terms": ["medieval"],
    })
    ps.reset_world_card_cache()
    kept = ps.filter_alt_blocklist(items)
    assert [p["id"] for p in kept] == [2], (
        "римские легионеры прошли текстовый фильтр при паспорте, который "
        "прямо исключает roman")


def test_multiword_anchor_is_matched_as_a_phrase(episode, monkeypatch):
    """Своя же ошибка, поймана ЗАМЕРОМ на 142 брифах эпизода 02 (17.09).

    Паспорт объявляет якорями и фразы (`plate armour`,
    `manuscript illumination`), а проверка сравнивала СЛОВА запроса с
    множеством якорей — двухсловный якорь не совпадал ни с чем никогда.
    Следствие: бриф, уже называющий эпоху, считался неякоренным, и ему
    сверху приписывался лишний `medieval`, вытесняя слово автора из лимита
    в пять слов. Так молча менялись 17 запросов из 142.
    """
    monkeypatch.setattr(ps, "VIDEO_FOLDER", str(episode))
    _write(episode, {
        "schema_version": 1, "register": "historical",
        "era": {"from": 1290, "to": 1480},
        "culture": {"include": ["english"], "exclude": []},
        "must_not_show": [], "expected_subjects": [],
        "era_anchor_terms": ["plate armour", "manuscript illumination"],
    })
    ps.reset_world_card_cache()
    got = ps.brief_to_stock_query(
        "a full plate armour harness on a stand", fallback="x y")
    assert got.startswith("full plate armour"), got
    assert "medieval" not in got

    # А бриф БЕЗ якоря обязан якорь получить — и односложным, чтобы не
    # съесть два слова из пяти.
    _write(episode, {
        "schema_version": 1, "register": "historical",
        "era": {"from": 1290, "to": 1480},
        "culture": {"include": ["english"], "exclude": []},
        "must_not_show": [], "expected_subjects": [],
        "era_anchor_terms": ["plate armour", "medieval"],
    })
    ps.reset_world_card_cache()
    got = ps.brief_to_stock_query("wet churned earth underfoot", fallback="x y")
    assert got.split()[0] == "medieval", got


# --- ЧЕТВЁРТАЯ ВЕЩЬ (21.09): must_not_show доходит до ВЕТО, а не только до
# запроса. До этого дня `world_card.forbidden_classes()` не вызывал НИКТО —
# прямой grep по scripts/ давал пусто, при том что её докстринг обещает
# «уходит в вопрос приёмки кадра». Цена была измерена на реальном эпизоде:
# паспорт дословно запрещал `modern tactical knife` и `napoleonic uniform`,
# и ровно это стояло на экране (нож на фразу «Вот кинжал», наполеоновский
# гусар на «Конница мчится»), потому что канальные 8 ловушек — про
# современность и культуру клинка, а весь брак был про ЭПОХУ.

def test_forbidden_classes_reach_the_veto(episode, monkeypatch):
    """Ловушки паспорта обязаны попадать в список, по которому судит вето."""
    monkeypatch.setattr(ps, "VIDEO_FOLDER", str(episode))
    _write(episode, _psy_card())
    ps.reset_world_card_cache()
    anchors = ps.episode_forbidden_anchors(str(episode))
    assert "plate armour" in anchors and "castle" in anchors, anchors


def test_veto_judges_against_channel_plus_episode(episode, monkeypatch):
    """negative_anchor_violation() судит по СУММЕ: канал + паспорт эпизода.

    Ловим сам факт подмешивания (какие тексты ушли в модель), а не вердикт:
    вердикт зависит от весов модели, а состав списка — от провода, который
    и был оборван.
    """
    monkeypatch.setattr(ps, "VIDEO_FOLDER", str(episode))
    _write(episode, _psy_card())
    ps.reset_world_card_cache()
    seen = {}

    def fake_multi(path, texts):
        seen["texts"] = list(texts)
        return [0.5] + [0.0] * (len(texts) - 1)   # цель уверенно выигрывает

    monkeypatch.setattr(ps, "clip_relevance_multi", fake_multi)
    monkeypatch.setattr(ps, "NEGATIVE_VETO_ENABLED", True)
    rejected, _ = ps.negative_anchor_violation("x.jpg", "some query")
    assert rejected is False
    assert seen["texts"][0] == "some query"
    for a in ps.CONTENT_NEGATIVE_ANCHORS:
        assert a in seen["texts"], "канальные ловушки пропали из вето"
    assert "plate armour" in seen["texts"], "ловушки паспорта не дошли до вето"


def test_no_card_keeps_veto_list_byte_identical(episode, monkeypatch):
    """Нет паспорта — список ловушек ровно прежний, ни одной лишней строки."""
    monkeypatch.setattr(ps, "VIDEO_FOLDER", str(episode))
    ps.reset_world_card_cache()
    assert ps.episode_forbidden_anchors(str(episode)) == ()
    seen = {}

    def fake_multi(path, texts):
        seen["texts"] = list(texts)
        return [0.5] + [0.0] * (len(texts) - 1)

    monkeypatch.setattr(ps, "clip_relevance_multi", fake_multi)
    monkeypatch.setattr(ps, "NEGATIVE_VETO_ENABLED", True)
    ps.negative_anchor_violation("x.jpg", "q")
    assert seen["texts"] == ["q"] + list(ps.CONTENT_NEGATIVE_ANCHORS)


def test_broken_card_does_not_kill_selection(episode, monkeypatch):
    """Сломанный паспорт не должен ронять отбор ЧЕРЕЗ вето: гейт обязан
    остаться в прежнем (канальном) составе, а не выбросить исключение
    посреди подбора кадра."""
    monkeypatch.setattr(ps, "VIDEO_FOLDER", str(episode))
    with open(wc.path(str(episode)), "w", encoding="utf-8") as f:
        f.write("{ это не json")
    ps.reset_world_card_cache()
    assert ps.episode_forbidden_anchors(str(episode)) == ()
