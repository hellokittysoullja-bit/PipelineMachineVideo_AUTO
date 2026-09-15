# -*- coding: utf-8 -*-
"""[shot:...] — описание кадра, написанное рядом со своей фразой.

Держим три вещи, каждая из которых уже ломалась в этом репозитории на
других тегах: бриф не уезжает в заказ TTS, не режет блок и не приклеивается
к соседней фразе.
"""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import script_parser as sp  # noqa: E402


def _blocks(tmp_path, text):
    p = tmp_path / "script.txt"
    p.write_text(text, encoding="utf-8")
    return sp.parse_blocks(str(p))


def test_brief_attaches_to_its_own_phrase(tmp_path):
    blocks = _blocks(tmp_path, """=== HOOK ===
[shot:a knight lying face down in churned mud] И всё равно ты лежишь лицом в грязи. [pause] [shot:a two-handed European sword, the full blade] Возьми настоящий боевой меч. [pause] Ничего не произойдёт.
""")
    assert len(blocks) == 3
    assert blocks[0]["shot_brief"] == "a knight lying face down in churned mud"
    assert blocks[1]["shot_brief"] == "a two-handed European sword, the full blade"
    # У фразы без брифа он обязан быть пустым, а не унаследованным от соседа:
    # бриф описывает КОНКРЕТНЫЙ кадр, и молчаливое наследование вернуло бы
    # ровно ту болезнь, из-за которой один запрос секции обслуживал десять
    # слотов.
    assert blocks[2]["shot_brief"] is None


def test_brief_never_reaches_the_tts_order(tmp_path):
    """Литеральный тег в заказе уже стоил эпизоду PHRASE LOCK (замер 14.09
    на [sfx:]). Бриф длиннее их всех — цена ошибки здесь выше."""
    blocks = _blocks(tmp_path, """=== HOOK ===
[shot:a full suit of plate armour standing in a dark museum hall] Доспех отработал каждую монету.
""")
    assert "shot:" not in blocks[0]["text"]
    assert "[" not in blocks[0]["text"]
    assert blocks[0]["text"] == "Доспех отработал каждую монету."
    assert "shot" not in sp.strip_pipeline_only_tags("A [shot:foo bar baz] B")


def test_brief_is_in_the_single_pipeline_only_registry():
    """Вторая копия словаря тегов уже один раз выключила PHRASE LOCK на
    целый эпизод — тег обязан жить в общем регекспе, а не в своей копии."""
    assert sp.PIPELINE_ONLY_TAG_RE.search("[shot:anything at all]")
    assert sp.strip_pipeline_only_tags("x [shot:y] z").strip() == "x   z".strip()


def test_brief_does_not_split_the_block(tmp_path):
    """Тег в СЕРЕДИНЕ фразы не имеет права породить лишний монтажный рез:
    ровно этот дефект был измерен у [sfx:] (три блока превращались в два)."""
    blocks = _blocks(tmp_path, """=== HOOK ===
Возьми настоящий боевой меч [shot:a plain steel blade, macro] и ударь им человека в доспехе.
""")
    assert len(blocks) == 1
    assert blocks[0]["shot_brief"] == "a plain steel blade, macro"
    assert blocks[0]["text"] == "Возьми настоящий боевой меч и ударь им человека в доспехе."


def test_empty_brief_is_ignored(tmp_path):
    blocks = _blocks(tmp_path, """=== HOOK ===
[shot:] Пустой бриф не считается брифом.
""")
    assert blocks[0]["shot_brief"] is None
    assert "[" not in blocks[0]["text"]


def test_subcut_inherits_the_brief_of_its_phrase():
    """Под-кадр — та же фраза, разрезанная по длительности. Бриф обязан
    ехать с ним: иначе половина длинной фразы искала бы вслепую."""
    import pipeline_smart as ps
    b = {"text": " ".join(f"слово{i}" for i in range(60)),
         "pause_after": 0.0, "words": 60, "section": "HOOK",
         "stat": None, "stat_word_pos": None, "is_climax": False,
         "sfx": [], "hush": False, "shot_brief": "a halberd on a dark background"}
    out, _w = ps.split_long_blocks([b], [60.0])
    assert len(out) > 1, "блок не разрезался — тест не про то"
    assert all(x.get("shot_brief") == "a halberd on a dark background" for x in out)


def test_shelf_gets_the_brief_stocks_get_the_query():
    """Развилка, ради которой всё сделано. Полка сравнивает описание с
    изображениями — ей нужен полный бриф; у стоков текстовый API, где каждое
    лишнее слово сужает выдачу, им по-прежнему уходит короткий запрос.

    Развилка не изменилась (15.09): полке по-прежнему уходит бриф, стокам —
    короткий запрос. Добавилось только то, что при ОТСУТСТВИИ брифа полка
    получает фразу блока вместо запроса секции, который делят 6-10 слотов.
    Функциональная проверка самого приоритета живёт рядом, в
    tests/test_shelf_question.py (там pipeline_smart уже импортирован);
    здесь — что развилка источников на месте."""
    src = open(os.path.join(REPO, "scripts", "pipeline_smart.py"),
               encoding="utf-8").read()
    start = src.index("for source_name, fetch in (")
    block = src[start:start + 6000]
    assert 'fetch(pq, brief=shelf_question(shot_brief, block_text) or None)' in block
    assert 'fetch(api_q)' in block


@pytest.mark.parametrize("tag", ["[stat:5 кг]", "[climax]", "[sfx:armour_clank]",
                                 "[hush]", "[shot:a sword]"])
def test_every_pipeline_only_tag_is_stripped_by_one_function(tag):
    assert sp.strip_pipeline_only_tags(f"до {tag} после").count("[") == 0
