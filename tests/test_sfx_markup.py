# -*- coding: utf-8 -*-
"""Разметка звука в сценарии: [sfx:концепт] и [hush].

Смысл слоя — не разбирать готовый текст, а записать намерение в тот момент,
когда автор этот костёр в текст и вписывает. Обе автоматические схемы были
проверены живьём и отклонены с числами: русский текст главы в текстовую
башню CLAP дал ОДИН И ТОТ ЖЕ вид на всех шести главах эпизода, а мост
«многоязычный эмбеддинг -> английская метка» — 7 верных из 18 на одиночных
словах при случайном выборе 1 из 20. Разметка в сценарии даёт точность по
построению: угадывать нечего.
"""
import os
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import script_parser


def parse(text):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     encoding="utf-8") as f:
        f.write("=== METADATA ===\nTITLE: t\n\n=== HOOK ===\n" + text + "\n")
        path = f.name
    try:
        return script_parser.parse_blocks(path)
    finally:
        os.unlink(path)


def test_sfx_tag_records_name_and_word_anchor():
    b = parse("Кузнец бьёт молотом по раскалённой полосе[sfx:hammer], и это слышно.")
    hit = [x for x in b if x["sfx"]]
    assert len(hit) == 1
    assert hit[0]["sfx"][0]["name"] == "hammer"
    # якорь — сколько слов уже произнесено к моменту тега
    assert hit[0]["sfx"][0]["word_pos"] == 6


def test_sfx_tag_does_not_cut_the_edit():
    """Реальный дефект первой версии: тег разбивает строку на части, и если
    висела несъеденная пауза, следующий огрызок — хоть одна точка — уходил
    в ОТДЕЛЬНЫЙ блок, то есть получал собственный слот под картинку."""
    b = parse("Первая.[pause]Дождь идёт третьи сутки[sfx:rain].")
    assert len(b) == 2, [x["text"] for x in b]
    assert b[1]["text"].startswith("Дождь идёт третьи сутки")
    assert not any(x["text"].strip() in {".", ",", ""} for x in b)


def test_several_sounds_in_one_block_all_survive():
    b = parse("Костёр горит[sfx:fire], и дождь стучит по крыше[sfx:rain].")
    hit = [x for x in b if x["sfx"]][0]
    assert [s["name"] for s in hit["sfx"]] == ["fire", "rain"]
    assert hit["sfx"][0]["word_pos"] < hit["sfx"][1]["word_pos"]


def test_hush_marks_deliberate_silence():
    """[hush] — не «здесь нет звука», а «здесь тишина НУЖНА». Пустое место
    планировщик вправе заполнить, помеченную тишину — нет."""
    b = parse("Первая фраза.[pause]А теперь молчи.[hush][pause]Третья.")
    marked = [x for x in b if x["hush"]]
    assert len(marked) == 1
    assert "молчи" in marked[0]["text"]


def test_sfx_never_reaches_the_tts_text():
    """Тот же принцип, что у [stat:...] и [climax]: пайплайн-only маркер,
    человек копирует ЧИСТЫЙ текст в TTS."""
    b = parse("Костёр догорает[sfx:fire] всю ночь.")
    joined = " ".join(x["text"] for x in b)
    assert "sfx" not in joined and "[" not in joined and "hush" not in joined


def test_unknown_tags_still_warn(capsys):
    """[sfx:]/[hush] не должны заглушить предупреждение о запрещённых тегах
    (ЧАСТЬ 10: [long pause] ломает TTS артефактами)."""
    parse("Фраза[sfx:fire] и ещё[long pause] одна.")
    assert "long pause" in capsys.readouterr().out


@pytest.mark.parametrize("bad", ["[sfx:]", "[sfx: ]"])
def test_empty_sfx_name_is_ignored(bad):
    b = parse("Просто фраза" + bad + " дальше.")
    assert not any(x["sfx"] for x in b)


def test_existing_markup_not_regressed():
    b = parse("Первая.[pause]Вторая[stat:ЧИСЛО].[pause]Третья.[climax][pause]Четвёртая.")
    assert any(x["stat"] == "ЧИСЛО" for x in b)
    assert any(x["is_climax"] for x in b)
    assert all("sfx" in x and "hush" in x for x in b), "новые ключи есть у КАЖДОГО блока"
