# -*- coding: utf-8 -*-
"""Предпросмотр экрана (script_preflight.py) и две правки детекторов, которые
он вскрыл на сценарии эп.04 (фикстура — исходные фразы ДО ручного обхода):
* автор цитаты в начале предложения («Юниус пишет дочери: «…»»);
* подпись места уходила на страну («Бамберг, небольшое княжество в Германии»).
"""
import os
import re
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import place_year as py  # noqa: E402
import screen_text as st  # noqa: E402
import script_preflight as pf  # noqa: E402

FIXTURE = os.path.join(REPO_ROOT, "tests", "fixtures", "script_preflight", "vedmy_original.txt")
RAW = open(FIXTURE, encoding="utf-8").read()

BAMBERG = ("Тысяча шестьсот двадцать шестой год. За четыре года до той родинки из начала. "
           "Бамберг, небольшое княжество в Германии, которым правит епископ по фамилии Фукс.")
JUNIUS = ("Юниус пишет дочери: «Невиновным я попал в тюрьму, невиновным меня пытали, "
          "невиновным я должен умереть». И объясняет ей правило этого места.")


# ---------------------------------------------------------------- место

def test_city_in_apposition_wins_over_country():
    assert py.place_and_year(BAMBERG, py.script_words(RAW)) == ("БАМБЕРГ", 1626)


@pytest.mark.parametrize("text,expected", [
    ("Бамберг, город на Майне, в 1626 году", ("БАМБЕРГ", 1626)),
    ("Вюрцбург, соседнее епископство, в 1627 году", ("ВЮРЦБУРГ", 1627)),
    ("Юниус, которого город выбирал, в 1628 году", None),      # «которого» — не прилагательное
    ("в 1626 году в Германии и во Франции", None),              # две страны — равноправны
    ("в 1626 году при Пуатье и при Креси", None),               # два города — равноправны
    ("в тысяча сто девяносто втором году его опознали в Австрии", ("АВСТРИЯ", 1192)),  # страна одна
])
def test_specific_place_only_over_region(text, expected):
    assert py.place_and_year(text) == expected


def test_person_is_not_a_place_by_genitive_form():
    """«при Шекспире» подтверждалось формой «Шекспира» (пьесы Шекспира) как
    место — подписи «ШЕКСПИРА 1600» быть не должно."""
    words = py.script_words("В Лондоне шли пьесы Шекспира. И вскоре Шекспир пишет «Макбета».")
    assert py.place_and_year("при Шекспире в 1600 году", words) is None


def test_era_is_not_a_place():
    assert py.place_and_year("в Средневековье в тысяча трёхсотом году") is None


def test_apposition_oblique_case_confirmed_by_script():
    words = py.script_words("Ньюкасл, середина семнадцатого века.")
    assert py.place_and_year("В английском городе Ньюкасле в 1649 году", words) == ("НЬЮКАСЛ", 1649)
    # без подтверждения — как написано (несклоняемое «Куртре» не режется)
    assert py.place_and_year("в городе Куртре в 1302 году") == ("КУРТРЕ", 1302)


# ---------------------------------------------------------------- автор цитаты

def test_author_at_sentence_start_confirmed_by_script():
    names = st.script_names(RAW)
    assert st.find_quote(JUNIUS, confirm=names) == (
        "Невиновным я попал в тюрьму, невиновным меня пытали, невиновным я должен умереть", "Юниус")


def test_author_at_sentence_start_without_confirmation_is_refused():
    """Одиночное слово с заглавной в начале предложения без подтверждения
    сценарием — не автор: ложный автор хуже пропуска."""
    assert st.find_quote(JUNIUS) is None
    assert st.find_quote(JUNIUS, confirm={"Бамберг"}) is None


@pytest.mark.parametrize("text", [
    "Король пишет: «Я устал от этой войны и хочу мира».",          # нарицательное
    "Он пишет: «Я устал от этой войны и хочу мира».",              # местоимение
    "Потом пишет: «Я устал от этой войны и хочу мира».",
    "Юниус пишет, что его пытали тисками.",                         # пересказ
    "Шпренгер написал «Молот ведьм».",                              # название книги
    "Потом Яков пишет о ведьмах целую книгу, «Демонологию».",      # название через запятую
    "Это книгопечатание: «Молот ведьм» выходит одним изданием за другим.",
])
def test_no_false_author(text):
    names = {"Юниус", "Яков", "Шпренгер", "Король", "Он", "Потом"}
    assert st.find_quote(text, confirm=names) is None


@pytest.mark.parametrize("text,author", [
    ("Потом Юниус пишет: «Я не сделал ничего дурного никому».", "Юниус"),
    ("Иоганнес Юниус пишет: «Я не сделал ничего дурного никому».", "Иоганнес Юниус"),
    ("В письме Юниус пишет дочери: «Я не сделал ничего дурного никому».", "Юниус"),
])
def test_author_not_at_start_or_two_words(text, author):
    assert st.find_quote(text)[1] == author


def test_mark_screen_text_passes_script_names(monkeypatch, tmp_path):
    """Рендер подтверждает автора тем же сценарием (иначе карточка Юниуса
    есть в предпросмотре и нет в ролике)."""
    pytest.importorskip("PIL")
    import pipeline_smart as ps
    monkeypatch.setenv("QUOTE_CARD", "1")
    monkeypatch.setenv("ON_SCREEN_TEXT", "1")
    monkeypatch.setenv("PLACE_CAPTION", "0")
    p = tmp_path / "script.txt"
    p.write_text(RAW, encoding="utf-8")
    monkeypatch.setattr(ps, "SCRIPT_FILE", str(p))
    blocks = [{"text": JUNIUS, "parent_text": JUNIUS, "orig_index": 0, "words": len(JUNIUS.split()),
               "section": "BLOCK 1: X"}]
    ps.mark_screen_text(blocks, [0.0], 20.0, True)
    assert blocks[0]["quote_card"]["author"] == "Юниус"


# ---------------------------------------------------------------- предпросмотр

def _report(tmp_path, text):
    p = tmp_path / "script.txt"
    p.write_text(text, encoding="utf-8")
    return pf.preflight(str(p))


def test_preflight_on_fixture_shows_elements_and_no_errors():
    rep = pf.preflight(FIXTURE)
    out = rep.render()
    assert "БАМБЕРГ 1626" in out and "ГЛАРУС 1782" in out
    assert "— Юниус" in out
    assert "ВЕДЬМ ЖГЛИ НЕ ПРИ" in out          # название после хука
    assert "BLOCK 2" not in out or "ПОСЛЕДНЯЯ ВЕДЬМА" in out
    assert rep.errors == []


def test_preflight_cli_exit_codes(tmp_path):
    assert pf.main([FIXTURE]) == 0
    bad = RAW.replace("Юниус пишет дочери:", "Дочери пишут:")
    p = tmp_path / "script.txt"
    p.write_text(bad, encoding="utf-8")
    assert pf.main([str(tmp_path)]) == 1


def test_preflight_errors_on_quote_without_author(tmp_path):
    rep = _report(tmp_path, RAW.replace("Юниус пишет дочери:", "Дочери пишут:"))
    assert any("автор не найден" in e for e in rep.errors)


def test_preflight_errors_on_region_caption_with_city(tmp_path):
    text = RAW.replace("Через два года очередь", "В тысяча шестьсот двадцать восьмом году в Германии "
                       "сожгли многих, и Бамберг тоже. Через два года очередь")
    rep = _report(tmp_path, text)
    assert any("ГЕРМАНИЯ" in e and "БАМБЕРГ" in e for e in rep.errors)


def test_preflight_tags_and_tts_lint(tmp_path):
    text = RAW.replace("Брали богатых", "[Pause] Брали — богатых [long pause] [foo]")
    text = text.replace("Проверяй", "[energetic] Проверяй").replace("[climax]", "")
    rep = _report(tmp_path, text)
    errs, warns = " ".join(rep.errors), " ".join(rep.warnings)
    assert "[Pause]" in errs and "[long pause]" in errs and "[foo]" in errs
    assert "[energetic] встречается 2" in warns
    assert "тире" in warns
    assert "нет [climax]" in warns


def test_preflight_chapter_title_too_long(tmp_path):
    long = "ОЧЕНЬ ДЛИННОЕ НАЗВАНИЕ ГЛАВЫ КОТОРОЕ НИКАК НЕ ВЛЕЗЕТ В ДВЕ СТРОКИ ЗАСТАВКИ ДАЖЕ МЕЛКО"
    rep = _report(tmp_path, RAW.replace("ПОСЛЕДНЯЯ ВЕДЬМА", long))
    assert any("не влезает в 2 строки" in e for e in rep.errors)


def test_preflight_long_block_and_stat_width(tmp_path):
    long_phrase = " ".join(["слово"] * 70)
    text = RAW.replace("Проверяй, что тебе показывают.",
                       f"[stat:ДВАДЦАТЬ ШИЛЛИНГОВ ЗА КАЖДУЮ ВЕДЬМУ] {long_phrase}.")
    rep = _report(tmp_path, text)
    assert any("шире кадра" in e for e in rep.errors)
    assert any("70 слов без [pause]" in w for w in rep.warnings)


def test_preflight_chapter_start_phrase_is_busy(tmp_path):
    """Подпись на короткой первой фразе главы не встанет — она целиком под
    заставкой (эп.03: КУРТРЕ 1302)."""
    text = RAW.replace("Последней казнённой ведьмой Европы часто называют швейцарскую служанку Анну "
                       "Гёльди. Её обезглавили в тысяча семьсот восемьдесят втором году в городе Гларус.",
                       "В тысяча семьсот восемьдесят втором году, город Гларус.")
    out = _report(tmp_path, text).render()
    assert "ГЛАРУС 1782  [не будет" in out


def test_subcut_threshold_matches_pipeline():
    src = open(os.path.join(REPO_ROOT, "scripts", "pipeline_smart.py"), encoding="utf-8").read()
    m = re.search(r"^SUBCUT_MIN_SOURCE_DUR = ([\d.]+)", src, re.M)
    assert float(m.group(1)) == pf.SUBCUT_MIN_SOURCE_DUR


def test_render_prints_preflight():
    src = open(os.path.join(REPO_ROOT, "scripts", "pipeline_smart.py"), encoding="utf-8").read()
    assert "script_preflight.print_preflight(SCRIPT_FILE)" in src


def test_print_preflight_never_raises(tmp_path, capsys):
    assert pf.print_preflight(str(tmp_path / "missing.txt")) == (0, 0)
    assert "не выполнен" in capsys.readouterr().out
