# -*- coding: utf-8 -*-
"""Название ролика после хука, место и год печатной машинкой, карточка цитаты.

Утверждено владельцем по ручному демо 03.10. Главное правило подписи места и
карточки цитаты — ничего не додумывать: нет явного места И года (или нет
дословной цитаты с автором) — нет и текста на экране.
"""
import re
import contextlib
import io
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]

import chapter_card as cc  # noqa: E402
import place_year as py  # noqa: E402

def _place_year(text, confirm=None):
    """Место и год подписи (прежний контракт place_and_year) — через рабочую
    функцию place_and_date: подпись без места здесь считается «нет места»."""
    got = py.place_and_date(text, confirm)
    if not got or not got[0]:
        return None
    m = re.search(r"(\d{3,4})$", got[1])
    return (got[0], int(m.group(1))) if m else None

import screen_text as st  # noqa: E402

SCRIPT_03 = os.path.join(REPO_ROOT, "videos", "03_plen", "script.txt")
FONT = cc.default_font()


# ---------------------------------------------------------------- год словами

@pytest.mark.parametrize("text,year", [
    ("на битве при Пуатье в тысяча триста пятьдесят шестом году", 1356),
    ("Двадцать пятое октября тысяча четыреста пятнадцатого года, север Франции", 1415),
    ("Одиннадцатое июля тысяча трёхсот второго года, Фландрия", 1302),
    ("в тысяча сто девяносто втором году Ричард", 1192),
    ("А в тысяча девятьсот девяносто шестом году нашли могилу", 1996),
    ("в 1356 году", 1356),
])
def test_years_in_words_and_digits(text, year):
    assert [y for y, _a, _b in py.years_in(text)] == [year]


@pytest.mark.parametrize("text", [
    "В тысяча сто семидесятых годах турнир был совсем другим",   # десятилетие
    "В Англии четырнадцатого века",                              # век
    "сто тысяч марок серебром",
])
def test_not_a_year(text):
    assert py.years_in(text) == []


# ---------------------------------------------------------------- место

@pytest.mark.parametrize("text,expected", [
    ("Лучше всего это видно на битве при Пуатье в тысяча триста пятьдесят шестом году.", ("ПУАТЬЕ", 1356)),
    ("При Креси в тысяча триста сорок шестом году король Эдуард", ("КРЕСИ", 1346)),
    ("Там, в Лондоне, в тысяча триста шестьдесят четвёртом году он и умер.", None),  # без подтверждения
    ("Одиннадцатое июля тысяча трёхсот второго года, Фландрия, город Куртре.", ("КУРТРЕ", 1302)),
    ("В тысяча двести семнадцатом году под английским городом Линкольн войска", ("ЛИНКОЛЬН", 1217)),
    ("в тысяча сто девяносто втором году его опознали в Австрии", ("АВСТРИЯ", 1192)),
    ("Короля увезли в Англию, по договору тысяча трёхсот шестидесятого года", None),  # винительный
    ("в тысяча четыреста шестьдесят первом году при Таутоне", None),  # -е без подтверждения
    ("Сам Карл Смелый через год погиб под Нанси", None),             # нет года
    ("В тысяча четыреста семьдесят шестом году они воевали с герцогом", None),  # нет места
])
def test_place_and_year_never_guesses(text, expected):
    assert _place_year(text) == expected


def test_ambiguous_form_needs_confirmation_by_script():
    t = "Там, в Лондоне, в тысяча триста шестьдесят четвёртом году он и умер."
    assert _place_year(t, {"Лондон"}) == ("ЛОНДОН", 1364)
    assert _place_year("при Легнице в тысяча двести сорок первом году", {"Легница"}) == ("ЛЕГНИЦА", 1241)


def test_person_is_not_confirmed_as_place():
    words = py.script_words("Его схватил король Генрих. Потом было другое.")
    assert "Генрих" not in words
    assert _place_year("при Генрихе в тысяча сотом году", words) is None


def test_two_places_or_two_years_no_caption():
    assert _place_year("при Пуатье в 1356 году и при Креси в 1346 году") is None


def test_places_on_episode_03_are_the_verified_ones():
    """Глазами сверено: каждое место и год ниже названы диктором в одной фразе
    и исторически верны."""
    from script_parser import parse_blocks
    text = open(SCRIPT_03, encoding="utf-8").read()
    with contextlib.redirect_stdout(io.StringIO()):
        blocks = parse_blocks(SCRIPT_03)
    found = {_place_year(b["text"], py.script_words(text)) for b in blocks} - {None}
    assert found == {("ФРАНЦИЯ", 1119), ("ЛИНКОЛЬН", 1217), ("БРЕМЮЛЬ", 1119), ("АВСТРИЯ", 1192),
                     ("ПУАТЬЕ", 1356), ("ЛОНДОН", 1364), ("КУРТРЕ", 1302), ("КРЕСИ", 1346),
                     ("АЗЕНКУР", 1415)}


# ---------------------------------------------------------------- расписание подписей

def _slots(texts, starts):
    return [{"text": t, "parent_text": t, "orig_index": k, "words": len(t.split())}
            for k, t in enumerate(texts)]


def test_captions_spacing_and_busy(monkeypatch):
    t1 = "при Пуатье в 1356 году"
    t2 = "при Креси в 1346 году"
    blocks = _slots([t1, t2, t1.replace("1356", "1360")], None)
    starts, ends = [0.0, 20.0, 60.0], [10.0, 30.0, 70.0]
    caps = st.plan_place_captions(blocks, starts, ends)
    assert sorted(caps) == [0, 2]                         # второй ближе 40 с — нет
    assert caps[0] == {"place": "ПУАТЬЕ", "year": "1356", "start": 0.55}
    blocks[0]["stat"] = "1356"
    assert sorted(st.plan_place_captions(blocks, starts, ends)) == [1, 2]   # на плашке — нет
    blocks[0].pop("stat")
    blocks[0]["chapter_card"] = {"title": "X"}
    assert 0 not in st.plan_place_captions(blocks, starts, ends)


def test_caption_skipped_on_too_short_slot():
    blocks = _slots(["при Пуатье в 1356 году"], None)
    assert st.plan_place_captions(blocks, [0.0], [2.0]) == {}


def test_caption_geometry_above_player_bar_and_bigger_than_demo():
    assert st.YEAR_Y + st.YEAR_SIZE <= 880
    assert st.PLACE_SIZE >= 48 * 1.15 and st.YEAR_SIZE >= 34 * 1.15
    chain = st.place_caption_chain({"place": "ПУАТЬЕ", "year": "1356"}, 0.5, 6.0, "f", "g",
                                   lambda s: s, FONT)
    assert chain.count("drawtext") == len("ПУАТЬЕ") + len("1356")
    assert "0xC8102E" in chain and "mod(t-" in chain     # красная полоска и мигающий курсор


def test_char_times_match_demo_rate():
    times, typed = st.place_char_times("ПУАТЬЕ", "1356")
    assert times[1] - times[0] == pytest.approx(0.075)
    assert typed == pytest.approx(10 * 0.075 + st.PLACE_LINE_PAUSE)


# ---------------------------------------------------------------- цитата

def test_quote_needs_guillemets_and_named_author():
    assert st.find_quote("Как писал Фруассар, «рыцари сражались до последнего».") == \
        ("рыцари сражались до последнего", "Фруассар")
    assert st.find_quote("«Я одолел около пятисот рыцарей», — вспоминал Маршал.") == \
        ("Я одолел около пятисот рыцарей", "Маршал")


@pytest.mark.parametrize("text", [
    "Фруассар пишет, что рыцари сражались до последнего.",       # пересказ
    "По словам монаха, погибло всего трое.",                       # пересказ
    "Об этом есть целая поэма «Песнь о Роланде».",                 # название, не цитата
    "«Рыцари сражались до последнего», — так это обычно пересказывают.",  # нет автора
])
def test_no_quote_card_without_verbatim_quote_and_author(text):
    assert st.find_quote(text) is None


def test_no_quote_cards_on_episode_03():
    from script_parser import parse_blocks
    with contextlib.redirect_stdout(io.StringIO()):
        blocks = parse_blocks(SCRIPT_03)
    for b in blocks:
        b["parent_text"] = b["text"]
    n = len(blocks)
    assert st.plan_quote_cards(blocks, [10.0 * k for k in range(n)], [10.0 * k + 9 for k in range(n)]) == {}


def test_quote_card_white_marks():
    g = st.quote_graph({"quote": "рыцари сражались до последнего", "author": "Фруассар"},
                       1.0, 6.0, {"mark": "m", "quote": "q", "author": "a"},
                       {"mark": FONT, "quote": FONT, "author": FONT}, lambda s: s, "0:v", "vq")
    assert "text='«':fontsize=150:fontcolor=white" in g
    assert "0xC8102E" not in g


# ---------------------------------------------------------------- название ролика

def _script(tmp_path, meta):
    p = tmp_path / "script.txt"
    p.write_text("=== METADATA ===\n" + meta + "\nSERIES: X\n\n=== HOOK ===\nРаз два.\n"
                 "=== BLOCK 1: ПОЛЕ ===\nТри.\n", encoding="utf-8")
    return str(p)


def test_title_card_field_wins(tmp_path):
    p = _script(tmp_path, "TITLE_CARD: Рыцари почти не убивали друг друга\n"
                          "TITLE: Что-то другое — и длинное (рабочее)")
    assert cc.film_title(p, FONT) == "Рыцари почти не убивали друг друга"


def test_title_without_brackets_and_cut_at_dash(tmp_path):
    p = _script(tmp_path, "TITLE: Рыцари почти не убивали друг друга — и день, когда это "
                          "закончилось (рабочее, финальное выбирает владелец)")
    assert cc.film_title(p, FONT) == "Рыцари почти не убивали друг друга"


def test_title_card_line_does_not_change_blocks(tmp_path):
    from script_parser import parse_blocks
    a = _script(tmp_path, "TITLE: Рыцари")
    b_dir = tmp_path / "b"
    b_dir.mkdir()
    b = _script(b_dir, "TITLE_CARD: Рыцари почти не убивали\nTITLE: Рыцари")
    with contextlib.redirect_stdout(io.StringIO()):
        assert parse_blocks(a) == parse_blocks(b)


def test_title_card_line_does_not_change_sound_plan_title(tmp_path):
    import sound_director
    _script(tmp_path, "TITLE_CARD: Короткое\nTITLE: Полное название")
    assert sound_director.episode_title(str(tmp_path)) == "Полное название"


def test_title_layout_two_lines_like_demo():
    size, lines, ok = cc.title_layout("РЫЦАРИ ПОЧТИ НЕ УБИВАЛИ ДРУГ ДРУГА", FONT, cc.TITLE)
    assert ok and size == 72 and lines == ["РЫЦАРИ ПОЧТИ НЕ УБИВАЛИ", "ДРУГ ДРУГА"]


def test_title_style_has_no_red_line_chapter_style_has():
    t = cc.card_filter("Рыцари почти не убивали друг друга", 2.4, 8.0, FONT, FONT, lambda s: s,
                       style=cc.TITLE)
    c = cc.card_filter("Поле", 1.8, 8.0, FONT, FONT, lambda s: s, style=cc.CHAPTER)
    assert "drawbox" not in t and "drawbox" in c
    assert "fontsize=72" in t and "boxblur=30:2" in t


def test_title_replaces_chapter_one_card(monkeypatch, tmp_path):
    pytest.importorskip("PIL")
    import pipeline_smart as ps
    monkeypatch.setenv("CHAPTER_CARD", "1")
    monkeypatch.setenv("TITLE_DROP", "1")
    p = _script(tmp_path, "TITLE_CARD: Рыцари почти не убивали")
    monkeypatch.setattr(ps, "SCRIPT_FILE", p)
    blocks = [{"section": s, "text": "x", "words": 1} for s in
              ("HOOK", "BLOCK 1: ПОЛЕ", "BLOCK 2: СЧЁТ")]
    monkeypatch.setattr(ps, "SPEECH_ENDS", [2.0, 8.0, 15.0])
    assert ps.mark_chapter_cards(blocks, [0.0, 5.0, 11.0]) == [1, 2]
    assert blocks[1]["chapter_card"]["style"] == "title"
    assert blocks[1]["chapter_card"]["title"] == "Рыцари почти не убивали"
    assert "style" not in blocks[2]["chapter_card"]
    monkeypatch.setenv("TITLE_DROP", "0")
    ps.mark_chapter_cards(blocks, [0.0, 5.0, 11.0])
    assert blocks[1]["chapter_card"]["title"] == "ПОЛЕ"


def test_title_pause_target_in_fix_pauses(monkeypatch, tmp_path):
    import csv
    import json
    import fix_pauses as fp
    monkeypatch.setenv("CHAPTER_CARD", "1")
    monkeypatch.setenv("TITLE_DROP", "1")
    d = tmp_path
    (d / "media_plan" / "alignment").mkdir(parents=True)
    (d / "script.txt").write_text(
        "=== METADATA ===\nTITLE_CARD: Рыцари\n\n=== HOOK ===\nОдин.\n=== BLOCK 1: ПОЛЕ ===\nДва.\n"
        "=== BLOCK 2: СЧЁТ ===\nТри.\n", encoding="utf-8")
    json.dump({"HOOK": 0.0, "BLOCK 1: ПОЛЕ": 10.0, "BLOCK 2: СЧЁТ": 20.0},
              open(d / "media_plan" / "section_offsets.json", "w"))
    for k in range(3):
        with open(d / "media_plan" / "alignment" / f"{k:02d}.csv", "w", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["char", "start", "end"])
            w.writerows([["О", 0.0, 0.1], [".", 9.0, 9.2]])
    wins = fp.chapter_card_windows(str(d))
    assert [w[2] for w in wins] == [cc.TITLE_PAUSE_SEC, cc.CARD_PAUSE_SEC]
    monkeypatch.setenv("TITLE_DROP", "0")
    assert [w[2] for w in fp.chapter_card_windows(str(d))] == [cc.CARD_PAUSE_SEC] * 2


def test_flags_off_no_marks_no_post(monkeypatch):
    pytest.importorskip("PIL")
    import pipeline_smart as ps
    blocks = _slots(["при Пуатье в 1356 году", "Он сказал: «рыцари бились», писал Фруассар."], None)
    monkeypatch.setenv("PLACE_CAPTION", "0")
    monkeypatch.setenv("QUOTE_CARD", "0")
    ps.mark_screen_text(blocks, [0.0, 20.0], 40.0, True)
    assert not any(b.get("place_caption") or b.get("quote_card") for b in blocks)
    assert ps.clip_post_spec(blocks[0], None, 0.0) is None
    monkeypatch.setenv("PLACE_CAPTION", "1")
    ps.mark_screen_text(blocks, [0.0, 20.0], 40.0, False)      # без PHRASE LOCK — тоже ничего
    assert not any(b.get("place_caption") for b in blocks)
    ps.mark_screen_text(blocks, [0.0, 20.0], 40.0, True)
    assert blocks[0]["place_caption"]["place"] == "ПУАТЬЕ"
    post = ps.clip_post_spec(blocks[0], None, 0.0)
    assert post["caption"]["local"] == pytest.approx(0.55)


@pytest.mark.skipif(not __import__("shutil").which("ffmpeg"), reason="нет ffmpeg")
def test_caption_and_quote_render_through_post_pass(tmp_path):
    pytest.importorskip("PIL")
    import subprocess
    import pipeline_smart as ps
    src = str(tmp_path / "src.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=1920x1080:r=24:d=5",
                    "-c:v", "libx264", "-preset", "ultrafast"] + ps.CLIP_PIX_ARGS + ps.COLOR_META_ARGS + [src],
                   check=True)
    post = {"caption": {"place": "ПУАТЬЕ", "year": "1356", "start": 0.5, "local": 0.5},
            "quote": {"quote": "рыцари сражались до последнего", "author": "Фруассар",
                      "q0": 1.0, "q1": 4.0, "l0": 1.0, "l1": 4.0}}
    out = str(tmp_path / "out.mp4")
    assert ps.apply_clip_post(src, out, 5.0, post)
    assert ps.verify_clip(out, 5.0)[0]
