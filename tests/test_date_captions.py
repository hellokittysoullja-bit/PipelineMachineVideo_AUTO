"""Подпись даты печатной машинкой понимает любую дату, а не только «место + год».

Решение владельца 08.10 после эп.05: подписей места и года в ролике не было ни
одной. «Азенкур, двадцать пятое октября тысяча четыреста пятнадцатого года» —
место в начале фразы без предлога правило не видело, а «Бой тридцати, тысяча
триста пятьдесят первый год» без места не давало подписи вовсе.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import place_year  # noqa: E402
import screen_text  # noqa: E402

CONFIRM = {"Азенкур", "Азенкуре", "Азенкура", "Пуатье"}


def test_day_month_year_in_words():
    got = place_year.dates_in("Азенкур, двадцать пятое октября тысяча четыреста пятнадцатого года.")
    assert [g[0] for g in got] == ["25 ОКТЯБРЯ 1415"]


def test_day_month_year_in_digits_and_alone():
    assert [g[0] for g in place_year.dates_in("Это было 3 мая 1415 года.")] == ["3 МАЯ 1415"]
    assert [g[0] for g in place_year.dates_in("Двадцатое августа, утро.")] == ["20 АВГУСТА"]
    assert [g[0] for g in place_year.dates_in("Одиннадцатое июля тысяча трёхсот второго года.")] == \
        ["11 ИЮЛЯ 1302"]


def test_decades_are_not_dates():
    assert place_year.dates_in("в тысяча сто семидесятых годах") == []


def test_place_at_sentence_start_before_a_date():
    t = "Возьмём пример. Азенкур, двадцать пятое октября тысяча четыреста пятнадцатого года."
    assert place_year.place_and_date(t, CONFIRM) == ("АЗЕНКУР", "25 ОКТЯБРЯ 1415")


def test_sentence_start_word_without_evidence_is_not_a_place():
    t = "Теперь, двадцать пятое октября тысяча четыреста пятнадцатого года."
    assert place_year.place_and_date(t, CONFIRM | {"Теперь"}) == (None, "25 ОКТЯБРЯ 1415")


def test_date_without_place_still_gets_a_caption():
    t = "«Бой тридцати», тысяча триста пятьдесят первый год."
    assert place_year.place_and_date(t, CONFIRM) == (None, "1351")


def test_two_dates_are_ambiguous():
    assert place_year.place_and_date("В 1346 году и в 1356 году.", CONFIRM) is None


def test_place_caption_wins_the_interval_over_a_date_only_caption():
    blocks = [{"text": "«Бой тридцати», тысяча триста пятьдесят первый год.", "orig_index": 0},
              {"text": "Азенкур, двадцать пятое октября тысяча четыреста пятнадцатого года.",
               "orig_index": 1}]
    caps = screen_text.plan_place_captions(blocks, [0.0, 30.0], [10.0, 40.0], (), CONFIRM)
    assert list(caps) == [1]
    assert caps[1]["place"] == "АЗЕНКУР"


def test_date_only_caption_is_one_line_at_the_place_row():
    cap = {"place": "", "year": "1351", "start": 0.0}
    chain = screen_text.place_caption_chain(cap, 0.5, 6.0, "p.ttf", "y.ttf", lambda x: x, "y.ttf")
    assert f"y={screen_text.PLACE_Y}" in chain
    assert f"y={screen_text.YEAR_Y}:" not in chain
    times, _ = screen_text.place_char_times("", "1351")
    assert len(times) == 4 and times[0] == 0.0
