"""Согласование формы единицы измерения с числом в тексте для озвучки
(scripts/speech_generate.normalize_pronunciation / russian_unit_form).

Отдельный файл, а не тесты внутри test_speech_generate.py, по одной
причине: тот файл целиком помечен skipif "нет ffmpeg в PATH" (там есть
тесты, реально декодирующие аудио), и чисто текстовая логика вместе с ним
пропускалась в любом окружении без ffmpeg — то есть ровно там, где дешевле
всего было бы поймать регрессию. Здесь ffmpeg не нужен вообще."""
import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

sys.argv = ["speech_generate.py", tempfile.gettempdir()]
import speech_generate as sg   # noqa: E402


def test_unit_form_agrees_with_number():
    # Раньше подстановка всегда ставила родительный падеж множественного
    # числа: "1 кг" -> "1 килограммов", "1,5 кг" -> "1,5 килограммов".
    # Текст уходил В ОЗВУЧКУ (платные кредиты TTS) и звучал неграмотно, а
    # для канала про вес меча "1,5 кг" — основной контент.
    cases = {
        "Меч весил 1 кг.": "1 килограмм",
        "Меч весил 2 кг.": "2 килограмма",
        "Меч весил 4 кг.": "4 килограмма",
        "Меч весил 5 кг.": "5 килограммов",
        "Меч весил 11 кг.": "11 килограммов",
        "Меч весил 14 кг.": "14 килограммов",
        "Меч весил 21 кг.": "21 килограмм",
        "Меч весил 22 кг.": "22 килограмма",
        "Меч весил 1,5 кг.": "1,5 килограмма",
        "Ровно 1%.": "1 процент",
        "Ровно 22%.": "22 процента",
        "Ровно 50%.": "50 процентов",
        "Длина 104 см.": "104 сантиметра",
        "Зазор 3 мм.": "3 миллиметра",
        "Дистанция 7 км.": "7 километров",
    }
    for text_in, expected in cases.items():
        out, subs = sg.normalize_pronunciation(text_in)
        assert expected in out, f"{text_in!r} -> {out!r}, ожидалось {expected!r}"
        assert subs, f"подстановка должна логироваться: {text_in!r}"


def test_russian_unit_form_edges():
    forms = ("килограмм", "килограмма", "килограммов")
    assert sg.russian_unit_form("1", forms) == "килограмм"
    assert sg.russian_unit_form("101", forms) == "килограмм"
    assert sg.russian_unit_form("111", forms) == "килограммов"
    assert sg.russian_unit_form("112", forms) == "килограммов"
    assert sg.russian_unit_form("3", forms) == "килограмма"
    assert sg.russian_unit_form("0", forms) == "килограммов"
    assert sg.russian_unit_form("0,5", forms) == "килограмма"
    assert sg.russian_unit_form("2.75", forms) == "килограмма"


def test_whole_number_is_part_of_substitution():
    # В паттерн входит ВСЁ число, а не последняя цифра — иначе форму
    # невозможно согласовать (и в отчёте "before" врал).
    _, subs = sg.normalize_pronunciation("Средний вес составлял 1.2 кг.")
    assert subs[0]["before"] == "1.2 кг"
    assert subs[0]["after"] == "1.2 килограмма"


def test_ambiguous_bare_g_untouched():
    # "г" (граммы vs год vs город) сознательно не в таблице.
    text, subs = sg.normalize_pronunciation("Событие произошло в 1932 г.")
    assert text == "Событие произошло в 1932 г." and subs == []


def test_no_number_no_substitution():
    text, subs = sg.normalize_pronunciation("Вес измеряли в килограммах, без цифр.")
    assert subs == [] and "килограммах" in text
