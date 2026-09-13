"""Случайный хэш-джиттер грейда выключен по измерению.

film_look() добавлял к контрасту/насыщенности/яркости вариацию, выведенную
из ХЭША ИМЕНИ ФАЙЛА — ±4.7% по контрасту и ±9.3% по насыщенности на
MOOD_GRADE["BODY"], случайно от кадра к кадру, без связи с содержимым.

Замер на 40 кадрах опубликованного эпизода (реальная математика цепочки:
auto_levels -> auto_wb -> контраст/яркость/насыщенность + сглаживание
luma_ema, как в main()):

    насыщенность, скачок между соседними кадрами  51.2 -> 16.5  (грейд работает)
    яркость,      скачок между соседними кадрами  45.1 -> 37.4  (работает слабо)
    джиттер: разброс яркости 169.3 -> 174.3, скачок насыщенности 16.5 -> 17.5

То есть джиттер расширяет ровно тот разнобой источников, который грейд на
сборке из разных источников обязан сводить. Задачу «рецепт не одинаков на
каждом кадре» решают три СОДЕРЖАТЕЛЬНЫХ модулятора — _scene_bias,
DOMAIN_GRADE_MODE и Look Management.
"""
import os
import re
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]

import feature_flags as ff  # noqa: E402
import pipeline_smart as ps  # noqa: E402


def _eq_values(vf):
    """(contrast, saturation) из первой творческой eq= в графе."""
    m = re.search(r"eq=contrast=([\d.]+):saturation=([\d.]+)", vf)
    assert m, vf[:200]
    return float(m.group(1)), float(m.group(2))


def test_default_is_off():
    assert ff.FLAGS["GRADE_HASH_JITTER"].default == "0"


def test_two_different_files_get_the_same_recipe_by_default(monkeypatch):
    """Главное следствие: два соседних кадра одной секции больше не
    расходятся по контрасту и насыщенности просто из-за имени файла."""
    monkeypatch.setenv("GRADE_HASH_JITTER", "0")
    a = _eq_values(ps.film_look(12345, "BLOCK 1"))
    b = _eq_values(ps.film_look(98765, "BLOCK 1"))
    assert a == b


def test_jitter_flag_restores_the_old_spread(monkeypatch):
    """Код не удалён — флаг возвращает прежнее поведение."""
    monkeypatch.setenv("GRADE_HASH_JITTER", "1")
    vals = [_eq_values(ps.film_look(h, "BLOCK 1")) for h in (12345, 98765, 4242)]
    assert len(set(vals)) > 1


def test_section_mood_still_differs(monkeypatch):
    """Выключение джиттера НЕ делает весь ролик одинаковым: хук/тело/финал
    по-прежнему имеют свой mood_grade."""
    monkeypatch.setenv("GRADE_HASH_JITTER", "0")
    hook = _eq_values(ps.film_look(1, "HOOK"))
    final = _eq_values(ps.film_look(1, "FINAL"))
    assert hook != final


def test_energy_bias_is_untouched(monkeypatch):
    """energy_bias — реальная громкость голоса в этом отрезке, сигнал
    содержательный, он остаётся."""
    monkeypatch.setenv("GRADE_HASH_JITTER", "0")
    quiet = _eq_values(ps.film_look(1, "BLOCK 1", energy_bias=-0.4))
    loud = _eq_values(ps.film_look(1, "BLOCK 1", energy_bias=0.4))
    assert quiet != loud
