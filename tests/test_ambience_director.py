"""Вето атмосферы: что считается ВЕРДИКТОМ, а что его отсутствием.

Односторонность вето (может только снять то, что выбрал словарь) держится
двумя вещами: точкой вызова в `ambience_plan.plan_ambience` — её проверяет
`tests/test_ambience_plan.py::TestLlmVeto` — и правилом «снимаем только по
явному отказу модели», которое проверяется здесь.

Правило пришлось написать после разбора 17.09: заявленный fail-open не
покрывал ГЛАВНЫЙ режим отказа локальной модели. `LocalBrain.ask()` при сбое
возвращает пустую строку, а не бросает исключение — то есть `except
Exception` не срабатывал, пустой ответ читался как «атмосфера не нужна», и
упавшая модель молча гасила фон во всём эпизоде.
"""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import ambience_director as ad  # noqa: E402


@pytest.mark.parametrize("raw", [
    "",                       # модель упала — LocalBrain.ask отдаёт "" без исключения
    "   \n  ",                # то же, но с пробелами
    "не знаю, наверное ветер",  # ответ не по форме
    "SCENE:",                 # форма есть, сцены нет
])
def test_a_non_answer_never_removes_the_dictionary_bed(raw):
    keep, why = ad.veto_decision(raw)
    assert keep is True, f"ответ без вердикта снял атмосферу: {raw!r}"
    assert why, "причина «почему нет вердикта» обязана быть названа, а не молчать"


def test_explicit_none_is_the_only_thing_that_removes():
    keep, why = ad.veto_decision("NONE")
    assert keep is False
    assert why is None


def test_a_valid_scene_keeps_the_bed():
    keep, why = ad.veto_decision("SCENE: wind over an open field")
    assert keep is True
    assert why is None


def test_discrete_event_wording_does_not_veto_in_veto_mode():
    """`DISCRETE_EVENT_RE` писался для режима, где модель ВЫБИРАЕТ сцену: там
    разовое действие — негодный ответ. В режиме вето сцена выбрасывается
    вызывающим кодом, и отклонять по слову внутри строки, которая на решение
    не влияет, значило бы снимать законный фон.

    Контроль формы: сама `parse_answer` этот ответ по-прежнему отклоняет —
    правило не ослаблено, оно просто не применяется там, где не про то."""
    answer = "SCENE: wind over a battlefield with distant clashing"
    wants, _scene, reason = ad.parse_answer(answer)
    assert wants is False and reason.startswith("rejected_discrete_event")
    keep, why = ad.veto_decision(answer)
    assert keep is True
    assert why and why.startswith("rejected_discrete_event")


def test_the_closure_uses_the_same_rule():
    """Второй копии правила не заводится: замыкание `veto()` обязано звать
    `veto_decision`, иначе тесты выше сторожили бы функцию, которой в проде
    никто не пользуется — ровно тот класс «слой есть, и его никто не зовёт»,
    которым этот репозиторий горел семь раз."""
    import inspect
    src = inspect.getsource(ad.build_veto_fn)
    assert "veto_decision(raw)" in src
