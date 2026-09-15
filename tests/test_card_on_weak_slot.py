# -*- coding: utf-8 -*-
"""Слот, который система САМА назвала слабым, не остаётся в ролике молча.

Замер на videos/_test60s (первый полный прогон): пайплайн пометил 4 слота
как семантически слабые по РЕАЛЬНОМУ тексту блока — и это ровно те слоты,
что забракованы глазами, — а карточек-фолбэков поставил НОЛЬ. Причина не в
бюджете: `_slot_known_bad_reason()` просто не считала вердикт Директора
причиной вообще, и знание выбрасывалось.
"""
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)
sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]

import pipeline_smart as ps  # noqa: E402

# Реальные числа того прогона: (слот, скор, вердикт глазами).
MEASURED = [(4, 0.0122, "брак"), (7, 0.0244, "терпимо"),
            (6, 0.0578, "годно"), (1, 0.0581, "терпимо")]
FLOOR = 0.061249086480936965


@pytest.fixture
def misses(monkeypatch):
    monkeypatch.setattr(ps, "ARBITER_REJECTED_ALL", [])
    monkeypatch.setattr(ps, "STOCK_EXHAUSTED_MISSES", [])
    monkeypatch.setattr(ps, "RELEVANCE_GATE_MISSES", [])
    monkeypatch.setattr(ps, "DIRECTOR_RELEVANCE_MISSES",
                        [{"index": i, "relevance": r, "threshold": FLOOR}
                         for i, r, _ in MEASURED])


def test_decisive_miss_is_a_reason(misses):
    """Слот 4 — музей в Барселоне на фразе про упавшего рыцаря, 0.20x пола."""
    assert ps._slot_known_bad_reason(4) == "director_relevance_decisive"


def test_marginal_miss_is_not_a_reason(misses):
    """Слоты У САМОГО ПОЛА (0.94x и 0.95x) карточку НЕ получают: там низкий
    скор даёт абстрактная фраза, а не плохой кадр. Негативный контроль
    правки — без порога решительности единственная карточка бюджета ушла бы
    именно сюда, на слот 1, а брак остался бы в ролике.

    ЧЕСТНО про границу: 0.5 отделяет ГОДНЫЕ от остальных, но брак от
    терпимого не отделяет — слот 7 (0.40x, «терпимо») тоже помечается. Это
    осознанно и безвредно: бюджет тратится на самый ранний слот (4), а
    среди помеченных нет ни одного годного кадра. Обещать здесь большее
    значило бы подогнать число под четыре точки и свой глаз."""
    for index in (1, 6):
        assert ps._slot_known_bad_reason(index) is None, index


def test_the_split_matches_the_eye_verdicts_on_the_measured_episode(misses):
    """Ось «решительный промах» обязана отделить брак от годного на тех
    самых числах, по которым порог и выбирался."""
    flagged = {i for i, _, _ in MEASURED if ps._slot_known_bad_reason(i)}
    good = {i for i, _, v in MEASURED if v == "годно"}
    assert flagged & good == set(), flagged & good
    assert 4 in flagged


def test_stronger_signals_still_win(monkeypatch, misses):
    """Порядок причин — по силе сигнала, вердикт Директора самый слабый."""
    monkeypatch.setattr(ps, "ARBITER_REJECTED_ALL", [{"index": 4}])
    assert ps._slot_known_bad_reason(4) == "arbiter_rejected_all"


def test_no_floor_never_flags(monkeypatch):
    """Пол 0 (Директор не работал) — не повод объявлять слот негодным."""
    monkeypatch.setattr(ps, "ARBITER_REJECTED_ALL", [])
    monkeypatch.setattr(ps, "STOCK_EXHAUSTED_MISSES", [])
    monkeypatch.setattr(ps, "RELEVANCE_GATE_MISSES", [])
    monkeypatch.setattr(ps, "DIRECTOR_RELEVANCE_MISSES",
                        [{"index": 3, "relevance": 0.0, "threshold": 0.0}])
    assert ps._slot_known_bad_reason(3) is None


def test_fraction_can_only_reduce_cards():
    """Гарантия односторонности: доля строго меньше единицы, то есть новый
    сигнал никогда не помечает больше слотов, чем наивное «ниже пола»."""
    assert 0.0 < ps.DIRECTOR_CARD_DECISIVE_FRACTION < 1.0


class TestTheReasonIsNotInertInProduction:
    """Правка, которая работает в тесте и молчит в проде, — не правка.

    Замер 14.09 (videos/_test60s, полный рендер): слот с современным музеем
    на фразе про упавшего рыцаря получил 0.20x от пола Директора, причина
    `director_relevance_decisive` честно возвращалась юнит-тестом — а
    карточек в ролике оказалось НОЛЬ. Причина в порядке: вердикт Директора
    пишется ПОСЛЕ отбора (иначе нечего оценивать), а решение о карточке
    принималось ДО, читая список, который для текущего слота всегда пуст."""

    SRC = open(os.path.join(SCRIPTS_DIR, "pipeline_smart.py"), encoding="utf-8").read()

    def test_a_decision_point_exists_after_the_director_verdict(self):
        i_verdict = self.SRC.index("DIRECTOR_RELEVANCE_MISSES.append")
        i_late = self.SRC.index("late_reason = _slot_known_bad_reason(i)")
        assert i_verdict < i_late, "поздняя проверка обязана стоять ПОСЛЕ вердикта"

    def test_the_late_point_runs_before_the_clip_is_queued(self):
        """Иначе карточка не доедет до экрана — клип уже отрендерен."""
        i_late = self.SRC.index("late_reason = _slot_known_bad_reason(i)")
        # Именно ВЫЗОВ, а не упоминание в комментарии выше по файлу.
        i_submit = self.SRC.index("future = render_pool.submit(")
        assert i_late < i_submit

    def test_the_late_point_handles_only_the_late_reason(self):
        """Остальные причины известны раньше отбора, и их обслуживает первая
        проверка — она же даёт видео-пути спасение фотографией ДО карточки."""
        tail = self.SRC[self.SRC.index("late_reason = _slot_known_bad_reason(i)"):]
        head = tail[:tail.index("luma = measure_luma")]
        assert 'late_reason == "director_relevance_decisive"' in head

    def test_a_slot_already_carded_is_not_carded_twice(self):
        tail = self.SRC[self.SRC.index("# ВТОРАЯ точка решения о карточке"):]
        head = tail[:tail.index("luma = measure_luma")]
        assert "FALLBACK_CARD_SLOTS" in head
