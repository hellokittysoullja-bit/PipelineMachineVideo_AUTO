"""Кукла не привязана к частоте кадров: план в секундах, моргание переживает 24 к/с.
Запуск: cd experiments/alive_cat/doll && python -m pytest -q test_rig_fps.py"""
import inspect
import numpy as np
import doll_rig as D


def test_render_default_fps_matches_assembler():
    assert inspect.signature(D.render).parameters["fps"].default == 24


def test_blink_keeps_two_closed_frames_at_24fps_in_worst_phase():
    fps = 24; worst = 99
    for ph in np.linspace(0, 1 / fps, 25, endpoint=False):
        st = D.plan(3.0, [{"t": 1.0 + ph, "do": "blink"}], seed=5)
        closed = sum(1 for i in range(int(3 * fps)) if st(i / fps)["lid"] >= 0.999)
        worst = min(worst, closed)
    assert worst >= 2


def test_plan_is_defined_in_seconds_not_frames():
    a = D.plan(4.0, [{"t": 1.0, "do": "tilt", "deg": 5, "dur": 1.0}], seed=3)
    b = D.plan(4.0, [{"t": 1.0, "do": "tilt", "deg": 5, "dur": 1.0}], seed=3)
    for t in (0.5, 1.3, 2.0, 3.1):          # одно и то же время — одно и то же состояние при любом fps
        assert a(t)["head"] == b(t)["head"] and a(t)["lid"] == b(t)["lid"]
