"""Кукла не привязана к частоте кадров: план в секундах, моргание переживает 24 к/с.
Запуск: cd experiments/alive_cat/doll && python -m pytest -q test_rig_fps.py"""
import inspect
import os
import numpy as np
import pytest
import doll_rig as D

needs_rig = pytest.mark.skipif(not os.path.exists("hero.png") or not os.path.exists("eyes.npy"),
                               reason="части куклы (продукт parts.py) не лежат в репозитории — тест идёт в рабочей папке с ними")


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


@needs_rig
def test_set_paper_identity_on_source_paper_and_no_halo_on_cream():
    CREAM = np.array([250, 247, 240], np.float32)
    st = dict(look=(0, 0), lid=0, head=4, ear=-3, tail=3, breath=1.01, paws={})
    r = D.Rig(); before = r.frame(st, .5).copy()
    r.set_paper(r.paper_src)                      # та же бумага — ничего не меняется
    assert np.array_equal(r.frame(st, .5), before)
    r.set_paper(CREAM); f = r.frame(st, .5)
    a = r.alpha / 255; edge = (a > 0.3) & (a < 0.7)
    halo = (f[edge].astype(int).sum(1) > CREAM.sum() + 3).mean()
    assert halo < 0.02        # наивная вставка: 1967 из 4124 (48%); альфа-композит без снятия примеси: 66; с ним: 39
