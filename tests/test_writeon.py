"""Карандашная надпись: форма закрыта целиком, порядок письма, точки, перенос, темп."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import writeon as w  # noqa: E402

W, H, FPS = 1920, 1080, 24


def _render(text, size=120, max_w=1500):
    L = w.layout(text, size, W, H, W/2, H/2, seed=3, max_w=max_w)
    ev, T = w.plan(L, FPS)
    ink = w.Ink(L, ev, H, W)
    frames = []
    for f in range(int(np.ceil(T*FPS)) + 2):
        ink.advance(f/FPS)
        frames.append(ink.alpha())
    return L, ev, T, frames


def test_full_coverage_with_dots_and_punctuation():
    L, ev, T, fr = _render("ёжик 2025, привет!: да?")
    mask = np.zeros((H, W), np.float32)
    for let in L:
        np.maximum(mask, let["soft"], out=mask)
        for d in let["dots"]:
            np.maximum(mask, d, out=mask)
    ink = fr[-1] > 0.3*0.86
    assert ink[mask > 0.5].mean() > 0.98             # ни одной недописанной части
    assert (fr[-1][mask == 0] == 0).all()           # и ничего за пределами формы


def test_dots_are_drawn_after_letter_body():
    L, ev, *_ = _render("ёжик!")
    yo = [let for let in L if let["ch"] == "ё"][0]
    assert yo["dots"], "точки ё должны быть отдельными касаниями"
    kinds = [e["kind"] for e in w.plan([yo], FPS)[0]]
    assert kinds[-len(yo["dots"]):] == ["dot"]*len(yo["dots"])


@pytest.mark.parametrize("ch", list("ркн"))
def test_stems_first(ch):
    L = w.layout(ch, 200, W, H, W/2, H/2, jitter=0)
    p = w.letter_strokes(L[0])[0]["path"]
    assert abs(p[-1, 0] - p[0, 0]) > 1.8*abs(p[-1, 1] - p[0, 1])   # первый — вертикальная стойка
    assert p[0, 0] < p[-1, 0]                                       # сверху вниз


def test_a_starts_with_bowl_counterclockwise():
    L = w.layout("а", 200, W, H, W/2, H/2, jitter=0)
    p = w.letter_strokes(L[0])[0]["path"]
    head = p[:len(p)//6]
    assert head[:, 1].mean() < p[0, 1]           # от стыка влево — по верху овала


def test_d_starts_at_bowl_not_at_top_curl():
    L = w.layout("д", 200, W, H, W/2, H/2, jitter=0)
    p = max(w.letter_strokes(L[0]), key=lambda s: len(s["path"]))["path"]
    assert p[0, 0] > p[-1, 0]                    # начало ниже конца: завиток — последним


def test_long_text_wraps_inside_width():
    L = w.layout("это очень длинная мысль которая не влезает в одну строку экрана", 120, W, H, W/2, H/2, max_w=1500)
    xs = np.nonzero(sum(let["soft"] for let in L).max(0) > 0.3)[0]
    ys = np.nonzero(sum(let["soft"] for let in L).max(1) > 0.3)[0]
    assert xs.max() - xs.min() <= 1500 + 40
    assert ys.max() - ys.min() > 1.6*L[0]["size"]                  # две строки


def test_empty_text():
    assert w.layout("", 120, W, H, 0, 0) == []
    assert w.layout("   ", 120, W, H, 0, 0) == []
    ev, T = w.plan([], FPS)
    assert ev == [] and T == 0


def test_tempo_about_a_second_per_word_and_no_pulse():
    words, total, empty, n = 0, 0.0, 0, 0
    for t in ["третий день", "только открыть", "нужен один шаг", "страшно начать"]:
        L, ev, T, fr = _render(t)
        words += len(t.split()); total += T
        inc = np.array([np.clip(b - a, 0, None).sum() for a, b in zip(fr, fr[1:])])[:int(T*FPS)]
        empty += int((inc < 0.05*inc.mean()).sum()); n += len(inc)
    assert 0.85 <= total/words <= 1.45                              # ~1 с/слово (решение владельца)
    assert empty/n < 0.15                                           # без «вспышек» и пустых кадров


def test_plan_offset_and_monotonic_time():
    L = w.layout("шаг", 120, W, H, W/2, H/2)
    ev, T = w.plan(L, FPS, t0=5.0)
    assert ev[0]["t0"] == pytest.approx(5.0)
    assert all(a["t1"] <= b["t0"] + 1e-9 for a, b in zip(ev, ev[1:]))
    assert ev[-1]["t1"] == pytest.approx(5.0 + T)
