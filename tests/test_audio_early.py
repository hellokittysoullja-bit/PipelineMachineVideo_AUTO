#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Звук эпизода, начатый до цикла рендера (AUDIO_EARLY): берётся только при
совпавших входах."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import pipeline_smart as ps  # noqa: E402


def _args(**kw):
    a = dict(blocks=[{"text": "a", "section": "HOOK"}], starts=[0.0], weights=[1.0], total=10.0,
             hook_end=3.0, final_start=8.0, clicks=[1.0], plates=[], locked=True)
    a.update(kw)
    return (a["blocks"], a["starts"], a["weights"], a["total"], a["hook_end"], a["final_start"],
            a["clicks"], a["plates"], a["locked"])


def test_same_inputs_take_the_early_result(monkeypatch):
    calls = []
    monkeypatch.setattr(ps, "master_audio_premix", lambda *a: calls.append(a) or ("p.wav", {"i": -14}))
    ea = ps.EarlyAudio(_args())
    got = ea.take(_args())
    assert got is not None
    pool, fut = got
    assert fut.result() == ("p.wav", {"i": -14}) and len(calls) == 1
    pool.shutdown()


def test_changed_inputs_discard_the_early_result(monkeypatch):
    calls = []
    monkeypatch.setattr(ps, "master_audio_premix", lambda *a: calls.append(a) or ("p.wav", {}))
    ea = ps.EarlyAudio(_args())
    assert ea.take(_args(hook_end=4.0)) is None
    assert len(calls) == 1, "ранний расчёт дождан и отброшен, не оставлен работать"
    ea2 = ps.EarlyAudio(_args())
    assert ea2.take(_args(blocks=[{"text": "b", "section": "HOOK"}])) is None


def test_early_failure_is_swallowed_on_stop(monkeypatch):
    def boom(*a):
        raise RuntimeError("x")
    monkeypatch.setattr(ps, "master_audio_premix", boom)
    ps.EarlyAudio(_args()).stop()


def test_main_starts_audio_before_the_slot_loop_and_stops_it_on_every_early_return():
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "scripts", "pipeline_smart.py"), encoding="utf-8").read()
    start = src.index("early_audio = EarlyAudio(")
    loop = src.index("    for i, (b, d) in enumerate(zip(blocks, durs)):", start)
    assert start < loop
    body = src[loop:src.index("audio_pool, audio_future = _taken")]
    assert body.count("early_audio.stop()") >= 4


def test_parallax_workers_follow_cores_and_env(monkeypatch):
    monkeypatch.delenv("PARALLAX_WORKERS", raising=False)
    # Ядра — выданные прогону (cpu_budget), а не хозяина контейнера.
    monkeypatch.setattr(ps.cpu_budget, "cores", lambda *a, **k: 4)
    assert ps.parallax_workers() == 1, "до 8 ядер — как было"
    monkeypatch.setattr(ps.cpu_budget, "cores", lambda *a, **k: 16)
    assert ps.parallax_workers() == 4
    monkeypatch.setattr(ps.cpu_budget, "cores", lambda *a, **k: 64)
    assert ps.parallax_workers() == 4
    monkeypatch.setenv("PARALLAX_WORKERS", "2")
    assert ps.parallax_workers() == 2


def test_parallel_highlight_clips_do_not_share_render_state():
    """Независимость клипа: путь временного файла и stderr выводятся из его
    собственного out, общего файла нет."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "scripts", "pipeline_smart.py"), encoding="utf-8").read()
    body = src[src.index("def parallax_kenburns("):src.index("def parallax_kenburns(") + 30000]
    assert "render_tmp_path(out)" in body
    assert "stderr_path = render_tmp_path(out)" in body
