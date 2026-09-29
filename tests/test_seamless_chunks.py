# -*- coding: utf-8 -*-
"""Склейка кусками (xfade_chain_chunked) = склейка одним проходом, кадр в
кадр, включая переходы на стыках кусков. Сравнение — без сжатия (ffv1),
на настоящем ffmpeg: если хоть один кадр отличается, тест падает."""
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.argv = ["pipeline_smart.py", REPO]
import pipeline_smart as ps  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="нет ffmpeg")
W, H = 96, 64


def _clip(path, k, frames):
    src = ("testsrc2", "smptebars", "rgbtestsrc", "testsrc", "mandelbrot", "life", "cellauto")[k % 7]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"{src}=s={W}x{H}:r={ps.FPS}",
                    "-vf", f"hue=h={k * 37},eq=brightness={0.03 * (k % 5)}", "-frames:v", str(frames),
                    "-c:v", "ffv1", "-pix_fmt", "yuv420p", path], check=True)


def _frames_of(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, W * H * 3 // 2)


@pytest.mark.parametrize("workers", ["3", "5"])
def test_chunked_is_frame_identical_to_one_pass(tmp_path, monkeypatch, workers):
    monkeypatch.setattr(ps, "final_pass_encode_args", lambda: ["-c:v", "ffv1", "-pix_fmt", "yuv420p"])
    monkeypatch.setattr(ps, "hwdec_input_args", lambda *a, **k: [])
    monkeypatch.setenv("FINAL_CHUNK_WORKERS", workers)
    n = 11
    lens = [30, 26, 34, 20, 28, 31, 22, 27, 33, 25, 29]      # кадров
    clips = []
    for k in range(n):
        p = str(tmp_path / f"c{k}.mkv")
        _clip(p, k, lens[k])
        clips.append(p)
    durs = [f / ps.FPS for f in lens]
    # Секции меняются часто — на стыках кусков встанут и растворения, и резы.
    sections = ["HOOK", "HOOK", "B1", "B1", "B2", "B2", "B2", "B3", "B4", "B4", "B5"]
    blocks = [{"section": s_, "text": f"фраза {i}", "words": 2, "is_subcut": False}
              for i, s_ in enumerate(sections)]
    plan = ps.plan_transitions(sections, blocks)
    one = str(tmp_path / "one.mkv")
    ok1, d1 = ps.xfade_chain(clips, durs, sections, one, blocks=blocks, plan=plan)
    many = str(tmp_path / "many.mkv")
    ok2, d2 = ps.xfade_chain_chunked(clips, durs, sections, many, str(tmp_path), blocks=blocks)
    assert ok1 and ok2
    a, b = _frames_of(one), _frames_of(many)
    assert a.shape == b.shape, (a.shape, b.shape)
    diff = [i for i in range(len(a)) if not np.array_equal(a[i], b[i])]
    assert not diff, f"кадры отличаются: {diff[:10]}"
    assert d2 == pytest.approx(d1, abs=1e-6)
    # переходы не нулевые: иначе тест не проверял бы стыки
    assert any(d > 1.5 / ps.FPS for _t, d in plan)
