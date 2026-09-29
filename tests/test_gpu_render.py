#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU-рендер фото-клипа (scripts/gpu_render.py): каждый шаг сверяется с
самим ffmpeg, а не с ожиданием автора. Считается torch на процессоре
(GPU_RENDER_DEVICE=cpu) — та же математика, что на видеокарте."""
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.argv = ["pipeline_smart.py", REPO]
import gpu_render as gr  # noqa: E402

torch = pytest.importorskip("torch")
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="нет ffmpeg")


def _ff(args, inp=None):
    return subprocess.run(["ffmpeg", "-v", "error", "-y"] + args, input=inp, capture_output=True,
                          check=True).stdout


def _smooth_image(w, h, seed=0):
    """Гладкая картинка с деталями — дебандингу и резкости есть что делать."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.stack([128 + 60 * np.sin(xx / 17 + c) * np.cos(yy / 23 - c) for c in (0, 1, 2)], -1)
    return np.clip(base + rng.normal(0, 3, base.shape), 0, 255).astype(np.uint8)


def _psnr(a, b):
    d = a.astype(float) - b.astype(float)
    return 10 * np.log10(255 ** 2 / max((d ** 2).mean(), 1e-12))


def test_known_tail_is_the_real_film_look_tail():
    """Правка хвоста грейда в film_look() обязана упасть здесь, а не молча
    отправить клипы на процессор (или, хуже, считать на карте старый хвост)."""
    import pipeline_smart as ps
    for section in ("HOOK", "BLOCK_1", "FINAL"):
        fl = ps.film_look(123, section, 0.01, 0.2, levels=(0.04, 0.95), wb=(0.5, 0.45, 0.4))
        assert gr.split_film_look(fl) is not None, section
    assert gr.split_film_look("eq=contrast=1,vignette=PI/5,unsharp=3:3:1") is None


def test_motion_expressions_match_python_math():
    import pipeline_smart as ps
    e = ps.piecewise_ease_expr("(on/48)", [(0.0, 0.0), (0.5, 0.72), (0.85, 0.95), (1.0, 1.0)])
    f = gr.ff_expr(f"'1.04+0.08*{e}'")
    assert f(on=0) == pytest.approx(1.04)
    assert f(on=48) == pytest.approx(1.12)
    assert f(on=24) == pytest.approx(1.04 + 0.08 * 0.72, abs=1e-4)
    with pytest.raises(ValueError):
        gr.ff_expr("__import__('os')")


@needs_ffmpeg
def test_deband_and_unsharp_are_bit_exact_with_ffmpeg(tmp_path):
    W, H = 96, 64
    img = _smooth_image(W, H)
    img[::8] = 250                     # резкие края: резкости есть что усиливать
    img[:, ::11] = 5
    raw = img.tobytes()
    yuv = np.frombuffer(_ff(["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-i", "-",
                             "-vf", "format=yuv420p", "-f", "rawvideo", "-"], raw), np.uint8)
    ref = np.frombuffer(_ff(["-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{W}x{H}", "-i", "-",
                             "-vf", "deband=range=22:1thr=0.04:2thr=0.04:3thr=0.04:4thr=0.04,"
                                    "unsharp=5:5:0.45:5:5:0.0", "-f", "rawvideo", "-"], yuv.tobytes()),
                        np.uint8)
    ops = gr._Ops("cpu", W, H)
    n = W * H
    c = (W // 2) * (H // 2)
    planes = [torch.from_numpy(yuv[:n].reshape(1, H, W).astype(np.float32)),
              torch.from_numpy(yuv[n:n + c].reshape(1, H // 2, W // 2).astype(np.float32)),
              torch.from_numpy(yuv[n + c:].reshape(1, H // 2, W // 2).astype(np.float32))]
    planes = [ops.deband(planes[0], 0), ops.deband(planes[1], 1), ops.deband(planes[2], 1)]
    planes[0] = ops.unsharp(planes[0])
    got = np.concatenate([p[0].numpy().astype(np.uint8).ravel() for p in planes])
    assert (got == ref).all(), f"не совпало пикселей: {(got != ref).sum()}"


@needs_ffmpeg
def test_color_table_matches_the_ffmpeg_chain_on_420_input():
    import pipeline_smart as ps
    W, H = 128, 96
    img = _smooth_image(W, H, 3)
    yuv = _ff(["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-i", "-", "-vf", "format=yuv420p",
               "-f", "rawvideo", "-"], img.tobytes())
    point, _ = gr.split_film_look(ps.film_look(7, "BLOCK_1", 0.02, 0.1, levels=(0.05, 0.93), wb=(0.45, 0.42, 0.40)))
    ref = np.frombuffer(_ff(["-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{W}x{H}", "-i", "-",
                             "-vf", f"{point},format=rgb24", "-f", "rawvideo", "-"], yuv), np.uint8)
    a = np.frombuffer(yuv, np.uint8)
    n, c = W * H, (W // 2) * (H // 2)
    F = torch.nn.functional
    Y = torch.from_numpy(a[:n].reshape(H, W).astype(np.float32))
    U = F.interpolate(torch.from_numpy(a[n:n + c].reshape(1, 1, H // 2, W // 2).astype(np.float32)), size=(H, W),
                      mode="bilinear", align_corners=False)[0, 0]
    V = F.interpolate(torch.from_numpy(a[n + c:].reshape(1, 1, H // 2, W // 2).astype(np.float32)), size=(H, W),
                      mode="bilinear", align_corners=False)[0, 0]
    ops = gr._Ops("cpu", W, H)
    g = ops.lut(gr.capture_yuv_lut(point), torch.stack([Y, U, V])[None] / 255.0, order="yuv")[0]
    got = (g.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).ravel()
    # Синтетика с шумом: 43.2 дБ (замер 29.09); реальное фото — 47.4 дБ.
    # Порог — страж от регресса, не заявка на качество.
    assert _psnr(got, ref) > 42


@needs_ffmpeg
@pytest.mark.parametrize("ext,pix", [("png", "rgb24"), ("jpg", "yuvj420p"), ("jpg", "yuvj422p"),
                                     ("jpg", "yuvj444p")])
def test_whole_clip_matches_the_cpu_render(tmp_path, monkeypatch, ext, pix):
    """Клип целиком: тот же рецепт через ffmpeg и через gpu_render на
    маленьком кадре, для каждой раскладки цвета исходника (ffmpeg строит
    холст в раскладке исходника). Сравнение до кодирования (ffv1)."""
    import pipeline_smart as ps
    monkeypatch.setenv("GPU_RENDER_DEVICE", "cpu")
    W, H, cw, ch, nw, nh, cx0, cy0, frames = 320, 180, 640, 360, 640, 400, 0, 20, 3
    photo = tmp_path / f"p.{ext}"
    img = _smooth_image(400, 250, 5)
    img[40:210:6, :, 0] = 240          # цветные полосы: раскладке цвета есть что путать
    img[:, 30:370:10, 2] = 230
    _ff(["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "400x250", "-i", "-", "-pix_fmt", pix, "-q:v", "2",
         str(photo)], img.tobytes())
    z = "'1.04+0.06*(on/3)'"
    x = "'iw/2-(iw/zoom/2)+3.0*sin(2*PI*(on/3)*0.8)'"
    y = "'ih/2-(ih/zoom/2)'"
    fl = ps.film_look(11, "BLOCK_1", 0.0, 0.0, levels=(0.05, 0.93), wb=(0.45, 0.42, 0.40))
    vf = (f"scale={nw}:{nh},crop={cw}:{ch}:{cx0}:{cy0},setsar=1,zoompan=z={z}:x={x}:y={y}:d={frames}:"
          f"s={W}x{H}:fps=24,{fl}")
    cpu = tmp_path / "cpu.mkv"
    _ff(["-framerate", "1", "-loop", "1", "-i", str(photo), "-vf", vf, "-frames:v", str(frames),
         "-c:v", "ffv1", "-pix_fmt", "yuv420p", str(cpu)])
    gpu = tmp_path / "gpu.mkv"
    ok, why = gr.render_kenburns(str(photo), str(gpu), frames, z, x, y, (nw, nh, cw, ch, cx0, cy0), fl,
                                 ["-c:v", "ffv1", "-pix_fmt", "yuv420p10le", str(gpu)], W=W, H=H)
    assert ok, why

    def dec(f):
        return np.frombuffer(_ff(["-i", str(f), "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"]), np.uint8)
    a, b = dec(cpu), dec(gpu)
    assert a.size == b.size
    n = W * H
    fs = n * 3 // 2
    ys = [_psnr(a[k * fs:k * fs + n], b[k * fs:k * fs + n]) for k in range(frames)]
    uvs = [_psnr(a[k * fs + n:(k + 1) * fs], b[k * fs + n:(k + 1) * fs]) for k in range(frames)]
    print(pix, "клип целиком, яркость по кадрам:", [round(v, 1) for v in ys],
          "цвет:", [round(v, 1) for v in uvs])
    assert min(ys) > 36, ys
    assert min(uvs) > 37, uvs
    # Систематический сдвиг — класс обеих ошибок, найденных сверкой 29.09
    # (двойной перевод диапазона JPEG, округление вместо floor после
    # виньетки): PSNR его почти не видит, среднее — видит.
    bias = np.mean([a[k * fs:k * fs + n].astype(float).mean() - b[k * fs:k * fs + n].astype(float).mean()
                    for k in range(frames)])
    assert abs(bias) < 0.5, bias


def test_kenburns_uses_the_gpu_and_falls_back_on_failure(monkeypatch, tmp_path):
    import pipeline_smart as ps
    import gpu_render
    calls = []
    monkeypatch.setattr(ps, "gpu_render_active", lambda: True)
    monkeypatch.setattr(ps, "focus_crop", lambda p: p)
    monkeypatch.setattr(ps, "aspect_fit_backdrop", lambda p: p)
    monkeypatch.setattr(ps, "estimate_busyness", lambda p: 0.0)
    monkeypatch.setattr(ps, "resolve_crop_anchor", lambda p: None)
    monkeypatch.setattr(ps, "image_size_as_rendered", lambda p: (1600, 900))
    monkeypatch.setattr(ps, "verify_clip", lambda path, dur: (True, "", dur))
    out = tmp_path / "c.mp4"

    def fake(photo, tmp, *a, **k):
        calls.append("gpu")
        open(tmp, "wb").write(b"x")
        return True, ""
    monkeypatch.setattr(gpu_render, "render_kenburns", fake)
    monkeypatch.setattr(ps, "run_ffmpeg_with_retry", lambda *a, **k: (calls.append("cpu"), (False, "x"))[1])
    assert ps.kenburns("p.jpg", str(out), 1.0, section="BLOCK_1") is True
    assert calls == ["gpu"]
    calls.clear()
    monkeypatch.setattr(gpu_render, "render_kenburns", lambda *a, **k: (calls.append("gpu"), (False, "сбой"))[1])
    ps.kenburns("p.jpg", str(out), 1.0, section="BLOCK_1")
    assert calls == ["gpu", "cpu"], "сбой карты — клип идёт процессором"
    calls.clear()
    ps.kenburns("p.jpg", str(out), 1.0, section="BLOCK_1", title="Надпись")
    assert "gpu" not in calls, "надписи пока только процессором"


def test_gray_and_alpha_sources_stay_on_the_cpu(tmp_path):
    for pix in ("gray", "rgba"):
        p = tmp_path / f"{pix}.png"
        _ff(["-f", "lavfi", "-i", "testsrc2=s=64x48", "-frames:v", "1", "-pix_fmt", pix, str(p)])
        assert gr.front_format(str(p)) is None
