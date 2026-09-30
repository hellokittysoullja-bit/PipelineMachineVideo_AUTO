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


def _ffmpeg_is_static():
    try:
        r = subprocess.run(["ldd", shutil.which("ffmpeg")], capture_output=True, text=True)
        return "not a dynamic executable" in (r.stdout + r.stderr)
    except Exception:  # noqa: BLE001
        return False


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
    bad = int((got != ref).sum())
    if bad and _ffmpeg_is_static():
        # Статическая сборка ffmpeg несёт свою libm: sinf смещений дебандинга
        # отличается от системной в последнем бите у единичных пикселей
        # (29.09, сборка 4.4.1 johnvansickle: 9 из 9216). ffmpeg из пакета
        # системы (под Runpod, рабочая машина) — та же libm, совпадение
        # побитовое. Расхождение здесь всё равно ловит самопроверка рендера.
        assert bad <= got.size // 200, f"не совпало пикселей: {bad}"
    else:
        assert bad == 0, f"не совпало пикселей: {bad}"


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
    g = ops.lut(ops.lut_tensor(gr.capture_yuv_lut(point)), torch.stack([Y, U, V])[None] / 255.0, order="yuv")[0]
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
    vf = (f"scale={nw}:{nh}:out_range=tv,crop={cw}:{ch}:{cx0}:{cy0},setsar=1,zoompan=z={z}:x={x}:y={y}:d={frames}:"
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
    print(pix, "сдвиг яркости", round(float(bias), 3))
    assert abs(bias) < 0.25, bias


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


@needs_ffmpeg
@pytest.mark.parametrize("mode,op", [("screen", 0.09), ("softlight", 0.1234), ("softlight", 1.0)])
def test_blend_table_is_what_this_ffmpeg_computes(mode, op):
    """Наложение таблицей = blend этого ffmpeg побитово, и роли слоёв (A —
    верхний, первый вход) не перепутаны. 29.09: формула softlight у ffmpeg
    4.4 (образ Runpod) другая, чем у 6.x, — зашитая формула уводила яркость
    зерна до 9.8 уровня."""
    rng = np.random.default_rng(3)
    a = rng.integers(0, 256, (48, 64), dtype=np.uint8)
    b = rng.integers(0, 256, (48, 64), dtype=np.uint8)
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        fa, fb = os.path.join(d, "a.gray"), os.path.join(d, "b.gray")
        open(fa, "wb").write(a.tobytes())
        open(fb, "wb").write(b.tobytes())
        ref = np.frombuffer(_ff(["-f", "rawvideo", "-pix_fmt", "gray", "-s", "64x48", "-i", fa,
                                 "-f", "rawvideo", "-pix_fmt", "gray", "-s", "64x48", "-i", fb,
                                 "-filter_complex", f"blend=all_mode={mode}:all_opacity={op}",
                                 "-f", "rawvideo", "-pix_fmt", "gray", "-"]), np.uint8).reshape(48, 64)
    t = gr.capture_blend_table(mode, op)
    assert (t[a.astype(int), b.astype(int)] == ref).all()


def _clip_setup(tmp_path, grain):
    import pipeline_smart as ps
    W, H, cw, ch, nw, nh, cx0, cy0, frames = 320, 180, 640, 360, 640, 400, 0, 20, 3
    photo = tmp_path / "p.jpg"
    img = _smooth_image(400, 250, 7)
    _ff(["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "400x250", "-i", "-", "-q:v", "2", str(photo)],
        img.tobytes())
    z = "'1.04+0.06*(on/3)'"
    x = "'iw/2-(iw/zoom/2)'"
    y = "'ih/2-(ih/zoom/2)'"
    fl = ps.film_look(5, "BLOCK_1", 0.0, 0.0, levels=(0.05, 0.93), wb=(0.45, 0.42, 0.40))
    vf = (f"scale={nw}:{nh}:out_range=tv,crop={cw}:{ch}:{cx0}:{cy0},setsar=1,zoompan=z={z}:x={x}:y={y}:d={frames}:"
          f"s={W}x{H}:fps=24,{fl}")
    ref = ["ffmpeg", "-v", "error", "-framerate", "1", "-loop", "1", "-i", str(photo)]
    if grain:
        ref += ["-stream_loop", "-1", "-i", grain,
                "-filter_complex", f"[0:v]{vf}[gr_in];[1:v]scale={W}:{H}:flags=bicubic,setsar=1[gr_scaled];"
                                   f"[gr_in][gr_scaled]blend=all_mode=softlight:all_opacity=0.2000[vout]",
                "-map", "[vout]"]
    else:
        ref += ["-vf", vf]
    return dict(photo=str(photo), frames=frames, z=z, x=x, y=y, canvas=(nw, nh, cw, ch, cx0, cy0), fl=fl,
                W=W, H=H, ref=ref)


@pytest.fixture
def fresh_parity(monkeypatch):
    monkeypatch.setattr(gr, "_PARITY", {"ok": set(), "disabled": None})
    monkeypatch.setenv("GPU_RENDER_DEVICE", "cpu")


GRAIN = os.path.join(REPO, "assets", "grain", "grain_loop.mp4")


@needs_ffmpeg
@pytest.mark.skipif(not os.path.exists(GRAIN), reason="нет grain_loop.mp4")
def test_whole_clip_with_grain_matches_and_passes_the_self_check(tmp_path, fresh_parity):
    """Клип с зерном целиком проходит самопроверку против настоящей команды
    процессорного пути — на ЭТОМ ffmpeg."""
    c = _clip_setup(tmp_path, GRAIN)
    trace = {}
    ok, why = gr.render_kenburns(c["photo"], str(tmp_path / "g.mkv"), c["frames"], c["z"], c["x"], c["y"],
                                 c["canvas"], c["fl"], ["-c:v", "ffv1", str(tmp_path / "g.mkv")],
                                 W=c["W"], H=c["H"], grain_path=GRAIN, grain_opacity=0.2,
                                 trace=trace, reference_cmd=c["ref"])
    assert ok, why
    py, puv, bias = trace["parity"]
    print("самопроверка с зерном:", round(py, 1), round(puv, 1), round(bias, 2))
    assert py > 36 and puv > 40 and abs(bias) < 0.5
    assert gr._PARITY["ok"] and not gr._PARITY["disabled"]


@needs_ffmpeg
@pytest.mark.skipif(not os.path.exists(GRAIN), reason="нет grain_loop.mp4")
def test_self_check_turns_the_gpu_path_off_when_it_disagrees(tmp_path, fresh_parity, monkeypatch):
    """Карта считает не то, что ffmpeg этой машины (здесь — чужая формула
    мягкого света, как у 4.4 против 6.x) -> клип НЕ кодируется, GPU-путь
    выключен до конца процесса, следующий клип сразу уходит процессору."""
    real = gr.capture_blend_table

    def other_formula(mode, op):
        t = real(mode, op).astype(int)
        return np.clip(t + 8, 0, 255).astype(np.uint8) if mode == "softlight" else t.astype(np.uint8)
    monkeypatch.setattr(gr, "capture_blend_table", other_formula)
    c = _clip_setup(tmp_path, GRAIN)
    out = tmp_path / "g.mkv"
    ok, why = gr.render_kenburns(c["photo"], str(out), c["frames"], c["z"], c["x"], c["y"], c["canvas"],
                                 c["fl"], ["-c:v", "ffv1", str(out)], W=c["W"], H=c["H"],
                                 grain_path=GRAIN, grain_opacity=0.2, reference_cmd=c["ref"])
    assert not ok and "самопроверка" in why
    assert gr._PARITY["disabled"]
    ok2, why2 = gr.render_kenburns(c["photo"], str(out), c["frames"], c["z"], c["x"], c["y"], c["canvas"],
                                   c["fl"], ["-c:v", "ffv1", str(out)], W=c["W"], H=c["H"],
                                   grain_path=GRAIN, grain_opacity=0.2, reference_cmd=c["ref"])
    assert not ok2 and why2 == gr._PARITY["disabled"]


def test_kenburns_passes_its_own_cpu_command_for_the_self_check(monkeypatch, tmp_path):
    """Сверка идёт с ТОЙ ЖЕ командой, которой рендерит процессорный путь."""
    import pipeline_smart as ps
    import gpu_render
    seen = {}
    monkeypatch.setattr(ps, "gpu_render_active", lambda: True)
    monkeypatch.setattr(ps, "focus_crop", lambda p: p)
    monkeypatch.setattr(ps, "aspect_fit_backdrop", lambda p: p)
    monkeypatch.setattr(ps, "estimate_busyness", lambda p: 0.0)
    monkeypatch.setattr(ps, "resolve_crop_anchor", lambda p: None)
    monkeypatch.setattr(ps, "image_size_as_rendered", lambda p: (1600, 900))
    monkeypatch.setattr(ps, "verify_clip", lambda path, dur: (True, "", dur))

    def fake(photo, tmp, *a, **k):
        seen["ref"] = k["reference_cmd"]
        return False, "стоп"
    monkeypatch.setattr(gpu_render, "render_kenburns", fake)
    cpu_cmds = []

    def fake_run(fn, *a, **k):
        real_run = subprocess.run
        monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: (cpu_cmds.append(cmd), real_run(["true"]))[1])
        try:
            fn()
        finally:
            monkeypatch.setattr(subprocess, "run", real_run)
        return False, "x"
    monkeypatch.setattr(ps, "run_ffmpeg_with_retry", fake_run)
    ps.kenburns("p.jpg", str(tmp_path / "c.mp4"), 1.0, section="BLOCK_1")
    cpu = cpu_cmds[0]
    graph = cpu[2:cpu.index("-frames:v")]
    assert seen["ref"][3:] == graph


@needs_ffmpeg
def test_vignette_with_default_dither_is_bit_exact_across_frames():
    """Виньетка в film_look() идёт с дизерингом по умолчанию: ffmpeg
    добавляет значения своего ЛКГ, сквозного по кадрам. Без него GPU-путь
    был темнее на 0.5 уровня на виньетке и на 0.2 на готовом кадре
    (все 17 фото замера 29.09). Проверка — побитово, и с середины клипа
    (пачка начинается не с нулевого кадра)."""
    W, H = 64, 40
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (4, H, W, 3), dtype=np.uint8)
    ref = np.frombuffer(_ff(["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "24", "-i", "-",
                             "-vf", "vignette=PI/5.000", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                            img.tobytes()), np.uint8).reshape(4, H, W, 3)
    ops = gr._Ops("cpu", W, H)
    x = torch.from_numpy(img.astype(np.float32)).permute(0, 3, 1, 2)
    head = ops.vignette(x[:2], 5.0, frame0=0).permute(0, 2, 3, 1).numpy().astype(np.uint8)
    tail = ops.vignette(x[2:], 5.0, frame0=2).permute(0, 2, 3, 1).numpy().astype(np.uint8)
    assert (np.concatenate([head, tail]) == ref).all()


@needs_ffmpeg
def test_10bit_expansion_is_what_this_ffmpeg_does():
    """Перевод 8 -> 10 бит — таблица, снятая с ffmpeg (6.1 и 4.4 делают v<<2,
    GPU-путь делал (v<<2)|(v>>6): +0.24 уровня яркости после кодирования)."""
    t = gr.capture_10bit_table()
    rng = np.random.default_rng(5)
    y = rng.integers(0, 256, (32, 64), dtype=np.uint8)
    u = rng.integers(0, 256, (16, 32), dtype=np.uint8)
    v = rng.integers(0, 256, (16, 32), dtype=np.uint8)
    ref = np.frombuffer(_ff(["-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", "64x32", "-i", "-",
                             "-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-"],
                            y.tobytes() + u.tobytes() + v.tobytes()), np.uint16)
    mine = t[np.concatenate([y.ravel(), u.ravel(), v.ravel()]).astype(int)]
    assert (mine == ref).all(), "перевод 8 -> 10 бит не совпал с ffmpeg"


@needs_ffmpeg
def test_p010_output_matches_the_planar_10bit_output(tmp_path, monkeypatch):
    """Для NVENC кадры идут сразу в p010le (без переупаковки ffmpeg на
    процессоре) — декодированный результат обязан совпасть с прежним
    входом yuv420p10le кадр в кадр."""
    monkeypatch.setenv("GPU_RENDER_DEVICE", "cpu")
    c = _clip_setup(tmp_path, None)

    def run(args, out):
        ok, why = gr.render_kenburns(c["photo"], out, c["frames"], c["z"], c["x"], c["y"], c["canvas"],
                                     c["fl"], args + [out], W=c["W"], H=c["H"])
        assert ok, why
        return np.frombuffer(_ff(["-i", out, "-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-"]), np.uint16)
    a = run(["-c:v", "ffv1", "-pix_fmt", "yuv420p10le"], str(tmp_path / "a.mkv"))
    # ffv1 p010le не хранит — ffmpeg переведёт в yuv420p10le сам; вход при
    # этом p010le (по "-pix_fmt p010le" в аргументах), что и проверяется.
    b = run(["-c:v", "ffv1", "-pix_fmt", "p010le"], str(tmp_path / "b.mkv"))
    assert a.shape == b.shape and (a == b).all()
