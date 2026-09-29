#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Рендер фото-клипа Ken Burns на видеокарте (GPU_RENDER=0/1).

ЗАЧЕМ. Клип kenburns() целиком считался фильтрами ffmpeg на процессоре:
холст 8000x4500, zoompan, двенадцать шагов грейда, свечение, дебандинг,
резкость, зерно — на 4 ядрах 60 с видео рендерились 221 с. Здесь та же
цепочка идёт на видеокарте, процессор только раздаёт задания, кодирует
NVENC.

КАК СОВПАДЕНИЕ С НЫНЕШНИМ РЕНДЕРОМ ОБЕСПЕЧЕНО, а не «похоже»:
  * поточечная часть грейда (eq, смешение каналов, eq, кривые,
    избирательный цвет, баланс цвета) — ОДНА функция «цвет -> цвет»; она
    снимается с самого ffmpeg прогоном эталонной картинки со всеми цветами
    (Hald CLUT уровня 12, 144^3 точек, 0.26 с) через ту же строку фильтров
    и применяется таблицей. Замер 29.09 на реальном кадре: 47.0 дБ PSNR
    против ffmpeg на той же цепочке.
  * виньетка — формула vf_vignette (102.7 дБ без дизеринга);
  * дебандинг и резкость — побитовые копии vf_deband (смещения — та же
    frand() через sinf из libm) и vf_unsharp (биномиальное ядро 5x5 в
    целых), совпадение 100% пикселей;
  * наложения «экран» (свечение) и «мягкий свет» (зерно) — целочисленные
    формулы blend_modes.c;
  * раскладка форматов — та же, что ffmpeg выбирает сам в нынешней
    цепочке (проверено по -v verbose): наезд и первый eq в YUV 4:2:0,
    середина грейда в RGB, наложение свечения, дебандинг, резкость и
    зерно — снова в YUV 4:2:0 по плоскостям.
  * геометрия наезда — vf_zoompan: окно int(iw/zoom) x int(ih/zoom), левый
    верхний угол int(x) с выравниванием на чётное, бикубика swscale.
  * выражения движения камеры НЕ переписаны: считаются из тех же строк,
    что уходят в ffmpeg (ff_expr), — правка движения в pipeline_smart
    не может разойтись с GPU-путём.

Что не поддержано (клип идёт прежним путём на процессоре, без ошибки):
надписи и плашки (drawtext), добавочный фильтр Look Management, строка
грейда, отличающаяся от известной структуры (тест держит структуру),
нет torch/CUDA. Отказ называет причину.
"""
import ctypes
import ctypes.util
import hashlib
import math
import os
import re
import subprocess
import threading

import numpy as np

HALD_LEVEL = 12
_LUT_CACHE = {}
_LUT_LOCK = threading.Lock()
_DEBAND_CACHE = {}

# Хвост film_look() после виньетки. GPU-путь повторяет ИМЕННО его; другой
# хвост (правка грейда) — отказ и процессорный путь, а тест сверяет эту
# строку с настоящим film_look(), чтобы правка не прошла мимо.
KNOWN_TAIL = ("split[fl_hi_a][fl_hi_b];[fl_hi_a]curves=all='0/0 0.6/0 0.8/0.22 1/0.6',"
              "colorbalance=rs=0.12:bs=-0.12,scale=iw/4:ih/4,gblur=sigma=3.5[fl_hi_small];"
              "[fl_hi_small][fl_hi_b]scale2ref=flags=bicubic[fl_hi_glow][fl_hi_bref];"
              "[fl_hi_bref][fl_hi_glow]blend=all_mode=screen:all_opacity=0.09,"
              "deband=range=22:1thr=0.04:2thr=0.04:3thr=0.04:4thr=0.04,"
              "unsharp=5:5:0.45:5:5:0.0")
HALATION_POINT = "curves=all='0/0 0.6/0 0.8/0.22 1/0.6',colorbalance=rs=0.12:bs=-0.12"
HALATION_OPACITY = 0.09
HALATION_SIGMA = 3.5
DEBAND_RANGE = 22
DEBAND_THR = 0.04
UNSHARP_AMOUNT = 0.45


def enabled():
    try:
        import feature_flags
        return feature_flags.enabled("GPU_RENDER")
    except Exception:  # noqa: BLE001
        return False


def device():
    """'cuda' или None (GPU-рендер только на видеокарте)."""
    forced = (os.environ.get("GPU_RENDER_DEVICE") or "").strip()
    if forced:
        return forced                     # сверка совпадения на процессоре (тесты)
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------- выражения
_FUNCS = {
    "pow": lambda a, b: float(a) ** float(b), "sin": math.sin, "cos": math.cos,
    "min": min, "max": max, "abs": abs, "sqrt": math.sqrt,
    "lt": lambda a, b: 1.0 if a < b else 0.0, "gt": lambda a, b: 1.0 if a > b else 0.0,
    "lte": lambda a, b: 1.0 if a <= b else 0.0, "gte": lambda a, b: 1.0 if a >= b else 0.0,
    "if_": lambda c, a, b=0.0: a if c else b, "PI": math.pi,
}


def ff_expr(expr):
    """Строка выражения ffmpeg (как в zoompan) -> функция(**переменные).
    Грамматика выражений пайплайна — подмножество, совпадающее с Python:
    числа, + - * /, скобки, вызовы; `\\,` — экранированная запятая."""
    s = expr.strip()
    if len(s) >= 2 and s[0] == s[-1] == "'":
        s = s[1:-1]
    s = s.replace("\\,", ",")
    s = re.sub(r"\bif\(", "if_(", s)
    if re.search(r"[^\w\s\.\+\-\*/\(\),]", s):
        raise ValueError(f"неподдержанный символ в выражении: {expr[:80]}")
    code = compile(s, "<ff_expr>", "eval")

    def f(**v):
        return float(eval(code, {"__builtins__": {}}, {**_FUNCS, **v}))  # noqa: S307 — своя грамматика
    return f


# ---------------------------------------------------------------- грейд
# Раскладка цвета, в которой ffmpeg строит холст и делает наезд, —
# раскладка ИСХОДНИКА (проверено -v verbose 29.09): JPEG 4:2:0 -> yuv420p,
# 4:2:2 -> yuv422p, 4:4:4 и любой RGB/палитра -> yuv444p. Хвост цепочки
# после виньетки всегда yuv420p. Серые и с прозрачностью — процессорный путь.
CHROMA_SHIFT = {"yuv420p": (1, 1), "yuv422p": (1, 0), "yuv444p": (0, 0)}


def front_format(path):
    """Формат начала цепочки для исходника или None (процессорный путь)."""
    try:
        pf = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                             "stream=pix_fmt", "-of", "csv=p=0", path],
                            capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        return None
    if not pf or pf.startswith(("gray", "ya", "yuva", "rgba", "bgra", "argb", "abgr", "gbrap")) \
            or pf.endswith(("a", "a64le", "a64be")):
        return None
    if "420" in pf or pf in ("nv12", "nv21"):
        return "yuv420p"
    if "422" in pf:
        return "yuv422p"
    if "444" in pf or pf.startswith(("rgb", "bgr", "gbr", "pal8", "0rgb", "rgb0", "bgr0", "0bgr")):
        return "yuv444p"
    return None


def split_film_look(fl):
    """(поточечная цепочка, делитель виньетки) или None, если хвост не тот."""
    m = re.match(r"^(.*),vignette=PI/([\d.]+),(.*)$", fl)
    if not m or m.group(3) != KNOWN_TAIL:
        return None
    return m.group(1), float(m.group(2))


def capture_lut(chain, level=HALD_LEVEL):
    """Таблица [n,n,n,3] (индекс b,g,r), снятая с ffmpeg: та же строка
    фильтров применяется к Hald CLUT со всеми цветами."""
    key = (chain, level)
    with _LUT_LOCK:
        if key in _LUT_CACHE:
            return _LUT_CACHE[key]
    n = level * level
    out = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"haldclutsrc=level={level}",
                          "-frames:v", "1", "-vf", f"format=rgb24,{chain},format=rgb24",
                          "-f", "rawvideo", "-"], capture_output=True, check=True, timeout=120).stdout
    lut = np.frombuffer(out, np.uint8).reshape(n, n, n, 3)
    with _LUT_LOCK:
        if len(_LUT_CACHE) > 256:
            _LUT_CACHE.clear()
        _LUT_CACHE[key] = lut
    return lut


def capture_yuv_lut(chain, n=64, block=2, fmt="yuv420p"):
    """Таблица [n,n,n,3] (индекс y,u,v) -> RGB, снятая с ffmpeg на ТОМ ЖЕ
    формате входа, что в нынешнем пути (YUV 4:2:0 после zoompan): каждая
    точка сетки — блок block x block одного цвета, читается центр блока.
    Замер 29.09: перевод 4:2:0 -> RGB у ffmpeg отличается от 4:4:4 на -0.8
    уровня в среднем, таблица с 4:4:4 давала сдвиг яркости всего клипа."""
    key = ("yuv", fmt, chain, n, block)
    with _LUT_LOCK:
        if key in _LUT_CACHE:
            return _LUT_CACHE[key]
    lv = np.round(np.linspace(0, 255, n)).astype(np.uint8)
    Y, U, V = np.meshgrid(lv, lv, lv, indexing="ij")
    pts = n ** 3
    gw = int(math.ceil(math.sqrt(pts)))
    gh = int(math.ceil(pts / gw))

    def grid(a):
        g = np.zeros(gw * gh, np.uint8)
        g[:pts] = a.ravel()
        return g.reshape(gh, gw)
    Yg, Ug, Vg = grid(Y), grid(U), grid(V)
    sx, sy = CHROMA_SHIFT[fmt]
    Yp = np.repeat(np.repeat(Yg, block, 0), block, 1)
    Up = np.repeat(np.repeat(Ug, block >> sy, 0), block >> sx, 1)
    Vp = np.repeat(np.repeat(Vg, block >> sy, 0), block >> sx, 1)
    W_, H_ = gw * block, gh * block
    raw = Yp.tobytes() + Up.tobytes() + Vp.tobytes()
    out = subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", fmt, "-s", f"{W_}x{H_}",
                          "-i", "-", "-vf", f"{chain},format=rgb24", "-f", "rawvideo", "-"],
                         input=raw, capture_output=True, check=True, timeout=300).stdout
    img = np.frombuffer(out, np.uint8).reshape(H_, W_, 3)
    c = block // 2
    lut = img[c::block, c::block].reshape(-1, 3)[:pts].reshape(n, n, n, 3)
    with _LUT_LOCK:
        if len(_LUT_CACHE) > 256:
            _LUT_CACHE.clear()
        _LUT_CACHE[key] = lut
    return lut


def deband_offsets(w, h, rng=DEBAND_RANGE):
    """x_pos, y_pos из vf_deband.config_input — frand() через sinf/cosf из
    libm (той же функции, что у ffmpeg), float32 как в C."""
    key = (w, h, rng)
    if key in _DEBAND_CACHE:
        return _DEBAND_CACHE[key]
    libm = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6")
    for fn in ("sinf", "cosf"):
        getattr(libm, fn).restype = ctypes.c_float
        getattr(libm, fn).argtypes = [ctypes.c_float]
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    arg = (xs[None, :] * np.float32(12.9898) + ys[:, None] * np.float32(78.233)).astype(np.float32).ravel()
    sv = np.fromiter((libm.sinf(float(v)) for v in arg), dtype=np.float32, count=arg.size)
    r = (sv * np.float32(43758.545)).astype(np.float32)
    r = (r - np.floor(r)).astype(np.float32)
    dirv = (r * np.float32(2 * math.pi)).astype(np.float32)
    dist = (r * np.float32(rng)).astype(np.int32)
    cs = np.fromiter((libm.cosf(float(v)) for v in dirv), dtype=np.float32, count=dirv.size)
    sn = np.fromiter((libm.sinf(float(v)) for v in dirv), dtype=np.float32, count=dirv.size)
    res = ((cs * dist).astype(np.int32).reshape(h, w), (sn * dist).astype(np.int32).reshape(h, w))
    _DEBAND_CACHE[key] = res
    return res


# ------------------------------------------------------------ на видеокарте
SWS_BICUBIC_A = -0.6   # swscale SWS_BICUBIC: B=0, C=0.6 (Keys a=-0.6)


def resample_matrix(n_in, n_out, dev, a=SWS_BICUBIC_A):
    """Матрица [n_out, n_in] бикубического масштабирования с ядром swscale
    (при уменьшении ядро растягивается по коэффициенту — сглаживание, как у
    swscale). Края — повтор крайнего пикселя. Замер 29.09: против zoompan
    51.0 дБ по яркости (torch bicubic a=-0.75 — 49.5)."""
    import torch
    s = n_in / n_out
    fs = max(s, 1.0)
    sup = 2.0 * fs
    taps = int(math.ceil(2 * sup)) + 2
    i = torch.arange(n_out, device=dev, dtype=torch.float64)
    c = (i + 0.5) * s - 0.5
    j0 = torch.floor(c - sup) + 1
    j = j0[:, None] + torch.arange(taps, device=dev, dtype=torch.float64)[None, :]
    t = ((j - c[:, None]) / fs).abs()
    w = torch.where(t < 1, (a + 2) * t ** 3 - (a + 3) * t ** 2 + 1,
                    torch.where(t < 2, a * t ** 3 - 5 * a * t ** 2 + 8 * a * t - 4 * a, torch.zeros_like(t)))
    w = w / w.sum(1, keepdim=True)
    A = torch.zeros(n_out, n_in, device=dev, dtype=torch.float64)
    A.scatter_add_(1, j.clamp(0, n_in - 1).long(), w)
    return A.float()


class _Ops:
    """Операции цепочки на torch. Все формулы — из исходников ffmpeg 6.1."""

    def __init__(self, dev, W, H):
        import torch
        self.t, self.dev, self.W, self.H = torch, dev, W, H
        F = torch.nn.functional
        self.F = F
        x = torch.arange(W, device=dev) - W // 2
        y = torch.arange(H, device=dev) - H // 2
        self.vig_d = torch.sqrt((x[None, :].double() ** 2 + y[:, None].double() ** 2)) / math.hypot(W / 2, H / 2)
        xp, yp = deband_offsets(W, H)
        self.db = {}
        for p, (pw, ph) in enumerate(((W, H), ((W + 1) // 2, (H + 1) // 2))):
            xpos = torch.from_numpy(xp[:ph, :pw].copy()).to(dev).long()
            ypos = torch.from_numpy(yp[:ph, :pw].copy()).to(dev).long()
            yy, xx = torch.meshgrid(torch.arange(ph, device=dev), torch.arange(pw, device=dev), indexing="ij")
            idx = []
            for sy, sx in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
                iy = (yy + sy * ypos).clamp(0, ph - 1)
                ix = (xx + sx * xpos).clamp(0, pw - 1)
                idx.append((iy * pw + ix).reshape(-1))
            self.db[p] = idx

    # --- таблица цвета: x float [B,3,H,W] 0..1 (RGB) -> то же
    def lut(self, lut_np, x, order="rgb"):
        """order: 'rgb' — таблица с индексом [b][g][r] (Hald), 'yuv' — [y][u][v].
        grid_sample берёт координаты (x, y, z) = (последний, средний, первый)
        индекс таблицы."""
        t = self.t
        L = t.from_numpy(lut_np.astype(np.float32) / 255.0).to(self.dev).permute(3, 0, 1, 2)[None]
        B = x.shape[0]
        if order == "yuv":
            x = x[:, [2, 1, 0]]
        g = (x * 2 - 1).permute(0, 2, 3, 1).reshape(1, 1, B * self.H, self.W, 3)
        out = self.F.grid_sample(L, g, mode="bilinear", align_corners=True)[0, :, 0]
        return out.reshape(3, B, self.H, self.W).permute(1, 0, 2, 3)

    def vignette(self, x, div):
        c = self.t.cos((math.pi / div) * self.vig_d)
        f = ((c * c) * (c * c)).float()
        return x * f[None, None]

    @staticmethod
    def q8(x):
        return (x * 255.0).round().clamp(0, 255)

    # RGB 0..255 -> YUV 4:2:0 BT.601 limited (как swscale rgb24->yuv420p)
    def rgb_to_yuv420(self, rgb):
        R, G, Bc = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        Y = ((66 * R + 129 * G + 25 * Bc + 128) / 256).floor() + 16
        U = ((-38 * R - 74 * G + 112 * Bc + 128) / 256).floor() + 128
        V = ((112 * R - 94 * G - 18 * Bc + 128) / 256).floor() + 128
        pool = lambda c: self.F.avg_pool2d(c[:, None], 2, ceil_mode=True)[:, 0]  # noqa: E731
        return Y.clamp(0, 255), pool(U).round().clamp(0, 255), pool(V).round().clamp(0, 255)

    def gauss(self, x, sigma):
        t, F = self.t, self.F
        r = int(math.ceil(3 * sigma))
        k = t.exp(-(t.arange(-r, r + 1, device=self.dev, dtype=t.float32) ** 2) / (2 * sigma * sigma))
        k = k / k.sum()
        c = x.shape[1]
        x = F.pad(x, (r, r, 0, 0), mode="replicate")
        x = F.conv2d(x, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
        x = F.pad(x, (0, 0, r, r), mode="replicate")
        return F.conv2d(x, k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)

    def deband(self, plane, p):
        """plane [B,h,w] целые 0..255 -> vf_deband blur=1, thr=int(255*0.04)."""
        B, h, w = plane.shape
        flat = plane.reshape(B, -1)
        refs = [flat[:, i] for i in self.db[p]]
        avg = (refs[0] + refs[1] + refs[2] + refs[3]).div(4, rounding_mode="floor").reshape(B, h, w)
        thr = int(255 * DEBAND_THR)
        return self.t.where((plane - avg).abs() < thr, avg, plane)

    def unsharp(self, Y):
        """vf_unsharp 5:5:0.45 по яркости: биномиальное 5x5 в целых."""
        # float32 точен: все промежуточные — целые < 2^24 (сумма ядра 256*255,
        # разность * amount <= 255*29491). float64 на картах NVIDIA в десятки
        # раз медленнее.
        t, F = self.t, self.F
        k = t.tensor([1., 4., 6., 4., 1.], device=self.dev, dtype=t.float32)
        p = F.pad(Y[:, None].float(), (2, 2, 2, 2), mode="replicate")
        hx = F.conv2d(p, k.view(1, 1, 1, 5))
        b = F.conv2d(hx, k.view(1, 1, 5, 1))[:, 0]
        blur = t.floor((b + 128) / 256)
        amount = int(UNSHARP_AMOUNT * 65536)
        res = Y.float() + t.floor((Y.float() - blur) * amount / 65536)
        return res.clamp(0, 255)

    @staticmethod
    def blend_trunc(top, expr, opacity):
        """dst = top + (expr - top) * opacity, запись в uint8 — усечение к нулю (C)."""
        v = top + (expr - top) * opacity
        return v.trunc().clamp(0, 255)

    def screen(self, A, Bv, opacity):
        return self.blend_trunc(A, 255 - ((255 - A) * (255 - Bv)).div(255, rounding_mode="floor"), opacity)

    def softlight(self, A, Bv, opacity):
        t = self.t
        e = (A * A).div(255, rounding_mode="floor") + 2 * (Bv * (A * (255 - A)).div(255, rounding_mode="floor")).div(255, rounding_mode="floor")
        return self.blend_trunc(A, e.clamp(0, 255), opacity)


def _decode_yuv420(path, w=None, h=None, vf=None, fmt="yuv420p"):
    """Картинка/видео -> список (Y,U,V) uint8 плоскостей в fmt (yuv420p/
    yuv422p/yuv444p, ограниченный диапазон — как ffmpeg переводит сам).
    vf — доп. фильтры до формата."""
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height", "-of", "csv=p=0", path],
                           capture_output=True, text=True, check=True).stdout.strip().split(",")
    W0, H0 = int(probe[0]), int(probe[1])
    chain = (vf + "," if vf else "") + f"format={fmt}"
    if w and h:
        W0, H0 = w, h
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", chain, "-f", "rawvideo", "-"],
                         capture_output=True, check=True).stdout
    sx, sy = CHROMA_SHIFT[fmt]
    cw, ch = -(-W0 // (1 << sx)), -(-H0 // (1 << sy))
    fs = W0 * H0 + 2 * cw * ch
    frames = []
    for k in range(len(raw) // fs):
        b = np.frombuffer(raw, np.uint8, fs, k * fs)
        frames.append((b[:W0 * H0].reshape(H0, W0), b[W0 * H0:W0 * H0 + cw * ch].reshape(ch, cw),
                       b[W0 * H0 + cw * ch:].reshape(ch, cw)))
    return frames


_GRAIN_CACHE = {}


def grain_frames(path, dev, W, H):
    """grain_loop.mp4 -> кадры YUV 4:2:0 1920x1080 на видеокарте (как
    [1:v]scale=W:H:flags=bicubic в нынешней цепочке), один раз на процесс."""
    key = (path, W, H, str(dev))
    if key not in _GRAIN_CACHE:
        import torch
        fr = _decode_yuv420(path, W, H, vf=f"scale={W}:{H}:flags=bicubic")
        _GRAIN_CACHE[key] = [tuple(torch.from_numpy(np.ascontiguousarray(p)).to(dev).float() for p in f)
                             for f in fr]
    return _GRAIN_CACHE[key]


def render_kenburns(photo, out, frames, z_expr, x_expr, y_expr, canvas, film_look_str,
                    encode_args, fps=24, W=1920, H=1080, grain_path=None, grain_opacity=None,
                    batch=8, trace=None):
    """Отрисовать клип наезда на видеокарте и закодировать. canvas =
    (nw, nh, cw, ch, cx0, cy0) — та же геометрия, что scale/crop в
    kenburns(). encode_args — кодек, частота, цветовые метки, путь
    (как в команде процессорного пути после входа). Возвращает (ok, причина)."""
    dev = device()
    if dev is None:
        return False, "нет CUDA"
    parts = split_film_look(film_look_str)
    if parts is None:
        return False, "строка грейда отличается от известной структуры"
    point, vig_div = parts
    import torch
    F = torch.nn.functional
    nw, nh, cw, ch, cx0, cy0 = canvas
    ops = _Ops(dev, W, H)
    ffmt = front_format(photo)
    if ffmt is None:
        return False, "исходник серый или с прозрачностью — процессорный путь"
    sx, sy = CHROMA_SHIFT[ffmt]
    lut_point = capture_yuv_lut(point, fmt=ffmt)
    lut_hal = capture_lut(HALATION_POINT)
    zf, xf, yf = ff_expr(z_expr), ff_expr(x_expr), ff_expr(y_expr)

    # Холст как у процессорного пути: фото -> YUV 4:2:0 ограниченного
    # диапазона (перевод диапазона JPEG делает сам ffmpeg при раскодировании
    # в yuv420p — второй раз его делать нельзя: живая сверка 29.09 поймала
    # двойной перевод, яркость уезжала на 2.75 уровня) -> scale -> crop.
    Yj, Uj, Vj = _decode_yuv420(photo, fmt=ffmt)[0]
    to = lambda a: torch.from_numpy(np.array(a)).to(dev).float()  # noqa: E731
    src = [to(Yj), to(Uj), to(Vj)]
    # Холст 8000x4500 НЕ строится: «фото -> холст -> окно -> кадр» — два
    # линейных масштабирования, их матрицы перемножаются заранее, и каждый
    # кадр — два умножения матриц прямо из исходного фото (без промежуточного
    # округления холста до 8 бит). Цветность — те же матрицы на половинном
    # размере (YUV 4:2:0, как zoompan).
    geo = []
    for p, (sh, sw) in enumerate([(src[0].shape[0], src[0].shape[1]), (src[1].shape[0], src[1].shape[1])]):
        dy_, dx_ = (1, 1) if p == 0 else (1 << sy, 1 << sx)
        Ay = resample_matrix(sh, -(-nh // dy_), dev)[cy0 // dy_:cy0 // dy_ + -(-ch // dy_)]
        Ax = resample_matrix(sw, -(-nw // dx_), dev)[cx0 // dx_:cx0 // dx_ + -(-cw // dx_)]
        geo.append((Ay, Ax))
    _mats = {}

    def mats(n_in, n_out):
        if (n_in, n_out) not in _mats:
            _mats[(n_in, n_out)] = resample_matrix(n_in, n_out, dev)
        return _mats[(n_in, n_out)]
    grains = grain_frames(grain_path, dev, W, H) if grain_path else None
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-s", f"{W}x{H}",
           "-r", str(fps), "-i", "-"] + list(encode_args)
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    prev = {"zoom": 1.0}
    try:
        for b0 in range(0, frames, batch):
            idx = list(range(b0, min(frames, b0 + batch)))
            Ys, Us, Vs = [], [], []
            for on in idx:
                zoom = min(10.0, max(1.0, zf(on=on, iw=cw, ih=ch, zoom=prev["zoom"])))
                w = int(cw * (1.0 / zoom))
                h = int(ch * (1.0 / zoom))
                dx = xf(on=on, iw=cw, ih=ch, zoom=zoom)
                x = int(min(max(dx, 0.0), max(cw - w, 0))) & ~((1 << sx) - 1)
                dy = yf(on=on, iw=cw, ih=ch, zoom=zoom)
                y = int(min(max(dy, 0.0), max(ch - h, 0))) & ~((1 << sy) - 1)
                prev["zoom"] = zoom
                (Ay, Ax), (Cy, Cx) = geo
                My = mats(h, H) @ Ay[y:y + h]
                Mx = mats(w, W) @ Ax[x:x + w]
                hc, wc = -(-h >> sy), -(-w >> sx)
                Ny = mats(hc, -(-H >> sy)) @ Cy[y >> sy:(y >> sy) + hc]
                Nx = mats(wc, -(-W >> sx)) @ Cx[x >> sx:(x >> sx) + wc]
                Ys.append((My @ src[0] @ Mx.T)[None, None])
                Us.append((Ny @ src[1] @ Nx.T)[None, None])
                Vs.append((Ny @ src[2] @ Nx.T)[None, None])
            Y = torch.cat(Ys)[:, 0].round().clamp(0, 255)
            U = torch.cat(Us)[:, 0].round().clamp(0, 255)
            V = torch.cat(Vs)[:, 0].round().clamp(0, 255)
            if trace is not None and b0 == 0:
                trace["zoompan"] = (Y[0], U[0], V[0])
            # Цвет -> полный размер повтором отсчёта (так swscale поднимает
            # цветность перед переводом в RGB: замер 29.09, повтор 45.4 дБ
            # против 43.2 у билинейной), затем таблица YUV -> RGB с цепочкой.
            Uu = U.repeat_interleave(1 << sy, 1).repeat_interleave(1 << sx, 2)[:, :H, :W]
            Vu = V.repeat_interleave(1 << sy, 1).repeat_interleave(1 << sx, 2)[:, :H, :W]
            yuv = torch.stack([Y, Uu, Vu], 1) / 255.0
            g = ops.lut(lut_point, yuv, order="yuv")
            if trace is not None and b0 == 0:
                trace["point_rgb"] = ops.q8(g)[0]
            # vf_vignette пишет floor(v * f) в 8 бит (без дизеринга совпадение
            # 102.7 дБ; округление вместо floor давало сдвиг -0.5 на весь кадр).
            base = ops.vignette(ops.q8(g), vig_div).floor().clamp(0, 255)
            if trace is not None and b0 == 0:
                trace["vignette_rgb"] = base[0]
            # свечение: таблица, /4 бикубика, размытие, обратно, «экран» в YUV 4:2:0
            hi = ops.lut(lut_hal, base / 255.0)
            sm = F.interpolate(hi, size=(H // 4, W // 4), mode="bicubic", antialias=True, align_corners=False)
            gl = F.interpolate(ops.gauss(sm, HALATION_SIGMA), size=(H, W), mode="bicubic", align_corners=False)
            gY, gU, gV = ops.rgb_to_yuv420(ops.q8(gl.clamp(0, 1)))
            bY, bU, bV = ops.rgb_to_yuv420(base)
            planes = [ops.screen(bY, gY, HALATION_OPACITY), ops.screen(bU, gU, HALATION_OPACITY),
                      ops.screen(bV, gV, HALATION_OPACITY)]
            if trace is not None and b0 == 0:
                trace["halation"] = tuple(p_[0] for p_ in planes)
            planes = [ops.deband(planes[0], 0), ops.deband(planes[1], 1), ops.deband(planes[2], 1)]
            if trace is not None and b0 == 0:
                trace["deband"] = tuple(p_[0] for p_ in planes)
            planes[0] = ops.unsharp(planes[0])
            if trace is not None and b0 == 0:
                trace["unsharp"] = tuple(p_[0] for p_ in planes)
            if grains:
                op = min(1.0, float(grain_opacity))
                gsel = [grains[on % len(grains)] for on in idx]
                planes = [ops.softlight(planes[k], torch.stack([gs[k] for gs in gsel]), op) for k in range(3)]
            # 10 бит: (v<<2)|(v>>6), как swscale расширяет 8 бит
            for k in range(len(idx)):
                buf = []
                for pl in planes:
                    v = pl[k].to(torch.int32)
                    buf.append(((v << 2) | (v >> 6)).to(torch.int16).cpu().numpy().tobytes())
                enc.stdin.write(b"".join(buf))
        enc.stdin.close()
        err = enc.stderr.read().decode("utf-8", "replace")
        code = enc.wait()
        return (code == 0), (err.strip()[-300:] or f"кодер вернул {code}")
    except Exception as e:  # noqa: BLE001 — любой сбой: процессорный путь
        try:
            enc.kill()
        except Exception:  # noqa: BLE001
            pass
        return False, f"{type(e).__name__}: {e}"[:300]


def signature():
    """Часть подписи рецепта рендера: исходник этого модуля."""
    return hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16]
