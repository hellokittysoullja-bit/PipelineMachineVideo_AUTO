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


# Узлы таблицы грейда — ровно через 3 восьмибитных уровня (255 = 85 * 3).
# Замер 29.09 на реальном кадре против цепочки ffmpeg: n=64 (узлы
# linspace попадают между уровнями неравномерно) — сдвиг -0.12..-0.22 в
# зависимости от n, n=86 — 49.6 дБ и сдвиг -0.02 при снятии за 0.15 с;
# точная n=256 совпадает побитово, но снимается 4.6 с на клип.
YUV_LUT_N = 86


def capture_yuv_lut(chain, n=YUV_LUT_N, block=2, fmt="yuv420p"):
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


def capture_blend_table(mode, opacity):
    """Таблица [256 (A, верхний слой), 256 (B, нижний)] -> uint8, снятая с
    ffmpeg ЭТОЙ машины: blend=all_mode=<mode>:all_opacity=<opacity> на паре
    градиентов. Замер 29.09: формула softlight в ffmpeg 4.4 (образ Runpod)
    другая, чем в 6.x — яркость зерна расходилась до 9.8 уровня, хотя все
    остальные этапы цепочки у 4.4 и 6.1 совпадают. Таблица точна для любой
    версии по построению (0.05 с)."""
    key = ("blend", mode, round(float(opacity), 6))
    with _LUT_LOCK:
        if key in _LUT_CACHE:
            return _LUT_CACHE[key]
    a = np.tile(np.arange(256, dtype=np.uint8), (256, 1))          # A по x
    b = a.T.copy()                                                  # B по y
    # Оба входа — yuv444p с одинаковой плоскостью во всех каналах: blend
    # 8 бит считает каждую плоскость одной и той же функцией.
    def raw(p):
        return p.tobytes() * 3
    fc = (f"[0:v][1:v]blend=all_mode={mode}:all_opacity={float(opacity):.6f},format=yuv444p")
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        fa, fb = os.path.join(d, "a.yuv"), os.path.join(d, "b.yuv")
        open(fa, "wb").write(raw(a))
        open(fb, "wb").write(raw(b))
        out = subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "yuv444p", "-s", "256x256",
                              "-i", fa, "-f", "rawvideo", "-pix_fmt", "yuv444p", "-s", "256x256", "-i", fb,
                              "-filter_complex", fc, "-frames:v", "1", "-f", "rawvideo", "-"],
                             capture_output=True, check=True, timeout=60).stdout
    t = np.frombuffer(out, np.uint8)[:65536].reshape(256, 256)      # [B][A]
    t = np.ascontiguousarray(t.T)                                   # [A][B]
    with _LUT_LOCK:
        _LUT_CACHE[key] = t
    return t


def capture_10bit_table():
    """uint16[256]: как ffmpeg ЭТОЙ машины переводит 8-битный yuv420p в
    yuv420p10le (процессорный путь делает это сам перед кодером). Замер
    29.09: и 6.1, и 4.4 — простой сдвиг v<<2, а GPU-путь писал
    (v<<2)|(v>>6) — кадр ярче на ~0.24 уровня после кодирования."""
    key = ("10bit",)
    with _LUT_LOCK:
        if key in _LUT_CACHE:
            return _LUT_CACHE[key]
    y = np.tile(np.arange(256, dtype=np.uint8), (2, 1))
    c = np.full((1, 128), 128, np.uint8)
    out = subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", "256x2",
                          "-i", "-", "-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-"],
                         input=y.tobytes() + c.tobytes() + c.tobytes(), capture_output=True, check=True,
                         timeout=60).stdout
    t = np.frombuffer(out, np.uint16)[:256].copy()
    with _LUT_LOCK:
        _LUT_CACHE[key] = t
    return t


def deband_offsets(w, h, rng=DEBAND_RANGE):
    """x_pos, y_pos из vf_deband.config_input — frand() через sinf/cosf из
    libm (той же функции, что у ffmpeg), float32 как в C."""
    key = (w, h, rng)
    if key in _DEBAND_CACHE:
        return _DEBAND_CACHE[key]
    try:
        libm = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6")
        for fn in ("sinf", "cosf"):
            getattr(libm, fn).restype = ctypes.c_float
            getattr(libm, fn).argtypes = [ctypes.c_float]

        def sinf(a):
            return np.fromiter((libm.sinf(float(v)) for v in a), dtype=np.float32, count=a.size)

        def cosf(a):
            return np.fromiter((libm.cosf(float(v)) for v in a), dtype=np.float32, count=a.size)
    except OSError:
        # Нет libm для ctypes (Windows): синус numpy в float32. Отличие от
        # sinf сборки ffmpeg — последний бит у единичных пикселей; заметно
        # ли оно, решает самопроверка рендера (parity_gate).
        def sinf(a):
            return np.sin(a.astype(np.float32))

        def cosf(a):
            return np.cos(a.astype(np.float32))
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    arg = (xs[None, :] * np.float32(12.9898) + ys[:, None] * np.float32(78.233)).astype(np.float32).ravel()
    sv = sinf(arg)
    r = (sv * np.float32(43758.545)).astype(np.float32)
    r = (r - np.floor(r)).astype(np.float32)
    dirv = (r * np.float32(2 * math.pi)).astype(np.float32)
    dist = (r * np.float32(rng)).astype(np.int32)
    cs = cosf(dirv)
    sn = sinf(dirv)
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
        self._dith = None
        self._dith_starts = [0]         # состояние ЛКГ в начале кадра k
        self._dith_lock = threading.Lock()
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
    def lut_tensor(self, lut_np):
        """Таблица цвета -> тензор на карте [1,3,n,n,n] (один раз на клип)."""
        return self.t.from_numpy(lut_np.astype(np.float32) / 255.0).to(self.dev).permute(3, 0, 1, 2)[None].contiguous()

    def lut(self, L, x, order="rgb"):
        """L — lut_tensor(). order: 'rgb' — таблица с индексом [b][g][r]
        (Hald), 'yuv' — [y][u][v]. grid_sample берёт координаты (x, y, z) =
        (последний, средний, первый) индекс таблицы."""
        B = x.shape[0]
        if order == "yuv":
            x = x[:, [2, 1, 0]]
        g = (x * 2 - 1).permute(0, 2, 3, 1).reshape(1, 1, B * self.H, self.W, 3)
        out = self.F.grid_sample(L, g, mode="bilinear", align_corners=True)[0, :, 0]
        return out.reshape(3, B, self.H, self.W).permute(1, 0, 2, 3)

    def vignette(self, x, div, frame0=None):
        """vf_vignette на RGB24. frame0=None — только умножение (float, как
        fmap * src в C). frame0=k — целиком, как ffmpeg с дизерингом по
        умолчанию: dst = (int)(float(src*f) + dv), dv — значения ЛКГ
        dither = dither*1664525 + 1013904223 (uint32, с нуля при старте
        фильтра, сквозное по кадрам и каналам R,G,B), делённые на 2^32.
        Замер 29.09: floor без дизеринга давал сдвиг яркости -0.5 на
        виньетке и -0.2 на готовом кадре на всех 17 фото."""
        t = self.t
        c = t.cos((math.pi / div) * self.vig_d)
        f = ((c * c) * (c * c)).float()
        p = x * f[None, None]
        if frame0 is None:
            return p
        # По кадру: значения генератора пачки целиком — ~1 ГиБ int64/float64.
        out = t.empty_like(p)
        for k in range(p.shape[0]):
            dv = self.vignette_dither(frame0 + k, 1)[0]
            out[k] = (p[k].double() + dv).floor().clamp(0, 255).float()
        return out

    def vignette_dither(self, frame0, n):
        """dv [n,3,H,W] (float64) для кадров frame0..frame0+n-1."""
        t = self.t
        if self._dith is None:
            N = self.W * self.H * 3
            a, c = 1664525, 1013904223
            A = np.full(N, a, dtype=np.uint32)
            A[0] = 1
            A = np.cumprod(A, dtype=np.uint32)            # a^i mod 2^32
            C = np.zeros(N, dtype=np.uint32)
            C[1:] = np.cumsum(A[:-1], dtype=np.uint32) * np.uint32(c)   # x_i при x_0 = 0
            aN = int(A[-1]) * a % (1 << 32)
            cN = (int(C[-1]) * a + c) % (1 << 32)
            self._dith = (t.from_numpy(A.astype(np.int64)).to(self.dev),
                          t.from_numpy(C.astype(np.int64)).to(self.dev), aN, cN)
        A, C, aN, cN = self._dith
        with self._dith_lock:
            starts = self._dith_starts
            while len(starts) < frame0 + n:
                starts.append((aN * starts[-1] + cN) % (1 << 32))
            first = starts[frame0:frame0 + n]
        xs = t.tensor(first, device=self.dev, dtype=t.int64)
        v = (A[None, :] * xs[:, None] + C[None, :]) & 0xFFFFFFFF
        v = v.double() / float(1 << 32)
        return v.reshape(n, self.H, self.W, 3).permute(0, 3, 1, 2)

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

    def table(self, table_np):
        """Таблица наложения -> тензор на карте (один раз на клип)."""
        return self.t.from_numpy(np.ascontiguousarray(table_np, dtype=np.float32).ravel()).to(self.dev)

    @staticmethod
    def blend(A, Bv, tab):
        """Наложение слоёв таблицей capture_blend_table (A — верхний, B — нижний)."""
        return tab[(A.long() * 256 + Bv.long()).clamp(0, 65535)]


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


def _canvas_planes(photo, nw, nh, cw, ch, cx0, cy0, fmt):
    """Холст наезда (Y, U, V) — той же строкой фильтров, что начинает
    процессорный путь kenburns(): scale, crop, setsar, в раскладке fmt."""
    vf = f"scale={nw}:{nh},crop={cw}:{ch}:{cx0}:{cy0},setsar=1,format={fmt}"
    raw = subprocess.run(["ffmpeg", "-v", "error", "-framerate", "1", "-loop", "1", "-i", photo,
                          "-vf", vf, "-frames:v", "1", "-f", "rawvideo", "-"],
                         capture_output=True, check=True, timeout=300).stdout
    sx, sy = CHROMA_SHIFT[fmt]
    cwc, chc = -(-cw // (1 << sx)), -(-ch // (1 << sy))
    ny = cw * ch
    if len(raw) < ny + 2 * cwc * chc:
        raise RuntimeError(f"холст: ffmpeg отдал {len(raw)} байт")
    b = np.frombuffer(raw, np.uint8)
    return (b[:ny].reshape(ch, cw), b[ny:ny + cwc * chc].reshape(chc, cwc),
            b[ny + cwc * chc:ny + 2 * cwc * chc].reshape(chc, cwc))


_GRAIN_CACHE = {}


def grain_frames(path, dev, W, H):
    """grain_loop.mp4 -> кадры YUV 4:2:0 1920x1080 на видеокарте (как
    [1:v]scale=W:H:flags=bicubic в нынешней цепочке), один раз на процесс."""
    key = (path, W, H, str(dev))
    if key not in _GRAIN_CACHE:
        import torch
        fr = _decode_yuv420(path, W, H, vf=f"scale={W}:{H}:flags=bicubic")
        # uint8: наложение берёт значения индексом таблицы; float32 держал
        # бы ~1 ГиБ на процесс.
        _GRAIN_CACHE[key] = [tuple(torch.from_numpy(np.array(p)).to(dev) for p in f) for f in fr]
    return _GRAIN_CACHE[key]


# ------------------------------------------------------- самопроверка
# GPU-путь ОБЯЗАН совпадать с процессорным путём ЭТОЙ машины, а формулы
# фильтров меняются между версиями ffmpeg (29.09: softlight в 4.4 другой,
# чем в 6.x — яркость зерна уезжала до 9.8 уровня, и ни один тест в
# контейнере с 6.1 этого не видел). Поэтому первый клип каждой раскладки
# цвета в процессе считается дважды — на карте и командой процессорного
# пути (первая пачка кадров, до кодирования) — и сравнивается. Не совпало —
# GPU-путь выключается до конца процесса, громко, и весь ролик идёт
# процессорным путём. Пороги — с запасом к измеренному: здоровые клипы
# (17 фото, ffmpeg 6.1 и 4.4, до правок холста и таблицы) — яркость от
# 36.8 дБ и сдвиг до 0.36; расхождение формулы давало 19.7-31.6 дБ и сдвиг
# до 9.8.
PARITY_MIN_PSNR_Y = 33.0
PARITY_MIN_PSNR_UV = 38.0
PARITY_MAX_BIAS = 0.5
_PARITY = {"ok": set(), "disabled": None}
_PARITY_LOCK = threading.Lock()


def _parity_needed(ffmt):
    with _PARITY_LOCK:
        return ffmt not in _PARITY["ok"]


def _psnr(d):
    return float(10 * np.log10(255.0 ** 2 / max(float((d * d).mean()), 1e-9)))


def parity_gate(ffmt, got, reference_cmd, W, H, trace=None):
    """got — список кадров [(Y, U, V) uint8] первой пачки GPU-пути.
    reference_cmd — входы и граф процессорного пути. (ok, причина)."""
    n = len(got)
    cmd = list(reference_cmd) + ["-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"]
    try:
        raw = subprocess.run(cmd, capture_output=True, check=True, timeout=600).stdout
    except Exception as e:  # noqa: BLE001 — сверить нечем: доверять карте нельзя
        why = f"самопроверка не состоялась ({type(e).__name__}) — GPU-путь выключен"
        _PARITY["disabled"] = why
        print(f"  ВИДЕОКАРТА: {why}")
        return False, why
    ny, nc = W * H, ((W + 1) // 2) * ((H + 1) // 2)
    fs = ny + 2 * nc
    if len(raw) < n * fs:
        why = f"самопроверка: процессорный путь отдал {len(raw) // fs} кадр(ов) из {n} — GPU-путь выключен"
        _PARITY["disabled"] = why
        print(f"  ВИДЕОКАРТА: {why}")
        return False, why
    ref = np.frombuffer(raw, np.uint8, n * fs).reshape(n, fs).astype(np.float32)
    mine = np.stack([np.concatenate([p.ravel() for p in f]) for f in got]).astype(np.float32)
    dy = mine[:, :ny] - ref[:, :ny]
    duv = mine[:, ny:] - ref[:, ny:]
    py, puv, bias = _psnr(dy), _psnr(duv), float(dy.mean())
    if trace is not None:
        trace["parity"] = (py, puv, bias)
    if py < PARITY_MIN_PSNR_Y or puv < PARITY_MIN_PSNR_UV or abs(bias) > PARITY_MAX_BIAS:
        why = (f"самопроверка не прошла на раскладке {ffmt}: яркость {py:.1f} дБ, цвет {puv:.1f} дБ, "
               f"сдвиг {bias:+.2f} — GPU-путь выключен до конца процесса, рендер на процессоре")
        with _PARITY_LOCK:
            _PARITY["disabled"] = why
        print(f"  ВИДЕОКАРТА: {why}")
        return False, why
    with _PARITY_LOCK:
        _PARITY["ok"].add(ffmt)
    print(f"  видеокарта: самопроверка {ffmt} пройдена (яркость {py:.1f} дБ, цвет {puv:.1f}, сдвиг {bias:+.2f})")
    return True, ""


_OPS_CACHE = {}
_OPS_LOCK = threading.Lock()


def _ops_for(dev, W, H):
    """Индексы дебандинга, карта виньетки и т.п. — одни на процесс и размер
    кадра (строились заново на каждый клип)."""
    key = (str(dev), W, H)
    with _OPS_LOCK:
        if key not in _OPS_CACHE:
            _OPS_CACHE[key] = _Ops(dev, W, H)
        return _OPS_CACHE[key]


# Профиль по этапам (GPU_RENDER_PROFILE=1): с синхронизацией карты после
# каждого этапа — только для замера, в рендере синхронизация убивала бы
# работу внахлёст.
PROFILE = {}
_PROFILE_LOCK = threading.Lock()


class _Prof:
    def __init__(self, dev):
        self.on = os.environ.get("GPU_RENDER_PROFILE") == "1"
        self.dev = dev
        self.t = None

    def mark(self, name):
        if not self.on:
            return
        import time
        if str(self.dev).startswith("cuda"):
            import torch
            torch.cuda.synchronize()
        now = time.perf_counter()
        if self.t is not None and name:
            with _PROFILE_LOCK:
                PROFILE[name] = PROFILE.get(name, 0.0) + (now - self.t)
        self.t = now


def render_kenburns(*args, **kwargs):
    """Любой сбой GPU-пути (нехватка памяти карты при создании буферов,
    драйвер) — (False, причина), и клип идёт процессором: исключение наружу
    уронило бы процесс пула карты целиком."""
    try:
        return _render_kenburns(*args, **kwargs)
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"[:300]


def _render_kenburns(photo, out, frames, z_expr, x_expr, y_expr, canvas, film_look_str,
                     encode_args, fps=24, W=1920, H=1080, grain_path=None, grain_opacity=None,
                     batch=8, trace=None, reference_cmd=None):
    """Отрисовать клип наезда на видеокарте и закодировать. canvas =
    (nw, nh, cw, ch, cx0, cy0) — та же геометрия, что scale/crop в
    kenburns(). encode_args — кодек, частота, цветовые метки, путь
    (как в команде процессорного пути после входа). reference_cmd — входы и
    граф фильтров процессорного пути ЭТОГО клипа (без выхода): по нему
    первый клип каждой раскладки цвета сверяется с ffmpeg этой машины
    (parity_gate). Возвращает (ok, причина)."""
    dev = device()
    if dev is None:
        return False, "нет CUDA"
    if _PARITY["disabled"]:
        return False, _PARITY["disabled"]
    parts = split_film_look(film_look_str)
    if parts is None:
        return False, "строка грейда отличается от известной структуры"
    point, vig_div = parts
    import torch
    F = torch.nn.functional
    nw, nh, cw, ch, cx0, cy0 = canvas
    ops = _ops_for(dev, W, H)
    prof = _Prof(dev)
    ffmt = front_format(photo)
    if ffmt is None:
        return False, "исходник серый или с прозрачностью — процессорный путь"
    sx, sy = CHROMA_SHIFT[ffmt]
    lut_point = ops.lut_tensor(capture_yuv_lut(point, fmt=ffmt))
    lut_hal = ops.lut_tensor(capture_lut(HALATION_POINT))
    tab_10bit = torch.from_numpy(capture_10bit_table().astype(np.int16)).to(dev)
    tab_screen = ops.table(capture_blend_table("screen", HALATION_OPACITY))
    tab_grain = (ops.table(capture_blend_table("softlight", min(1.0, float(grain_opacity))))
                 if grain_path else None)
    zf, xf, yf = ff_expr(z_expr), ff_expr(x_expr), ff_expr(y_expr)

    # Холст строит ТА ЖЕ команда ffmpeg, что у процессорного пути (scale,
    # crop, setsar в раскладке ffmt): он совпадает побитово по построению.
    # Замер 29.09: собственный перевод диапазона JPEG плюс общая матрица
    # «фото -> холст -> окно» расходились со swscale на цветности до -0.6
    # уровня (swscale переводит диапазон и масштабирует одним проходом, при
    # нецелом коэффициенте с собственным округлением). Стоимость — 0.25 с
    # процессора на клип; на карте остаётся масштабирование окна наезда.
    cvs = _canvas_planes(photo, nw, nh, cw, ch, cx0, cy0, ffmt)
    to = lambda a: torch.from_numpy(np.array(a)).to(dev).float()  # noqa: E731
    src = [to(cvs[0]), to(cvs[1]), to(cvs[2])]
    _mats = {}

    def mats(n_in, n_out):
        if (n_in, n_out) not in _mats:
            _mats[(n_in, n_out)] = resample_matrix(n_in, n_out, dev)
        return _mats[(n_in, n_out)]
    grains = grain_frames(grain_path, dev, W, H) if grain_path else None
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-s", f"{W}x{H}",
           "-r", str(fps), "-i", "-"] + list(encode_args)
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    # Запись в кодер — отдельным потоком: карта считает пачку n+1, пока
    # пачка n уходит в трубу кодера. Очередь на две пачки держит память.
    import queue
    q = queue.Queue(maxsize=2)
    werr = []

    def writer():
        while True:
            item = q.get()
            if item is None:
                return
            if werr:
                continue
            ev, host = item
            try:
                if ev is not None:
                    ev.synchronize()
                enc.stdin.write(memoryview(host.numpy()).cast("B"))
            except Exception as e:  # noqa: BLE001 — кодер упал: причина — в его stderr
                werr.append(e)
    wt = threading.Thread(target=writer, daemon=True)
    wt.start()
    cuda = str(dev).startswith("cuda")
    ring = [[None] for _ in range(4)]
    prev = {"zoom": 1.0}
    prof.mark(None)
    try:
        for b0 in range(0, frames, batch):
            if werr:
                raise werr[0]
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
                hc, wc = -(-h >> sy), -(-w >> sx)
                My, Mx = mats(h, H), mats(w, W)
                Ny, Nx = mats(hc, -(-H >> sy)), mats(wc, -(-W >> sx))
                Ys.append((My @ src[0][y:y + h, x:x + w] @ Mx.T)[None, None])
                cy_, cx_ = y >> sy, x >> sx
                Us.append((Ny @ src[1][cy_:cy_ + hc, cx_:cx_ + wc] @ Nx.T)[None, None])
                Vs.append((Ny @ src[2][cy_:cy_ + hc, cx_:cx_ + wc] @ Nx.T)[None, None])
            Y = torch.cat(Ys)[:, 0].round().clamp(0, 255)
            U = torch.cat(Us)[:, 0].round().clamp(0, 255)
            V = torch.cat(Vs)[:, 0].round().clamp(0, 255)
            prof.mark("наезд")
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
            # vf_vignette с дизерингом по умолчанию — тот же ЛКГ, что у ffmpeg.
            base = ops.vignette(ops.q8(g), vig_div, frame0=b0)
            prof.mark("грейд таблицей и виньетка")
            if trace is not None and b0 == 0:
                trace["vignette_rgb"] = base[0]
            # свечение: таблица, /4 бикубика, размытие, обратно, «экран» в YUV 4:2:0
            hi = ops.lut(lut_hal, base / 255.0)
            sm = F.interpolate(hi, size=(H // 4, W // 4), mode="bicubic", antialias=True, align_corners=False)
            gl = F.interpolate(ops.gauss(sm, HALATION_SIGMA), size=(H, W), mode="bicubic", align_corners=False)
            gY, gU, gV = ops.rgb_to_yuv420(ops.q8(gl.clamp(0, 1)))
            bY, bU, bV = ops.rgb_to_yuv420(base)
            planes = [ops.blend(bY, gY, tab_screen), ops.blend(bU, gU, tab_screen),
                      ops.blend(bV, gV, tab_screen)]
            prof.mark("свечение")
            if trace is not None and b0 == 0:
                trace["halation"] = tuple(p_[0] for p_ in planes)
            planes = [ops.deband(planes[0], 0), ops.deband(planes[1], 1), ops.deband(planes[2], 1)]
            if trace is not None and b0 == 0:
                trace["deband"] = tuple(p_[0] for p_ in planes)
            planes[0] = ops.unsharp(planes[0])
            if trace is not None and b0 == 0:
                trace["unsharp"] = tuple(p_[0] for p_ in planes)
            if grains:
                gsel = [grains[on % len(grains)] for on in idx]
                planes = [ops.blend(planes[k], torch.stack([gs[k] for gs in gsel]), tab_grain) for k in range(3)]
            prof.mark("дебандинг, резкость, зерно")
            if b0 == 0 and reference_cmd is not None and _parity_needed(ffmt):
                got = [[pl[k].to(torch.uint8).cpu().numpy() for pl in planes] for k in range(len(idx))]
                ok_p, why_p = parity_gate(ffmt, got, reference_cmd, W, H, trace)
                prof.mark(None)
                if not ok_p:
                    enc.kill()
                    q.put(None)
                    return False, why_p
            # 10 бит — таблицей ffmpeg этой машины (capture_10bit_table); кадры
            # пачки подряд (Y, U, V каждого кадра) — одна пересылка на пачку.
            nb = len(idx)
            packed = torch.cat([pl.long().reshape(nb, -1) for pl in planes], 1)
            packed = tab_10bit[packed.clamp(0, 255)]
            if cuda:
                # Кольцо из 4 закреплённых буферов (очередь 2 + пишется 1 +
                # заполняется 1): выделение закреплённой памяти на каждую
                # пачку стоило бы миллисекунды вызова драйвера.
                slot = ring[(b0 // batch) % len(ring)]
                if slot[0] is None or slot[0].shape[1] != packed.shape[1] or slot[0].shape[0] < nb:
                    slot[0] = torch.empty((batch, packed.shape[1]), dtype=torch.int16, pin_memory=True)
                host = slot[0][:nb]
                host.copy_(packed, non_blocking=True)
                ev = torch.cuda.Event()
                ev.record()
            else:
                host, ev = packed.contiguous(), None
            prof.mark("упаковка и пересылка")
            q.put((ev, host))
        q.put(None)
        wt.join()
        prof.mark("ожидание записи в кодер")
        if werr:
            raise werr[0]
        enc.stdin.close()
        err = enc.stderr.read().decode("utf-8", "replace")
        code = enc.wait()
        prof.mark("кодер: хвост")
        return (code == 0), (err.strip()[-300:] or f"кодер вернул {code}")
    except Exception as e:  # noqa: BLE001 — любой сбой: процессорный путь
        try:
            enc.kill()
        except Exception:  # noqa: BLE001
            pass
        try:
            q.put(None, timeout=30)       # после kill запись падает сразу — поток выйдет
        except Exception:  # noqa: BLE001
            pass
        try:
            err = enc.stderr.read().decode("utf-8", "replace").strip()[-200:]
        except Exception:  # noqa: BLE001
            err = ""
        return False, (f"{type(e).__name__}: {e}"[:200] + (f" | кодер: {err}" if err else ""))


def signature():
    """Часть подписи рецепта рендера: исходник этого модуля."""
    return hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16]
