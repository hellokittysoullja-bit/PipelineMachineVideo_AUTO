#!/usr/bin/env python3
"""Увеличение рисунка для крупных планов: Real-ESRGAN (аниме, лёгкая версия) +
смешивание с Lanczos.

Почему смесь, а не чистый ESRGAN: нейросеть на рисованной линии чуть
«пластилинит» штриховку и зерно бумаги. Низкие частоты берутся у Lanczos
(геометрия и цвет ровно исходные), высокие — у ESRGAN (чистая линия) и
частично у Lanczos (родное зерно). Замер на кадрах демо (04.10): без ореолов
и без мыла на 2.5x; критики подтвердили.

Модель: xiongjie/lightweight-real-ESRGAN-anime, RealESRGAN_x4plus_anime_4B32F.onnx
(MIT, 5 МБ), качается при первом вызове и сверяется по sha256. Нет сети/
onnxruntime — чистый Lanczos с громким предупреждением (кадр не пропадает).
Обработка кусками 256 px с перекрытием 24 — без стыков и без роста памяти
(целиком 2528x1696 съедало всю память контейнера)."""
import hashlib
import os
import sys
import urllib.request

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_URL = ("https://huggingface.co/xiongjie/lightweight-real-ESRGAN-anime/resolve/main/"
             "RealESRGAN_x4plus_anime_4B32F.onnx")
MODEL_SHA = "2208c7ae8db793330abf1248fbce15585ad317e921c456265572836b92926c9a"
MODEL_PATH = os.path.join(ROOT, "models", "esrgan", "RealESRGAN_x4plus_anime_4B32F.onnx")
TILE, PAD = 256, 24
FLAT_DEV = 12            # кусок, где 99.5% пикселей отличаются от его медианы меньше — бумага или ровная
                         # заливка: линий нет, нейросеть там не нужна (замер 04.10: ~45% кусков кадра)
LOW_SIGMA, HI_ESR, HI_LANCZOS = 1.4, 0.85, 0.6
VERSION = 2

_SESSION = None
WARNED = []


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def ensure_model(path=MODEL_PATH):
    if os.path.exists(path) and _sha(path) == MODEL_SHA:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    urllib.request.urlretrieve(MODEL_URL, tmp)
    if _sha(tmp) != MODEL_SHA:
        os.remove(tmp)
        raise RuntimeError("ESRGAN: скачанный файл не совпал по sha256 — не используем")
    os.replace(tmp, path)
    return path


def session():
    """Сессия onnxruntime или None (с предупреждением один раз)."""
    global _SESSION
    if os.environ.get("UPSCALE_NEURAL", "1") == "0":     # явное «без нейросети» (тесты, слабая машина)
        return None
    if _SESSION is None:
        try:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = int(os.environ.get("UPSCALE_THREADS", "1"))
            _SESSION = ort.InferenceSession(ensure_model(), so, providers=["CPUExecutionProvider"])
        except Exception as e:  # noqa: BLE001
            _SESSION = False
            if not WARNED:
                WARNED.append(str(e))
                print(f"  ВНИМАНИЕ: увеличение без нейросети (Lanczos) — {type(e).__name__}: {e}"[:300],
                      file=sys.stderr)
    return _SESSION or None


def _esrgan(a, scale, sess, L):
    """a — HxWx3 uint8 -> (H*scale)x(W*scale)x3 float32, кусками с перекрытием.
    Пустые куски (бумага) берутся из Lanczos L — у рисунка на бумаге это
    большая часть кадра, а результат там тот же."""
    h, w = a.shape[:2]
    out = np.zeros((h*scale, w*scale, 3), np.float32)
    name = sess.get_inputs()[0].name
    src = a.astype(np.float32)/255
    for ty in range(0, h, TILE):
        for tx in range(0, w, TILE):
            a0, b0 = max(0, ty - PAD), max(0, tx - PAD)
            a1, b1 = min(h, ty + TILE + PAD), min(w, tx + TILE + PAD)
            ey, ex = min(ty + TILE, h), min(tx + TILE, w)
            t_ = a[a0:a1, b0:b1].astype(np.int16)
            dev = np.abs(t_ - np.median(t_.reshape(-1, 3), axis=0).astype(np.int16)).max(2)
            if float(np.percentile(dev, 99.5)) < FLAT_DEV:
                out[ty*scale:ey*scale, tx*scale:ex*scale] = L[ty*scale:ey*scale, tx*scale:ex*scale]
                continue
            y = sess.run(None, {name: src[a0:a1, b0:b1].transpose(2, 0, 1)[None]})[0][0].transpose(1, 2, 0)
            y = np.clip(y, 0, 1)*255
            if scale != 4:
                y = np.asarray(Image.fromarray(y.round().astype(np.uint8)).resize(
                    ((b1 - b0)*scale, (a1 - a0)*scale), Image.LANCZOS), np.float32)
            out[ty*scale:ey*scale, tx*scale:ex*scale] = \
                y[(ty - a0)*scale:(ey - a0)*scale, (tx - b0)*scale:(ex - b0)*scale]
    return out


def upscale(a, scale=2, cache_dir=None):
    """Рисунок в scale (2 или 4) раза: смесь ESRGAN и Lanczos. uint8 -> uint8.
    cache_dir — дисковый кэш по содержимому (повторный рендер не платит временем)."""
    a = np.ascontiguousarray(a, dtype=np.uint8)
    key = None
    if cache_dir:
        neural = os.environ.get("UPSCALE_NEURAL", "1") != "0"
        key = hashlib.sha256(a.tobytes() + f"{a.shape}|{scale}|{VERSION}|{LOW_SIGMA}|{HI_ESR}|{HI_LANCZOS}|{neural}|{FLAT_DEV}"
                             .encode()).hexdigest()[:24]
        p = os.path.join(cache_dir, f"up_{key}.png")
        if os.path.exists(p):
            try:
                return np.asarray(Image.open(p).convert("RGB"))
            except OSError:
                pass
    h, w = a.shape[:2]
    L = np.asarray(Image.fromarray(a).resize((w*scale, h*scale), Image.LANCZOS), np.float32)
    sess = session()
    if sess is None:
        out = L
    else:
        U = _esrgan(a, scale, sess, L)

        def lo(x):
            return ndimage.gaussian_filter(x, (LOW_SIGMA, LOW_SIGMA, 0))
        lL = lo(L)
        out = lL + (U - lo(U))*HI_ESR + (L - lL)*HI_LANCZOS
    out = np.clip(out, 0, 255).round().astype(np.uint8)
    if key:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = os.path.join(cache_dir, f"up_{key}.{os.getpid()}.png")
        Image.fromarray(out).save(tmp)
        os.replace(tmp, os.path.join(cache_dir, f"up_{key}.png"))
    return out
