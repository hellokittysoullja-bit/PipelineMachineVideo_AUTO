#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WeMM-Embedding-9B — модель ранжирования каскада (CASCADE_MODEL=wemm9b).

ЗАЧЕМ. Замер 29.09 на 72 размеченных кучах двух ниш (эп.94 — история, 49
куч; эп.93 — глубоководье, 23 кучи), разметка первых 20 доведена слепой
доразметкой до полной, контроль согласия с прежней разметкой 80% точно и
100% в пределах ±1:

                          годных в первых 20     лучший кадр кучи дошёл
                          эп.94     эп.93          эп.94     эп.93
  SigLIP2-base            277       332            39/44     20/23
  WeMM-9B                 324       354            41/44     22/23
  WeMM-9B + реранкер 30   365       354            44/46*    23/23

  * знаменатель другой: там лучший кадр считается только среди кадров с
    превью (две кучи эп.94 без превью у лучшего кадра).

По каждой из 72 куч связка «WeMM + реранкер 30» лучший кадр НЕ теряет ни
против SigLIP2-base, ни против Qwen3-VL-Embedding + реранкер 24, ни против
WeMM без реранкера. Реранкер 40 на эп.93 теряет одну кучу — поэтому 30.
Замеры, листы разметки и числа — scratchpad/ens (analyze5.py, final_eval.py).

КАК СЧИТАЕТСЯ — ровно как в замере: sentence-transformers с кодом модели
на закреплённой ревизии (REVISION; код модели исполняется с Hugging Face —
ревизия закреплена, чтобы он не сменился без нас), bf16, картинка — EXIF-
поворот, RGB, уменьшение до MAX_PIXELS (bicubic), документ — картинка,
запрос — текст; векторы нормированы.

НЕСКОЛЬКО ВИДЕОКАРТ. На каждой видеокарте — своя копия модели, вызов берёт
свободную карту, большая пачка делится между всеми картами. Модель одна и
та же, вектор кадра от карты не зависит (в пределах bf16 — как и сейчас от
состава пачки), время делится на число карт.

Нет torch / sentence-transformers / весов / видеокарты — available() ложно
и каскад считает прежней моделью (решает вызывающий)."""
import os
import queue
import threading

MODEL_NAME = "tencent/WeMM-Embedding-9B"
REVISION = "00c52839de57a6d4fd5b78cf5522ccf0ac8ea482"
MAX_PIXELS = 256 * 32 * 32
# Версия способа счёта: входит в ключи кэша каскада.
VERSION = 1

_LOCK = threading.Lock()
_STATE = {"models": {}, "free": None, "broken": None}
_TEXT = {}


def selected():
    """Каскад ранжирует WeMM-9B (CASCADE_MODEL=wemm9b в .env). Одно правило
    для каскада (pipeline_smart.cascade_model) и для готовности моделей
    (vision_model): иначе рендер мог бы стартовать без модели, которой будет
    ранжировать."""
    raw = (os.environ.get("CASCADE_MODEL") or "").strip().lower()
    return raw in ("wemm", "wemm9b", "wemm-9b")


def batch_size():
    raw = (os.environ.get("WEMM_BATCH") or "").strip()
    return max(1, int(raw)) if raw.isdigit() else 32


def signature():
    return f"{MODEL_NAME}@{REVISION[:12]}|px{MAX_PIXELS}|v{VERSION}"


def broken_reason():
    return _STATE["broken"]


def devices():
    """Карты для копий модели: WEMM_GPUS (число) или все видимые."""
    import ml_device
    n = ml_device.cuda_count()
    raw = (os.environ.get("WEMM_GPUS") or "").strip()
    if raw.isdigit() and int(raw) > 0:
        n = min(n, int(raw))
    return [f"cuda:{k}" for k in range(n)]


def prepare_image(im):
    """Как в замере: EXIF-поворот, RGB, не больше MAX_PIXELS (bicubic)."""
    import math
    from PIL import Image, ImageOps
    try:
        im = ImageOps.exif_transpose(im)
    except Exception:  # noqa: BLE001 — битый EXIF: как есть
        pass
    im = im.convert("RGB")
    w, h = im.size
    if w * h > MAX_PIXELS:
        f = math.sqrt(MAX_PIXELS / (w * h))
        im = im.resize((max(32, int(w * f)), max(32, int(h * f))), Image.BICUBIC)
    return im


def _load():
    if _STATE["models"]:
        return True
    if _STATE["broken"]:
        return False
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        devs = devices()
        if not devs:
            raise RuntimeError("нет видеокарты CUDA")
        free = queue.Queue()
        for dev in devs:
            m = SentenceTransformer(MODEL_NAME, revision=REVISION, trust_remote_code=True,
                                    device=dev, model_kwargs={"torch_dtype": torch.bfloat16})
            m.eval()
            _STATE["models"][dev] = m
            free.put(dev)
        _STATE["free"] = free
        return True
    except Exception as e:  # noqa: BLE001 — нет модели: решает вызывающий
        _STATE["models"].clear()
        _STATE["broken"] = f"{type(e).__name__}: {e}"[:300]
        print(f"  WeMM-9B недоступна ({_STATE['broken']})")
        return False


def available():
    with _LOCK:
        return _load()


def _fail(e):
    if not _STATE["broken"]:
        _STATE["broken"] = f"{type(e).__name__}: {e}"[:300]
        print(f"  WeMM-9B сорвалась ({_STATE['broken']})")
    _STATE["models"].clear()
    return None


def _encode_on_free_device(fn):
    import ml_device
    free = _STATE["free"]
    dev = free.get()
    try:
        m = _STATE["models"].get(dev)
        if m is None:
            raise RuntimeError("модель выгружена")
        return ml_device.run(lambda: fn(m), dev)
    finally:
        free.put(dev)


def _release_cache():
    """Вернуть карте кэш аллокатора после пачки: процесс отбора держал 38.5
    ГиБ из 47 (прогон 29.09), а рендер клипа на карте получал OOM на 190 МиБ
    и откатывался на процессор. Кэш — не нужные модели данные, только
    зарезервированная память."""
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 — освобождение кэша необязательно
        pass


def embed_images(images):
    """Нормированные векторы картинок (np.ndarray [n, d], float32) или None.
    Пачка делится на куски по batch_size() и идёт на все свободные карты
    одновременно; порядок векторов — порядок картинок."""
    import numpy as np
    with _LOCK:
        if not _load():
            return None
    if not images:
        return np.zeros((0, 0), np.float32)
    try:
        prepared = [prepare_image(im) for im in images]
        bs = batch_size()
        # Пачки из картинок близкого размера: меньше пустых токенов в пачке.
        # Замер 29.09 (RTX PRO 6000, 2067 превью): 25.4 -> 30.9 карт/с вместе
        # с быстрыми ядрами; векторы с эталоном — медиана косинуса 0.9999, как
        # и у обычного порядка (разница — шум bf16 от состава пачки).
        order = sorted(range(len(prepared)),
                       key=lambda k: (prepared[k].size[0] * prepared[k].size[1], k))
        chunks = [[prepared[j] for j in order[k:k + bs]] for k in range(0, len(order), bs)]

        def one(chunk):
            return _encode_on_free_device(lambda m: m.encode_document(
                [{"image": im} for im in chunk], batch_size=len(chunk),
                normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False))

        n_dev = max(1, len(_STATE["models"]))
        if len(chunks) == 1 or n_dev == 1:
            parts = [one(c) for c in chunks]
        else:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(min(n_dev, len(chunks))) as ex:
                parts = list(ex.map(one, chunks))
        _release_cache()
        stacked = np.concatenate([np.asarray(p, np.float32) for p in parts])
        out = np.empty_like(stacked)
        out[np.asarray(order)] = stacked
        return out
    except Exception as e:  # noqa: BLE001 — см. _fail
        with _LOCK:
            return _fail(e)


def embed_text(text):
    """Нормированный вектор запроса или None. Кэш на процесс."""
    import numpy as np
    v = _TEXT.get(text)
    if v is not None:
        return v
    with _LOCK:
        if not _load():
            return None
    try:
        v = _encode_on_free_device(lambda m: m.encode_query(
            [text], batch_size=1, normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False))
        v = np.asarray(v, np.float32)[0]
        _TEXT[text] = v
        return v
    except Exception as e:  # noqa: BLE001 — см. _fail
        with _LOCK:
            return _fail(e)
