#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen3-VL-Reranker: точная оценка пары «описание кадра — картинка».

ЧТО ЭТО. Модель поиска (qwen_vl_embed) кодирует текст и картинку ПО
ОТДЕЛЬНОСТИ и сравнивает векторы — быстро, но на составной фразе («рука
держит кинжал») это сравнение мешка признаков. Реранкер смотрит на пару
ЦЕЛИКОМ: описание и картинка идут в модель одним диалогом, и модель отвечает
«подходит / не подходит» (Apache-2.0, январь 2026). Дорого — один проход
большой модели на пару, поэтому он доранжирует только верх кучи (каскад) и
проверяет победителя слота (вторая проверка), а не каждого кандидата.

КАК СЧИТАЕТСЯ — ровно как в официальном коде (scripts/qwen3_vl_reranker.py
в репозитории модели на Hugging Face): системное сообщение «Judge whether the
Document meets the requirements...», в сообщении пользователя «<Instruct>:»,
«<Query>:» с текстом и «\\n<Document>:» с картинкой; картинка приводится к
сетке 32 px (smart_resize, как у модели поиска) в пределах 4..1280
визуальных токенов; оценка — сигмоида от проекции скрытого состояния
последнего токена на разность строк lm_head для «yes» и «no». Пары идут по
одной, как в официальном коде, — без дополнения пачки, числа те же.

Только видеокарта CUDA: 2B-модель на процессоре — минуты на слот. Нет
torch/transformers/весов/видеокарты — available() ложно, вызывающий
решает сам (vision_model.require_ready отказывает рендеру до начала).
Сбой посреди прогона выключает реранкер до конца прогона громко; первая
причина остаётся первой.
"""
import hashlib
import json
import os
import threading

import qwen_vl_embed

MODEL_NAME = os.environ.get("QWEN_RERANK_MODEL", "Qwen/Qwen3-VL-Reranker-2B").strip()
# Закреплённые ревизии (аудит 30.09): калибровка порогов (assets/calibration,
# 29.09) снята на этих коммитах — у обеих моделей последний был 16.04.2026.
# Без закрепления новая ревизия на Hugging Face молча сменила бы числа под
# старыми порогами. Другая модель через переменную окружения — без закрепления.
PINNED_REVISIONS = {"Qwen/Qwen3-VL-Reranker-2B": "4bd860ac4f15ad1897a214615cccc700f8f71818"}
REVISION = PINNED_REVISIONS.get(MODEL_NAME)
SYSTEM_PROMPT = ('Judge whether the Document meets the requirements based on the Query and the '
                 'Instruct provided. Note that the answer can only be "yes" or "no".')
DEFAULT_INSTRUCTION = "Given a search query, retrieve relevant candidates that answer the query."
MIN_PIXELS = 4 * qwen_vl_embed.FACTOR * qwen_vl_embed.FACTOR
MAX_PIXELS = 1280 * qwen_vl_embed.FACTOR * qwen_vl_embed.FACTOR
# Версия способа счёта: входит в ключ дискового кэша оценок — смена промпта
# или разрешения не отдаёт старые числа.
RERANK_VERSION = 1

_LOCK = threading.Lock()
_STATE = {"model": None, "processor": None, "linear": None, "device": None, "broken": None}
_MEMO = {}


def signature():
    return f"{MODEL_NAME}|v{RERANK_VERSION}|px{MAX_PIXELS}"


def cache_dir():
    return os.environ.get("RERANK_CACHE_DIR") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp_rerank_cache")


def prepare_image(im):
    """RGB и размер по сетке модели в пределах MIN..MAX_PIXELS — как
    fetch_image в qwen-vl-utils для официального реранкера."""
    im = qwen_vl_embed.within_ratio(qwen_vl_embed._to_rgb(im))
    w, h = im.size
    rh, rw = qwen_vl_embed.smart_resize(h, w, min_pixels=MIN_PIXELS, max_pixels_=MAX_PIXELS)
    return im.resize((rw, rh))


def conversation(query, image, instruction=None):
    """Диалог одной пары — порядок частей как в официальном format_mm_instruction."""
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "text", "text": "<Instruct>: " + (instruction or DEFAULT_INSTRUCTION)},
            {"type": "text", "text": "<Query>:"},
            {"type": "text", "text": query if query else "NULL"},
            {"type": "text", "text": "\n<Document>:"},
            {"type": "image", "image": image},
        ]},
    ]


def _load():
    if _STATE["model"] is not None or _STATE["broken"]:
        return _STATE["model"] is not None
    try:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        import ml_device
        if ml_device.device() != "cuda":
            raise RuntimeError(f"нужна видеокарта CUDA, устройство моделей: {ml_device.device()}")
        dev = ml_device.device_for("rerank")
        ml_device.require_bf16(torch, dev)
        lm = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_NAME, revision=REVISION, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        processor = AutoProcessor.from_pretrained(MODEL_NAME, revision=REVISION, padding_side="left")
        vocab = processor.tokenizer.get_vocab()
        w = lm.lm_head.weight.data
        linear = torch.nn.Linear(w.shape[1], 1, bias=False)
        with torch.no_grad():
            linear.weight[0] = w[vocab["yes"]] - w[vocab["no"]]
        model = lm.model.to(dev).eval()
        linear = linear.to(dev).to(model.dtype).eval()
        del lm
        _STATE.update(model=model, processor=processor, linear=linear, device=dev)
        return True
    except Exception as e:  # noqa: BLE001 — нет модели: решает вызывающий
        _STATE["broken"] = f"{type(e).__name__}: {e}"[:300]
        print(f"  Qwen3-VL-Reranker недоступен ({_STATE['broken']})")
        return False


def available():
    with _LOCK:
        return _load()


def broken_reason():
    return _STATE["broken"]


def _fail(e):
    if _STATE["broken"] and _STATE["model"] is None:
        return None
    _STATE["broken"] = f"{type(e).__name__}: {e}"[:300]
    _STATE["model"] = None
    print(f"  Qwen3-VL-Reranker сорвался ({_STATE['broken']}) — дальше без него")
    return None


def _inputs(query, image, instruction):
    """Входы одной пары — на процессоре (шаблон диалога, токенизация,
    нарезка картинки на патчи). Не трогает видеокарту и не зависит от
    других пар, поэтому готовится параллельно с проходами соседних пар."""
    processor = _STATE["processor"]
    if processor is None:
        raise RuntimeError("реранкер отключён другим потоком")
    conv = conversation(query, image, instruction)
    text = processor.apply_chat_template([conv], tokenize=False, add_generation_prompt=True)
    return processor(text=text, images=[image], truncation=False, padding=True,
                     do_resize=False, return_tensors="pt")


def _forward(inputs):
    """Проход модели по ОДНОЙ паре — как в официальном коде: пачка с
    дополнением меняла бы числа от соседей по пачке (bf16), а порог
    smart_rerank калибруется на оценках по одной паре."""
    import torch
    import ml_device
    model, linear, dev = _STATE["model"], _STATE["linear"], _STATE["device"]
    if model is None:
        raise RuntimeError("реранкер отключён другим потоком")
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    with torch.inference_mode():
        h = ml_device.run(lambda: model(**inputs).last_hidden_state[:, -1], dev)
        s = torch.sigmoid(linear(h)).squeeze(-1).float()
    return float(s.cpu()[0])


def _score_one(query, image, instruction):
    return _forward(_inputs(query, image, instruction))


# Сколько пар готовится на процессоре одновременно (см. _inputs). Проходы
# по видеокарте от этого не меняются: они всё равно по одному.
PREP_WORKERS = 4


def _digest(im):
    return hashlib.sha1(im.tobytes() + repr(im.size).encode()).hexdigest()


def _key(query, digest, instruction):
    raw = json.dumps([signature(), instruction or DEFAULT_INSTRUCTION, query, digest],
                     ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _cached(key):
    v = _MEMO.get(key)
    if v is not None:
        return v
    fp = os.path.join(cache_dir(), key[:2], key + ".json")
    try:
        with open(fp, encoding="utf-8") as f:
            v = float(json.load(f)["score"])
        _MEMO[key] = v
        return v
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _store(key, v):
    _MEMO[key] = v
    d = os.path.join(cache_dir(), key[:2])
    fp = os.path.join(d, key + ".json")
    try:
        os.makedirs(d, exist_ok=True)
        part = f"{fp}.{os.getpid()}.{threading.get_ident()}.part"
        with open(part, "w", encoding="utf-8") as f:
            json.dump({"score": v}, f)
        os.replace(part, fp)
    except OSError:
        pass


def score(query, images, instruction=None):
    """Оценки 0..1 пар (query, картинка) — по одной на картинку, в том же
    порядке; картинки — PIL.Image или пути. None — реранкера нет или он
    сорвался (вызывающий решает сам). Нечитаемый файл — None на его месте."""
    from PIL import Image
    with _LOCK:
        if not _load():
            return None
    try:
        prepared = []
        for img in images:
            try:
                if isinstance(img, str):
                    with Image.open(img) as im:
                        pil = prepare_image(im)
                else:
                    pil = prepare_image(img)
            except Exception:  # noqa: BLE001 — нечитаемый файл: оценки нет
                prepared.append(None)
                continue
            key = _key(query, _digest(pil), instruction)
            prepared.append((key, pil, _cached(key)))
        todo = [p for p in prepared if p is not None and p[2] is None]
        fresh = {}
        if todo:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(min(PREP_WORKERS, len(todo))) as ex:
                futs = [ex.submit(_inputs, query, pil, instruction) for _k, pil, _v in todo]
                for (key, _pil, _v), fut in zip(todo, futs):
                    if key not in fresh:
                        fresh[key] = _forward(fut.result())
                        _store(key, fresh[key])
        return [None if p is None else (p[2] if p[2] is not None else fresh[p[0]])
                for p in prepared]
    except Exception as e:  # noqa: BLE001 — см. _fail
        with _LOCK:
            return _fail(e)
