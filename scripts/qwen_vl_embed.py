#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen3-VL-Embedding-8B — модель зрения отбора (GPU-ветка с 29.09).

ЧТО ЭТО. Мультимодальная модель поиска (Apache-2.0, MMEB-V2 77.9, поиск
картинок 80.0): вектор текста и картинки в одном пространстве, понимает
составную фразу («рука держит кинжал»), а не мешок признаков. С 29.09 — во
всех ролях, где была SigLIP2 (решение владельца): гейты релевантности и вето,
каскад, оценка фразы режиссёром, домен кадра, полка. Пороги — из калибровки
(vision_model.py, scripts/calibrate_vision.py); верх каскада доранжирует
Qwen3-VL-Reranker (qwen_vl_rerank.py).

КАК СЧИТАЕТСЯ — ровно как в официальном коде (QwenLM/Qwen3-VL-Embedding,
src/models/qwen3_vl_embedding.py): диалог «system: инструкция, user:
картинка или текст», картинка приводится к сетке 32 px (smart_resize из
qwen-vl-utils, тот же расчёт), вектор — скрытое состояние ПОСЛЕДНЕГО
токена по маске внимания, нормированное. Картинке — инструкция модели по
умолчанию («Represent the user's input.»), запросу — инструкция поиска.

СКОРОСТЬ. На видеокарте — bf16 (модель так и обучена и так опубликована;
float32 удвоил бы память без выигрыша в качестве) и внимание SDPA. Размер
картинки ограничен QWEN_EMBED_MAX_PIXELS (по умолчанию 256 визуальных
токенов после слияния — превью каскада меньше этого, картинка не
теряет деталей), пачка — QWEN_EMBED_BATCH. Потоки упреждающего отбора зовут
модель из разных нитей. Прогон на видеокарте — по одному через общий замок
видеокарты (ml_device.run), иначе параллельные пачки делили бы видеопамять;
свой замок модуля держит только загрузку и отключение модели (аудит 29.09:
под ним же шли подготовка картинок и токенизация на процессоре, и потоки
ждали друг друга при свободной видеокарте). Пачки те же — числа те же.

Нет torch/transformers/весов/видеокарты — available() ложно; другой модели
на замену нет (решение владельца 29.09): рендер отказывает до начала работы
(vision_model.require_ready)."""
import math
import os
import threading
import unicodedata

MODEL_NAME = os.environ.get("QWEN_EMBED_MODEL", "Qwen/Qwen3-VL-Embedding-8B").strip()
FACTOR = 32                         # patch 16 x merge 2
MIN_PIXELS = 4 * FACTOR * FACTOR
DEFAULT_MAX_PIXELS = 256 * FACTOR * FACTOR
MAX_LENGTH = 8192
IMAGE_INSTRUCTION = "Represent the user's input."
QUERY_INSTRUCTION = "Retrieve images or text relevant to the user's query."
MAX_RATIO = 200

_LOCK = threading.Lock()
_STATE = {"model": None, "processor": None, "device": None, "broken": None}


def max_pixels():
    raw = (os.environ.get("QWEN_EMBED_MAX_PIXELS") or "").strip()
    return int(raw) if raw.isdigit() and int(raw) >= MIN_PIXELS else DEFAULT_MAX_PIXELS


def batch_size():
    raw = (os.environ.get("QWEN_EMBED_BATCH") or "").strip()
    return max(1, int(raw)) if raw.isdigit() else 16


def signature():
    """Часть ключей кэша каскада: модель и размер картинки меняют вектор."""
    return f"{MODEL_NAME}|px{max_pixels()}"


def smart_resize(height, width, factor=FACTOR, min_pixels=MIN_PIXELS, max_pixels_=None):
    """Тот же расчёт, что qwen_vl_utils.vision_process.smart_resize."""
    max_pixels_ = max_pixels_ or max_pixels()
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError("слишком вытянутая картинка")
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels_:
        beta = math.sqrt((height * width) / max_pixels_)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _to_rgb(im):
    from PIL import Image
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        return bg
    return im.convert("RGB")


def prepare_image(im):
    """RGB и размер по сетке модели — как fetch_image в qwen-vl-utils."""
    im = _to_rgb(im)
    w, h = im.size
    rh, rw = smart_resize(h, w)
    return im.resize((rw, rh))


def _instruction(text):
    text = (text or "").strip()
    if text and not unicodedata.category(text[-1]).startswith("P"):
        text += "."
    return text


def conversation(text=None, image=None, instruction=None):
    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    if text is not None:
        content.append({"type": "text", "text": text})
    if not content:
        content.append({"type": "text", "text": "NULL"})
    return [{"role": "system", "content": [{"type": "text", "text": _instruction(
                instruction or IMAGE_INSTRUCTION)}]},
            {"role": "user", "content": content}]


def pool_last(hidden, mask):
    """Скрытое состояние последнего токена по маске (дополнение справа)."""
    import torch
    last = mask.flip(dims=[1]).argmax(dim=1)
    col = mask.shape[1] - last - 1
    row = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[row, col]


def embedding_model_class():
    """Класс модели, как в официальном коде: Qwen3VLModel внутри self.model.
    Веса опубликованы с префиксом model.* (model.language_model.*,
    model.visual.*), и этот класс грузит их тем же путём, что официальный
    Qwen3VLForEmbedding."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (Qwen3VLModel,
                                                                Qwen3VLPreTrainedModel)

    class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
        _checkpoint_conversion_mapping = {}
        accepts_loss_kwargs = False

        def __init__(self, config):
            super().__init__(config)
            self.model = Qwen3VLModel(config)
            self.post_init()

        def forward(self, **kwargs):
            return self.model(**kwargs)
    return Qwen3VLForEmbedding


def _load():
    if _STATE["model"] is not None or _STATE["broken"]:
        return _STATE["model"] is not None
    try:
        import torch
        from transformers import AutoProcessor
        import ml_device
        dev = ml_device.device()
        if dev != "cuda":
            # Без видеокарты 8B-модель грузилась на процессор в fp32 — ~32 ГБ
            # памяти и часы на эпизод (аудит 28.09) — лучше отказ до начала.
            raise RuntimeError(f"нужна видеокарта CUDA, устройство моделей: {dev}")
        dtype = torch.bfloat16
        kwargs = {"torch_dtype": dtype}
        if dev == "cuda":
            kwargs["attn_implementation"] = "sdpa"
        model = embedding_model_class().from_pretrained(MODEL_NAME, **kwargs)
        model = model.to(dev).eval()
        processor = AutoProcessor.from_pretrained(MODEL_NAME, padding_side="right")
        _STATE.update(model=model, processor=processor, device=dev)
        return True
    except Exception as e:  # noqa: BLE001 — нет модели: решает vision_model.require_ready
        _STATE["broken"] = f"{type(e).__name__}: {e}"[:300]
        print(f"  Qwen3-VL-Embedding недоступна ({_STATE['broken']})")
        return False


def available():
    with _LOCK:
        return _load()


def _encode(conversations, images):
    import torch
    model, processor, dev = _STATE["model"], _STATE["processor"], _STATE["device"]
    if model is None:
        raise RuntimeError("модель отключена другим потоком")
    text = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, images=images or None, truncation=True, max_length=MAX_LENGTH,
                       padding=True, do_resize=False, return_tensors="pt")
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    import ml_device
    with torch.inference_mode():
        out = ml_device.run(lambda: model(**inputs), dev)
        emb = pool_last(out.last_hidden_state, inputs["attention_mask"])
        emb = torch.nn.functional.normalize(emb.float(), p=2, dim=-1)
    return emb.cpu().numpy().astype("float32")


def _fail(e):
    """Сбой модели посреди прогона (нехватка памяти после повтора, картинка,
    которую не разобрал smart_resize): раньше исключение уходило из
    cascade_reorder наружу и могло уронить слот (аудит 28.09). Теперь модель
    выключается до конца прогона громко: гейты дальше не судят кадры (их
скоры None), каскад оставляет порядок пула.
    Вызывается под _LOCK; второй поток, сорвавшийся на уже отключённой
    модели, ничего не перезаписывает — причина остаётся первой."""
    if _STATE["broken"] and _STATE["model"] is None:
        return None
    _STATE["broken"] = f"{type(e).__name__}: {e}"[:300]
    _STATE["model"] = None
    print(f"  Qwen3-VL-Embedding сорвалась ({_STATE['broken']}) — дальше гейты и каскад без неё")
    return None


def _area(im):
    """Площадь после smart_resize — она же число патчей картинки."""
    w, h = getattr(im, "size", (0, 0))
    return w * h


def embed_images(images):
    """Нормированные векторы картинок (np.ndarray [n, d]) или None."""
    import numpy as np
    with _LOCK:
        if not _load():
            return None
    try:
        prepared = [prepare_image(im) for im in images]
        if not prepared:
            return None
        # Пачки из картинок близкой длины (число патчей): меньше пустого
        # выравнивания в языковой части. Замер 29.09 на RTX PRO 6000:
        # 38.9 -> 48.6 картинки/с, отличие от расчёта «по одной» то же
        # (мин. косинус 0.99603 против 0.99608 у нынешних пачек). Порядок
        # ответа — порядок входа.
        order = sorted(range(len(prepared)), key=lambda k: (_area(prepared[k]), k))
        bs = batch_size()
        parts = [[prepared[k] for k in order[i:i + bs]] for i in range(0, len(order), bs)]
        out = [_encode([conversation(image=im) for im in part], part) for part in parts]
        vecs = np.concatenate(out)
        res = np.empty_like(vecs)
        res[np.asarray(order)] = vecs
        return res
    except Exception as e:  # noqa: BLE001 — см. _fail
        with _LOCK:
            return _fail(e)


def embed_text(text, instruction=QUERY_INSTRUCTION):
    """Нормированный вектор запроса или None."""
    with _LOCK:
        if not _load():
            return None
    try:
        return _encode([conversation(text=text, instruction=instruction)], None)[0]
    except Exception as e:  # noqa: BLE001 — см. _fail
        with _LOCK:
            return _fail(e)
