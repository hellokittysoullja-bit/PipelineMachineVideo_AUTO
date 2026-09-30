#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Где считать локальные модели отбора (SigLIP2, CLIP эстетики): видеокарта,
если она есть, иначе процессор.

ЗАЧЕМ. До этого модуля все модели отбора считались только на процессоре,
даже на машине с видеокартой: ни одного `.to(device)` в коде не было.
Каскад оценивает до 1000 превью на слот, 0.15-0.19 с на картинку на четырёх
ядрах (замер 26.09), а видеокарта делает это в десятки раз быстрее.

ЧТО НЕ МЕНЯЕТСЯ НА ПРОЦЕССОРЕ. Здесь нет ни одного преобразования для
процессора: модель и входы остаются там, где были, числа — байт в байт
прежние, ключи кэшей и подписи отбора — прежние (tag() пустой). Правка
касается только машин, где torch видит CUDA или MPS.

ТОЧНОСТЬ НА ВИДЕОКАРТЕ. Считается в float32, как на процессоре; TF32 у
CUDA выключен (иначе свёртка входа SigLIP давала бы расхождение около 1e-3).
Остаток — порядок 1e-6, то есть различие возможно только между кадрами,
равными по оценке до шестого знака. Такие эмбеддинги в кэшах не смешиваются
с посчитанными на процессоре: tag() входит в ключи кэша эмбеддингов и в
подпись отбора.

ВЫБОР — флаг ML_DEVICE (реестр feature_flags): auto (по умолчанию: CUDA,
затем MPS, затем процессор), cpu (всегда процессор — откат), cuda, mps
(только это устройство, если оно есть; иначе процессор)."""
import functools

import feature_flags


@functools.lru_cache(maxsize=1)
def device():
    """'cuda', 'mps' или 'cpu'. Решается один раз на процесс."""
    choice = feature_flags.mode("ML_DEVICE")
    if choice == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        return "cpu"
    if choice in ("auto", "cuda") and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if choice in ("auto", "mps") and mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def tag():
    """Часть ключей кэша и подписи: пустая на процессоре (ключи прежние)."""
    d = device()
    return "" if d == "cpu" else f"@{d}"


def place(model):
    """Модель на устройство. На процессоре — та же модель без изменений."""
    d = device()
    return model if d == "cpu" else model.to(d)


def inputs(batch):
    """Входы модели (BatchFeature или dict тензоров) на устройство. На
    процессоре — тот же объект."""
    d = device()
    if d == "cpu":
        return batch
    return {k: (v.to(d) if hasattr(v, "to") else v) for k, v in batch.items()}


def host(tensor):
    """Тензор обратно на процессор (для numpy и кэшей)."""
    return tensor if device() == "cpu" else tensor.cpu()
