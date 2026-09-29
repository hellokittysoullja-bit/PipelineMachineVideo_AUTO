#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Модель зрения отбора (GPU-ветка): Qwen3-VL вместо SigLIP2 во всех ролях.

РЕШЕНИЕ ВЛАДЕЛЬЦА 29.09. SigLIP2 (base-256 в гейтах и каскаде, so400m + Jina
во второй проверке) убрана из GPU-ветки целиком: «даёт много браков и грубо
считает кандидатов». Вместо неё:

  * Qwen3-VL-Embedding-8B (qwen_vl_embed) — вектор картинки и текста во всех
    местах, где был SigLIP2: гейты релевантности и вето, каскад, оценка фразы
    режиссёром, домен кадра для лука, полка;
  * Qwen3-VL-Reranker-2B (qwen_vl_rerank) — доранжирует верх каскада (его и
    видит судья) и заменяет вторую проверку победителя (была so400m + Jina;
    у Jina к тому же лицензия CC-BY-NC — некоммерческая).

ПОРОГИ — ТОЛЬКО ИЗ КАЛИБРОВКИ. Все пороги гейтов были откалиброваны на числах
SigLIP2; у Qwen шкала сходства другая, и прежние числа с ней пропускали бы
брак или резали годное. Пороги выводит scripts/calibrate_vision.py на
видеокарте по размеченному золотому набору (те же правила, что записаны у
каждого порога: ноль потерь годных и терпимых кадров) и пишет файл
калибровки. Без файла, с файлом от другой модели или без видеокарты рендер
ОТКАЗЫВАЕТСЯ до начала работы (решение владельца: никаких кадров по
непроверенным порогам) — require_ready().

ПЕРЕНОС ШКАЛЫ для констант, которых разметка не покрывает (порог
уверенности домена у лука, веса режиссёра): значение переводится со шкалы
SigLIP2 на шкалу Qwen через среднее и разброс сходства обеих моделей на
ОДНОЙ И ТОЙ ЖЕ матрице «все кадры золотого набора × все их запросы» —
тот же приём, которым проект раньше переносил Jina на шкалу SigLIP2.
Числа SigLIP2 заморожены ниже (сняты 29.09, до удаления модели), числа Qwen
пишет калибровка.
"""
import json
import os

CALIBRATION_SCHEMA = 1
# Версия ПРОТОКОЛА гейтов: инструкции Qwen для запроса и картинки, промпт
# реранкера, состав матриц шкалы. Смена любого из них делает старую
# калибровку недействительной — её числа описывали другие вопросы к модели.
GATE_PROTOCOL_VERSION = 1

DEFAULT_CALIBRATION_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets", "calibration", "vision_qwen3vl.json")

CALIBRATE_COMMAND = "python scripts/calibrate_vision.py"

# Шкалы прежних моделей на золотом наборе (40 кадров эп.01): среднее и
# разброс сходства по полной матрице «кадр × текст» (1600 пар). Сняты 29.09,
# до удаления моделей:
#   * гейты — SigLIP2-base-256, английские запросы набора;
#   * оценка фразы режиссёром — ансамбль SigLIP2-so400m + Jina (0.5/0.5 с
#     z-переносом Jina, ровно формула прежнего sentence_relevance; сверка с
#     прод-функцией на трёх парах — расхождение 6.7e-8), русские фразы.
LEGACY_GATE_SCALE = {"mean": 0.05138, "std": 0.05258}
LEGACY_SENTENCE_SCALE = {"mean": 0.04766, "std": 0.02995}

REQUIRED_THRESHOLDS = ("relevance", "risky_margin", "negative_veto_margin", "smart_rerank")


class NotCalibrated(RuntimeError):
    """Порог спрошен, а калибровки для текущей модели нет."""


def calibration_path():
    return os.environ.get("VISION_CALIBRATION") or DEFAULT_CALIBRATION_PATH


def current_signature():
    import qwen_vl_embed
    import qwen_vl_rerank
    return {"embed": qwen_vl_embed.signature(), "rerank": qwen_vl_rerank.signature(),
            "protocol": GATE_PROTOCOL_VERSION, "schema": CALIBRATION_SCHEMA}


_CACHE = {"path": None, "mtime": None, "data": None, "problem": None}


def _read():
    path = calibration_path()
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        _CACHE.update(path=path, mtime=None, data=None,
                      problem=f"нет файла калибровки моделей зрения ({path})")
        return
    if _CACHE["path"] == path and _CACHE["mtime"] == mtime:
        return
    data, problem = None, None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        problem = f"файл калибровки не читается ({path}): {e}"
    if data is not None:
        want = current_signature()
        got = data.get("signature") or {}
        if got != want:
            problem = (f"калибровка снята для других моделей или другого протокола "
                       f"(в файле {got}, нужно {want})")
            data = None
        else:
            miss = [k for k in REQUIRED_THRESHOLDS
                    if not isinstance((data.get("thresholds") or {}).get(k), (int, float))]
            if miss:
                problem = f"в калибровке нет порогов: {', '.join(miss)}"
                data = None
    _CACHE.update(path=path, mtime=mtime, data=data, problem=problem)


def calibration():
    """Данные калибровки текущих моделей или None (см. calibration_problem)."""
    _read()
    return _CACHE["data"]


def calibration_problem():
    _read()
    return _CACHE["problem"]


def threshold(name, default=None):
    """Порог из калибровки. Нет калибровки — NotCalibrated (рендер до
    этого места не доходит: require_ready). default — только для порогов,
    у которых разметки нет вовсе (частицы): None значит «слой выключен»."""
    cal = calibration()
    if cal is None:
        raise NotCalibrated(f"{calibration_problem()} — запусти {CALIBRATE_COMMAND}")
    v = (cal.get("thresholds") or {}).get(name, default)
    return v


def domain_guard_threshold(name):
    cal = calibration()
    if cal is None:
        raise NotCalibrated(f"{calibration_problem()} — запусти {CALIBRATE_COMMAND}")
    return ((cal.get("thresholds") or {}).get("domain_guard") or {}).get(name)


def _scale(kind):
    cal = calibration()
    if cal is None:
        raise NotCalibrated(f"{calibration_problem()} — запусти {CALIBRATE_COMMAND}")
    s = (cal.get("scale") or {}).get(kind)
    if not s or not s.get("std"):
        raise NotCalibrated(f"в калибровке нет шкалы «{kind}» — запусти {CALIBRATE_COMMAND}")
    return s


def legacy_margin(value, kind="gate"):
    """Разница сходств (порог уверенности, запас), заданная на шкале SigLIP2,
    — на шкале Qwen: умножение на отношение разбросов."""
    legacy = LEGACY_GATE_SCALE if kind == "gate" else LEGACY_SENTENCE_SCALE
    return value * _scale(kind)["std"] / legacy["std"]


def to_legacy_level(value, kind="sentence"):
    """Обратный перенос: сходство Qwen -> шкала SigLIP2. Нужен там, где
    сложение с бонусами, откалиброванными на старой шкале (режиссёр)."""
    legacy = LEGACY_GATE_SCALE if kind == "gate" else LEGACY_SENTENCE_SCALE
    s = _scale(kind)
    return (value - s["mean"]) / s["std"] * legacy["std"] + legacy["mean"]


def readiness(embed=True, rerank=True):
    """Список причин, по которым отбор на Qwen сейчас невозможен (пустой —
    готово). embed/rerank — какие модели прогон реально позовёт (см.
    pipeline_smart.vision_models_needed): модель, которой прогон не
    пользуется, не требуется и не грузится. Грузит нужные модели: они всё
    равно нужны первому слоту, а проверка «видеокарта есть, веса читаются»
    без загрузки ничего не доказывает."""
    if not embed and not rerank:
        return []
    problems = []
    import ml_device
    if ml_device.device() != "cuda":
        problems.append(f"нужна видеокарта CUDA, устройство моделей: {ml_device.device()}")
        return problems
    import qwen_vl_embed
    import qwen_vl_rerank
    if embed and not qwen_vl_embed.available():
        problems.append(f"Qwen3-VL-Embedding не загрузилась: {qwen_vl_embed._STATE['broken']}")
    if rerank and not qwen_vl_rerank.available():
        problems.append(f"Qwen3-VL-Reranker не загрузился: {qwen_vl_rerank.broken_reason()}")
    if calibration() is None:
        problems.append(calibration_problem())
    return problems


def require_ready(embed=True, rerank=True):
    """Отказ рендера ДО начала работы (до платных вызовов), если отбор на
    Qwen невозможен. Решение владельца 29.09: никаких кадров по
    непроверенным порогам и никакого молчаливого отката на другую модель."""
    problems = readiness(embed, rerank)
    if problems:
        raise SystemExit(
            "ОТКАЗ: отбор кадров на Qwen3-VL не готов:\n  - " + "\n  - ".join(problems)
            + f"\nКалибровка порогов (на видеокарте, веса скачаются сами): {CALIBRATE_COMMAND}")
