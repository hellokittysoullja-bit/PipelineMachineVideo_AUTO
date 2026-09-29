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

# Видеопамять, которую займут веса (bf16, сумма .safetensors на Hugging Face,
# замер 29.09) — проверяется ДО загрузки по СВОБОДНОЙ памяти карты (другие
# процессы, рабочий стол, соседний рендер тоже в счёт). Без проверки на карте
# в 16 ГБ отказ приходил невнятной ошибкой CUDA из глубины загрузки. Модель
# с другим именем (QWEN_EMBED_MODEL/QWEN_RERANK_MODEL) в таблице не значится
# — её размер не угадывается, проверка по ней пропускается.
WEIGHTS_GIB = {"Qwen/Qwen3-VL-Embedding-8B": 15.17, "Qwen/Qwen3-VL-Reranker-2B": 3.96,
               # model.safetensors ревизии 00c52839 (bf16); на КАЖДОЙ карте
               # своя копия (wemm_embed.devices).
               "tencent/WeMM-Embedding-9B": 17.6}
# Запас WeMM-9B сверх весов на карту: пачка 32 превью (замер 29.09, RTX PRO
# 6000: пик 26.4 ГиБ при пачке 64 и 19.8 при 16 вместе с весами).
WEMM_HEADROOM_GIB = 5.0
# Сверх весов: пачка каскада (64 превью через 8B), CLIP эстетики, модель
# глубины, контекст CUDA. Оценка, а не замер (видеокарты в среде, где это
# писалось, нет) — с запасом, чтобы отказ был здесь, а не посреди слота.
HEADROOM_GIB = 2.5
# Реранкер на второй карте (ml_device.device_for): только его активации на
# одну пару и контекст CUDA.
SECOND_GPU_HEADROOM_GIB = 1.0


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
    lack = vram_shortage(embed, rerank)
    if lack:
        problems.append(lack)
        return problems
    if embed and not qwen_vl_embed.available():
        problems.append(f"Qwen3-VL-Embedding не загрузилась: {qwen_vl_embed._STATE['broken']}")
    if rerank and not qwen_vl_rerank.available():
        problems.append(f"Qwen3-VL-Reranker не загрузился: {qwen_vl_rerank.broken_reason()}")
    if embed:
        import wemm_embed
        if wemm_embed.selected() and not wemm_embed.available():
            problems.append(f"WeMM-Embedding-9B (CASCADE_MODEL) не загрузилась: {wemm_embed.broken_reason()}")
    if calibration() is None:
        problems.append(calibration_problem())
    return problems


def vram_need_gib(embed=True, rerank=True, wemm=None):
    """{устройство: ГиБ} свободной видеопамяти, нужной моделям, которые
    прогон позовёт (веса + запас); None — размер какой-то из них неизвестен.
    Эмбеддинг — на cuda:0 вместе с запасом на пачки каскада и прочие модели;
    реранкер — там, где его разместит ml_device.device_for (вторая карта,
    если их две, со своим меньшим запасом на свои активации)."""
    import ml_device
    import qwen_vl_embed
    import qwen_vl_rerank
    if (embed and qwen_vl_embed.MODEL_NAME not in WEIGHTS_GIB) or \
            (rerank and qwen_vl_rerank.MODEL_NAME not in WEIGHTS_GIB):
        return None
    need = {}
    if embed:
        need["cuda:0"] = WEIGHTS_GIB[qwen_vl_embed.MODEL_NAME] + HEADROOM_GIB
    if rerank:
        dev = ml_device.device_for("rerank")
        base = need.get(dev, 0.0)
        if not base:
            base = HEADROOM_GIB if dev == "cuda:0" else SECOND_GPU_HEADROOM_GIB
        need[dev] = base + WEIGHTS_GIB[qwen_vl_rerank.MODEL_NAME]
    import wemm_embed
    if (embed if wemm is None else wemm) and wemm_embed.selected():
        for dev in wemm_embed.devices():
            need[dev] = need.get(dev, 0.0) + WEIGHTS_GIB[wemm_embed.MODEL_NAME] + WEMM_HEADROOM_GIB
    return need or None


def vram_shortage(embed=True, rerank=True):
    """Текст отказа, если на какой-то карте свободной видеопамяти меньше
    нужного, иначе None. Уже загруженные модели свою память заняли — их доля
    не требуется снова."""
    import qwen_vl_embed
    import qwen_vl_rerank
    import wemm_embed
    need = vram_need_gib(embed and qwen_vl_embed._STATE.get("model") is None,
                         rerank and qwen_vl_rerank._STATE.get("model") is None,
                         wemm=embed and not wemm_embed._STATE["models"])
    if not need:
        return None
    try:
        import torch
        short = []
        for dev, gib in sorted(need.items()):
            idx = int(dev.split(":")[1])
            free, total = torch.cuda.mem_get_info(idx)
            if free / 2 ** 30 < gib:
                short.append(f"{torch.cuda.get_device_name(idx)} ({dev}) — свободно "
                             f"{free / 2 ** 30:.1f} из {total / 2 ** 30:.1f} ГиБ, нужно ~{gib:.1f}")
    except Exception:  # noqa: BLE001 — не спросилось: решит сама загрузка
        return None
    if not short:
        return None
    return ("мало видеопамяти для моделей зрения (веса + запас на пачки): "
            + "; ".join(short) + ". Закрыть другие процессы на карте или взять карту от 24 ГБ")


_REQUIRED = {"embed": False, "rerank": False}


def mark_required(embed, rerank):
    """Рендер прошёл require_ready с этими моделями — с этого момента их
    потеря посреди прогона не «пропуск проверки», а стоп (lost())."""
    _REQUIRED.update(embed=bool(embed), rerank=bool(rerank))


def lost():
    """Причина, если модель, которую рендер потребовал на старте, выключилась
    посреди прогона (нехватка видеопамяти после повторов, сбой); иначе None.
    Без этого гейт получал бы None вместо скора и пропускал кадр
    НЕПРОВЕРЕННЫМ — ровно то, что решение владельца 29.09 запрещает."""
    import qwen_vl_embed
    import qwen_vl_rerank
    if _REQUIRED["embed"] and qwen_vl_embed._STATE.get("model") is None \
            and qwen_vl_embed._STATE.get("broken"):
        return f"Qwen3-VL-Embedding: {qwen_vl_embed._STATE['broken']}"
    if _REQUIRED["rerank"] and qwen_vl_rerank._STATE.get("model") is None \
            and qwen_vl_rerank._STATE.get("broken"):
        return f"Qwen3-VL-Reranker: {qwen_vl_rerank._STATE['broken']}"
    if _REQUIRED["embed"]:
        import wemm_embed
        if wemm_embed.selected() and not wemm_embed._STATE["models"] and wemm_embed.broken_reason():
            return f"WeMM-Embedding-9B: {wemm_embed.broken_reason()}"
    return None


def require_ready(embed=True, rerank=True):
    """Отказ рендера ДО начала работы (до платных вызовов), если отбор на
    Qwen невозможен. Решение владельца 29.09: никаких кадров по
    непроверенным порогам и никакого молчаливого отката на другую модель."""
    problems = readiness(embed, rerank)
    if problems:
        raise SystemExit(
            "ОТКАЗ: отбор кадров на Qwen3-VL не готов:\n  - " + "\n  - ".join(problems)
            + f"\nКалибровка порогов (на видеокарте, веса скачаются сами): {CALIBRATE_COMMAND}")
