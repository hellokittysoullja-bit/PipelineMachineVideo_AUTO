#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Кадр вокруг смысловой детали: рамка (доли изображения) -> область 16:9.

ЗАЧЕМ. Архивный рисунок — это страница: миниатюра битвы занимает четверть
листа, остальное — текст и поля (замер 24.09 на эпизоде 94: 7 кадров из 9,
найденных «исследователем», — страницы рукописей и трактатов). Показать
страницу целиком значит показать зрителю мелкую картинку среди текста.
Документалист делает иначе: медленный наезд на ту сцену, о которой
говорит фраза. Рамку сцены находит модель-судья (shot_judge.locate_box),
здесь — только геометрия и вырезка, без сети и без моделей.

Один модуль на всех читателей: рендер (pipeline_smart.focus_crop) и оба
контактных листа (runs_sheet.py, shotlist_contact.py) обязаны показывать
ОДНУ и ту же вырезку, иначе лист врёт о том, что увидит зритель."""
import hashlib
import json
import os

MARGIN = 0.18          # запас вокруг рамки с каждой стороны, доля её размера
MIN_WIDTH_PX = 900     # уже — вырезка на кадре 1920 превращается в мыло
MIN_SHARE = 0.30       # вырезка не уже этой доли ширины изображения: деталь в контексте
PAGE_ASPECT = 1.25     # кадр уже этого (портрет, квадрат) — страница, а не сцена
ASPECT = 16 / 9
NEAR_FULL = 0.97       # вырезка почти во весь кадр — незачем
# Метаданные кадра лежат рядом с файлом (pipeline_smart.media_sidecar_path);
# совпадение суффикса держит тест.
SIDECAR_SUFFIX = ".meta.json"


def crop_region(iw, ih, box, aspect=ASPECT):
    """Область (x0, y0, x1, y1) в пикселях вокруг рамки или None.

    Рамка расширяется на MARGIN с каждой стороны и до пропорции кадра; не
    уже MIN_SHARE ширины изображения и не уже MIN_WIDTH_PX. Изображение
    уже нужной области — берётся наибольшая область нужной пропорции с
    центром на рамке (у узкой страницы — во всю её ширину). None — вырезка
    бессмысленна (уже MIN_WIDTH_PX или почти весь кадр) или рамка в неё не
    входит целиком."""
    x0, y0, x1, y1 = (float(v) for v in box)
    bx0, by0, bx1, by1 = x0 * iw, y0 * ih, x1 * iw, y1 * ih
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    w = (bx1 - bx0) * (1 + 2 * MARGIN)
    h = (by1 - by0) * (1 + 2 * MARGIN)
    w = max(w, h * aspect, MIN_SHARE * iw, MIN_WIDTH_PX)
    h = w / aspect
    if h > ih:
        h = float(ih)
        w = h * aspect
    if w > iw:
        w = float(iw)
        h = min(float(ih), w / aspect)
    if w < MIN_WIDTH_PX or (w >= NEAR_FULL * iw and h >= NEAR_FULL * ih):
        return None
    left = min(max(cx - w / 2, 0.0), iw - w)
    top = min(max(cy - h / 2, 0.0), ih - h)
    # Деталь обязана войти целиком: высокий предмет (кинжал во всю
    # страницу трактата) в полосу 16:9 не помещается, и вырезка показала бы
    # середину клинка без острия и рукояти. Тогда лучше страница целиком
    # (её покажет подложка aspect_fit_backdrop), чем обрубок предмета.
    if bx0 < left - 0.5 or bx1 > left + w + 0.5 or by0 < top - 0.5 or by1 > top + h + 0.5:
        return None
    return (int(round(left)), int(round(top)), int(round(left + w)), int(round(top + h)))


def box_of(path):
    """Рамка смысловой детали из метаданных кадра или None."""
    try:
        with open(path + SIDECAR_SUFFIX, encoding="utf-8") as f:
            box = json.load(f).get("focus_box")
    except (OSError, ValueError, AttributeError):
        return None
    return box if isinstance(box, list) and len(box) == 4 else None


def crop_image(im, box):
    """(изображение, вырезано ли): PIL-картинка, уже повёрнутая по EXIF, ->
    вырезка вокруг рамки или она же. Для листов, которые показывают кадр
    так, как его увидит зритель."""
    region = crop_region(im.width, im.height, box) if box else None
    return (im.crop(region), True) if region else (im, False)


def is_page_like(width, height):
    """Высокий кадр (страница, лист, портрет) — кандидат на вырезку."""
    return width / max(1, height) < PAGE_ASPECT


def crop_file(src, box, out_dir, quality=95):
    """Путь к вырезке src вокруг рамки box или None (вырезка не нужна).
    Кэш по (путь, mtime, размер, рамка, параметры); исключения — наружу,
    fail-open решает вызывающий."""
    from PIL import Image, ImageOps
    st = os.stat(src)
    key = hashlib.md5(json.dumps([os.path.abspath(src), st.st_mtime_ns, st.st_size, list(box),
                                  MARGIN, MIN_WIDTH_PX, MIN_SHARE, ASPECT]).encode()).hexdigest()[:16]
    out = os.path.join(out_dir, key + ".jpg")
    if os.path.exists(out):
        return out
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        region = crop_region(im.width, im.height, box)
        if region is None:
            return None
        os.makedirs(out_dir, exist_ok=True)
        tmp = out + ".tmp.jpg"
        im.crop(region).save(tmp, "JPEG", quality=quality)
    os.replace(tmp, out)
    return out
