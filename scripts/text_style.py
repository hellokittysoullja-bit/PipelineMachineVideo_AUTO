#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Оформление экранного текста: классическая антиква + разреженная подпись.

Утверждено владельцем 25.10 по макетам «было / стало» и видео-примеру
bylo_stalo_v2 (эп.05 для второго канала): вместо широкого гротеска Benzin
капсом — Cormorant Garamond (OFL) строчными, над названием главы мелкая
разреженная строка «ГЛАВА ВТОРАЯ» и тонкая золотистая линия; текст
проявляется из лёгкого размытия в резкость и едва заметно плывёт.

Почему текст рисуется картинкой (PIL), а не drawtext: drawtext не умеет ни
разрядку букв, ни размытие текста. Блок рисуется ОДИН раз в PNG (резкий и
размытый вариант), а в ffmpeg накладывается через movie= — поэтому граф
остаётся одной строкой filter_complex и вызывающему не нужно добавлять
входы. Модуль чистый: картинки и строки фильтров, без запуска ffmpeg.
"""
import hashlib
import os

FONTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "fonts")
SERIF = os.path.join(FONTS_DIR, "CormorantGaramond-SemiBold.ttf")
SERIF_ITALIC = os.path.join(FONTS_DIR, "CormorantGaramond-MediumItalic.ttf")
SANS = os.path.join(FONTS_DIR, "Montserrat-Medium.ttf")

IVORY = (238, 231, 218)
GOLD = (201, 167, 106)
W, H = 1920, 1080
SAFE_BOTTOM = 880          # ниже — зона кнопок плеера YouTube
MARGIN = 96                # 5% ширины кадра

PULL_SEC = 0.7             # проявление из размытия в резкость
OUT_SEC = 0.45
RISE_PX = 12
BLUR_RADIUS = 9

_ORDINALS = ("ПЕРВАЯ", "ВТОРАЯ", "ТРЕТЬЯ", "ЧЕТВЁРТАЯ", "ПЯТАЯ", "ШЕСТАЯ", "СЕДЬМАЯ",
             "ВОСЬМАЯ", "ДЕВЯТАЯ", "ДЕСЯТАЯ", "ОДИННАДЦАТАЯ", "ДВЕНАДЦАТАЯ")
_ROMAN = ("I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII")


def chapter_kicker(n):
    """1 -> 'ГЛАВА ПЕРВАЯ'; вне словаря — 'ГЛАВА 13'."""
    return "ГЛАВА " + (_ORDINALS[n - 1] if 1 <= n <= len(_ORDINALS) else str(n))


def roman(n):
    return _ROMAN[n - 1] if 1 <= n <= len(_ROMAN) else str(n)


def _font(path, size):
    from PIL import ImageFont
    return ImageFont.truetype(path, size)


def _line_width(text, font, tracking):
    if not text:
        return 0.0
    return sum(font.getlength(ch) for ch in text) + tracking * (len(text) - 1)


def text_width(text, path, size, tracking=0.0):
    try:
        return _line_width(text, _font(path, size), tracking)
    except Exception:
        return 0.55 * size * len(text)


def wrap(text, path, size, max_w, max_lines=3):
    """Перенос по словам; None — не влезло в max_lines."""
    words, lines, cur = (text or "").split(), [], ""
    for w in words:
        cand = (cur + " " + w).strip()
        if cur and text_width(cand, path, size) > max_w:
            lines.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return lines if len(lines) <= max_lines else None


def balance_two(text, path, size, max_w):
    """Одна строка, если влезает; иначе две примерно равные по ширине."""
    if text_width(text, path, size) <= max_w:
        return [text]
    words = text.split()
    best = None
    for k in range(1, len(words)):
        a, b = " ".join(words[:k]), " ".join(words[k:])
        w = max(text_width(a, path, size), text_width(b, path, size))
        if best is None or w < best[0]:
            best = (w, [a, b])
    return best[1] if best and best[0] <= max_w else None


def _draw_line(draw, x, y, text, font, fill, tracking):
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill)
        x += font.getlength(ch) + tracking


def _render(items, width, height, shadow=True):
    """items: [(kind, ...)] в координатах холста width x height.
    ('text', x, y, text, path, size, rgba, tracking, align)
    ('rule', x, y, w, h, rgba). Возвращает RGBA Image."""
    from PIL import Image, ImageDraw, ImageFilter
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    mask = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d, dm = ImageDraw.Draw(img), ImageDraw.Draw(mask)
    for it in items:
        if it[0] == "rule":
            _, x, y, w, h, rgba = it
            d.rectangle((x, y, x + w - 1, y + h - 1), fill=rgba)
            continue
        _, x, y, text, path, size, rgba, tracking, align = it
        font = _font(path, size)
        lw = _line_width(text, font, tracking)
        if align == "center":
            x = x - lw / 2.0
        elif align == "right":
            x = x - lw
        _draw_line(d, x, y, text, font, rgba, tracking)
        _draw_line(dm, x, y + 2, text, font, (0, 0, 0, 170), tracking)
    if shadow:
        sh = mask.filter(ImageFilter.GaussianBlur(14))
        out = Image.alpha_composite(sh, img)
        return out
    return img


def _cache_path(tmp_dir, key):
    os.makedirs(tmp_dir, exist_ok=True)
    return os.path.join(tmp_dir, "txt_" + hashlib.md5(key.encode("utf-8")).hexdigest()[:14])


def save_pair(img, tmp_dir, key):
    """(резкий.png, размытый.png) — размытый вариант для проявления."""
    from PIL import ImageFilter
    base = _cache_path(tmp_dir, key)
    sharp, blur = base + ".png", base + "_b.png"
    if not (os.path.exists(sharp) and os.path.exists(blur)):
        img.save(sharp)
        img.filter(ImageFilter.GaussianBlur(BLUR_RADIUS)).save(blur)
    return sharp, blur


# --- блоки ---------------------------------------------------------------------

def title_lines(title, size=104, min_size=72):
    """(кегль, строки) названия заставки или None — не влезло в две строки
    шириной 1640 даже кеглем min_size. Та же вёрстка, что у title_block."""
    while size >= min_size:
        lines = balance_two(title or "", SERIF, size, 1640)
        if lines:
            return size, lines
        size -= 4
    return None


def title_block(title, kicker=None, size=104, min_size=72):
    """Заставка: [разреженная строка + золотая линия] + название по центру
    кадра. Полноразмерный холст 1920x1080 (позиция — центр). None — не влезло."""
    got = title_lines(title, size, min_size)
    if not got:
        return None
    size, lines = got
    lh = int(size * 1.08)
    block_h = lh * len(lines)
    k_size, k_track = 22, 22 * 0.62
    head = (k_size + 26 + 1 + 30) if kicker else 0
    top = (H - (head + block_h)) // 2
    items = []
    if kicker:
        items.append(("text", W // 2 + k_track / 2, top, kicker, SANS, k_size,
                      (*IVORY, 184), k_track, "center"))
        items.append(("rule", W // 2 - 45, top + k_size + 26, 90, 1, (*GOLD, 255)))
    y = top + head
    for ln in lines:
        items.append(("text", W // 2, y - int(size * 0.18), ln, SERIF, size, (*IVORY, 255), 0.0, "center"))
        y += lh
    return _render(items, W, H)


def stat_block(text, size=168):
    """Цифра внизу слева: золотая линия и крупная антиква. Низ с учётом
    выносных элементов (у антиквы «5», «9» опускаются ниже строки) и тени —
    выше SAFE_BOTTOM. Холст 1920x1080."""
    while size > 96 and text_width(text, SERIF, size) > 1100:
        size -= 8
    bb = _font(SERIF, size).getbbox(text)
    y = SAFE_BOTTOM - 22 - bb[3]          # 22 px — запас на размытую тень
    rule_y = y + bb[1] - 20
    items = [("rule", MARGIN + 6, rule_y, 120, 1, (*GOLD, 255)),
             ("text", MARGIN, y, text, SERIF, size, (*IVORY, 255), 0.0, "left")]
    return _render(items, W, H)


def marker_block(n, title):
    """Маркер главы в верхнем левом углу: «II —— ЛАБОРАТОРИЯ В ЛИДСЕ»."""
    rn = roman(n)
    t = (title or "").upper()
    rw = text_width(rn, SERIF, 34)
    track = 17 * 0.42
    items = [("text", MARGIN, 62, rn, SERIF, 34, (*GOLD, 255), 0.0, "left"),
             ("rule", int(MARGIN + rw + 18), 84, 34, 1, (*GOLD, 255)),
             ("text", MARGIN + rw + 70, 74, t, SANS, 17, (*IVORY, 200), track, "left")]
    return _render(items, W, H)


def source_block(line1, line2=None):
    """Подпись источника справа внизу: курсив + разреженная строка
    учреждения, над ними короткая золотая линия. Ширина колонки 300."""
    col_w, x_right = 300, W - 66
    l1 = wrap(line1, SERIF_ITALIC, 30, col_w, 3) or [line1]
    l2 = wrap((line2 or "").upper(), SANS, 14, col_w, 3) if line2 else []
    h = 14 + 1 + len(l1) * 35 + (10 + len(l2) * 22 if l2 else 0)
    top = SAFE_BOTTOM - h
    items = [("rule", x_right - 40, top, 40, 1, (*GOLD, 255))]
    y = top + 14
    for ln in l1:
        items.append(("text", x_right, y, ln, SERIF_ITALIC, 30, (*IVORY, 235), 0.0, "right"))
        y += 35
    y += 10
    for ln in l2 or []:
        items.append(("text", x_right, y, ln, SANS, 14, (*IVORY, 170), 14 * 0.32, "right"))
        y += 22
    return _render(items, W, H)


# --- ffmpeg ------------------------------------------------------------------

def movie_src(path_value, dur, fps, label):
    """Зацикленная картинка как поток (movie=); path_value уже экранирован
    для фильтра (pipeline_smart.ffmpeg_filter_path)."""
    return (f"movie='{path_value}',loop=loop=-1:size=1:start=0,setpts=N/{fps}/TB,"
            f"trim=duration={dur + 1.0 / fps:.3f},format=rgba[{label}]")


def pulled_overlay(in_label, out_label, sharp, blur, t_in, t_out, dur, fps, tag,
                   rise=RISE_PX, drift_x=0.0):
    """Наложение блока с проявлением из размытия: размытая копия входит
    быстро и гаснет за PULL_SEC, резкая проявляется за тот же PULL_SEC и
    поднимается на rise px; после t_out обе гаснут за OUT_SEC. drift_x —
    медленный сдвиг за время показа (живой, а не мёртво стоящий текст)."""
    t_in = max(0.0, float(t_in))
    t_out = max(t_in + PULL_SEC, float(t_out))
    span = max(0.1, t_out + OUT_SEC - t_in)
    y = f"'{rise}*(1-min(max(t-{t_in:.3f}\\,0)/{PULL_SEC:.3f}\\,1))'" if rise else "0"
    x = f"'{drift_x}*min(max(t-{t_in:.3f}\\,0)/{span:.3f}\\,1)'" if drift_x else "0"
    en = f"between(t,{t_in:.3f},{t_out + OUT_SEC + 0.05:.3f})"
    return ";".join([
        movie_src(sharp, dur, fps, f"{tag}s0"),
        movie_src(blur, dur, fps, f"{tag}b0"),
        f"[{tag}s0]fade=t=in:st={t_in:.3f}:d={PULL_SEC:.3f}:alpha=1,"
        f"fade=t=out:st={t_out:.3f}:d={OUT_SEC:.3f}:alpha=1[{tag}s]",
        f"[{tag}b0]fade=t=in:st={t_in:.3f}:d={PULL_SEC * 0.35:.3f}:alpha=1,"
        f"fade=t=out:st={t_in + PULL_SEC * 0.35:.3f}:d={PULL_SEC * 0.75:.3f}:alpha=1[{tag}b]",
        f"[{in_label}][{tag}b]overlay=x={x}:y={y}:eval=frame:format=yuv420p10:enable='{en}'[{tag}m]",
        f"[{tag}m][{tag}s]overlay=x={x}:y={y}:eval=frame:format=yuv420p10:enable='{en}'[{out_label}]",
    ])


def signature_source():
    import inspect
    import sys
    return inspect.getsource(sys.modules[__name__])
