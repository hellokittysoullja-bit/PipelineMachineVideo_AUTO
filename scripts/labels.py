#!/usr/bin/env python3
"""Русские подписи на кадре — кодом, а не нейросетью.

Нейросеть рисует кадр БЕЗ текста (стрелки и пустые места под подписи — да,
буквы — нет). Подписи кладёт этот модуль, поэтому в них не бывает ни одной
ошибки в буквах.

Шрифт: Shantell Sans ExtraBold (жирный маркер «от руки»); если в подписи есть
знак, которого в нём нет, — Balsamiq Sans Bold. Оба — SIL OFL 1.1
(assets/fonts/OFL-*.txt), коммерческое использование разрешено.
Обводки нет нигде (решение владельца): цвет текста выбирается по фону под
подписью — чёрный на светлом, белый на тёмном.

Где ставить:
  * caption (одна крупная подпись) — нижняя полоса кадра, если она пустая;
    иначе место находит модель со зрением;
  * diagram (подписи к частям схемы) — модель со зрением возвращает рамку
    пустого места для каждой подписи рядом с тем, что она подписывает.
Каждое место проверяется по ПИКСЕЛЯМ исходного кадра: внутри рамки не должно
быть рисунка (доля «чернил» ниже порога). Не пустое — подпись туда не
ставится, кадр идёт в брак, а не на экран с текстом поверх рисунка."""
import base64
import io
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT_PRIMARY = os.path.join(ROOT, "assets", "fonts", "ShantellSans-ExtraBold.ttf")
FONT_FALLBACK = os.path.join(ROOT, "assets", "fonts", "BalsamiqSans-Bold.ttf")
INK_DARK, INK_LIGHT = (29, 29, 31), (255, 255, 255)
LINE_SPACING = 1.08
# Доля «рисунка» в рамке, выше — место не пустое. Замер 30.09 на дудл-кадре:
# пустой фон 0.000-0.004, любой рисунок в рамке (край фигуры, ряд человечков,
# одна линия, облако) 0.065-0.106. Порог с запасом в обе стороны.
MAX_INK_SHARE = 0.015
CAPTION_BAND = (0.05, 0.79, 0.95, 0.97)   # доли кадра: x1, y1, x2, y2
# Рисунок стоит на всю высоту ролика, и наезд камеры (assemble_frames.ZOOM,
# 4%) срезает по 2% сверху и снизу. Рамка подписи не ближе 3% к краю кадра —
# иначе буквы у края уходят за кадр на первой или последней секунде клипа.
EDGE_SAFE = 0.03
CANVAS_ASPECT = 16 / 9
EDGE_VISUAL = 0.015   # там, где наезд не режет: текст вплотную к краю рисунка читается как обрезанный
# Шкала кегля — доля высоты кадра, одна на весь ролик: подпись не прыгает
# от кадра к кадру. Меньше потолка — только если текст не влезает в место.
SIZE_CAPTION = 0.075
SIZE_DIAGRAM = 0.045
LOCATE_VERSION = 1

_CMAP = {}


def _cmap(path):
    if path not in _CMAP:
        from fontTools.ttLib import TTFont
        _CMAP[path] = set(TTFont(path).getBestCmap())
    return _CMAP[path]


def font_for(text):
    """Основной шрифт, если в нём есть все знаки подписи, иначе запасной."""
    need = {ord(c) for c in text if not c.isspace()}
    for path in (FONT_PRIMARY, FONT_FALLBACK):
        if need <= _cmap(path):
            return path
    missing = "".join(sorted(chr(c) for c in need - _cmap(FONT_FALLBACK)))
    raise ValueError(f"знаков нет ни в одном шрифте: {missing!r}")


def _wrap(draw, font, words, max_w):
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=font) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def fit(text, box_w, box_h, font_path, max_size=220, min_size=18):
    """(размер, строки) — самый крупный размер, при котором текст с переносом
    по словам влезает в рамку. Слово длиннее рамки не рвётся: уменьшается шрифт."""
    from PIL import Image, ImageDraw, ImageFont
    draw = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    words = text.split()
    lo, hi, best = min_size, max_size, None
    while lo <= hi:
        size = (lo + hi) // 2
        font = ImageFont.truetype(font_path, size)
        lines = _wrap(draw, font, words, box_w)
        widths = [draw.textlength(ln, font=font) for ln in lines]
        asc, desc = font.getmetrics()
        height = (asc + desc) * (1 + (len(lines) - 1) * LINE_SPACING)
        if max(widths) <= box_w and height <= box_h:
            best, lo = (size, lines), size + 1
        else:
            hi = size - 1
    if best is None:
        raise ValueError(f"подпись «{text}» не влезает в рамку {box_w}x{box_h} даже кеглем {min_size}")
    return best


def ink_share(img, box):
    """Доля пикселей рамки, заметно отличающихся от её основного цвета
    (медианы) — «сколько рисунка внутри». Ровный фон, бумага с лёгкой
    фактурой — около нуля; линия, штриховка, фигура — выше."""
    import numpy as np
    x1, y1, x2, y2 = box
    a = np.asarray(img.convert("RGB").crop((x1, y1, x2, y2)), dtype=np.int16)
    if a.size == 0:
        return 1.0
    med = np.median(a.reshape(-1, 3), axis=0)
    diff = np.abs(a - med).sum(axis=2)
    return float((diff > 90).mean())


def text_color(img, box):
    import numpy as np
    x1, y1, x2, y2 = box
    a = np.asarray(img.convert("L").crop((x1, y1, x2, y2)), dtype=np.float32)
    return INK_DARK if (a.size == 0 or a.mean() >= 110) else INK_LIGHT


def draw(img, box, text, pad=0.06, size_cap=SIZE_CAPTION):
    """Нарисовать подпись в рамке (центр по горизонтали и вертикали), кегль не
    больше size_cap × высота кадра. Возвращает запись (шрифт, кегль, цвет, строки)."""
    from PIL import ImageDraw, ImageFont
    x1, y1, x2, y2 = box
    px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
    inner = (x1 + px, y1 + py, x2 - px, y2 - py)
    path = font_for(text)
    size, lines = fit(text, inner[2] - inner[0], inner[3] - inner[1], path,
                      max_size=max(18, round(size_cap * img.size[1])))
    font = ImageFont.truetype(path, size)
    color = text_color(img, box)
    d = ImageDraw.Draw(img)
    asc, desc = font.getmetrics()
    lh = (asc + desc) * LINE_SPACING
    total = (asc + desc) + lh * (len(lines) - 1)
    y = inner[1] + ((inner[3] - inner[1]) - total) / 2
    cx = (inner[0] + inner[2]) / 2
    for ln in lines:
        d.text((cx - d.textlength(ln, font=font) / 2, y), ln, font=font, fill=color)
        y += lh
    return {"text": text, "font": os.path.basename(path), "size": size, "lines": lines,
            "color": "dark" if color == INK_DARK else "light", "box": list(box)}


# ------------------------------------------------------------------ где ставить

LOCATE_PROMPT = """This is a drawing for an explainer video. It has empty spaces left for {n} short Russian text label(s), which will be added later.
{items}
For EACH label return the empty area where it should be written: plain background with no drawing inside, next to the part of the picture it describes (at the tail of the arrow that points to that part, if there is one). The area must be big enough for the text and must not cover any drawn line, figure or arrow.
Coordinates are 0-1000 relative to the image width and height: [x1, y1, x2, y2].
Reply with JSON only: {{"boxes": {{"1": [x1, y1, x2, y2], ...}}}}"""


def _image_part(img, max_side=1024):
    im = img.convert("RGB").copy()
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}}


def parse_boxes(answer, n):
    m = re.search(r"\{.*\}", answer or "", re.S)
    if not m:
        return None
    try:
        raw = json.loads(m.group(0)).get("boxes") or {}
    except ValueError:
        return None
    out = {}
    for k in range(1, n + 1):
        b = raw.get(str(k))
        if not (isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b)):
            return None
        x1, y1, x2, y2 = (max(0.0, min(1000.0, float(v))) for v in b)
        if x2 - x1 < 30 or y2 - y1 < 20:
            return None
        out[k] = (x1, y1, x2, y2)
    return out


def locate(gateway, model, img, labels, picture, cache_path=None):
    """Рамки под подписи от модели со зрением, в пикселях, или None."""
    if cache_path and os.path.exists(cache_path):
        try:
            return {int(k): tuple(v) for k, v in json.load(open(cache_path, encoding="utf-8")).items()}
        except (OSError, ValueError):
            pass
    items = "\n".join(f"{i}. «{lab}»" for i, lab in enumerate(labels, 1))
    text = LOCATE_PROMPT.format(n=len(labels), items=f"The picture: {picture}\nLabels:\n{items}")
    answer, _u, _p = gateway.chat(model, [{"type": "text", "text": text}, _image_part(img)], 300, 1500, reasoning=False)
    boxes = parse_boxes(answer, len(labels))
    if boxes is None:
        return None
    w, h = img.size
    px = {k: (round(b[0] * w / 1000), round(b[1] * h / 1000), round(b[2] * w / 1000), round(b[3] * h / 1000))
          for k, b in boxes.items()}
    if cache_path:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(px, f)
    return px


def _inside_safe(box, w, h):
    """Рамка, ужатая в безопасную зону кадра: EDGE_SAFE от тех краёв, которые
    срезает наезд. Рисунок у́же 16:9 стоит на всю высоту, по бокам — поля,
    и наезд съедает поля, а не рисунок: боковой отступ там не нужен и только
    отнимает место у подписи (живой случай 01.10 — место под «ДОФАМИН» у
    правого края кадра 3:2). Шире 16:9 — наоборот."""
    narrow = w / h < CANVAS_ASPECT * (1 - 2 * EDGE_SAFE)
    wide = w / h > CANVAS_ASPECT / (1 - 2 * EDGE_SAFE)
    mx = round((EDGE_VISUAL if narrow else EDGE_SAFE) * w)
    my = round((EDGE_VISUAL if wide else EDGE_SAFE) * h)
    x1, y1, x2, y2 = box
    return (max(x1, mx), max(y1, my), min(x2, w - mx), min(y2, h - my))


MIN_FRAME_SIZE_SHARE = 0.6   # мельче — кадр в перерисовку, а не мелкий шрифт на экране
GROW_STEP = 0.015      # шаг расширения тесной рамки, доля кадра
GROW_MAX = (0.40, 0.25)   # рамка не шире 40% и не выше 25% кадра — подпись остаётся у своей стрелки


def _fitted_size(box, text, size_cap, h, pad=0.06):
    x1, y1, x2, y2 = box
    bw, bh = (x2 - x1) * (1 - 2 * pad), (y2 - y1) * (1 - 2 * pad)
    try:
        return fit(text, int(bw), int(bh), font_for(text), max_size=max(18, round(size_cap * h)))[0]
    except ValueError:
        return 0


def _grow(img, box, text, size_cap, others):
    """Тесная рамка расширяется в пустой фон, пока подпись не встанет кеглем
    шкалы ролика. Модель со зрением иногда отводит под слово щель (живой
    случай 01.10: «ДОФАМИН» вдвое мельче соседей на том же кадре). Каждая
    добавленная полоска должна быть пустой (MAX_INK_SHARE), в безопасной зоне
    и не задевать рамки других подписей; не вышло — рамка остаётся как есть."""
    w, h = img.size
    target = max(18, round(size_cap * h))
    if _fitted_size(box, text, size_cap, h) >= target:
        return box
    sx, sy = max(1, round(GROW_STEP * w)), max(1, round(GROW_STEP * h))
    lim = _inside_safe((0, 0, w, h), w, h)
    x1, y1, x2, y2 = box
    grew = True
    while grew and _fitted_size((x1, y1, x2, y2), text, size_cap, h) < target:
        grew = False
        for side in ("left", "right", "up", "down"):
            if side == "left":
                cand, strip = (max(lim[0], x1 - sx), y1, x2, y2), (max(lim[0], x1 - sx), y1, x1, y2)
            elif side == "right":
                cand, strip = (x1, y1, min(lim[2], x2 + sx), y2), (x2, y1, min(lim[2], x2 + sx), y2)
            elif side == "up":
                cand, strip = (x1, max(lim[1], y1 - sy), x2, y2), (x1, max(lim[1], y1 - sy), x2, y1)
            else:
                cand, strip = (x1, y1, x2, min(lim[3], y2 + sy)), (x1, y2, x2, min(lim[3], y2 + sy))
            if strip[2] <= strip[0] or strip[3] <= strip[1]:
                continue
            if cand[2] - cand[0] > GROW_MAX[0] * w or cand[3] - cand[1] > GROW_MAX[1] * h:
                continue
            if ink_share(img, strip) > MAX_INK_SHARE or any(_overlap(cand, o) for o in others):
                continue
            x1, y1, x2, y2 = cand
            grew = True
    return (x1, y1, x2, y2)


def _overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def background_color(img):
    """Цвет фона рисунка — медиана пикселей по краю кадра."""
    import numpy as np
    a = np.asarray(img.convert("RGB"), dtype=np.int16)
    edge = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    return tuple(int(v) for v in np.median(edge, axis=0))


def _band(img, labels):
    """Запасной путь: полоса цвета фона внизу кадра, подписи в одну строку
    через точку. Для кадра, на котором модель не оставила пустого места."""
    from PIL import ImageDraw
    w, h = img.size
    band = tuple(round(v * (w if i % 2 == 0 else h)) for i, v in enumerate(CAPTION_BAND))
    full = (0, band[1] - round(0.02 * h), w, h)
    ImageDraw.Draw(img).rectangle(full, fill=background_color(img))
    rec = draw(img, band, " · ".join(labels), size_cap=SIZE_CAPTION if len(labels) == 1 else SIZE_DIAGRAM)
    rec["fallback"] = "band"
    return [rec]


def compose(img_path, out_path, frame, gateway=None, model=None, cache_dir=None, fallback=False):
    """Положить подписи кадра. Возвращает (True, записи) или (False, причина).
    Кадр без подписей копируется как есть. fallback=True — места нет: подписи
    на полосе цвета фона внизу (а не выброс годного рисунка)."""
    from PIL import Image
    img = Image.open(img_path).convert("RGB")
    labels = frame.get("labels") or []
    w, h = img.size
    placed = []
    if labels and fallback:
        try:
            placed = _band(img, labels)
        except ValueError as e:
            return False, str(e)
    elif labels:
        boxes = None
        if frame.get("kind") == "caption" and len(labels) == 1:
            band = tuple(round(v * (w if i % 2 == 0 else h)) for i, v in enumerate(CAPTION_BAND))
            if ink_share(img, band) <= MAX_INK_SHARE:
                boxes = {1: band}
        if boxes is None:
            if gateway is None:
                return False, "нет пустого места под подпись и нет модели, чтобы его найти"
            cp = None
            if cache_dir:
                import hashlib
                digest = hashlib.sha256(open(img_path, "rb").read()).hexdigest()
                key = hashlib.sha256(f"{LOCATE_VERSION}|{model}|{labels}|{digest}".encode()).hexdigest()[:20]
                cp = os.path.join(cache_dir, f"locate_{key}.json")
            try:
                boxes = locate(gateway, model, img, labels, frame.get("picture", ""), cp)
            except Exception as e:  # noqa: BLE001
                return False, f"модель не нашла места: {type(e).__name__}: {e}"[:200]
            if boxes is None:
                return False, "модель не вернула рамки под подписи"
        cap = SIZE_CAPTION if frame.get("kind") == "caption" else SIZE_DIAGRAM
        safe = {k: _inside_safe(boxes[k], w, h) for k in boxes}
        grown = {}
        for k in range(1, len(labels) + 1):
            others = [safe[j] for j in safe if j != k] + list(grown.values())
            grown[k] = _grow(img, safe[k], labels[k - 1], cap, others)
        # Одна величина на кадр — по самой тесной рамке: разный кегль у
        # соседних подписей одной схемы читается как брак вёрстки.
        target = max(18, round(cap * h))
        frame_size = min(_fitted_size(grown[k], labels[k - 1], cap, h) for k in grown)
        if frame_size < MIN_FRAME_SIZE_SHARE * target:
            return False, (f"под подписи мало места: кегль {frame_size} при шкале {target} "
                           f"(меньше {MIN_FRAME_SIZE_SHARE:.0%})")
        cap = frame_size / h
        for k in range(1, len(labels) + 1):
            b = grown[k]
            share = ink_share(img, b)
            if share > MAX_INK_SHARE:
                return False, f"место под подпись {k} не пустое (рисунка {share:.0%})"
            if any(_overlap(b, p["box"]) for p in placed):
                return False, f"подпись {k} налезает на другую"
            try:
                placed.append(draw(img, b, labels[k - 1], size_cap=cap))
            except ValueError as e:
                return False, str(e)
    tmp = out_path + ".tmp.png"
    img.save(tmp, "PNG")
    os.replace(tmp, out_path)
    return True, placed
