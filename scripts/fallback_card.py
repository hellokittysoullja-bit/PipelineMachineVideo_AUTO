# -*- coding: utf-8 -*-
"""Процедурная карточка — уровень видеоряда, который НЕ МОЖЕТ БЫТЬ НЕВЕРНЫМ.

Зачем это существует. Подбор кадра устроен как гейт, а не как фильтр: если
весь просмотренный пул провалил проверки, побеждает «лучший из плохих», и
слот всё равно заполняется (философия «слот не должен остаться пустым»,
CLAUDE.md ЧАСТЬ 13). На опубликованном эпизоде это дало кадры, где система
УЖЕ ЗНАЛА, что кандидат плохой: относительно слота #13 гейт релевантности
прямо отклонял победителя, а для слота #5 VLM-арбитр явно ответил «ни один
из кандидатов не подходит» — и оба кадра ушли зрителю, потому что показать
было больше нечего.

Пока у системы нет варианта, который не может быть «не про то», ужесточение
любых гейтов меняет только ИМЯ показанного брака. Карточка и есть этот
вариант: она собирается из СОБСТВЕННЫХ слов диктора, звучащих в этот самый
момент. Она физически не может изображать чужую эпоху, другую культуру или
не тот предмет — на ней нет изображения вообще.

Это не затычка. Типографская карточка с ключевой цифрой или фразой —
нормальный приём исторической документалистики, и для канала, где вся суть в
числах («меч весил полтора килограмма, а не пятнадцать»), она работает на
смысл сильнее случайного стокового кадра.

Честные ограничения:
  * карточка НЕ заменяет хороший кадр — она заменяет ПЛОХОЙ. Использовать её
    там, где подбор справился, значит обеднить ролик;
  * подряд идущие карточки читаются как сбой вёрстки, а не как приём — за
    частоту отвечает вызывающий код (см. can_use_card_at в pipeline_smart);
  * текст берётся из фразы блока и НИКОГДА не сочиняется: единственная
    гарантия неошибочности в том, что зритель слышит ровно эти слова.
"""
import hashlib
import math
import os
import random
import re

from PIL import Image, ImageDraw, ImageFilter, ImageFont

# Палитра — под тёмный, десатурированный грейд канала (см. mood_grade в
# channel_profile.json: контраст 1.06, насыщенность 0.82-0.90). Светлая
# «пергаментная» карточка между тёмными клипами била бы по глазам вспышкой;
# карточка обязана читаться как часть того же ролика, а не как вставка.
CARD_BG_DARK = (14, 15, 17)
CARD_BG_LIGHT = (46, 44, 40)
CARD_INK = (232, 226, 214)
CARD_ACCENT = (176, 141, 87)     # приглушённая латунь: металл, а не золото

CARD_W, CARD_H = 1920, 1080
CARD_MARGIN_FRAC = 0.12          # поля: текст не должен упираться в край кадра
CARD_MAX_LINES = 3
CARD_MAX_WORDS = 7               # длиннее — это уже субтитр, а не карточка

# Единицы измерения канала: числа с ними — самая ценная подпись, какую можно
# поставить на карточку (вся суть ролика в том, сколько весил меч).
_UNIT_WORDS = (
    "килограмм", "килограмма", "килограммов", "кг",
    "грамм", "грамма", "граммов",
    "сантиметр", "сантиметра", "сантиметров", "см",
    "метр", "метра", "метров",
    "год", "года", "лет", "век", "века", "веков",
)
_NUM_WORDS = (
    "ноль", "один", "одна", "два", "две", "три", "четыре", "пять", "шесть",
    "семь", "восемь", "девять", "десять", "одиннадцать", "двенадцать",
    "тринадцать", "четырнадцать", "пятнадцать", "шестнадцать", "семнадцать",
    "восемнадцать", "девятнадцать", "двадцать", "тридцать", "сорок",
    "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят", "девяносто",
    "сто", "двести", "триста", "четыреста", "пятьсот", "тысяча", "тысячи",
    "полтора", "полторы",
)


def _words(text):
    return [w for w in re.split(r"\s+", (text or "").strip()) if w]


def _clean(word):
    return word.strip(" \t\n\r«»\"'()[]—–-").rstrip(".,:;!?")


def choose_card_text(text, max_words=CARD_MAX_WORDS):
    """Что вынести на карточку из фразы блока.

    Порядок приоритетов — не вкусовщина, а то, что для этого канала несёт
    смысл: сначала измеримое (число с единицей), потом остальное. Текст
    ВСЕГДА берётся из самой фразы, дословно: это и есть гарантия, что
    карточка не может быть «не про то» — зритель слышит ровно эти слова.

    Возвращает (строка, вид) — вид нужен вёрстке: число ставится крупно,
    фраза мельче и в несколько строк.
    """
    ws = _words(text)
    if not ws:
        return "", "empty"

    lowered = [_clean(w).lower() for w in ws]

    # 1. Число (цифрами или словом) вместе с его единицей измерения.
    for i, w in enumerate(lowered):
        is_num = bool(re.search(r"\d", w)) or w in _NUM_WORDS
        if not is_num:
            continue
        # Единица может стоять сразу после числа или через одно слово
        # ("пятнадцать целых килограммов").
        for j in range(i + 1, min(i + 3, len(lowered))):
            if any(lowered[j].startswith(u) for u in _UNIT_WORDS):
                phrase = " ".join(_clean(x) for x in ws[i:j + 1])
                return phrase, "number"
    # 2. Число без единицы — всё ещё сильнее любой абстрактной фразы.
    for i, w in enumerate(lowered):
        if re.search(r"\d", w):
            return _clean(ws[i]), "number"

    # 3. Короткая фраза целиком — ДОСЛОВНО, с внутренней пунктуацией.
    # «Не дрались. Несли.» без точки превращается в «Не дрались Несли» —
    # набор слов вместо фразы, ради ритма которой её и писали.
    if len(ws) <= max_words:
        return text.strip().strip("—–- "), "phrase"

    # 4. Иначе — САМАЯ КОРОТКАЯ законченная клауза, а не обрезанная первая.
    # Обрезка по счётчику слов даёт «Историки до сих пор спорят врёт эта» —
    # оборванную мысль, которая читается как баг вёрстки. Короткая клауза —
    # это законченное высказывание автора, и на карточке она работает.
    clauses = [c.strip(" \t—–-") for c in re.split(r"[.!?;:]+", text) if c.strip()]
    usable = [c for c in clauses if 2 <= len(_words(c)) <= max_words]
    if usable:
        return min(usable, key=lambda c: len(_words(c))), "phrase"
    # Ни одно предложение не коротко: пробуем границу запятой. «Историки до
    # сих пор спорят, врёт эта табличка или нет» — одно предложение на 11
    # слов, и без этого шага оно обрезалось до «...спорят, врёт эта».
    sub = [c.strip(" \t—–-") for c in re.split(r"[,.!?;:]+", text) if c.strip()]
    usable = [c for c in sub if 2 <= len(_words(c)) <= max_words]
    if usable:
        return max(usable, key=lambda c: len(_words(c))), "phrase"
    # Ни одной короткой клаузы: берём самую короткую вообще и подрезаем по
    # границе слова — последний рубеж, лучше чем пустая карточка.
    shortest = min(clauses, key=lambda c: len(_words(c))) if clauses else text
    return " ".join(_words(shortest)[:max_words]).strip(), "phrase"


def _texture(size, seed):
    """Процедурная фактура: крупное мягкое пятно + мелкое зерно.

    Собирается из шума и блюра, а не из файла-ассета: карточка не должна
    зависеть от наличия картинки на диске — уровень, который «не может
    провалиться», не имеет права провалиться из-за отсутствующего файла.
    """
    w, h = size
    rnd = random.Random(seed)
    # Крупное пятно — редкий шум, растянутый и размытый.
    small = Image.new("L", (max(2, w // 90), max(2, h // 90)))
    small.putdata([rnd.randint(70, 185) for _ in range(small.width * small.height)])
    blot = small.resize((w, h), Image.BICUBIC).filter(ImageFilter.GaussianBlur(w / 90))
    # Мелкое зерно — чтобы плоская заливка не выглядела цифровой.
    grain = Image.new("L", (w, h))
    grain.putdata([rnd.randint(112, 143) for _ in range(w * h)])
    grain = grain.filter(ImageFilter.GaussianBlur(0.6))
    return Image.blend(blot, grain, 0.35)


def _vignette(size, strength=0.85):
    w, h = size
    cx, cy = w / 2.0, h / 2.0
    maxd = math.hypot(cx, cy)
    mask = Image.new("L", (w, h))
    px = mask.load()
    # Считаем по столбцу и переиспользуем симметрию — полный двойной цикл по
    # 2 Мп пикселям в Python занимал бы секунды на каждый кадр.
    for y in range(h):
        dy2 = (y - cy) ** 2
        for x in range(0, w // 2 + 1):
            d = math.hypot(x - cx, 0) if False else math.sqrt((x - cx) ** 2 + dy2)
            v = int(255 * (1.0 - strength * (d / maxd) ** 2.2))
            v = max(0, min(255, v))
            px[x, y] = v
            px[w - 1 - x, y] = v
    return mask


def _fit_font(font_path, text, max_w, max_h, start=200, min_size=28):
    """Наибольший кегль, при котором строка влезает в отведённый прямоугольник."""
    size = start
    while size > min_size:
        try:
            font = ImageFont.truetype(font_path, size)
        except Exception:
            return None, size
        box = font.getbbox(text)
        if (box[2] - box[0]) <= max_w and (box[3] - box[1]) <= max_h:
            return font, size
        size -= 4
    try:
        return ImageFont.truetype(font_path, min_size), min_size
    except Exception:
        return None, min_size


def _wrap(text, font, max_w, draw, max_lines=CARD_MAX_LINES):
    words, lines, cur = _words(text), [], ""
    for w in words:
        probe = (cur + " " + w).strip()
        if draw.textlength(probe, font=font) <= max_w or not cur:
            cur = probe
        else:
            lines.append(cur)
            cur = w
            if len(lines) == max_lines:
                break
    if cur and len(lines) < max_lines:
        lines.append(cur)
    return lines


def build_fallback_card(text, out_path, font_path=None, accent_font_path=None,
                        size=(CARD_W, CARD_H), seed=None):
    """Собрать карточку и записать в out_path. Возвращает (путь, что_написано).

    seed=None -> выводится из текста: одна и та же фраза всегда даёт одну и ту
    же карточку (детерминизм важен для кэша клипов — иначе каждый прогон
    перерендеривал бы слот заново), но РАЗНЫЕ фразы получают разную фактуру и
    раскладку, и подряд идущие карточки не выглядят копиями друг друга.
    """
    w, h = size
    card_text, kind = choose_card_text(text)
    if not card_text:
        card_text, kind = "•", "phrase"
    if seed is None:
        seed = int(hashlib.md5(card_text.encode("utf-8")).hexdigest()[:8], 16)
    rnd = random.Random(seed)

    # Фон: вертикальный градиент между двумя тёмными тонами + фактура.
    base = Image.new("RGB", (w, h), CARD_BG_DARK)
    grad = Image.new("L", (1, h))
    grad.putdata([int(255 * (y / max(1, h - 1)) ** 1.3) for y in range(h)])
    grad = grad.resize((w, h))
    base = Image.composite(Image.new("RGB", (w, h), CARD_BG_LIGHT), base, grad)
    tex = _texture((w, h), seed).convert("RGB")
    base = Image.blend(base, tex, 0.18)
    base = Image.composite(base, Image.new("RGB", (w, h), CARD_BG_DARK),
                           _vignette((w, h)))

    draw = ImageDraw.Draw(base)
    margin = int(min(w, h) * CARD_MARGIN_FRAC)
    box_w, box_h = w - 2 * margin, h - 2 * margin

    fp = font_path
    if not fp or not os.path.exists(fp):
        return None, card_text   # без шрифта карточка не собирается — честный отказ

    if kind == "number":
        font, _ = _fit_font(fp, card_text, box_w, int(box_h * 0.55), start=int(h * 0.42))
        lines = [card_text]
    else:
        font, _ = _fit_font(fp, card_text, box_w, int(box_h * 0.30), start=int(h * 0.16))
        if font is None:
            return None, card_text
        lines = _wrap(card_text, font, box_w, draw)
    if font is None:
        return None, card_text

    # Вертикальная центровка по ФАКТИЧЕСКИМ чернилам, а не по метрике шрифта.
    # getbbox() у display-гарнитуры возвращает бокс со смещённым верхом
    # (у Benzin — заметно), и наивное y = (h - line_h*n)//2 ставит блок
    # выше середины: на первом рендере текст визуально «висел» в верхней
    # трети кадра при формально верной арифметике.
    boxes = [font.getbbox(line) for line in lines]
    line_h = int(max((b[3] - b[1]) for b in boxes) * 1.34)
    ink_top = boxes[0][1]
    ink_bottom = boxes[-1][3] + line_h * (len(lines) - 1)
    y = (h - (ink_bottom - ink_top)) // 2 - ink_top

    # Тонкая линейка над текстом — типографский якорь, который отличает
    # карточку от субтитра. Её ширина слегка гуляет от seed: две соседние
    # карточки не должны выглядеть одним шаблоном.
    rule_w = int(box_w * rnd.uniform(0.18, 0.34))
    rule_y = y + ink_top - int(line_h * 0.62)
    draw.rectangle([(w - rule_w) // 2, rule_y, (w + rule_w) // 2, rule_y + max(2, h // 540)],
                   fill=CARD_ACCENT)

    for i, line in enumerate(lines):
        tw = draw.textlength(line, font=font)
        x = (w - tw) / 2
        ly = y + i * line_h
        # Мягкая тень: на тёмной фактуре чистый текст без опоры «плывёт».
        draw.text((x + 2, ly + 3), line, font=font, fill=(0, 0, 0))
        draw.text((x, ly), line, font=font, fill=CARD_INK)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp = out_path + ".tmp.png"
    base.save(tmp, "PNG")
    os.replace(tmp, out_path)
    return out_path, card_text
