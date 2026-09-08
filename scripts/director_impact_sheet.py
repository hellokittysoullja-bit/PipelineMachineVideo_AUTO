# -*- coding: utf-8 -*-
"""Контактный лист «было / стало» по результату measure_director_impact.py.

Зачем он вообще нужен, если есть числа. Потому что числа тут врут в одну
конкретную сторону, и это уже доказано на этом канале: кадр с современной
кухней из опубликованного эпизода ПРОШЁЛ все гейты (relevance 0.226 при
пороге 0.19) и всё равно был браком. Гейт спрашивает «похож ли кадр на
ЗАПРОС», а зритель видит «подходит ли кадр к ФРАЗЕ». Поэтому решает
проверка глазами (Шаг 7.5 в CLAUDE.md), а автоматические оси — только
вспомогательные.

Лист строится парами: слева победитель СЕГОДНЯШНЕГО механизма (baseline),
справа — победитель того пула, который реально будет в проде (union).
Подпись содержит русскую фразу слота, оба запроса и обе оценки, чтобы
глазами сравнивалось то же самое, что считалось числами.

Запуск: .venv/bin/python scripts/director_impact_sheet.py docs/quality/director_impact.json
"""
import argparse
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CELL_W, CELL_H = 460, 280
PAD = 12
TEXT_H = 96
COLS = 2   # baseline | union — сравнение всегда парное


def _font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _wrap(draw, text, font, max_w):
    words, lines, cur = (text or "").split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=font) <= max_w:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _thumb(path):
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        im = Image.new("RGB", (CELL_W, CELL_H), (40, 40, 40))
        d = ImageDraw.Draw(im)
        d.text((10, CELL_H // 2), "нет файла", fill=(200, 80, 80), font=_font(16))
        return im
    im.thumbnail((CELL_W, CELL_H))
    canvas = Image.new("RGB", (CELL_W, CELL_H), (18, 18, 18))
    canvas.paste(im, ((CELL_W - im.width) // 2, (CELL_H - im.height) // 2))
    return canvas


def build(impact_path, out_path, per_page=6):
    with open(impact_path, encoding="utf-8") as f:
        data = json.load(f)
    cases = data["cases"]

    f_small, f_mid = _font(13), _font(15)
    row_h = CELL_H + TEXT_H + PAD
    pages, page = [], []
    for c in cases:
        page.append(c)
        if len(page) == per_page:
            pages.append(page)
            page = []
    if page:
        pages.append(page)

    written = []
    for pi, page in enumerate(pages, 1):
        W = COLS * CELL_W + (COLS + 1) * PAD
        H = len(page) * row_h + PAD + 30
        sheet = Image.new("RGB", (W, H), (12, 12, 12))
        d = ImageDraw.Draw(sheet)
        d.text((PAD, 8),
               f"«было» (сегодняшний механизм)   |   «стало» (пул с режиссёром)   "
               f"— {data.get('model')}, промпт v{data.get('prompt_version')}",
               fill=(200, 200, 200), font=f_mid)

        for ri, c in enumerate(page):
            y = 30 + ri * row_h
            for ci, key in enumerate(("baseline", "union")):
                x = PAD + ci * (CELL_W + PAD)
                p = c[key].get("winner_path")
                sheet.paste(_thumb(p) if p else Image.new("RGB", (CELL_W, CELL_H), (30, 30, 30)),
                            (x, y))
                sr = c[key].get("winner_sentence_rel")
                gate = "гейт+" if c[key].get("winner_gate_passed") else "гейт-"
                label = f"{'БЫЛО' if key == 'baseline' else 'СТАЛО'}  {gate}  смысл={sr}"
                d.text((x, y + CELL_H + 2), label,
                       fill=(150, 220, 150) if key == "union" else (200, 200, 120),
                       font=f_mid)
                q = c["baseline_query"] if key == "baseline" else ", ".join(c["director_queries"])
                for li, line in enumerate(_wrap(d, q, f_small, CELL_W)[:2]):
                    d.text((x, y + CELL_H + 22 + li * 15), line,
                           fill=(140, 140, 160), font=f_small)
            # Фраза слота — под парой, общая для обоих кадров.
            head = f"{c['id']} ({c['difficulty_class']}, было: {c['baseline_verdict']})"
            d.text((PAD, y + CELL_H + 54), head, fill=(170, 170, 170), font=f_small)
            for li, line in enumerate(_wrap(d, c.get("text", ""), f_small, W - 2 * PAD)[:2]):
                d.text((PAD, y + CELL_H + 70 + li * 15), line,
                       fill=(225, 225, 225), font=f_small)

        p = out_path if len(pages) == 1 else out_path.replace(".jpg", f"_{pi:02d}.jpg")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        sheet.save(p, quality=88)
        written.append(p)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("impact_json")
    ap.add_argument("--out", default=os.path.join(REPO, "docs", "quality",
                                                   "director_impact_sheet.jpg"))
    ap.add_argument("--per-page", type=int, default=6)
    args = ap.parse_args()
    for p in build(args.impact_json, args.out, args.per_page):
        print(f"Готово: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
