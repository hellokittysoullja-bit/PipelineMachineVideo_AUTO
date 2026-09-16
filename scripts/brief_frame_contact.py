# -*- coding: utf-8 -*-
"""Контактный лист к A/B готовых кадров: пары «Claude / локальная модель».

ЗАЧЕМ ОТДЕЛЬНО ОТ ЧИСЛА. `brief_frame_ab.py` честно считает ровно одно —
долю слотов, где кадр СМЕНИЛСЯ. Это объективно и проверяется хэшем файла.
Вопрос «а в какую сторону сменился» числом не берётся: в этом репозитории
уже трижды измерено и трижды отклонено, что автоматический скор
(сырой косинус CLIP, маржа полки, «документальность») класс кадра не
разделяет. Значит вердикт выносит глаз — и ему нужен лист, где пара стоит
рядом и подписана фразой, ради которой кадр искали.

Слева ВСЕГДА рука Claude, справа локальная модель, порядок не
рандомизируется: лист смотрит автор правки и владелец, которые и так
знают, что где — слепота тут не создаётся, а путаница создавалась бы.
"""
import argparse
import json
import os
import sys
import textwrap

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT = os.path.join(REPO, "assets", "fonts", "Montserrat-Medium.ttf")
FONT_BOLD = os.path.join(REPO, "assets", "fonts", "Montserrat-Bold.ttf")

CELL_W, CELL_H = 460, 300
PAD, TEXT_H = 18, 150
ROW_H = CELL_H + TEXT_H


def _font(path, size):
    from PIL import ImageFont
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def _fit(img, w, h):
    """Вписать кадр целиком, не обрезая: на контактном листе важно, что
    на картинке есть, а не как она сядет в 16:9 — кроп решается позже и
    другим кодом."""
    from PIL import Image
    im = img.copy()
    im.thumbnail((w, h), Image.LANCZOS)
    canvas = Image.new("RGB", (w, h), (24, 24, 26))
    canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return canvas


def build(data, out_path, per_page=4):
    from PIL import Image, ImageDraw
    slots = [s for s in data["slots"]
             if s.get("claude_file") and s.get("local_file")]
    f_small = _font(FONT, 15)
    f_head = _font(FONT_BOLD, 17)
    f_phrase = _font(FONT, 16)
    pages = []
    for start in range(0, len(slots), per_page):
        chunk = slots[start:start + per_page]
        W = CELL_W * 2 + PAD * 3
        H = ROW_H * len(chunk) + PAD * 2 + 40
        sheet = Image.new("RGB", (W, H), (16, 16, 18))
        dr = ImageDraw.Draw(sheet)
        # Подписи рук берутся ИЗ ОТЧЁТА, а не зашиты: тот же файл собирается
        # и для сравнения двух мозгов, и «слева Claude» там было бы неправдой
        # о том, что на плитке.
        labels = (data.get("summary") or {}).get(
            "arm_labels") or ["бриф Claude", "бриф локальной модели"]
        dr.text((PAD, PAD), f"СЛЕВА — {labels[0]}       СПРАВА — {labels[1]}",
                font=f_head, fill=(255, 220, 120))
        y = PAD + 40
        for s in chunk:
            dr.text((PAD, y), f"[{s['index']}] {s['text'][:110]}",
                    font=f_phrase, fill=(235, 235, 240))
            y += 26
            # В режиме двух мозгов правая рука использует ТОТ ЖЕ бриф, что
            # левая, и отличается только добавленным в пул запросом. Писать
            # под ней чужой бриф было бы неправдой о том, чем нашли кадр.
            right_key = ("second_brain_query"
                         if (data.get("summary") or {}).get("mode") == "one_vs_two_brains"
                         else "brief_local")
            for col, (arm, brief_key) in enumerate(
                    (("claude_file", "brief_claude"), ("local_file", right_key))):
                x = PAD + col * (CELL_W + PAD)
                try:
                    with Image.open(s[arm]) as im:
                        sheet.paste(_fit(im.convert("RGB"), CELL_W, CELL_H), (x, y))
                except Exception as e:
                    dr.rectangle([x, y, x + CELL_W, y + CELL_H], fill=(60, 20, 20))
                    dr.text((x + 10, y + 10), f"нет кадра: {e}", font=f_small,
                            fill=(255, 160, 160))
                ty = y + CELL_H + 6
                for line in textwrap.wrap(s.get(brief_key, ""), 58)[:4]:
                    dr.text((x, ty), line, font=f_small, fill=(170, 200, 255))
                    ty += 19
            y += ROW_H - 26
        p = out_path.replace(".jpg", f"_{len(pages) + 1:02d}.jpg")
        sheet.save(p, quality=90)
        pages.append(p)
        print(f"страница: {p}")
    return pages


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("ab_json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-page", type=int, default=4)
    a = ap.parse_args(argv[1:])
    with open(a.ab_json, encoding="utf-8") as f:
        data = json.load(f)
    pages = build(data, a.out, a.per_page)
    print(f"\nвсего страниц: {len(pages)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
