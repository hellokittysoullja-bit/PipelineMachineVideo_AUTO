#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Контактный лист «до/после» по слотам: строка — слот, колонка — прогон.

Мерило каждой правки отбора — не отчёт, а кадры на экране рядом: тот же
слот в разных прогонах (эпизод до правки, прогоны харнесса
selection_freeze.py). Каждая колонка — папка эпизода с
media_plan/shotlist.json; подпись под плиткой — вид медиа и источник.

Плитка показывает то, что увидит зритель: если у кадра есть рамка
смысловой детали (метаданные кадра, focus_box), — вырезку вокруг неё той же
геометрией, что у рендера (focus_frame), с пометкой «наезд».

  python scripts/runs_sheet.py out.jpg "до" videos/94_x "после" \
      temp_selection_freeze/94_x/runs/judge4/episode
"""
import json
import os
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFont, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import focus_frame  # noqa: E402  — та же вырезка, что у рендера

TW, TH = 420, 236
FONTS = ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf")


def _font(size):
    for fp in FONTS:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()


def thumb(path, kind, box=None):
    """(плитка, вырезана ли деталь). У видео — кадр на первой секунде; у
    фото с рамкой детали — вырезка вокруг неё. Нет файла — (None, False)."""
    if not path or not os.path.exists(path):
        return None, False
    tmp = None
    try:
        if kind == "video" or path.endswith(".mp4"):
            fd, tmp = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "1", "-i", path,
                            "-frames:v", "1", tmp], check=False)
            if not os.path.getsize(tmp):
                return None, False
            path = tmp
        with Image.open(path) as im:
            im, cropped = focus_frame.crop_image(ImageOps.exif_transpose(im).convert("RGB"), box)
            im.thumbnail((TW, TH))
        bg = Image.new("RGB", (TW, TH), (16, 16, 16))
        bg.paste(im, ((TW - im.width) // 2, (TH - im.height) // 2))
        return bg, cropped
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)


def load_column(ep):
    """{слот: (путь, вид, источник)} и {слот: фраза} из шотлиста эпизода."""
    with open(os.path.join(ep, "media_plan", "shotlist.json"), encoding="utf-8") as f:
        d = json.load(f)
    shots, texts = {}, {}
    for s in d["shots"]:
        texts[s["index"]] = s.get("text") or ""
        p = s.get("file")
        shots[s["index"]] = (os.path.join(ep, p) if p else None, s.get("kind"), s.get("provider"))
    return shots, texts


def _wrap(text, width=34):
    lines, line = [], ""
    for w in text.split():
        if line and len(line) + len(w) > width:
            lines.append(line.rstrip())
            line = ""
        line += w + " "
    return lines + [line.rstrip()]


def build(out, columns):
    """columns: [(заголовок, папка эпизода)]."""
    f, fs = _font(17), _font(14)
    cols, texts = [], {}
    for _title, ep in columns:
        shots, t = load_column(ep)
        cols.append(shots)
        texts.update(t)
    slots = sorted(texts)
    lw, hh, rh = 330, 40, TH + 30
    sheet = Image.new("RGB", (lw + len(columns) * (TW + 10), hh + len(slots) * rh), (28, 28, 28))
    dr = ImageDraw.Draw(sheet)
    for c, (title, _ep) in enumerate(columns):
        dr.text((lw + c * (TW + 10) + 6, 10), title, font=f, fill=(240, 220, 120))
    for r, i in enumerate(slots):
        y = hh + r * rh
        dr.text((8, y + 6), f"#{i + 1}", font=f, fill=(230, 230, 230))
        for k, ln in enumerate(_wrap(texts[i])[:8]):
            dr.text((8, y + 30 + k * 18), ln, font=fs, fill=(200, 200, 200))
        for c, shots in enumerate(cols):
            x = lw + c * (TW + 10)
            path, kind, prov = shots.get(i, (None, None, None))
            box = focus_frame.box_of(path) if path and kind != "video" else None
            t, cropped = thumb(path, kind, box)
            if t is None:
                dr.rectangle([x, y, x + TW, y + TH], fill=(70, 20, 20))
                dr.text((x + 10, y + 10), "нет кадра", font=f, fill=(240, 200, 200))
            else:
                sheet.paste(t, (x, y))
            label = f"{kind or '—'}/{prov or '—'}" + (" · наезд" if cropped else "")
            dr.text((x + 4, y + TH + 4), label, font=fs, fill=(170, 170, 170))
    sheet.save(out, quality=88)
    return sheet.size


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 3 or len(argv) % 2 == 0:
        print(__doc__)
        return 2
    size = build(argv[0], [(argv[k], argv[k + 1]) for k in range(1, len(argv), 2)])
    print(argv[0], size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
