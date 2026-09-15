# -*- coding: utf-8 -*-
"""Контактный лист БРИФОВ: что полка отвечает на каждую фразу, ДО рендера.

ЗАЧЕМ. Все числа качества подбора в этом проекте до сих пор были либо
метрикой ГЕЙТОВ (`golden_set_eval.py` — она про «пропустили ли брак», а не
про «хорош ли кадр»), либо глазами автора правки на девяти слотах, что сам
этот репозиторий честно записал как «не приёмка». Настоящая разметка
требовала полного рендера: аудио, ключи стоков, часы сборки. То есть между
правкой ВХОДА и первой цифрой качества стоял целый ролик.

Этот лист убирает и эту стену. Он берёт бриф каждой фразы (`[shot:...]`),
спрашивает им полку теми же функциями, что будут отвечать на рендере,
и кладёт ответ картинкой рядом с фразой. Владельцу остаётся пройтись
глазами и сказать «годен / терпимо / брак» — то есть ровно та разметка, на
которой стоит золотой набор, но полученная БЕЗ рендера, без `audio.mp3`,
без единого ключа стока и без единого платного вызова.

ЧЕСТНО О ГРАНИЦАХ. Лист показывает ПОЛКУ, а не готовый кадр. Полка только
пополняет пул; победителя слота выбирают гейты и ранжирование, и среди
кандидатов будут ещё сток, архивы и музейный путь. Поэтому «годен» здесь
означает «полка ответила тем, что просил бриф», а не «этот кадр будет в
ролике». Это меньше, чем разметка готового эпизода, и больше, чем ноль,
который был до него.

Использование:
    python scripts/shelf_contact.py videos/02_ne-mechom
    python scripts/shelf_contact.py videos/02_ne-mechom --top 3 --per-page 12

Результат: <video_dir>/media_plan/shelf_contact_01.jpg, _02.jpg, ...
плюс <video_dir>/media_plan/shelf_contact.json — та же выдача текстом,
чтобы разметку можно было вести не только глазами по картинке.
Ничего не чинит, ничего не блокирует, рендер не трогает.
"""
import argparse
import json
import os
import sys
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from PIL import Image, ImageDraw  # noqa: E402

# Шрифт и перенос — из уже существующего контактного листа, а не второй
# копией: две реализации одного переноса рано или поздно разойдутся, и
# один из двух листов станет нечитаемым незаметно.
from shotlist_contact import load_font, wrap_text  # noqa: E402

THUMB_W, THUMB_H = 320, 320
CAPTION_H = 128
PAD = 10
CACHE_DIR = os.path.join(REPO, "temp_shelf_index", "contact_cache")
_UA = {"User-Agent": "Mozilla/5.0 (faceless-pipeline shelf contact)"}


def _cached_thumb(rec):
    """Превью предмета на диск. Снимки полки лежат в чужих хранилищах, и
    повторный лист не имеет права качать их заново — это чужой трафик."""
    url = rec.get("thumb") or rec.get("image")
    if not url:
        return None
    os.makedirs(CACHE_DIR, exist_ok=True)
    name = str(rec.get("id", "x")).replace(":", "_").replace("/", "_") + ".jpg"
    path = os.path.join(CACHE_DIR, name)
    if not (os.path.exists(path) and os.path.getsize(path) > 2000):
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            if len(data) < 2000:
                return None
            with open(path, "wb") as fh:
                fh.write(data)
        except Exception:
            return None
    return path


def _tile(path):
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return None
    img.thumbnail((THUMB_W, THUMB_H))
    canvas = Image.new("RGB", (THUMB_W, THUMB_H), (18, 18, 18))
    canvas.paste(img, ((THUMB_W - img.width) // 2, (THUMB_H - img.height) // 2))
    return canvas


def collect(video_dir, top=1, limit=None):
    import script_parser
    import shelf_index

    blocks = script_parser.parse_blocks(os.path.join(video_dir, "script.txt"))
    if limit:
        blocks = blocks[: int(limit)]
    st = shelf_index.stats()
    cells = []
    for i, b in enumerate(blocks):
        brief = b.get("shot_brief")
        if not brief:
            continue
        for rank, rec in enumerate(shelf_index.search(brief, limit=int(top)) or [None]):
            cells.append({"index": i, "rank": rank, "section": b.get("section"),
                          "text": (b.get("text") or "").strip(), "brief": brief,
                          "rec": rec})
    return cells, st


def render_page(cells, cols, out_path):
    rows = (len(cells) + cols - 1) // cols
    cell_w, cell_h = THUMB_W + PAD, THUMB_H + CAPTION_H + PAD
    page = Image.new("RGB", (cols * cell_w + PAD, rows * cell_h + PAD), (28, 28, 28))
    draw = ImageDraw.Draw(page)
    f_head, f_text, f_small = load_font(16), load_font(13), load_font(12)
    for k, c in enumerate(cells):
        x0 = PAD + (k % cols) * cell_w
        y0 = PAD + (k // cols) * cell_h
        rec = c.get("rec")
        tile = _tile(_cached_thumb(rec)) if rec else None
        if tile is None:
            tile = Image.new("RGB", (THUMB_W, THUMB_H), (70, 20, 20))
            msg = "ПОЛКА МОЛЧИТ" if not rec else "снимок не скачался"
            ImageDraw.Draw(tile).text((12, 12), msg, fill=(255, 200, 200), font=f_head)
        page.paste(tile, (x0, y0))
        y = y0 + THUMB_H + 4
        rank = f" ({c['rank'] + 1})" if c["rank"] else ""
        draw.text((x0 + 2, y), f"#{c['index']}{rank}  {c['section']}"[:46],
                  fill=(220, 220, 220), font=f_head)
        y += 19
        for line in wrap_text(c["text"], f_text, THUMB_W - 4, draw, max_lines=2):
            draw.text((x0 + 2, y), line, fill=(175, 175, 175), font=f_text)
            y += 15
        for line in wrap_text(c["brief"], f_small, THUMB_W - 4, draw, max_lines=2):
            draw.text((x0 + 2, y), line, fill=(120, 170, 220), font=f_small)
            y += 14
        if rec:
            tail = f"{rec.get('score', 0):+.3f}  {rec.get('name') or '—'}  " \
                   f"[{rec.get('b')}-{rec.get('e')}]"
            draw.text((x0 + 2, y), tail[:52], fill=(150, 200, 150), font=f_small)
    page.save(out_path, quality=85)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video_dir")
    ap.add_argument("--top", type=int, default=1,
                    help="сколько кандидатов полки показывать на бриф")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--per-page", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None,
                    help="разобрать только первые N юнитов сценария")
    a = ap.parse_args(argv)

    import shelf_index
    if not shelf_index.available():
        # Громко и сразу: лист без полки показывал бы пустые плитки и
        # выглядел бы как измерение, которым он не является.
        print("Полка НЕ собрана — печатать нечего. "
              "Собрать: python scripts/shelf_index.py build --corpus all")
        return 1

    cells, st = collect(a.video_dir, top=a.top, limit=a.limit)
    if not cells:
        print("Ни у одного юнита нет [shot:...] — размечать нечего.")
        return 1
    out_dir = os.path.join(a.video_dir, "media_plan")
    os.makedirs(out_dir, exist_ok=True)

    silent = sum(1 for c in cells if not c.get("rec"))
    print(f"Полка: {st['items']} предметов, модель {st['model']}")
    print(f"Брифов на листе: {len(cells)}   молчит: {silent}")

    pages = []
    for p in range(0, len(cells), a.per_page):
        out = os.path.join(out_dir, f"shelf_contact_{p // a.per_page + 1:02d}.jpg")
        pages.append(render_page(cells[p:p + a.per_page], a.cols, out))
        print("  ", out)

    report = {"shelf": st, "briefs": len(cells), "silent": silent,
              "pages": [os.path.basename(p) for p in pages],
              "cells": [{"index": c["index"], "rank": c["rank"],
                         "section": c["section"], "text": c["text"],
                         "brief": c["brief"],
                         "shelf": {k: (c["rec"] or {}).get(k)
                                   for k in ("id", "score", "name", "title",
                                             "dept", "b", "e", "source",
                                             "rights", "image_size", "page")}
                                   if c["rec"] else None}
                        for c in cells]}
    with open(os.path.join(out_dir, "shelf_contact.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
