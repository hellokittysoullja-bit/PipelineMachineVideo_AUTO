#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Слепые контактные листы пулов для разметки качества подбора.

ЗАЧЕМ. Любое правило отбора (судить кадр по брифу, выбирать фото или видео
по действию, вес ритма крупностей) калибруется порогом, а порог без
разметки — догадка. Разметка обязана быть:
  * по ТОМУ пулу, из которого выбирает код — поэтому вход здесь не
    пересобранный рядом пул, а pools.jsonl прогона харнесса
    (selection_freeze.install_pool_capture): пул на входе ранжирования;
  * СЛЕПОЙ — на листе нет ни одного числа модели, ни источника, ни того,
    какой кандидат победил. Только фраза, бриф и кадры. Иначе разметчик
    (человек или модель, смотрящая глазами) подтягивается к ответу системы.

Команды:
  build  <pools.jsonl> --out DIR [--depth N] [--tag ЭПИЗОД]
         скачивает превью первых N кандидатов каждого пула, пишет листы
         DIR/sheets/<tag>_<слот>.jpg и шаблон разметки DIR/labels/<tag>_<слот>.json
         (метка null у каждого кандидата). Повторный запуск докачивает
         недостающее и не трогает уже проставленные метки.

  mark   DIR ИМЯ --s3 0,3 --s2 5 --s1 7 [--s0 1,2 | --rest 0]
         проставляет метки по номерам на листе, не открывая файл разметки
         (в нём текст источника — подсказка, которой на листе нет).

Метки — шкала судьи кадров:
  3 — показан требуемый предмет и действие, в нужной эпохе и культуре;
  2 — предмет тот, но действие или композиция другие, или близкая замена;
  1 — связано, но не тот предмет, эпоха или культура;
  0 — не по теме, современное там, где эпоха историческая, или негодно.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import textwrap
import urllib.request

# Та же шкала, что у судьи кадров (scripts/shot_judge.py): ответы судьи и
# глаз сравниваются напрямую, без перевода между шкалами.
LABELS = ("3", "2", "1", "0")
UA = "Mozilla/5.0 (X11; Linux x86_64) pool-sheet/1"
THUMB = (240, 160)
COLS = 8
FONT_CANDIDATES = ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def read_pools(path):
    """Пулы из pools.jsonl. Слот может звать ядро несколько раз (фото после
    негодного видео и т.п.); одинаковый пул одного слота берётся один раз,
    разные пулы одного слота — все, с номером вызова."""
    out, seen = [], set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["index"], tuple(str(c["id"]) for c in rec["pool"]))
            if key in seen:
                continue
            seen.add(key)
            rec["call"] = sum(1 for r in out if r["index"] == rec["index"])
            out.append(rec)
    return out


def image_name(cid):
    return hashlib.sha256(str(cid).encode("utf-8")).hexdigest()[:16] + ".jpg"


def fetch(url, headers, dst, timeout=30):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return True
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
    except Exception:
        return False
    if not data:
        return False
    tmp = dst + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, dst)
    return True


def _font(size):
    from PIL import ImageFont
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def draw_sheet(rec, cands, img_dir, dst):
    from PIL import Image, ImageDraw
    rows = (len(cands) + COLS - 1) // COLS
    head_lines = []
    for label, text in (("ФРАЗА", rec.get("block_text")), ("БРИФ", rec.get("shot_brief"))):
        for i, line in enumerate(textwrap.wrap(text or "—", 150) or ["—"]):
            head_lines.append((label if i == 0 else "", line))
    head_h = 16 + 22 * len(head_lines)
    W, H = COLS * (THUMB[0] + 6) + 6, head_h + rows * (THUMB[1] + 26) + 6
    sheet = Image.new("RGB", (W, H), (24, 24, 24))
    d = ImageDraw.Draw(sheet)
    f_head, f_num = _font(16), _font(15)
    y = 8
    for label, line in head_lines:
        if label:
            d.text((8, y), label, fill=(255, 210, 90), font=f_head)
        d.text((80, y), line, fill=(235, 235, 235), font=f_head)
        y += 22
    for k, c in enumerate(cands):
        x = 6 + (k % COLS) * (THUMB[0] + 6)
        yy = head_h + (k // COLS) * (THUMB[1] + 26)
        p = os.path.join(img_dir, image_name(c["id"]))
        try:
            im = Image.open(p).convert("RGB")
            im.thumbnail(THUMB)
            sheet.paste(im, (x + (THUMB[0] - im.width) // 2, yy + (THUMB[1] - im.height) // 2))
        except Exception:
            d.rectangle([x, yy, x + THUMB[0], yy + THUMB[1]], outline=(120, 60, 60))
            d.text((x + 8, yy + 8), "нет превью", fill=(200, 120, 120), font=f_num)
        d.text((x + 2, yy + THUMB[1] + 3), f"{k}", fill=(255, 255, 255), font=f_num)
    sheet.save(dst, quality=88)


def cmd_build(args):
    out = os.path.abspath(args.out)
    img_dir, sheets, labels = (os.path.join(out, n) for n in ("img", "sheets", "labels"))
    for p in (img_dir, sheets, labels):
        os.makedirs(p, exist_ok=True)
    pools = read_pools(args.pools)
    jobs = {}
    for rec in pools:
        for c in rec["pool"][:args.depth]:
            if c.get("probe_url"):
                jobs[c["id"]] = (c["probe_url"], c.get("headers"))
    with concurrent.futures.ThreadPoolExecutor(8) as ex:
        got = dict(zip(jobs, ex.map(
            lambda kv: fetch(kv[1][0], kv[1][1], os.path.join(img_dir, image_name(kv[0]))),
            jobs.items())))
    missing = [k for k, ok in got.items() if not ok]
    for rec in pools:
        name = f"{args.tag}_{rec['index']:03d}" + (f"_{rec['call']}" if rec["call"] else "")
        cands = rec["pool"][:args.depth]
        draw_sheet(rec, cands, img_dir, os.path.join(sheets, name + ".jpg"))
        lp = os.path.join(labels, name + ".json")
        old = {}
        if os.path.exists(lp):
            old = {str(c["id"]): c.get("label") for c in json.load(open(lp, encoding="utf-8"))["candidates"]}
        doc = {"tag": args.tag, "index": rec["index"], "call": rec["call"],
               "query": rec["query"], "shot_brief": rec.get("shot_brief"),
               "block_text": rec.get("block_text"), "pool_size": len(rec["pool"]),
               "candidates": [{"n": k, "id": c["id"], "channel": c["channel"],
                               "text": c["text"], "label": old.get(str(c["id"]))}
                              for k, c in enumerate(cands)]}
        with open(lp, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
    print(f"листов: {len(pools)}; превью: {len(jobs) - len(missing)} из {len(jobs)} "
          f"(не скачалось {len(missing)})")
    return 0


def _nums(spec):
    return {int(x) for x in (spec or "").split(",") if x.strip()}


def cmd_mark(args):
    lp = os.path.join(os.path.abspath(args.dir), "labels", args.name + ".json")
    doc = json.load(open(lp, encoding="utf-8"))
    marks = {}
    for label in LABELS:
        for n in _nums(getattr(args, "s" + label)):
            if n in marks and marks[n] != label:
                raise SystemExit(f"кандидат {n} помечен дважды: {marks[n]} и {label}")
            marks[n] = label
    known = {c["n"] for c in doc["candidates"]}
    unknown = set(marks) - known
    if unknown:
        raise SystemExit(f"нет таких номеров на листе: {sorted(unknown)}")
    for c in doc["candidates"]:
        if c["n"] in marks:
            c["label"] = marks[c["n"]]
        elif args.rest:
            c["label"] = args.rest
    left = [c["n"] for c in doc["candidates"] if c["label"] is None]
    with open(lp, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print(f"{args.name}: размечено {len(doc['candidates']) - len(left)} из "
          f"{len(doc['candidates'])}" + (f", без метки: {left}" if left else ""))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("pools")
    b.add_argument("--out", required=True)
    b.add_argument("--depth", type=int, default=32)
    b.add_argument("--tag", required=True)
    b.set_defaults(fn=cmd_build)
    m = sub.add_parser("mark")
    m.add_argument("dir")
    m.add_argument("name")
    for label in LABELS:
        m.add_argument(f"--s{label}", default="")
    m.add_argument("--rest", choices=LABELS)
    m.set_defaults(fn=cmd_mark)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
