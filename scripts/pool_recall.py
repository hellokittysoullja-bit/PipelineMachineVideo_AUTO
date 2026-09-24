#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Есть ли нужный кадр в пуле, и доходит ли он до проверки.

ЗАЧЕМ. Все прежние числа подбора мерили ПОБЕДИТЕЛЯ слота. Они не отвечают
на главный вопрос: плохой кадр на экране — потому что хорошего не было в
выдаче источников, или потому что он был, но отбор до него не дошёл
(фильтр выбросил, порядок пула поставил его 300-м при 20 местах). Это
разные болезни с разным лечением, и без замера их путают.

ВХОД — pools.jsonl прогона харнесса (selection_freeze.py): пул каждого
вызова отбора в том порядке, в каком его получает ранжирование, плюс
кандидаты, которых выбросил фильтр, с причиной (поле removed).

КОМАНДЫ
  sheet  RUN_DIR OUT_DIR [--k 12]
      Для каждого слота и вида — лист кандидатов на разметку: объединение
      первых K трёх порядков (как сейчас / по описанию кадра в пуле после
      фильтров / по описанию кадра в пуле ДО фильтров). Под плиткой — номер,
      источник, отметка «выброшен фильтром» и в каких порядках кандидат
      стоит в первых K. Номер плитки -> id кандидата лежит в OUT_DIR/index.json.
  recall RUN_DIR LABELS [--handoff 20]
      LABELS — json {"<слот>:<вид>": {"<номер плитки>": 2|1|0}}: 2 — точный
      кадр, 1 — годная замена, 0 — брак. Печатает по слоту лучшую метку в
      пуле до фильтров, после фильтров и в первых N каждого порядка.

ЧЕСТНЫЕ ПРЕДЕЛЫ. «Лучшее в пуле» — лучшее среди РАЗМЕЧЕННЫХ: размечается
объединение верхушек порядков, а не весь пул в 700 кандидатов, поэтому
recall@pool — оценка снизу. Порядок «по описанию кадра» — тот же каскад,
что в проде (эмбеддинг превью моделью гейта против текста брифа), и тот же
дисковый кэш эмбеддингов, если он передан.
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.request

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import runs_sheet  # noqa: E402

UA = "Mozilla/5.0 (X11; Linux x86_64) pool_recall"
TW, TH = 300, 170


def load_pools(run_dir):
    """{(слот, вид): запись} — первая запись каждого слота и вида (повторные
    попытки того же вида строят тот же пул)."""
    out = {}
    with open(os.path.join(run_dir, "pools.jsonl"), encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            out.setdefault((rec["index"], rec["kind"]), rec)
    return out


def section_last(rec, phrase_queries):
    """Порядок «сначала запросы фразы, запросы секции последними»: внутри
    каждой группы — прежний круг по запросам. Кандидат приходит в пул с
    меткой запроса (via), поэтому порядок восстанавливается без повторного
    поиска."""
    own = set(phrase_queries or []) | {rec.get("query")}
    first = [r for r in rec["pool"] if r.get("via") in own]
    rest = [r for r in rec["pool"] if r.get("via") not in own]
    return first + rest


class Embedder:
    """Эмбеддинг превью и текста моделью гейта pipeline_smart — ровно тот,
    что у каскада. Кэш: ключ как у каскада (модель + адрес превью)."""

    def __init__(self, cache_dirs, write_dir):
        import pipeline_smart as ps
        self.ps = ps
        self.cache_dirs = [d for d in cache_dirs if d and os.path.isdir(d)]
        self.write_dir = write_dir
        os.makedirs(write_dir, exist_ok=True)

    def _key(self, url):
        return self.ps._cascade_key(url)

    def image(self, url, headers):
        import numpy as np
        if not url:
            return None
        key = self._key(url)
        for d in self.cache_dirs + [self.write_dir]:
            fp = os.path.join(d, key + ".npy")
            if os.path.exists(fp):
                try:
                    return np.load(fp)
                except Exception:
                    pass
        path = fetch(url, headers)
        if not path:
            return None
        try:
            with Image.open(path) as im:
                vec = self.ps._gate_embed(images=[im.convert("RGB")])
        except Exception:
            vec = None
        finally:
            os.remove(path)
        if vec is None:
            return None
        np.save(os.path.join(self.write_dir, key + ".npy"), vec[0])
        return vec[0]

    def text(self, text):
        v = self.ps._gate_embed(text=text)
        return None if v is None else v[0]


def fetch(url, headers):
    """Превью во временный файл; None при сбое."""
    h = {"User-Agent": UA}
    h.update(headers or {})
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=20) as r, \
                open(path, "wb") as f:
            f.write(r.read())
        if os.path.getsize(path) > 0:
            return path
    except Exception:
        pass
    os.remove(path)
    return None


def rank_by_text(rows, text, emb):
    """Кандидаты по убыванию близости превью к тексту; без превью — в хвост
    в прежнем порядке."""
    t = emb.text(text) if text else None
    if t is None:
        return list(rows)
    scored, rest = [], []
    for k, r in enumerate(rows):
        v = emb.image(r.get("probe_url"), r.get("headers"))
        if v is None:
            rest.append(r)
        else:
            scored.append((-float(v @ t), k, r))
    return [r for _s, _k, r in sorted(scored, key=lambda x: (x[0], x[1]))] + rest


VIDEO_MAX_TIME_STRETCH = 1.5   # как в pipeline_smart; держит тест синхронности


def base_slot_durs(pools):
    """{слот: настоящая длительность}. В режиме «только пул» слот остаётся
    без кадра, и его время уходит следующему (поглощение): slot_dur слота k
    = его собственная длительность + slot_dur слота k-1. Восстанавливается
    разностью соседних. Первый слот и слот после слота без записи — как есть."""
    sd = {}
    for (slot, _kind), rec in pools.items():
        if rec.get("slot_dur"):
            sd.setdefault(slot, float(rec["slot_dur"]))
    out = {}
    for slot in sorted(sd):
        prev = sd.get(slot - 1)
        base = sd[slot] - prev if prev is not None and sd[slot] > prev else sd[slot]
        out[slot] = round(base, 3)
    return out


def too_short(row, slot_dur):
    """Та же формула, что _video_candidate_too_short: неизвестная длина —
    годен (fail-open)."""
    try:
        dur = float(row.get("duration") or 0)
    except (TypeError, ValueError):
        return False
    return bool(slot_dur) and dur > 0 and dur * VIDEO_MAX_TIME_STRETCH < slot_dur


def full_rows(rec):
    """Все кандидаты в порядке ДО фильтра; выброшенные помечены _removed и
    причиной. Старая запись без prefilter_order — пул после фильтра, за ним
    выброшенные."""
    by_id = {}
    for r in rec["pool"]:
        by_id[str(r.get("id"))] = dict(r)
    for r in rec.get("removed", []):
        by_id[str(r.get("id"))] = dict(r, _removed=True)
    order = rec.get("prefilter_order")
    if not order:
        return list(by_id.values())
    return [by_id[str(i)] for i in order if str(i) in by_id]


def real_pool(rec, base_dur, blocklist=True):
    """Пул, который получил бы отбор в настоящем прогоне: запрещённые id и
    слишком короткие ролики (по настоящей длительности слота) выброшены;
    жанровый список запретов — если blocklist."""
    out = []
    for r in full_rows(rec):
        reason = r.get("reason") or ""
        if reason == "blocked_id":
            continue
        if blocklist and reason.startswith("blocklist:"):
            continue
        if rec["kind"] == "video" and too_short(r, base_dur):
            continue
        out.append(r)
    return out


def orders_for(rec, emb, phrase_queries, base_dur=None):
    """Именованные порядки кандидатов слота."""
    brief = rec.get("shot_brief") or rec.get("query")
    now = real_pool(rec, base_dur)
    noblock = real_pool(rec, base_dur, blocklist=False)
    as_rec = dict(rec, pool=now)
    return {
        "now": now,
        "section_last": section_last(as_rec, phrase_queries),
        "cascade": rank_by_text(now, brief, emb),
        "cascade_noblock": rank_by_text(noblock, brief, emb),
    }


def tile(row, label):
    path = fetch(row.get("probe_url"), row.get("headers"))
    bg = Image.new("RGB", (TW, TH + 34), (16, 16, 16))
    if path:
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((TW, TH))
                bg.paste(im, ((TW - im.width) // 2, (TH - im.height) // 2))
        except Exception:
            pass
        os.remove(path)
    d = ImageDraw.Draw(bg)
    d.text((4, TH + 2), label, fill=(255, 220, 0) if row.get("_removed") else (230, 230, 230),
           font=runs_sheet._font(13))
    d.text((4, TH + 18), (row.get("text") or "")[:44], fill=(160, 160, 160), font=runs_sheet._font(11))
    return bg


def cmd_sheet(a):
    pools = load_pools(a.run_dir)
    plan = load_phrase_queries(a.phrase_queries)
    emb = Embedder(a.emb_cache or [], os.path.join(a.out_dir, "emb"))
    os.makedirs(a.out_dir, exist_ok=True)
    base = base_slot_durs(pools)
    print("  настоящие длительности слотов:", base)
    index = {}
    for (slot, kind), rec in sorted(pools.items()):
        orders = orders_for(rec, emb, plan.get(rec.get("block_text") or "", []), base.get(slot))
        picked, where = [], {}
        for name, rows in orders.items():
            for r in rows[:a.k]:
                key = str(r.get("id"))
                where.setdefault(key, []).append(name)
                if key not in {str(p.get("id")) for p in picked}:
                    picked.append(r)
        cols = 6
        rows_n = (len(picked) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * TW, 40 + rows_n * (TH + 34)), (0, 0, 0))
        d = ImageDraw.Draw(sheet)
        d.text((6, 4), f"слот {slot} {kind}: {rec.get('block_text')}", fill=(255, 255, 255),
               font=runs_sheet._font(16))
        d.text((6, 22), f"бриф: {rec.get('shot_brief')}"[:150], fill=(180, 180, 180), font=runs_sheet._font(12))
        key = f"{slot}:{kind}"
        index[key] = {}
        for n, r in enumerate(picked, 1):
            lab = f"#{n} {r.get('channel')} [{','.join(sorted(where[str(r.get('id'))]))}]"
            if r.get("_removed"):
                lab += " ВЫБРОШЕН: " + (r.get("reason") or "")
            sheet.paste(tile(r, lab[:52]), ((n - 1) % cols * TW, 40 + (n - 1) // cols * (TH + 34)))
            index[key][str(n)] = {"id": r.get("id"), "removed": bool(r.get("_removed")),
                                  "reason": r.get("reason"), "text": r.get("text")}
        out = os.path.join(a.out_dir, f"slot{slot:02d}_{kind}.jpg")
        sheet.save(out, quality=85)
        print(f"  {out}: {len(picked)} кандидатов")
    with open(os.path.join(a.out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)
    return 0


def load_phrase_queries(path):
    """{текст фразы: [запросы]} из media_plan/stock_queries.json."""
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return {u.get("text", ""): u.get("queries", []) for u in d.get("units", {}).values()}


def recall_table(orders, labels, handoff):
    """{порядок: лучшая метка в первых handoff}; плюс 'pool' и 'prefilter'."""
    def best(rows):
        vals = [labels[str(r.get("id"))] for r in rows if str(r.get("id")) in labels]
        return max(vals) if vals else None
    out = {name: best(rows[:handoff]) for name, rows in orders.items()}
    out["pool"] = best(orders["now"])
    out["pool_noblock"] = best(orders["cascade_noblock"])
    return out


def cmd_recall(a):
    pools = load_pools(a.run_dir)
    plan = load_phrase_queries(a.phrase_queries)
    with open(a.labels, encoding="utf-8") as f:
        raw = json.load(f)
    with open(a.index, encoding="utf-8") as f:
        index = json.load(f)
    emb = Embedder(a.emb_cache or [], os.path.join(os.path.dirname(a.index), "emb"))
    base = base_slot_durs(pools)
    table = {}
    for key, marks in raw.items():
        slot, kind = key.split(":")
        rec = pools.get((int(slot), kind))
        if not rec:
            continue
        labels = {str(index[key][n]["id"]): v for n, v in marks.items() if n in index.get(key, {})}
        orders = orders_for(rec, emb, plan.get(rec.get("block_text") or "", []), base.get(int(slot)))
        table[key] = recall_table(orders, labels, a.handoff)
        print(key, json.dumps(table[key], ensure_ascii=False))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(table, f, ensure_ascii=False, indent=1)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheet")
    s.add_argument("run_dir")
    s.add_argument("out_dir")
    s.add_argument("--k", type=int, default=12)
    r = sub.add_parser("recall")
    r.add_argument("run_dir")
    r.add_argument("labels")
    r.add_argument("--index", required=True)
    r.add_argument("--handoff", type=int, default=20)
    r.add_argument("--out")
    for sp in (s, r):
        sp.add_argument("--phrase-queries", help="media_plan/stock_queries.json эпизода")
        sp.add_argument("--emb-cache", action="append",
                        help="папка кэша эмбеддингов каскада (можно несколько)")
    s.set_defaults(fn=cmd_sheet)
    r.set_defaults(fn=cmd_recall)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
