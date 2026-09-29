#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Замер каскада Qwen3-VL (эмбеддинг + реранкер верха) на размеченных кучах
эпизода 94 против базовой линии SigLIP2.

Гейт регрессий (pool_recall.py rankcheck) хранит векторы SigLIP2 и базовую
линию: 49 куч эп.94 (judge9/11/12), 856 меток, годных в первых 20 — 204.
Картинок в нём нет. Этот скрипт берёт сами превью из записей сети
харнесса (temp_selection_freeze/94_dagger_test: основная запись и
net_overlay прогонов), считает векторы Qwen3-VL-Embedding и ранжирует
ПРОД-функцией cascade_reorder — вместе с реранкером верха (превью для него
берутся из той же записи), по тем же меткам.

НУЖНО: видеокарта и папка записи эпизода 94 (--freeze). Запись живёт там,
где снимались прогоны judge9-17; в git её нет (гигабайты). Без неё скрипт
честно отказывает — калибровка порогов (calibrate_vision.py) от неё не
зависит и идёт на золотом наборе из git.

    python scripts/cascade_model_eval.py --freeze temp_selection_freeze/94_dagger_test

Вывод «лучше SigLIP2 без потерь» — по двум числам: годных в первых 20 больше
базовой линии и ни одной кучи, где лучший размеченный кадр ушёл из первых
20 (rankcheck_failures). Кучи, где размеченное превью не нашлось в записи,
не сравниваются (их число печатается). Метки — Claude (см. labels.json,
имя оценщика)."""
import argparse
import gzip
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
FIX = os.path.join(REPO, "tests", "fixtures", "pool_regression")
FREEZE = os.path.join(REPO, "temp_selection_freeze", "94_dagger_test")


def recorded_bodies(freeze, runs):
    """ключ запроса -> путь к телу ответа (только успешные GET)."""
    roots = [os.path.join(freeze, "net")] + [os.path.join(freeze, "runs", r, "net_overlay")
                                             for r in runs]
    out = {}
    for root in roots:
        idx = os.path.join(root, "index.jsonl")
        if not os.path.exists(idx):
            continue
        with open(idx, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("kind") != "response" or not 200 <= (rec.get("status") or 0) < 300:
                    continue
                out.setdefault(rec["key"], os.path.join(root, "bodies", rec["body"][:2], rec["body"]))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--freeze", default=FREEZE)
    p.add_argument("--handoff", type=int, default=20)
    a = p.parse_args(argv)
    if not os.path.isdir(os.path.join(a.freeze, "net")):
        raise SystemExit(f"нет записи эпизода 94 ({a.freeze}) — см. докстринг: замер каскада "
                         f"идёт только там, где снимались прогоны judge9-17")
    os.environ.setdefault("CASCADE_CACHE_DIR", tempfile.mkdtemp(prefix="casc_eval_"))
    os.environ.setdefault("RERANK_CACHE_DIR", tempfile.mkdtemp(prefix="rerank_eval_"))
    sys.argv = ["pipeline_smart.py", REPO]
    from PIL import Image
    import net_recorder
    import pipeline_smart as ps
    import pool_recall as pr
    import qwen_vl_embed
    import qwen_vl_rerank
    if not qwen_vl_embed.available() or not qwen_vl_rerank.available():
        raise SystemExit("Qwen3-VL недоступна — см. строки выше")

    labels = json.load(open(os.path.join(FIX, "labels.json"), encoding="utf-8"))
    base = json.load(open(os.path.join(FIX, "baseline.json"), encoding="utf-8"))["metrics"]
    lab = pr.merged_labels(labels)
    pools = json.load(gzip.open(os.path.join(FIX, "pools.json.gz"), "rt", encoding="utf-8"))
    plan = json.load(open(os.path.join(FIX, "specs.json"), encoding="utf-8"))
    specs = {u.get("text"): dict(u, queries=u.get("queries_for") or u.get("queries") or [])
             for u in ((plan or {}).get("units") or {}).values()}
    runs_dir = os.path.join(a.freeze, "runs")
    bodies = recorded_bodies(a.freeze, sorted(os.listdir(runs_dir)) if os.path.isdir(runs_dir) else [])

    def body_of(url):
        return bodies.get(net_recorder.request_key("GET", url, None))

    # Векторы превью — пачками, заранее: cascade_reorder берёт их из кэша
    # процесса; превью без записи остаются без вектора (как в рендере
    # превью, которое не скачалось, — в хвост).
    todo, lost_labeled = {}, set()
    for rec in pools:
        for r in rec["rows"]:
            cand = pr._cand(r, rec["kind"])
            ident = ps._cascade_ident(cand, r["probe_url"])
            path = body_of(r["probe_url"])
            if path is None:
                if lab.get(f"{rec['index']}|{rec['kind']}|{r['id']}") is not None:
                    lost_labeled.add(f"{rec['run']}|{rec['index']}|{rec['kind']}")
                continue
            todo[ident] = path
    idents = sorted(todo)
    print(f"превью в записи: {len(idents)}; куч с потерянными размеченными превью: "
          f"{len(lost_labeled)}")
    bs = 64
    for k in range(0, len(idents), bs):
        part = idents[k:k + bs]
        imgs = []
        for ident in part:
            with Image.open(todo[ident]) as im:
                imgs.append(im.convert("RGB"))
        vecs = qwen_vl_embed.embed_images(imgs)
        if vecs is None:
            raise SystemExit("Qwen3-VL-Embedding сорвалась — см. строку выше")
        for ident, v in zip(part, vecs):
            ps._CASCADE_EMB[ps._cascade_key(ident)] = v
        print(f"  {min(k + bs, len(idents))}/{len(idents)}", end="\r", flush=True)
    print()

    def probe_from_record(c, dest):
        path = body_of(c["_probe"])
        if path is None:
            raise OSError("превью нет в записи")
        shutil.copyfile(path, dest)

    orders = {}
    tmp = tempfile.mkdtemp(prefix="casc_eval_run_")
    try:
        for rec in pools:
            kind = rec["kind"]
            cands = [pr._cand(r, kind) for r in rec["rows"]]
            spec = specs.get(rec.get("block_text"))
            brief = rec.get("shot_brief") or rec.get("query")
            ranked = ps.cascade_reorder(
                cands, ps.cascade_texts(spec, brief, kind), os.path.join(tmp, "x"),
                probe_from_record, index=rec["index"], url_of=lambda c: c["_probe"],
                claims=ps.cascade_claims(spec, kind),
                rerank_text=ps.cascade_rerank_text(spec, brief))
            orders[f"{rec['run']}|{rec['index']}|{kind}"] = [str(c["id"]) for c in ranked]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    now = {k: m for k, m in pr.rank_metrics(orders, labels, a.handoff).items()
           if k not in lost_labeled}
    base_same = {k: m for k, m in base.items() if k in now}
    g_now = sum(m["good"] for m in now.values())
    g_base = sum(b["good"] for b in base_same.values())
    bad = pr.rankcheck_failures(now, base_same)
    better = [k for k, m in now.items() if k in base_same
              and (m["best"] if m["best"] is not None else -1)
              > (base_same[k]["best"] if base_same[k]["best"] is not None else -1)]
    print(f"Qwen3-VL (эмбеддинг + реранкер верха) — годных в первых {a.handoff}: {g_now} "
          f"(SigLIP2 на тех же {len(base_same)} кучах: {g_base})")
    print(f"куч, где лучший кадр в первых {a.handoff} стал лучше: {len(better)}; хуже: "
          f"{sum(1 for x in bad if 'лучший' in x)}")
    for line in bad:
        print("  ХУЖЕ:", line)
    verdict = not bad and g_now > g_base
    print("ВЫВОД:", "лучше SigLIP2 без потерь" if verdict else "НЕ лучше SigLIP2 без потерь")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
