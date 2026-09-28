#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Замер модели каскада на размеченных кучах: та же формула каскада,
другие векторы.

Гейт регрессий (pool_recall.py rankcheck) хранит векторы SigLIP2 и сверяет
порядок с базовой линией: 49 куч эп.94 (judge9/11/12), 856 меток, годных в
первых 20 — 204. Картинок в нём нет. Этот скрипт берёт сами превью из
записей сети харнесса (temp_selection_freeze/94_dagger_test: основная
запись и net_overlay прогонов judge9/11/12 — ровно те прогоны, из которых
сняты кучи), считает векторы выбранной моделью каскада и ранжирует теми же
функциями прода (cascade_reorder, cascade_texts, cascade_claims) по тем же
меткам. Кадры без вектора в базовой линии не участвуют и здесь — сравнение
на одном наборе.

    CASCADE_MODEL=qwen3vl python scripts/cascade_model_eval.py
    python scripts/cascade_model_eval.py --model siglip2     # контроль: 204

Решение о смене CASCADE_MODEL — по двум числам: годных в первых 20 больше
204 и ни одной кучи, где лучший размеченный кадр ушёл из первых 20
(rankcheck_failures). Метки — Claude (см. labels.json, имя оценщика)."""
import argparse
import gzip
import json
import os
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


def _orders_without_siglip(pr, fix):
    """fixture_orders, но без векторов SigLIP2 снимка: у Qwen свои ключи, а
    чужие векторы в кэше каскада процесса лишь занимают память."""
    import numpy as np
    real = np.load

    def empty_load(path, *a, **k):
        if str(path).endswith("emb.npz"):
            return {"keys": [], "vecs": []}
        return real(path, *a, **k)
    np.load = empty_load
    try:
        return pr.fixture_orders(fix)
    finally:
        np.load = real


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", choices=("siglip2", "qwen3vl"),
                   default=os.environ.get("CASCADE_MODEL", "qwen3vl"))
    p.add_argument("--freeze", default=FREEZE)
    p.add_argument("--handoff", type=int, default=20)
    a = p.parse_args(argv)
    os.environ["CASCADE_MODEL"] = a.model
    os.environ.setdefault("CASCADE_CACHE_DIR", tempfile.mkdtemp(prefix="casc_eval_"))
    sys.argv = ["pipeline_smart.py", REPO]
    import numpy as np
    from PIL import Image
    import net_recorder
    import pipeline_smart as ps
    import pool_recall as pr

    labels = json.load(open(os.path.join(FIX, "labels.json"), encoding="utf-8"))
    base = json.load(open(os.path.join(FIX, "baseline.json"), encoding="utf-8"))
    if a.model == "siglip2":
        now = pr.rank_metrics(pr.fixture_orders(FIX), labels, a.handoff)
    else:
        if ps.cascade_model() != "qwen3vl":
            raise SystemExit("Qwen3-VL-Embedding недоступна — см. строку выше")
        import qwen_vl_embed
        z = np.load(os.path.join(FIX, "emb.npz"))
        have_base = set(str(k) for k in z["keys"])
        pools = json.load(gzip.open(os.path.join(FIX, "pools.json.gz"), "rt", encoding="utf-8"))
        runs_dir = os.path.join(a.freeze, "runs")
        bodies = recorded_bodies(a.freeze, sorted(os.listdir(runs_dir)) if os.path.isdir(runs_dir)
                                 else [])
        # Какие превью ранжировались в базовой линии (ключ SigLIP2) — их и
        # считаем моделью каскада, под её ключом.
        os.environ["CASCADE_MODEL"] = "siglip2"
        todo, missing, uncovered = {}, 0, set()
        for rec in pools:
            for r in rec["rows"]:
                cand = pr._cand(r, rec["kind"])
                ident = ps._cascade_ident(cand, r["probe_url"])
                if ps._cascade_key(ident) not in have_base:
                    continue
                path = bodies.get(net_recorder.request_key("GET", r["probe_url"], None))
                if path is None:
                    missing += 1
                    uncovered.add(ps._cascade_key(ident))
                    continue
                todo[ident] = path
        # Честное сравнение — на одном наборе: у SigLIP2 тоже убираются
        # превью, картинки которых нет в записях (у Qwen их не посчитать).
        keep = [k for k in z["keys"] if str(k) not in uncovered]
        sub_fix = tempfile.mkdtemp(prefix="casc_fix_")
        for name in ("labels.json", "pools.json.gz", "specs.json", "baseline.json"):
            os.symlink(os.path.join(FIX, name), os.path.join(sub_fix, name))
        vec = dict(zip((str(k) for k in z["keys"]), z["vecs"]))
        np.savez_compressed(os.path.join(sub_fix, "emb.npz"), keys=np.array(keep),
                            vecs=np.stack([vec[str(k)] for k in keep]))
        base_same = pr.rank_metrics(pr.fixture_orders(sub_fix), labels, a.handoff)
        ps._CASCADE_EMB.clear()
        print(f"SigLIP2 на том же наборе: годных в первых {a.handoff} "
              f"{sum(m['good'] for m in base_same.values())} (полный набор: "
              f"{sum(b['good'] for b in base['metrics'].values())})")
        base = {"metrics": base_same}
        os.environ["CASCADE_MODEL"] = a.model
        idents = sorted(todo)
        print(f"превью для модели: {len(idents)} (нет в записи сети: {missing})")
        bs = 64
        for k in range(0, len(idents), bs):
            part = idents[k:k + bs]
            imgs = []
            for ident in part:
                with Image.open(todo[ident]) as im:
                    imgs.append(im.convert("RGB"))
            vecs = qwen_vl_embed.embed_images(imgs)
            for ident, v in zip(part, vecs):
                ps._CASCADE_EMB[ps._cascade_key(ident)] = v
            print(f"  {min(k + bs, len(idents))}/{len(idents)}", end="\r", flush=True)
        print()
        now = pr.rank_metrics(_orders_without_siglip(pr, sub_fix), labels, a.handoff)
    g_now = sum(m["good"] for m in now.values())
    g_base = sum(b["good"] for b in base["metrics"].values())
    bad = pr.rankcheck_failures(now, base["metrics"])
    better = [k for k, m in now.items() if k in base["metrics"]
              and (m["best"] if m["best"] is not None else -1)
              > (base["metrics"][k]["best"] if base["metrics"][k]["best"] is not None else -1)]
    print(f"модель каскада: {a.model} — годных в первых {a.handoff}: {g_now} "
          f"(SigLIP2 на том же наборе: {g_base})")
    print(f"куч, где лучший кадр в первых {a.handoff} стал лучше: {len(better)}; хуже: "
          f"{sum(1 for x in bad if 'лучший' in x)}")
    for line in bad:
        print("  ХУЖЕ:", line)
    verdict = not bad and g_now > g_base
    print("ВЫВОД:", "лучше SigLIP2 без потерь — можно ставить CASCADE_MODEL=" + a.model
          if verdict else "не лучше SigLIP2 без потерь — оставить siglip2")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
