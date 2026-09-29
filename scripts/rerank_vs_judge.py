#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Реранкер Qwen3-VL-Reranker-2B против судьи — на тех же размеченных кадрах.

    python scripts/rerank_vs_judge.py --judge <bench_photo_*.json> [ещё ...] \
        --img-store <папка превью> --run-dir <runs/pool0> --episode <videos/94_...> \
        [--out отчёт.json]

ЗАЧЕМ. Судья (Qwen 3.7 Plus через шлюз) — платный и медленный: на слот сетка
плюс проверка до пяти финалистов по пунктам. Реранкер считается на своей
видеокарте, бесплатно и быстро. Вопрос владельца: можно ли, поставив его
перед судьёй, задавать судье меньше вопросов — без потери выбора. Отвечаем
числами на кадрах, где ответы судьи УЖЕ записаны (pool_recall.py bench), —
то есть без единого нового платного вызова:

  1. пары: верно ли реранкер упорядочивает кадры с разной меткой (как
     «пары верно» у судьи);
  2. сам по себе: какой кадр он поставил бы на экран из первых --handoff
     (лучший ли, брак ли) — против выбора судьи на тех же кадрах;
  3. как фильтр: попадает ли кадр, выбранный судьёй, в первые K по
     реранкеру (K = 1..handoff). Попадает во всех слотах при малом K —
     судье можно показывать K кадров вместо handoff при ТОМ ЖЕ выборе;
     для сравнения — то же для нынешнего порядка каскада (эмбеддинг).

Метки — разметка Claude (labels_claude.json стенда), не владельца.
Реранкер оценивает пару «текст — картинка» тремя текстами: фокус
спецификации кадра (так спрашивает рендер), фраза сценария, бриф.
"""
import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _img_path(store, cand_id):
    return os.path.join(store, re.sub(r"[^A-Za-z0-9_.-]", "_", str(cand_id)) + ".jpg")


def _slot_texts(run_dir, episode):
    """{ключ слота: {focus, phrase, brief}} — тексты, которыми спрашивать."""
    import pool_recall
    import stock_query_planner
    pools = pool_recall.load_pools(run_dir)
    specs = stock_query_planner.load_specs(episode) if episode else {}
    units = pool_recall._plan_unit_texts(episode) if episode else []
    out = {}
    for (slot, kind), rec in pools.items():
        phrase = rec.get("block_text") or ""
        spec = pool_recall._spec_for_block(specs, units, phrase) if specs else None
        brief = rec.get("shot_brief") or rec.get("query") or ""
        out[f"{slot}:{kind}"] = {"focus": (spec or {}).get("focus") or brief, "phrase": phrase,
                                 "brief": brief}
    return out


def evaluate(tiles_by_key, judge_slots, scores, handoff):
    """Метрики одного варианта текста. scores: {(ключ, id): оценка}."""
    res = {"pairs": 0, "pairs_ok": 0.0, "slots": 0, "rr_best": 0, "rr_brak": 0, "rr_label_sum": 0,
           "judge_best": 0, "judge_brak": 0, "judge_label_sum": 0,
           "judge_pick_in_rr_top": {}, "judge_pick_in_cascade_top": {},
           "best_in_rr_top": {}, "best_in_cascade_top": {}, "per_slot": []}
    for key, tiles in tiles_by_key.items():
        sc = [(t, scores.get((key, str(t["id"])))) for t in tiles]
        sc = [(t, s) for t, s in sc if s is not None]
        for i in range(len(sc)):
            for j in range(i + 1, len(sc)):
                (a, sa), (b, sb) = sc[i], sc[j]
                if a["label"] == b["label"]:
                    continue
                hi, lo = (sa, sb) if a["label"] > b["label"] else (sb, sa)
                res["pairs"] += 1
                res["pairs_ok"] += 1.0 if hi > lo else 0.5 if hi == lo else 0.0
        head = sorted([(t, s) for t, s in sc if t["pos"] < handoff], key=lambda ts: ts[0]["pos"])
        if not head:
            continue
        js = judge_slots.get(key)
        res["slots"] += 1
        best = max(t["label"] for t, _s in head)
        rr_order = sorted(range(len(head)), key=lambda k: (-head[k][1], k))
        pick = head[rr_order[0]][0]["label"]
        res["rr_best"] += pick == best
        res["rr_brak"] += pick == 0
        res["rr_label_sum"] += pick
        row = {"key": key, "best": best, "rr_pick": pick, "rr_pick_id": head[rr_order[0]][0]["id"]}
        if js:
            res["judge_best"] += js["pick"] == best
            res["judge_brak"] += js["pick"] == 0
            res["judge_label_sum"] += js["pick"]
            row["judge_pick"] = js["pick"]
            ids = [str(head[k][0]["id"]) for k in rr_order]
            cas = [str(t["id"]) for t, _s in head]
            jid = str(js.get("pick_id"))
            row["judge_pick_rr_rank"] = ids.index(jid) + 1 if jid in ids else None
            row["judge_pick_cascade_rank"] = cas.index(jid) + 1 if jid in cas else None
        best_ids = {str(t["id"]) for t, _s in head if t["label"] == best}
        rr_ids = [str(head[k][0]["id"]) for k in rr_order]
        cas_ids = [str(t["id"]) for t, _s in head]
        for K in range(1, handoff + 1):
            res["best_in_rr_top"][K] = res["best_in_rr_top"].get(K, 0) + bool(best_ids & set(rr_ids[:K]))
            res["best_in_cascade_top"][K] = (res["best_in_cascade_top"].get(K, 0)
                                             + bool(best_ids & set(cas_ids[:K])))
            if js:
                r_ = row.get("judge_pick_rr_rank")
                c_ = row.get("judge_pick_cascade_rank")
                res["judge_pick_in_rr_top"][K] = res["judge_pick_in_rr_top"].get(K, 0) + bool(r_ and r_ <= K)
                res["judge_pick_in_cascade_top"][K] = (res["judge_pick_in_cascade_top"].get(K, 0)
                                                       + bool(c_ and c_ <= K))
        res["per_slot"].append(row)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="append", required=True, help="выход pool_recall.py bench")
    ap.add_argument("--img-store", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--handoff", type=int, default=10)
    ap.add_argument("--texts", default="focus,phrase,brief")
    ap.add_argument("--out")
    a = ap.parse_args(argv)

    import qwen_vl_rerank
    benches = {os.path.basename(p): json.load(open(p, encoding="utf-8")) for p in a.judge}
    first = next(iter(benches.values()))
    tiles_by_key = {}
    for t in first["tiles"]:
        tiles_by_key.setdefault(t["key"], []).append(t)
    texts = _slot_texts(a.run_dir, a.episode)
    report = {"handoff": a.handoff, "frames": len(first["tiles"]), "slots": len(tiles_by_key),
              "labels": "Claude (labels_claude.json)", "variants": {}}
    t0 = time.time()
    if not qwen_vl_rerank.available():
        raise SystemExit(f"реранкер не загрузился: {qwen_vl_rerank.broken_reason()}")
    report["load_sec"] = round(time.time() - t0, 1)
    for variant in a.texts.split(","):
        scores, n, t = {}, 0, time.time()
        for key, tiles in tiles_by_key.items():
            q = texts.get(key, {}).get(variant)
            paths = [_img_path(a.img_store, x["id"]) for x in tiles]
            ok = [(x, p) for x, p in zip(tiles, paths) if os.path.exists(p)]
            if not q or not ok:
                continue
            got = qwen_vl_rerank.score(q, [p for _x, p in ok])
            if got is None:
                raise SystemExit(f"реранкер сорвался: {qwen_vl_rerank.broken_reason()}")
            for (x, _p), s in zip(ok, got):
                if s is not None:
                    scores[(key, str(x["id"]))] = float(s)
                    n += 1
        dt = time.time() - t
        report["variants"][variant] = {"scored": n, "sec": round(dt, 1),
                                       "pairs_per_sec": round(n / dt, 1) if dt else None,
                                       "by_judge_run": {}}
        for name, b in benches.items():
            js = {s["key"]: s for s in b["slots"]}
            ev = evaluate(tiles_by_key, js, scores, a.handoff)
            report["variants"][variant]["by_judge_run"][name] = ev
            print(f"\n=== текст «{variant}», судья {name}: оценено пар «текст—картинка» {n} "
                  f"за {dt:.1f} с ({n / max(dt, 1e-9):.1f}/с)")
            print(f"  пары верно: реранкер {ev['pairs_ok']:.1f}/{ev['pairs']} "
                  f"({100 * ev['pairs_ok'] / max(ev['pairs'], 1):.1f}%); судья этого прогона "
                  f"{b['pairs_ok']:.1f}/{b['pairs']} ({100 * b['pairs_ok'] / max(b['pairs'], 1):.1f}%)")
            print(f"  на экране из первых {a.handoff} ({ev['slots']} слотов): реранкер — лучший "
                  f"{ev['rr_best']}, брак {ev['rr_brak']}, сумма меток {ev['rr_label_sum']}; судья — "
                  f"лучший {ev['judge_best']}, брак {ev['judge_brak']}, сумма {ev['judge_label_sum']}")
            ks = [1, 2, 3, 5, a.handoff]
            print("  выбор судьи в первых K: по реранкеру "
                  + ", ".join(f"K={k}: {ev['judge_pick_in_rr_top'].get(k, 0)}" for k in ks)
                  + "; по каскаду " + ", ".join(f"K={k}: {ev['judge_pick_in_cascade_top'].get(k, 0)}"
                                               for k in ks))
            print("  лучший кадр в первых K: по реранкеру "
                  + ", ".join(f"K={k}: {ev['best_in_rr_top'].get(k, 0)}" for k in ks)
                  + "; по каскаду " + ", ".join(f"K={k}: {ev['best_in_cascade_top'].get(k, 0)}" for k in ks))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
