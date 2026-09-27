"""Сетка судьи в БЕСПЛАТНОЙ зоне (слоты 25+): лучше ли выбранный кадр.

Кучи — pools.jsonl прогонов харнесса в порядке бесплатной зоны (жанровый
фильтр выбрасывает, каскада нет). Бесплатная зона смотрит фото — первые
FAST_PHOTO_DEDUP_MAX_TRIES, видео — первые VIDEO_PREVIEW_POOL. На ЭТИХ ЖЕ
кандидатах сравниваются два выбора:
  A — как сейчас: прод-гейты (clip_relevance, is_relevant_candidate,
      frame_readable, aesthetic_score) и прод-ранжирование _score_and_pick;
  B — то же, но оценка сетки судьи (shot_judge.judge, прод-вопрос: фокус
      спецификации, строка мира паспорта) — первым ключом, как в платной зоне.
Метка победителя — разметка 0/1/2 (Claude). Видео — по одному кадру превью
(в проде: средний кадр и лента из трёх), это предел стенда.

    python grid_bench.py <папка эпизода> <labels.json>[,<ещё>] <run_dir>[,<ещё>]
    POOL_RECALL_IMG_STORE=<папка превью>
"""
import json, os, sys, time
REPO = "/home/user/PipelineMachineVideo_AUTO"
sys.path.insert(0, os.path.join(REPO, "scripts"))
EP, LABELS, RUNS = sys.argv[1], sys.argv[2].split(","), sys.argv[3].split(",")
sys.argv = ["x", EP]
os.chdir(REPO)
from dotenv import load_dotenv
load_dotenv(os.path.join(REPO, ".env"))
import pipeline_smart as ps
import pool_recall, shot_judge, world_card, llm_gateway, stock_query_planner

OUT = os.path.dirname(os.path.abspath(__file__))
tag = os.path.basename(EP.rstrip("/"))
labels = {}
for f in LABELS:
    d = json.load(open(f))
    for k, v in d.items():
        if isinstance(v, dict):
            labels.update(v)
        else:
            labels[k] = v
card = ps.episode_world_card()
setting = world_card.judge_setting(card) if card else None
specs = stock_query_planner.load_specs(EP) or {}
by_text = {}
for u in specs.values():
    if isinstance(u, dict) and u.get("text"):
        by_text[u["text"]] = u
pools = {}
for rd in RUNS:
    for k, rec in pool_recall.load_pools(rd).items():
        pools.setdefault(k, rec)
gw = llm_gateway.Gateway(spend_cap=int(os.environ.get("BENCH_SPEND", "60000")))
model = ps.shot_judge_model()
cache = os.path.join(OUT, "grid_cache")


def lab(idx, kind, cid):
    return labels.get(f"{idx}|{kind}|{cid}")


res = []
for (idx, kind), rec in sorted(pools.items()):
    head_n = ps.FAST_PHOTO_DEDUP_MAX_TRIES if kind == "photo" else ps.VIDEO_PREVIEW_POOL
    pool = ps.filter_alt_blocklist([dict(r) for r in rec["pool"]])[:head_n]
    q = rec["query"]
    info, paths = [], []
    for r in pool:
        p = pool_recall.fetch(r.get("probe_url"), r.get("headers"), r.get("id"))
        if not p:
            continue
        paths.append(p)
        rel = ps.clip_relevance(p, q)
        relevant = ps.is_relevant_candidate(p, q, relevance=rel)
        luma = None
        try:
            luma = ps.measure_luma(p)
        except Exception:
            pass
        if kind == "video" and luma is not None and luma < ps.VIDEO_MIN_LUMA_HARD:
            relevant = False
        aes = ps.aesthetic_score(p)
        info.append({"path": p, "p": {"id": r["id"]}, "is_dup_free": 1, "size_ok": 1,
                     "is_readable": ps.frame_readable(p), "is_relevant": 1 if relevant else 0,
                     "sharp_ok": 1 if (kind == "photo" or luma is None or luma >= ps.VIDEO_PREFER_MIN_LUMA) else 0,
                     "aesthetic_val": aes if aes is not None else 0.0, "luma_score": 0.0,
                     "min_d": 99, "relevance": rel})
    if len(info) < 2:
        continue
    a, _ = ps._score_and_pick(info, None)
    u = by_text.get(rec.get("block_text") or "")
    brief = (u or {}).get("focus") or rec.get("shot_brief") or q
    t = time.time()
    scores = shot_judge.judge(gw, model, phrase=rec.get("block_text"), brief=brief,
                              candidates=[(str(c["p"]["id"]), c["path"]) for c in info],
                              cache_dir=cache, report={}, kind=kind, setting=setting)
    dt = time.time() - t
    b = a
    if scores:
        for c in info:
            c["judge"] = scores.get(str(c["p"]["id"]))
        b, _ = ps._score_and_pick(info, None)
    la, lb = lab(idx, kind, a["p"]["id"]), lab(idx, kind, b["p"]["id"])
    best = max((x for x in (lab(idx, kind, c["p"]["id"]) for c in info) if x is not None), default=None)
    res.append({"index": idx, "kind": kind, "phrase": rec.get("block_text"), "n": len(info),
                "A": a["p"]["id"], "B": b["p"]["id"], "label_A": la, "label_B": lb, "best_in_head": best,
                "judge_ok": bool(scores), "sec": round(dt, 1),
                "cands": [(c["p"]["id"], lab(idx, kind, c["p"]["id"]), c.get("judge"), c["is_relevant"],
                           round(c["aesthetic_val"], 2)) for c in info]})
    print(idx, kind, len(info), "A", la, "B", lb, "лучший", best, f"{dt:.1f}s", "" if scores else "СУДЬЯ НЕ ОТВЕТИЛ")
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass

json.dump(res, open(os.path.join(OUT, f"grid_bench_{tag}.json"), "w"), ensure_ascii=False, indent=1)
for kind in ("photo", "video"):
    rs = [r for r in res if r["kind"] == kind]
    if not rs:
        continue
    known = [r for r in rs if r["label_A"] is not None and r["label_B"] is not None]
    better = sum(r["label_B"] > r["label_A"] for r in known)
    worse = sum(r["label_B"] < r["label_A"] for r in known)
    print(f"== {kind}: куч {len(rs)}, с метками обоих {len(known)}: лучше {better}, хуже {worse}, "
          f"сумма A {sum(r['label_A'] for r in known)} -> B {sum(r['label_B'] for r in known)}, "
          f"брак A {sum(r['label_A'] == 0 for r in known)} -> B {sum(r['label_B'] == 0 for r in known)}, "
          f"без метки у победителя A {sum(r['label_A'] is None for r in rs)} / B {sum(r['label_B'] is None for r in rs)}")
print("потрачено", gw.spent)
