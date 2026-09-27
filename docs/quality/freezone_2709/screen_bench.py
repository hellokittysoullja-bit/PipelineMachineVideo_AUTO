"""Отсев по подписи в БЕСПЛАТНОЙ зоне: что он делает с тем, из чего она выбирает.

Кучи — pools.jsonl прогонов эп.94 (порядок кучи до каскада = порядок
бесплатной зоны). Бесплатная зона смотрит: фото — первые 5, видео — первые 20
(VIDEO_PREVIEW_POOL). Отсев спрашивается по первым HEAD кандидатам, затем
берётся та же голова из оставшихся. Метки 0/1/2 — Claude (fixture + extra).
"""
import gzip, json, os, sys, time
REPO = "/home/user/PipelineMachineVideo_AUTO"
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.argv = ["x", os.path.join(REPO, "videos/94_dagger_test")]
os.chdir(REPO)
from dotenv import load_dotenv
load_dotenv(os.path.join(REPO, ".env"))
import pipeline_smart as ps
import caption_screen, world_card, llm_gateway

HEAD = int(os.environ.get("HEAD", "30"))
OUT = os.path.dirname(os.path.abspath(__file__))
card = ps.episode_world_card()
print("active_for:", caption_screen.active_for(card))
labels = {}
fx = json.load(open("tests/fixtures/pool_regression/labels.json"))
for v in fx.values():
    labels.update(v)
labels.update(json.load(open("docs/quality/prejudge_2509/labels_ep94_extra.json")))
specs = json.load(open("tests/fixtures/pool_regression/specs.json"))["units"]
by_text = {u.get("text"): u for u in specs.values() if u.get("text")}

pools, seen = [], set()
for run in ("judge9", "judge11", "judge12"):
    for line in open(f"temp_selection_freeze/94_dagger_test/runs/{run}/pools.jsonl"):
        p = json.loads(line)
        k = (p["index"], p["kind"])
        if k in seen:
            continue
        seen.add(k)
        pools.append(p)
print("pools", len(pools))
gw = llm_gateway.Gateway(spend_cap=60000)
cache = os.path.join(OUT, "cache")
res = []
for p in pools:
    pool = ps.filter_alt_blocklist([dict(r) for r in p["pool"]])
    look = 5 if p["kind"] == "photo" else ps.VIDEO_PREVIEW_POOL
    head = pool[:HEAD]
    rows = [(str(c["id"]), c.get("channel"), (c.get("text") or "")[:300]) for c in head]
    u = by_text.get(p["block_text"]) or {}
    focus = u.get("focus") or p.get("shot_brief") or p["query"]
    t = time.time()
    drop, info = caption_screen.screen(gw, p["block_text"], focus, card, rows, cache_dir=cache)
    dt = time.time() - t
    after = [c for c in pool if str(c["id"]) not in drop]
    lab = lambda xs: [labels.get(f"{p['index']}|{p['kind']}|{c['id']}") for c in xs[:look]]
    b, a = lab(pool), lab(after)
    dropped_lab = [labels.get(f"{p['index']}|{p['kind']}|{i}") for i in drop]
    res.append({"index": p["index"], "kind": p["kind"], "phrase": p["block_text"], "look": look,
                "before": b, "after": a, "dropped": sorted(drop), "dropped_labels": dropped_lab,
                "dropped_text": [d["text"] for d in info["dropped"]], "sec": round(dt, 1),
                "price": info["price"], "error": info["error"]})
    print(p["index"], p["kind"], f"{dt:.1f}s", info["price"], "drop", len(drop), dropped_lab,
          "| before", b[:look], "| after", a[:look], info["error"] or "")
json.dump(res, open(os.path.join(OUT, f"screen_bench_head{HEAD}.json"), "w"), ensure_ascii=False, indent=1)
