#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Судья с заданием оркестратора: те же кадры, та же разметка — другое задание.

ЗАЧЕМ. Текстовый замер (orchestrator_bench.py) отвечает, понял ли
оркестратор фразу. Этот — меняется ли от этого РЕШЕНИЕ СУДЬИ на настоящих
кадрах с известным ответом: снимок пулов эпизода 94 (tests/fixtures/
pool_regression: пулы трёх прогонов, превью, метки 0/1/2), те же кадры —
первые 20 по прод-каскаду версии 3, — и ровно одно отличие между руками:
задание кадра (v3 из снимка или v5 из плана оркестратора). Ловушки
оркестратора в вопросе проверки замерены здесь же 26.09 и сняты: выбор
тот же (4 против 4), а вопрос длиннее.

Как в пайплайне: сетка судьи по фокусу задания, финалисты — лучшие по
сетке ∪ первые по порядку, проверка по утверждениям с миром отдельным
вопросом, вектор claims_vector, оценка сетки 0 — «ничего не найдено».
Выбранный кадр — лучший по (проверка, сетка, место в порядке).

ЧЕСТНЫЕ ПРЕДЕЛЫ. Метки — Claude, а не владельца; один эпизод, только фото,
девять фраз. Подписи кадров восстановлены по API источников (как их видит
прод), у части — нет. Пул — снятый, то есть влияние нового задания на
ВЫДАЧУ здесь не видно (это отдельный замер), только на выбор из готового.
"""
import argparse
import concurrent.futures
import gzip
import json
import os
import sys
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
FIX = os.path.join(REPO, "tests", "fixtures", "pool_regression")
UA = {"User-Agent": "PipelineMachineVideo/1.0 (research; orchestrator bench)"}


def _get_json(url, headers=None):
    req = urllib.request.Request(url, headers=dict(UA, **(headers or {})))
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_caption(cid):
    """Подпись кандидата так, как её собирает прод (текст источника плюс
    паспорт музейного предмета) — по номеру, через API источника."""
    cid = str(cid)
    try:
        if cid.startswith("met:"):
            j = _get_json("https://collectionapi.metmuseum.org/public/collection/v1/objects/" + cid[4:])
            extra = []
            if j.get("objectBeginDate") or j.get("objectEndDate"):
                extra.append(f"dated {j.get('objectBeginDate')}-{j.get('objectEndDate')}")
            if j.get("culture"):
                extra.append(j["culture"])
            return (j.get("title") or "") + ("; " + ", ".join(extra) if extra else "")
        if cid.startswith("chicago:"):
            j = _get_json(f"https://api.artic.edu/api/v1/artworks/{cid[8:]}?fields=title,date_start,date_end,place_of_origin")
            d = j.get("data") or {}
            return f"{d.get('title')}; dated {d.get('date_start')}-{d.get('date_end')}, {d.get('place_of_origin')}"
        if cid.startswith("cleveland:"):
            j = _get_json(f"https://openaccess-api.clevelandart.org/api/artworks/{cid[10:]}")
            d = j.get("data") or {}
            return f"{d.get('title')}; {d.get('creation_date')}, {', '.join(d.get('culture') or [])}"
        if cid.startswith("pixabay:"):
            key = os.environ.get("PIXABAY_API_KEY", "")
            j = _get_json("https://pixabay.com/api/?" + urllib.parse.urlencode({"key": key, "id": cid[8:]}))
            hits = j.get("hits") or []
            return hits[0].get("tags", "") if hits else ""
        if cid.startswith("openverse:"):
            j = _get_json(f"https://api.openverse.org/v1/images/{cid[10:]}/")
            return f"{j.get('title') or ''} [{j.get('source') or ''}]"
        if cid.startswith("commons:"):
            return ""
        if cid.isdigit():
            key = os.environ.get("PEXELS_API_KEY", "")
            j = _get_json(f"https://api.pexels.com/v1/photos/{cid}", {"Authorization": key})
            slug = (j.get("url") or "").rstrip("/").split("/")[-1].replace("-", " ")
            return f"{j.get('alt') or ''} {slug}".strip()
    except Exception:  # noqa: BLE001 — нет подписи: кадр судится без неё
        return ""
    return ""


def load_fixture(run, kind):
    with gzip.open(os.path.join(FIX, "pools.json.gz"), "rt", encoding="utf-8") as f:
        pools = [p for p in json.load(f) if p["run"] == run and p["kind"] == kind]
    labels = json.load(open(os.path.join(FIX, "labels.json"), encoding="utf-8"))
    return sorted(pools, key=lambda p: p["index"]), labels


def v3_specs():
    plan = json.load(open(os.path.join(FIX, "specs.json"), encoding="utf-8"))
    return {u["text"]: u for u in plan["units"].values()}


def v5_specs(plan_path):
    plan = json.load(open(plan_path, encoding="utf-8"))
    out = {}
    for u in plan["units"].values():
        spec = {"focus": u["focus"], "claims": u["claims"], "queries": u.get("queries_for") or []}
        for k in ("meaning", "about", "reading", "vehicle", "traps"):
            if u.get(k):
                spec[k] = u[k]
        out[u["text"]] = spec
    return out


def cascade_heads(specs, handoff=20):
    """Порядок КАЖДОГО снятого пула (все прогоны, фото и видео) прод-каскадом
    по тексту задания данной руки — эмбеддинги превью из снимка, без сети.
    Возвращает метрики головы (pool_recall.rank_metrics): лучший кадр среди
    первых handoff и число годных там. Пулы собраны запросами v3, поэтому
    мерится только то, как задание сортирует готовую кучу для судьи."""
    import tempfile
    import shutil
    import numpy as np
    import pipeline_smart as ps
    import pool_recall as pr
    z = np.load(os.path.join(FIX, "emb.npz"))
    for k, v in zip(z["keys"], z["vecs"]):
        ps._CASCADE_EMB[str(k)] = v.astype(np.float32)
    with gzip.open(os.path.join(FIX, "pools.json.gz"), "rt", encoding="utf-8") as f:
        pools = json.load(f)

    def no_probe(p, path):
        raise OSError("снимок: превью без эмбеддинга не скачивается")
    orders, tmp = {}, tempfile.mkdtemp(prefix="cascade_")
    try:
        for rec in pools:
            kind = rec["kind"]
            spec = specs.get(rec.get("block_text"))
            if not spec:
                continue
            cands = [pr._cand(r, kind) for r in rec["rows"]]
            ranked = ps.cascade_reorder(
                cands, ps.cascade_texts(spec, rec.get("shot_brief") or rec.get("query"), kind),
                os.path.join(tmp, "x"), no_probe, index=rec["index"], url_of=lambda p: p["_probe"],
                claims=ps.cascade_claims(spec, kind))
            orders[f"{rec['run']}|{rec['index']}|{kind}"] = [str(p["id"]) for p in ranked]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    labels = json.load(open(os.path.join(FIX, "labels.json"), encoding="utf-8"))
    return pr.rank_metrics(orders, labels, handoff)


def cmd_cascade(a):
    """Каскад платной зоны с заданием v3 (снимок) и v5 (план): уходит ли
    лучший размеченный кадр из первых 20, которых видит судья."""
    sys.argv = ["pipeline_smart.py", "/tmp"]
    v3 = cascade_heads(v3_specs(), a.head)
    v5 = cascade_heads(v5_specs(a.plan), a.head)
    worse, better = [], []
    for key in sorted(set(v3) & set(v5)):
        b3 = v3[key]["best"] if v3[key]["best"] is not None else -1
        b5 = v5[key]["best"] if v5[key]["best"] is not None else -1
        if b5 < b3:
            worse.append(f"{key}: {b3} -> {b5}")
        elif b5 > b3:
            better.append(f"{key}: {b3} -> {b5}")
    common = sorted(set(v3) & set(v5))
    out = {"pools": len(common), "good_v3": sum(v3[k]["good"] for k in common),
           "good_v5": sum(v5[k]["good"] for k in common), "best_worse": worse, "best_better": better}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "cascade_v3_v5.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": out, "v3": v3, "v5": v5}, f, ensure_ascii=False, indent=1)
    return 0


def cached_orders(store, pr, ps):
    """Порядок пулов снимка по прод-каскаду не зависит от задания руки —
    считается один раз и кладётся рядом с превью. Ключ — файлы снимка и
    исходник каскада: правка любого из них пересчитывает порядок."""
    import hashlib
    import inspect
    h = hashlib.sha256()
    for name in ("pools.json.gz", "specs.json", "emb.npz"):
        st = os.stat(os.path.join(FIX, name))
        h.update(f"{name}|{st.st_size}|{st.st_mtime_ns}".encode())
    for fn in (ps.cascade_reorder, ps.cascade_texts, ps.cascade_claims):
        h.update(inspect.getsource(fn).encode("utf-8"))
    path = os.path.join(store, "orders_" + h.hexdigest()[:16] + ".json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    orders = pr.fixture_orders(FIX)
    os.makedirs(store, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(orders, f)
    return orders


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spec", required=True, choices=("v3", "v5"))
    ap.add_argument("--plan", help="план оркестратора v5 (stock_queries.json)")
    ap.add_argument("--card", help="паспорт мира эпизода (для судьи)")
    ap.add_argument("--store", help="хранилище превью (для судьи; переиспользуется)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", default="judge12")
    ap.add_argument("--kind", default="photo")
    ap.add_argument("--head", type=int, default=20)
    ap.add_argument("--cap", type=int, default=150000)
    ap.add_argument("--cascade", action="store_true",
                    help="без модели: каскад по заданию v3 и v5 на всех снятых пулах")
    a = ap.parse_args(argv)
    if a.cascade:
        return cmd_cascade(a)
    if not a.card or not a.store:
        ap.error("для судьи нужны --card и --store")
    from dotenv import load_dotenv
    load_dotenv(os.path.join(REPO, ".env"))
    sys.argv = ["pipeline_smart.py", "/tmp"]
    import llm_gateway
    import pipeline_smart as ps
    import pool_recall as pr
    import shot_judge
    import world_card
    pr.IMG_STORE = a.store
    card = json.load(open(a.card, encoding="utf-8"))
    grid_setting = world_card.judge_setting(card)
    setting = world_card.claims_setting(card)
    cg_veto = not world_card.renders_allowed(card)
    pools, labels = load_fixture(a.run, a.kind)
    lab = pr.merged_labels(labels)
    orders = cached_orders(a.store, pr, ps)
    specs = v3_specs() if a.spec == "v3" else v5_specs(a.plan)
    model = ps.shot_judge_model()
    gw = llm_gateway.Gateway(spend_cap=a.cap)
    cache = os.path.join(a.store, "judge_cache")
    cap_path = os.path.join(a.store, "captions.json")
    captions = json.load(open(cap_path, encoding="utf-8")) if os.path.exists(cap_path) else {}
    stats = {"pairs": 0, "pairs_ok": 0.0, "bad": 0, "bad_accepted": 0, "good": 0, "good_vetoed": 0,
             "slots": []}
    for rec in pools:
        idx, kind = rec["index"], rec["kind"]
        spec = specs.get(rec["block_text"])
        if not spec:
            print(f"  слот {idx}: нет задания в плане — пропуск")
            continue
        order = orders[f"{a.run}|{idx}|{kind}"][:a.head]
        rows = {str(r["id"]): r for r in rec["rows"]}
        frames = []
        for pos, cid in enumerate(order):
            label = lab.get(f"{idx}|{kind}|{cid}")
            if label is None:
                continue
            if cid not in captions:
                captions[cid] = fetch_caption(cid)
            path = pr.fetch(rows[cid].get("probe_url"), None, cid)
            if path:
                keep = os.path.join(a.store, "keep", cid.replace(":", "_").replace("/", "_") + ".jpg")
                os.makedirs(os.path.dirname(keep), exist_ok=True)
                os.replace(path, keep)
                frames.append({"id": cid, "pos": pos, "label": label, "path": keep})
        grid = shot_judge.judge(gw, model, phrase=rec["block_text"], brief=spec["focus"],
                                candidates=[(f["id"], f["path"]) for f in frames], cache_dir=cache,
                                report={}, kind=kind, setting=grid_setting) or {}
        for f in frames:
            f["grid"] = grid.get(f["id"])
        top = sorted(frames, key=lambda f: (-(f["grid"] if isinstance(f["grid"], int) else -1), f["pos"]))
        fin_ids = {f["id"] for f in top[:ps.VERIFY_FINALISTS]} | {f["id"] for f in frames[:ps.VERIFY_FINALISTS]}
        finalists = [f for f in frames if f["id"] in fin_ids]
        def ask(f):
            return shot_judge.verify_claims(gw, model, phrase=rec["block_text"], spec=spec, setting=setting,
                                            path=f["path"], kind=kind, cache_dir=cache,
                                            reasoning=ps.VERIFY_REASONING, caption=captions.get(f["id"]),
                                            frames=1, world_separate=True)
        with concurrent.futures.ThreadPoolExecutor(len(finalists) or 1) as ex:
            got = list(ex.map(ask, finalists))
        for f, (ans, _info) in zip(finalists, got):
            f["answers"] = ans
            if ans is None:
                # Сбой шлюза — не ответ о кадре: замер с такими дырами не
                # сравнивает руки честно, число печатается рядом с итогом.
                stats["verify_failed"] = stats.get("verify_failed", 0) + 1
                continue
            vec = shot_judge.claims_vector(spec, ans, world_veto=True, cg_veto=cg_veto)
            nothing = shot_judge.nothing_met(spec, ans) or f.get("grid") == 0
            if vec is None:
                f["key"] = (-9,)
            elif nothing:
                f["key"] = (-5,)
            else:
                f["key"] = vec
            if f["label"] == 0:
                stats["bad"] += 1
                stats["bad_accepted"] += 1 if vec is not None and not nothing else 0
            else:
                stats["good"] += 1
                stats["good_vetoed"] += 1 if vec is None else 0
        # Как в рендере (26.09, «непроверенный кадр не обходит проверенных»):
        # если проверка в слоте состоялась, кадр, которого она не спросила,
        # стоит НИЖЕ отклонённых и сам считается браком.
        any_verified = any(f.get("answers") is not None for f in finalists)
        for f in frames:
            f.setdefault("key", (-7,) if any_verified else (-1,))
        judged = [f for f in frames if "answers" in f and f["answers"] is not None]
        for i in range(len(judged)):
            for j in range(i + 1, len(judged)):
                x, y = judged[i], judged[j]
                if x["label"] == y["label"]:
                    continue
                hi, lo = (x, y) if x["label"] > y["label"] else (y, x)
                stats["pairs"] += 1
                stats["pairs_ok"] += 1.0 if hi["key"] > lo["key"] else 0.5 if hi["key"] == lo["key"] else 0.0

        def pick():
            if not frames:
                return None
            best_f = max(frames, key=lambda f: (f["key"], f["grid"] if isinstance(f["grid"], int) else -1,
                                                -f["pos"]))
            # Лучший — отклонён проверкой (или не спрошен при состоявшейся
            # проверке): в рендере слот уходит во второй круг и поглощается
            # соседом, на экран этот кадр не встаёт.
            return None if best_f["key"][0] <= -5 else best_f
        p1 = pick()
        best = max((f["label"] for f in frames), default=None)
        stats["slots"].append({"index": idx, "phrase": rec["block_text"], "best": best,
                               "pick": p1 and p1["label"], "pick_id": p1 and p1["id"],
                               "frames": len(frames), "finalists": len(finalists),
                               "focus": spec["focus"], "claims": spec["claims"],
                               "detail": [{"id": f["id"], "label": f["label"], "grid": f.get("grid"),
                                           "key": list(f["key"]) if isinstance(f.get("key"), tuple) else f.get("key"),
                                           "answers": f.get("answers")} for f in frames]})
        print(f"  слот {idx}: кадров {len(frames)}, лучший {best}, выбран {p1 and p1['label']}", flush=True)
        with open(cap_path, "w", encoding="utf-8") as fcap:
            json.dump(captions, fcap, ensure_ascii=False)
    s = stats
    picked = sum(sl["pick"] or 0 for sl in s["slots"])
    best = sum(sl["best"] or 0 for sl in s["slots"])
    print(f"сумма меток выбранных: {picked} из лучших возможных {best}")
    if s.get("verify_failed"):
        print(f"ВНИМАНИЕ: проверка не состоялась у {s['verify_failed']} финалистов (сбой шлюза) — "
              f"перезапуск спросит только их (остальное в кэше)")
    print(f"пары верно {s['pairs_ok']:.1f}/{s['pairs']}; брак принят {s['bad_accepted']}/{s['bad']}; "
          f"годных отклонено {s['good_vetoed']}/{s['good']}; {gw.summary()}")
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"judge_{a.spec}.json"), "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
