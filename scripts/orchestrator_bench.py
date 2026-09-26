#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Замер оркестратора кадров на тексте: понял ли он фразу, до всякого поиска.

ЗАЧЕМ. Прежние замеры планировщика смотрели на кадры одного эпизода или на
совпадение с брифами, написанными Claude. Здесь — ожидания, записанные ДО
прогона моделей (tests/fixtures/orchestrator_bench/gold.json) на шести нишах
(средневековье двумя эпизодами, психология, нейронаука, глубоководье,
биржевой крах 1929): местоимение разрешено в нужный предмет, образ
метафоры не попал ни в главное, ни в описание кадра, ни в запросы,
абстракция получила вещь, которую можно снять, а фразы не потеряны.
Ниши разделены на «разработку» и «проверку»: промпт правится только по
первым, итог — по вторым, чтобы не подогнать его под свой же набор.

Версия 3 (прежняя) замеряется своим замороженным кодом: путь к копии
модуля — --v3 (git show <ревизия>:scripts/stock_query_planner.py).

ЧЕСТНЫЕ ПРЕДЕЛЫ. Ожидания — Claude, а не владельца; совпадение ищется
словами, поэтому верный кадр другими словами считается промахом (число
снизу), а слово-ловушка в правильном контексте — утечкой (число сверху).
Этот замер про ПОНИМАНИЕ; что в стоках найдётся лучший кадр, он не
доказывает — это отдельный замер выдачи.
"""
import argparse
import contextlib
import importlib.util
import io
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
GOLD = os.path.join(REPO, "tests", "fixtures", "orchestrator_bench", "gold.json")


def kw_re(kw):
    """Слово ожидания: «слово» — целиком с окончанием s/es, «слово*» — начало
    слова, сочетание с пробелом — подстрокой."""
    kw = kw.lower().strip()
    if kw.endswith("*"):
        return re.compile(r"\b" + re.escape(kw[:-1]), re.I)
    if " " in kw:
        return re.compile(re.escape(kw), re.I)
    return re.compile(r"\b" + re.escape(kw) + r"(?:s|es)?\b", re.I)


def hits(text, words):
    return [w for w in words if kw_re(w).search(text or "")]


def load_blocks(ep_dir):
    import script_parser
    path = os.path.join(ep_dir, "script.txt")
    if not os.path.exists(path):
        path = os.path.join(ep_dir, "script_psychology.txt")
    with contextlib.redirect_stdout(io.StringIO()):
        return script_parser.parse_blocks(path), path


def load_card(ep_dir):
    import world_card
    for p in (os.path.join(ep_dir, "media_plan", "world_card.json"),
              os.path.join(ep_dir, "world", "world_card.json")):
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                card = json.load(f)
            if not world_card.validate(card):
                return card
    return None


def workdir_for(ep, ep_dir, out_root, strip_briefs):
    """Своя папка эпизода на прогон: сценарий (без [shot:], если так надо),
    паспорт; кэш ответов модели — внутри, повторный прогон бесплатен."""
    d = os.path.join(out_root, ep)
    os.makedirs(os.path.join(d, "media_plan"), exist_ok=True)
    blocks, src = load_blocks(ep_dir)
    with open(src, encoding="utf-8") as f:
        text = f.read()
    if strip_briefs:
        text = re.sub(r"\[shot:[^\]]*\]", "", text)
    with open(os.path.join(d, "script.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    card = load_card(ep_dir)
    if card:
        with open(os.path.join(d, "media_plan", "world_card.json"), "w", encoding="utf-8") as f:
            json.dump(card, f, ensure_ascii=False, indent=1)
    import script_parser
    with contextlib.redirect_stdout(io.StringIO()):
        blocks = script_parser.parse_blocks(os.path.join(d, "script.txt"))
    return d, blocks, card


def import_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_v3(v3, d, blocks, gw, model):
    """Прежний планировщик своим кодом. Он писал в тот же файл плана, но
    своей версии — план и кэш уводятся в свою подпапку."""
    import shot_planner_llm
    v3.PLAN_NAME = "stock_queries_v3.json"
    v3.CACHE_DIR_NAME = "stock_query_cache_v3"
    v3.plan_episode(d, blocks, gw, model=model, verbose=False)
    with open(os.path.join(d, "media_plan", v3.PLAN_NAME), encoding="utf-8") as f:
        units = json.load(f)["units"]
    out = {}
    for b in blocks:
        u = units.get(shot_planner_llm.unit_key(b["text"]))
        if u:
            out[b["text"]] = {"meaning": u.get("focus"), "shot": u.get("focus"),
                              "core": u["claims"][0]["text"], "claims": u["claims"],
                              "queries": u.get("queries") or [], "reading": None, "vehicle": [],
                              "traps": []}
    return out, {}


def run_v5(sqp, d, blocks, gw, model, reasoning, bible_on):
    import shot_planner_llm
    card = load_card(d)
    bible = "auto" if bible_on else None
    sqp.plan_episode(d, blocks, gw, model=model, verbose=False, reasoning=reasoning, card=card,
                     bible=bible)
    with open(os.path.join(d, "media_plan", sqp.PLAN_NAME), encoding="utf-8") as f:
        plan = json.load(f)
    out = {}
    for b in blocks:
        u = plan["units"].get(shot_planner_llm.unit_key(b["text"]))
        if u:
            out[b["text"]] = {"meaning": u.get("meaning"), "about": u.get("about"), "shot": u.get("focus"),
                              "core": u["claims"][0]["text"], "claims": u["claims"],
                              "queries": u.get("queries") or [], "reading": u.get("reading"),
                              "vehicle": u.get("vehicle") or [], "traps": u.get("traps") or [],
                              "brief": u.get("brief")}
    bible_obj = sqp.load_bible(d) if bible_on else None
    return out, {"failed": plan.get("failed") or {}, "stats": plan.get("stats") or {},
                 "bible": bible_obj}


def score(gold, results, bibles):
    """Оценка по ожиданиям. results — {эпизод: {текст фразы: задание}}."""
    rows = []
    for it in gold["items"]:
        res = results.get(it["ep"]) or {}
        spec = next((v for t, v in res.items() if t.startswith(it["text"])), None)
        row = {"ep": it["ep"], "text": it["text"], "planned": spec is not None, "judge": bool(it.get("judge"))}
        if spec:
            visible = " ".join([spec.get("core") or "", spec.get("shot") or ""])
            must = [c["text"] for c in spec.get("claims") or [] if c.get("tier") == "must"]
            groups = it.get("subject") or []
            row["subject_ok"] = all(hits(visible, g) for g in groups) if groups else None
            bad = it.get("not_core") or []
            row["leak_core"] = hits(" ".join([visible] + must), bad) if bad else []
            badq = it.get("not_query", bad)
            row["leak_query"] = sorted({w for q in spec.get("queries") or [] for w in hits(q, badq)})
            ok_read = it.get("reading")
            row["reading_ok"] = (spec.get("reading") in ok_read) if ok_read and spec.get("reading") else None
            row["spec"] = spec
        rows.append(row)
    never = {}
    for ep, exp in (gold.get("never") or {}).items():
        b = (bibles or {}).get(ep)
        if not b:
            continue
        txt = " ; ".join(b.get("never") or [])
        never[ep] = {"far_hits": hits(txt, exp["far"]), "own_banned": hits(txt, exp["own"]),
                     "never": b.get("never")}
    return rows, never


def summary(rows, split, gold):
    out = {}
    splits = list(gold["split"].items()) + [("all", [e for v in gold["split"].values() for e in v])]
    for name, eps in splits:
        rs = [r for r in rows if r["ep"] in eps]
        planned = [r for r in rs if r["planned"]]
        subj = [r for r in planned if r.get("subject_ok") is not None]
        read = [r for r in planned if r.get("reading_ok") is not None]
        out[name] = {
            "items": len(rs), "planned": len(planned),
            "subject_ok": f"{sum(1 for r in subj if r['subject_ok'])}/{len(subj)}",
            "leak_core": sum(1 for r in planned if r.get("leak_core")),
            "leak_query": sum(1 for r in planned if r.get("leak_query")),
            "reading_ok": f"{sum(1 for r in read if r['reading_ok'])}/{len(read)}" if read else "—",
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", required=True, help="v3 | v5")
    ap.add_argument("--out", required=True, help="папка прогона (кэш, планы, итог)")
    ap.add_argument("--model", default="ds/deepseek-v4-flash")
    ap.add_argument("--reasoning", default="on", choices=("on", "off", "default"))
    ap.add_argument("--no-bible", action="store_true")
    ap.add_argument("--keep-briefs", action="store_true", help="не снимать [shot:] автора")
    ap.add_argument("--v3", help="путь к замороженному модулю v3")
    ap.add_argument("--eps", default="", help="через запятую; пусто — все")
    ap.add_argument("--cap", type=int, default=300000)
    a = ap.parse_args(argv)
    from dotenv import load_dotenv
    load_dotenv(os.path.join(REPO, ".env"))
    import llm_gateway
    with open(GOLD, encoding="utf-8") as f:
        gold = json.load(f)
    eps = [e for e in gold["episodes"] if not a.eps or e in a.eps.split(",")]
    gw = llm_gateway.Gateway(spend_cap=a.cap)
    reasoning = {"on": True, "off": False, "default": None}[a.reasoning]
    results, bibles, extra = {}, {}, {}
    t0 = time.time()
    for ep in eps:
        d, blocks, _card = workdir_for(ep, os.path.join(REPO, gold["episodes"][ep]), a.out,
                                       strip_briefs=not a.keep_briefs)
        t = time.time()
        if a.arm == "v3":
            v3 = import_path("sqp_v3_bench", a.v3)
            res, info = run_v3(v3, d, blocks, gw, a.model)
        else:
            import stock_query_planner as sqp
            res, info = run_v5(sqp, d, blocks, gw, a.model, reasoning, not a.no_bible)
            bibles[ep] = info.get("bible")
        results[ep] = res
        extra[ep] = {"units": len(blocks), "planned": len(res), "sec": round(time.time() - t, 1),
                     **{k: v for k, v in info.items() if k != "bible"}}
        print(f"{ep}: заданий {len(res)} из {len(blocks)} за {extra[ep]['sec']} с", flush=True)
    rows, never = score(gold, results, bibles)
    summ = summary(rows, None, gold)
    report = {"arm": a.arm, "model": a.model, "reasoning": a.reasoning, "bible": not a.no_bible,
              "briefs": a.keep_briefs, "summary": summ, "never": never, "episodes": extra,
              "gateway": gw.summary(), "sec": round(time.time() - t0, 1), "rows": rows}
    with open(os.path.join(a.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    for ep, v in never.items():
        print(f"never {ep}: далёкое {v['far_hits']} | своё под запретом {v['own_banned']}")
    print(f"шлюз: {gw.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
