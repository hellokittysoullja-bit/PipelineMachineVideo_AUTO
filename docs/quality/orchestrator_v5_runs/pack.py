#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Сборка архива замеров оркестратора v5 из черновой папки сессии (разово, 27.09).

Запускался один раз, из черновой папки сессии (S — папка этого файла там).
Черновая папка не сохраняется, поэтому повторно скрипт не запустить: он
лежит здесь, чтобы было видно, что взято в архив и как проверено.

Кладёт в docs/quality/orchestrator_v5_runs/ только текст: планы, сырые ответы
моделей, ответы судьи, метки, итоги, журналы и код замеров. Картинки,
эмбеддинги и кэши отбора (temp_smart) не берутся. Локальные пути заменены на
<scratchpad>/<repo>. Перед записью — проверка на ключи из .env, «sk-»,
значения заголовка Authorization и адреса почты: найдено — архив не пишется.
"""
import glob
import gzip
import io
import json
import os
import re
import sys

S = os.path.dirname(os.path.abspath(__file__))
REPO = "/home/user/PipelineMachineVideo_AUTO"
OUT = os.path.join(REPO, "docs", "quality", "orchestrator_v5_runs")
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, S)


def rd(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


def jl(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def rel(p):
    return os.path.relpath(p, S)


# ---------- 4.1 планировщик на тексте ----------
def planner_bench():
    reports, plans, raw = {}, {}, {}
    for d in sorted(glob.glob(os.path.join(S, "bench", "*"))):
        run = os.path.basename(d)
        if os.path.exists(os.path.join(d, "report.json")):
            reports[run] = jl(os.path.join(d, "report.json"))
        for ep_dir in sorted(glob.glob(os.path.join(d, "*"))):
            if not os.path.isdir(ep_dir):
                continue
            ep = os.path.basename(ep_dir)
            mp = os.path.join(ep_dir, "media_plan")
            entry = {}
            for name, key in (("stock_queries.json", "plan"), ("stock_queries_v3.json", "plan_v3"),
                              ("film_bible.json", "bible")):
                if os.path.exists(os.path.join(mp, name)):
                    entry[key] = jl(os.path.join(mp, name))
            # shot_plan.txt не берётся: это читаемая выдача того же плана (3.2 МБ).
            if entry:
                plans.setdefault(run, {})[ep] = entry
            for cache in ("stock_query_cache", "stock_query_cache_v3"):
                for p in sorted(glob.glob(os.path.join(mp, cache, "*.txt"))):
                    raw.setdefault(run, {}).setdefault(ep, {})[f"{cache}/{os.path.basename(p)}"] = rd(p)
    return reports, plans, raw


# ---------- 4.2 судья на снимке пулов ----------
def judge_bench():
    dumps = {}
    for p in sorted(glob.glob(os.path.join(S, "judge_dump*", "judge_*.json"))
                    + glob.glob(os.path.join(S, "judge_bench", "*.json"))
                    + glob.glob(os.path.join(S, "cascade_v5e*", "*.json"))):
        dumps[rel(p)] = jl(p)
    store = {}
    for p in sorted(glob.glob(os.path.join(S, "judge_store", "**", "*.json"), recursive=True)):
        store[rel(p)] = jl(p)
    return dumps, store


# ---------- 4.3 бесплатная зона, живой пул ----------
def free_zone():
    import m5_labels
    gens = {}
    # M5: руки двух процессов; метки — листы M5 (перенос cN + приманки), M5b, M5c.
    runs = {}
    for g in ("m5_A", "m5_B"):
        runs.update(jl(os.path.join(S, g, "runs.json")))
    pre = {}
    for g in ("m5_A", "m5_B"):
        p = os.path.join(S, g, "runs_pre_m5c.json")
        if os.path.exists(p):
            pre.update(jl(p))
    raw_labels = {}
    for p in ("m5_final_labels.json", "m5_sheet_labels.json", "m5b_sheet_labels.json", "m5c_sheet_labels.json"):
        raw_labels[p] = jl(os.path.join(S, p))
    sheets = {rel(p): jl(p) for p in sorted(glob.glob(os.path.join(S, "m5_final", "sheets", "*.json")))}
    gens["M5"] = {"runs": runs, "runs_pre_m5c": pre, "labels": m5_labels.labels(),
                  "labels_raw": raw_labels, "sheets": sheets,
                  "score_snapshot_34": jl(os.path.join(S, "m5_final", "score.json"))}
    # Ранние поколения (M2, M4, M4c): итог и строки прогона, как их считал свой скрипт.
    for gen, final, parts, lab_files in (
            ("M2", "m2_final", ("m2_A", "m2_B", "m2n_A", "m2n_B", "m2_all"), ("m2_labels.json", "m2_final_labels.json")),
            ("M4", "m4_final", ("m4_A", "m4_B"), ("m4_final_labels.json",)),
            ("M4c", "m4c_final", ("m4c_A", "m4c_B"), ("m4c_final_labels.json", "m4c_sheet_labels.json"))):
        g = {"runs": jl(os.path.join(S, final, "runs.json")),
             "score": jl(os.path.join(S, final, "score.json")),
             "sheets": {rel(p): jl(p) for p in sorted(glob.glob(os.path.join(S, final, "sheets", "*.json")))},
             "labels_raw": {f: jl(os.path.join(S, f)) for f in lab_files if os.path.exists(os.path.join(S, f))},
             "parts": {p: jl(os.path.join(S, p, "runs.json")) for p in parts
                       if os.path.exists(os.path.join(S, p, "runs.json"))}}
        gens[gen] = g
    for extra in ("m4t_out", "m2_out_v5h_partial", "m3_out_v5i_partial"):
        p = os.path.join(S, extra, "runs.json")
        if os.path.exists(p):
            gens.setdefault("partial", {})[extra] = jl(p)
    return gens


# ---------- 4.5 судья в бесплатной пятёрке ----------
def free_judge():
    return {"eval": jl(os.path.join(S, "free_judge_eval.json")),
            "raw": {rel(p): jl(p) for p in sorted(glob.glob(os.path.join(S, "judge_free_cache", "*.json")))}}


def free_rank():
    return {rel(p): jl(p) for p in sorted(glob.glob(os.path.join(S, "frs", "*.json")))}


def e2e():
    out = {}
    for base in ("e2e", "e2e_hyb", "e2e_hybS"):
        for p in sorted(glob.glob(os.path.join(S, base, "**", "*"), recursive=True)):
            if not os.path.isfile(p) or "temp_smart" in p or p.endswith((".jpg", ".npy", ".mp4", ".srt")):
                continue
            if "stock_query_cache" in p:
                out[rel(p)] = rd(p)
            elif p.endswith((".json", ".jsonl", ".txt")):
                out[rel(p)] = jl(p) if p.endswith(".json") else rd(p)
    return out


# ---------- отсев брака силами DeepSeek (27.09): слова подписи, проверка подписей, ловушки ----------
def garbage_cut():
    import lex_feasibility as lf
    import cap_check as cc
    out = {"lexicon": jl(os.path.join(S, "lex_feasibility.json")), "lexicon_prompt": lf.PROMPT,
           "caption_check_prompt": cc.PROMPT, "caption_check": []}
    for p in sorted(glob.glob(os.path.join(S, "cap_check_cache", "*.json"))):
        out["caption_check"].append(jl(p))
    # Какой пятёрке какой ответ: ключ кэша = хэш вопроса; вопрос восстанавливается из данных.
    import hashlib
    index = {}
    for key, row in cc.runs.items():
        for a in cc.ARMS:
            head = row["arms"][a]["head"]
            ids = tuple(h["id"] for h in head)
            for tag, reasoning in (("r1", True), ("r0", False)):
                h = hashlib.sha256(f"{tag}|ds/deepseek-v4-flash|{reasoning}|{cc.prompt_for(key, head)}".encode()).hexdigest()[:24]
                cp = os.path.join(S, "cap_check_cache", h + ".json")
                if os.path.exists(cp):
                    index.setdefault(tag, {})[f"{key}|{'|'.join(ids)}"] = jl(cp)
    out["caption_check"] = index
    d = jl(os.path.join(S, "m5feat", "traps_all.json"))
    out["features"] = d["feat"]
    out["items"] = d["items"]
    return out


NOISE_LOGS = {"apt.log", "torch.log", "clip_dl.log"}


def logs():
    out = {}
    for p in sorted(glob.glob(os.path.join(S, "*.log")) + glob.glob(os.path.join(S, "bench", "*", "*.log"))
                    + glob.glob(os.path.join(S, "frs", "*.log"))):
        if os.path.basename(p) in NOISE_LOGS:
            continue
        out[rel(p)] = rd(p)
    return out


ARM_CODE = ["v3/sqp_v3.py", "v3/shot_research_v3.py", "sqp_hybN.py", "tsqp_hybN.py", "sqp_hybS_final.py",
            "tsqp_hybS_final.py", "hybS_final.diff", "pr_now.txt"]
SKIP_CODE = {"ps_ctrl_backup.py", "ps_ctrl_keep.py", "ps_keep.py", "sqp_ctrl_keep.py", "sqp_full.py", "sqp_keep.py",
             "sqp_v3_orig.py", "tsqp_full.py", "sqp_hybS_keep.py", "hyb_ctrl_backup.py"}


def code():
    out = {}
    for p in sorted(glob.glob(os.path.join(S, "*.py")) + glob.glob(os.path.join(S, "*.sh"))):
        b = os.path.basename(p)
        if b in SKIP_CODE or b == "pack_archive.py":
            continue
        out[b] = rd(p)
    for p in ARM_CODE:
        out[p] = rd(os.path.join(S, p))
    return out


# ---------- проверка на секреты ----------
def secret_values():
    vals = []
    env = os.path.join(REPO, ".env")
    if os.path.exists(env):
        for line in rd(env).splitlines():
            if "=" not in line or line.strip().startswith("#"):
                continue
            k, v = line.split("=", 1)
            v = v.strip().strip('"').strip("'")
            if len(v) >= 8 and re.search(r"KEY|SECRET|TOKEN|PASSWORD|CLIENT", k, re.I):
                vals.append(v)
    return vals


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
AUTH_RE = re.compile(r"Authorization[\"']?\s*[:=]\s*[\"']?(?:Bearer\s+)?[A-Za-z0-9_\-]{12,}")
SK_RE = re.compile(r"\bsk-[A-Za-z0-9_\-]{10,}")
KEYPARAM_RE = re.compile(r"[?&](?:key|api_key|apikey|client_secret|access_token)=([A-Za-z0-9_\-]{8,})")


def scan(name, text, secrets):
    bad = []
    for v in secrets:
        if v in text:
            bad.append("значение ключа из .env")
    if SK_RE.search(text):
        bad.append("sk-...")
    if AUTH_RE.search(text):
        bad.append("Authorization: <значение>")
    m = KEYPARAM_RE.search(text)
    if m:
        bad.append("ключ в параметре адреса")
    emails = sorted(set(EMAIL_RE.findall(text)))
    return bad, emails


def dump(name, obj, secrets, allowed_emails):
    text = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    text = text.replace(S, "<scratchpad>").replace(REPO, "<repo>")
    bad, emails = scan(name, text, secrets)
    strange = [e for e in emails if e not in allowed_emails]
    if bad or strange:
        masked = [e[0] + "***@" + e.split("@", 1)[1] for e in strange]
        raise SystemExit(f"{name}: найдено {bad} почта {masked} — архив не пишется")
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0, compresslevel=9) as g:
        g.write(text.encode("utf-8"))
    path = os.path.join(OUT, name)
    with open(path, "wb") as f:
        f.write(buf.getvalue())
    print(f"{name}: {len(text) / 1e6:.2f} МБ текста -> {len(buf.getvalue()) / 1e6:.2f} МБ", flush=True)
    return emails


def main():
    os.makedirs(OUT, exist_ok=True)
    secrets = secret_values()
    # Почта, которой место в архиве: только общеизвестные служебные адреса из
    # подписей источников. Адрес пользователя сюда не входит никогда.
    allowed = set(json.loads(sys.argv[1])) if len(sys.argv) > 1 else set()
    reports, plans, raw = planner_bench()
    found = {}
    # Ожидания, по которым считались отчёты, — копия на день замера: правка
    # gold.json потом не должна менять архивные числа.
    gold = jl(os.path.join(REPO, "tests", "fixtures", "orchestrator_bench", "gold.json"))
    found["planner_bench_reports.json.gz"] = dump("planner_bench_reports.json.gz",
                                                  {"gold": gold, "runs": reports}, secrets, allowed)
    found["planner_bench_plans.json.gz"] = dump("planner_bench_plans.json.gz", plans, secrets, allowed)
    found["planner_bench_raw.json.gz"] = dump("planner_bench_raw.json.gz", raw, secrets, allowed)
    dumps, store = judge_bench()
    found["judge_bench.json.gz"] = dump("judge_bench.json.gz", dumps, secrets, allowed)
    found["judge_bench_raw.json.gz"] = dump("judge_bench_raw.json.gz", store, secrets, allowed)
    found["free_zone.json.gz"] = dump("free_zone.json.gz", free_zone(), secrets, allowed)
    found["free_judge.json.gz"] = dump("free_judge.json.gz", free_judge(), secrets, allowed)
    found["free_rank.json.gz"] = dump("free_rank.json.gz", free_rank(), secrets, allowed)
    found["e2e.json.gz"] = dump("e2e.json.gz", e2e(), secrets, allowed)
    found["garbage_cut.json.gz"] = dump("garbage_cut.json.gz", garbage_cut(), secrets, allowed)
    found["logs.json.gz"] = dump("logs.json.gz", logs(), secrets, allowed)
    found["code.json.gz"] = dump("code.json.gz", code(), secrets, allowed)
    print("почта в архиве:", {k: [e[0] + "***@" + e.split("@", 1)[1] for e in v] for k, v in found.items() if v})


if __name__ == "__main__":
    main()
