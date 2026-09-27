#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Архив замеров оркестратора v5: числа документа пересчитываются из данных.

docs/quality/orchestrator_v5_runs/ хранит то, чем замерялся оркестратор 26–27.09:
планы и сырые ответы DeepSeek, ответы судьи на кадры, слепые метки, итоги
прогонов. Здесь таблицы docs/quality/ORCHESTRATOR_V5.md (разделы 4.1–4.5)
пересчитываются заново — прод-функциями там, где решение принимает код
(оценка ожиданий, вектор утверждений судьи), — и сверяются с документом.

Тест не ходит в сеть и не платит. Упал — значит, разошлись документ и
данные, или прод-логика решает иначе, чем решала на замере: тогда числа
документа описывают уже не текущий код, и это надо записать, а не подогнать.
"""
import glob
import gzip
import json
import os
import re
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
ARCH = os.path.join(REPO, "docs", "quality", "orchestrator_v5_runs")


def load(name):
    with gzip.open(os.path.join(ARCH, name)) as f:
        return json.loads(f.read().decode("utf-8"))


# ---------------------------------------------------------------- архив чистый

def test_archive_is_text_without_keys_or_mail():
    """В архиве только сжатый текст, README и скрипт сборки: ни картинок, ни
    ключей, ни почты. Значения ключей из .env сверяются, если файл есть."""
    names = sorted(os.listdir(ARCH))
    assert names and all(n.endswith((".json.gz", ".md", ".py")) for n in names), names
    secrets = []
    env = os.path.join(REPO, ".env")
    if os.path.exists(env):
        for line in open(env, encoding="utf-8"):
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                v = v.strip().strip('"').strip("'")
                if len(v) >= 8 and re.search(r"KEY|SECRET|TOKEN|PASSWORD|CLIENT", k, re.I):
                    secrets.append(v)
    for n in names:
        p = os.path.join(ARCH, n)
        text = (gzip.open(p).read() if n.endswith(".gz") else open(p, "rb").read()).decode("utf-8")
        assert not re.search(r"\bsk-[A-Za-z0-9_\-]{10,}", text), n
        assert not re.search(r"Authorization[\"']?\s*[:=]\s*[\"']?(?:Bearer\s+)?[A-Za-z0-9_\-]{12,}", text), n
        assert not re.search(r"[?&](?:key|api_key|apikey|client_secret|access_token)=[A-Za-z0-9_\-]{8,}", text), n
        assert not re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text), n
        assert not any(v in text for v in secrets), n


# ------------------------------------------------ 4.1 понимание фразы (текст)

def _bench_summary(runs, gold, run):
    import orchestrator_bench as ob
    res = {}
    for row in runs[run]["rows"]:
        if row.get("planned") and row.get("spec"):
            res.setdefault(row["ep"], {})[row["text"]] = row["spec"]
    rows, _never = ob.score(gold, res, None)
    return rows, ob.summary(rows, None, gold)


def _split(s, name):
    x = s[name]
    return x["subject_ok"], x["leak_core"], x["leak_query"], x["reading_ok"]


def test_planner_bench_table_4_1():
    """Таблица 4.1: v3 — три прогона по наборам, v5 — прогон v5n_all
    (редакция, по которой написана таблица); ожидания — копия gold.json на
    день замера. Оценка — прод-функция orchestrator_bench.score."""
    arch = load("planner_bench_reports.json.gz")
    gold, runs = arch["gold"], arch["runs"]
    v3 = {"dev": _split(_bench_summary(runs, gold, "v3_dev")[1], "dev"),
          "holdout": _split(_bench_summary(runs, gold, "v3_holdout")[1], "holdout"),
          "holdout2": _split(_bench_summary(runs, gold, "v3_holdout2")[1], "holdout2")}
    _rows, s5 = _bench_summary(runs, gold, "v5n_all")
    v5 = {k: _split(s5, k) for k in ("dev", "holdout", "holdout2")}
    assert v3 == {"dev": ("43/44", 13, 5, "—"), "holdout": ("19/20", 8, 3, "—"), "holdout2": ("23/23", 7, 4, "—")}
    assert v5 == {"dev": ("42/44", 1, 1, "14/14"), "holdout": ("19/20", 1, 1, "9/9"),
                  "holdout2": ("23/23", 1, 0, "21/23")}
    # Итог по трём наборам: 3 из 105 против 28, в запросах 2 против 12.
    assert sum(v[1] for v in v5.values()) == 3 and sum(v[1] for v in v3.values()) == 28
    assert sum(v[2] for v in v5.values()) == 2 and sum(v[2] for v in v3.values()) == 12


def test_planner_bench_shipped_prompt_and_second_v3_sample():
    """Текст, который стоит в коде (правило запросов v5o), на тех же ожиданиях:
    образа метафоры нет ни в главном, ни в запросах ни на одном наборе. И
    раздел 4.1 «вторая выборка плана v3 понимает не лучше первой» на 59
    фразах шести ниш живого пула (без ep02 и neuro): 16 и 12 фраз с образом
    в главном или в кадре, 7 и 8 — в запросах, у v5 — 0 и 0; главный
    предмет 51 из 52 у всех трёх."""
    arch = load("planner_bench_reports.json.gz")
    gold, runs = arch["gold"], arch["runs"]
    _rows, s = _bench_summary(runs, gold, "v5o_all")
    assert [_split(s, k)[:3] for k in ("dev", "holdout", "holdout2")] == \
        [("42/44", 0, 0), ("19/20", 0, 0), ("23/23", 0, 0)]
    live = {"ep94_hook", "psych", "deepsea", "crash1929", "gutenberg", "immunity"}

    def count(rows):
        rs = [r for r in rows if r["ep"] in live and r["planned"]]
        subj = [r for r in rs if r.get("subject_ok") is not None]
        return (len(rs), sum(1 for r in rs if r.get("leak_core")), sum(1 for r in rs if r.get("leak_query")),
                f"{sum(1 for r in subj if r['subject_ok'])}/{len(subj)}")

    first = []
    for run in ("v3_dev", "v3_holdout", "v3_holdout2"):
        first += _bench_summary(runs, gold, run)[0]
    assert count(first) == (59, 16, 7, "51/52")
    assert count(_bench_summary(runs, gold, "v3_rep1")[0]) == (59, 12, 8, "51/52")
    assert count(_bench_summary(runs, gold, "v5o_all")[0]) == (59, 0, 0, "51/52")


def test_planner_bench_cost_numbers():
    """Цена: v5 с нуля — 55 688 токенов баланса на 231 фразу (≈240 на фразу),
    v3 — около 40 на фразу."""
    runs = load("planner_bench_reports.json.gz")["runs"]
    units = sum(e["units"] for e in runs["v5n_all"]["episodes"].values())
    assert (runs["v5n_all"]["gateway"]["spent"], units) == (55688, 231)
    v3 = runs["v3_dev"]
    per = v3["gateway"]["spent"] / sum(e["units"] for e in v3["episodes"].values())
    assert 30 <= per <= 45, per


def test_planner_bench_archive_keeps_plans_and_raw_answers():
    """Каждый прогон с планом хранит и сырые ответы модели, по которым он
    собран: без них число «из ответа модели» проверить нечем."""
    plans, raw = load("planner_bench_plans.json.gz"), load("planner_bench_raw.json.gz")
    for run in ("v5n_all", "v5o_all", "v5p_all", "hybS2"):
        for ep, entry in plans[run].items():
            assert entry["plan"]["units"], (run, ep)
            assert raw[run][ep], (run, ep)
    assert sum(len(files) for eps in raw.values() for files in eps.values()) >= 1200


# ------------------------------------------------- 4.2 судья на снимке пулов

JUDGE_TABLE = {
    # дамп: (годный на экране, брак на экране, слот на второй круг,
    #        брак принят проверкой / брак, годный отклонён миром / годный, пары)
    "judge_dump/judge_v3_trapsoff.json": (5.0, 2.0, 3, (7, 38), (1, 22), (47.0, 61)),
    "judge_dump/judge_v5_trapsoff.json": (2.7, 2.3, 5, (3, 39), (4, 27), (58.5, 89)),
    "judge_dump_v5j/judge_v5.json": (4.0, 1.0, 5, (1, 34), (2, 23), (50.0, 68)),
    "judge_dump_v5k/judge_v5.json": (1.3, 3.3, 5, (4, 38), (1, 27), (64.5, 93)),
    "judge_dump_v5n1/judge_v5.json": (5.5, 1.0, 4, (8, 39), (2, 24), (53.5, 75)),
    "judge_dump_v5n2/judge_v5.json": (5.0, 0.5, 5, (2, 38), (2, 22), (44.5, 63)),
    "judge_dump_v5o1/judge_v5.json": (8.3, 1.3, 1, (11, 41), (2, 30), (85.5, 104)),
    "judge_dump_v5o2/judge_v5.json": (1.5, 3.5, 4, (9, 37), (2, 24), (38.0, 65)),
    "judge_dump_v5p2/judge_v5.json": (4.2, 2.3, 3, (5, 37), (1, 25), (51.0, 72)),
    "judge_dump_v5p1/judge_v5.json": (4.3, 1.3, 4, (4, 40), (2, 26), (68.5, 93)),
}
FINAL_PLANS = ["judge_dump_v5n1/judge_v5.json", "judge_dump_v5n2/judge_v5.json",
               "judge_dump_v5o1/judge_v5.json", "judge_dump_v5o2/judge_v5.json"]


def _judge_row(dump):
    """Пересчёт строки таблицы 4.2 из ответов судьи. Ключ кадра считает
    ТЕКУЩАЯ прод-логика (claims_vector, nothing_met) и сверяется с ключом,
    записанным на замере; выбор — как в рендере; ничья разводится случайно,
    поэтому берётся ожидание."""
    import shot_judge
    exp = bad_screen = 0.0
    none = bad = bad_acc = good = good_veto = 0
    pairs, pairs_ok = 0, 0.0
    for sl in dump["slots"]:
        spec = {"claims": sl["claims"], "focus": sl["focus"]}
        asked = any(f.get("answers") is not None for f in sl["detail"])
        frames = []
        for f in sl["detail"]:
            a = f.get("answers")
            g = f["grid"] if isinstance(f.get("grid"), int) else -1
            if a is None:
                frames.append((((-7,) if asked else (-1,)), g, f["label"], False))
                continue
            vec = shot_judge.claims_vector(spec, a, world_veto=True, cg_veto=True)
            nothing = shot_judge.nothing_met(spec, a) or f.get("grid") == 0
            key = (-9,) if vec is None else (-5,) if nothing else tuple(vec)
            assert list(key) == f["key"], (sl["index"], f["id"])
            if f["label"] == 0:
                bad += 1
                bad_acc += vec is not None and not nothing
            else:
                good += 1
                good_veto += vec is None
            frames.append((key, g, f["label"], True))
        judged = [x for x in frames if x[3]]
        for i in range(len(judged)):
            for j in range(i + 1, len(judged)):
                x, y = judged[i], judged[j]
                if x[2] == y[2]:
                    continue
                hi, lo = (x, y) if x[2] > y[2] else (y, x)
                pairs += 1
                pairs_ok += 1.0 if hi[0] > lo[0] else 0.5 if hi[0] == lo[0] else 0.0
        best = max((k, g) for k, g, _l, _a in frames)
        if best[0][0] <= -5:
            none += 1
            continue
        tied = [lab for k, g, lab, _a in frames if (k, g) == best]
        exp += sum(tied) / len(tied)
        bad_screen += sum(1 for lab in tied if lab == 0) / len(tied)
    return round(exp, 1), round(bad_screen, 1), none, (bad_acc, bad), (good_veto, good), (pairs_ok, pairs)


def test_judge_bench_table_4_2():
    dumps = load("judge_bench.json.gz")
    got = {name: _judge_row(dumps[name]) for name in JUDGE_TABLE}
    assert got == JUDGE_TABLE
    # Среднее четырёх планов итоговой редакции: 5.1 / 1.6 / 3.5 / 30 из 155 / 8 из 100 / 221.5 из 307.
    rows = [got[n] for n in FINAL_PLANS]
    assert round(sum(r[0] for r in rows) / 4, 1) == 5.1
    assert round(sum(r[1] for r in rows) / 4, 1) == 1.6
    assert sum(r[2] for r in rows) / 4 == 3.5
    assert tuple(map(sum, zip(*[r[3] for r in rows]))) == (30, 155)
    assert tuple(map(sum, zip(*[r[4] for r in rows]))) == (8, 100)
    assert tuple(map(sum, zip(*[r[5] for r in rows]))) == (221.5, 307)


def test_judge_bench_stored_summaries_match_the_detail():
    """Сводка, записанная замером, совпадает с пересчётом по кадрам."""
    dumps = load("judge_bench.json.gz")
    for name in JUDGE_TABLE:
        s = dumps[name]
        _e, _b, _n, (ba, b), (gv, g), (po, p) = _judge_row(s)
        assert (s["bad_accepted"], s["bad"], s["good_vetoed"], s["good"], s["pairs_ok"], s["pairs"]) == \
            (ba, b, gv, g, po, p), name


# -------------------------------------------- 4.3 бесплатная зона, живой пул

def _m5_labels_from_raw(m5):
    """Метки M5 из исходных листов — так же, как их сводил скрипт замера:
    перенесённые метки (cN) и лист M5, затем листы M5b и M5c."""
    sheets, raw = m5["sheets"], m5["labels_raw"]
    out = {}
    idx, car, sh = (sheets["m5_final/sheets/index.json"], raw["m5_final_labels.json"],
                    raw["m5_sheet_labels.json"])
    for k, m in idx.items():
        if k.startswith("_"):
            continue
        for t, cid in m.items():
            v = car.get(k, {}).get(t) if t.startswith("c") else sh.get(k, {}).get(t)
            if v is not None:
                out.setdefault(k, {})[cid] = v
    for suf in ("b", "c"):
        for k, m in sheets[f"m5_final/sheets/index_{suf}.json"].items():
            for t, cid in m.items():
                v = raw[f"m5{suf}_sheet_labels.json"].get(k, {}).get(t)
                if v is not None:
                    out.setdefault(k, {}).setdefault(cid, v)
    return out


FREE_TABLE = {
    # рука: (сумма меток победителей, брак-победитель, фраз без победителя,
    #        лучший в пятёрке, годных в пятёрке)
    "v3N": (40, 12, 0, 54, 81), "v3R": (41, 11, 0, 54, 81), "v3R2": (41, 11, 0, 54, 81),
    "v3N2": (32, 17, 1, 48, 65), "v5oN": (35, 16, 0, 51, 85), "v5pN": (33, 16, 0, 52, 91),
    "v5pN2": (31, 16, 1, 47, 79), "hybS": (37, 12, 0, 47, 67),
}


def test_free_zone_table_4_3():
    m5 = load("free_zone.json.gz")["M5"]
    lab = _m5_labels_from_raw(m5)
    assert lab == m5["labels"], "сводные метки не выводятся из листов"
    runs = m5["runs"]
    assert len(runs) == 40 and set(runs) <= set(lab)
    got, wins = {}, {}
    for arm in FREE_TABLE:
        s = bad = none = best = good = 0
        for key, row in runs.items():
            e = row["arms"][arm]
            w = lab[key].get(e["winner"]) if e.get("winner") else None
            wins[(arm, key)] = w
            head = [lab[key].get(h["id"]) for h in e.get("head") or []]
            if w is None:
                none += 1
            else:
                s += w
                bad += w == 0
            best += max([h for h in head if h is not None] or [0])
            good += sum(1 for h in head if h)
        got[arm] = (s, bad, none, best, good)
    assert got == FREE_TABLE

    def pair(a, b):
        r = [(wins[(a, k)], wins[(b, k)]) for k in runs if None not in (wins[(a, k)], wins[(b, k)])]
        return (sum(x > y for x, y in r), sum(x < y for x, y in r), sum(x == y for x, y in r))

    # План Б против плана А в том же процессе; план Б против итогового правила задания.
    assert pair("v3N2", "v3R2") == (5, 13, 21)
    assert pair("v3N2", "v5oN") == (5, 6, 28)


def test_free_zone_decoys_agree_with_earlier_labels():
    """Приманки на листах, размеченных вслепую: 114 из 126 — прежней меткой,
    остальные 12 — на одну ступень."""
    m5 = load("free_zone.json.gz")["M5"]
    sheets, raw = m5["sheets"], m5["labels_raw"]

    def old_labels(skip):
        out = {}
        idx, car, sh = (sheets["m5_final/sheets/index.json"], raw["m5_final_labels.json"],
                        raw["m5_sheet_labels.json"])
        for k, m in idx.items():
            if k.startswith("_"):
                continue
            for t, cid in m.items():
                v = car.get(k, {}).get(t) if t.startswith("c") else sh.get(k, {}).get(t)
                if v is not None:
                    out.setdefault(k, {})[cid] = v
        for suf in ("b", "c"):
            if suf == skip:
                continue
            for k, m in sheets[f"m5_final/sheets/index_{suf}.json"].items():
                for t, cid in m.items():
                    v = raw[f"m5{suf}_sheet_labels.json"].get(k, {}).get(t)
                    if v is not None:
                        out.setdefault(k, {}).setdefault(cid, v)
        return out

    blind = {"b": {f"ep94_hook#{i}" for i in range(9)} | {"gutenberg#0", "gutenberg#1"}
             | {f"deepsea#{i}" for i in range(6)} | {f"psych#{i}" for i in range(6)}, "c": None}
    same = step1 = step2 = n = 0
    for suf in ("b", "c"):
        old, mine = old_labels(suf), raw[f"m5{suf}_sheet_labels.json"]
        for k, m in sheets[f"m5_final/sheets/decoys_{suf}.json"].items():
            if blind[suf] is not None and k not in blind[suf]:
                continue
            for t, cid in m.items():
                a, b = mine.get(k, {}).get(t), old.get(k, {}).get(cid)
                if a is None or b is None:
                    continue
                n += 1
                d = abs(a - b)
                same += d == 0
                step1 += d == 1
                step2 += d == 2
    assert (n, same, step1, step2) == (126, 114, 12, 0)


def test_free_zone_other_ranking_text_does_not_help():
    """Раздел 4.3: на сохранённых признаках 285 кандидатов перебраны тексты,
    с которыми отбор сравнивает пятёрку. Сумма меток выбранных: первый запрос
    121, лучший из запросов 122, главное 114, кадр задания 114, эстетика 68.

    Признаки кандидата, стоящего в двух наборах (M4 и M4c), ОБЪЕДИНЯЮТСЯ:
    скрипт замера затирал их признаками набора, прочитанного последним, и
    числа зависели от порядка файлов на диске (первая запись документа —
    122 / 122 / 115 / 113 / 66). У общего текста значение одно и то же —
    это проверяется здесь же."""
    fr = load("free_rank.json.gz")
    feat = {}
    for name in sorted(k for k in fr if k.endswith("_feat.json")):
        for cid, f in fr[name].items():
            g = feat.setdefault(cid, {"rel": {}, "ok": {}, "aes": f["aes"]})
            for t, v in f["rel"].items():
                assert t not in g["rel"] or abs(g["rel"][t] - v) < 1e-6, (cid, t)
                g["rel"][t] = v
            g["ok"].update(f["ok"])
    items = [it for v in fr["frs/items_all.json"].values() for it in v]

    def b(x):
        return 0 if x is None else int(round(float(x) / 0.02))

    def rel(f, t):
        return f["rel"].get(t) if t else None

    rules = {
        "q1": lambda it, f: (b(rel(f, it["q1"])), f["aes"] or 0),
        "max_q": lambda it, f: (b(max((rel(f, q) or -1) for q in it["queries"])), f["aes"] or 0),
        "core": lambda it, f: (b(rel(f, it["core"])), f["aes"] or 0),
        "focus": lambda it, f: (b(rel(f, it["focus"])), f["aes"] or 0),
        "aesthetic": lambda it, f: (f["aes"] or 0,),
    }
    sums = dict.fromkeys(rules, 0)
    n_cands = set()
    for it in items:
        cands = [(cid, lab) for cid, _path, lab in it["cands"] if cid in feat and lab is not None]
        if len(cands) != len(it["cands"]):
            continue
        n_cands.update(c for c, _l in cands)
        for name, fn in rules.items():
            best = max(cands, key=lambda c: (feat[c[0]]["ok"].get(it["q1"], 1),) + fn(it, feat[c[0]]))
            sums[name] += best[1]
    assert sums == {"q1": 121, "max_q": 122, "core": 114, "focus": 114, "aesthetic": 68}
    # Признаки есть у 285 кандидатов; в полных пятёрках (у каждого — метка и признаки) — 274.
    assert (len(feat), len(n_cands)) == (285, 274)


# ---------------------------------------- 4.5 судья в бесплатной пятёрке

JUDGE5_TABLE = {
    # рука: (сейчас, сетка судьи, брак сейчас, брак с судьёй, лучший в пятёрке, лучше, хуже)
    "v3N": (40, 47.5, 12, 6, 54, 8, 3), "v3N2": (32, 36.7, 17, 13, 45, 6, 2),
    "v5oN": (35, 41.0, 16, 13, 52, 8, 3), "v5pN": (33, 42.0, 16, 8, 52, 10, 2),
    "v5pN2": (31, 37.5, 16, 11, 44, 6, 1),
}


def _judge5_rows():
    """Выбор сетки заново из её оценок: высшая оценка; ничья — нынешний
    победитель, если он среди равных, иначе ожидание по равным."""
    fz = load("free_zone.json.gz")["M5"]
    ev = load("free_judge.json.gz")["eval"]
    lab = fz["labels"]
    out = {}
    for arm, rows in ev.items():
        for r in rows:
            e = fz["runs"][r["key"]]["arms"][arm]
            L = lab[r["key"]]
            cur = L.get(e.get("winner")) if e.get("winner") else None
            assert cur == r["cur"], (arm, r["key"])
            if r["scores"]:
                top = max(r["scores"].values())
                tied = [c for c, v in r["scores"].items() if v == top]
                pick = cur if e.get("winner") in tied else sum(L[c] for c in tied) / len(tied)
            else:
                pick = cur
            if r["judge"] is not None:
                assert abs(pick - r["judge"]) < 1e-9, (arm, r["key"])
            out.setdefault(arm, []).append({"key": r["key"], "cur": cur, "judge": pick, "best": r["best"],
                                            "rel": e.get("relevance"), "cost": r["cost"], "calls": r["calls"]})
    return out


def test_free_zone_judge_table_4_5():
    rows = _judge5_rows()
    got = {}
    for arm, rr in rows.items():
        k = [r for r in rr if r["cur"] is not None]
        got[arm] = (sum(r["cur"] for r in k), round(sum(r["judge"] for r in k), 1),
                    sum(r["cur"] == 0 for r in k), sum(r["judge"] == 0 for r in k), sum(r["best"] for r in k),
                    sum(r["judge"] > r["cur"] for r in k), sum(r["judge"] < r["cur"] for r in k))
    assert got == JUDGE5_TABLE
    tot = [sum(v[i] for v in got.values()) for i in range(7)]
    assert (tot[0], round(tot[1], 1), tot[2], tot[3], tot[4], tot[5], tot[6]) == (171, 204.7, 77, 51, 247, 38, 11)
    assert (sum(r["cost"] for rr in rows.values() for r in rr),
            sum(r["calls"] for rr in rows.values() for r in rr)) == (61599, 194)


def test_free_zone_judge_routing_4_5():
    """Где судья нужен: по сходству нынешнего победителя с первым запросом и
    по типу чтения фразы в задании v5 (план v5o_all из архива)."""
    import stock_query_planner as sqp
    rows = [r for rr in _judge5_rows().values() for r in rr if r["cur"] is not None]
    total = sum(r["judge"] - r["cur"] for r in rows)
    assert (len(rows), round(total, 1)) == (198, 33.7)

    def route(sel):
        return (len(sel), round(sum(r["judge"] - r["cur"] for r in sel), 1),
                sum(r["judge"] > r["cur"] for r in sel), sum(r["judge"] < r["cur"] for r in sel))

    assert route([r for r in rows if r["rel"] < 0.15]) == (148, 35.2, 35, 7)
    assert route([r for r in rows if r["rel"] < 0.142]) == (128, 31.7, 32, 7)
    assert route([r for r in rows if r["rel"] < 0.114]) == (66, 23.7, 22, 4)
    plans = load("planner_bench_plans.json.gz")["v5o_all"]
    runs = load("free_zone.json.gz")["M5"]["runs"]

    def reading(key):
        row = runs[key]
        units = {sqp._unit_key(u.get("text") or ""): u for u in plans[row["ep"]]["plan"]["units"].values()}
        u = units.get(sqp._unit_key(row["text"]))
        if u is None:
            t = row["text"].strip().rstrip(".…")
            u = next(u for u in units.values()
                     if (u.get("text") or "").strip().startswith(t) or t.startswith((u.get("text") or "").strip().rstrip(".…")))
        return u.get("reading")

    by = {}
    for r in rows:
        by.setdefault(reading(r["key"]), []).append(r)
    assert route(by["figurative"]) == (79, 23.5, 20, 3)
    assert route(by["literal"]) == (99, 9.2, 16, 7)
    assert route(by["abstract"]) == (20, 1.0, 2, 1)


@pytest.mark.parametrize("name", sorted(n for n in os.listdir(ARCH) if n.endswith(".json.gz")))
def test_every_archive_file_is_readable(name):
    assert load(name)


def test_readme_lists_every_archive_file():
    readme = open(os.path.join(ARCH, "README.md"), encoding="utf-8").read()
    for p in glob.glob(os.path.join(ARCH, "*.json.gz")):
        assert os.path.basename(p) in readme, os.path.basename(p)


# ------------------------------------ 4.6 отсев брака DeepSeek-ом по тексту

GC_ARMS = ["v3N", "v3N2", "v5oN", "v5pN", "v5pN2"]   # руки, у которых есть и оценки сетки судьи
GC_ARCHIVE = {"met", "chicago", "cleveland", "commons", "openverse", "europeana"}


def _gc_norm(t):
    import unicodedata
    t = unicodedata.normalize("NFKD", t or "")
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    return " " + re.sub(r"[^a-z0-9]+", " ", t) + " "


def _gc_has_accept(text, terms):
    """Слово из списка DeepSeek есть в подписи: сочетание — подстрокой, слово —
    началом слова подписи (множественное число снимается)."""
    n = _gc_norm(text)
    for term in terms:
        w = _gc_norm(term).strip()
        if not w:
            continue
        if " " in w:
            if f" {w}" in n:
                return True
        elif re.search(r" " + re.escape(w[:-1] if len(w) > 4 and w.endswith("s") else w), n):
            return True
    return False


def _gc_has_reject(text, terms):
    n = _gc_norm(text)
    return any(re.search(r" " + re.escape(_gc_norm(t).strip()) + r"(?:s|es)? ", n)
               for t in terms if _gc_norm(t).strip())


def _gc_setup():
    gc = load("garbage_cut.json.gz")
    m5 = load("free_zone.json.gz")["M5"]
    judge = {(r["key"], arm): r["scores"] for arm, rr in load("free_judge.json.gz")["eval"].items() for r in rr}
    texts = {(it["key"], it["arm"]): it["texts"] for its in gc["items"].values() for it in its}
    feat, lab = gc["features"], m5["labels"]
    rows = []
    for key, row in m5["runs"].items():
        for arm in GC_ARMS:
            e = row["arms"][arm]
            head = e["head"]
            labs = [lab[key].get(h["id"]) for h in head]
            if any(y is None for y in labs) or any(h["id"] not in feat for h in head):
                continue
            rows.append((key, arm, e, head, labs))
    return gc, feat, judge, texts, rows


def _gc_rule(feat, head, i, q1):
    """Правило рендера бесплатной зоны: гейт, корзина 0.02 сходства с первым запросом, эстетика."""
    f = feat[head[i]["id"]]
    x = f["rel"].get(q1)
    return (f["ok"].get(q1, 1), 0 if x is None else int(round(float(x) / 0.02)), f["aes"] or 0)


def _gc_eval(feat, rows, marker):
    marked, s0, s1, b0, b1, better, worse = [0, 0, 0], 0, 0, 0, 0, 0, 0
    for key, arm, e, head, labs in rows:
        q1 = e["queries"][0]
        m = marker(key, arm, e, head)
        for i, y in enumerate(labs):
            marked[y] += i in m
        base = max(range(5), key=lambda i: _gc_rule(feat, head, i, q1))
        keep = [i for i in range(5) if i not in m] or list(range(5))
        new = max(keep, key=lambda i: _gc_rule(feat, head, i, q1))
        s0 += labs[base]; s1 += labs[new]; b0 += labs[base] == 0; b1 += labs[new] == 0
        better += labs[new] > labs[base]; worse += labs[new] < labs[base]
    return tuple(marked), (s0, s1), (b0, b1), (better, worse)


def test_text_garbage_cut_does_not_lift_the_free_five_4_6():
    """Раздел 4.6: три способа отсечь брак текстовой моделью в бесплатной
    пятёрке — слова подписи в плане, проверка подписей пятёрки, ловушки
    задания как контрастивное вето. Ловят до 80% брачных кандидатов, но
    выбор правилом рендера не поднимают: помечают и годные, а брак, который
    побеждает, по подписи неотличим. Глаз (сетка судьи) на той же базе
    поднимает выбор. База — 190 пятёрок пяти рук с метками и признаками."""
    gc, feat, judge, texts, rows = _gc_setup()
    assert len(rows) == 190
    lex = gc["lexicon"]
    got = {
        "lexicon_archive_no_word": _gc_eval(feat, rows, lambda k, a, e, h: {
            i for i, x in enumerate(h) if x["channel"] in GC_ARCHIVE and not _gc_has_accept(x["alt"], lex[k]["accept"])}),
        "lexicon_forbidden_word": _gc_eval(feat, rows, lambda k, a, e, h: {
            i for i, x in enumerate(h) if _gc_has_reject(x["alt"], lex[k]["reject"])}),
    }
    for tag in ("r1", "r0"):
        ans = gc["caption_check"][tag]
        got[f"caption_check_{tag}"] = _gc_eval(feat, rows, lambda k, a, e, h, ans=ans: {
            i - 1 for i in ans[k + "|" + "|".join(x["id"] for x in h)]["drop"] if 1 <= i <= 5})

    def traps(target, margin):
        def m(k, a, e, h):
            t = texts[(k, a)]
            tgt = {"q1": t[0], "core": t[1], "shot": t[2]}[target]
            out = set()
            for i, x in enumerate(h):
                rel = feat[x["id"]]["rel"]
                negs = [rel[n] for n in t[3:] if rel.get(n) is not None]
                if rel.get(tgt) is not None and negs and rel[tgt] - max(negs) < margin:
                    out.add(i)
            return out
        return m

    got["traps_q1_-0.06"] = _gc_eval(feat, rows, traps("q1", -0.06))
    got["traps_core_-0.03"] = _gc_eval(feat, rows, traps("core", -0.03))
    assert got == {
        # (помечено брак/проходных/годных), (сумма до, после), (брак до, после), (лучше, хуже)
        "lexicon_archive_no_word": ((238, 59, 9), (178, 168), (66, 76), (2, 12)),
        "lexicon_forbidden_word": ((6, 3, 3), (178, 177), (66, 66), (1, 2)),
        "caption_check_r1": ((464, 140, 40), (178, 180), (66, 64), (16, 16)),
        "caption_check_r0": ((454, 127, 29), (178, 185), (66, 64), (14, 11)),
        "traps_q1_-0.06": ((50, 9, 1), (178, 178), (66, 66), (1, 1)),
        "traps_core_-0.03": ((114, 34, 10), (178, 180), (66, 66), (6, 4)),
    }

    # Сетка судьи на той же базе: высшая оценка, ничья — порядок правила рендера.
    s0 = s1 = b0 = b1 = better = worse = 0
    for key, arm, e, head, labs in rows:
        q1 = e["queries"][0]
        base = max(range(5), key=lambda i: _gc_rule(feat, head, i, q1))
        sc = judge.get((key, arm))
        new = base
        if sc:
            vals = [sc.get(x["id"], -1) for x in head]
            tied = [i for i, v in enumerate(vals) if v == max(vals)]
            new = max(tied, key=lambda i: _gc_rule(feat, head, i, q1))
        s0 += labs[base]; s1 += labs[new]; b0 += labs[base] == 0; b1 += labs[new] == 0
        better += labs[new] > labs[base]; worse += labs[new] < labs[base]
    assert (s0, s1, b0, b1, better, worse) == (178, 199, 66, 52, 26, 12)


def test_caption_check_gain_is_not_consistent_across_plans_4_6():
    """Единственный способ с плюсом по сумме (проверка подписей без
    рассуждения, 178 -> 185) держится на одном плане — плане Б v3 с
    негодными запросами; в двух планах из пяти он опускает выбор. Сетка
    судьи поднимает выбор в каждом плане."""
    gc, feat, judge, texts, rows = _gc_setup()
    ans = gc["caption_check"]["r0"]
    per = {}
    for arm in GC_ARMS:
        rr = [r for r in rows if r[1] == arm]
        _m, (s0, s1), _b, _bw = _gc_eval(feat, rr, lambda k, a, e, h: {
            i - 1 for i in ans[k + "|" + "|".join(x["id"] for x in h)]["drop"] if 1 <= i <= 5})
        per[arm] = s1 - s0
    assert per == {"v3N": -2, "v3N2": 6, "v5oN": -1, "v5pN": 4, "v5pN2": 0}
