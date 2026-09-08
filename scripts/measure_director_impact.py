# -*- coding: utf-8 -*-
"""Решающий замер Контура A: реальный сток, реальные гейты, сравнение победителей.

Чем отличается от analyze_director_benchmark.py. Тот меряет КАЧЕСТВО
ТЕКСТА брифа (английский ли, разнообразны ли запросы, угадана ли ловушка) —
это косвенные признаки. Здесь меряется то, ради чего всё делалось: КАКОЙ
КАДР В ИТОГЕ ПРИДЁТ. Запросы уходят в живой Pexels, кандидаты реально
скачиваются и проходят те же самые гейты, что работают в проде
(`is_relevant_candidate`, домен-гвард, контрастивное вето, резкость),
победитель выбирается тем же `_score_and_pick`.

ТРИ ПУЛА, а не два — потому что в проде запросы ДОБАВЛЯЮТСЯ к авторскому,
а не заменяют его, и у этих двух вариантов разные риски:

  baseline — только исходный запрос слота. Это то, что есть сегодня.
  director — только запросы режиссёра. Показывает потолок нового
             механизма, но в проде так не будет.
  union    — исходный + запросы режиссёра. РОВНО то, что произойдёт в
             проде, и единственный пул, по которому честно решать,
             включать фичу или нет.

Зачем нужен именно union, а не director. Добавление кандидатов меняет,
кто победит: новый кандидат может вытеснить хорошего старого. Пул
`director` этого не покажет (там старого просто нет), а `union` покажет.
Без этого замера утверждение «только upgrade» было бы обещанием, а не
измерением.

Запуск:
    .venv/bin/python scripts/measure_director_impact.py docs/quality/director_bench_v2.json
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

# pipeline_smart читает sys.argv[1] как папку эпизода ПРЯМО НА ИМПОРТЕ,
# поэтому argv приходится подменить. Настоящие аргументы сохраняем до
# подмены и возвращаем сразу после — иначе argparse ниже разбирал бы
# подставленный путь вместо того, что набрал пользователь (реально
# пойманная на себе ошибка: IsADirectoryError на /tmp).
_REAL_ARGV = list(sys.argv)
sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]
import pipeline_smart as ps  # noqa: E402
sys.argv = _REAL_ARGV

# Сколько кандидатов на запрос реально скачивать и оценивать. Не больше,
# чем реальный пайплайн смотрит на слот (BASE_MIN_POOL=4 / DIRECTOR_MIN_
# POOL=8) — иначе замер мерил бы более щедрый режим, чем прод.
PER_QUERY = 4


def _download(url, dest, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": ps.UA})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def evaluate_pool(queries, own_query, work_dir, seen_ids):
    """Скачивает кандидатов по queries и прогоняет РЕАЛЬНЫЕ гейты прода.

    Возвращает список кандидатов с полями решения. `own_query` — запрос,
    против которого гейт сверяет релевантность (в проде это запрос СЛОТА,
    а не тот, из которого кандидат приплыл; см. разбор query-swap бага в
    pipeline_smart)."""
    cands = []
    for q in queries:
        try:
            found = ps.filter_alt_blocklist(ps._pexels_search_photos(
                ps.disambiguate_search_query(q)))
        except Exception as e:
            print(f"      поиск «{q}» упал: {type(e).__name__}")
            continue
        taken = 0
        for p in found:
            if taken >= PER_QUERY:
                break
            pid = p.get("id")
            if pid in seen_ids:
                continue
            src = (p.get("src") or {}).get("large2x")
            if not src:
                continue
            dest = os.path.join(work_dir, f"{pid}.jpg")
            if not os.path.exists(dest):
                try:
                    _download(src, dest)
                except Exception:
                    continue
            seen_ids.add(pid)
            taken += 1
            cands.append({"id": pid, "path": dest, "origin_query": q,
                           "alt": p.get("alt"), "url": p.get("url")})

    for c in cands:
        rel = ps.clip_relevance(c["path"], own_query)
        c["relevance"] = round(rel, 4) if rel is not None else None
        # Тот же составной гейт, что в проде: порог + risky-margin +
        # домен-гвард (анахронизм) + контрастивное вето.
        c["gate_passed"] = bool(ps.is_relevant_candidate(c["path"], own_query, relevance=rel))
        vetoed, trap = ps.negative_anchor_violation(c["path"], own_query)
        c["negative_veto"] = trap if vetoed else None
        guard, gname = ps.visual_domain_guard_violation(c["path"], own_query)
        c["domain_guard"] = gname if guard else None
        try:
            c["aesthetic"] = round(float(ps.aesthetic_score(c["path"]) or 0), 3)
        except Exception:
            c["aesthetic"] = 0.0
        try:
            c["sharp_ok"] = bool(ps.image_sharpness_score(c["path"]) >= ps.PHOTO_SHARPNESS_REJECT)
        except Exception:
            c["sharp_ok"] = True
    return cands


def pick_winner(cands):
    """Тот же лексикографический выбор, что у прода (_score_and_pick):
    гейты идут ПЕРЕД эстетикой, а не складываются с ней в сумму."""
    if not cands:
        return None
    info = [{"path": c["path"], "p": {"_origin_query": c["origin_query"]},
              "is_dup_free": 1, "size_ok": 1,
              "is_relevant": 1 if c["gate_passed"] else 0,
              "sharp_ok": 1 if c["sharp_ok"] else 0,
              "aesthetic_val": c["aesthetic"], "luma_score": 0, "min_d": 0}
             for c in cands]
    base, _ = ps._score_and_pick(info, None)
    if base is None:
        return None
    return next(c for c in cands if c["path"] == base["path"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_json", help="результат shot_brief_planner --benchmark")
    ap.add_argument("--out", default=os.path.join(REPO, "docs", "quality",
                                                   "director_impact.json"))
    ap.add_argument("--work", default="/tmp/director_impact")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not ps.PEXELS_API_KEY:
        print("Нет PEXELS_API_KEY — замер невозможен")
        return 1

    with open(args.run_json, encoding="utf-8") as f:
        run = json.load(f)
    cases = [c for c in run["cases"] if c["valid"]]
    if args.limit:
        cases = cases[:args.limit]
    os.makedirs(args.work, exist_ok=True)

    print(f"Модель: {run.get('model')}, промпт v{run.get('prompt_version')}")
    print(f"Случаев с валидным брифом: {len(cases)}\n")

    results = []
    t0 = time.time()
    for i, c in enumerate(cases, 1):
        own = c["baseline_query"]
        wd = os.path.join(args.work, c["id"])
        os.makedirs(wd, exist_ok=True)
        dq = c["brief"]["queries_en"]
        print(f"[{i}/{len(cases)}] {c['id']} ({c['difficulty_class']}, "
              f"было: {c['baseline_verdict']})")
        print(f"    базовый запрос : {own}")
        print(f"    режиссёр       : {dq}")

        seen = set()
        base_c = evaluate_pool([own], own, wd, seen)
        dir_c = evaluate_pool(dq, own, wd, seen)
        # union — те же файлы, без повторного скачивания: кандидат,
        # найденный обоими путями, физически один и тот же.
        union_c = base_c + dir_c

        row = {"id": c["id"], "difficulty_class": c["difficulty_class"],
                "baseline_verdict": c["baseline_verdict"],
                "baseline_query": own, "director_queries": dq}
        for name, pool in (("baseline", base_c), ("director", dir_c), ("union", union_c)):
            w = pick_winner(pool)
            row[name] = {
                "n_candidates": len(pool),
                "n_gate_passed": sum(1 for x in pool if x["gate_passed"]),
                "winner_id": w["id"] if w else None,
                "winner_path": w["path"] if w else None,
                "winner_query": w["origin_query"] if w else None,
                "winner_gate_passed": w["gate_passed"] if w else None,
                "winner_relevance": w["relevance"] if w else None,
                "winner_veto": w["negative_veto"] if w else None,
                "winner_guard": w["domain_guard"] if w else None,
            }
            print(f"    {name:<9}: {len(pool):2d} кандидатов, "
                  f"{row[name]['n_gate_passed']:2d} прошли гейт, "
                  f"победитель {'ПРОШЁЛ' if row[name]['winner_gate_passed'] else 'НЕ прошёл'} "
                  f"(rel={row[name]['winner_relevance']})")
        results.append(row)

    summary = {}
    for name in ("baseline", "director", "union"):
        passed = sum(1 for r in results if r[name]["winner_gate_passed"])
        pool_rate = sum(r[name]["n_gate_passed"] for r in results)
        pool_total = sum(r[name]["n_candidates"] for r in results)
        summary[name] = {
            "winner_passes_gates": passed,
            "winner_passes_rate": round(passed / max(1, len(results)), 4),
            "pool_gate_pass_rate": round(pool_rate / max(1, pool_total), 4),
            "total_candidates": pool_total,
        }

    out = {"model": run.get("model"), "prompt_version": run.get("prompt_version"),
            "n_cases": len(results), "seconds": round(time.time() - t0, 1),
            "summary": summary, "cases": results}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    os.replace(tmp, args.out)

    print("\n=== ИТОГ ===")
    print(f"{'пул':<10} {'победитель прошёл гейты':<26} {'доля годных в пуле':<20}")
    for name in ("baseline", "director", "union"):
        s = summary[name]
        print(f"{name:<10} {s['winner_passes_gates']}/{len(results)} "
              f"({s['winner_passes_rate']:.0%})".ljust(37) +
              f"{s['pool_gate_pass_rate']:.0%} из {s['total_candidates']}")
    print(f"\nОтчёт: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
