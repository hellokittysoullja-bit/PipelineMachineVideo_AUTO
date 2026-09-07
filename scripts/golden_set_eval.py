#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Метрика качества подбора: прогон НАСТОЯЩИХ гейтов по золотому набору.

Зачем это существует. У пайплайна десятки цифр КОРРЕКТНОСТИ (сколько слотов
заполнено, сколько тестов зелёных, какое расстояние между хэшами дублей) и ни
одной цифры КАЧЕСТВА. Единственная, которая когда-либо звучала — «4 годных
кадра из 15» — родилась из того, что человек посмотрел контактный лист и
сказал это вслух, а не из системы. Значит любую правку скоринга, гвардов или
корпуса можно выполнить целиком и не суметь ответить числом, стало ли лучше.

Что тут меряется. Золотой набор (tests/fixtures/golden_set/) — это кадры,
которые РЕАЛЬНО ушли в опубликованный эпизод, с вердиктами, проставленными
глазами по Шагу 7.5 (CLAUDE.md). Скрипт прогоняет по ним те же самые функции,
что принимают решение в проде — is_relevant_candidate(),
visual_domain_guard_violation(), clip_relevance() — не их копию и не
упрощённую модель. Ответ: сколько брака гейты ПРОПУСКАЮТ и сколько годного
ложно ОТКЛОНЯЮТ.

Что тут НЕ меряется, честно:

* Это метрика ГЕЙТОВ, а не всего подбора. Гейт отвечает «допустим ли этот
  кандидат», а ранжирование («какой из допустимых лучше») здесь не
  проверяется — для него нужен пул кандидатов на слот, а в наборе по одному
  победителю на слот. Пропуск брака гейтом — необходимая, но не достаточная
  причина брака в ролике.
* Ось резкости на этом наборе не является приёмкой: кадры сохранены в 768 px
  по ширине, а PHOTO_SHARPNESS_REJECT калиброван на полном разрешении.
* Кадры видео-слотов взяты с 1-й секунды и не обязаны совпадать с тем, что
  видно в ролике в момент фразы.

Fail-open — главная ловушка этого измерения. clip_relevance() возвращает None
при недоступной модели, а None во всех гейтах читается как «пропустить».
Тогда скрипт напечатал бы «пропуск брака 100%» и это выглядело бы как
измерение, хотя на деле не работала модель. Поэтому перед подсчётом идёт
канарейка: если CLIP не отвечает — скрипт НЕ печатает ни одной цифры и
завершается с ошибкой.

Использование:
    python scripts/golden_set_eval.py [--out docs/quality/golden_set_report.json]
                                      [--baseline docs/quality/golden_set_baseline.json]

Код возврата: 0 — метрика посчитана; 1 — посчитать невозможно (нет модели,
нет набора); 2 — посчитана и есть регрессия против --baseline.
"""
import argparse
import json
import os
import sys
from collections import Counter, OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_DIR = os.path.join(REPO, "tests", "fixtures", "golden_set")

# Вердикты, которые для ПРОФЕССИОНАЛЬНОГО уровня считаются непройденными.
# Прямое требование владельца канала (07.09) после просмотра контактного
# листа первых 16 слотов: «остальные кадры многие тоже подбираются криво» —
# то есть планка не «нет явного брака», а «кадр работает на смысл». Поэтому
# у строгой метрики tolerable идёт в минус наравне с reject, а мягкая
# (совместимая с прежним разговором про «4 годных из 15») считается отдельно.
STRICT_FAIL_VERDICTS = ("reject", "tolerable")

# Причины отказа, которые описывают культурный/эпохальный анахронизм —
# именно тот класс, ради которого написаны VISUAL_DOMAIN_GUARDS и
# CONTENT_ALT_BLOCKLIST (CLAUDE.md, ЧАСТЬ 14).
ANACHRONISM_REASONS = ("non_european", "modern_intrusion")


def load_manifest(path=None):
    path = path or os.path.join(GOLDEN_DIR, "manifest.json")
    if not os.path.exists(path):
        raise SystemExit(f"нет золотого набора: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _canary(ps, items):
    """Доказать, что CLIP реально отвечает, ДО того как считать проценты.

    Без этого недоступная модель даёт None на каждом кадре, None читается
    гейтами как «пропустить», и отчёт показал бы идеальный пропуск брака —
    ровно та ловушка, о которой предупреждает CLAUDE.md («пустой
    автоматический отчёт — не то же самое, что проблема решена»).
    """
    for it in items:
        img = os.path.join(GOLDEN_DIR, it["image"])
        if not os.path.exists(img):
            continue
        rel = ps.clip_relevance(img, it["query"])
        if rel is not None:
            return rel
    return None


def evaluate(items, *, with_sentence=False):
    import pipeline_smart as ps

    canary = _canary(ps, items)
    if canary is None:
        raise SystemExit(
            "CLIP не отвечает (clip_relevance вернул None на каждом кадре набора).\n"
            "Метрика НЕ посчитана: при недоступной модели все гейты fail-open, "
            "и любые проценты были бы измерением тишины, а не качества.\n"
            "Проверь torch/transformers и CLIP_RELEVANCE в окружении."
        )

    sentence_fn = None
    if with_sentence:
        try:
            from visual_director import sentence_relevance as sentence_fn  # noqa: F401
        except Exception:
            sentence_fn = None

    rows = []
    for it in items:
        img = os.path.join(GOLDEN_DIR, it["image"])
        query = it["query"]
        relevance = ps.clip_relevance(img, query)
        risky = ps.is_risky_query(query)
        anchor = ps.clip_relevance(img, ps.NEGATIVE_ANCHOR_PROMPT) if risky else None
        guard_hit, guard_name = ps.visual_domain_guard_violation(img, query)
        passed = ps.is_relevant_candidate(img, query, relevance)
        row = OrderedDict(
            id=it["id"], index=it["index"], section=it["section"],
            verdict=it["verdict"], reject_reason=it.get("reject_reason"),
            query=query, text=it["text"],
            relevance=None if relevance is None else round(relevance, 4),
            threshold=ps.CLIP_RELEVANCE_THRESHOLD,
            risky_query=risky,
            anchor_margin=(None if (anchor is None or relevance is None)
                           else round(relevance - anchor, 4)),
            anchor_margin_threshold=ps.RISKY_QUERY_MARGIN if risky else None,
            domain_guard=guard_name,
            gate_passed=bool(passed),
        )
        if sentence_fn is not None:
            try:
                row["sentence_relevance"] = sentence_fn(img, it["text"])
            except Exception:
                row["sentence_relevance"] = None
        rows.append(row)
    return rows, canary


def summarize(rows):
    def frac(num, den):
        return None if den == 0 else round(num / den, 4)

    rejects = [r for r in rows if r["verdict"] == "reject"]
    goods = [r for r in rows if r["verdict"] == "good"]
    anach = [r for r in rejects if r["reject_reason"] in ANACHRONISM_REASONS]
    strict_fails = [r for r in rows if r["verdict"] in STRICT_FAIL_VERDICTS]
    hook = [r for r in rows if r["section"] == "HOOK"]

    return OrderedDict(
        n_items=len(rows),
        n_good=len(goods),
        n_tolerable=sum(1 for r in rows if r["verdict"] == "tolerable"),
        n_reject=len(rejects),
        # Главные три числа.
        reject_leak=frac(sum(1 for r in rejects if r["gate_passed"]), len(rejects)),
        good_false_reject=frac(sum(1 for r in goods if not r["gate_passed"]), len(goods)),
        anachronism_leak=frac(sum(1 for r in anach if r["gate_passed"]), len(anach)),
        # Строгая планка (tolerable тоже минус) — требование владельца канала.
        strict_leak=frac(sum(1 for r in strict_fails if r["gate_passed"]), len(strict_fails)),
        # Годность самого набора, без гейтов: чем ролик был на самом деле.
        episode_good_rate=frac(len(goods), len(rows)),
        hook_good_rate=frac(sum(1 for r in hook if r["verdict"] == "good"), len(hook)),
        by_reason=dict(Counter(
            r["reject_reason"] for r in rejects if r["reject_reason"])),
        leaked_by_reason=dict(Counter(
            r["reject_reason"] for r in rejects
            if r["gate_passed"] and r["reject_reason"])),
        domain_guard_hits=sum(1 for r in rows if r["domain_guard"]),
    )


CORPUS_SAMPLE = os.path.join(REPO, "tests", "fixtures", "pexels_slugs_sample.json")


def evaluate_corpus(path=None):
    """Вторая ось: сколько брака отсекается ДО скачивания, по тексту выдачи.

    Зачем отдельно от гейтов. Метрика выше меряет пиксельные гейты на уже
    скачанных кадрах — она по построению слепа ко всему, что происходит
    раньше, на этапе выдачи поиска. А самый дешёвый рычаг качества лежит
    именно там: у видео-объектов Pexels нет ни alt, ни тегов, но есть
    человекочитаемый слаг в url, и по нему видно и «historical reenactment
    event», и «two fencers at their fighting position». Правка жанрового
    фильтра не двигает ни одного числа выше — и без этой функции у неё не
    было бы числа вообще.

    Считается на ЗАМОРОЖЕННОЙ живой выдаче (tests/fixtures/
    pexels_slugs_sample.json) — без сети и без ключа, воспроизводимо.
    """
    import pipeline_smart as ps

    with open(path or CORPUS_SAMPLE, encoding="utf-8") as f:
        items = json.load(f)["items"]
    by_query = {}
    for it in items:
        by_query.setdefault((it["query"], it["kind"]), []).append(it)

    removed = starved = 0
    per_kind = {"photo": [0, 0], "video": [0, 0]}
    for (_q, kind), group in by_query.items():
        kept = ps.filter_alt_blocklist(list(group))
        blocked = [g for g in group
                   if any(t in ps.pexels_candidate_text(g)
                          for t in ps.CONTENT_ALT_BLOCKLIST)]
        # Откат «отфильтровалось всё» возвращает исходный список: фильтр
        # формально отработал, а эффекта ноль. Это надо видеть отдельно.
        if blocked and len(kept) == len(group):
            starved += 1
        removed += len(group) - len(kept)
        per_kind.setdefault(kind, [0, 0])
        per_kind[kind][0] += len(group) - len(kept)
        per_kind[kind][1] += len(group)
    return OrderedDict(
        n_candidates=len(items),
        blocked_share=round(removed / len(items), 4) if items else None,
        blocked_share_photo=(round(per_kind["photo"][0] / per_kind["photo"][1], 4)
                             if per_kind["photo"][1] else None),
        blocked_share_video=(round(per_kind["video"][0] / per_kind["video"][1], 4)
                             if per_kind["video"][1] else None),
        starved_queries=starved,
    )


# Направление, в котором метрика УЛУЧШАЕТСЯ: -1 — меньше лучше, +1 — больше лучше.
METRIC_DIRECTION = {
    "reject_leak": -1, "good_false_reject": -1, "anachronism_leak": -1,
    "strict_leak": -1, "episode_good_rate": +1, "hook_good_rate": +1,
    # Больше отсеянного брака — лучше, но только пока не появляются запросы,
    # где отсеивается ВСЁ: там срабатывает откат и ужесточение бессмысленно.
    "blocked_share": +1, "starved_queries": -1,
}
REGRESSION_EPS = 1e-9


def compare_baseline(summary, baseline):
    """Сравнение с зафиксированной базовой линией — по каждой оси отдельно."""
    out = OrderedDict()
    regressed = []
    for key, direction in METRIC_DIRECTION.items():
        now, was = summary.get(key), baseline.get(key)
        if now is None or was is None:
            continue
        delta = round(now - was, 4)
        worse = (delta > REGRESSION_EPS) if direction < 0 else (delta < -REGRESSION_EPS)
        out[key] = {"baseline": was, "now": now, "delta": delta, "regressed": worse}
        if worse:
            regressed.append(key)
    return out, regressed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out", default=os.path.join(REPO, "docs", "quality",
                                                  "golden_set_report.json"))
    ap.add_argument("--baseline", default=None,
                    help="JSON с прошлым summary: падать кодом 2 при регрессии")
    ap.add_argument("--with-sentence", action="store_true",
                    help="считать ещё и sentence_relevance (медленно: ~3 с на кадр)")
    args = ap.parse_args(argv)

    meta = load_manifest(args.manifest)
    items = meta["items"]
    rows, canary = evaluate(items, with_sentence=args.with_sentence)
    summary = summarize(rows)
    summary.update(evaluate_corpus())

    report = OrderedDict(
        schema_version=1,
        manifest_source=meta.get("source"),
        canary_relevance=round(canary, 4),
        summary=summary,
        items=rows,
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)

    print(f"Золотой набор: {summary['n_items']} кадров "
          f"({summary['n_good']} годных / {summary['n_tolerable']} терпимых / "
          f"{summary['n_reject']} брака)")
    print(f"  пропуск брака гейтами      : {summary['reject_leak']}")
    print(f"  из них анахронизмы         : {summary['anachronism_leak']}")
    print(f"  ложный отказ годным        : {summary['good_false_reject']}")
    print(f"  строгий пропуск (+терпимые): {summary['strict_leak']}")
    print(f"  годность эпизода / хука    : {summary['episode_good_rate']} / "
          f"{summary['hook_good_rate']}")
    if summary["leaked_by_reason"]:
        print(f"  утечки по причинам         : {summary['leaked_by_reason']}")
    print(f"Живая выдача ({summary['n_candidates']} кандидатов, заморожена):")
    print(f"  отсеяно по тексту до скачивания: {summary['blocked_share']} "
          f"(фото {summary['blocked_share_photo']} / видео {summary['blocked_share_video']})")
    print(f"  запросов, где отсеялось всё    : {summary['starved_queries']}")
    print(f"Отчёт: {os.path.relpath(args.out, REPO)}")

    if args.baseline:
        with open(args.baseline, encoding="utf-8") as f:
            base = json.load(f)
        base = base.get("summary", base)
        diff, regressed = compare_baseline(summary, base)
        report["baseline_diff"] = diff
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
        for key, d in diff.items():
            mark = "ХУЖЕ" if d["regressed"] else "    "
            print(f"  {mark} {key}: {d['baseline']} -> {d['now']} ({d['delta']:+})")
        if regressed:
            print("РЕГРЕССИЯ по осям: " + ", ".join(regressed))
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
