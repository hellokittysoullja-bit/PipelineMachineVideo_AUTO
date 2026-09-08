# -*- coding: utf-8 -*-
"""Разбор прогона локального режиссёра по осям, которые можно ПОСЧИТАТЬ.

Зачем отдельно от самого прогона: «посмотрел глазами, вроде лучше» — это
ровно тот способ оценки, который в этом проекте уже один раз дал ложную
уверенность. Здесь каждая ось — число, и по каждой видно направление
(больше — лучше или хуже), чтобы сравнение версий промпта было честным,
а не подгонкой под запомнившийся пример.

Оси и почему именно они:

1. valid_json_rate — доля ответов, переживших строгую валидацию. Ниже
   единицы — это не «модель глупая», а «столько-то слотов молча остались
   бы на прежнем поведении».
2. reading_accuracy — САМАЯ ВАЖНАЯ ось. Отличить буквальное от
   фигурального — то, чего эмбеддинг не умеет в принципе; если модель
   этого тоже не умеет, весь Контур A теряет смысл. Считается только по
   классам с однозначным ожидаемым ответом (literal -> literal,
   figurative/metaphor -> figurative); compositional/fragment/
   anachronism_risk сознательно НЕ засчитываются ни в плюс, ни в минус —
   у них правильный ответ неоднозначен, и притягивать их к метрике
   значило бы мерить собственную разметку, а не модель.
3. self_harm_rate — доля слотов, где модель сама предложила запрос с
   термином из блоклиста канала (katana/samurai/...). Прямой
   членовредительский случай: система принесёт ровно то, что попросили,
   а свои же гейты это отбракуют. У авторских запросов такой случай
   реально был (`katana sword` в сценарии про Европу).
4. cross_slot_distinct — доля УНИКАЛЬНЫХ запросов среди всех слотов.
   Базовая линия: один запрос `greatsword warrior fight` обслуживал
   четыре слота, и все четыре дали брак. Чем ближе к 1, тем меньше
   слотов делят общую судьбу.
5. intra_slot_diversity — насколько три запроса ОДНОГО слота отличаются
   друг от друга (1 - средняя доля общих слов). Три синонима — это один
   запрос, записанный трижды: пул шире не становится.
6. trap_hit_rate — по слотам, которые РЕАЛЬНО провалились в
   опубликованном эпизоде: назвала ли модель в must_not_contain ту самую
   ловушку, в которую система попала. Единственная ось, которая
   проверяет понимание против ФАКТА, а не против моей разметки.

Запуск: .venv/bin/python scripts/analyze_director_benchmark.py <run.json> [...]
"""
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

# Ожидаемое чтение — только там, где ответ однозначен (см. докстринг, ось 2).
EXPECTED_READING = {
    "literal": "literal",
    "figurative": "figurative",
    "metaphor": "figurative",
}

# Реальная ловушка, в которую система попала на этом слоте в
# опубликованном эпизоде (из вердиктов золотого набора). Слова — то, что
# должно было бы стоять в must_not_contain, чтобы этот брак не прошёл.
ACTUAL_TRAPS = {
    "ep01_000": ["kitchen", "milk", "dairy", "cereal", "domestic", "modern"],
    "ep01_005": ["baby", "infant", "bottle", "milk", "child", "modern"],
    "ep01_008": ["modern", "spectator", "crowd", "tourist", "contemporary"],
    "ep01_068": ["fencing", "sport", "mask", "modern", "competition"],
    "ep01_002": ["asian", "japanese", "chinese", "katana", "samurai", "eastern", "korean"],
    "ep01_006": ["asian", "japanese", "chinese", "katana", "samurai", "eastern", "korean"],
    "ep01_122": ["asian", "japanese", "chinese", "katana", "samurai", "eastern", "korean"],
    "ep01_140": ["blur", "unclear", "abstract", "modern", "photograph"],
    "ep01_084": ["modern", "souvenir", "decorative", "replica", "shop"],
    "ep01_009": ["modern", "spectator", "crowd", "tourist", "contemporary"],
}

_WORD = re.compile(r"[a-z]+")


def _words(s):
    return set(_WORD.findall((s or "").lower()))


def _blocklist():
    try:
        with open(os.path.join(REPO, "channel_profile.json"), encoding="utf-8") as f:
            return [t.lower() for t in json.load(f).get("content_alt_blocklist", [])]
    except Exception:
        return []


def analyze(run_path):
    with open(run_path, encoding="utf-8") as f:
        run = json.load(f)
    cases = run["cases"]
    blocklist = _blocklist()

    valid = [c for c in cases if c["valid"]]

    # 2. reading_accuracy
    graded, correct = 0, 0
    reading_detail = []
    for c in valid:
        exp = EXPECTED_READING.get(c["difficulty_class"])
        if exp is None:
            continue
        graded += 1
        got = c["brief"]["reading"]
        ok = (got == exp)
        correct += ok
        reading_detail.append({"id": c["id"], "class": c["difficulty_class"],
                                "expected": exp, "got": got, "ok": ok})

    # 3. self_harm_rate
    self_harm = []
    for c in valid:
        hits = set()
        for q in c["brief"]["queries_en"]:
            ql = q.lower()
            for term in blocklist:
                if term in ql:
                    hits.add(term)
        if hits:
            self_harm.append({"id": c["id"], "terms": sorted(hits),
                               "queries": c["brief"]["queries_en"]})

    # 4. cross_slot_distinct
    all_q = [q.lower() for c in valid for q in c["brief"]["queries_en"]]
    distinct = len(set(all_q)) / len(all_q) if all_q else 0.0
    # Сколько слотов делят хотя бы один общий запрос — прямой аналог
    # базовой болезни (один запрос на четыре слота).
    q_to_slots = {}
    for c in valid:
        for q in c["brief"]["queries_en"]:
            q_to_slots.setdefault(q.lower(), set()).add(c["id"])
    shared = {q: sorted(s) for q, s in q_to_slots.items() if len(s) > 1}

    # 5. intra_slot_diversity
    div = []
    for c in valid:
        qs = c["brief"]["queries_en"]
        if len(qs) < 2:
            continue
        pairs, total = 0, 0.0
        for i in range(len(qs)):
            for j in range(i + 1, len(qs)):
                a, b = _words(qs[i]), _words(qs[j])
                if a or b:
                    total += len(a & b) / max(1, len(a | b))
                    pairs += 1
        if pairs:
            div.append(1.0 - total / pairs)
    intra = sum(div) / len(div) if div else 0.0

    # 6. trap_hit_rate
    trap_graded, trap_hit = 0, 0
    trap_detail = []
    for c in valid:
        traps = ACTUAL_TRAPS.get(c["id"])
        if not traps:
            continue
        trap_graded += 1
        neg_words = _words(" ".join(c["brief"]["must_not_contain"]))
        hit = bool(neg_words & set(traps))
        trap_hit += hit
        trap_detail.append({"id": c["id"], "hit": hit,
                             "negatives": c["brief"]["must_not_contain"],
                             "actual_trap": traps})

    return {
        "model": run.get("model"),
        "prompt_version": run.get("prompt_version"),
        "avg_seconds_per_case": run.get("avg_seconds_per_case"),
        "n_cases": len(cases),
        "valid_json_rate": round(len(valid) / max(1, len(cases)), 4),
        "reading_accuracy": round(correct / max(1, graded), 4) if graded else None,
        "reading_graded_n": graded,
        "self_harm_rate": round(len(self_harm) / max(1, len(valid)), 4),
        "cross_slot_distinct": round(distinct, 4),
        "shared_queries_n": len(shared),
        "intra_slot_diversity": round(intra, 4),
        "trap_hit_rate": round(trap_hit / max(1, trap_graded), 4) if trap_graded else None,
        "trap_graded_n": trap_graded,
        "_detail": {"reading": reading_detail, "self_harm": self_harm,
                     "shared_queries": shared, "traps": trap_detail},
    }


def _fmt(v):
    return "н/д" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))


def main():
    paths = sys.argv[1:]
    if not paths:
        print(__doc__)
        return 1
    results = [(p, analyze(p)) for p in paths]

    axes = [
        ("valid_json_rate", "валидный JSON", "выше"),
        ("reading_accuracy", "буквальное/фигуральное", "выше"),
        ("self_harm_rate", "запрос против блоклиста", "НИЖЕ"),
        ("cross_slot_distinct", "уникальность запросов", "выше"),
        ("intra_slot_diversity", "разнообразие внутри слота", "выше"),
        ("trap_hit_rate", "предсказал реальную ловушку", "выше"),
        ("avg_seconds_per_case", "секунд на слот", "НИЖЕ"),
    ]
    name_w = max(len(a[1]) for a in axes) + 2
    header = "ось".ljust(name_w) + "лучше  " + "  ".join(
        os.path.basename(p)[:18].ljust(18) for p, _ in results)
    print(header)
    print("-" * len(header))
    for key, label, direction in axes:
        line = label.ljust(name_w) + direction.ljust(7)
        line += "  ".join(_fmt(r.get(key)).ljust(18) for _, r in results)
        print(line)

    # Подробности по последнему прогону — что именно не так.
    _, last = results[-1]
    d = last["_detail"]
    bad_reading = [x for x in d["reading"] if not x["ok"]]
    if bad_reading:
        print(f"\nОшибки чтения ({len(bad_reading)}):")
        for x in bad_reading:
            print(f"  {x['id']} ({x['class']}): ждали {x['expected']}, получили {x['got']}")
    if d["self_harm"]:
        print(f"\nЗапросы против собственного блоклиста ({len(d['self_harm'])}):")
        for x in d["self_harm"]:
            print(f"  {x['id']}: {x['terms']} -> {x['queries']}")
    if d["shared_queries"]:
        print(f"\nЗапросы, поделённые между слотами ({len(d['shared_queries'])}):")
        for q, slots in list(d["shared_queries"].items())[:8]:
            print(f"  «{q}» -> {slots}")
    missed = [x for x in d["traps"] if not x["hit"]]
    if missed:
        print(f"\nНе предсказал реальную ловушку ({len(missed)} из {last['trap_graded_n']}):")
        for x in missed:
            print(f"  {x['id']}: назвал {x['negatives']}")
            print(f"      реально попал в: {x['actual_trap']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
