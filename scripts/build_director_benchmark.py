# -*- coding: utf-8 -*-
"""Сборка набора ТРУДНЫХ случаев для замера локального режиссёра (Контур A).

Зачем отдельный набор, а не золотой набор целиком. Золотой набор
(tests/fixtures/golden_set/) меряет ГЕЙТЫ на уже скачанных кадрах — он
отвечает на вопрос "пропустил ли гейт брак". Здесь другой вопрос:
"стал ли САМ ЗАПРОС к стоку осмысленнее" — то есть то, что происходит
ДО скачивания, и чего золотой набор по построению не видит.

Принцип отбора, критичный для честности замера: набор ОБЯЗАН содержать
и кадры, которые сегодня ПЛОХИЕ (цель улучшения), и кадры, которые
сегодня ХОРОШИЕ (защита от регресса). Набор только из провалов умеет
показать лишь улучшение и слеп к тому, что новый механизм сломал
работавшее — а требование пользователя прямое: "только upgrade, без
минусов и компромиссов", проверить это можно единственным способом —
включив в замер то, что уже работает.

Классы сложности выделены по ПРИЧИНЕ, по которой эмбеддинг ошибается,
а не по теме — иначе замер смешал бы разные механизмы отказа:

  literal        — предметная фраза, объект существует и назван прямо.
                   Эмбеддингу тут и так неплохо; класс нужен как
                   регрессионный якорь.
  compositional  — объект назван, но решает АТРИБУТ или отношение
                   ("меч на четыре килограмма", "положи на палец").
                   Ровно тот режим отказа, который задокументирован в
                   ARO/SugarCrepe/Winoground: модель ведёт себя как
                   мешок слов и атрибут теряет.
  figurative     — искомого объекта физически НЕ существует
                   ("пятнадцать килограммов" — про миф, не про предмет).
                   Буквальный поиск обязан промахнуться по построению.
  fragment       — во фразе нет ни одного визуального существительного
                   ("Не дрались. Несли."). Смысл целиком в соседях.
  metaphor       — образ, а не предмет ("человек в термосе").
                   Здесь "правильного" кадра не существует ни для кого —
                   меряется согласованность, не попадание.
  anachronism_risk — тема, где сток систематически подсовывает не ту
                   культуру/эпоху (в опубликованном эпизоде это дало
                   4 брака подряд с одного запроса).

Запуск: .venv/bin/python scripts/build_director_benchmark.py
Результат: tests/fixtures/director_benchmark/cases.json
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

GOLDEN = os.path.join(REPO, "tests", "fixtures", "golden_set", "manifest.json")
OUT_DIR = os.path.join(REPO, "tests", "fixtures", "director_benchmark")
OUT = os.path.join(OUT_DIR, "cases.json")

# id -> класс сложности. Отобрано вручную по РЕАЛЬНЫМ вердиктам
# опубликованного эпизода (см. докстринг про смешанный состав).
CASE_CLASSES = {
    # --- регрессионные якоря: сегодня ХОРОШО, обязаны остаться хорошими ---
    "ep01_011": "literal",
    "ep01_026": "literal",
    "ep01_128": "literal",
    "ep01_076": "literal",
    "ep01_090": "fragment",       # сегодня good, но держится на контексте соседей
    "ep01_133": "metaphor",       # сегодня good — "человек в термосе"
    "ep01_160": "metaphor",       # сегодня good — "лучшая метафора"
    "ep01_048": "literal",
    # --- цели улучшения: сегодня БРАК ---
    "ep01_000": "figurative",     # "Пятнадцать килограммов" -> современная кухня
    "ep01_005": "figurative",     # "пакет молока в руке" -> младенец с бутылочкой
    "ep01_008": "compositional",  # "меч на четыре килограмма" -> современное вторжение
    "ep01_068": "compositional",  # "положи на палец" -> спортивное фехтование
    "ep01_002": "anachronism_risk",
    "ep01_006": "anachronism_risk",
    "ep01_122": "anachronism_risk",
    "ep01_084": "compositional",  # "несли перед знатным человеком" -> не тот предмет
    "ep01_140": "anachronism_risk",
    "ep01_009": "fragment",       # "врёт эта табличка" -> современное вторжение
    # --- пограничные: сегодня "терпимо" ---
    "ep01_055": "fragment",
    "ep01_146": "metaphor",
}


def main():
    with open(GOLDEN, encoding="utf-8") as f:
        golden = json.load(f)
    by_id = {it["id"]: it for it in golden["items"]}
    # Порядок блоков в эпизоде — для окна контекста (±1 фраза).
    ordered = sorted(golden["items"], key=lambda it: it["index"])
    pos = {it["id"]: i for i, it in enumerate(ordered)}

    cases = []
    missing = []
    for cid, cls in CASE_CLASSES.items():
        it = by_id.get(cid)
        if it is None:
            missing.append(cid)
            continue
        i = pos[cid]
        prev_text = ordered[i - 1]["text"] if i > 0 else ""
        next_text = ordered[i + 1]["text"] if i + 1 < len(ordered) else ""
        cases.append({
            "id": cid,
            "difficulty_class": cls,
            "section": it["section"],
            "index": it["index"],
            "text": it["text"],
            # Окно контекста — вход режиссёра. НЕ то же, что
            # semantic_context_text() (та добавляет соседа только когда
            # своя фраза короче порога): режиссёру контекст даётся всегда,
            # он сам решает, нужен ли.
            "context_prev": prev_text,
            "context_next": next_text,
            # Базовая линия — что реально делала система на этом слоте.
            "baseline_query": it["query"],
            "baseline_verdict": it["verdict"],
            "baseline_reject_reason": it.get("reject_reason"),
            "baseline_note": it.get("note"),
        })
    if missing:
        raise SystemExit(f"нет в золотом наборе: {missing}")

    cases.sort(key=lambda c: c["index"])
    by_class = {}
    for c in cases:
        by_class.setdefault(c["difficulty_class"], []).append(c["id"])
    by_verdict = {}
    for c in cases:
        by_verdict[c["baseline_verdict"]] = by_verdict.get(c["baseline_verdict"], 0) + 1

    payload = {
        "schema_version": 1,
        "source": "tests/fixtures/golden_set/manifest.json — реальные слоты "
                  "опубликованного эпизода 01_ves-mecha с вердиктами глазами",
        "purpose": "замер КАЧЕСТВА ЗАПРОСА (то, что происходит до скачивания), "
                   "а не гейтов на уже скачанном кадре",
        "honest_composition": {
            "по классам": by_class,
            "по базовому вердикту": by_verdict,
            "почему смешанный": "набор только из провалов показал бы лишь "
                                 "улучшение и был бы слеп к регрессу на том, "
                                 "что уже работает",
        },
        "cases": cases,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    os.replace(tmp, OUT)

    print(f"Случаев: {len(cases)}")
    print(f"  по базовому вердикту: {by_verdict}")
    for cls in sorted(by_class):
        print(f"  {cls:<18} {len(by_class[cls])}: {', '.join(by_class[cls])}")
    print(f"Записано: {OUT}")


if __name__ == "__main__":
    main()
