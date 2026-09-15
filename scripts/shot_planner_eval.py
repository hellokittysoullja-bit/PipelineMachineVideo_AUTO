# -*- coding: utf-8 -*-
"""Замер локального режиссёра на РЕАЛЬНЫХ фразах эпизода.

Зачем отдельный инструмент. Первое измерение качества планировщика было
сделано на трёх юнитах — этого мало, чтобы принимать решение о флаге, и
воспроизвести его можно было только руками. Здесь тот же замер становится
одной командой: выбрать разнотипные юниты, прогнать боевым промптом,
положить ответы рядом с фразами.

Инструмент НИЧЕГО не решает и ничего не чинит. Он печатает, что модель
ответила, чтобы человек посмотрел глазами — ровно тот же принцип, что у
shot_brief_review.py (сверка брифов с полкой) и у контактного листа: между
«написал промпт» и «увидел результат» не должно стоять часа рендера.

Запуск:
    LLAMA_CLI_BIN=... LLAMA_MODEL_GGUF=... SHOT_PLANNER_LLM=1 \
        python scripts/shot_planner_eval.py videos/02_ne-mechom --take 8
"""
import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import script_parser  # noqa: E402
import shot_planner_llm as planner  # noqa: E402

# Классы фраз, на которых планировщик ломается по-разному. Выборка «первые
# N юнитов» скрыла бы это: начало эпизода — почти всегда предметные кадры.
CLASSES = {
    "метафора": ("как ", "словно", "будто"),
    "отрицание": ("не ", " ни "),
    "число": ("килограмм", "процент", "тысяч", "метр"),
    "сцена": ("поле", "грязь", "земл", "лагер", "дорог"),
    "вопрос": ("?",),
}


def classify(text):
    t = (text or "").lower()
    hits = [name for name, keys in CLASSES.items() if any(k in t for k in keys)]
    return hits or ["простая"]


def pick_units(blocks, take):
    """По юниту каждого класса, потом добор — чтобы выборка была
    разнотипной, а не первыми N подряд."""
    seen, picked = set(), []
    for cls in list(CLASSES) + ["простая"]:
        for i, b in enumerate(blocks):
            if i in seen or cls not in classify(b.get("text")):
                continue
            picked.append((i, b, cls))
            seen.add(i)
            break
    for i, b in enumerate(blocks):
        if len(picked) >= take:
            break
        if i not in seen:
            picked.append((i, b, "/".join(classify(b.get("text")))))
            seen.add(i)
    return picked[:take]


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("video_dir")
    ap.add_argument("--take", type=int, default=8)
    ap.add_argument("--out", default=None, help="куда положить JSON результата")
    a = ap.parse_args(argv[1:])

    if not planner.runtime_ready():
        print("Нет LLAMA_CLI_BIN / LLAMA_MODEL_GGUF — замерять нечем")
        return 2

    blocks = script_parser.parse_blocks(os.path.join(a.video_dir, "script.txt"))
    units = pick_units(blocks, a.take)
    print(f"Модель: {os.path.basename(planner.LLAMA_MODEL)}  "
          f"промпт v{planner.PLANNER_PROMPT_VERSION}  юнитов: {len(units)}\n")

    rows, t0 = [], time.time()
    for n, (i, b, cls) in enumerate(units, 1):
        text = (b.get("text") or "").strip()
        t1 = time.time()
        # Контекст строится ТОЙ ЖЕ функцией, что в проде. Мерить модель на
        # промпте, отличном от рабочего, — значит получить число, которое
        # ни к чему не относится.
        ctx = planner.unit_context(blocks, i)
        got = planner.plan_unit(text, ctx, None)  # без кэша: меряем модель
        dt = time.time() - t1
        print(f"[{i:3}] ({cls}) {text[:88]}")
        if got:
            print(f"      shot_en : {got['shot_en']}")
            print(f"      function: {got['function']}   forbidden: {got['forbidden']}")
        else:
            print("      ОТВЕТ НЕ РАЗОБРАН")
        print(f"      {dt:.0f}с\n")
        rows.append({"index": i, "class": cls, "text": text,
                     "result": got, "sec": round(dt, 1)})

    ok = sum(1 for r in rows if r["result"])
    print(f"Разобрано {ok} из {len(rows)}; всего {time.time() - t0:.0f}с, "
          f"живых вызовов {planner.STATS['calls']}, "
          f"невалидных {planner.STATS['invalid']}, "
          f"таймаутов {planner.STATS['timeouts']}")
    print("\nВердикт по каждому кадру — ГЛАЗАМИ. Инструмент показывает "
          "ответ модели, а не оценивает его.")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"model": os.path.basename(planner.LLAMA_MODEL),
                       "prompt_version": planner.PLANNER_PROMPT_VERSION,
                       "rows": rows}, f, ensure_ascii=False, indent=2)
        print(f"JSON: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
