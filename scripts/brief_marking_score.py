# -*- coding: utf-8 -*-
"""Расшифровка слепой разметки: превратить отметки владельца в вердикт.

ЗАЧЕМ. Лист (`brief_marking_sheet.py`) обезличивает варианты и прячет
ключ, чтобы разметка была честной. Значит сам по себе размеченный лист
нечитаем — буква «Б» у каждого юнита своя. Этот скрипт соединяет отметки
с ключом и печатает то, ради чего всё делалось: сколько раз ЧЕЛОВЕК
выбрал каждую систему.

ЧЕМ ЭТО ОТЛИЧАЕТСЯ ОТ ВСЕГО ОСТАЛЬНОГО ЗАМЕРА. Все прочие числа
(docs/quality/SHOT_BRIEF_DIRECTOR.md) считаны против эталона, который
написал Claude в прошлой сессии, — то есть одна из рук отчасти мерит саму
себя, а метрика «совпал ли предмет» не отличает ДРУГОЙ ХОРОШИЙ кадр от
промаха. Здесь ни того, ни другого: судит человек, вслепую.

ЧТО СЧИТАЕТСЯ ЧЕСТНО, А ЧТО НЕТ.
  * «выбран N раз» — честно: это прямой счёт голосов.
  * «нет» (ни один вариант не годится) считается отдельно и НЕ
    приписывается никому: это тоже результат, и важный — он говорит, что
    задачу не решил никто.
  * доверительных интервалов здесь не печатается сознательно: 40 юнитов
    — это 40 юнитов, и раздутая статистика на такой выборке создаёт
    ложную уверенность (тот же урок, что в этом проекте уже дважды
    оплачен порогами на сырой косинус).
"""
import argparse
import collections
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

_UNIT_RE = re.compile(r"^###\s+(\d+)\.")
_MARK_RE = re.compile(r"^лучший:\s*(.*?)(?:\s{2,}почему|$)", re.I)
_NONE = {"нет", "ни один", "никто", "-", "—", "none"}


def read_sheet(path):
    """Отметки из размеченного листа: {номер юнита: [буквы] или NONE}."""
    marks, unit = {}, None
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = _UNIT_RE.match(line.strip())
            if m:
                unit = int(m.group(1))
                continue
            m = _MARK_RE.match(line.strip())
            if not m or unit is None:
                continue
            raw = m.group(1).replace("_", " ").strip(" .:")
            if not raw:
                unit = None
                continue
            if raw.lower() in _NONE:
                marks[unit] = None
            else:
                letters = [c for c in raw.upper()
                           if c in "АБВГДЕABCDEF"]
                if letters:
                    marks[unit] = letters
            unit = None
    return marks


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("sheet", help="размеченный .md")
    ap.add_argument("key", help="…_KEY.json, снятый при сборке листа")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv[1:])

    with open(a.key, encoding="utf-8") as f:
        key = json.load(f)
    by_unit = {u["unit"]: u["map"] for u in key["units"]}
    marks = read_sheet(a.sheet)

    wins = collections.Counter()
    nobody, unmarked, ties = 0, 0, 0
    detail = []
    for unit, mapping in by_unit.items():
        got = marks.get(unit, "НЕТ ОТМЕТКИ")
        if got == "НЕТ ОТМЕТКИ":
            unmarked += 1
            continue
        if got is None:
            nobody += 1
            detail.append({"unit": unit, "выбрано": None})
            continue
        chosen = [mapping[l] for l in got if l in mapping]
        if not chosen:
            unmarked += 1
            continue
        if len(chosen) > 1:
            ties += 1
        for name in chosen:
            wins[name] += 1
        detail.append({"unit": unit, "выбрано": chosen})

    judged = len(by_unit) - unmarked
    print(f"Размечено {judged} юнитов из {len(by_unit)}"
          + (f" (без отметки {unmarked})" if unmarked else ""))
    print(f"Ни один вариант не годится: {nobody}")
    if ties:
        print(f"Отмечено несколько равных: {ties} (голос идёт каждому)")
    print()
    width = max(len(n) for n in key["arms"])
    print(f"{'система'.ljust(width)}  выбран  доля от размеченных")
    print("-" * (width + 30))
    for name in key["arms"]:
        n = wins[name]
        print(f"{name.ljust(width)}  {n:6d}  {n / max(1, judged):.0%}")
    print()
    print("Это ЕДИНСТВЕННОЕ число в замере, где судит человек и вслепую. "
          "Все остальные считаны против эталона, написанного моделью.")

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"judged": judged, "nobody_fits": nobody,
                       "ties": ties, "unmarked": unmarked,
                       "wins": dict(wins), "units": detail},
                      f, ensure_ascii=False, indent=2)
        print(f"\nJSON: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
