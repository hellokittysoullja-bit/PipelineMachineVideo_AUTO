#!/usr/bin/env python3
"""Перемеряет graded_lab_mean/graded_recipe_fingerprint (см.
look_reference.graded_reference_lab) для УЖЕ существующих записей
lookbook.json — без этого поля closed-loop проверка (_closed_loop_improves)
честно пропускает эталон (fail-open, см. докстринг там), то есть P1-3
форензик-защита молчит для немигрированных записей.

Когда запускать: (1) один раз сразу после того, как графики film_look()
не имели graded_lab_mean вообще (эта миграция); (2) снова — после любой
правки film_look()/MOOD_GRADE/DOMAIN_WARM_PUSH_SCALE, которая меняет
реальный грейд: graded_recipe_fingerprint у СТАРЫХ записей перестанет
совпадать с текущим кодом, и _closed_loop_improves будет их пропускать,
пока не перемеряешь заново.

Usage: python scripts/lookbook_remeasure.py [--force]
--force — перемерить даже записи, у которых fingerprint уже совпадает с
текущим film_look() (иначе они по умолчанию пропускаются — дёшево, не
дёргать ffmpeg зря)."""
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

_real_argv = sys.argv
sys.argv = ["look_reference.py", os.path.join(REPO_ROOT, "videos")]
import look_reference as lr   # noqa: E402
import pipeline_smart          # noqa: E402
sys.argv = _real_argv


def atomic_write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main():
    force = "--force" in _real_argv[1:]
    if not os.path.exists(lr.LOOKBOOK_PATH):
        print(f"Lookbook не найден: {lr.LOOKBOOK_PATH}")
        return 1
    data = json.load(open(lr.LOOKBOOK_PATH, encoding="utf-8"))
    references = data.get("references", [])
    if not references:
        print("Lookbook пуст — нечего перемерять.")
        return 0

    current_fp = lr._grade_recipe_fingerprint()
    updated = 0
    skipped = 0
    for ref in references:
        if not force and ref.get("graded_recipe_fingerprint") == current_fp and ref.get("graded_lab_mean"):
            skipped += 1
            continue
        image_path = os.path.join(REPO_ROOT, ref["image"])
        if not os.path.exists(image_path):
            print(f"  ПРОПУСК {ref['id']!r}: файл не найден {image_path}")
            continue
        levels, wb = pipeline_smart.measure_levels(image_path, want_wb=True)
        if wb is None or levels is None or levels[0] is None:
            print(f"  ПРОПУСК {ref['id']!r}: не удалось измерить levels/wb")
            continue
        graded = lr.graded_reference_lab(image_path, ref["domain"], levels, wb)
        ref["graded_lab_mean"] = {
            k: ([round(x, 4) for x in v] if v is not None else None)
            for k, v in graded.items()
        }
        ref["graded_recipe_fingerprint"] = current_fp
        failed = [k for k, v in graded.items() if v is None]
        status = f" (ВНИМАНИЕ: не измерилось для {failed})" if failed else ""
        print(f"  {ref['id']!r}: graded_lab_mean обновлён{status}")
        updated += 1

    atomic_write_json(lr.LOOKBOOK_PATH, data)
    print(f"Готово: обновлено {updated}, пропущено (уже актуальны) {skipped}, всего {len(references)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
