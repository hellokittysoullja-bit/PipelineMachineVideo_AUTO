#!/usr/bin/env python3
"""Единственный способ пополнить assets/lookbook/lookbook.json (см.
scripts/look_reference.py) — вручную, одним реально одобренным кадром
канала за раз. Сам pipeline (pipeline_smart.py) lookbook никогда не
пишет, только читает — эталон утверждает человек, не автоматика (та же
ошибка, за которую критиковали _scene_bias: коррекция без внешней,
реально проверенной цели).

Usage: python scripts/lookbook_add.py <photo_path> --domain <domain> [--id <id>]
Домены — см. look_reference.DOMAIN_PROMPTS (snow/night/museum_daylight/
portrait/urban/archive_bw/ai_illustration/battle).

Копирует фото в assets/lookbook/<id>.<ext> (lookbook не зависит от того,
останется ли исходный файл на месте) и дописывает запись в lookbook.json:
lab_mean/brightness/contrast/temperature — реальные измеренные величины
(pipeline_smart.measure_levels/look_reference._srgb_to_lab/_scene_bias, те
же функции, что использует рендер, не отдельная копия логики)."""
import datetime
import json
import os
import shutil
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

_real_argv = sys.argv
sys.argv = ["look_reference.py", tempfile.gettempdir()]
import look_reference as lr   # noqa: E402
import pipeline_smart          # noqa: E402
sys.argv = _real_argv


def atomic_write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def parse_args(argv):
    if not argv:
        return None
    photo_path = argv[0]
    domain = None
    ref_id = None
    i = 1
    while i < len(argv):
        if argv[i] == "--domain" and i + 1 < len(argv):
            domain = argv[i + 1]
            i += 2
        elif argv[i] == "--id" and i + 1 < len(argv):
            ref_id = argv[i + 1]
            i += 2
        else:
            i += 1
    return photo_path, domain, ref_id


def main():
    parsed = parse_args(_real_argv[1:])
    if not parsed or not parsed[1]:
        print("Usage: python scripts/lookbook_add.py <photo_path> --domain <domain> [--id <id>]")
        print(f"Домены: {', '.join(sorted(lr.DOMAIN_PROMPTS))}")
        return 1
    photo_path, domain, ref_id = parsed

    if domain not in lr.DOMAIN_PROMPTS:
        print(f"Неизвестный домен {domain!r}. Домены: {', '.join(sorted(lr.DOMAIN_PROMPTS))}")
        return 1
    if not os.path.exists(photo_path):
        print(f"Файл не найден: {photo_path}")
        return 1

    levels, wb = pipeline_smart.measure_levels(photo_path, want_wb=True)
    if wb is None or levels is None or levels[0] is None:
        print(f"Не удалось измерить levels/wb для {photo_path} (битый файл/вырожденная гистограмма?)")
        return 1

    lab_mean = lr._srgb_to_lab(wb)
    warm_bias, _, _ = pipeline_smart._scene_bias(levels, wb)
    brightness = (levels[0] + levels[1]) / 2.0
    contrast = levels[1] - levels[0]

    ext = os.path.splitext(photo_path)[1] or ".jpg"
    if not ref_id:
        ref_id = f"{domain}_{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

    lookbook_dir = os.path.dirname(lr.LOOKBOOK_PATH)   # НЕ модульная константа —
                                                          # читается заново на каждый
                                                          # вызов, чтобы тесты могли
                                                          # безопасно monkeypatch'ить
                                                          # lr.LOOKBOOK_PATH на tmp_path,
                                                          # не трогая реальный lookbook
    os.makedirs(lookbook_dir, exist_ok=True)
    dest_path = os.path.join(lookbook_dir, f"{ref_id}{ext}")
    shutil.copyfile(photo_path, dest_path)

    lookbook = lr.load_lookbook()
    references = [r for r in lookbook.get("references", []) if r.get("id") != ref_id]
    references.append({
        "id": ref_id,
        "domain": domain,
        "image": os.path.relpath(dest_path, REPO_ROOT),
        "added_at": datetime.datetime.utcnow().isoformat() + "Z",
        "lab_mean": [round(x, 4) for x in lab_mean],
        "brightness": round(brightness, 4),
        "contrast": round(contrast, 4),
        "temperature": round(warm_bias, 4),
        "max_correction_delta": list(lr.DEFAULT_MAX_CORRECTION_DELTA),
    })
    atomic_write_json(lr.LOOKBOOK_PATH, {"references": references})

    print(f"Добавлен эталон {ref_id!r} (домен={domain}) в {lr.LOOKBOOK_PATH}")
    print(f"  lab_mean={[round(x, 2) for x in lab_mean]} brightness={brightness:.3f} "
          f"contrast={contrast:.3f} temperature={warm_bias:.3f}")
    print(f"  Всего эталонов в lookbook: {len(references)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
