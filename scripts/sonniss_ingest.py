#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sonniss GDC Game Audio Bundle -> библиотека звуков, одной командой.

ЗАЧЕМ. Openverse/Freesound отдают CC0-превью 128 kbps, и это честный предел
(оригинал требует OAuth пользователя). Sonniss GDC — профессиональные
студийные WAV, раздаются бесплатно и целиком. `sound_library.py ingest --dir`
уже умеет индексировать локальную папку ТЕМИ ЖЕ гейтами, но принимает ОДИН
вид на запуск и обходит всю папку — а в одной части бандла тысячи файлов от
десятков студий, и живые гейты (CLAP + AST, секунды на файл) по всему
дереву на каждый из десяти видов — это часы впустую.

Этот скрипт делает три вещи, которых у `ingest --dir` нет, и НЕ дублирует
ни одного решения о качестве: отбор по-прежнему делает `ingest_dir()`.

1. СКАЧИВАНИЕ. Измерено 14.09: `sonniss.com` и `downloads.sonniss.com`
   отдают скрипту HTTP 403 (Cloudflare), а зеркала — 200. Поэтому по
   умолчанию берётся зеркало; часть GDC-2019 №1 — 3.82 ГБ. Докачка
   поддерживается (`curl -C -`), готовый файл не перекачивается.

2. МАРШРУТИЗАЦИЯ ПО ВИДАМ. Шорт-лист строится по ИМЕНИ файла
   (`sound_library.title_relevance()` — те же слова, что в `queries` вида) и
   по длительности из спецификации вида. Это ТОЛЬКО ускорение: шорт-лист
   может лишь СУЗИТЬ круг, решение «годится или нет» принимают настоящие
   измерительные гейты и модели внутри `ingest_dir()`, не имя файла.
   **Честный предел, названный заранее:** звук, который подошёл бы по
   содержанию, но назван неинформативно («FX_0421.wav»), до гейтов не
   доедет вообще. Это цена за то, чтобы прогон занимал минуты, а не часы;
   снимается флагом `--no-name-filter` (тогда гейты видят всё, и это долго).

3. ЛИЦЕНЗИЯ НЕ УГАДЫВАЕТСЯ И НЕ СПРАШИВАЕТСЯ. Условия Sonniss проверены на
   их же странице 14.09: коммерческое использование разрешено, атрибуция не
   требуется, «Use for AI/ML training is strictly prohibited». Поэтому
   лицензия проставляется одной и той же строкой на каждую запись, а
   `allow_ai_embeddings` ЖЁСТКО False — это не флаг и не умолчание, которое
   можно случайно переключить: пакет прямо запрещает обучение, и вопрос
   «можно ли публиковать эмбеддинги этих записей» обязан иметь ответ в
   данных. Инференс ради отбора под запрет не попадает — пайплайн не учит
   модели, он их спрашивает.

ПОЧЕМУ РЕЗУЛЬТАТ ПЕРЕЖИВАЕТ КОНТЕЙНЕР. Принятые файлы кладутся в
`assets/library/<kind>/<вид>/` и коммитятся в git (папка `object/` весит
2.8 МБ — замер 14.09; в .gitignore она попала по ошибочной оценке «сотни
МБ», верной только для `ambience/`, 639 МБ). То есть после одного прогона
звуки есть у любого клона навсегда — в отличие от `restore`, который
качает по URL, а у локального пакета URL нет.

Использование:
    python scripts/sonniss_ingest.py --year 2019 --part 1
    python scripts/sonniss_ingest.py --dir /путь/к/распакованному
    python scripts/sonniss_ingest.py --year 2019 --part 1 --kinds object
    ... [--keep-download] [--no-name-filter] [--dry-run]

По умолчанию берутся виды `object` и `sfx`. `ambience` намеренно НЕ по
умолчанию: записи атмосферы длинные, их папка и так 639 МБ и в git не
хранится — добавлять туда ещё сотни МБ без явного запроса неправильно.
"""
import argparse
import os
import shutil
import subprocess
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sound_library as sl  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Условия проверены на sonniss.com/gameaudiogdc 14.09: royalty free,
# коммерческое использование разрешено, атрибуция НЕ требуется, обучение
# ИИ запрещено. Строка едет в манифест на каждую запись.
SONNISS_LICENSE = "Sonniss GDC Game Audio Bundle (royalty-free, commercial use, no attribution)"
SONNISS_LICENSE_URL = "https://sonniss.com/gameaudiogdc"
SONNISS_CREDIT = "Sonniss GDC Game Audio Bundle"

# Зеркала, отдающие скрипту 200 (сам sonniss.com — 403 за Cloudflare,
# измерено 14.09). Порядок = порядок попыток.
MIRRORS = {
    2019: ["https://ftpmirror.your.org/pub/misc/sonniss2019/"
           "Sonniss.com%20-%20GDC%202019%20-%20Game%20Audio%20Bundle%20Part%20{part}of8.zip"],
    2018: ["https://ftpmirror.your.org/pub/misc/sonniss2018/"
           "Sonniss.com%20-%20GDC%202018%20-%20Game%20Audio%20Bundle%20Part%20{part}of8.zip"],
    2017: ["https://ftpmirror.your.org/pub/misc/sonniss2017/"
           "Sonniss.com%20-%20GDC%202017%20-%20Game%20Audio%20Bundle%20Part%20{part}of9.zip"],
    2016: ["https://ftpmirror.your.org/pub/misc/sonniss2016/"
           "Sonniss.com%20-%20GDC%202016-%20Game%20Audio%20Bundle%20Part%20{part}of6.zip"],
    2024: ["https://hippolytus.feralhosting.com/sonniss/"
           "Sonniss.com-GDC2024-GameAudioBundle{part}of9.zip"],
    2023: ["https://hippolytus.feralhosting.com/sonniss/"
           "Sonniss.com-GDC2023-GameAudioBundle{part}of14.zip"],
}
PARTS = {2024: 9, 2023: 14, 2019: 8, 2018: 8, 2017: 9, 2016: 6}

DEFAULT_KINDS = ("object", "sfx")


def bundle_urls(year, part):
    if year not in MIRRORS:
        raise SystemExit(f"нет зеркала для {year}; доступны: {sorted(MIRRORS)}")
    if not 1 <= part <= PARTS[year]:
        raise SystemExit(f"у GDC {year} частей {PARTS[year]}, запрошена {part}")
    return [m.format(part=part) for m in MIRRORS[year]]


def download(urls, dst):
    """Скачать первым отвечающим зеркалом, с докачкой. Готовый файл не
    трогаем: 3.8 ГБ на часть — перекачивать «на всякий случай» нельзя."""
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print(f"  уже скачано: {dst} ({os.path.getsize(dst)/1e9:.2f} ГБ)")
        return dst
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    for url in urls:
        print(f"  качаю {url}")
        r = subprocess.run(["curl", "-fL", "-C", "-", "--retry", "3",
                            "-A", "Mozilla/5.0", "-o", dst, url])
        if r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 0:
            print(f"  готово: {os.path.getsize(dst)/1e9:.2f} ГБ")
            return dst
        print(f"  зеркало не отдало (код {r.returncode})")
    raise SystemExit("ни одно зеркало не отдало файл")


def extract_audio(zip_path, out_dir):
    """Распаковать ТОЛЬКО аудио. В бандле рядом лежат pdf/txt/картинки
    студий — они занимают место и в отборе не участвуют."""
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            if not info.filename.lower().endswith(sl.LOCAL_AUDIO_EXT):
                continue
            # Плоское имя: вложенность папок студий в отборе не нужна, а
            # длинные пути упираются в лимит Windows.
            safe = os.path.basename(info.filename)
            dst = os.path.join(out_dir, safe)
            if os.path.exists(dst):
                stem, ext = os.path.splitext(safe)
                dst = os.path.join(out_dir, f"{stem}__{n}{ext}")
            with z.open(info) as src, open(dst, "wb") as out:
                shutil.copyfileobj(src, out)
            n += 1
    print(f"  распаковано аудио-файлов: {n}")
    return n


def shortlist_for(spec, name, files, use_names=True):
    """Кандидаты вида — по СЛОВАМ запросов вида в имени файла.

    Только сужает: всё, что попало сюда, дальше судят настоящие гейты
    внутри ingest_dir(). Порядок — по числу совпавших слов (та же
    title_relevance, что уже упорядочивает выдачу стока).
    """
    if not use_names:
        return list(files)
    out = []
    for f in files:
        base = os.path.basename(f)
        if sl.title_blocked(name, base):
            continue
        rel = sl.title_relevance(spec, base.replace("_", " ").replace("-", " "))
        if rel > 0:
            out.append((rel, f))
    out.sort(key=lambda t: -t[0])
    return [f for _, f in out]


def ingest_bundle(src_dir, kinds, label, use_names=True, dry_run=False):
    manifest = sl.load_manifest()
    files = []
    for root, _, fs in os.walk(src_dir):
        for f in sorted(fs):
            if f.lower().endswith(sl.LOCAL_AUDIO_EXT):
                files.append(os.path.join(root, f))
    print(f"\nВсего аудио в пакете: {len(files)}")
    if not files:
        raise SystemExit(f"в {src_dir} нет аудио")

    work = os.path.join(REPO, "temp_library", "sonniss_shortlist")
    total = 0
    for kind in kinds:
        for name, spec in sl.LIBRARY_SPEC[kind].items():
            cand = shortlist_for(dict(spec, _name=name), name, files, use_names)
            print(f"\n== {kind}/{name}: шорт-лист по имени {len(cand)} из {len(files)}")
            if not cand:
                continue
            if dry_run:
                for f in cand[:8]:
                    print("     " + os.path.basename(f)[:70])
                continue
            # Папка ссылок, а не копий: гейтам нужен путь к файлу, а
            # копировать гигабайты ради переименования папки незачем.
            # Имя папки уезжает в манифест полем "query" — по нему потом
            # видно, из какого пакета пришла запись.
            d = os.path.join(work, f"{label}__{name}")
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)
            for f in cand[:400]:
                link = os.path.join(d, os.path.basename(f))
                if os.path.exists(link):
                    continue
                try:
                    os.symlink(os.path.abspath(f), link)
                except OSError:
                    shutil.copy2(f, link)
            before = len(manifest["items"])
            sl.ingest_dir(manifest, d, kind, name, SONNISS_LICENSE, SONNISS_LICENSE_URL,
                          credit=SONNISS_CREDIT, allow_ai_embeddings=False)
            manifest = sl.load_manifest()
            total += len(manifest["items"]) - before
            shutil.rmtree(d, ignore_errors=True)
    shutil.rmtree(work, ignore_errors=True)
    print(f"\nВсего принято новых записей: {total}")
    return total


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=2019)
    ap.add_argument("--part", type=int, default=1)
    ap.add_argument("--dir", default="", help="уже распакованная папка (тогда без скачивания)")
    ap.add_argument("--kinds", default=",".join(DEFAULT_KINDS),
                    help="через запятую: object,sfx,ambience")
    ap.add_argument("--work", default="", help="куда качать/распаковывать (по умолчанию temp_library/sonniss)")
    ap.add_argument("--keep-download", action="store_true", help="не удалять zip и распаковку")
    ap.add_argument("--no-name-filter", action="store_true",
                    help="не сужать по имени файла — гейты увидят ВСЁ (долго)")
    ap.add_argument("--dry-run", action="store_true", help="только показать шорт-листы")
    args = ap.parse_args(argv)

    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    for k in kinds:
        if k not in sl.LIBRARY_SPEC:
            raise SystemExit(f"нет вида {k}; есть: {', '.join(sl.LIBRARY_SPEC)}")

    if args.dir:
        src, label, cleanup = args.dir, os.path.basename(args.dir.rstrip("/")), None
    else:
        work = args.work or os.path.join(REPO, "temp_library", "sonniss")
        label = f"sonniss-gdc{args.year}-part{args.part}"
        zip_path = os.path.join(work, label + ".zip")
        print(f"Sonniss GDC {args.year}, часть {args.part}")
        download(bundle_urls(args.year, args.part), zip_path)
        src = os.path.join(work, label)
        if not os.path.isdir(src) or not os.listdir(src):
            print("  распаковываю аудио...")
            extract_audio(zip_path, src)
        cleanup = (zip_path, src)

    try:
        ingest_bundle(src, kinds, label, use_names=not args.no_name_filter,
                      dry_run=args.dry_run)
    finally:
        if cleanup and not args.keep_download and not args.dry_run:
            print("  убираю временные файлы пакета")
            for p in cleanup:
                shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else (
                    os.path.exists(p) and os.remove(p))
    print("\nПринятые файлы лежат в assets/library/ и коммитятся в git — "
          "после коммита они есть у любого клона, без повторного скачивания.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
