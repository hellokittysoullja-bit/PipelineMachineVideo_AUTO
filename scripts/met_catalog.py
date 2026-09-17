"""Локальный каталог Метрополитена: точный запрос ПО ПОЛЯМ вместо угадывания.

Зачем он вообще
---------------
Музейный поиск через API устроен как окошко: мы кричим `q=medieval plate
armour museum`, а сервер сам решает, что мы имели в виду, и отдаёт то, что
совпало по ОПИСАНИЯМ. Отсюда керамические тарелки на слово *plate* (замер
14.09: 480 objectID, первые — настенные часы и «Plate with Water Bird»).
`departmentId` это смягчил, но не убрал: внутри отдела всё равно решает
текстовое совпадение, а не факт «это меч».

Мет публикует ВЕСЬ свой каталог одним файлом под CC0
(github.com/metmuseum/openaccess, MetObjects.csv, ~303 МБ). В нём есть ровно
те поля, по которым надо спрашивать: `Object Name` (что это за предмет),
`Culture`, `Object Begin/End Date`, `Medium`, `Department`, `Classification`,
`Tags`, `Is Public Domain`. То есть музей сам отвечает на вопрос «это меч
или тарелка» — не надо выводить это из описания.

Что это меняет ИЗМЕРИМО (индекс, собранный 14.09)
-------------------------------------------------
Паспортный фильтр применён ОДИН раз ко всему каталогу, теми же функциями,
что работают на API-пути (`era_overlaps`, `culture_is_foreign` из
`museum_sources` — не вторая копия правил):

    всего записей          484 956
    public domain          248 472
    + пересекает эпоху      48 086
    - чужая культура       -16 904
    ОСТАЛОСЬ                31 182

Это ИСТИННЫЙ размер корпуса, доступного каналу. Для сравнения: сегодняшний
API-путь тянет максимум 60 карточек на запрос и после паспорта оставляет
около 20.

Совпадение по полю `Object Name` на словаре канала (всё уже public domain,
в эпохе, европейской культуры):

    sword 59 · helmet 66 · armor 140 · halberd 96 · spur 102 · shield 45
    dagger 23 · gauntlet 35 · breastplate 37 · crossbow 31 · mail 19

Ни один из 59 мечей не японский — не потому, что сработал гейт, а потому
что в колонке `Culture` написано «Italian, Venice» / «Western European» /
«European, probably Scandinavia». Уточнитель «european» на этом пути
становится не нужен: у нас есть поле вместо догадки по слову.

Побочная находка, ради которой каталог полезен сам по себе
-----------------------------------------------------------
`poleaxe` в каталоге — НОЛЬ. Мет называет этот предмет `Halberd` (96 штук)
и `Partisan` (35). То есть авторский запрос и словарь музея расходятся, и
сегодня это лечится случайно — тем, что свободный текст цепляется за
описание. Каталог даёт настоящий словарь музея, и синонимы берутся ИЗ
ДАННЫХ (`vocabulary()`), а не из моей головы.

Честные пределы, названные сразу
--------------------------------
* **Ссылок на снимки в дампе НЕТ.** Есть только `Object ID`. Значит за
  картинкой всё равно идём в API — но ТОЛЬКО за теми предметами, которые
  уже прошли паспорт локально. Сегодня тянется 60 карточек и выживает ~20,
  то есть 40 запросов тратятся впустую; здесь впустую не тратится ни один.
  Это сокращение бесполезных запросов, а НЕ отмена сети.
* **Только Метрополитен.** Кливленд и Чикаго своих дампов не публикуют,
  они остаются на API.
* **Дамп — снимок на дату.** Для предметов XIV века это не проблема, но
  дата сборки индекса пишется в сам индекс и видна в отчёте.
* Индекс — ПРОДУКТ, а не исходник: он не хранится в git (те же ~20 МБ и
  та же причина, что у источников атмосферы), пересобирается командой
  `python scripts/met_catalog.py build`.
"""
import csv
import json
import os
import re
import sys
import time

CATALOG_DIR = os.environ.get(
    "MET_CATALOG_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "temp_met_catalog"))
CSV_PATH = os.path.join(CATALOG_DIR, "MetObjects.csv")
INDEX_PATH = os.path.join(CATALOG_DIR, "index.json")
DUMP_URL = ("https://media.githubusercontent.com/media/metmuseum/openaccess"
            "/master/MetObjects.csv")

# Версия правил индексации: индекс, собранный старым кодом, не должен молча
# считаться свежим — тот же принцип, что у подписи отбора.
CATALOG_VERSION = 1

_INDEX = None


def _load():
    """Индекс в память один раз за процесс. Нет файла — None, и вызывающий
    код честно падает на API-путь (fail-open, как и весь остальной код)."""
    global _INDEX
    if _INDEX is None:
        try:
            with open(INDEX_PATH, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("version") != CATALOG_VERSION:
                _INDEX = False
            else:
                _INDEX = data
        except Exception:
            _INDEX = False
    return _INDEX or None


def available():
    return _load() is not None


# Дамп у Мет лежит под git-LFS, и «сырая» ссылка на файл отдаёт не его, а
# 134-байтовый УКАЗАТЕЛЬ на него. Найдено живой попыткой собрать индекс
# 15.09, а не чтением: сборка прошла успешно, вернула код 0 и записала
# индекс из НУЛЯ предметов — «всего 2 · public domain 0 · в индексе: 0».
# Дальше `available()` честно сказал бы «каталог есть», и путь отбора
# молча остался бы без единого музейного кандидата.
#
# Настоящий файл — media.githubusercontent.com/media/... (317 МБ).
LFS_POINTER_HEAD = b"version https://git-lfs"
# Меньше этого числа строк дамп быть не может: в нём 484 956 записей.
# Порог намеренно грубый — он ловит «не тот файл», а не «файл чуть
# устарел».
MIN_DUMP_ROWS = 100_000


def _refuse_if_not_a_dump(csv_path):
    """Отказаться ГРОМКО, если на входе не каталог.

    Тихо собранный пустой индекс — худший из возможных исходов: он
    выглядит как готовый каталог и отнимает у эпизода весь музейный пул.
    """
    try:
        size = os.path.getsize(csv_path)
        with open(csv_path, "rb") as f:
            head = f.read(len(LFS_POINTER_HEAD))
    except OSError as exc:
        raise SystemExit(f"Каталог не читается: {exc}")
    if head == LFS_POINTER_HEAD:
        raise SystemExit(
            f"{csv_path} — это указатель git-LFS, а не сам дамп "
            f"({size} байт). Скачивать надо по адресу "
            "https://media.githubusercontent.com/media/metmuseum/openaccess/"
            "master/MetObjects.csv (~317 МБ).")
    if size < 50_000_000:
        raise SystemExit(
            f"{csv_path} — {size} байт, а дамп Мет весит ~317 МБ. "
            "Скорее всего скачалась страница ошибки, а не каталог.")


def build(csv_path=CSV_PATH, out_path=INDEX_PATH):
    """Собрать индекс из дампа ТЕМИ ЖЕ паспортными правилами, что и API-путь.

    Правила импортируются из museum_sources, а не переписываются здесь:
    расхождение между локальным и сетевым паспортом означало бы, что кадр
    проходит или не проходит в зависимости от того, каким путём он найден.
    """
    import museum_sources as ms
    # Индекс СОБИРАЕТСЯ ПОД ОКНО ЭПОХИ и живёт месяцами. Собрать его под
    # окно, которое никто не объявлял (константа модуля 900-1600), значит
    # получить средневековый корпус для канала любой другой ниши и узнать
    # об этом только по пустой выдаче. Живой путь (search_museums) в этом
    # случае музеи просто не спрашивает; здесь команду запускает человек,
    # поэтому достаточно назвать причину громко, а не отказать.
    if ms.foreign_culture_terms_declared() is None:
        print("  ВНИМАНИЕ: список чужих культур не объявлен в "
              "channel_profile.json (foreign_culture_terms) — индекс будет "
              "собран по списку из константы модуля (европейское "
              "Средневековье). Для канала другой ниши это отсечёт его "
              "собственную тему.")
    if ms.era_window_declared() is None:
        lo, hi = ms.era_window()
        print(f"  ВНИМАНИЕ: окно эпохи не объявлено ни в channel_profile.json "
              f"(era_from/era_to), ни авто-нишей — индекс будет собран под "
              f"{lo}-{hi} из константы модуля. Для канала другой ниши это "
              f"заведомо не тот корпус; объяви окно до сборки.")
    _refuse_if_not_a_dump(csv_path)
    csv.field_size_limit(10_000_000)
    rows, stats = [], {"total": 0, "public_domain": 0, "in_era": 0, "foreign": 0}
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            stats["total"] += 1
            if (r.get("Is Public Domain") or "").strip().lower() != "true":
                continue
            stats["public_domain"] += 1
            try:
                b = int(float(r.get("Object Begin Date") or 0))
                e = int(float(r.get("Object End Date") or 0))
            except ValueError:
                continue
            if not ms.era_overlaps(b, e):
                continue
            stats["in_era"] += 1
            if ms.culture_is_foreign(r.get("Culture"), r.get("Country"),
                                     r.get("Title"), r.get("Classification")):
                stats["foreign"] += 1
                continue
            rows.append({
                "id": r.get("Object ID"), "dept": r.get("Department"),
                "name": (r.get("Object Name") or "").strip(),
                "title": (r.get("Title") or "").strip(),
                "culture": r.get("Culture"), "b": b, "e": e,
                "medium": r.get("Medium"), "cls": r.get("Classification"),
                "tags": r.get("Tags"),
            })
    if stats["total"] < MIN_DUMP_ROWS:
        raise SystemExit(
            f"В дампе {stats['total']} записей вместо сотен тысяч — "
            "индекс не пишется. Пустой каталог, выглядящий как готовый, "
            "хуже отсутствующего.")
    payload = {"version": CATALOG_VERSION, "built_at": time.strftime("%Y-%m-%d"),
               "era": list(ms.era_window()), "stats": stats, "rows": rows}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, out_path)
    return payload


def vocabulary(department=None, min_count=2):
    """Словарь РЕАЛЬНЫХ названий предметов музея, а не наших догадок.

    Существует потому, что авторский запрос и каталог расходятся: `poleaxe`
    в каталоге ноль, а нужный предмет там лежит под именем `Halberd` (96) и
    `Partisan` (35). Синонимы для таких случаев берутся отсюда — из данных.
    """
    idx = _load()
    if not idx:
        return {}
    counts = {}
    for r in idx["rows"]:
        if department and r["dept"] != department:
            continue
        nm = (r["name"] or "").strip()
        if nm:
            counts[nm] = counts.get(nm, 0) + 1
    return {k: v for k, v in counts.items() if v >= min_count}


def _terms(query):
    """Слова запроса без служебных — по ним ищем в ПОЛЯХ, а не в описании."""
    stop = {"medieval", "european", "museum", "display", "closeup", "close",
            "up", "macro", "detail", "shot", "the", "a", "of", "and", "in",
            "on", "with", "knight", "knights"}
    return [w for w in re.findall(r"[a-z]+", (query or "").lower())
            if len(w) >= 3 and w not in stop]


def search(query, department_name=None, limit=60):
    """Точный поиск по полям каталога. Возвращает записи, УЖЕ прошедшие
    паспорт (public domain + эпоха + культура) — фильтровать после нечего.

    Ранжирование честно простое и объяснимое: совпадение в `Object Name`
    сильнее, чем в `Classification`, а то — сильнее, чем в `Tags`/`Title`.
    Никакой релевантностной магии здесь нет и не нужно: это каталог, а не
    поисковик, и вопрос «это меч?» решается полем, а не догадкой.
    """
    idx = _load()
    if not idx:
        return []
    terms = _terms(query)
    if not terms:
        return []
    pats = [re.compile(rf"\b{re.escape(t)}", re.I) for t in terms]
    scored = []
    for r in idx["rows"]:
        if department_name and r["dept"] != department_name:
            continue
        name, cls = r["name"] or "", r["cls"] or ""
        tags, title = r["tags"] or "", r["title"] or ""
        score = 0
        for p in pats:
            if p.search(name):
                score += 10
            elif p.search(cls):
                score += 4
            elif p.search(tags):
                score += 2
            elif p.search(title):
                score += 1
        if score:
            scored.append((score, r))
    scored.sort(key=lambda sr: (-sr[0], sr[1]["id"]))
    return [r for _, r in scored[:limit]]


def _cli():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        p = build()
        s = p["stats"]
        print(f"всего {s['total']} · public domain {s['public_domain']} · "
              f"в эпохе {s['in_era']} · чужих культур -{s['foreign']}")
        print(f"в индексе: {len(p['rows'])}  ->  {INDEX_PATH}")
    elif cmd == "search":
        for r in search(" ".join(sys.argv[2:])):
            print(f"  [{r['id']}] {r['name'][:30]:32} {r['b']}-{r['e']:<5} "
                  f"{(r['culture'] or '—')[:32]}")
    elif cmd == "vocab":
        dept = sys.argv[2] if len(sys.argv) > 2 else None
        for k, v in sorted(vocabulary(dept).items(), key=lambda kv: -kv[1])[:40]:
            print(f"  {v:5}  {k}")


if __name__ == "__main__":
    _cli()
