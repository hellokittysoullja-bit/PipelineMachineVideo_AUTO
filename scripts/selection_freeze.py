#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Харнесс эквивалентности отбора: заморозить прогон и воспроизвести его.

Зачем. Отбор кадра предстоит перестроить целиком (единое ядро для фото и
видео, точка коммита, объектив вместо лексикографии). Правка такого размера
безопасна ровно настолько, насколько можно доказать, что новый код выбирает
тех же победителей, что старый, там, где он обязан, — и отличается только
там, где исправлен названный дефект. Этот инструмент и есть доказательство.

КАК РАБОТАЕТ.
  record  — снимок входов эпизода + прогон НАСТОЯЩЕГО main() в режиме
            --select-only с записью всей сети (scripts/net_recorder.py).
  replay  — тот же main() на копии тех же входов, сеть — только из записи.
  verify  — replay + сравнение с записью.
  compare — сравнение двух любых прогонов одной заморозки.

ГЕРМЕТИЧНОСТЬ — не пожелание, а список закрытых источников расхождения:
  * сеть: 100% вызовов через urllib.request.urlopen, перехвачены на границе
    процесса (без правки продакшн-кода);
  * дисковые кэши поиска (музеи, Openverse, эмбеддинги): у каждого прогона
    свои пустые — иначе «воспроизведение» тайком читает кэш, прогретый чужим
    прогоном (в реальном отчёте эпизода 94 музейный поиск целиком пришёл из
    кэша: met_requests 0, search_cache_hits 8);
  * окружение: переменные, которые читает пайплайн, выводятся из КОДА
    (все os.environ.get/getenv в scripts/ + реестр флагов + ключи .env) и
    передаются дочернему процессу явно; сам .env дочерний процесс не читает
    вовсе — load_dotenv добавляет отсутствующие переменные, и правка .env
    между записью и воспроизведением молча изменила бы поведение;
  * секреты: в снимке — только отпечаток (sha256). Ключ изменился после
    записи — воспроизведение отказывается с объяснением, а не тихо
    расходится (у Pixabay ключ стоит прямо в URL, то есть в ключе записи);
  * хэш-соль: PYTHONHASHSEED фиксирован и записан; определить ДЕТЕРМИНИЗМ
    относительно соли можно отдельно, воспроизведением с другой солью;
  * модели: HF_HUB_OFFLINE — веса только из локального кэша.

ЧТО СРАВНИВАЕТСЯ — по слоту, по полю: вид, файл, sha256 самого файла,
источник, поставщик, id кандидата, relevance, chosen_by, запрос; шапка
гейтов; отчёты отбора (промахи, поглощения, спасения, вклад источников);
мультимножество сетевых обращений по ключу. Каждый слот получает ровно один
класс: СОВПАЛ / РАЗОШЁЛСЯ (с перечнем полей). Ожидаемые расхождения (когда
этап законно меняет вывод) подаются явным файлом --expect и сверяются по
причине — «где-то разошлось» без названной причины не проходит.

ПОКРЫТИЕ. Трассировщик фиксирует исполненные строки функций отбора (вместе
со всеми вложенными в них включениями, лямбдами и функциями — на Python 3.11
у включения свой кадр) и отчитывается по классам ветвей из плана (кэш-хит,
ре-пик вето, запасные ярусы видео и т.д.). Якорь класса — фрагмент
исходника, обязанный встречаться в своей функции ровно объявленное число
раз; потерянный якорь печатается как «якорь потерян», а не молча считается
непокрытым. «НЕ покрыто» — не успех: эквивалентность по такой ветке этой
заморозкой не доказана, и это печатается поимённо.

ОСТАТОЧНЫЙ РИСК, названный прямо. Гарантия стоит на допущении, что
записанные пулы представительны. Они представительны для тех прогонов, из
которых сняты: источник, сменивший корпус (новый снимок в выдаче Pexels,
дособранная полка), вне этой гарантии, и закрывается это только
перезаписью заморозки. Это единственное место, где гарантия не даётся
конструкцией.

Использование:
  python scripts/selection_freeze.py record videos/94_dagger_test
  python scripts/selection_freeze.py verify temp_selection_freeze/94_dagger_test
  python scripts/selection_freeze.py replay temp_selection_freeze/94_dagger_test --hashseed 1 --label seed1
  python scripts/selection_freeze.py compare temp_selection_freeze/94_dagger_test record seed1
"""
import argparse
import ast
import datetime
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO_ROOT, "scripts")
PIPELINE = os.path.join(SCRIPTS, "pipeline_smart.py")
FREEZE_ROOT = os.environ.get("SELECTION_FREEZE_ROOT") or os.path.join(REPO_ROOT, "temp_selection_freeze")
FREEZE_VERSION = 1
DEFAULT_HASHSEED = "0"

# Входы эпизода копируются целиком, кроме того, что порождает сам прогон и
# что не участвует в отборе: кэш рендера и готовые ролики. media_plan
# копируется весь — там и входы (паспорт, alignment, speech_plan, кэш
# семантического назначения запросов), и lock-и шотлиста.
INPUT_EXCLUDE_DIRS = ("temp_smart",)
INPUT_EXCLUDE_GLOBS = ("*.mp4", "render_log*.txt")

# Отчёты, которые пишет ОТБОР (артефакты рендера сюда не входят — режим
# --select-only их не пишет по построению).
SELECTION_REPORTS = (
    "relevance_gate_report.json", "stock_exhausted_report.json",
    "arbiter_rejected_report.json", "smart_veto_report.json",
    "absorbed_slots_report.json", "fallback_cards_report.json",
    "video_photo_rescue_report.json", "director_relevance_report.json",
    "visual_director_report.json", "source_contribution.json",
    "text_truncation_report.json",
)

SHOT_FIELDS = ("index", "section", "text", "query", "kind", "file", "file_sha256",
               "source", "provider", "candidate_id", "relevance", "chosen_by", "clip", "lock")

_SECRET_NAME_RE = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|CLIENT_ID)", re.I)
_ENV_READ_RE = re.compile(r"""(?:os\.environ\.get|os\.getenv|environ\.get)\(\s*["']([A-Z][A-Z0-9_]+)["']""")
_ENV_INDEX_RE = re.compile(r"""os\.environ\[\s*["']([A-Z][A-Z0-9_]+)["']\s*\]""")

# Классы ветвей из плана: функция + фрагмент исходника, который исполняется
# ровно тогда, когда исполнена ветка. Фрагмент обязан встречаться в своей
# функции ровно один раз — иначе класс печатается как «якорь потерян».
# Если строка ветки дословно повторяется в функции (две разные ветки пишут
# одну и ту же запись), четвёртым элементом задаётся (номер, всего): якорь —
# N-е вхождение из РОВНО стольких. Поменялось число вхождений — якорь
# потерян, а не молча сдвинут на соседнюю ветку.
BRANCH_ANCHORS = (
    ("фото: кэш-хит", "pexels_photo", "if os.path.exists(cf) and os.path.getsize(cf) > 0:"),
    ("фото: путь без анти-дубля", "pexels_photo", "            pick = candidates[0]"),
    ("фото: оценка Режиссёра", "_score_and_pick", "extra = director_score_fn("),
    ("фото: ре-пик по резкости", "pexels_photo", '"+sharp_repick"'),
    ("фото: спасение скачивания", "pexels_photo", '"+download_rescue"'),
    ("фото: ре-пик вето", "pexels_photo", '"+veto_repick"'),
    ("фото: вето отклонило всех", "pexels_photo", 'SMART_VETO_MISSES.append({"index": index, "query": query, "kind": "photo"})'),
    ("фото: сток исчерпан", "pexels_photo", "STOCK_EXHAUSTED_MISSES.append("),
    ("фото: победитель ниже порога", "pexels_photo", "RELEVANCE_GATE_MISSES.append("),
    ("фото: арбитр отказал всем", "pexels_photo", "ARBITER_REJECTED_ALL.append("),
    ("видео: кэш-хит", "pexels_video", "register_cached_media(cf, used_ids=used_ids"),
    ("видео: фильтр длины отсеял", "pexels_video", "VIDEO_TOO_SHORT_FILTERED.append("),
    ("видео: есть прошедшие гейт", "pexels_video", "        if good:"),
    ("видео: запасной — релевантный дубль", "pexels_video", "dup_fallback = (trial, v.get(\"id\"), cand_hash)"),
    ("видео: запасной — первый скачанный", "pexels_video", "plain_fallback = (trial, v.get(\"id\"), cand_hash)"),
    ("видео: арбитр отказал всем", "pexels_video", "ARBITER_REJECTED_ALL.append("),
    ("видео: ре-пик вето", "pexels_video", "veto_repicks += 1"),
    ("видео: вето отклонило всех (основной путь)", "pexels_video",
     'SMART_VETO_MISSES.append({"index": index, "query": query, "kind": "video"})', (1, 2)),
    ("видео: вето отклонило всех (запасной путь)", "pexels_video",
     'SMART_VETO_MISSES.append({"index": index, "query": query, "kind": "video"})', (2, 2)),
    ("видео: второй запасной принят", "pexels_video", "взят второй запасной"),
    ("видео: победитель ниже порога", "pexels_video", "RELEVANCE_GATE_MISSES.append("),
    ("видео: сток исчерпан", "pexels_video", "STOCK_EXHAUSTED_MISSES.append("),
    ("слот: видео заменено фото", "main", "VIDEO_RESCUED_BY_PHOTO.append("),
    ("слот: поглощён", "main", "ABSORBED_SLOTS.append("),
)
TRACED_FUNCTIONS = tuple(sorted({a[1] for a in BRANCH_ANCHORS}))


# ------------------------------------------------------------------ окружение

def _dotenv_values():
    path = os.path.join(REPO_ROOT, ".env")
    if not os.path.exists(path):
        return {}
    try:
        from dotenv import dotenv_values
        return {k: v for k, v in dotenv_values(path).items() if v is not None}
    except Exception:
        return {}


def pipeline_env_names():
    """Все переменные окружения, которые читает код пайплайна — выведено из
    исходников, а не записано списком: новая переменная попадает в снимок
    сама, без правки этого файла."""
    names = set()
    for fn in os.listdir(SCRIPTS):
        if fn.endswith(".py"):
            src = open(os.path.join(SCRIPTS, fn), encoding="utf-8", errors="replace").read()
            names |= set(_ENV_READ_RE.findall(src)) | set(_ENV_INDEX_RE.findall(src))
    sys.path.insert(0, SCRIPTS)
    try:
        import feature_flags
        names |= set(getattr(feature_flags, "FLAGS", {}))
    except Exception:
        pass
    names |= set(_dotenv_values())
    return sorted(names)


def is_secret(name):
    return bool(_SECRET_NAME_RE.search(name))


def effective_env():
    """Окружение так, как его увидит пайплайн: переменные процесса побеждают
    .env — ровно семантика load_dotenv(override=False)."""
    env = dict(_dotenv_values())
    env.update(os.environ)
    return env


def env_snapshot():
    eff = effective_env()
    snap = {}
    for name in pipeline_env_names():
        v = eff.get(name)
        if v is None:
            snap[name] = None
        elif is_secret(name):
            snap[name] = {"sha256": hashlib.sha256(v.encode()).hexdigest()}
        else:
            snap[name] = v
    return snap


def child_env(snapshot, run_dir, hashseed):
    """Окружение дочернего процесса строго из снимка. Секреты — текущие
    значения, но только если их отпечаток совпал с записанным."""
    eff = effective_env()
    env = dict(os.environ)
    mismatched = []
    for name, v in snapshot.items():
        env.pop(name, None)
        if v is None:
            continue
        if isinstance(v, dict):
            cur = eff.get(name)
            if cur is None or hashlib.sha256(cur.encode()).hexdigest() != v["sha256"]:
                mismatched.append(name)
                continue
            env[name] = cur
        else:
            env[name] = v
    if mismatched:
        raise SystemExit(
            f"ОТКАЗ: после записи изменились секреты {mismatched}. Ключ входит в адрес "
            f"запроса (у Pixabay — параметр key=), воспроизведение по другому ключу не "
            f"совпадёт с записью ни по одному адресу. Перезапиши заморозку командой record.")
    caches = os.path.join(run_dir, "caches")
    for var, sub in (("MUSEUM_CACHE_DIR", "museum"), ("OPENVERSE_CACHE_DIR", "openverse"),
                     ("EMB_CACHE_DIR", "emb")):
        env[var] = os.path.join(caches, sub)
        os.makedirs(env[var], exist_ok=True)
    env["PYTHONHASHSEED"] = str(hashseed)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


# ------------------------------------------------------------------ снимок входов

def _ignore(src_dir, names):
    out = set()
    for n in names:
        if n in INPUT_EXCLUDE_DIRS and os.path.isdir(os.path.join(src_dir, n)):
            out.add(n)
        elif any(fnmatch.fnmatch(n, g) for g in INPUT_EXCLUDE_GLOBS):
            out.add(n)
    return out


def _git_rev():
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
        dirty = subprocess.run(["git", "status", "--porcelain", "scripts"], cwd=REPO_ROOT,
                               capture_output=True, text=True, encoding="utf-8",
                               errors="replace").stdout.strip()
        return {"head": r.stdout.strip(), "scripts_dirty": bool(dirty)}
    except Exception:
        return {"head": None, "scripts_dirty": None}


def _stack_versions():
    out = {"python": sys.version.split()[0]}
    for mod in ("torch", "transformers", "onnxruntime", "numpy", "PIL"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "?")
        except Exception:
            out[mod] = None
    return out


# ------------------------------------------------------------------ запуск

def prune_run_media(sandbox):
    """Удалить то, что прогон СКАЧАЛ (кэш отбора в песочнице эпизода).
    Вызывается строго ПОСЛЕ collect(): хэш каждого файла-победителя уже в
    result.json, а сами байты лежат в записи сети и воспроизводятся из неё
    бит-в-бит. Хранить их ещё и в каждом прогоне — удвоение диска без
    новой информации (эпизод из 9 слотов: 63 МБ на прогон). Входы эпизода
    и отчёты (media_plan) не трогаются."""
    freed = 0
    for d in INPUT_EXCLUDE_DIRS:
        path = os.path.join(sandbox, d)
        if os.path.isdir(path):
            for root, _dirs, files in os.walk(path):
                for f in files:
                    try:
                        freed += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
            shutil.rmtree(path)
    return freed


def run_pipeline(freeze, mode, label, hashseed, keep_media=False):
    meta = json.load(open(os.path.join(freeze, "meta.json"), encoding="utf-8"))
    run_dir = os.path.join(freeze, "runs", label)
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir)
    sandbox = os.path.join(run_dir, "episode")
    shutil.copytree(os.path.join(freeze, "input"), sandbox)
    env = child_env(meta["env"], run_dir, hashseed)
    cmd = [sys.executable, os.path.abspath(__file__), "_child", mode,
           os.path.join(freeze, "net"), sandbox, run_dir]
    with open(os.path.join(run_dir, "stdout.log"), "w", encoding="utf-8") as out, \
            open(os.path.join(run_dir, "stderr.log"), "w", encoding="utf-8") as err:
        started = datetime.datetime.now(datetime.timezone.utc)
        rc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=out, stderr=err).returncode
        took = (datetime.datetime.now(datetime.timezone.utc) - started).total_seconds()
    result = collect(sandbox, run_dir)
    if not keep_media:
        prune_run_media(sandbox)
    result["returncode"] = rc
    result["hashseed"] = str(hashseed)
    result["mode"] = mode
    result["seconds"] = round(took, 1)
    with open(os.path.join(run_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1, sort_keys=True)
    return result


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_paths(obj, sandbox):
    raw = json.dumps(obj, ensure_ascii=False)
    for p in {os.path.abspath(sandbox), os.path.realpath(sandbox)}:
        raw = raw.replace(p, "<EP>")
    return json.loads(raw)


def collect(sandbox, run_dir):
    mp = os.path.join(sandbox, "media_plan")
    shots, gates = [], {}
    sl_path = os.path.join(mp, "shotlist.json")
    if os.path.exists(sl_path):
        data = json.load(open(sl_path, encoding="utf-8"))
        gates = data.get("gates", {})
        for s in data.get("shots", []):
            row = {k: s.get(k) for k in SHOT_FIELDS if k != "file_sha256"}
            f = s.get("file")
            fp = os.path.join(sandbox, f) if f else None
            row["file_sha256"] = _sha256_file(fp) if fp and os.path.exists(fp) else None
            shots.append(row)
    reports = {}
    for name in SELECTION_REPORTS:
        p = os.path.join(mp, name)
        if os.path.exists(p):
            reports[name] = json.load(open(p, encoding="utf-8"))
    net = {}
    np_ = os.path.join(run_dir, "net_summary.json")
    if os.path.exists(np_):
        net = json.load(open(np_, encoding="utf-8"))
    cov = {}
    cp = os.path.join(run_dir, "coverage.json")
    if os.path.exists(cp):
        cov = json.load(open(cp, encoding="utf-8"))
    return _normalize_paths({"shots": shots, "gates": gates, "reports": reports,
                             "net": net, "coverage": cov}, sandbox)


# ------------------------------------------------------------------ дочерний процесс

def nested_code_objects(code):
    """Код функции и ВСЕ вложенные в неё объекты кода: включения,
    генераторы, лямбды, вложенные функции. На Python 3.11 списковое
    включение исполняется в собственном кадре со своим f_code — трассировка
    только кода самой функции не видит строк внутри включения, и ветка,
    которая реально исполнилась, числилась бы «НЕ покрыто» (так и было:
    фильтр длины отсеял 92 кандидата и значился непокрытым)."""
    out, stack = [], [code]
    while stack:
        c = stack.pop()
        out.append(c)
        stack.extend(k for k in c.co_consts if isinstance(k, type(code)))
    return out


def _install_tracer(pipeline_smart):
    """Исполненные строки функций отбора. Трассируются только их кадры:
    на остальных вызовах трассировщик возвращает None, цена — одна проверка
    на вызов функции."""
    targets = {}
    for name in TRACED_FUNCTIONS:
        fn = getattr(pipeline_smart, name, None)
        if fn is not None:
            for code in nested_code_objects(fn.__code__):
                targets[code] = name
    hits = {name: set() for name in targets.values()}

    def local(frame, event, _arg):
        if event == "line":
            hits[targets[frame.f_code]].add(frame.f_lineno)
        return local

    def glob(frame, event, _arg):
        if event == "call" and frame.f_code in targets:
            return local
        return None

    import threading
    sys.settrace(glob)
    threading.settrace(glob)
    return hits


def _anchor_lines(src_lines, tree):
    """Класс ветви -> номер строки якоря (или причина, почему якоря нет)."""
    # Только верхний уровень модуля — ровно то, что трассировщик берёт через
    # getattr(модуль, имя): одноимённая вложенная функция не подменит якорь.
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    out = {}
    for anchor in BRANCH_ANCHORS:
        cls, fname, frag = anchor[:3]
        nth, total = anchor[3] if len(anchor) > 3 else (1, 1)
        node = funcs.get(fname)
        if node is None:
            out[cls] = {"error": f"функции {fname} нет"}
            continue
        # Фрагмент с ведущими пробелами сравнивается с НАЧАЛОМ строки: тогда
        # отступ значим, и ветка на одном уровне не путается с одноимённой
        # строкой глубже (pick = candidates[0] встречается в pexels_photo дважды).
        hit = ((lambda ln: ln.startswith(frag)) if frag.startswith(" ")
               else (lambda ln: frag in ln))
        lines = [i for i in range(node.lineno, node.end_lineno + 1) if hit(src_lines[i - 1])]
        if len(lines) != total:
            out[cls] = {"error": f"якорь потерян: {len(lines)} совпадений в {fname}, "
                                 f"ожидалось {total}"}
        else:
            out[cls] = {"function": fname, "line": lines[nth - 1]}
    return out


def child(mode, net_dir, sandbox, run_dir):
    sys.path.insert(0, SCRIPTS)
    import dotenv
    dotenv.load_dotenv = lambda *a, **k: False   # окружение целиком передал родитель
    import net_recorder
    rec = net_recorder.NetRecorder(net_dir, mode).install()
    sys.argv = [PIPELINE, sandbox, "--select-only"]
    rc = 1
    hits = {}
    try:
        import pipeline_smart
        hits = _install_tracer(pipeline_smart)
        try:
            rc = pipeline_smart.main()
        except SystemExit as e:
            rc = e.code if isinstance(e.code, int) else 1
    finally:
        sys.settrace(None)
        with open(os.path.join(run_dir, "net_summary.json"), "w", encoding="utf-8") as f:
            json.dump(rec.summary(), f, ensure_ascii=False, indent=1)
        rec.uninstall()
        src = open(PIPELINE, encoding="utf-8").read()
        anchors = _anchor_lines(src.split("\n"), ast.parse(src))
        cov = {}
        for cls, a in anchors.items():
            if "error" in a:
                cov[cls] = {"state": "якорь потерян", "detail": a["error"]}
            else:
                hit = a["line"] in hits.get(a["function"], set())
                cov[cls] = {"state": "покрыто" if hit else "НЕ покрыто", "line": a["line"]}
        with open(os.path.join(run_dir, "coverage.json"), "w", encoding="utf-8") as f:
            json.dump({"branches": cov,
                       "lines_executed": {k: len(v) for k, v in hits.items()}},
                      f, ensure_ascii=False, indent=1)
    sys.exit(rc if isinstance(rc, int) else 1)


# ------------------------------------------------------------------ сравнение

def _load_result(freeze, label):
    p = os.path.join(freeze, "runs", label, "result.json")
    if not os.path.exists(p):
        raise SystemExit(f"нет результата прогона {label!r}: {p}")
    return json.load(open(p, encoding="utf-8"))


def compare(a, b, expect=None):
    """Классификация по слоту. expect: {index: причина} — слоты, которые
    ОБЯЗАНЫ разойтись; разошедшийся слот без названной причины — провал,
    совпавший слот из expect — тоже провал (правка не сработала)."""
    expect = {int(k): v for k, v in (expect or {}).items()}
    sa = {s["index"]: s for s in a.get("shots", [])}
    sb = {s["index"]: s for s in b.get("shots", [])}
    slots = []
    for i in sorted(set(sa) | set(sb)):
        x, y = sa.get(i), sb.get(i)
        if x is None or y is None:
            diff = ["слот отсутствует в одном из прогонов"]
        else:
            diff = [k for k in SHOT_FIELDS if x.get(k) != y.get(k)]
        if not diff:
            klass = "НЕОЖИДАННО СОВПАЛ" if i in expect else "СОВПАЛ"
        else:
            klass = "ОЖИДАЕМО РАЗОШЁЛСЯ" if i in expect else "РАЗОШЁЛСЯ"
        slots.append({"index": i, "class": klass, "fields": diff,
                      "a": {k: (x or {}).get(k) for k in diff},
                      "b": {k: (y or {}).get(k) for k in diff},
                      "expected_reason": expect.get(i)})
    reports = {}
    for name in sorted(set(a.get("reports", {})) | set(b.get("reports", {}))):
        ra, rb = a.get("reports", {}).get(name), b.get("reports", {}).get(name)
        if ra != rb:
            reports[name] = {"a": ra, "b": rb}
    ca = a.get("net", {}).get("calls_by_key", {})
    cb = b.get("net", {}).get("calls_by_key", {})
    net_diff = {k: [ca.get(k, 0), cb.get(k, 0)] for k in sorted(set(ca) | set(cb))
                if ca.get(k, 0) != cb.get(k, 0)}
    divergences = b.get("net", {}).get("divergences", [])
    bad = [s for s in slots if s["class"] in ("РАЗОШЁЛСЯ", "НЕОЖИДАННО СОВПАЛ")]
    ok = (not bad and not reports and not net_diff and not divergences
          and a.get("gates") == b.get("gates")
          and a.get("returncode") == b.get("returncode"))
    return {"ok": ok, "slots": slots, "reports_differ": reports,
            "gates_differ": a.get("gates") != b.get("gates"),
            "returncode": [a.get("returncode"), b.get("returncode")],
            "net_calls_differ": net_diff, "replay_divergences": divergences}


def print_report(rep, a, b):
    print(f"\nСЛОТЫ ({len(rep['slots'])}):")
    for s in rep["slots"]:
        extra = ""
        if s["fields"]:
            extra = "  поля: " + ", ".join(f"{k}: {s['a'].get(k)!r} -> {s['b'].get(k)!r}"
                                          for k in s["fields"])
        if s["expected_reason"]:
            extra += f"  [причина: {s['expected_reason']}]"
        print(f"  #{s['index'] + 1:<3} {s['class']:<18}{extra}")
    print(f"\nшапка гейтов: {'РАЗОШЛАСЬ' if rep['gates_differ'] else 'совпала'}")
    print(f"коды возврата: {rep['returncode'][0]} / {rep['returncode'][1]}")
    print(f"отчёты отбора: {'разошлись: ' + ', '.join(rep['reports_differ']) if rep['reports_differ'] else 'совпали'}")
    na, nb = a.get("net", {}), b.get("net", {})
    print(f"сеть: {na.get('distinct_requests')} адресов / {na.get('total_calls')} обращений  против  "
          f"{nb.get('distinct_requests')} / {nb.get('total_calls')}; "
          f"расходящихся ключей {len(rep['net_calls_differ'])}; "
          f"запросов вне записи {len(rep['replay_divergences'])}")
    for d in rep["replay_divergences"][:10]:
        print(f"    вне записи: {d['method']} {d['url']} (обращение #{d['seq'] + 1})")
    cov = b.get("coverage", {}).get("branches", {})
    if cov:
        print("\nПОКРЫТИЕ ВЕТВЕЙ ОТБОРА:")
        for cls, v in cov.items():
            print(f"  {v['state']:<14} {cls}" + (f"  ({v['detail']})" if "detail" in v else ""))
    print(f"\nИТОГ: {'ЭКВИВАЛЕНТНО' if rep['ok'] else 'НЕ ЭКВИВАЛЕНТНО'}")


# ------------------------------------------------------------------ команды

def cmd_record(args):
    episode = os.path.abspath(args.episode)
    if not os.path.exists(os.path.join(episode, "script.txt")):
        raise SystemExit(f"не эпизод (нет script.txt): {episode}")
    freeze = os.path.abspath(args.out or os.path.join(FREEZE_ROOT, os.path.basename(episode.rstrip("/"))))
    if os.path.exists(freeze):
        if not args.force:
            raise SystemExit(f"заморозка уже есть: {freeze} (перезаписать — --force)")
        shutil.rmtree(freeze)
    os.makedirs(freeze)
    shutil.copytree(episode, os.path.join(freeze, "input"), ignore=_ignore)
    meta = {"freeze_version": FREEZE_VERSION, "episode": os.path.basename(episode.rstrip("/")),
            "source_path": episode, "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "git": _git_rev(), "stack": _stack_versions(), "hashseed": str(args.hashseed),
            "env": env_snapshot()}
    with open(os.path.join(freeze, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(f"запись: {meta['episode']} -> {freeze}")
    res = run_pipeline(freeze, "record", "record", args.hashseed, args.keep_media)
    shots_with = sum(1 for s in res["shots"] if s.get("file"))
    print(f"записано за {res['seconds']}с: код {res['returncode']}, слотов {len(res['shots'])}, "
          f"с кадром {shots_with}, сеть {res['net'].get('distinct_requests')} адресов / "
          f"{res['net'].get('total_calls')} обращений")
    return 0 if res["returncode"] in (0, 2) else 1


def cmd_replay(args):
    freeze = os.path.abspath(args.freeze)
    label = args.label or f"replay-seed{args.hashseed}"
    res = run_pipeline(freeze, "replay", label, args.hashseed, args.keep_media)
    print(f"воспроизведено за {res['seconds']}с: код {res['returncode']}, "
          f"вне записи {len(res['net'].get('divergences', []))}")
    return 0


def _expect(path):
    if not path:
        return None
    return json.load(open(path, encoding="utf-8"))


def cmd_compare(args):
    freeze = os.path.abspath(args.freeze)
    a, b = _load_result(freeze, args.a), _load_result(freeze, args.b)
    rep = compare(a, b, _expect(args.expect))
    print_report(rep, a, b)
    with open(os.path.join(freeze, "runs", args.b, f"compare_vs_{args.a}.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    return 0 if rep["ok"] else 1


def cmd_verify(args):
    freeze = os.path.abspath(args.freeze)
    meta = json.load(open(os.path.join(freeze, "meta.json"), encoding="utf-8"))
    seed = args.hashseed if args.hashseed is not None else meta["hashseed"]
    label = args.label or f"verify-seed{seed}"
    run_pipeline(freeze, "replay", label, seed, args.keep_media)
    a, b = _load_result(freeze, "record"), _load_result(freeze, label)
    rep = compare(a, b, _expect(args.expect))
    print_report(rep, a, b)
    with open(os.path.join(freeze, "runs", label, "compare_vs_record.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    return 0 if rep["ok"] else 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_child":
        return child(*argv[1:5])
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("episode")
    r.add_argument("--out")
    r.add_argument("--force", action="store_true")
    r.add_argument("--hashseed", default=DEFAULT_HASHSEED)
    r.set_defaults(fn=cmd_record)
    rp = sub.add_parser("replay")
    rp.add_argument("freeze")
    rp.add_argument("--hashseed", default=DEFAULT_HASHSEED)
    rp.add_argument("--label")
    rp.set_defaults(fn=cmd_replay)
    v = sub.add_parser("verify")
    v.add_argument("freeze")
    v.add_argument("--hashseed")
    v.add_argument("--label")
    v.add_argument("--expect", help="JSON {индекс слота: причина} — слоты, обязанные разойтись")
    v.set_defaults(fn=cmd_verify)
    for sp in (r, rp, v):
        sp.add_argument("--keep-media", action="store_true",
                        help="оставить скачанные прогоном файлы (для глазной проверки); "
                             "по умолчанию удаляются после подсчёта хэшей")
    c = sub.add_parser("compare")
    c.add_argument("freeze")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--expect")
    c.set_defaults(fn=cmd_compare)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
