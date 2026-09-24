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
import threading

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
    "text_truncation_report.json", "run_journal.jsonl", "shot_judge_report.json",
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
# одну и ту же запись), у фрагмента задаётся (номер, всего): якорь — N-е
# вхождение из РОВНО стольких. Поменялось число вхождений — якорь потерян, а
# не молча сдвинут на соседнюю ветку.
#
# Одна ветка — несколько ФОРМ: харнесс судит и старую ревизию (добытчик
# pexels_photo, вердикт SMART_VETO_MISSES.append), и новую (_select_photo,
# record_verdict). Разрешается ровно одна форма; если в исходнике нашлись
# две — это неоднозначность, и класс тоже «потерян», а не выбран наугад.
_P, _V = ("PhotoAdapter.choose", "_select_photo", "pexels_photo"), ("_select_video", "pexels_video")
# Видео в общем ядре (этап 3): те же классы ветвей, где они пережили
# переписывание, — в методах VideoAdapter.
_VA = ("VideoAdapter._pick",)


def _forms(fns, *frags, occ=(1, 1)):
    return tuple((fn, frag, occ) for fn in fns for frag in frags)


BRANCH_ANCHORS = (
    ("фото: кэш-хит", _forms(("_select_photo", "pexels_photo"),
                             "if os.path.exists(cf) and os.path.getsize(cf) > 0:")
     + _forms(("PhotoAdapter.cache_hit",),
              "used_ids, used_hashes = request.used_photo_ids, request.used_hashes")),
    ("фото: путь без анти-дубля", _forms(_P, "            pick = candidates[0]")),
    ("фото: оценка Режиссёра", _forms(("_score_and_pick",), "extra = director_score_fn(")),
    ("фото: ре-пик по резкости", _forms(_P, '"+sharp_repick"')),
    ("фото: спасение скачивания", _forms(_P, '"+download_rescue"')),
    ("фото: ре-пик вето", _forms(_P, '"+veto_repick"')),
    ("фото: вето отклонило всех", _forms(
        _P, 'record_verdict("smart_veto", {"index": index, "query": query, "kind": "photo"})',
        'SMART_VETO_MISSES.append({"index": index, "query": query, "kind": "photo"})')),
    ("фото: сток исчерпан", _forms(_P, 'record_verdict("stock", {', "STOCK_EXHAUSTED_MISSES.append(")),
    ("фото: ни один кандидат не проверен", _forms(_P, "ни один кандидат не дошёл до проверки")),
    ("фото: победитель ниже порога", _forms(_P, 'record_verdict("relevance", {',
                                             "RELEVANCE_GATE_MISSES.append(")),
    ("фото: арбитр отказал всем", _forms(_P, 'record_verdict("arbiter", {', "ARBITER_REJECTED_ALL.append(")),
    ("фото: судья не одобрил победителя", _forms(_P, 'record_verdict("judge", {')),
    ("видео: кэш-хит", _forms(_V, "register_cached_media(cf, used_ids=used_ids")
     + _forms(("VideoAdapter.cache_hit",), "register_cached_media(cf, used_ids=request.used_video_ids")),
    ("видео: кэш-хит отвергнут как повтор", _forms(("VideoAdapter.cache_hit",),
                                                   "            return None")),
    ("видео: фильтр длины отсеял", _forms(_V + ("VideoAdapter.filter_pool",),
                                          "VIDEO_TOO_SHORT_FILTERED.append(")),
    ("видео: оценено по превью", _forms(("VideoAdapter.choose",), "candidates_info.append({")),
    ("видео: превью не скачалось", _forms(("VideoAdapter.choose",),
                                          '_source_bump(candidate_channel(v), "download_errors")')),
    ("видео: ре-пик после скачивания", _forms(_VA, "repicks += 1")),
    ("видео: судья не одобрил победителя", _forms(_VA, 'record_verdict("judge", {')),
    ("видео: есть прошедшие гейт", _forms(_V, "        if good:")),
    ("видео: запасной — релевантный дубль", _forms(_V, 'dup_fallback = (trial, v.get("id"), cand_hash)')),
    ("видео: запасной — первый скачанный", _forms(_V, 'plain_fallback = (trial, v.get("id"), cand_hash)')),
    ("видео: арбитр отказал всем", _forms(_V + _VA, 'record_verdict("arbiter", {', "ARBITER_REJECTED_ALL.append(")),
    ("видео: ре-пик вето", _forms(_V, "veto_repicks += 1")),
    ("видео: вето отклонило всех (основной путь)", _forms(
        _V, 'record_verdict("smart_veto", {"index": index, "query": query, "kind": "video"})',
        'SMART_VETO_MISSES.append({"index": index, "query": query, "kind": "video"})', occ=(1, 2))
     + _forms(_VA, 'record_verdict("smart_veto", {"index": index, "query": query, "kind": "video",')),
    ("видео: вето отклонило всех (запасной путь)", _forms(
        _V, 'record_verdict("smart_veto", {"index": index, "query": query, "kind": "video"})',
        'SMART_VETO_MISSES.append({"index": index, "query": query, "kind": "video"})', occ=(2, 2))),
    ("видео: второй запасной принят", _forms(_V, "взят второй запасной")),
    ("видео: победитель ниже порога", _forms(_V + _VA, 'record_verdict("relevance", {',
                                              "RELEVANCE_GATE_MISSES.append(")),
    ("видео: сток исчерпан", _forms(_V + _VA, 'record_verdict("stock", {', "STOCK_EXHAUSTED_MISSES.append(")),
    ("слот: видео заменено фото", _forms(("main",), "VIDEO_RESCUED_BY_PHOTO.append(")),
    ("слот: поглощён", _forms(("main",), "ABSORBED_SLOTS.append(")),
)
TRACED_FUNCTIONS = tuple(sorted({fn for _cls, forms in BRANCH_ANCHORS for fn, _f, _o in forms}))

# Ветви, существующие только в одной ревизии видео-отбора. Переписывание
# видео (этап 3) убрало ветви прежнего добытчика (запасные ярусы, отдельный
# цикл вето) и принесло свои (оценка по превью, судья). Класс чужой ревизии —
# «ветви нет в этой ревизии», а не «якорь потерян»: потерянный якорь
# означает, что правка кода незаметно сдвинула ветку, и роняет тест.
# Ревизия узнаётся по коду, а не объявляется: есть VideoAdapter._pick — ядро.
ANCHOR_REVISION = {
    "видео: есть прошедшие гейт": "legacy",
    "видео: запасной — релевантный дубль": "legacy",
    "видео: запасной — первый скачанный": "legacy",
    "видео: ре-пик вето": "legacy",
    "видео: вето отклонило всех (запасной путь)": "legacy",
    "видео: второй запасной принят": "legacy",
    "видео: кэш-хит отвергнут как повтор": "engine",
    "видео: оценено по превью": "engine",
    "видео: превью не скачалось": "engine",
    "видео: ре-пик после скачивания": "engine",
    "видео: судья не одобрил победителя": "engine",
}


def video_revision(funcs):
    return "engine" if "VideoAdapter._pick" in funcs else "legacy"


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
                     ("COMMONS_CACHE_DIR", "commons"), ("EMB_CACHE_DIR", "emb")):
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


def _git_rev(root=REPO_ROOT):
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
        dirty = subprocess.run(["git", "status", "--porcelain", "scripts"], cwd=root,
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


def run_pipeline(freeze, mode, label, hashseed, keep_media=False, pipeline=None,
                 live_fallback=False, overlay_from=None):
    meta = json.load(open(os.path.join(freeze, "meta.json"), encoding="utf-8"))
    run_dir = os.path.join(freeze, "runs", label)
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir)
    sandbox = os.path.join(run_dir, "episode")
    shutil.copytree(os.path.join(freeze, "input"), sandbox)
    env = child_env(meta["env"], run_dir, hashseed)
    pipeline = os.path.abspath(pipeline or PIPELINE)
    overlay = os.path.join(run_dir, "net_overlay") if live_fallback else ""
    extra = os.path.join(freeze, "runs", overlay_from, "net_overlay") if overlay_from else ""
    if extra and not os.path.exists(os.path.join(extra, "index.jsonl")):
        raise SystemExit(f"у прогона {overlay_from!r} нет слоя живых запросов: {extra}")
    cmd = [sys.executable, os.path.abspath(__file__), "_child", mode,
           os.path.join(freeze, "net"), sandbox, run_dir, pipeline, overlay, extra]
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
    result["pipeline"] = pipeline
    result["live_fallback"] = bool(live_fallback)
    result["net_overlay_from"] = overlay_from
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
            with open(p, encoding="utf-8") as f:
                reports[name] = ([json.loads(line) for line in f if line.strip()]
                                 if name.endswith(".jsonl") else json.load(f))
    net = {}
    np_ = os.path.join(run_dir, "net_summary.json")
    if os.path.exists(np_):
        net = json.load(open(np_, encoding="utf-8"))
    inputs = {}
    ip = os.path.join(run_dir, "slot_inputs.json")
    if os.path.exists(ip):
        inputs = json.load(open(ip, encoding="utf-8"))
    attempts = {}
    ap = os.path.join(run_dir, "slot_attempts.json")
    if os.path.exists(ap):
        attempts = json.load(open(ap, encoding="utf-8"))
    cov = {}
    cp = os.path.join(run_dir, "coverage.json")
    if os.path.exists(cp):
        cov = json.load(open(cp, encoding="utf-8"))
    return _normalize_paths({"shots": shots, "gates": gates, "reports": reports,
                             "net": net, "coverage": cov, "slot_inputs": inputs,
                             "slot_attempts": attempts}, sandbox)


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


def slot_loop_range(tree):
    """Строки слотового цикла main(): тот for, внутри которого слот
    объявляется разрешённым (RESOLVED_SLOTS_THIS_RUN.add(i)). Только внутри
    него номер слота — это `i` цикла; до и после сеть слоту не принадлежит."""
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    for node in ast.walk(main):
        if not isinstance(node, ast.For):
            continue
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr == "add" and isinstance(sub.func.value, ast.Name)
                    and sub.func.value.id == "RESOLVED_SLOTS_THIS_RUN"):
                inner = [f for f in ast.walk(node) if isinstance(f, ast.For) and f is not node
                         and f.lineno <= sub.lineno <= f.end_lineno]
                if not inner:
                    return node.lineno, node.end_lineno, node.body[0].lineno
    raise ValueError("слотовый цикл main() не найден")


# Общее состояние, с которым слот входит в отбор: всё, через что один слот
# влияет на следующий. Два прогона с одинаковым входом слота и без утечки в
# нём самом обязаны выбрать одно и то же.
SLOT_INPUT_NAMES = ("used_photo_ids", "used_video_ids", "used_photo_hashes",
                    "recent_shot_sizes", "recent_media_types", "recent_semantic_tags",
                    "luma_ema", "_carry_sec", "stat_carry", "use_pexels", "use_local")


def _digest(value):
    if isinstance(value, (set, frozenset)):
        value = sorted(value, key=repr)
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, default=repr)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _resolve(module, dotted):
    obj = module
    for part in dotted.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


# Поля запроса слота, которые читает каждый вид отбора — по сигнатурам
# добытчиков этапа 1 (их аргументы и есть эти поля под прежними именами).
ATTEMPT_FIELDS = {
    "photo": ("query", "index", "used_photo_ids", "used_hashes", "recent_sizes", "target_luma",
              "director_score_fn", "director_assist", "director_report", "extra_queries",
              "text_key", "arbiter_text", "is_opening", "shot_brief", "block_text"),
    "video": ("query", "index", "used_video_ids", "used_hashes", "action_qualifier",
              "extra_queries", "video_score_fn", "text_key", "arbiter_text", "is_opening",
              "recent_sizes", "slot_dur", "shot_brief",
              # читаются с этапа 3 (судья видит фразу; Режиссёр решает, как у фото)
              "block_text", "director_assist"),
}


def _field_digest(name, value):
    if callable(value):
        value = "<callable>"      # адрес объекта в процессе смыслом не обладает
    elif name == "extra_queries":
        value = list(value or [])  # None и пустой кортеж — одно и то же «нет»
    return _digest(value)


def attempt_fields_from_call(entry, loc):
    """(вид медиа, {поле запроса: отпечаток}) по кадру вызова отбора."""
    if entry == "select_media":
        req, kind = loc.get("request"), loc.get("kind")
        values = {f: getattr(req, f, None) for f in ATTEMPT_FIELDS.get(kind, ())}
    else:
        kind = "photo" if entry == "_select_photo" else "video"
        alias = {"used_ids": "used_photo_ids" if kind == "photo" else "used_video_ids",
                 "is_opening_shot": "is_opening", "sentence_score_fn": "video_score_fn"}
        values = {alias.get(k, k): v for k, v in loc.items()}
        values = {f: values.get(f) for f in ATTEMPT_FIELDS[kind]}
    return kind, {f: _field_digest(f, v) for f, v in values.items()}


def _install_tracer(pipeline_smart, slot_range=None):
    """Исполненные строки функций отбора. Трассируются только их кадры:
    на остальных вызовах трассировщик возвращает None, цена — одна проверка
    на вызов функции. Попутно — номер текущего слота (для меток сети):
    строка main() внутри слотового цикла -> его `i`, вне цикла -> None."""
    targets = {}
    for name in TRACED_FUNCTIONS:
        fn = _resolve(pipeline_smart, name)
        if fn is not None:
            for code in nested_code_objects(fn.__code__):
                targets[code] = name
    # Вход каждой попытки отбора: у кода с запросом слота — select_media,
    # у кода этапа 1 — сами добытчики (их аргументы приводятся к именам
    # полей запроса, см. attempt_fields_from_call).
    entry_names = (("select_media",) if hasattr(pipeline_smart, "select_media")
                   else ("_select_photo", "_select_video"))
    entries = {}
    for name in entry_names:
        fn = getattr(pipeline_smart, name, None)
        if fn is not None:
            entries[fn.__code__] = name
    attempts = {}
    hits = {name: set() for name in targets.values()}
    main_fn = getattr(pipeline_smart, "main", None)
    main_code = main_fn.__code__ if main_fn is not None else None
    slot = {"i": None}
    inputs = {}
    lo, hi, first = slot_range if slot_range is not None else (None, None, None)

    def local(frame, event, _arg):
        if event == "line":
            hits[targets[frame.f_code]].add(frame.f_lineno)
            if frame.f_code is main_code and lo is not None:
                inside = lo <= frame.f_lineno <= hi
                loc = frame.f_locals if inside else None
                cur = loc.get("i") if inside else None
                slot["i"] = cur if isinstance(cur, int) else None
                if frame.f_lineno == first and isinstance(cur, int):
                    inputs[cur] = {n: _digest(loc.get(n)) for n in SLOT_INPUT_NAMES}
        return local

    def glob(frame, event, _arg):
        if event != "call":
            return None
        entry = entries.get(frame.f_code)
        if entry is not None and slot["i"] is not None:
            kind, fields = attempt_fields_from_call(entry, frame.f_locals)
            attempts.setdefault(slot["i"], []).append({"kind": kind, "fields": fields})
        if frame.f_code in targets:
            return local
        return None

    import threading
    sys.settrace(glob)
    threading.settrace(glob)
    hits["__slot__"] = slot
    hits["__inputs__"] = inputs
    hits["__attempts__"] = attempts
    return hits


def _anchor_lines(src_lines, tree):
    """Класс ветви -> номер строки якоря (или причина, почему якоря нет)."""
    # Только верхний уровень модуля — ровно то, что трассировщик берёт через
    # getattr(модуль, имя): одноимённая вложенная функция не подменит якорь.
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
        for m in cls.body:
            if isinstance(m, ast.FunctionDef):
                funcs[f"{cls.name}.{m.name}"] = m
    out = {}
    revision = video_revision(funcs)
    for cls, forms in BRANCH_ANCHORS:
        if ANCHOR_REVISION.get(cls, revision) != revision:
            out[cls] = {"absent": f"ветви нет в этой ревизии видео-отбора ({revision})"}
            continue
        resolved, notes = [], []
        for fname, frag, (nth, total) in forms:
            node = funcs.get(fname)
            if node is None:
                continue
            # Фрагмент с ведущими пробелами сравнивается с НАЧАЛОМ строки: тогда
            # отступ значим, и ветка на одном уровне не путается с одноимённой
            # строкой глубже (pick = candidates[0] встречается в добытчике дважды).
            hit = ((lambda ln, f=frag: ln.startswith(f)) if frag.startswith(" ")
                   else (lambda ln, f=frag: f in ln))
            lines = [i for i in range(node.lineno, node.end_lineno + 1) if hit(src_lines[i - 1])]
            if len(lines) == total:
                resolved.append({"function": fname, "line": lines[nth - 1]})
            elif lines:
                notes.append(f"{len(lines)} совпадений в {fname}, ожидалось {total}")
        if len(resolved) == 1:
            out[cls] = resolved[0]
        elif resolved:
            out[cls] = {"error": f"неоднозначно: {len(resolved)} форм разрешились"}
        else:
            out[cls] = {"error": "якорь потерян" + (": " + "; ".join(notes) if notes else "")}
    return out


def coverage_from(source, hits):
    """Классы ветвей по исполненным строкам (hits: функция -> строки)."""
    anchors = _anchor_lines(source.split("\n"), ast.parse(source))
    cov = {}
    for cls, a in anchors.items():
        if "error" in a:
            cov[cls] = {"state": "якорь потерян", "detail": a["error"]}
        elif "absent" in a:
            cov[cls] = {"state": "нет в ревизии", "detail": a["absent"]}
        else:
            hit = a["line"] in set(hits.get(a["function"], ()))
            cov[cls] = {"state": "покрыто" if hit else "НЕ покрыто", "line": a["line"]}
    return cov


def recompute_coverage(freeze, label):
    """Пересчитать покрытие прогона по его сырым строкам — например, после
    того как якоря научились узнавать форму ветки другой ревизии. Исходник
    берётся по пути из результата и обязан совпасть по хэшу с исполненным:
    иначе номера строк относятся к другому тексту, и пересчёт был бы ложью."""
    run_dir = os.path.join(freeze, "runs", label)
    res_path = os.path.join(run_dir, "result.json")
    res = json.load(open(res_path, encoding="utf-8"))
    cov_path = os.path.join(run_dir, "coverage.json")
    raw = json.load(open(cov_path, encoding="utf-8"))
    src = open(res["pipeline"], encoding="utf-8").read()
    if hashlib.sha256(src.encode("utf-8")).hexdigest() != raw.get("pipeline_sha256"):
        raise SystemExit(f"{res['pipeline']} не совпадает с исполненным кодом прогона {label!r}")
    raw["branches"] = coverage_from(src, raw.get("lines") or {})
    with open(cov_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=1)
    res["coverage"] = raw
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, sort_keys=True)
    return raw["branches"]


def load_module_from_source(name, path, source):
    """Импорт модуля из заданного текста (а не из файла на диске): модуль
    регистрируется в sys.modules до исполнения — как при обычном импорте,
    поэтому модули, импортирующие его в ответ, получают этот же объект."""
    import linecache
    import types
    # inspect.getsource() (им пайплайн строит подписи кэша) читает текст
    # через linecache, то есть с ДИСКА. Запись без mtime linecache не
    # перепроверяет — getsource видит ровно исполняемый снимок.
    linecache.cache[path] = (len(source), None, source.splitlines(True), path)
    mod = types.ModuleType(name)
    mod.__file__ = path
    sys.modules[name] = mod
    try:
        exec(compile(source, path, "exec"), mod.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


POOL_FIELDS = ("id", "_origin_query")
PENDING_REMOVED = {}


ABLATIONS = {
    # запрос уходит в источник как написан: без уточнителя культуры и якоря эпохи
    "nolengthen": lambda ps: setattr(ps, "disambiguate_search_query", lambda q: q),
    # жанровый список запретов канала не действует (запрещённые id — действуют)
    "noblocklist": lambda ps: setattr(ps, "content_blocklist_effective", lambda: []),
}


def apply_ablation(pipeline_smart, spec):
    """Замер этапа 0: выключить слой ИСПЫТУЕМОГО кода, не правя его файл.
    spec — имена через запятую из ABLATIONS; неизвестное имя — отказ, а не
    тихий прогон без абляции (он выглядел бы как измерение)."""
    names = [n.strip() for n in (spec or "").split(",") if n.strip()]
    unknown = [n for n in names if n not in ABLATIONS]
    if unknown:
        raise SystemExit(f"ОТКАЗ: неизвестная абляция {unknown}; есть {sorted(ABLATIONS)}")
    for n in names:
        ABLATIONS[n](pipeline_smart)
    return names


def install_pool_capture(pipeline_smart, path):
    """Пишет в path (jsonl) пул каждого вызова ядра отбора ровно в том виде,
    в каком его получает ранжирование: после сборки из источников, дедупа и
    жанрового фильтра. Это вход разметки качества: размечается ТОТ пул, из
    которого выбирает код, а не пересобранный рядом по памяти.

    Кандидат сохраняется компактно (id, канал, текст, адрес превью и нужные
    для скачивания заголовки) — полный объект источника не нужен и весит
    мегабайты. Пулы отдают адаптеры ядра: фото с этапа 2, видео с этапа 3;
    код без ядра пулов не отдаёт — файла нет."""
    ps = pipeline_smart
    adapters = [a for a in (getattr(ps, "PHOTO_ADAPTER", None), getattr(ps, "VIDEO_ADAPTER", None))
                if a is not None]
    if not adapters:
        return False
    lock = threading.Lock()

    def wrap(adapter):
        original = adapter.choose

        def probe_url(c):
            if adapter.kind == "photo":
                return ps.candidate_probe_url(c)
            urls = ps.video_preview_urls(c)   # середина — кадр, по которому судят гейты
            return urls[len(urls) // 2] if urls else None

        def row(c):
            return {
                "id": c.get("id"), "channel": ps.candidate_channel(c),
                "via": c.get("_origin_query"),
                "text": (ps.pexels_candidate_text(c) or "")[:300],
                "probe_url": probe_url(c),
                "headers": c.get("_download_headers") or {},
                "duration": c.get("duration"),
            }

        original_filter = getattr(adapter, "filter_pool", None)

        def filter_pool(request, pool):
            # Что фильтр ВЫБРОСИЛ и почему — это вход замера «есть ли нужный
            # кадр в выдаче до фильтров» (план, этап 0). Причина считается
            # теми же функциями, что решают в проде, а не своей копией.
            out = original_filter(request, pool)
            kept = {id(c) for c in out}
            removed = []
            terms = ps.content_blocklist_effective()
            for c in pool:
                if id(c) in kept:
                    continue
                text = ps.pexels_candidate_text(c) or ""
                hit = next((t for t in terms if t in text), None)
                if hit:
                    reason = f"blocklist:{hit}"
                elif ps._candidate_block_key(c) in ps.CONTENT_BLOCKED_CANDIDATE_IDS:
                    reason = "blocked_id"
                elif adapter.kind == "video" and getattr(request, "slot_dur", None) \
                        and ps._video_candidate_too_short(c, request.slot_dur):
                    reason = "too_short"
                else:
                    reason = "other"
                removed.append(dict(row(c), reason=reason))
            PENDING_REMOVED[(request.index, adapter.kind, threading.get_ident())] = (
                removed, [c.get("id") for c in pool])
            return out
        if original_filter is not None:
            adapter.filter_pool = filter_pool

        def choose(request, pool, cf):
            removed, order = PENDING_REMOVED.pop(
                (request.index, adapter.kind, threading.get_ident()), ([], []))
            # slot_dur пишется, потому что в режиме «только пул» слот не
            # получает кадра и его время уходит следующему: фильтр длины
            # видео у следующих слотов тогда режет по раздутой длительности.
            # Анализ пересчитывает его по настоящей (pool_recall.base_slot_durs).
            rec = {"index": request.index, "kind": adapter.kind, "query": request.query,
                   "extra_queries": list(request.extra_queries), "shot_brief": request.shot_brief,
                   "block_text": request.block_text, "slot_dur": getattr(request, "slot_dur", None),
                   "pool": [row(c) for c in pool], "removed": removed,
                   "prefilter_order": order}
            with lock, open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if os.environ.get("SELECTION_POOL_ONLY") == "1":
                return None   # замер пула: без скачиваний и без судьи
            return original(request, pool, cf)
        adapter.choose = choose

    for a in adapters:
        wrap(a)
    return True


def child(mode, net_dir, sandbox, run_dir, pipeline=PIPELINE, overlay="", extra_net=""):
    """Дочерний процесс прогона. pipeline — КАКОЙ код отбора гонять: по
    умолчанию соседний, но харнесс умеет судить и код другой ревизии
    (например, из git worktree) — харнесс и испытуемый код разделены."""
    # Модули ХАРНЕССА берутся из харнесса, модули ОТБОРА — из испытуемого
    # кода: у старой ревизии свои, более старые копии net_recorder и т.п.
    sys.path.insert(0, SCRIPTS)
    import net_recorder
    import time_decisions
    sys.path.insert(0, os.path.dirname(os.path.abspath(pipeline)))
    import dotenv
    dotenv.load_dotenv = lambda *a, **k: False   # окружение целиком передал родитель
    rec = net_recorder.NetRecorder(net_dir, mode, overlay=overlay or None,
                                   extra_roots=(extra_net,) if extra_net else ()).install()
    import museum_sources
    clock = time_decisions.MetCooldownRecorder(
        os.path.join(net_dir, "time_decisions.jsonl"), mode).install(museum_sources)
    sys.argv = [pipeline, sandbox, "--select-only"]
    rc = 1
    hits = {}
    slot_inputs = {}
    slot_attempts = {}
    # Исходник снимается один раз, и модуль компилируется ИМЕННО из этого
    # снимка: якоря покрытия, хэш кода в результате и исполняемый код —
    # один и тот же текст. Читать файл с диска повторно нельзя: его могли
    # поправить за время прогона (так и случилось: запись длилась полчаса,
    # якоря посчитались по чужой версии и все вышли «потерянными»).
    with open(pipeline, encoding="utf-8") as f:
        loaded_src = f.read()
    tree = ast.parse(loaded_src)
    try:
        pipeline_smart = load_module_from_source("pipeline_smart", pipeline, loaded_src)
        hits = _install_tracer(pipeline_smart, slot_loop_range(tree))
        slot = hits.pop("__slot__")
        slot_inputs = hits.pop("__inputs__")
        slot_attempts = hits.pop("__attempts__")
        rec.tagger = lambda: slot["i"]
        clock.tagger = rec.tagger
        install_pool_capture(pipeline_smart, os.path.join(run_dir, "pools.jsonl"))
        apply_ablation(pipeline_smart, os.environ.get("SELECTION_ABLATE", ""))
        try:
            rc = pipeline_smart.main()
        except SystemExit as e:
            rc = e.code if isinstance(e.code, int) else 1
    finally:
        sys.settrace(None)
        hits.pop("__slot__", None)
        hits.pop("__inputs__", None)
        hits.pop("__attempts__", None)
        summary = rec.summary()
        summary["time_decisions"] = clock.summary()
        with open(os.path.join(run_dir, "net_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
        with open(os.path.join(run_dir, "slot_inputs.json"), "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in sorted(slot_inputs.items())}, f,
                      ensure_ascii=False, indent=1)
        with open(os.path.join(run_dir, "slot_attempts.json"), "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in sorted(slot_attempts.items())}, f,
                      ensure_ascii=False, indent=1)
        rec.uninstall()
        cov = coverage_from(loaded_src, hits)
        with open(os.path.join(run_dir, "coverage.json"), "w", encoding="utf-8") as f:
            json.dump({"branches": cov,
                       "pipeline_sha256": hashlib.sha256(loaded_src.encode("utf-8")).hexdigest(),
                       "lines_executed": {k: len(v) for k, v in hits.items()},
                       # Сырые строки — чтобы покрытие можно было пересчитать
                       # по тому же исходнику, не повторяя прогон.
                       "lines": {k: sorted(v) for k, v in hits.items()}},
                      f, ensure_ascii=False, indent=1)
    sys.exit(rc if isinstance(rc, int) else 1)


# ------------------------------------------------------------------ сравнение

def _load_result(freeze, label):
    p = os.path.join(freeze, "runs", label, "result.json")
    if not os.path.exists(p):
        raise SystemExit(f"нет результата прогона {label!r}: {p}")
    return json.load(open(p, encoding="utf-8"))


def parse_expect(expect):
    """Ожидания этапа, законно меняющего вывод:

    {"slots":      {"3": "причина"},              — слот ОБЯЗАН разойтись;
     "reports":    {"run_journal.jsonl": "причина"}, — отчёт ОБЯЗАН разойтись;
     "returncode": "причина",                     — код возврата ОБЯЗАН измениться;
     "attempt_fields": {"shot_brief": "причина"}, — поле запроса попытки,
                                                    которое этап вправе менять
                                                    (см. compare: причина слота
                                                    «ЗАПРОС ПОПЫТКИ ИЗМЕНИЛСЯ»);
     "signature":  "причина",                     — подпись отбора сменилась:
                                                    имена файлов кэша сверяются
                                                    без неё;
     "rewritten_kinds": {"video": "причина"}}     — вид медиа переписан
                                                    целиком (см. compare:
                                                    «ВИД ПЕРЕПИСАН»).
    Плоский словарь {"3": "причина"} — прежняя форма, только слоты.

    Остальные слоты «ниже по течению» автор не перечисляет: харнесс сам
    видит, у каких слотов изменился ВХОД (общее состояние на начало слота) и
    в каких журнал показывает утечку по старой семантике, — см. compare()."""
    expect = expect or {}
    if not ({"slots", "reports", "returncode", "attempt_fields", "signature",
             "rewritten_kinds"} & set(expect)):
        expect = {"slots": expect}
    must = {}
    for k, v in (expect.get("slots") or {}).items():
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"слот {k}: причина обязана быть непустой строкой, дано {v!r}")
        must[int(k)] = v
    rc = expect.get("returncode")
    if rc is not None and (not isinstance(rc, str) or not rc.strip()):
        raise ValueError("returncode: причина обязана быть непустой строкой")
    fields = dict(expect.get("attempt_fields") or {})
    for k, v in fields.items():
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"поле попытки {k}: причина обязана быть непустой строкой")
    return must, dict(expect.get("reports") or {}), rc, fields


def parse_expect_ext(expect):
    """Ожидания, добавленные этапом 3: смена подписи отбора и переписанный
    вид медиа. Отдельно от parse_expect, чтобы прежние вызовы не менялись."""
    expect = expect or {}
    sig = expect.get("signature")
    if sig is not None and (not isinstance(sig, str) or not sig.strip()):
        raise ValueError("signature: причина обязана быть непустой строкой")
    kinds = dict(expect.get("rewritten_kinds") or {})
    for k, v in kinds.items():
        if k not in ("photo", "video"):
            raise ValueError(f"rewritten_kinds: неизвестный вид медиа {k!r}")
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"rewritten_kinds.{k}: причина обязана быть непустой строкой")
    return sig, kinds


# Имя файла кэша кончается подписью отбора: <слот>_<запрос>_<подпись>.<ext>.
_CACHE_SIG_RE = re.compile(r"_[0-9a-f]{10}(\.[A-Za-z0-9]+)$")


def shot_diff(x, y, signature_changed):
    """Поля кадра, в которых прогоны разошлись. При объявленной смене
    подписи имя файла кэша сверяется без неё, а имя клипа (оно выводится из
    пути файла) — только если сам файл разошёлся: тот же файл с теми же
    байтами под другой подписью — тот же кадр."""
    diff = [k for k in SHOT_FIELDS if x.get(k) != y.get(k)]
    if not signature_changed:
        return diff
    fa, fb = (_CACHE_SIG_RE.sub(r"_<sig>\1", v.get("file") or "") for v in (x, y))
    if "file" in diff and fa == fb:
        diff.remove("file")
    if "clip" in diff and "file" not in diff and "file_sha256" not in diff:
        diff.remove("clip")
    return diff


def strip_cache_signature(value):
    """Отчёт с именами файлов кэша без подписи отбора — рекурсивно по
    вложенным словарям и спискам. Нужна при объявленной смене подписи:
    журнал прогона пишет полный путь кадра, и без этого тот же файл под
    новой подписью читался бы как расхождение слота (найдено на приёмке
    этапа 3: слот 1 совпал бит-в-бит, а журнал разошёлся одной строкой
    пути)."""
    if isinstance(value, str):
        return _CACHE_SIG_RE.sub(r"_<sig>\1", value)
    if isinstance(value, list):
        return [strip_cache_signature(v) for v in value]
    if isinstance(value, dict):
        return {k: strip_cache_signature(v) for k, v in value.items()}
    return value


def rewritten_slot(xa, xb, shot_a, shot_b, kinds):
    """Слот, где решал переписанный вид медиа.

    Возвращает (затронут ли слот, обязан ли совпасть). Слот затронут, если
    хоть в одном прогоне в нём была попытка переписанного вида. Обязан
    совпасть он, если в обоих прогонах на экране кадр НЕ переписанного вида
    и попытка, давшая его, получила одинаковый запрос: тогда переписанная
    попытка только проиграла, и её переписывание не может менять исход.
    Так харнесс не превращает «вид переписан» в разрешение на любое
    расхождение слота."""
    kinds_a = {e.get("kind") for e in (xa or [])}
    kinds_b = {e.get("kind") for e in (xb or [])}
    if not ((kinds_a | kinds_b) & set(kinds)):
        return False, False
    ka, kb = (shot_a or {}).get("kind"), (shot_b or {}).get("kind")
    if ka is None or kb is None or ka in kinds or kb in kinds or ka != kb:
        return True, False
    last_a = next((e for e in reversed(xa or []) if e.get("kind") == ka), None)
    last_b = next((e for e in reversed(xb or []) if e.get("kind") == kb), None)
    if last_a is None or last_b is None:
        return True, False
    fa, fb = last_a.get("fields") or {}, last_b.get("fields") or {}
    same = all(fa[f] == fb[f] for f in set(fa) & set(fb))
    return True, same


def journal_leak_slots(result):
    """Слоты, где у НОВОГО прогона есть утечка по СТАРОЙ семантике: попытка,
    кадр которой не встал на экран, несла резервы/победу источника (старый
    код применял их в момент добычи) или вердикты, не определявшие исход
    (старый код держал их в отчётах слота и решал по ним). Только в таких
    слотах, и ниже них через изменившийся вход, правка этапа 1 вправе
    изменить исход."""
    journal = (result.get("reports") or {}).get("run_journal.jsonl") or []
    attempts = {r["attempt_id"]: r for r in journal if r.get("record") == "attempt"}
    leaks = set()
    for r in journal:
        if r.get("record") != "slot":
            continue
        for aid in r.get("attempts") or []:
            att = attempts.get(aid) or {}
            if aid == r.get("shown"):
                continue
            if att.get("effects"):
                leaks.add(r["index"])
            if att.get("verdicts") and r.get("decisive") is not None and aid != r.get("decisive"):
                leaks.add(r["index"])
    return leaks


def _slot_lists(report):
    """Ключи отчёта, значения которых — списки записей с номером слота."""
    return {k for k, v in report.items()
            if isinstance(v, list) and v and all(isinstance(m, dict) and "index" in m for m in v)}


def slot_report_stray(ra, rb, allowed):
    """Отчёт с записями по слотам ({"misses": [{"index": …}, …], …}; так же
    любой другой ключ со списком записей, несущих "index"): поля вне этих
    списков обязаны совпасть, записи — расходиться только в слотах из
    allowed. Возвращает слоты с расхождением без причины ([] — всё
    объяснено) или None, если отчёт не по слотам или разошлось что-то вне
    записей (такое расхождение объяснить слотом нельзя)."""
    if isinstance(ra, list) and isinstance(rb, list):
        # Отчёт-список записей (журнал прогона): тот же разбор по слотам.
        ra, rb = {"records": ra}, {"records": rb}
    if not (isinstance(ra, dict) and isinstance(rb, dict)):
        return None
    keys = _slot_lists(ra) | _slot_lists(rb)
    for k in list(keys):
        if not (isinstance(ra.get(k), list) and isinstance(rb.get(k), list)):
            return None
    keys |= {k for k in ("misses",) if isinstance(ra.get(k), list) and isinstance(rb.get(k), list)}
    if not keys:
        return None
    if {k: v for k, v in ra.items() if k not in keys} != {k: v for k, v in rb.items() if k not in keys}:
        return None

    def by_slot(misses):
        out = {}
        for m in misses:
            if not isinstance(m, dict) or "index" not in m:
                return None
            out.setdefault(m["index"], []).append(m)
        return out
    stray = set()
    for k in keys:
        ga, gb = by_slot(ra[k]), by_slot(rb[k])
        if ga is None or gb is None:
            return None
        stray |= {i for i in set(ga) | set(gb) if ga.get(i) != gb.get(i) and i not in allowed}
    return sorted(stray)


def attempt_input_changes(xa, xb):
    """Поля запроса, в которых разошлись попытки слота. Сравниваются пары
    попыток одного вида ДО первого расхождения видов: дальнейшие попытки —
    следствие изменившегося исхода, а не его причина."""
    if not xa or not xb:
        return set()
    changed = set()
    for ea, eb in zip(xa, xb):
        if ea.get("kind") != eb.get("kind"):
            break
        fa, fb = ea.get("fields") or {}, eb.get("fields") or {}
        changed |= {f for f in set(fa) & set(fb) if fa[f] != fb[f]}
    return changed


CONTRIBUTION_REPORT = "source_contribution.json"


def _gates_sans_contribution(run):
    return {k: v for k, v in (run.get("gates") or {}).items() if k != "source_contribution"}


def contribution_vs_screen(run):
    """Сводный счёт побед источников против кадров на экране.

    Сводный отчёт нельзя разложить по слотам, поэтому его расхождение
    судится не разрешением, а проверкой: побед у каждого источника ровно
    столько, сколько его кадров на экране. Возвращает (сходится?, различия)
    или (None, None), если отчёта нет."""
    rep = (run.get("reports") or {}).get(CONTRIBUTION_REPORT)
    if not isinstance(rep, dict):
        return None, None
    won = {k: v.get("won", 0) for k, v in (rep.get("sources") or {}).items() if v.get("won")}
    screen = {}
    for s in run.get("shots", []):
        if s.get("provider"):
            screen[s["provider"]] = screen.get(s["provider"], 0) + 1
    diff = {k: [won.get(k, 0), screen.get(k, 0)] for k in set(won) | set(screen)
            if won.get(k, 0) != screen.get(k, 0)}
    return not diff, diff


def compare(a, b, expect=None):
    """Классификация по слоту — ровно один класс на слот:

      СОВПАЛ               — все поля слота совпали;
      ОЖИДАЕМО РАЗОШЁЛСЯ   — автор назвал слот исправлением, и он изменился;
      НЕОЖИДАННО СОВПАЛ    — назван исправлением, но не изменился (провал:
                             правка не сработала);
      ВХОД ИЗМЕНИЛСЯ       — слот изменился, и у него изменилось общее
                             состояние на входе (названы поля);
      ЗАПРОС ПОПЫТКИ ИЗМЕНИЛСЯ — вход слота тот же, но у его попытки
                             изменились поля запроса, и ВСЕ они объявлены
                             этапом в attempt_fields (необъявленное поле
                             причиной не считается — иначе ошибка передачи
                             запроса объясняла бы сама себя);
      УТЕЧКА УСТРАНЕНА     — слот изменился при том же входе, и журнал
                             нового прогона показывает в нём утечку старой
                             семантики;
      РАЗОШЁЛСЯ            — изменился без всякой причины (провал).

    Отчёты: названные в ожиданиях обязаны разойтись; отчёты по слотам
    (misses) вправе расходиться только в записях слотов с причиной; прочие —
    совпасть.
    Сеть: расхождение числа обращений или обращение вне записи допустимо
    только в слотах, у которых есть причина (ожидание, изменившийся вход,
    утечка) — по меткам слотов обоих прогонов."""
    must, rep_expect, rc_expect, field_expect = parse_expect(expect)
    sig_expect, kind_expect = parse_expect_ext(expect)
    aa, ab = a.get("slot_attempts") or {}, b.get("slot_attempts") or {}
    leaks = journal_leak_slots(b)
    ia, ib = a.get("slot_inputs") or {}, b.get("slot_inputs") or {}
    sa = {s["index"]: s for s in a.get("shots", [])}
    sb = {s["index"]: s for s in b.get("shots", [])}
    slots, allowed = [], set(must) | leaks
    rewritten_touched, rewritten_changed = [], []
    for i in sorted(set(sa) | set(sb)):
        x, y = sa.get(i), sb.get(i)
        if x is None or y is None:
            diff = ["слот отсутствует в одном из прогонов"]
        else:
            diff = shot_diff(x, y, bool(sig_expect))
        xin, yin = ia.get(str(i)), ib.get(str(i))
        changed_in = sorted(n for n in set(xin or {}) | set(yin or {})
                            if (xin or {}).get(n) != (yin or {}).get(n)) if (xin and yin) else []
        if changed_in:
            allowed.add(i)
        changed_att = attempt_input_changes(aa.get(str(i)), ab.get(str(i)))
        declared_att = bool(changed_att) and changed_att <= set(field_expect)
        if declared_att:
            allowed.add(i)
        touched, must_match = rewritten_slot(aa.get(str(i)), ab.get(str(i)), x, y, kind_expect)
        if touched:
            allowed.add(i)      # сеть слота законно другая: переписанная попытка ходит иначе
            rewritten_touched.append(i)
            if diff:
                rewritten_changed.append(i)
        reason = None
        if i in must:
            klass = "ОЖИДАЕМО РАЗОШЁЛСЯ" if diff else "НЕОЖИДАННО СОВПАЛ"
            reason = must[i]
        elif not diff:
            klass = "СОВПАЛ"
        elif changed_in:
            klass = "ВХОД ИЗМЕНИЛСЯ"
            reason = "изменилось на входе: " + ", ".join(changed_in)
        elif declared_att:
            klass = "ЗАПРОС ПОПЫТКИ ИЗМЕНИЛСЯ"
            reason = "; ".join(f"{f}: {field_expect[f]}" for f in sorted(changed_att))
        elif touched and not must_match:
            klass = "ВИД ПЕРЕПИСАН"
            reason = "; ".join(f"{k}: {v}" for k, v in sorted(kind_expect.items()))
        elif i in leaks:
            klass = "УТЕЧКА УСТРАНЕНА"
            reason = "в слоте была утечка старой семантики (журнал)"
        else:
            klass = "РАЗОШЁЛСЯ"
        slots.append({"index": i, "class": klass, "fields": diff,
                      "a": {k: (x or {}).get(k) for k in diff},
                      "b": {k: (y or {}).get(k) for k in diff},
                      "expected_reason": reason})
    reports, reports_bad = {}, []
    for name in sorted(set(a.get("reports", {})) | set(b.get("reports", {})) | set(rep_expect)):
        ra, rb = a.get("reports", {}).get(name), b.get("reports", {}).get(name)
        if sig_expect:
            ra, rb = strip_cache_signature(ra), strip_cache_signature(rb)
        if ra == rb:
            if name in rep_expect:
                reports_bad.append(name)      # заявленное изменение не произошло
            continue
        if name in rep_expect:
            reports[name] = {"a": ra, "b": rb, "expected_reason": rep_expect[name]}
            continue
        if name == CONTRIBUTION_REPORT:
            # Законно, только если изменились слоты с причиной (иначе пулы
            # и победители те же) И новый счёт сходится с экраном.
            ok_b, diff_b = contribution_vs_screen(b)
            ok_a, diff_a = contribution_vs_screen(a)
            # Переписанный вид меняет счёт «предложено/рассмотрено» и там,
            # где его попытка проиграла, — это тоже законная причина.
            slots_moved = (any(s["class"] not in ("СОВПАЛ", "РАЗОШЁЛСЯ") and s["fields"]
                               for s in slots) or bool(rewritten_touched))
            reports[name] = {"a": ra, "b": rb, "expected_reason": None,
                             "won_vs_screen": {"a": diff_a, "b": diff_b}}
            if slots_moved and ok_b:
                reports[name]["expected_reason"] = ("сменились слоты с причиной; "
                                                    "побед у источников = кадров на экране")
            else:
                reports_bad.append(name)
            continue
        stray = slot_report_stray(ra, rb, allowed)
        reports[name] = {"a": ra, "b": rb, "expected_reason": None,
                         "slots_outside_cause": stray}
        if stray is None or stray:
            reports_bad.append(name)
    ca = a.get("net", {}).get("calls_by_key", {})
    cb = b.get("net", {}).get("calls_by_key", {})
    ta = a.get("net", {}).get("slots_by_key", {})
    tb = b.get("net", {}).get("slots_by_key", {})
    net_diff, net_bad = {}, []
    for k in sorted(set(ca) | set(cb)):
        if ca.get(k, 0) == cb.get(k, 0) and ta.get(k) == tb.get(k):
            continue
        sa, sb = ta.get(k) or {}, tb.get(k) or {}
        # Судится пара «адрес × слот», а не адрес целиком: общий адрес
        # (превью, поиск по запросу секции) звучит во многих слотах, и
        # лишнее обращение в слоте с названной причиной не делает
        # необъяснёнными слоты, где число обращений совпало. Нет разметки
        # слотами при разном числе обращений — судить не по чему, провал.
        changed = {lb for lb in set(sa) | set(sb) if sa.get(lb, 0) != sb.get(lb, 0)}
        net_diff[k] = {"calls": [ca.get(k, 0), cb.get(k, 0)],
                       "slots": [ta.get(k), tb.get(k)], "slots_changed": sorted(changed)}
        if not (changed and all(lb != "none" and int(lb) in allowed for lb in changed)):
            net_bad.append(k)
    divergences = b.get("net", {}).get("divergences", [])
    div_bad = [d for d in divergences if d.get("slot") not in allowed]
    clock = (b.get("net", {}).get("time_decisions") or {}).get("divergences") or []
    # Решение по часам вне записи законно только в слоте с причиной; запись
    # без слота (старый формат, решение вне цикла слотов) — провал.
    clock_bad = [d for d in clock
                 if not (isinstance(d, dict) and d.get("slot") in allowed)]
    bad = [s for s in slots if s["class"] in ("РАЗОШЁЛСЯ", "НЕОЖИДАННО СОВПАЛ")]
    rc_ok = ((a.get("returncode") != b.get("returncode")) if rc_expect
             else (a.get("returncode") == b.get("returncode")))
    # Счёт источников в шапке гейтов — копия сводного отчёта, судится вместе
    # с ним (выше); остальная шапка обязана совпасть.
    gates_differ = _gates_sans_contribution(a) != _gates_sans_contribution(b)
    # Объявленное переписывание обязано быть наблюдаемым: вид переписан, а
    # ни один слот с его попыткой не изменился — либо правка не доехала до
    # испытуемого кода, либо эпизод её не задевает; оба случая — не приёмка.
    rewrite_unseen = bool(kind_expect) and not rewritten_changed
    ok = (not bad and not reports_bad and not net_bad and not div_bad and not clock_bad
          and not gates_differ and rc_ok and not rewrite_unseen)
    return {"ok": ok, "slots": slots, "reports_differ": reports, "reports_unexpected": reports_bad,
            "gates_differ": gates_differ,
            "returncode": [a.get("returncode"), b.get("returncode")],
            "net_calls_differ": net_diff, "net_unexpected": net_bad,
            "replay_divergences": divergences, "divergences_unexpected": div_bad,
            "clock_divergences": clock_bad, "leak_slots": sorted(leaks),
            "rewritten_slots": rewritten_touched, "rewritten_changed": rewritten_changed,
            "rewrite_unseen": rewrite_unseen}


def print_report(rep, a, b):
    print(f"\nСЛОТЫ ({len(rep['slots'])}):")
    for s in rep["slots"]:
        extra = ""
        if s["fields"]:
            extra = "  поля: " + ", ".join(f"{k}: {s['a'].get(k)!r} -> {s['b'].get(k)!r}"
                                          for k in s["fields"])
        if s["expected_reason"]:
            extra += f"  [причина: {s['expected_reason']}]"
        print(f"  #{s['index'] + 1:<3} {s['class']:<19}{extra}")
    if rep.get("clock_divergences"):
        first = rep["clock_divergences"][0]
        first = first.get("key") if isinstance(first, dict) else first
        print(f"\nрешения по часам вне записи и вне слотов с причиной: "
              f"{len(rep['clock_divergences'])} (первое: {first})")
    if rep.get("rewritten_slots"):
        print(f"\nпереписанный вид: попытки в слотах "
              f"{', '.join(str(i + 1) for i in rep['rewritten_slots'])}; изменились "
              f"{', '.join(str(i + 1) for i in rep['rewritten_changed']) or 'НИ ОДИН'}")
    if rep.get("rewrite_unseen"):
        print("    ПЕРЕПИСЫВАНИЕ НЕ НАБЛЮДАЕТСЯ: ни один слот с попыткой переписанного "
              "вида не изменился — правка не доехала до испытуемого кода или эпизод её не задевает")
    print(f"\nшапка гейтов: {'РАЗОШЛАСЬ' if rep['gates_differ'] else 'совпала'}")
    print(f"коды возврата: {rep['returncode'][0]} / {rep['returncode'][1]}")
    print(f"отчёты отбора: {'разошлись: ' + ', '.join(rep['reports_differ']) if rep['reports_differ'] else 'совпали'}")
    wvs = (rep["reports_differ"].get(CONTRIBUTION_REPORT) or {}).get("won_vs_screen") \
        if isinstance(rep["reports_differ"], dict) else None
    if wvs:
        for side, lbl in (("a", "было"), ("b", "стало")):
            d = wvs.get(side)
            print(f"    побед источников против кадров на экране ({lbl}): "
                  + ("сходится" if d == {} else ("нет отчёта" if d is None else
                     ", ".join(f"{k} {w} побед / {n} кадров" for k, (w, n) in sorted(d.items())))))
    if rep["reports_unexpected"]:
        print(f"    БЕЗ названной причины (или ожидались, но не разошлись): "
              f"{', '.join(rep['reports_unexpected'])}")
    na, nb = a.get("net", {}), b.get("net", {})
    print(f"сеть: {na.get('distinct_requests')} адресов / {na.get('total_calls')} обращений  против  "
          f"{nb.get('distinct_requests')} / {nb.get('total_calls')}; "
          f"расходящихся ключей {len(rep['net_calls_differ'])}; "
          f"запросов вне записи {len(rep['replay_divergences'])}")
    for d in rep["replay_divergences"][:10]:
        print(f"    вне записи: {d['method']} {d['url']} (обращение #{d['seq'] + 1}, "
              f"слот {d.get('slot')}, {d.get('served', 'refused')})")
    if rep["net_unexpected"] or rep["divergences_unexpected"]:
        print(f"    сеть расходится ВНЕ слотов с названной причиной: ключей "
              f"{len(rep['net_unexpected'])}, запросов вне записи {len(rep['divergences_unexpected'])}")
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
            "git": _git_rev(os.path.dirname(os.path.dirname(os.path.abspath(args.pipeline or PIPELINE)))),
            "pipeline": os.path.abspath(args.pipeline or PIPELINE),
            "stack": _stack_versions(), "hashseed": str(args.hashseed),
            "env": env_snapshot()}
    with open(os.path.join(freeze, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(f"запись: {meta['episode']} -> {freeze}")
    res = run_pipeline(freeze, "record", "record", args.hashseed, args.keep_media,
                       pipeline=args.pipeline)
    shots_with = sum(1 for s in res["shots"] if s.get("file"))
    print(f"записано за {res['seconds']}с: код {res['returncode']}, слотов {len(res['shots'])}, "
          f"с кадром {shots_with}, сеть {res['net'].get('distinct_requests')} адресов / "
          f"{res['net'].get('total_calls')} обращений")
    return 0 if res["returncode"] in (0, 2) else 1


def cmd_replay(args):
    freeze = os.path.abspath(args.freeze)
    label = args.label or f"replay-seed{args.hashseed}"
    res = run_pipeline(freeze, "replay", label, args.hashseed, args.keep_media,
                       pipeline=args.pipeline, live_fallback=args.live_fallback,
                       overlay_from=args.net_overlay_from)
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
    run_pipeline(freeze, "replay", label, seed, args.keep_media,
                 pipeline=args.pipeline, live_fallback=args.live_fallback,
                 overlay_from=args.net_overlay_from)
    a, b = _load_result(freeze, args.against), _load_result(freeze, label)
    rep = compare(a, b, _expect(args.expect))
    print_report(rep, a, b)
    with open(os.path.join(freeze, "runs", label, f"compare_vs_{args.against}.json"), "w",
              encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    return 0 if rep["ok"] else 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_child":
        return child(*argv[1:8])
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
        sp.add_argument("--pipeline", help="какой pipeline_smart.py гонять (по умолчанию соседний)")
    for sp in (rp, v):
        sp.add_argument("--net-overlay-from", metavar="ПРОГОН",
                        help="дополнительно отдавать живые запросы указанного прогона "
                             "(его net_overlay), продолжая по каждому адресу запись")
        sp.add_argument("--live-fallback", action="store_true",
                        help="запросы вне записи выполнять живьём в отдельный слой (названные)")
    v.add_argument("--against", default="record",
                   help="с каким прогоном сравнивать (по умолчанию запись)")
    for sp in (r, rp, v):
        sp.add_argument("--keep-media", action="store_true",
                        help="оставить скачанные прогоном файлы (для глазной проверки); "
                             "по умолчанию удаляются после подсчёта хэшей")
    cv = sub.add_parser("coverage", help="пересчитать покрытие прогона по сырым строкам")
    cv.add_argument("freeze")
    cv.add_argument("label")
    cv.set_defaults(fn=lambda a: (print(json.dumps(recompute_coverage(os.path.abspath(a.freeze), a.label),
                                                   ensure_ascii=False, indent=1)), 0)[1])
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
