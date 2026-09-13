"""Машинная проверка «слой написан, включён — и его никто не вызывает».

Это САМЫЙ ЧАСТЫЙ класс дефекта в истории этого репозитория, и он каждый раз
находился глазами, постфактум, по готовому ролику:

* Openverse — код написан, флаг включён, живой путь отбора не вызывал ни разу,
  вклад в опубликованный эпизод ровно ноль (07.09);
* Pixabay и Unsplash — то же самое, два источника сразу (13.09);
* reveal-акценты — ассеты сгенерированы, шаг сборки не написан никогда, за
  23-минутный эпизод звучал ОДИН тип эффекта из четырёх (13.09);
* filter_alt_blocklist() — существовала ровно в одном месте, и половина
  слотов эпизода (видео) не проходила её никогда (07.09);
* DEFLICKER_ENABLED — документированное имя флага не совпадало с читаемым в
  коде, то есть откат по документации молча не срабатывал (03.09).

Все пять — одно и то же: между «функция есть» и «функция вызывается из
рендера» не было НИ ОДНОЙ автоматической проверки. Этот тест её делает:
строит граф достижимости по исходникам (вызовы И ссылки на функции — в этом
коде колбэки передаются как значения: director_score_fn, пул рендера) и
требует, чтобы каждая публичная функция была достижима хотя бы от одной точки
входа, либо СТОЯЛА В СПИСКЕ НИЖЕ С ПРИЧИНОЙ.

Список — не «глушилка», а инвентарь: он заставляет назвать вслух, почему
код существует и не работает. Устаревшая запись в нём тоже валит тест.
"""
import ast
import os
import re
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

import feature_flags  # noqa: E402


# Публичные функции, сознательно НЕ достижимые из рабочих путей. Ключ —
# "модуль.функция", значение — причина, которую обязан назвать автор.
ALLOWED_UNREACHABLE = {
    "assemble.estimate_xfade_budget":
        "альтернативный сборщик по слотам (assemble.py) — второй, ручной путь "
        "сборки; в основной рендер (pipeline_smart.py) не входит",
    "feature_flags.is_boolean":
        "служебный интроспектор реестра — используется тестами и отчётами, "
        "не рендером",
    "shot_director.reset_call_counter":
        "сброс счётчика живых вызовов между прогонами — нужен тестам, в "
        "продовом однопрогонном пути вызывать нечего",
    "stress_placement.accentize_with_homograph_correction":
        "полная правка ударения в ТЕКСТЕ — сознательно не подключена к "
        "рендеру (см. CLAUDE.md: резы завязаны на [pause]-границы, не на "
        "слог). Живой путь использует detected_homographs() того же модуля",
    "shot_brief_planner.channel_era_window": "исследовательский модуль локального "
        "режиссёра (docs/quality/DIRECTOR_LOCAL_LLM.md) — в рендер не подключён",
    "shot_brief_planner.generate": "то же",
    "shot_brief_planner.load": "то же",
    "shot_brief_planner.validate_brief": "то же",
    "visual_director.cache_signature":
        "вызывается через ССЫЛКУ НА МОДУЛЬ-ПАРАМЕТР "
        "(pipeline_smart._visual_director_cache_signature(director_ref)) — "
        "статический граф такой вызов увидеть не может",
    "look_reference.text_domain_hint":
        "вызывается через ссылку на модуль-параметр, как и предыдущая",
}


def _parse_all():
    mods = {}
    for fn in sorted(os.listdir(SCRIPTS_DIR)):
        if not fn.endswith(".py"):
            continue
        with open(os.path.join(SCRIPTS_DIR, fn), encoding="utf-8") as f:
            mods[fn[:-3]] = ast.parse(f.read())
    return mods


def _func_defs(tree):
    return {n.name: n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _references(node):
    """И вызовы, и ССЫЛКИ на функции: в этом коде функция сплошь и рядом
    передаётся значением (director_score_fn, render_pool.submit(kenburns,...),
    functools.partial) — граф только по ast.Call объявил бы мёртвыми десятки
    работающих функций."""
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            out.add((None, n.id))
        elif (isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load)
              and isinstance(n.value, ast.Name)):
            out.add((n.value.id, n.attr))
    return out


def build_graph():
    mods = _parse_all()
    defs = {m: _func_defs(t) for m, t in mods.items()}
    alias, from_imports = {}, {}
    for m, tree in mods.items():
        a, fi = {}, {}
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for x in n.names:
                    if x.name in defs:
                        a[x.asname or x.name] = x.name
            elif isinstance(n, ast.ImportFrom) and n.module in defs:
                for x in n.names:
                    fi[x.asname or x.name] = (n.module, x.name)
        alias[m], from_imports[m] = a, fi

    graph = {}
    for m, tree in mods.items():
        for name, node in defs[m].items():
            graph[(m, name)] = _references(node)
        top = set()
        for n in tree.body:
            if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                top |= _references(n)
        graph[(m, "<module>")] = top
    return mods, defs, alias, from_imports, graph


def reachable(defs, alias, from_imports, graph):
    """Точки входа — модульный уровень каждого скрипта и его main(): ровно то,
    что реально запускает протокол производства (ЧАСТЬ 13 CLAUDE.md)."""
    entries = [(m, "<module>") for m in defs]
    entries += [(m, "main") for m in defs if "main" in defs[m]]

    def resolve(m, ref):
        mod, name = ref
        if mod is None:
            if name in defs.get(m, {}):
                return (m, name)
            if name in from_imports.get(m, {}):
                mm, nn = from_imports[m][name]
                return (mm, nn) if nn in defs.get(mm, {}) else None
            return None
        real = alias.get(m, {}).get(mod, mod)
        return (real, name) if (real in defs and name in defs[real]) else None

    seen, stack = set(), list(entries)
    while stack:
        cur = stack.pop()
        seen.add(cur)
        for ref in graph.get(cur, ()):
            nxt = resolve(cur[0], ref)
            if nxt and nxt not in seen:
                stack.append(nxt)
    return seen


@pytest.fixture(scope="module")
def analysis():
    mods, defs, alias, from_imports, graph = build_graph()
    return {"defs": defs, "graph": graph,
            "seen": reachable(defs, alias, from_imports, graph)}


def test_no_public_function_is_unreachable(analysis):
    public = [(m, n) for (m, n) in analysis["graph"]
              if n != "<module>" and not n.startswith("_")]
    dead = sorted(f"{m}.{n}" for (m, n) in public
                  if (m, n) not in analysis["seen"])
    unexplained = [d for d in dead if d not in ALLOWED_UNREACHABLE]
    assert not unexplained, (
        "Эти публичные функции не вызываются ни из одного рабочего пути — "
        "ровно тот класс дефекта, которым уже отличились Openverse, Pixabay, "
        "Unsplash, reveal-акценты и видео-блоклист: код есть, ролику не даёт "
        "ничего. Подключить — или внести в ALLOWED_UNREACHABLE с причиной:\n  "
        + "\n  ".join(unexplained))


def test_allowlist_has_no_stale_entries(analysis):
    """Запись, которая уже подключена (или удалена), обязана уйти из списка —
    иначе он перестаёт быть инвентарём и становится свалкой."""
    public = {f"{m}.{n}" for (m, n) in analysis["graph"] if n != "<module>"}
    seen = {f"{m}.{n}" for (m, n) in analysis["seen"]}
    stale = sorted(k for k in ALLOWED_UNREACHABLE
                   if k not in public or k in seen)
    assert not stale, (
        "Записи в ALLOWED_UNREACHABLE больше не описывают реальность "
        "(функция подключена или удалена): " + ", ".join(stale))


def _flag_read_sites():
    """Где в коде реально читается флаг: feature_flags.enabled("X"),
    feature_flags.mode("X"), os.environ.get("X"), os.getenv("X")."""
    sites = {}
    for fn in sorted(os.listdir(SCRIPTS_DIR)):
        if not fn.endswith(".py"):
            continue
        path = os.path.join(SCRIPTS_DIR, fn)
        with open(path, encoding="utf-8") as f:
            src = f.read()
        for m in re.finditer(r"""(?:enabled|mode|environ\.get|getenv)\(\s*["']([A-Z0-9_]+)["']""", src):
            sites.setdefault(m.group(1), set()).add(fn[:-3])
    return sites


def test_every_registered_flag_is_read_somewhere():
    """Флаг, объявленный в реестре и не читаемый ниоткуда, — это обещание
    отката, которого нет. Ровно так молча не работал DEFLICKER_ENABLED:
    документация называла одно имя, код читал другое."""
    sites = _flag_read_sites()
    names = [f.name for f in feature_flags.FLAGS.values()] \
        if isinstance(feature_flags.FLAGS, dict) else [f.name for f in feature_flags.FLAGS]
    missing = []
    for name in names:
        flag = (feature_flags.FLAGS[name] if isinstance(feature_flags.FLAGS, dict)
                else next(f for f in feature_flags.FLAGS if f.name == name))
        known = {name} | set(getattr(flag, "aliases", ()) or ())
        if not (known & set(sites)):
            missing.append(name)
    assert not missing, (
        "Эти флаги объявлены в реестре, но не читаются ни одной строкой кода — "
        "выставить их в .env значит ничего не изменить: " + ", ".join(missing))


def test_sound_resolver_has_one_implementation(tmp_path, monkeypatch):
    """Раньше здесь был тест «две копии не разошлись» — это компромисс, а не
    решение: дубликат лишь сторожился. Теперь pipeline_smart.library_sounds()
    ДЕЛЕГИРУЕТ в sound_library.library_files(), и тест держит именно это:
    подмена оригинала обязана быть видна через обёртку, копии не существует."""
    import sound_library as sl
    import pipeline_smart as ps

    src = open(os.path.join(SCRIPTS_DIR, "pipeline_smart.py"), encoding="utf-8").read()
    assert "_SOUND_LIBRARY_DIR" not in src, "в pipeline_smart снова завёлся свой путь к библиотеке"

    root = tmp_path / "library"
    d = root / "sfx" / "plate_tick"
    d.mkdir(parents=True)
    for name in ("b.flac", "a.flac", "c.wav"):
        (d / name).write_bytes(b"")
    monkeypatch.setattr(sl, "LIBRARY_ROOT", str(root))
    monkeypatch.setattr(ps, "SOUND_LIBRARY_ENABLED", True)
    assert [os.path.basename(p) for p in ps.library_sounds("sfx", "plate_tick")] == ["a.flac", "b.flac"]

    monkeypatch.setattr(sl, "library_files", lambda kind, name: ["/patched/only.flac"])
    assert ps.library_sounds("sfx", "plate_tick") == ["/patched/only.flac"]

    monkeypatch.setattr(ps, "SOUND_LIBRARY_ENABLED", False)
    assert ps.library_sounds("sfx", "plate_tick") == []
