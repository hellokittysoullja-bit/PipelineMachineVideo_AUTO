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
    **{f"level_regression.{f}":
       "обратный замер уровней по отрендеренному звуку — измерительная "
       "оснастка регрессии (tests/test_level_regression.py), в рендер не "
       "входит СОЗНАТЕЛЬНО: она рендерит сцену дважды, со слоем и без, и "
       "вычитает результаты по сэмплам. Поставить её в рабочий путь значило "
       "бы удваивать каждый рендер ради проверки"
       for f in ("build_fixture", "decode_mono", "layer_contribution",
                 "limiter_reduction_db", "loudness_of_samples", "render_scene",
                 "window")},
    **{name:
       "генерация кадра по брифу — ЗАДЕЛ, сознательно не подключённый к "
       "отбору (решение владельца 23.09: все цели подбора выполняются на "
       "найденных кадрах, генерация — будущая добавка). Модуль и клиент "
       "шлюза держатся рабочими своими тестами (test_shot_generator.py, "
       "test_llm_gateway.py). Подключат — эти строки обязаны уйти: "
       "устаревшая запись валит тест так же, как новая находка"
       for name in ("shot_generator.generate", "shot_generator.prompt_for",
                    "shot_generator.cache_key", "llm_gateway.image",
                    "llm_gateway.image_cost")},
    "assemble.estimate_xfade_budget":
        "альтернативный сборщик по слотам (assemble.py) — второй, ручной путь "
        "сборки; в основной рендер (pipeline_smart.py) не входит",
    "pipeline_smart.reset_world_card_cache":
        "сброс кэша паспорта мира между прогонами — нужен тестам и "
        "повторному вызову main() в одном процессе; в продовом "
        "однопрогонном пути сбрасывать нечего (тот же случай, что "
        "reset_call_counter ниже)",
    "shot_director.reset_call_counter":
        "сброс счётчика живых вызовов между прогонами — нужен тестам, в "
        "продовом однопрогонном пути вызывать нечего",
    "stress_placement.accentize_with_homograph_correction":
        "полная правка ударения в ТЕКСТЕ — сознательно не подключена к "
        "рендеру (см. CLAUDE.md: резы завязаны на [pause]-границы, не на "
        "слог). Живой путь использует detected_homographs() того же модуля",
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


# Метка «атрибут на выражении, а не на имени»: такое имя никогда не
# разрешается как функция модуля, оно учитывается только как имя метода.
ATTR_ON_EXPR = "<expr>"


def _def_references(m, tree):
    """(модуль, имя) -> ссылки. Узел графа — ИМЯ, а одноимённых определений
    в модуле бывает несколько (методы разных классов, `install` у двух
    классов). Прежде словарь по имени оставлял последнее определение, и
    ссылки остальных молча терялись: живое, вызываемое только из них,
    объявлялось мёртвым. Теперь ссылки объединяются."""
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.setdefault((m, n.name), set()).update(_references(n))
    return out


def _references(node):
    """И вызовы, и ССЫЛКИ на функции: в этом коде функция сплошь и рядом
    передаётся значением (director_score_fn, render_pool.submit(kenburns,...),
    functools.partial) — граф только по ast.Call объявил бы мёртвыми десятки
    работающих функций."""
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            out.add((None, n.id))
        elif isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load):
            # `mod.f` — ссылка на функцию модуля; `obj.method` при ЛЮБОМ
            # выражении слева (в том числе `Cls(...).install()`) — ещё и
            # вызов метода по имени, см. reachable().
            base = n.value.id if isinstance(n.value, ast.Name) else ATTR_ON_EXPR
            out.add((base, n.attr))
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
        graph.update(_def_references(m, tree))
        top = set()
        for n in tree.body:
            if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                top |= _references(n)
        graph[(m, "<module>")] = top
    return mods, defs, alias, from_imports, graph


def _class_methods(mods):
    """Модуль -> {класс: [имена его методов]} для классов верхнего уровня."""
    out = {}
    for m, tree in mods.items():
        out[m] = {n.name: [b.name for b in n.body
                           if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef))]
                  for n in tree.body if isinstance(n, ast.ClassDef)}
    return out


def reachable(defs, alias, from_imports, graph, classes=None):
    """Точки входа — модульный уровень каждого скрипта и его main(): ровно то,
    что реально запускает протокол производства (ЧАСТЬ 13 CLAUDE.md).

    МЕТОДЫ. Вызов метода на объекте (`rec.install()`, `brain.ask(...)`)
    статически не привязывается к классу: тип `rec` известен только в
    рантайме. Раньше из-за этого живые методы объявлялись мёртвыми и
    вносились в список исключений поимённо. Теперь правило: метод живой,
    если его КЛАСС упомянут из живого кода (создан, передан, унаследован) И
    имя метода вызывается как атрибут где-то в живом коде; конструктор и
    прочие dunder-методы живы вместе с классом — их зовёт сам язык.
    Требуются оба условия: класс, на который никто не ссылается, остаётся
    мёртвым целиком, а метод, чьё имя не зовёт никто, — мёртвым внутри
    живого класса."""
    classes = classes or {}
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

    def resolve_class(m, ref):
        mod, name = ref
        if mod is None:
            if name in classes.get(m, {}):
                return (m, name)
            if name in from_imports.get(m, {}):
                mm, nn = from_imports[m][name]
                return (mm, nn) if nn in classes.get(mm, {}) else None
            return None
        real = alias.get(m, {}).get(mod, mod)
        return (real, name) if name in classes.get(real, {}) else None

    seen, stack = set(), list(entries)
    live_classes, attr_names = set(), set()
    while True:
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            for ref in graph.get(cur, ()):
                if ref[0] is not None:
                    attr_names.add(ref[1])
                nxt = resolve(cur[0], ref)
                if nxt and nxt not in seen:
                    stack.append(nxt)
                cls = resolve_class(cur[0], ref)
                if cls:
                    live_classes.add(cls)
        for cm, cn in live_classes:
            for meth in classes[cm][cn]:
                node = (cm, meth)
                dunder = meth.startswith("__") and meth.endswith("__")
                if node not in seen and (dunder or meth in attr_names):
                    stack.append(node)
        if not stack:
            return seen


def is_interface_stub(fn):
    """Объявление интерфейса: тело — только докстринг и `raise
    NotImplementedError`. Такой метод ничего не делает и слоем, «который
    есть, но ролику не даёт ничего», быть не может; работу делают его
    реализации в подклассах, и их достижимость проверяется как обычно."""
    body = list(fn.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    if len(body) != 1 or not isinstance(body[0], ast.Raise) or body[0].exc is None:
        return False
    exc = body[0].exc.func if isinstance(body[0].exc, ast.Call) else body[0].exc
    return isinstance(exc, ast.Name) and exc.id == "NotImplementedError"


def _interface_stubs(mods):
    return {(m, n.name) for m, tree in mods.items() for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and is_interface_stub(n)}


@pytest.fixture(scope="module")
def analysis():
    mods, defs, alias, from_imports, graph = build_graph()
    return {"defs": defs, "graph": graph, "stubs": _interface_stubs(mods),
            "seen": reachable(defs, alias, from_imports, graph, _class_methods(mods))}


def test_no_public_function_is_unreachable(analysis):
    public = [(m, n) for (m, n) in analysis["graph"]
              if n != "<module>" and not n.startswith("_") and (m, n) not in analysis["stubs"]]
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
    feature_flags.mode("X"), feature_flags.value("X"), os.environ.get("X"),
    os.getenv("X").

    `value(` добавлен 15.09: без него гвард ложно объявлял мёртвым
    `LUMA_MATCH`, который читается строкой
    `feature_flags.value("LUMA_MATCH")` в pipeline_smart.luma_match_params().
    Детектор знал три аксессора из четырёх, и четвёртый — не экзотика, а
    штатный способ прочитать флаг со списком значений. Добавление может
    только превратить ложное падение в проход: `value(` — настоящее
    чтение, и мёртвый флаг им не замаскируешь."""
    sites = {}
    for fn in sorted(os.listdir(SCRIPTS_DIR)):
        if not fn.endswith(".py"):
            continue
        path = os.path.join(SCRIPTS_DIR, fn)
        with open(path, encoding="utf-8") as f:
            src = f.read()
        for m in re.finditer(r"""(?:enabled|mode|value|environ\.get|getenv)\(\s*["']([A-Z0-9_]+)["']""", src):
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


def test_the_guard_knows_every_public_reader_of_the_registry():
    """Охранник ищет чтение флага по ИМЕНИ функции-читателя. Появись в
    feature_flags новый публичный читатель — охранник перестал бы его
    видеть и объявил бы живой флаг мёртвым. Ровно это и случилось с
    `value()` и `LUMA_MATCH`: красный тест доехал до общей ветки.
    Поэтому список читателей сверяется с самим модулем, а не живёт
    отдельной копией в регулярном выражении.
    """
    readers = {n for n in dir(feature_flags)
               if not n.startswith("_")
               and callable(getattr(feature_flags, n))
               and n in ("enabled", "mode", "value")}
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "test_no_dead_layers.py"), encoding="utf-8").read()
    pattern = re.search(r"re\.finditer\(r\"{3}(.+?)\"{3}", src, re.S)
    assert pattern, "не найдено выражение поиска читателей"
    known = pattern.group(1)
    unseen = sorted(r for r in readers if r not in known)
    assert not unseen, (
        "feature_flags экспортирует читателей, которых охранник не ищет — "
        "живой флаг будет объявлен мёртвым: " + ", ".join(unseen))


def test_method_rule_keeps_dead_code_dead():
    """Правило методов не должно ослеплять охранника: живой класс не делает
    живыми ВСЕ свои методы, а класс без ссылок не оживает вовсе."""
    src = {
        "app": ("import lib\n"
                "def main():\n"
                "    lib.Rec(1).install()\n"
                "    lib.helper()\n"),
        "lib": ("def helper():\n"
                "    return 1\n"
                "def used_by_method():\n"
                "    return 2\n"
                "def orphan():\n"
                "    return 3\n"
                "class Rec:\n"
                "    def __init__(self, x):\n"
                "        self.x = x\n"
                "    def install(self):\n"
                "        return self._inner()\n"
                "    def _inner(self):\n"
                "        return used_by_method()\n"
                "    def never_called(self):\n"
                "        return orphan()\n"
                "class Unused:\n"
                "    def install(self):\n"
                "        return 0\n"),
    }
    mods = {m: ast.parse(s) for m, s in src.items()}
    defs = {m: _func_defs(t) for m, t in mods.items()}
    alias = {"app": {"lib": "lib"}, "lib": {}}
    from_imports = {"app": {}, "lib": {}}
    graph = {}
    for m, tree in mods.items():
        graph.update(_def_references(m, tree))
        graph[(m, "<module>")] = set()
    seen = reachable(defs, alias, from_imports, graph, _class_methods(mods))
    for live in ("helper", "install", "_inner", "used_by_method", "__init__"):
        assert ("lib", live) in seen, f"{live} обязан быть живым"
    for dead in ("never_called", "orphan"):
        assert ("lib", dead) not in seen, f"{dead} ожил — охранник ослеп"


def test_interface_stub_rule_is_narrow():
    """Исключаются ТОЛЬКО объявления без тела; метод с настоящей работой
    остаётся под проверкой, даже если рядом стоит raise NotImplementedError."""
    tree = ast.parse(
        "class A:\n"
        "    def stub(self):\n"
        "        \"\"\"doc\"\"\"\n"
        "        raise NotImplementedError\n"
        "    def stub_call(self):\n"
        "        raise NotImplementedError('x')\n"
        "    def real(self, x):\n"
        "        if not x:\n"
        "            raise NotImplementedError\n"
        "        return x\n"
        "    def other_error(self):\n"
        "        raise ValueError\n")
    got = {n.name: is_interface_stub(n) for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert got == {"stub": True, "stub_call": True, "real": False, "other_error": False}

