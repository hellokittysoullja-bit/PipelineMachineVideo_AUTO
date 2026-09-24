#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Подпись КОДА, который принимает решение, — собирается сама.

Зачем. Кэш кандидатов и клипов ключуется подписью правил отбора: правка
правил без смены подписи на прогретом temp_smart/ не доходит до экрана —
кадр берётся готовым по старым правилам. Подписи держали РУЧНЫЕ списки
функций, и список отставал по построению: замер 25.09 — от адаптеров фото и
видео достижимы 161 функция pipeline_smart, 109 из них не входили ни в одну
подпись (_repick, judge_rank, relevance_rank_bucket, frame_readable...).
Класс «забыли добавить в подпись» записан в CLAUDE.md больше пяти раз.

Здесь список не пишется руками: от точек входа обходятся все функции,
которые они вызывают или на которые ссылаются (в том же модуле и в
перечисленных соседних), и хэшируется их текст БЕЗ комментариев и
докстрингов (ast.unparse) — правка комментария подпись не меняет, правка
кода меняет. Константы сюда не входят сознательно: их значения зависят от
состояния прогона (флаги поломки, счётчики), и подпись по ним была бы
нестабильной; пороги и режимы объявляются в подписях явно, как раньше.

Граница (stop) — функции, код которых входит в другую подпись: правка
судьи не должна перекачивать слоты вне платной зоны."""
import ast
import hashlib
import inspect
import textwrap
import types


def _norm_source(obj):
    """Текст функции/класса без комментариев и докстрингов."""
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(obj)))
    except (OSError, TypeError, SyntaxError):
        return None
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module))
                and body and isinstance(body[0], ast.Expr)
                and isinstance(getattr(body[0], "value", None), ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def _referenced(obj):
    """Имена и пары (имя модуля, атрибут), на которые ссылается код объекта."""
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(obj)))
    except (OSError, TypeError, SyntaxError):
        return set(), set()
    names, attrs = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            names.add(n.id)
        elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
            attrs.add((n.value.id, n.attr))
    return names, attrs


def _is_code(v):
    return isinstance(v, (types.FunctionType, type))


def _module_name(obj):
    """Имя модуля объекта; скрипт, запущенный как __main__, называется по
    файлу — иначе подпись прогона из командной строки и из харнесса
    (импорт) различалась бы при одном и том же коде."""
    mod = getattr(obj, "__module__", None)
    if mod == "__main__":
        import os
        import sys
        path = getattr(sys.modules.get("__main__"), "__file__", "") or ""
        return os.path.splitext(os.path.basename(path))[0] or mod
    return mod


def _module(name, allowed):
    """Модуль по имени, если в него разрешено заходить; ленивый импорт
    внутри функции (import shot_judge) в __globals__ не виден."""
    import importlib
    import sys
    if name not in allowed:
        return None
    m = sys.modules.get(name)
    if m is None:
        try:
            m = importlib.import_module(name)
        except Exception:  # noqa: BLE001 — нет модуля: нечего и подписывать
            return None
    return m


def reachable(entries, modules, stop=()):
    """{полное имя: объект} всего кода, достижимого от entries. modules —
    имена модулей, в которые разрешено заходить (остальное — внешние
    библиотеки, их версия — забота подписи стека)."""
    allowed = set(modules)
    stop_ids = {id(s) for s in stop}
    seen, out, todo = set(), {}, list(entries)
    while todo:
        obj = todo.pop()
        if id(obj) in seen or id(obj) in stop_ids:
            continue
        seen.add(id(obj))
        mod = _module_name(obj)
        if mod not in allowed:
            continue
        out[f"{mod}.{getattr(obj, '__qualname__', repr(obj))}"] = obj
        g = getattr(obj, "__globals__", None)
        if g is None:
            import sys
            owner = sys.modules.get(getattr(obj, "__module__", ""))
            g = vars(owner) if owner is not None else {}
        names, attrs = _referenced(obj)
        for name in names:
            v = g.get(name)
            if _is_code(v):
                todo.append(v)
            elif v is not None and not isinstance(v, types.ModuleType) \
                    and _module_name(type(v)) in allowed:
                todo.append(type(v))        # экземпляр своего класса (PHOTO_ADAPTER)
        for base, attr in attrs:
            m = g.get(base)
            if not isinstance(m, types.ModuleType):
                m = _module(base, allowed) if m is None else None
            if isinstance(m, types.ModuleType) and m.__name__ in allowed:
                v = getattr(m, attr, None)
                if _is_code(v):
                    todo.append(v)
        if isinstance(obj, type):
            for v in vars(obj).values():
                if isinstance(v, (staticmethod, classmethod)):
                    v = v.__func__
                if _is_code(v):
                    todo.append(v)
    return out


def signature(entries, modules, stop=()):
    """Короткий хэш текста всего достижимого кода (порядок — по имени)."""
    code = reachable(entries, modules, stop)
    h = hashlib.sha256()
    for name in sorted(code):
        src = _norm_source(code[name])
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update((src or "?").encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16], len(code)
