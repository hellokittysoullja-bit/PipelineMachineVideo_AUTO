#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""--select-only: весь отбор кадра, ни одного рендера.

Режим нужен дважды: человеку — увидеть решение о кадре до многочасового
рендера (Шаг 7.2 CLAUDE.md), и харнессу эквивалентности
(scripts/selection_freeze.py) — гонять НАСТОЯЩИЙ main(), а не вторую копию
его логики.

Обе пользы исчезают, если режим отбирает НЕ ТО ЖЕ САМОЕ, что полный рендер.
Поэтому здесь две независимые защиты одного свойства:

1. Статическая (AST): после точки среза в цикле слотов нет ни одной
   мутации состояния, от которого зависит отбор СЛЕДУЮЩЕГО слота. Какое
   состояние «отборное», выводится из самого кода, а не из рукописного
   списка имён: переносимая между итерациями переменная — та, что читается
   до среза раньше, чем безусловно присваивается в той же итерации.
   Рукописный список отстал бы от кода при первой же новой переменной.
   ЧЕСТНЫЙ ПРЕДЕЛ: межпроцедурные мутации (функция, вызванная в хвосте,
   меняет глобал, который читает отбор) статикой не ловятся — их ловит
   защита 2.
2. Поведенческая: шотлист режима отбора совпадает с шотлистом полного
   рендера на том же эпизоде, поле в поле.
"""
import ast
import json
import os
import shutil
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _episode_factory import N_BLOCKS, PIPELINE, build_episode, run_pipeline  # noqa: E402

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe не найдены в PATH",
)

RENDER_ARTIFACTS = ("render_manifest.json", "render_qc_report.json",
                    "look_manifest.json", "camera_language_report.json")

MUTATING_METHODS = {"append", "extend", "insert", "pop", "remove", "clear", "update",
                    "add", "discard", "setdefault", "sort", "reverse", "popitem",
                    "__setitem__", "__delitem__"}


# ---------------------------------------------------------------- статика

def _main_and_cut():
    tree = ast.parse(open(PIPELINE, encoding="utf-8").read())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    cuts = []
    for loop in ast.walk(main):
        if not isinstance(loop, ast.For):
            continue
        for k, st in enumerate(loop.body):
            if (isinstance(st, ast.If) and isinstance(st.test, ast.Name)
                    and st.test.id == "SELECT_ONLY"
                    and any(isinstance(x, ast.Continue) for x in st.body)):
                cuts.append((loop, k))
    return tree, main, cuts


def _receiver_nodes(node):
    """Name-узлы, которые загружаются ТОЛЬКО как получатель мутации:
    x.append(...), x[k] = ..., del x[k]. Такое чтение не берёт значение —
    это запись в накопитель, а не вход решения."""
    out = set()
    for n in ast.walk(node):
        bases = []
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                and n.func.attr in MUTATING_METHODS:
            bases.append(n.func.value)
        elif isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            bases += [t for t in targets if isinstance(t, (ast.Subscript, ast.Attribute))]
        elif isinstance(n, ast.Delete):
            bases += [t for t in n.targets if isinstance(t, (ast.Subscript, ast.Attribute))]
        for b in bases:
            while isinstance(b, (ast.Subscript, ast.Attribute)):
                b = b.value
            if isinstance(b, ast.Name):
                out.add(id(b))
    return out


def _loaded_names(node, values_only=False):
    skip = _receiver_nodes(node) if values_only else set()
    return {n.id for n in ast.walk(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and id(n) not in skip}


def _assigned_top_level(st):
    """Имена, которые оператор присваивает БЕЗУСЛОВНО (на верхнем уровне тела
    цикла). Условные присваивания внутри if/try имя не определяют."""
    out = set()
    if isinstance(st, (ast.Assign, ast.AnnAssign)):
        targets = st.targets if isinstance(st, ast.Assign) else [st.target]
        for t in targets:
            for n in ast.walk(t):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                    out.add(n.id)
    return out


def _loop_carried(pre_cut, loop):
    """Входы отбора следующей итерации: чьё ЗНАЧЕНИЕ читается до среза
    раньше, чем безусловно присваивается в той же итерации."""
    defined = set()
    for n in ast.walk(loop.target):
        if isinstance(n, ast.Name):
            defined.add(n.id)
    carried = set()
    for st in pre_cut:
        carried |= (_loaded_names(st, values_only=True) - defined)
        defined |= _assigned_top_level(st)
    return carried


def _is_select_only_test(test, negated):
    if negated:
        return (isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not)
                and isinstance(test.operand, ast.Name) and test.operand.id == "SELECT_ONLY")
    return isinstance(test, ast.Name) and test.id == "SELECT_ONLY"


def _selection_outputs(main, loop):
    """Что после цикла читает ПУТЬ РЕЖИМА ОТБОРА: от конца цикла до
    `if SELECT_ONLY: return`, без тел `if not SELECT_ONLY:` (их else —
    как раз путь отбора, он учитывается). Накопитель, который читается здесь,
    — выход отбора (шотлист, отчёты); тот, что только под `if not
    SELECT_ONLY`, — бухгалтерия рендера."""
    ret_line = min(n.lineno for n in ast.walk(main)
                   if isinstance(n, ast.If) and _is_select_only_test(n.test, False)
                   and n.lineno > loop.end_lineno
                   and any(isinstance(x, ast.Return) for x in n.body))
    render_only = set()
    for n in ast.walk(main):
        if isinstance(n, ast.If) and _is_select_only_test(n.test, True):
            for st in n.body:
                for m in ast.walk(st):
                    render_only.add(id(m))
    names = set()
    for n in ast.walk(main):
        if (isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                and getattr(n, "lineno", 0) > loop.end_lineno
                and n.lineno < ret_line and id(n) not in render_only):
            names.add(n.id)
    return names


def _selection_state(main, loop, k):
    pre_cut = loop.body[:k]
    carried = _loop_carried(pre_cut, loop)
    outputs = _selection_outputs(main, loop)
    globals_read = {n for n in (_loaded_names(ast.Module(body=pre_cut, type_ignores=[])) | outputs)
                    if n.isupper()}
    return carried | outputs | globals_read


def _mutations(stmts):
    """(имя, как) для каждой мутации в операторах."""
    found = []
    for st in stmts:
        for n in ast.walk(st):
            if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                for t in targets:
                    base = t
                    while isinstance(base, (ast.Subscript, ast.Attribute)):
                        base = base.value
                    if isinstance(base, ast.Name):
                        found.append((base.id, type(n).__name__))
            elif isinstance(n, ast.Delete):
                for t in n.targets:
                    base = t
                    while isinstance(base, (ast.Subscript, ast.Attribute)):
                        base = base.value
                    if isinstance(base, ast.Name):
                        found.append((base.id, "del"))
            elif (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr in MUTATING_METHODS):
                base = n.func.value
                while isinstance(base, (ast.Subscript, ast.Attribute)):
                    base = base.value
                if isinstance(base, ast.Name):
                    found.append((base.id, f".{n.func.attr}()"))
    return found


def test_exactly_one_cut_in_main():
    _tree, _main, cuts = _main_and_cut()
    assert len(cuts) == 1, f"точек среза SELECT_ONLY в main(): {len(cuts)}, ожидалась ровно 1"


def test_tail_after_cut_never_touches_selection_state():
    """Хвост итерации после среза — только рендер. Любая мутация состояния,
    которое отбор читает в следующей итерации, сделала бы --select-only
    отбором ДРУГИХ кадров, чем полный прогон."""
    _tree, main, cuts = _main_and_cut()
    loop, k = cuts[0]
    state = _selection_state(main, loop, k)
    offenders = [(name, how) for name, how in _mutations(loop.body[k + 1:]) if name in state]
    assert not offenders, (
        "после точки среза меняется состояние отбора — режим отбора разойдётся "
        f"с полным рендером: {offenders}")


def test_loop_carried_analysis_is_not_vacuous():
    """Защита защиты: если анализ ничего не считает переносимым, предыдущий
    тест зелёный по построению. Состояние, которое отбор ТОЧНО переносит
    между слотами, обязано быть найдено."""
    _tree, main, cuts = _main_and_cut()
    loop, k = cuts[0]
    state = _selection_state(main, loop, k)
    for must in ("luma_ema", "used_photo_ids", "recent_shot_sizes", "recent_media_types",
                 "shot_entries", "SOURCE_STATS"):
        assert must in state, f"анализ не увидел состояние отбора {must!r}"
    # и обратное: бухгалтерия рендера состоянием отбора не считается
    assert "pending_jobs" not in state, "очередь рендера ошибочно считается состоянием отбора"


def test_analysis_catches_an_injected_mutation():
    """Контроль: мутация отборного состояния, подложенная в хвост, ловится."""
    _tree, main, cuts = _main_and_cut()
    loop, k = cuts[0]
    state = _selection_state(main, loop, k)
    for injected in ("used_photo_ids.add('x')", "luma_ema = 0.0",
                     "recent_shot_sizes.append('wide')", "shot_entries[i] = {}",
                     "SOURCE_STATS.clear()"):
        tail = loop.body[k + 1:] + ast.parse(injected).body
        hit = [n for n, _h in _mutations(tail) if n in state]
        assert hit, f"подложенная мутация не поймана: {injected}"


# ---------------------------------------------------------------- поведение

@pytest.fixture
def episode(tmp_path):
    return build_episode(tmp_path / "sel_ep")


@needs_ffmpeg
def test_select_only_selects_without_rendering(episode):
    r = run_pipeline(episode, args=("--select-only",))
    assert r.returncode in (0, 2), f"отбор упал:\n{r.stdout[-3000:]}\n{r.stderr[-1500:]}"
    assert "ОТБОР ГОТОВ" in r.stdout, r.stdout[-2000:]

    shotlist = json.load(open(episode / "media_plan" / "shotlist.json", encoding="utf-8"))
    assert len(shotlist["shots"]) == N_BLOCKS

    assert not (episode / "final.mp4").exists(), "режим отбора собрал ролик"
    temp = episode / "temp_smart"
    clips = list(temp.glob("clip_*.mp4")) if temp.exists() else []
    assert not clips, f"режим отбора отрендерил клипы: {clips}"
    for name in RENDER_ARTIFACTS:
        assert not (episode / "media_plan" / name).exists(), \
            f"режим отбора записал артефакт рендера {name}"


@needs_ffmpeg
def test_select_only_never_overwrites_existing_render_artifacts(episode):
    """Эпизод уже отрендерен — прогон отбора не имеет права испортить его
    манифест рендера пустым."""
    mp = episode / "media_plan"
    mp.mkdir(exist_ok=True)
    sentinel = {"sentinel": "настоящий манифест прошлого рендера"}
    for name in RENDER_ARTIFACTS:
        (mp / name).write_text(json.dumps(sentinel, ensure_ascii=False), encoding="utf-8")
    r = run_pipeline(episode, args=("--select-only",))
    assert r.returncode in (0, 2), r.stdout[-3000:]
    for name in RENDER_ARTIFACTS:
        assert json.loads((mp / name).read_text(encoding="utf-8")) == sentinel, \
            f"режим отбора перезаписал {name}"


@needs_ffmpeg
def test_select_only_picks_exactly_what_the_full_render_picks(tmp_path):
    """Главное свойство режима: он отбирает РОВНО то же, что полный рендер.
    Два одинаковых эпизода в разных папках — чтобы кэш клипов полного
    прогона не превратил второй прогон в кэш-хиты."""
    full = build_episode(tmp_path / "full")
    sel = build_episode(tmp_path / "sel")
    rf = run_pipeline(full)
    rs = run_pipeline(sel, args=("--select-only",))
    assert rf.returncode in (0, 2), f"полный рендер упал:\n{rf.stdout[-3000:]}"
    assert rs.returncode in (0, 2), f"отбор упал:\n{rs.stdout[-3000:]}"

    def shots(ep):
        data = json.load(open(ep / "media_plan" / "shotlist.json", encoding="utf-8"))
        return data["shots"], data["gates"]

    full_shots, full_gates = shots(full)
    sel_shots, sel_gates = shots(sel)
    assert len(full_shots) == len(sel_shots) == N_BLOCKS
    for a, b in zip(full_shots, sel_shots):
        assert a == b, f"слот {a.get('index')}: полный рендер {a} против отбора {b}"
    assert full_gates == sel_gates, "шапка гейтов разошлась между режимами"
