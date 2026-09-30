#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Подпись кода отбора собирается сама (М6, 25.09).

Ручные списки функций в подписях отставали по построению: от адаптеров
фото и видео достижимы 161 функция pipeline_smart, 109 не входили ни в
одну подпись — правка любой из них на прогретом кэше не доходила до экрана."""
import importlib
import os
import sys
import textwrap

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import code_signature as cs  # noqa: E402


def _mod(tmp_path, name, body):
    (tmp_path / f"{name}.py").write_text(textwrap.dedent(body), encoding="utf-8")
    sys.path.insert(0, str(tmp_path))
    sys.modules.pop(name, None)
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(tmp_path))


BASE = '''
def entry(x):
    """Точка входа."""
    return helper(x) + judge(x)

def helper(x):
    # коментарий
    return x * 2

def judge(x):
    return x
'''


def _sig(tmp_path, body, name, stop_judge=False):
    m = _mod(tmp_path, name, body)
    return cs.signature([m.entry], (name,), stop=(m.judge,) if stop_judge else ())[0]


def test_a_deep_helper_change_changes_the_signature(tmp_path):
    a = _sig(tmp_path, BASE, "cs_a")
    b = _sig(tmp_path, BASE.replace("x * 2", "x * 3"), "cs_a")
    assert a != b


def test_comments_and_docstrings_do_not(tmp_path):
    a = _sig(tmp_path, BASE, "cs_b")
    b = _sig(tmp_path, BASE.replace("# коментарий", "# другой").replace("Точка входа.", "Иначе."),
             "cs_b")
    assert a == b


def test_code_behind_the_stop_belongs_to_its_own_signature(tmp_path):
    a = _sig(tmp_path, BASE, "cs_c", stop_judge=True)
    b = _sig(tmp_path, BASE.replace("return x\n", "return -x\n"), "cs_c", stop_judge=True)
    assert a == b


def test_pipeline_selection_reaches_the_ranking_code():
    sys.argv = ["pipeline_smart.py", REPO]
    import pipeline_smart as ps
    import selection_engine
    code = cs.reachable([ps.PhotoAdapter, ps.VideoAdapter, selection_engine.select],
                        ps.SELECTION_CODE_MODULES, stop=ps._judge_code_entries())
    names = {k.split(".", 1)[1] for k in code}
    for fn in ("_repick", "judge_rank", "relevance_rank_bucket", "frame_readable",
               "is_relevant_candidate", "stock_api_query"):
        assert fn in names, fn
    assert "judge_candidates" not in names, "код судьи — в своей подписи"
