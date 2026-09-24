#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ядро отбора (scripts/selection_engine.py) и запрос слота.

Правила, на которых стоит устранение «починили фото, забыли видео»:
в ядре нет ни одной ветки по виду медиа и ни одного импорта pipeline_smart
или адаптеров — видовой логике негде спрятаться; запрос слота один, без
значений по умолчанию, и строится в одном месте.
"""
import ast
import dataclasses
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS)
import selection_attempt as sa  # noqa: E402
import selection_engine as se  # noqa: E402

ENGINE_TREE = ast.parse(open(os.path.join(SCRIPTS, "selection_engine.py"), encoding="utf-8").read())
PIPELINE_TREE = ast.parse(open(os.path.join(SCRIPTS, "pipeline_smart.py"), encoding="utf-8").read())


# ------------------------------------------------------------------ устройство

def test_engine_imports_no_pipeline_and_no_adapter():
    imported = set()
    for n in ast.walk(ENGINE_TREE):
        if isinstance(n, ast.Import):
            imported |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            imported.add(n.module)
    assert "pipeline_smart" not in imported
    assert imported <= {"dataclasses", "itertools", "os", "selection_attempt"}, imported


def test_engine_has_no_branch_on_media_kind():
    """Ни сравнения со строкой вида медиа, ни чтения .kind в ядре."""
    for n in ast.walk(ENGINE_TREE):
        if isinstance(n, ast.Constant) and n.value in ("photo", "video"):
            raise AssertionError(f"вид медиа упомянут в ядре, строка {n.lineno}")
        if isinstance(n, ast.Attribute) and n.attr == "kind" and not isinstance(n.ctx, ast.Store):
            raise AssertionError(f"ядро читает вид медиа, строка {n.lineno}")


def test_slot_request_has_no_defaults_and_is_frozen():
    for f in dataclasses.fields(se.SlotRequest):
        assert f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING, f.name
    req = _request()
    with pytest.raises(dataclasses.FrozenInstanceError):
        req.query = "другое"


def test_slot_request_is_built_in_exactly_one_place():
    calls = [n for n in ast.walk(PIPELINE_TREE) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "SlotRequest"]
    assert len(calls) == 1
    owner = next(f.name for f in PIPELINE_TREE.body if isinstance(f, ast.FunctionDef)
                 and any(c is n for n in ast.walk(f) for c in calls))
    assert owner == "build_slot_request"


def test_every_attempt_of_a_slot_gets_the_same_request():
    """Спасающий вызов фото годами шёл без брифа: аргументы перечислялись
    заново в каждом вызове. Теперь все вызовы отбора в main() передают
    ОДНУ переменную запроса."""
    main = next(f for f in PIPELINE_TREE.body if isinstance(f, ast.FunctionDef) and f.name == "main")
    calls = [n for n in ast.walk(main) if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "fetch_in_attempt"]
    assert len(calls) >= 5
    for c in calls:
        assert isinstance(c.args[3], ast.Name) and c.args[3].id == "select_media"
        assert isinstance(c.args[4], ast.Name) and c.args[4].id == "request", ast.dump(c)


# ------------------------------------------------------------------ пул

def _request(**over):
    fields = {f.name: None for f in dataclasses.fields(se.SlotRequest)}
    fields.update(index=0, query="q", extra_queries=(), is_opening=False, director_assist=False)
    fields.update(over)
    return se.SlotRequest(**fields)


class FakeAdapter(se.MediaAdapter):
    kind = "fake"

    def __init__(self, tmp, per_query, brief=None, blocked=()):
        self.tmp, self.per_query, self.brief, self.blocked = tmp, per_query, brief, set(blocked)
        self.failures, self.offered, self.chosen_pool = [], [], None

    def cache_path(self, request):
        return os.path.join(self.tmp, "cache", f"{request.index}.bin")

    def cache_hit(self, request, path):
        return path

    def brief_query(self, request):
        return self.brief

    def sources(self, request, pq):
        return [[dict(c, _origin_query=pq) for c in lst] for lst in self.per_query.get(pq, [])]

    def filter_pool(self, request, pool):
        return [c for c in pool if c["id"] not in self.blocked]

    def note_offered(self, pool):
        self.offered.extend(c["id"] for c in pool)

    def choose(self, request, pool, cf):
        self.chosen_pool = [c["id"] for c in pool]
        with open(cf, "wb") as f:
            f.write(b"x")
        return cf

    def on_failure(self, request, exc):
        self.failures.append(exc)


def _run(adapter, request, tmp):
    att = sa.Attempt(request.index, "fake", str(tmp / "staging"))
    with sa.activate(att):
        return se.select(request, adapter)


def test_pool_order_brief_first_sources_and_queries_round_robin(tmp_path):
    per_query = {
        "brief": [[{"id": "b1"}]],
        "q": [[{"id": "m1"}, {"id": "m2"}], [{"id": "s1"}]],
        "x": [[{"id": "x1"}, {"id": "m1"}]],
    }
    ad = FakeAdapter(str(tmp_path), per_query, brief="brief")
    _run(ad, _request(extra_queries=("x", "q")), tmp_path)
    # запросы: brief, q, x; внутри q источники по кругу: m1, s1, m2
    assert ad.chosen_pool == ["b1", "m1", "x1", "s1", "m2"]


def test_pool_is_filtered_after_interleave_and_counted_once(tmp_path):
    ad = FakeAdapter(str(tmp_path), {"q": [[{"id": "a"}, {"id": "b"}]]}, blocked={"a"})
    _run(ad, _request(), tmp_path)
    assert ad.chosen_pool == ["b"] and ad.offered == ["b"]


def test_pool_emptied_by_the_filter_is_no_candidates_not_a_source_failure(tmp_path):
    """Раньше фото-путь падал на candidates[0] (IndexError), и это
    исключение засчитывалось как сбой API Pexels — шаг к отключению стока
    на весь эпизод из-за того, что все кандидаты оказались не по жанру."""
    ad = FakeAdapter(str(tmp_path), {"q": [[{"id": "a"}]]}, blocked={"a"})
    assert _run(ad, _request(), tmp_path) is None
    assert ad.failures == [] and ad.chosen_pool is None


def test_adapter_gets_a_staged_path_and_cache_hit_short_circuits(tmp_path):
    ad = FakeAdapter(str(tmp_path), {"q": [[{"id": "a"}]]})
    staged = _run(ad, _request(), tmp_path)
    assert "staging" in staged and not os.path.exists(ad.cache_path(_request()))
    os.makedirs(os.path.dirname(ad.cache_path(_request())), exist_ok=True)
    open(ad.cache_path(_request()), "wb").write(b"old")
    ad2 = FakeAdapter(str(tmp_path), {"q": [[{"id": "a"}]]})
    assert _run(ad2, _request(), tmp_path) == ad.cache_path(_request())
    assert ad2.chosen_pool is None


def test_exception_in_choose_goes_to_the_adapter(tmp_path):
    class Boom(FakeAdapter):
        def choose(self, request, pool, cf):
            raise RuntimeError("сбой")
    ad = Boom(str(tmp_path), {"q": [[{"id": "a"}]]})
    assert _run(ad, _request(), tmp_path) is None
    assert [str(e) for e in ad.failures] == ["сбой"]


def test_build_slot_request_rejects_a_missing_field():
    sys.argv = ["pipeline_smart.py", str(REPO_ROOT)]
    import pipeline_smart as ps
    fields = {f.name: None for f in dataclasses.fields(se.SlotRequest)}
    del fields["shot_brief"]
    with pytest.raises(TypeError):
        ps.build_slot_request(**fields)


def test_select_media_accepts_only_a_slot_request():
    import pipeline_smart as ps
    with pytest.raises(TypeError):
        ps.select_media({"query": "q", "index": 0}, "photo")


def test_phrase_queries_fill_the_pool_before_section_queries():
    """Спецификация фразы: её запросы — первый ярус пула, общие запросы
    секции — после них, а не вперемешку по кругу."""
    import types
    spec = {"queries": [{"q": "arrow armor", "for": ["c1"]}, {"q": "archer", "for": ["c1"]}]}
    req = types.SimpleNamespace(query="arrow armor", shot_spec=spec)
    qs = ["arrow armor", "archer", "knight armour", "medieval battle"]
    assert se.query_tiers(req, qs) == [["arrow armor", "archer"],
                                                      ["knight armour", "medieval battle"]]
    no_spec = types.SimpleNamespace(query="arrow armor", shot_spec=None)
    assert se.query_tiers(no_spec, qs) == [qs], "без спецификации — прежний порядок"
