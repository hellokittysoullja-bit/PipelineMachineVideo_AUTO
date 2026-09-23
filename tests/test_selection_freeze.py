#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Харнесс эквивалентности (scripts/selection_freeze.py) — сам по себе.

Харнесс — устройство, на котором стоит безопасность всей перестройки
отбора. Если он сам ошибается (называет эквивалентным то, что разошлось;
молча читает кэш; выдаёт секрет в запись), то каждое «эквивалентно» после
него ничего не стоит. Поэтому проверяется каждое его решение отдельно, без
настоящего эпизода и без сети.
"""
import ast
import hashlib
import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
import selection_freeze as sf  # noqa: E402


def _result(shots, reports=None, net=None, gates=None, rc=0):
    return {"shots": shots, "reports": reports or {}, "net": net or {"calls_by_key": {}},
            "gates": gates or {"g": 1}, "returncode": rc}


def _shot(i, **kw):
    base = {k: None for k in sf.SHOT_FIELDS}
    base.update(index=i, kind="photo", file=f"temp_smart/pexels_cache/{i:04d}.jpg",
                file_sha256=f"h{i}", provider="pexels", candidate_id=i)
    base.update(kw)
    return base


# ------------------------------------------------------------------ сравнение

def test_identical_runs_are_equivalent():
    a = _result([_shot(0), _shot(1)])
    rep = sf.compare(a, json.loads(json.dumps(a)))
    assert rep["ok"] and all(s["class"] == "СОВПАЛ" for s in rep["slots"])


def test_one_changed_file_hash_is_a_named_divergence():
    a = _result([_shot(0), _shot(1)])
    b = _result([_shot(0), _shot(1, file_sha256="other")])
    rep = sf.compare(a, b)
    assert not rep["ok"]
    s1 = next(s for s in rep["slots"] if s["index"] == 1)
    assert s1["class"] == "РАЗОШЁЛСЯ" and s1["fields"] == ["file_sha256"]


def test_expected_divergence_passes_only_with_its_reason():
    a = _result([_shot(0), _shot(1)])
    b = _result([_shot(0), _shot(1, candidate_id=99, file_sha256="x")])
    rep = sf.compare(a, b, expect={"1": "бриф вошёл в пул rescue-слота"})
    assert rep["ok"]
    assert next(s for s in rep["slots"] if s["index"] == 1)["class"] == "ОЖИДАЕМО РАЗОШЁЛСЯ"


def test_expected_divergence_that_did_not_happen_fails():
    """Этап заявил исправление слота, а слот не изменился — правка не
    сработала, и это провал, а не успех."""
    a = _result([_shot(0), _shot(1)])
    rep = sf.compare(a, json.loads(json.dumps(a)), expect={"1": "обязан смениться"})
    assert not rep["ok"]
    assert next(s for s in rep["slots"] if s["index"] == 1)["class"] == "НЕОЖИДАННО СОВПАЛ"


def test_missing_slot_is_a_divergence():
    rep = sf.compare(_result([_shot(0), _shot(1)]), _result([_shot(0)]))
    assert not rep["ok"]


@pytest.mark.parametrize("mutate", [
    lambda b: b["reports"].update({"smart_veto_report.json": {"misses": [{"index": 3}]}}),
    lambda b: b.update(gates={"g": 2}),
    lambda b: b.update(returncode=2),
    lambda b: b["net"]["calls_by_key"].update({"k": 2}),
    lambda b: b["net"].update(divergences=[{"method": "GET", "url": "u", "seq": 0}]),
])
def test_any_non_slot_difference_breaks_equivalence(mutate):
    """Победители совпали, но отчёт, шапка, код возврата или сеть разошлись —
    это не эквивалентность: новый код мог прийти к тем же кадрам другим
    путём (например, удвоив трафик)."""
    a = _result([_shot(0)], reports={"smart_veto_report.json": {"misses": []}},
                net={"calls_by_key": {"k": 1}})
    b = json.loads(json.dumps(a))
    b["net"].setdefault("divergences", [])
    mutate(b)
    assert not sf.compare(a, b)["ok"]


# ------------------------------------------------------------------ окружение

def test_env_names_are_derived_from_code():
    names = set(sf.pipeline_env_names())
    # читаются кодом напрямую
    for must in ("MUSEUM_CACHE_DIR", "OPENVERSE_CACHE_DIR", "EMB_CACHE_DIR", "PEXELS_API_KEY"):
        assert must in names, f"{must} не выведен из кода"
    # читаются через реестр флагов
    for must in ("SMART_RELEVANCE_VETO", "NEVER_SHOW_KNOWN_BAD", "VIDEO_PHOTO_RESCUE"):
        assert must in names, f"флаг реестра {must} не попал в снимок"


def test_secrets_are_snapshotted_as_fingerprints_only(monkeypatch):
    monkeypatch.setenv("PEXELS_API_KEY", "real-secret-value-123")
    snap = sf.env_snapshot()
    v = snap["PEXELS_API_KEY"]
    assert isinstance(v, dict) and set(v) == {"sha256"}
    assert v["sha256"] == hashlib.sha256(b"real-secret-value-123").hexdigest()
    assert "real-secret-value-123" not in json.dumps(snap)


def test_changed_secret_refuses_replay(monkeypatch, tmp_path):
    monkeypatch.setenv("PEXELS_API_KEY", "old-key-aaaaaaaa")
    snap = sf.env_snapshot()
    monkeypatch.setenv("PEXELS_API_KEY", "new-key-bbbbbbbb")
    with pytest.raises(SystemExit) as e:
        sf.child_env(snap, str(tmp_path), "0")
    assert "PEXELS_API_KEY" in str(e.value)


def test_child_env_is_hermetic(monkeypatch, tmp_path):
    """Кэши — свои у прогона, соль и офлайн моделей — заданы, переменная,
    отсутствовавшая при записи, не просачивается из текущего окружения."""
    monkeypatch.delenv("SMART_RELEVANCE_VETO", raising=False)
    snap = sf.env_snapshot()
    monkeypatch.setenv("SMART_RELEVANCE_VETO", "0")   # появилась ПОСЛЕ записи
    env = sf.child_env(snap, str(tmp_path), "7")
    if snap.get("SMART_RELEVANCE_VETO") is None:
        assert "SMART_RELEVANCE_VETO" not in env, "переменная после записи просочилась"
    for var in ("MUSEUM_CACHE_DIR", "OPENVERSE_CACHE_DIR", "EMB_CACHE_DIR"):
        assert env[var].startswith(str(tmp_path)), f"{var} указывает вне песочницы"
    assert env["PYTHONHASHSEED"] == "7"
    assert env["HF_HUB_OFFLINE"] == "1"


# ------------------------------------------------------------------ входы и якоря

def test_input_snapshot_skips_render_cache_and_videos(tmp_path):
    ep = tmp_path / "ep"
    (ep / "temp_smart").mkdir(parents=True)
    (ep / "temp_smart" / "clip.mp4").write_bytes(b"x")
    (ep / "media_plan").mkdir()
    (ep / "media_plan" / "world_card.json").write_text("{}")
    (ep / "script.txt").write_text("s")
    (ep / "final.mp4").write_bytes(b"x")
    (ep / "render_log5.txt").write_text("l")
    import shutil
    shutil.copytree(ep, tmp_path / "copy", ignore=sf._ignore)
    got = sorted(os.path.relpath(os.path.join(d, f), tmp_path / "copy")
                 for d, _ds, fs in os.walk(tmp_path / "copy") for f in fs)
    assert got == ["media_plan/world_card.json", "script.txt"]


def test_every_branch_anchor_resolves_exactly_once():
    """Якорь потерялся после правки кода — покрытие молча стало бы «НЕ
    покрыто» там, где ветка на самом деле исполняется. Поэтому потеря якоря
    роняет тест, а не отчёт."""
    src = open(sf.PIPELINE, encoding="utf-8").read()
    anchors = sf._anchor_lines(src.split("\n"), ast.parse(src))
    lost = {cls: a["error"] for cls, a in anchors.items() if "error" in a}
    assert not lost, f"якоря покрытия потеряны: {lost}"


# ------------------------------------------------------------------ трассировка

def test_tracer_sees_lines_inside_comprehensions():
    """На Python 3.11 включение исполняется в своём кадре. Трассировщик,
    следящий только за кодом самой функции, объявил бы ветку внутри
    включения непокрытой — ровно так «фильтр длины» числился непокрытым
    при 92 реально отсеянных кандидатах."""
    import types
    src = ("def target(xs):\n"
           "    kept = [x for x in xs\n"
           "            if x > 1]\n"
           "    return kept\n")
    mod = types.ModuleType("fake_pipeline")
    exec(compile(src, "<fake>", "exec"), mod.__dict__)
    old = sf.TRACED_FUNCTIONS
    sf.TRACED_FUNCTIONS = ("target",)
    try:
        hits = sf._install_tracer(mod)
        mod.target([1, 2, 3])
    finally:
        sys.settrace(None)
        import threading
        threading.settrace(None)
        sf.TRACED_FUNCTIONS = old
    assert 3 in hits["target"], f"строка внутри включения не увидена: {sorted(hits['target'])}"


def test_repeated_anchor_needs_declared_occurrence_count(monkeypatch):
    src = ("def f(a):\n"
           "    if a:\n"
           "        log.append(1)\n"
           "    else:\n"
           "        log.append(1)\n")
    tree = ast.parse(src)
    lines = src.split("\n")
    monkeypatch.setattr(sf, "BRANCH_ANCHORS", (
        ("первая", "f", "log.append(1)", (1, 2)),
        ("вторая", "f", "log.append(1)", (2, 2)),
        ("без номера", "f", "log.append(1)"),
        ("число сменилось", "f", "log.append(1)", (1, 3)),
    ))
    got = sf._anchor_lines(lines, tree)
    assert got["первая"]["line"] == 3 and got["вторая"]["line"] == 5
    assert "error" in got["без номера"], "дословный повтор без номера обязан теряться, а не брать первое"
    assert "error" in got["число сменилось"]


def test_run_media_is_pruned_only_after_hashes_are_taken(tmp_path, monkeypatch):
    """Порядок критичен: удалить файлы ДО подсчёта хэшей — и file_sha256
    победителя станет None у обоих прогонов, то есть сравнение перестанет
    видеть подмену файла и скажет «СОВПАЛ» там, где кадр другой."""
    freeze = tmp_path / "fr"
    (freeze / "input" / "media_plan").mkdir(parents=True)
    (freeze / "input" / "script.txt").write_text("s")
    (freeze / "meta.json").write_text(json.dumps({"env": {}}))
    payload = b"winner-bytes"

    def fake_run(cmd, cwd=None, env=None, stdout=None, stderr=None):
        sandbox = cmd[5]
        os.makedirs(os.path.join(sandbox, "temp_smart", "pexels_cache"))
        with open(os.path.join(sandbox, "temp_smart", "pexels_cache", "0.jpg"), "wb") as f:
            f.write(payload)
        with open(os.path.join(sandbox, "media_plan", "shotlist.json"), "w") as f:
            json.dump({"gates": {}, "shots": [{"index": 0, "file": "temp_smart/pexels_cache/0.jpg"}]}, f)

        class R:
            returncode = 0
        return R()

    monkeypatch.setattr(sf.subprocess, "run", fake_run)
    res = sf.run_pipeline(str(freeze), "replay", "t", "0")
    assert res["shots"][0]["file_sha256"] == hashlib.sha256(payload).hexdigest()
    sandbox = freeze / "runs" / "t" / "episode"
    assert not (sandbox / "temp_smart").exists(), "скачанное прогоном не удалено"
    assert (sandbox / "media_plan" / "shotlist.json").exists(), "отчёты отбора удалены"

    res2 = sf.run_pipeline(str(freeze), "replay", "t2", "0", keep_media=True)
    assert (freeze / "runs" / "t2" / "episode" / "temp_smart").exists()
    assert res2["shots"][0]["file_sha256"] == res["shots"][0]["file_sha256"]
