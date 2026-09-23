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
import dataclasses
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
        ("первая", (("f", "log.append(1)", (1, 2)),)),
        ("вторая", (("f", "log.append(1)", (2, 2)),)),
        ("без номера", (("f", "log.append(1)", (1, 1)),)),
        ("число сменилось", (("f", "log.append(1)", (1, 3)),)),
    ))
    got = sf._anchor_lines(lines, tree)
    assert got["первая"]["line"] == 3 and got["вторая"]["line"] == 5
    assert "error" in got["без номера"], "дословный повтор без номера обязан теряться, а не брать первое"
    assert "error" in got["число сменилось"]


def test_two_resolving_forms_are_ambiguous_not_a_guess(monkeypatch):
    """Старая и новая форма одной ветки нашлись обе — харнесс не знает, какую
    из двух считать веткой, и честно говорит об этом."""
    src = "def old(a):\n    X.append(1)\n\ndef new(a):\n    record(1)\n"
    monkeypatch.setattr(sf, "BRANCH_ANCHORS", (
        ("ветка", sf._forms(("new", "old"), "record(1)", "X.append(1)")),))
    got = sf._anchor_lines(src.split("\n"), ast.parse(src))
    assert "неоднозначно" in got["ветка"]["error"]


def test_coverage_is_recomputed_from_raw_lines(tmp_path, monkeypatch):
    src = "def f(a):\n    if a:\n        X.append(1)\n"
    monkeypatch.setattr(sf, "BRANCH_ANCHORS", (("ветка", sf._forms(("f",), "X.append(1)")),))
    cov = sf.coverage_from(src, {"f": {3}})
    assert cov["ветка"]["state"] == "покрыто"
    assert sf.coverage_from(src, {"f": {2}})["ветка"]["state"] == "НЕ покрыто"


# ------------------------------------------------------------------ ожидания этапа

def _inputs(**per_slot):
    return {str(k[1:]): v for k, v in per_slot.items()}


def test_slot_may_differ_only_with_a_measured_cause():
    """Слот изменился при том же входе и без утечки в нём — провал; с
    изменившимся входом — законно, и отчёт называет, ЧТО изменилось."""
    a = _result([_shot(0), _shot(1), _shot(2)])
    a["slot_inputs"] = _inputs(s0={"luma_ema": "x"}, s1={"luma_ema": "x"}, s2={"luma_ema": "x"})
    b = _result([_shot(0), _shot(1, file_sha256="fixed"), _shot(2, file_sha256="shifted")])
    b["slot_inputs"] = _inputs(s0={"luma_ema": "x"}, s1={"luma_ema": "x"}, s2={"luma_ema": "y"})
    rep = sf.compare(a, b, {"slots": {"1": "вердикт чужой попытки"}})
    assert rep["ok"], rep
    assert [s["class"] for s in rep["slots"]] == ["СОВПАЛ", "ОЖИДАЕМО РАЗОШЁЛСЯ", "ВХОД ИЗМЕНИЛСЯ"]
    assert "luma_ema" in rep["slots"][2]["expected_reason"]
    b["slot_inputs"]["2"] = {"luma_ema": "x"}
    rep = sf.compare(a, b, {"slots": {"1": "вердикт чужой попытки"}})
    assert not rep["ok"] and rep["slots"][2]["class"] == "РАЗОШЁЛСЯ"


def test_leak_in_the_slot_itself_is_a_cause():
    """Отброшенная попытка несла резерв — по старой семантике он был бы
    применён; устранение утечки вправе изменить исход этого слота."""
    journal = [
        {"record": "attempt", "attempt_id": "v", "index": 0, "effects": [{"effect": "reserve_hash"}],
         "verdicts": []},
        {"record": "attempt", "attempt_id": "p", "index": 0, "effects": [], "verdicts": []},
        {"record": "slot", "index": 0, "shown": "p", "decisive": "p", "attempts": ["v", "p"]},
    ]
    a = _result([_shot(0)])
    b = _result([_shot(0, file_sha256="other")], reports={"run_journal.jsonl": journal})
    rep = sf.compare(a, b, {"reports": {"run_journal.jsonl": "журнал появился"}})
    assert rep["ok"] and rep["slots"][0]["class"] == "УТЕЧКА УСТРАНЕНА"
    assert sf.journal_leak_slots(b) == {0}


def test_clock_decision_outside_the_record_breaks_equivalence():
    a = _result([_shot(0)])
    b = _result([_shot(0)])
    b["net"]["time_decisions"] = {"divergences": ["get|0|u"]}
    assert not sf.compare(a, b)["ok"], "старый формат без слота — судить не по чему"
    b["net"]["time_decisions"] = {"divergences": [{"key": "get|0|u", "slot": 0}]}
    assert not sf.compare(a, b)["ok"], "слот без причины"


def test_clock_decision_inside_a_slot_with_cause_is_allowed():
    a = _result([_shot(0), _shot(1)])
    b = _result([_shot(0), _shot(1, file_sha256="x")])
    b["net"]["time_decisions"] = {"divergences": [{"key": "get|0|u", "slot": 1}]}
    assert sf.compare(a, b, {"slots": {"1": "fix"}})["ok"]
    b["net"]["time_decisions"] = {"divergences": [{"key": "get|0|u", "slot": None}]}
    assert not sf.compare(a, b, {"slots": {"1": "fix"}})["ok"]


@pytest.mark.parametrize("bad", [{"slots": {"1": {"downstream_of": 0}}}, {"slots": {"1": ""}}])
def test_expectation_is_a_named_reason(bad):
    with pytest.raises(ValueError):
        sf.parse_expect(bad)


def test_expected_report_must_actually_differ():
    a = _result([_shot(0)], reports={"run_journal.jsonl": [{"record": "slot", "index": 0,
                                                           "attempts": [], "shown": None,
                                                           "decisive": None}]})
    same = json.loads(json.dumps(a))
    exp = {"reports": {"run_journal.jsonl": "журнал появился"}}
    assert not sf.compare(a, same, exp)["ok"], "заявленное изменение отчёта не произошло"
    changed = json.loads(json.dumps(a))
    changed["reports"]["run_journal.jsonl"][0]["outcome"] = "shown"
    assert sf.compare(a, changed, exp)["ok"]


def _net(calls, slots, divergences=()):
    return {"calls_by_key": calls, "slots_by_key": slots, "divergences": list(divergences)}


def test_network_may_differ_only_inside_expected_slots():
    """Правка слота 1 законно меняет его скачивания — но если сеть поехала в
    слоте 0 или вне слотового цикла, это уже не та правка."""
    base = _result([_shot(0), _shot(1)], net=_net({"k": 1}, {"k": {"1": 1}}))
    inside = _result([_shot(0), _shot(1, file_sha256="x")],
                     net=_net({"k": 2}, {"k": {"1": 2}},
                              [{"method": "GET", "url": "u", "seq": 0, "slot": 1, "served": "live"}]))
    exp = {"slots": {"1": "fix"}}
    assert sf.compare(base, inside, exp)["ok"]
    outside = _result([_shot(0), _shot(1, file_sha256="x")],
                      net=_net({"k": 1, "j": 1}, {"k": {"1": 1}, "j": {"0": 1}}))
    rep = sf.compare(base, outside, exp)
    assert not rep["ok"] and rep["net_unexpected"] == ["j"]
    unlabelled = _result([_shot(0), _shot(1, file_sha256="x")],
                         net=_net({"k": 1, "j": 1}, {"k": {"1": 1}, "j": {"none": 1}}))
    assert not sf.compare(base, unlabelled, exp)["ok"], "обращение вне слотового цикла не приписано правке"
    stray = _result([_shot(0), _shot(1, file_sha256="x")], net=_net(
        {"k": 1}, {"k": {"1": 1}}, [{"method": "GET", "url": "u", "seq": 0, "slot": 0}]))
    assert not sf.compare(base, stray, exp)["ok"]


def test_shared_address_is_judged_per_slot():
    """Реальный случай эпизода 93: адрес превью звучал в 18 слотах, новый код
    добавил ОДНО обращение в слоте с причиной. Раньше адрес целиком считался
    разошедшимся «вне слотов с причиной», хотя в остальных 18 слотах число
    обращений совпало."""
    base = _result([_shot(0), _shot(1)], net=_net({"k": 2}, {"k": {"0": 1, "1": 1}}))
    extra_in_cause = _result([_shot(0), _shot(1, file_sha256="x")],
                             net=_net({"k": 3}, {"k": {"0": 1, "1": 2}}))
    rep = sf.compare(base, extra_in_cause, {"slots": {"1": "fix"}})
    assert rep["ok"] and rep["net_calls_differ"]["k"]["slots_changed"] == ["1"]
    extra_elsewhere = _result([_shot(0), _shot(1, file_sha256="x")],
                              net=_net({"k": 3}, {"k": {"0": 2, "1": 1}}))
    assert sf.compare(base, extra_elsewhere, {"slots": {"1": "fix"}})["net_unexpected"] == ["k"]
    untagged = _result([_shot(0), _shot(1, file_sha256="x")], net=_net({"k": 3}, {}))
    assert sf.compare(base, untagged, {"slots": {"1": "fix"}})["net_unexpected"] == ["k"], \
        "разное число обращений без разметки слотами — судить не по чему"


def test_slot_loop_range_is_the_loop_that_resolves_slots():
    src = ("def main():\n"
           "    for i in range(3):\n"
           "        prep(i)\n"
           "    for i, b in enumerate(blocks):\n"
           "        for j in range(2):\n"
           "            pass\n"
           "        RESOLVED_SLOTS_THIS_RUN.add(i)\n"
           "        fetch(i)\n"
           "    done()\n")
    assert sf.slot_loop_range(ast.parse(src)) == (4, 8, 5)


def test_slot_report_may_differ_only_in_slots_with_a_cause():
    base = {"slots_evaluated_this_run": [0, 1], "misses": [{"index": 0}, {"index": 1}]}
    fixed = {"slots_evaluated_this_run": [0, 1], "misses": [{"index": 0}]}
    assert sf.slot_report_stray(base, fixed, allowed={1}) == []
    assert sf.slot_report_stray(base, fixed, allowed={0}) == [1]
    other_field = dict(fixed, slots_evaluated_this_run=[0])
    assert sf.slot_report_stray(base, other_field, allowed={1}) is None
    a = _result([_shot(0), _shot(1, file_sha256="x")], reports={"r.json": base})
    b = _result([_shot(0), _shot(1, file_sha256="y")], reports={"r.json": fixed})
    assert sf.compare(a, b, {"slots": {"1": "исправление"}})["ok"]
    assert not sf.compare(a, b, {"slots": {"0": "не тот слот"}})["ok"]


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



def test_module_is_executed_from_the_snapshot_not_from_disk(tmp_path):
    """Файл на диске поменялся после снимка — исполняется снимок."""
    path = tmp_path / "m.py"
    path.write_text("def f():\n    return 'новое на диске'\n")
    mod = sf.load_module_from_source("_freeze_probe_mod", str(path),
                                     "def f():\n    return 'снимок'\n")
    try:
        import inspect
        assert mod.f() == "снимок"
        assert sys.modules["_freeze_probe_mod"] is mod
        # подписи кэша пайплайна строятся inspect.getsource — он обязан видеть
        # исполняемый снимок, а не файл на диске
        assert "снимок" in inspect.getsource(mod.f)
    finally:
        sys.modules.pop("_freeze_probe_mod", None)
        import linecache
        linecache.cache.pop(str(path), None)


def test_any_list_of_indexed_records_is_compared_by_slot():
    """Не только misses: например, clips отчёта Режиссёра."""
    base = {"enabled": False, "clips": [{"index": 0, "d": "a"}, {"index": 1, "d": "absorbed"}]}
    new = {"enabled": False, "clips": [{"index": 0, "d": "a"}, {"index": 1, "d": "skipped"}]}
    assert sf.slot_report_stray(base, new, allowed={1}) == []
    assert sf.slot_report_stray(base, new, allowed=set()) == [1]


def test_returncode_may_change_only_when_named():
    a = _result([_shot(0)], rc=2)
    b = _result([_shot(0)], rc=0)
    assert not sf.compare(a, b)["ok"]
    assert sf.compare(a, b, {"returncode": "пустых слотов больше нет"})["ok"]
    assert not sf.compare(a, json.loads(json.dumps(a)), {"returncode": "заявлено, но не изменилось"})["ok"]


def _att(kind, **fields):
    return {"kind": kind, "fields": fields}


def test_changed_attempt_request_is_a_cause_only_when_declared():
    """Спасающая фото-попытка теперь получает бриф фразы: её запрос
    изменился при том же входе слота. Причина засчитывается, только если
    этап объявил поле заранее; необъявленное — ошибка передачи запроса."""
    a = _result([_shot(0, file_sha256="old")])
    b = _result([_shot(0, file_sha256="new")])
    a["slot_attempts"] = {"0": [_att("video", query="q"), _att("photo", query="q", shot_brief="n")]}
    b["slot_attempts"] = {"0": [_att("video", query="q"), _att("photo", query="q", shot_brief="b")]}
    rep = sf.compare(a, b, {"attempt_fields": {"shot_brief": "бриф в спасающей попытке"}})
    assert rep["ok"] and rep["slots"][0]["class"] == "ЗАПРОС ПОПЫТКИ ИЗМЕНИЛСЯ"
    assert not sf.compare(a, b)["ok"], "необъявленное изменение запроса объяснило само себя"
    b["slot_attempts"]["0"][1]["fields"]["query"] = "other"
    assert not sf.compare(a, b, {"attempt_fields": {"shot_brief": "x"}})["ok"], \
        "изменилось и необъявленное поле — причина не засчитывается"


def test_attempts_after_a_kind_mismatch_are_consequences():
    xa = [_att("video", q="1"), _att("photo", q="1")]
    xb = [_att("photo", q="2")]
    assert sf.attempt_input_changes(xa, xb) == set()


def test_old_and_new_call_shapes_map_to_the_same_fields():
    """Аргументы добытчика этапа 1 и поля SlotRequest — одни и те же
    величины под разными именами; отпечатки обязаны совпасть."""
    import types
    loc_old = {"query": "q", "index": 3, "used_ids": {1, 2}, "used_hashes": ["h"],
               "is_opening_shot": False, "extra_queries": None, "shot_brief": "b",
               "block_text": "t", "director_score_fn": lambda x: x}
    kind, old = sf.attempt_fields_from_call("_select_photo", loc_old)
    req = types.SimpleNamespace(query="q", index=3, used_photo_ids={2, 1}, used_hashes=["h"],
                                is_opening=False, extra_queries=(), shot_brief="b",
                                block_text="t", director_score_fn=print, recent_sizes=None,
                                target_luma=None, director_assist=None, director_report=None,
                                text_key=None, arbiter_text=None)
    kind2, new = sf.attempt_fields_from_call("select_media", {"request": req, "kind": "photo"})
    assert kind == kind2 == "photo" and old == new


def test_journal_like_list_report_is_compared_by_slot():
    a = [{"record": "slot", "index": 0, "outcome": "shown"},
         {"record": "slot", "index": 1, "outcome": "absorbed"}]
    b = [{"record": "slot", "index": 0, "outcome": "shown"},
         {"record": "slot", "index": 1, "outcome": "shown"}]
    assert sf.slot_report_stray(a, b, allowed={1}) == []
    assert sf.slot_report_stray(a, b, allowed=set()) == [1]


def _contrib(**won):
    return {"sources": {k: {"won": v, "offered": 10 + v} for k, v in won.items()}}


def test_contribution_report_is_judged_against_the_screen():
    """Сводный счёт источников по слотам не раскладывается. Законно его
    расхождение, только когда сменились слоты с причиной И побед у каждого
    источника ровно столько, сколько его кадров на экране. Реальный случай
    эпизода 93: старый код писал Pexels 24 победы при 23 кадрах на экране."""
    a = _result([_shot(0), _shot(1)], reports={sf.CONTRIBUTION_REPORT: _contrib(pexels=3)})
    moved = [_shot(0), _shot(1, provider="pixabay", file_sha256="x")]
    exp = {"slots": {"1": "fix"}}
    good = _result(moved, reports={sf.CONTRIBUTION_REPORT: _contrib(pexels=1, pixabay=1)})
    rep = sf.compare(a, good, exp)
    assert rep["ok"]
    assert rep["reports_differ"][sf.CONTRIBUTION_REPORT]["won_vs_screen"] == {
        "a": {"pexels": [3, 2]}, "b": {}}
    lying = _result(moved, reports={sf.CONTRIBUTION_REPORT: _contrib(pexels=2, pixabay=1)})
    assert sf.CONTRIBUTION_REPORT in sf.compare(a, lying, exp)["reports_unexpected"]
    unmoved = _result([_shot(0), _shot(1)], reports={sf.CONTRIBUTION_REPORT: _contrib(pexels=2)})
    assert sf.CONTRIBUTION_REPORT in sf.compare(a, unmoved)["reports_unexpected"], \
        "слоты не менялись — сводный счёт обязан совпасть"


def test_gates_header_copy_of_contribution_is_judged_with_the_report():
    a = _result([_shot(0)], gates={"g": 1, "source_contribution": {"pexels": 1}})
    b = _result([_shot(0)], gates={"g": 1, "source_contribution": {"pexels": 2}})
    assert not sf.compare(a, b)["gates_differ"]
    c = _result([_shot(0)], gates={"g": 2, "source_contribution": {"pexels": 1}})
    assert sf.compare(a, c)["gates_differ"]


def test_pool_capture_records_the_pool_the_ranking_receives(tmp_path):
    """Разметка качества обязана смотреть на ТОТ пул, из которого выбирает
    код, — перехват стоит на входе ранжирования, а решение не меняется."""
    import types
    import selection_engine as se

    class Adapter:
        kind = "photo"

        def choose(self, request, pool, cf):
            return f"chosen:{pool[0]['id']}"

    fake = types.SimpleNamespace(
        PHOTO_ADAPTER=Adapter(),
        candidate_channel=lambda c: "pexels",
        pexels_candidate_text=lambda c: c.get("alt"),
        candidate_probe_url=lambda c: f"https://x/{c['id']}.jpg",
    )
    path = str(tmp_path / "pools.jsonl")
    assert sf.install_pool_capture(fake, path)
    fields = {f.name: None for f in dataclasses.fields(se.SlotRequest)}
    fields.update(index=3, query="q", extra_queries=("e",), shot_brief="b", block_text="t")
    req = se.SlotRequest(**fields)
    pool = [{"id": 7, "alt": "sword", "_origin_query": "q"}, {"id": 8, "alt": "axe"}]
    assert fake.PHOTO_ADAPTER.choose(req, pool, "cf") == "chosen:7"
    rec = json.loads(open(path, encoding="utf-8").read())
    assert rec["index"] == 3 and rec["shot_brief"] == "b" and rec["extra_queries"] == ["e"]
    assert [r["id"] for r in rec["pool"]] == [7, 8]
    assert rec["pool"][0]["probe_url"] == "https://x/7.jpg" and rec["pool"][0]["via"] == "q"


def test_pool_capture_is_absent_for_code_without_the_engine(tmp_path):
    import types
    assert not sf.install_pool_capture(types.SimpleNamespace(), str(tmp_path / "p.jsonl"))
    assert not (tmp_path / "p.jsonl").exists()
