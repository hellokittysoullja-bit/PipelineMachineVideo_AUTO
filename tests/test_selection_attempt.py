#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Этап 1 перестройки отбора: состояние эпизода меняет только коммит.

Две части. Первая — сам модуль попытки (scripts/selection_attempt.py):
коммит применяет записанное и переносит файлы, отказ не применяет ничего,
закрытая попытка не принимает записей. Вторая — машинные инварианты по
исходнику pipeline_smart.py: ни одна строка вне единственной точки
применения эффектов не мутирует анти-дубль, ритм, историю, счёт побед
источников, лицензионный манифест и отчёты промахов. Инвариант по AST, а
не по памяти автора: прежний механизм (снимок/откат вердиктов) держался на
том, что кто-то помнит все места, куда утекает запись, — и один забытый
список стоил эпизоду слота.
"""
import ast
import os
import sys
import threading

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS)
import selection_attempt as sa  # noqa: E402

PIPELINE_SRC = open(os.path.join(SCRIPTS, "pipeline_smart.py"), encoding="utf-8").read()
PIPELINE_TREE = ast.parse(PIPELINE_SRC)


def _collect_effects():
    applied = []

    def apply(kind, *args):
        applied.append((kind, args))
        if kind == "add":
            args[0].add(args[1])
        elif kind == "append":
            args[0].append(args[1])
    return applied, apply


# ------------------------------------------------------------------ попытка

def test_commit_applies_effects_in_recorded_order_and_moves_files(tmp_path):
    cache = tmp_path / "cache"
    ids, hashes = set(), []
    att = sa.Attempt(3, "photo", str(tmp_path / "staging"))
    with sa.activate(att):
        staged = sa.stage_path(str(cache / "0003_x.jpg"))
        with open(staged, "wb") as f:
            f.write(b"img")
        with open(staged + ".meta.json", "w") as f:
            f.write("{}")
        sa.record_effect("add", ids, 42)
        sa.record_effect("append", hashes, "h1")
        sa.record_effect("append", hashes, "h2")
    att.media = staged
    applied, apply = _collect_effects()
    final = att.commit(apply)
    assert final == str(cache / "0003_x.jpg")
    assert (cache / "0003_x.jpg").read_bytes() == b"img"
    assert (cache / "0003_x.jpg.meta.json").exists(), "sidecar обязан переехать вместе с кадром"
    assert ids == {42} and hashes == ["h1", "h2"]
    assert [k for k, _a in applied] == ["add", "append", "append"]
    assert not (tmp_path / "staging" / att.attempt_id).exists()


def test_discard_applies_nothing_and_leaves_no_file(tmp_path):
    """Выброшенный кадр не должен оказаться в кэше: иначе следующий прогон
    вернул бы его кэш-хитом без единого вердикта (так и было)."""
    cache = tmp_path / "cache"
    ids = set()
    att = sa.Attempt(3, "photo", str(tmp_path / "staging"))
    with sa.activate(att):
        staged = sa.stage_path(str(cache / "0003_x.jpg"))
        open(staged, "wb").write(b"img")
        sa.record_effect("add", ids, 42)
    att.discard()
    assert ids == set()
    assert not (cache / "0003_x.jpg").exists()
    assert not os.path.exists(staged)


def test_closed_attempt_refuses_writes(tmp_path):
    att = sa.Attempt(0, "photo", str(tmp_path))
    att.discard()
    for write in (lambda: att.verdict("stock", {}), lambda: att.effect("add", set(), 1),
                  lambda: att.stage(str(tmp_path / "f.jpg")), att.discard,
                  lambda: att.commit(lambda *a: None)):
        with pytest.raises(sa.AttemptStateError):
            write()


def test_commit_from_another_thread_is_refused(tmp_path):
    att = sa.Attempt(0, "photo", str(tmp_path))
    errors = []

    def other():
        try:
            att.commit(lambda *a: None)
        except sa.AttemptStateError as e:
            errors.append(e)
    t = threading.Thread(target=other)
    t.start()
    t.join()
    assert errors and att.state == sa.OPEN


def test_recording_outside_an_attempt_is_an_error_not_a_silent_apply():
    """Второй путь изменения состояния в обход коммита — ровно то, что
    модуль убирает; «применить сразу» здесь было бы им."""
    assert sa.current() is None
    with pytest.raises(sa.AttemptStateError):
        sa.record_effect("add", set(), 1)
    with pytest.raises(sa.AttemptStateError):
        sa.record_verdict("stock", {})


def test_nested_attempt_is_refused(tmp_path):
    a, b = sa.Attempt(0, "video", str(tmp_path)), sa.Attempt(0, "photo", str(tmp_path))
    with sa.activate(a):
        with pytest.raises(sa.AttemptStateError):
            with sa.activate(b):
                pass


def test_exception_inside_fetcher_discards_the_attempt(tmp_path):
    att = sa.Attempt(0, "photo", str(tmp_path / "st"))

    def boom():
        path = sa.stage_path(str(tmp_path / "c" / "f.jpg"))
        open(path, "wb").write(b"x")
        raise RuntimeError("сбой")
    with pytest.raises(RuntimeError):
        sa.run_attempt(att, boom)
    assert att.state == sa.DISCARDED
    assert not (tmp_path / "st" / att.attempt_id).exists()
    assert sa.current() is None


def test_sweep_removes_orphaned_attempt_dirs(tmp_path):
    root = tmp_path / "staging"
    (root / "00001-3-photo" / "d0").mkdir(parents=True)
    (root / "00001-3-photo" / "d0" / "x.jpg").write_bytes(b"x")
    sa.sweep_orphans(str(root))
    assert not root.exists()


def test_attempt_ids_are_local_to_the_slot(tmp_path):
    """Номер попытки считается внутри слота: лишняя попытка одного слота не
    сдвигает id попыток других слотов (иначе журналы двух прогонов
    расходились бы везде после первого изменившегося слота)."""
    a = sa.Attempt(101, "photo", str(tmp_path))
    b = sa.Attempt(101, "video", str(tmp_path))
    c = sa.Attempt(102, "photo", str(tmp_path))
    na = int(a.attempt_id.rsplit("-", 1)[1])
    assert a.attempt_id.startswith("101-photo-") and b.attempt_id == f"101-video-{na + 1}"
    assert c.attempt_id == "102-photo-1"


# ------------------------------------------------------------------ инварианты

LEDGER_NAMES = {"used_ids", "used_hashes", "recent_sizes", "used_photo_ids",
                "used_video_ids", "used_photo_hashes", "recent_shot_sizes",
                "recent_media_types", "recent_semantic_tags"}
REPORT_LISTS = {"RELEVANCE_GATE_MISSES", "STOCK_EXHAUSTED_MISSES", "ARBITER_REJECTED_ALL",
                "SMART_VETO_MISSES", "DIRECTOR_RELEVANCE_MISSES"}
MUTATORS = {"add", "append", "extend", "insert", "remove", "discard", "pop", "clear",
            "update", "__setitem__", "__delitem__"}


def _enclosing_functions(tree):
    """узел -> имя ближайшей функции-владельца."""
    owner = {}

    def walk(node, fname):
        for child in ast.iter_child_nodes(node):
            name = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else fname
            owner[child] = name
            walk(child, name)
    walk(tree, "<module>")
    return owner


OWNER = _enclosing_functions(PIPELINE_TREE)


def _mutations_of(names):
    """(функция, строка, имя) — каждая мутация контейнера с одним из имён:
    вызов мутирующего метода, del по срезу/индексу, присваивание по
    индексу, += ."""
    out = []
    for node in ast.walk(PIPELINE_TREE):
        target = None
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in MUTATORS and isinstance(node.func.value, ast.Name)):
            target = node.func.value.id
        elif isinstance(node, ast.Delete):
            for t in node.targets:
                if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name):
                    target = t.value.id
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name):
                    target = t.value.id
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            target = node.target.id
        if target in names:
            out.append((OWNER.get(node), node.lineno, target))
    return out


def test_episode_state_is_mutated_only_by_the_commit_applier():
    offenders = [m for m in _mutations_of(LEDGER_NAMES)]
    assert not offenders, (
        "анти-дубль/ритм/история меняются в обход коммита (apply_selection_effect): "
        f"{offenders}")


def test_miss_reports_are_filled_only_by_the_projection():
    offenders = _mutations_of(REPORT_LISTS)
    assert not offenders, f"отчёты промахов пишутся мимо close_slot: {offenders}"


def _calls(func_name):
    for node in ast.walk(PIPELINE_TREE):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)
            if name == func_name:
                yield node


def test_source_win_and_license_only_through_the_applier():
    for node in _calls("_source_bump"):
        field = node.args[1].value if len(node.args) > 1 and isinstance(node.args[1], ast.Constant) else None
        if field == "won":
            assert OWNER[node] == "apply_selection_effect", (
                f"победа источника засчитана мимо коммита, строка {node.lineno}")
    owners = {OWNER[n] for n in _calls("log_candidate_license")}
    assert owners == {"apply_selection_effect"}, owners


def test_only_close_slot_commits_or_discards():
    for method in ("commit", "discard"):
        owners = {OWNER[n] for n in _calls(method)
                  if isinstance(n.func, ast.Attribute)}
        assert owners <= {"close_slot"}, (method, owners)


def _frame_writes(fn):
    writers = []
    for n in ast.walk(fn):
        if not isinstance(n, ast.Call):
            continue
        name = n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", None)
        args = [a for a in n.args if isinstance(a, ast.Name)]
        if name in ("download", "replace", "write_media_sidecar", "atomic_url_download") and \
                any(a.id == "cf" for a in args):
            writers.append(n.lineno)
    return writers


def _stage_assignments(fn, target):
    return [n.lineno for n in ast.walk(fn)
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == "stage_path"
            and any(isinstance(t, ast.Name) and t.id == target for t in n.targets)]


def test_video_fetcher_writes_only_into_its_attempt():
    """Каждая запись файла кадра в добытчике идёт ПОСЛЕ перевода пути кэша
    в стейджинг попытки: запись раньше этой строки — это файл в кэше до
    решения, то есть ровно воскрешение выброшенного кадра."""
    fn = next(n for n in PIPELINE_TREE.body
              if isinstance(n, ast.FunctionDef) and n.name == "_select_video")
    stage_lines = _stage_assignments(fn, "cf")
    assert len(stage_lines) == 1, stage_lines
    writers = _frame_writes(fn)
    assert writers, "не найдено ни одной записи кадра — проверка пуста"
    assert all(line > stage_lines[0] for line in writers), (stage_lines, writers)


def test_engine_hands_the_adapter_only_a_staged_path():
    """Ядро переводит путь кэша в стейджинг ДО того, как отдаёт его адаптеру,
    а адаптер фото пишет файл кадра только в переданный ему путь и сам
    стейджинг не вычисляет (иначе мог бы вычислить не тот)."""
    engine = ast.parse(open(os.path.join(SCRIPTS, "selection_engine.py"), encoding="utf-8").read())
    select = next(n for n in engine.body if isinstance(n, ast.FunctionDef) and n.name == "select")
    staged = _stage_assignments(select, "cf")
    assert len(staged) == 1
    choose_calls = [n.lineno for n in ast.walk(select) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute) and n.func.attr == "choose"]
    assert choose_calls and all(c > staged[0] for c in choose_calls)
    assert not _frame_writes(select), "ядро само не пишет файл кадра"
    adapter = next(n for n in PIPELINE_TREE.body
                   if isinstance(n, ast.ClassDef) and n.name == "PhotoAdapter")
    methods = {m.name: m for m in adapter.body if isinstance(m, ast.FunctionDef)}
    assert _frame_writes(methods["choose"]), "проверка пуста: адаптер фото не пишет кадр"
    assert not _stage_assignments(methods["choose"], "cf")
    for name, m in methods.items():
        if name != "choose":
            assert not _frame_writes(m), f"{name} пишет файл кадра до решения"


def test_the_invariant_checks_are_not_vacuous():
    """Контроль самих проверок: внедрённая мутация обязана ловиться."""
    global PIPELINE_TREE, OWNER
    saved = PIPELINE_TREE, OWNER
    try:
        PIPELINE_TREE = ast.parse("def f(used_ids, recent_media_types):\n"
                                  "    used_ids.add(1)\n"
                                  "    del recent_media_types[:-6]\n"
                                  "    RELEVANCE_GATE_MISSES.append({})\n")
        OWNER = _enclosing_functions(PIPELINE_TREE)
        assert len(_mutations_of(LEDGER_NAMES)) == 2
        assert len(_mutations_of(REPORT_LISTS)) == 1
    finally:
        PIPELINE_TREE, OWNER = saved


# ------------------------------------------------------------------ close_slot

@pytest.fixture
def ps(tmp_path, monkeypatch):
    import pipeline_smart as _ps
    monkeypatch.setattr(_ps, "TEMP_FOLDER", str(tmp_path / "temp_smart"))
    for name in REPORT_LISTS:
        monkeypatch.setattr(_ps, name, [])
    monkeypatch.setattr(_ps, "RUN_JOURNAL", [])
    monkeypatch.setattr(_ps, "SOURCE_STATS", {})
    return _ps


def _fetched(ps, tmp_path, index, kind, ids, hashes, pid, h, verdicts=()):
    """Попытка, которая «добыла» файл, как это делает добытчик: файл в
    стейджинге, резервы и вердикты — записи."""
    att = ps.new_attempt(index, kind)
    with sa.activate(att):
        path = sa.stage_path(str(tmp_path / "temp_smart" / "cache" / f"{index:04d}_{kind}.bin"))
        open(path, "wb").write(kind.encode())
        sa.record_effect("reserve_id", ids, pid)
        sa.record_effect("reserve_hash", hashes, h)
        sa.record_effect("source_won", ps.candidate_channel(pid))
        for v in verdicts:
            sa.record_verdict(*v)
    att.media = path
    return att


def test_rescued_video_reserves_nothing(ps, tmp_path):
    """Видео, которое заменили фотографией, не должно занимать анти-дубль:
    раньше его хэш успевал лечь в used_hashes и обеднял следующие слоты."""
    ids, hashes = set(), []
    video = _fetched(ps, tmp_path, 2, "video", ids, hashes, "pixabay:1", "vh",
                     [("relevance", {"index": 2})])
    photo = _fetched(ps, tmp_path, 2, "photo", ids, hashes, "met:2", "ph")
    final = ps.close_slot(2, [video, photo], shown=photo, decisive=photo)
    assert ids == {"met:2"} and hashes == ["ph"]
    assert os.path.exists(final) and not os.path.exists(video.final_path(video.media))
    assert ps.RELEVANCE_GATE_MISSES == [], "вердикт отвергнутого видео попал в отчёт слота"
    assert ps.SOURCE_STATS.get("met", {}).get("won") == 1
    assert "pixabay" not in ps.SOURCE_STATS or not ps.SOURCE_STATS["pixabay"].get("won")


def test_absorbed_slot_reserves_nothing_and_caches_nothing(ps, tmp_path):
    ids, hashes = set(), []
    bad = _fetched(ps, tmp_path, 4, "photo", ids, hashes, "123", "h",
                   [("smart_veto", {"index": 4})])
    assert ps.close_slot(4, [bad], shown=None, decisive=bad, outcome="absorbed") is None
    assert ids == set() and hashes == []
    assert not os.path.exists(bad.final_path(bad.media)), \
        "поглощённый кадр лёг в кэш — следующий прогон вернул бы его кэш-хитом"
    assert ps.SMART_VETO_MISSES == [{"index": 4}], "причина поглощения обязана остаться в отчёте"


def test_late_card_reserves_nothing_of_the_rejected_frame(ps, tmp_path):
    ids, hashes = set(), []
    rejected = _fetched(ps, tmp_path, 6, "photo", ids, hashes, "77", "h77",
                        [("director", {"index": 6, "relevance": 0.01, "threshold": 0.06})])
    slot = [rejected]
    card = ps.given_attempt(slot, 6, "card", str(tmp_path / "card.jpg"))
    ps.close_slot(6, slot, shown=card, decisive=rejected, outcome="card")
    assert ids == set() and hashes == []
    assert ps.DIRECTOR_RELEVANCE_MISSES == [{"index": 6, "relevance": 0.01, "threshold": 0.06}]


def test_empty_slot_reports_every_attempt(ps, tmp_path):
    a = ps.new_attempt(1, "video")
    a.verdict("smart_veto", {"index": 1, "kind": "video"})
    b = ps.new_attempt(1, "photo")
    b.verdict("stock", {"index": 1, "kind": "photo"})
    ps.close_slot(1, [a, b], shown=None, decisive=None, outcome="missing")
    assert ps.SMART_VETO_MISSES and ps.STOCK_EXHAUSTED_MISSES


def test_journal_records_every_attempt_and_the_decision(ps, tmp_path):
    ids, hashes = set(), []
    video = _fetched(ps, tmp_path, 2, "video", ids, hashes, "v", "vh")
    photo = _fetched(ps, tmp_path, 2, "photo", ids, hashes, "p", "ph")
    ps.close_slot(2, [video, photo], shown=photo, decisive=photo)
    recs = ps.RUN_JOURNAL
    states = {r["attempt_id"]: r["state"] for r in recs if r["record"] == "attempt"}
    assert states == {video.attempt_id: "discarded", photo.attempt_id: "committed"}
    slot = [r for r in recs if r["record"] == "slot"]
    assert slot == [{"record": "slot", "index": 2, "outcome": "shown",
                     "shown": photo.attempt_id, "decisive": photo.attempt_id,
                     "attempts": [video.attempt_id, photo.attempt_id]}]
    # журнал сериализуем (контейнеры анти-дубля в него не попадают)
    import json
    json.dumps(recs)


def test_standalone_fetch_is_select_and_commit(ps, tmp_path):
    """Вызов добытчика вне main() — «выбрать и сразу принять» тем же
    close_slot; внутри чужой попытки — отказ, а не тихая смесь."""
    ids = set()

    def fetcher(query, index):
        path = sa.stage_path(str(tmp_path / "temp_smart" / "c" / f"{index}.jpg"))
        open(path, "wb").write(b"x")
        sa.record_effect("reserve_id", ids, "id1")
        return path
    final = ps.select_standalone("photo", 9, fetcher, "q", 9)
    assert final == str(tmp_path / "temp_smart" / "c" / "9.jpg") and os.path.exists(final)
    assert ids == {"id1"}
    other = ps.new_attempt(3, "video")
    with sa.activate(other):
        with pytest.raises(sa.AttemptStateError):
            ps.select_standalone("photo", 9, fetcher, "q", 9)
