"""Частичный прогон не должен стирать аудит эпизода.

Фон (реальный, измеренный случай 04.09): после точечного ре-рендера 4 слотов из
165 файлы relevance_gate_report.json / stock_exhausted_report.json /
director_relevance_report.json оказались ПУСТЫМИ — при том что в готовом ролике
было 10 явно бракованных кадров. Отчёты переписывались целиком списком промахов
только текущего прогона, а 161 слот шёл кэш-хитом и в этот список не попадал.

Итог: единственные машинные свидетельства о подборе уничтожались следующим же
запуском, и вопрос "почему в хуке катана" становился непроверяемым. Это ровно та
ловушка, от которой предостерегает CLAUDE.md ("пустой автоматический отчёт — не
то же самое, что проблема решена"), встроенная в саму механику записи.
"""
import json
import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]

import pipeline_smart as ps  # noqa: E402


def _write(path, misses):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"misses": misses}, f)


def test_untouched_slots_survive_a_partial_run(tmp_path):
    """Главный регрессионный тест: слот, не пересобранный в этом прогоне,
    обязан сохранить прошлый вердикт."""
    p = str(tmp_path / "media_plan" / "relevance_gate_report.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    _write(p, [{"index": 9, "query": "medieval knight", "relevance": 0.11},
               {"index": 133, "query": "exhausted soldier", "relevance": 0.09}])

    # Прогон пересобрал ТОЛЬКО слот 133 и промахов на нём не нашёл.
    out = ps.merge_slot_report(p, [], resolved_slots={133})

    idxs = [m["index"] for m in out["misses"]]
    assert idxs == [9], (
        "промах по слоту 9 исчез из отчёта, хотя слот в этом прогоне даже "
        "не пересобирался — это и есть потеря аудита эпизода"
    )
    assert out["misses"][0]["from_previous_run"] is True
    assert out["slots_evaluated_this_run"] == 1


def test_resolved_slot_verdict_is_replaced_not_merged(tmp_path):
    """Пересобранный слот — это ДРУГОЙ кадр: старый вердикт к нему не относится."""
    p = str(tmp_path / "r.json")
    _write(p, [{"index": 76, "query": "old", "relevance": 0.05}])
    out = ps.merge_slot_report(
        p, [{"index": 76, "query": "new", "relevance": 0.42}], resolved_slots={76})
    assert len(out["misses"]) == 1
    assert out["misses"][0]["query"] == "new"
    assert "from_previous_run" not in out["misses"][0]


def test_resolved_slot_with_no_fresh_miss_is_cleared(tmp_path):
    """Слот пересобран и промаха больше нет -> запись обязана исчезнуть."""
    p = str(tmp_path / "r.json")
    _write(p, [{"index": 76, "query": "old", "relevance": 0.05}])
    out = ps.merge_slot_report(p, [], resolved_slots={76})
    assert out["misses"] == []


def test_missing_or_corrupt_previous_report_is_not_fatal(tmp_path):
    """Нет файла/битый JSON — пишем свежий, как раньше, без исключения."""
    p = str(tmp_path / "nope.json")
    out = ps.merge_slot_report(p, [{"index": 1}], resolved_slots={1})
    assert [m["index"] for m in out["misses"]] == [1]

    with open(p, "w", encoding="utf-8") as f:
        f.write("{сломано")
    out = ps.merge_slot_report(p, [{"index": 2}], resolved_slots={2})
    assert [m["index"] for m in out["misses"]] == [2]


def test_extra_fields_are_preserved(tmp_path):
    """Служебные поля отчёта (checked/note) не должны теряться при слиянии."""
    p = str(tmp_path / "r.json")
    out = ps.merge_slot_report(p, [], resolved_slots=set(),
                               extra={"checked": False, "note": "CLIP не грузился"})
    assert out["checked"] is False
    assert out["note"] == "CLIP не грузился"


def test_report_is_written_atomically_and_sorted(tmp_path):
    p = str(tmp_path / "r.json")
    ps.merge_slot_report(p, [{"index": 30}, {"index": 4}, {"index": 12}],
                         resolved_slots={4, 12, 30})
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    assert [m["index"] for m in data["misses"]] == [4, 12, 30]
    assert not os.path.exists(p + ".tmp"), "временный файл не убран"
