"""Отчёт о ФАКТИЧЕСКИ исполнявшихся смысловых слоях.

Зачем этот файл существует. 18.09 автор карты стека записал в документ, что
смысловая раздача авторских запросов (Jina text-text) и визуальная полка в
трёх тестовых эпизодах работали. Обе не работали: onnxruntime в окружении не
установлен, индекса полки на диске нет. Оба слоя честно откатились fail-open
и НИЧЕГО об этом не сообщили, а media_plan/feature_flags.json показывал
включённые флаги — то есть артефакты прогона подтверждали неверный вывод.

Проверяется ровно это: declared (флаг) и active (что произошло) — разные
поля, и расхождение между ними видно.
"""
import json
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]
import pipeline_smart as ps   # noqa: E402


@pytest.fixture(autouse=True)
def _clean_stats():
    ps.reset_capability_stats()
    yield
    ps.reset_capability_stats()


def test_declared_but_not_working_is_not_reported_as_working():
    """Флаг включён, модель недоступна -> declared=True, active=False."""
    ps._capability_slot("semantic_query_assignment")["sections_fell_back"] = 3
    rows = ps.capability_report()
    row = rows["semantic_query_assignment"]
    assert row["declared"] is True
    assert row["active"] is False
    assert "ПОЗИЦИОННО" in row["reason"]
    assert row["detail"]["sections_fell_back"] == 3


def test_working_layer_is_reported_active():
    ps._capability_slot("semantic_query_assignment")["sections_assigned"] = 5
    row = ps.capability_report()["semantic_query_assignment"]
    assert row["active"] is True
    assert row["reason"] is None


def test_off_by_flag_is_not_the_same_as_broken(monkeypatch):
    """Четыре состояния, не два: выключенный слой не должен попадать в
    список «заявлен, но не работал» — иначе предупреждение о настоящей
    деградации утонет в шуме осознанных выключений."""
    monkeypatch.setattr(ps.feature_flags, "mode", lambda name: "off"
                        if name == "VISUAL_DIRECTOR_MODE" else "on")
    row = ps.capability_report()["visual_director_ensemble"]
    assert row["declared"] is False
    assert row["active"] is False


def test_shelf_flag_on_without_index_is_degraded(monkeypatch):
    """Ровно случай трёх тестовых эпизодов 18.09: SHELF_INDEX=1, индекса нет.

    Флаг включается принудительно: tests/conftest.py гасит SHELF_INDEX на
    всю суиту (иначе любой тест, дошедший до сборки пула, тянул бы модель на
    4.3 ГБ), поэтому проверяется логика отчёта, а не дефолт окружения."""
    monkeypatch.setattr(ps.feature_flags, "enabled",
                        lambda name: True if name == "SHELF_INDEX" else False)
    ps.SOURCE_STATS.pop("shelf", None)
    row = ps.capability_report()["shelf_index"]
    assert row["declared"] is True      # флаг включён
    assert row["active"] is False       # но кандидатов ноль
    assert row["detail"]["offered"] == 0


def test_museum_sources_counts_all_three_museums(monkeypatch):
    """Найдено живым прогоном 19.09, не чтением: candidate_source() отдаёт
    ТРИ отдельных префикса ('met'/'cleveland'/'chicago'), общего ключа
    'museum' в SOURCE_STATS не существует и не существовало никогда.
    Отчёт лгал «музеи не дали кандидатов» даже когда Мет и Кливленд реально
    выиграли слоты (медиевал-тест: met offered 102, cleveland offered 6,
    capability_report при этом писал offered=0)."""
    monkeypatch.setattr(ps.feature_flags, "enabled",
                        lambda name: True if name == "MUSEUM_SOURCES_ENABLED" else False)
    monkeypatch.setitem(ps.SOURCE_STATS, "met", {"offered": 102})
    monkeypatch.setitem(ps.SOURCE_STATS, "cleveland", {"offered": 6})
    monkeypatch.setitem(ps.SOURCE_STATS, "chicago", {"offered": 225})
    row = ps.capability_report()["museum_sources"]
    assert row["declared"] is True
    assert row["active"] is True
    assert row["detail"]["offered"] == 102 + 6 + 225


def test_museum_sources_degraded_when_all_three_empty(monkeypatch):
    monkeypatch.setattr(ps.feature_flags, "enabled",
                        lambda name: True if name == "MUSEUM_SOURCES_ENABLED" else False)
    for name in ("met", "cleveland", "chicago"):
        monkeypatch.delitem(ps.SOURCE_STATS, name, raising=False)
    row = ps.capability_report()["museum_sources"]
    assert row["declared"] is True
    assert row["active"] is False
    assert row["detail"]["offered"] == 0


def test_report_file_written_and_lists_degraded(tmp_path):
    ps._capability_slot("semantic_query_assignment")["sections_fell_back"] = 1
    path = ps.write_capability_report(str(tmp_path))
    data = json.load(open(path, encoding="utf-8"))
    caps = data["capabilities"]
    assert caps["semantic_query_assignment"]["active"] is False
    degraded = [n for n, v in caps.items() if v["declared"] and not v["active"]]
    assert "semantic_query_assignment" in degraded
