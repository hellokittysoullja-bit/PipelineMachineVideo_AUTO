"""Линт голодающего пула: сколько слотов приходится на один авторский запрос.

РЕАЛЬНЫЙ провал, из-за которого линт написан. Годность кадров ХУКА
опубликованного эпизода по золотому набору — 0 из 10. Разбор показал, что
дело не в гейтах: правки закрыли и метафору-запрос («milk bottle hand» →
современная кухня), и современное вторжение (контрастивное вето). Пустым
был ПУЛ: на 30 слотов хука приходилось 3 авторских запроса, то есть с
каждого требовалось 10 РАЗНЫХ годных кадров подряд. Дедуп по id/aHash
заставляет брать 7-й и 8-й результат выдачи, где по узкому средневековому
запросу уже нет ни Европы, ни рыцарей — и система честно показывает
«лучшее из плохого».

Увидеть это было негде: обе существующие оси линта (блоклист канала и
якорь эпохи) смотрят на ТЕКСТ запроса и ничего не знают о том, сколько
слотов он должен закрыть.
"""
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]

import pipeline_smart as ps  # noqa: E402


def _blocks(section, n):
    return [{"section": section, "text": "текст", "words": 5, "pause_after": 0.0}
            for _ in range(n)]


def test_starved_section_is_reported(capsys):
    """30 слотов на 3 запроса — ровно конфигурация, давшая 0 годных из 10."""
    ps.lint_authored_queries({"HOOK": ["a medieval sword", "b medieval helmet",
                                       "c medieval armour"]}, _blocks("HOOK", 30))
    out = capsys.readouterr().out
    assert "голодающим пулом" in out
    assert "30 слотов на 3" in out


def test_healthy_section_is_silent(capsys):
    ps.lint_authored_queries({"HOOK": [f"q{i} medieval armour" for i in range(8)]},
                             _blocks("HOOK", 30))
    out = capsys.readouterr().out
    assert "голодающим пулом" not in out


def test_section_name_underscore_form_is_matched(capsys):
    """В script.txt секции пишутся BLOCK_1, а у блоков они BLOCK 1 — без
    сопоставления линт молча не находил бы ни одной секции."""
    ps.lint_authored_queries({"BLOCK_1": ["x medieval sword"]}, _blocks("BLOCK 1", 20))
    assert "BLOCK_1: 20 слотов" in capsys.readouterr().out


def test_without_blocks_the_new_axis_is_skipped(capsys):
    """Старый вызов из двух аргументов остаётся валидным — ноль регресса."""
    ps.lint_authored_queries({"HOOK": ["a medieval sword"]})
    assert "голодающим пулом" not in capsys.readouterr().out


def test_no_slots_for_a_section_is_not_a_false_alarm(capsys):
    """Секция, у которой в сценарии нет блоков, не должна давать деление
    на пустоту и ложное предупреждение."""
    ps.lint_authored_queries({"BLOCK_9": ["x medieval sword"]}, _blocks("HOOK", 30))
    assert "BLOCK_9" not in capsys.readouterr().out


def test_empty_query_strings_do_not_inflate_the_count(capsys):
    """Пустая строка от лишней запятой не должна считаться запросом и
    прятать голод пула."""
    ps.lint_authored_queries({"HOOK": ["a medieval sword", "", ""]}, _blocks("HOOK", 30))
    assert "30 слотов на 1" in capsys.readouterr().out


def test_threshold_is_below_the_measured_failure():
    """Порог обязан срабатывать на измеренном провале (10.0) и не быть
    настолько низким, чтобы кричать на здоровой секции."""
    assert 3.0 < ps.QUERY_SLOTS_PER_QUERY_WARN < 10.0


def test_lint_is_called_with_blocks_from_the_pipeline():
    import inspect
    assert "lint_authored_queries(authored_queries, blocks)" in inspect.getsource(ps.main)


def test_real_episode_hook_is_no_longer_starved(capsys):
    """Якорь на реальных данных: после расширения запросов хук эпизода 02
    уходит из списка голодающих секций."""
    import re
    import script_parser
    script = os.path.join(REPO_ROOT, "videos", "02_ne-mechom", "script.txt")
    if not os.path.exists(script):
        pytest.skip("сценарий эпизода 02 недоступен")
    blocks, _ = ps.split_long_blocks(script_parser.parse_blocks(script), None)
    q = {}
    for line in open(script, encoding="utf-8"):
        m = re.match(r"^(HOOK|BLOCK_\d+|FINAL):\s*(.+)$", line.strip())
        if m:
            q[m.group(1)] = [x.strip() for x in m.group(2).split(",")]
    capsys.readouterr()
    ps.lint_authored_queries(q, blocks)
    out = capsys.readouterr().out
    assert "HOOK:" not in out, out
    assert len(q["HOOK"]) >= 8
