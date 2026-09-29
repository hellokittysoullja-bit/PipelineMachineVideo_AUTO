"""Честная видимость молчаливой обрезки текста по лимиту токенов модели.

История (08.09): у SigLIP2 текстовая башня держала 64 токена, у Jina — 77,
и processor(padding="max_length", max_length=N) молча обрезал каждую
четвёртую фразу блока (24 из 96 на двух опубликованных сценариях). Отчёт
media_plan/text_truncation_report.json сделал это видимым.

GPU-ветка (29.09): модель зрения — Qwen3-VL-Embedding, контекст 32k токенов,
фраза блока в него помещается целиком. Отчёт остаётся (пишется всегда, в том
числе пустым — «нечего сообщить» тоже факт), и эти тесты держат, что он
по-прежнему подключён к прогону.
"""
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

sys.argv = ["visual_director.py", tempfile.gettempdir()]
import visual_director as vd  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_report():
    vd.reset_text_truncation_report()
    yield
    vd.reset_text_truncation_report()


def test_reset_clears_the_report():
    vd.TEXT_TRUNCATION_REPORT.append({"model": "m", "text": "t", "tokens": 9, "limit": 8})
    vd.reset_text_truncation_report()
    assert vd.TEXT_TRUNCATION_REPORT == []


def test_qwen_context_holds_a_whole_block_phrase():
    """Причина, по которой отчёт в GPU-ветке пуст: лимит модели на порядки
    больше самой длинной фразы блока (93 токена SigLIP2 — худшая из 142 фраз
    эп.02)."""
    import qwen_vl_embed
    assert qwen_vl_embed.MAX_LENGTH >= 8192


class TestWiredIntoEpisodeReport:
    """main() (pipeline_smart.py) реально пишет media_plan/
    text_truncation_report.json и печатает предупреждение — не просто
    существующий, но никуда не подключённый отчёт."""

    def test_main_writes_and_prints_the_report(self):
        src = open(os.path.join(SCRIPTS_DIR, "pipeline_smart.py"), encoding="utf-8").read()
        assert "text_truncation_report.json" in src
        assert "visual_director.TEXT_TRUNCATION_REPORT" in src
        assert "молча обрезаны по лимиту токенов" in src

    def test_main_resets_the_report_when_visual_director_is_imported(self):
        src = open(os.path.join(SCRIPTS_DIR, "pipeline_smart.py"), encoding="utf-8").read()
        start = src.index('import visual_director as visual_director')
        block = src[start:start + 400]
        assert "reset_text_truncation_report()" in block
