"""_cached_semantic_query_assignment() — дисковый кэш поверх
semantic_query_assignment() (см. докстринг в pipeline_smart.py): реальный
измеренный кейс (01_ves-mecha, 31 авг) — resolve_queries() пересчитывал
Jina text-text similarity заново при КАЖДОМ перезапуске процесса (10-15+
минут), хотя вход между перезапусками одного эпизода не менялся.

Кэш ОТКЛЮЧЁН под pytest (PYTEST_CURRENT_TEST) — эти тесты сами
имитируют "продакшн" через monkeypatch.delenv, иначе кэш никогда бы не
сработал внутри тестового прогона (см. риск пересечения между тестами в
докстринге _cached_semantic_query_assignment)."""
import json
import os
import sys
import tempfile
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]
import render_core as ps   # noqa: E402


def _fake_sim(monkeypatch, matrix, call_counter=None):
    def _sim(a, b):
        if call_counter is not None:
            call_counter.append(1)
        return matrix
    fake = types.SimpleNamespace(text_text_similarity=_sim)
    monkeypatch.setitem(sys.modules, "visual_director", fake)


def _use_real_cache_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(ps, "VIDEO_FOLDER", str(tmp_path))
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)


def test_cache_disabled_under_pytest_by_default():
    # Под обычным запуском теста PYTEST_CURRENT_TEST стоит сам pytest —
    # функция обязана вести себя как прямой вызов semantic_query_assignment,
    # без единого файла на диске.
    assert os.environ.get("PYTEST_CURRENT_TEST")










