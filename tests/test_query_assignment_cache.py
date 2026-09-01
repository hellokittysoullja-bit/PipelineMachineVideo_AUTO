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
import pipeline_smart as ps   # noqa: E402


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


def test_cache_hit_avoids_second_model_call(monkeypatch, tmp_path):
    monkeypatch.setattr(ps, "SEMANTIC_QUERY_ASSIGNMENT", True)
    calls = []
    _fake_sim(monkeypatch, [[0.1, 0.9], [0.8, 0.2]], call_counter=calls)
    _use_real_cache_dir(monkeypatch, tmp_path)

    out1 = ps._cached_semantic_query_assignment(["a", "b"], ["q0", "q1"])
    assert out1 == [1, 0]
    assert len(calls) == 1

    # Второй вызов с ТЕМ ЖЕ входом — модель больше не должна дёргаться,
    # даже если бы она теперь возвращала другую матрицу.
    _fake_sim(monkeypatch, [[0.9, 0.1], [0.1, 0.9]], call_counter=calls)
    out2 = ps._cached_semantic_query_assignment(["a", "b"], ["q0", "q1"])
    assert out2 == [1, 0]   # из кэша, не пересчитано по новой (другой) матрице
    assert len(calls) == 1  # второй вызов модели не произошёл


def test_cache_miss_on_different_input(monkeypatch, tmp_path):
    monkeypatch.setattr(ps, "SEMANTIC_QUERY_ASSIGNMENT", True)
    calls = []
    _fake_sim(monkeypatch, [[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]], call_counter=calls)
    _use_real_cache_dir(monkeypatch, tmp_path)

    ps._cached_semantic_query_assignment(["a", "b"], ["q0", "q1"])
    ps._cached_semantic_query_assignment(["a", "b", "c"], ["q0", "q1"])
    assert len(calls) == 2   # разный вход -> разный ключ -> оба раза считает


def test_cache_file_written_to_media_plan_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(ps, "SEMANTIC_QUERY_ASSIGNMENT", True)
    _fake_sim(monkeypatch, [[0.1, 0.9], [0.8, 0.2]])
    _use_real_cache_dir(monkeypatch, tmp_path)

    ps._cached_semantic_query_assignment(["a", "b"], ["q0", "q1"])
    cache_dir = os.path.join(str(tmp_path), "media_plan", "query_resolution_cache")
    files = os.listdir(cache_dir)
    assert len(files) == 1
    with open(os.path.join(cache_dir, files[0]), encoding="utf-8") as f:
        assert json.load(f) == [1, 0]


def test_fail_open_result_not_cached(monkeypatch, tmp_path):
    monkeypatch.setattr(ps, "SEMANTIC_QUERY_ASSIGNMENT", True)
    calls = []
    _fake_sim(monkeypatch, None, call_counter=calls)   # модель "недоступна"
    _use_real_cache_dir(monkeypatch, tmp_path)

    out = ps._cached_semantic_query_assignment(["a"], ["q0", "q1"])
    assert out is None
    cache_dir = os.path.join(str(tmp_path), "media_plan", "query_resolution_cache")
    assert not os.path.isdir(cache_dir) or not os.listdir(cache_dir)
    assert len(calls) == 1

    # Повторный вызов — модель обязана дёрнуться СНОВА (None не кэшируется).
    ps._cached_semantic_query_assignment(["a"], ["q0", "q1"])
    assert len(calls) == 2


def test_pytest_current_test_bypasses_cache_entirely(monkeypatch, tmp_path):
    # PYTEST_CURRENT_TEST выставлен (мы внутри теста) -> функция обязана
    # вести себя как голый semantic_query_assignment(), НЕ читая/не пиша
    # на диск, даже если VIDEO_FOLDER указывает на реальную tmp_path.
    monkeypatch.setattr(ps, "SEMANTIC_QUERY_ASSIGNMENT", True)
    monkeypatch.setattr(ps, "VIDEO_FOLDER", str(tmp_path))
    calls = []
    _fake_sim(monkeypatch, [[0.1, 0.9], [0.8, 0.2]], call_counter=calls)
    assert os.environ.get("PYTEST_CURRENT_TEST")   # sanity: точно внутри pytest

    ps._cached_semantic_query_assignment(["a", "b"], ["q0", "q1"])
    ps._cached_semantic_query_assignment(["a", "b"], ["q0", "q1"])
    assert len(calls) == 2   # кэш не сработал ни разу
    cache_dir = os.path.join(str(tmp_path), "media_plan", "query_resolution_cache")
    assert not os.path.isdir(cache_dir)
