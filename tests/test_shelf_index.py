# -*- coding: utf-8 -*-
"""Визуальная полка: инварианты, которые нельзя ломать молча.

Тесты держат не «функция что-то вернула», а ровно те свойства, из-за
нарушения которых этот репозиторий уже горел: additive-подключение,
совпадение пространств векторов, честный откат при половинном индексе и
единый ID предмета между путями (иначе дедуп пропустит дубль).
"""
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import shelf_index  # noqa: E402
import shot_types  # noqa: E402


def _write_index(tmp_path, n=4, dim=8, model=None, drop_vectors=0):
    """Маленький синтетический индекс на диске в РЕАЛЬНОМ формате модуля."""
    d = tmp_path / "shelf"
    d.mkdir(exist_ok=True)
    items, vecs = [], []
    for i in range(n):
        v = np.zeros(dim, dtype="float32")
        v[i % dim] = 1.0
        vecs.append(v)
        items.append({"id": f"met:{1000+i}", "dim": dim,
                      "model": model or shelf_index.SHELF_MODEL,
                      "version": shelf_index.SHELF_INDEX_VERSION,
                      "name": f"Object {i}", "title": f"Title {i}",
                      "culture": "Italian", "b": 1400, "e": 1450,
                      "dept": "Arms and Armor",
                      "image": f"https://example.invalid/{i}.jpg",
                      "thumb": f"https://example.invalid/{i}_s.jpg",
                      "page": f"https://example.invalid/page/{i}"})
    with open(d / "items.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    keep = n - drop_vectors
    with open(d / "vectors.f32", "wb") as f:
        for v in vecs[:keep]:
            f.write(v.tobytes())
    return d


@pytest.fixture
def indexed(tmp_path, monkeypatch):
    def _make(**kw):
        d = _write_index(tmp_path, **kw)
        monkeypatch.setattr(shelf_index, "INDEX_DIR", str(d))
        monkeypatch.setattr(shelf_index, "ITEMS_PATH", str(d / "items.jsonl"))
        monkeypatch.setattr(shelf_index, "VECTORS_PATH", str(d / "vectors.f32"))
        shelf_index._CACHE.update({"loaded": False, "vectors": None,
                                   "items": None, "dim": None})
        return d
    return _make


def test_no_index_on_disk_is_a_clean_no_op(tmp_path, monkeypatch):
    """Полки нет — путь отбора обязан остаться прежним, а не упасть."""
    monkeypatch.setattr(shelf_index, "ITEMS_PATH", str(tmp_path / "nope.jsonl"))
    monkeypatch.setattr(shelf_index, "VECTORS_PATH", str(tmp_path / "nope.f32"))
    shelf_index._CACHE.update({"loaded": False, "vectors": None, "items": None})
    assert shelf_index.available() is False
    assert shelf_index.search("a two-handed sword") == []


def test_half_written_index_truncates_to_the_common_prefix(indexed):
    """Обрыв процесса между строкой метаданных и вектором — штатный случай
    append-only файла. Молча сдвинуть соответствие «строка -> вектор»
    нельзя: это дало бы кандидатов с чужими метаданными."""
    indexed(n=5, drop_vectors=2)
    vecs, items = shelf_index.load()
    assert vecs is not None
    assert len(items) == vecs.shape[0] == 3


def test_index_from_another_model_is_refused_not_used(indexed, capsys):
    """Косинус между разными пространствами посчитается без ошибки и будет
    шумом. Отказ должен быть громким, а не тихим продолжением."""
    indexed(n=3, model="some-other-model")
    vecs, items = shelf_index.load()
    assert vecs is None and items is None
    assert "другой моделью" in capsys.readouterr().out


def test_search_ranks_by_vector_similarity(indexed, monkeypatch):
    indexed(n=4, dim=8)
    q = np.zeros(8, dtype="float32")
    q[2] = 1.0
    monkeypatch.setattr(shelf_index, "_brief_vector", lambda _t: q)
    res = shelf_index.search("что угодно", limit=3)
    assert [r["id"] for r in res][0] == "met:1002"
    assert res[0]["score"] == pytest.approx(1.0, abs=1e-6)


def test_empty_brief_never_queries_the_shelf(indexed, monkeypatch):
    indexed(n=3)
    called = []
    monkeypatch.setattr(shelf_index, "_brief_vector",
                        lambda t: called.append(t) or np.zeros(8, dtype="float32"))
    assert shelf_index.search("", limit=3) == []
    assert shelf_index.search("   ", limit=3) == []
    assert called == []


def test_dimension_mismatch_fails_open(indexed, monkeypatch):
    """Вектор запроса другой длины — не падение рендера, а пустой список."""
    indexed(n=3, dim=8)
    monkeypatch.setattr(shelf_index, "_brief_vector",
                        lambda _t: np.zeros(16, dtype="float32"))
    assert shelf_index.search("бриф", limit=3) == []


def test_shelf_declares_itself_in_the_routing_table():
    """Источник объявляется ДАННЫМИ, а не развилкой в коде."""
    assert "shelf" in shot_types.SOURCE_CAPABILITIES
    caps = shot_types.SOURCE_CAPABILITIES["shelf"]["supports"]
    assert "object" in caps and "illustration" in caps
    # Сценические слоты пока НЕ заявлены: музейный корпус на них измеренно
    # слаб, а отдельного A/B по победителям сцен ещё не было. Заявить их без
    # замера — ровно та ошибка, которую исправил замер маршрутизации 14.09.
    assert "scene" not in caps


def test_shelf_candidate_id_matches_the_museum_path(indexed, monkeypatch):
    """ID предмета обязан быть ОДИН на оба пути: общий used_ids/дедуп должен
    видеть один и тот же предмет как один, каким бы путём он ни нашёлся."""
    indexed(n=2, dim=8)
    q = np.zeros(8, dtype="float32")
    q[0] = 1.0
    monkeypatch.setattr(shelf_index, "_brief_vector", lambda _t: q)
    res = shelf_index.search("бриф", limit=2)
    assert all(r["id"].startswith("met:") for r in res)


def test_stats_reports_what_is_actually_on_disk(indexed):
    indexed(n=6, dim=8)
    st = shelf_index.stats()
    assert st["available"] is True
    assert st["items"] == 6 and st["dim"] == 8
    assert st["model"] == shelf_index.SHELF_MODEL
