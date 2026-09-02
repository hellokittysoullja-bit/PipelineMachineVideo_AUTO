"""Кэш эмбеддингов и заглушка pixel_values размера 1 (visual_director.py).

Ни SigLIP2, ни Jina здесь не грузятся: обе тяжёлые (so400m — сотни
мегабайт, ~322с холодного старта по замеру 02.09), а проверять нужно не
их арифметику, а ЛОГИКУ кэша и самопроверки — она полностью отделима и
тестируется на фейковой сессии, тем же паттерном, что
test_shot_director.py применяет к Gemini.
"""
import os
import sys
import time

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

import visual_director as vd  # noqa: E402


class _FakeSession:
    """Имитирует ONNX-сессию Jina: возвращает эмбеддинги по числу текстов и
    записывает, с каким батчем pixel_values её звали."""

    def __init__(self, agree_on_batch1=True):
        self.agree_on_batch1 = agree_on_batch1
        self.pixel_batches = []

    def run(self, outputs, feeds):
        ids = feeds["input_ids"]
        pix_n = feeds["pixel_values"].shape[0]
        self.pixel_batches.append(pix_n)
        n = ids.shape[0]
        base = np.tile(np.arange(4, dtype=np.float32), (n, 1))
        if pix_n == 1 and not self.agree_on_batch1:
            base = base + 1.0        # «тихо другой результат» — худший случай
        return [base]


@pytest.fixture(autouse=True)
def _reset_state():
    vd._JINA_DUMMY_BATCH1_OK = None
    for c in (vd._siglip2_text_emb_cache, vd._siglip2_img_emb_cache,
              vd._jina_text_emb_cache, vd._jina_img_emb_cache):
        c.clear()
    yield
    vd._JINA_DUMMY_BATCH1_OK = None


def test_batch1_dummy_used_after_selfcheck_passes():
    """Совпало на самопроверке -> дальше только быстрый путь (батч 1)."""
    sess = _FakeSession(agree_on_batch1=True)
    ids = np.zeros((5, 7), dtype=np.int64)
    vd._jina_text_emb_batch(sess, ids)
    assert vd._JINA_DUMMY_BATCH1_OK is True
    # первый вызов — сверка: один быстрый + один полный прогон
    assert sorted(sess.pixel_batches) == [1, 5]
    sess.pixel_batches.clear()
    vd._jina_text_emb_batch(sess, ids)
    vd._jina_text_emb_batch(sess, ids)
    assert sess.pixel_batches == [1, 1], "после сверки полный батч больше не нужен"


def test_falls_back_to_full_batch_when_results_differ():
    """Быстрый путь молча вернул ДРУГОЕ — навсегда откатываемся на полный
    батч. Это и есть защита от «тихо испорченной раздачи запросов»."""
    sess = _FakeSession(agree_on_batch1=False)
    ids = np.zeros((5, 7), dtype=np.int64)
    out = vd._jina_text_emb_batch(sess, ids)
    assert vd._JINA_DUMMY_BATCH1_OK is False
    assert out.shape == (5, 4)
    np.testing.assert_allclose(out, np.tile(np.arange(4, dtype=np.float32), (5, 1)))
    sess.pixel_batches.clear()
    vd._jina_text_emb_batch(sess, ids)
    assert sess.pixel_batches == [5], "после провала сверки — только полный батч"


def test_single_text_never_takes_fast_path():
    """При n=1 заглушка и так размера 1 — сверку гонять незачем."""
    sess = _FakeSession()
    vd._jina_text_emb_batch(sess, np.zeros((1, 7), dtype=np.int64))
    assert sess.pixel_batches == [1]
    assert vd._JINA_DUMMY_BATCH1_OK is None, "сверка не должна была запускаться"


def test_selfcheck_failure_is_fail_safe():
    """Исключение внутри сверки -> полный батч, а не падение."""
    class _Boom(_FakeSession):
        def run(self, outputs, feeds):
            if feeds["pixel_values"].shape[0] == 1:
                raise RuntimeError("INVALID_ARGUMENT")
            return super().run(outputs, feeds)

    sess = _Boom()
    out = vd._jina_text_emb_batch(sess, np.zeros((3, 7), dtype=np.int64))
    assert vd._JINA_DUMMY_BATCH1_OK is False
    assert out.shape == (3, 4)


def test_emb_cache_respects_upper_bound():
    cache = {}
    for i in range(vd.EMB_CACHE_MAX + 5):
        vd._emb_cache_put(cache, f"k{i}", i)
    assert len(cache) <= vd.EMB_CACHE_MAX
    assert cache[f"k{vd.EMB_CACHE_MAX + 4}"] == vd.EMB_CACHE_MAX + 4


def test_image_cache_key_changes_when_file_changes(tmp_path):
    """Перезаписанный на месте файл обязан дать ДРУГОЙ ключ — иначе кэш
    отдал бы эмбеддинг старой картинки под новым содержимым."""
    p = tmp_path / "frame.jpg"
    p.write_bytes(b"a" * 100)
    k1 = vd._image_cache_key(str(p))
    time.sleep(0.01)
    p.write_bytes(b"b" * 250)
    k2 = vd._image_cache_key(str(p))
    assert k1 != k2


def test_image_cache_key_survives_missing_file():
    """Нет файла -> ключ по пути, без исключения (кэш живёт один прогон)."""
    k = vd._image_cache_key("/нет/такого/файла.jpg")
    assert k == ("path", "/нет/такого/файла.jpg")


def test_cache_signature_unchanged_by_embedding_cache():
    """cache_signature() хэширует ТОЛЬКО творческие константы отбора —
    добавление кэша не должно её двигать, иначе правка ради скорости молча
    инвалидировала бы кэш кандидатов на диске."""
    assert vd.cache_signature() == vd.cache_signature()
    assert "EMB_CACHE_MAX" not in str(vd.cache_signature())
