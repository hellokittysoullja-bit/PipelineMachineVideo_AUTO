"""Честная видимость молчаливой обрезки текста по лимиту токенов модели.

Реальный, измеренный вживую случай (08.09): SigLIP2 текстовая башня
(SIGLIP2_MAX_TEXT_LENGTH=64) — жёсткий предел max_position_embeddings,
Jina (JINA_TEXT_MAX_LENGTH=77) — свой. semantic_context_text()
(pipeline_smart.py) передаёт сюда полную фразу блока (иногда с соседней),
processor(padding="max_length", max_length=N) молча ОБРЕЗАЕТ всё, что не
влезло, без единой строчки в логе. Прямой замер на 96 реальных смысловых
юнитах двух опубликованных сценариев этого канала (01_ves-mecha,
_test20s): 24 из 96 (25%) обрезаются — модель оценивала соответствие
картинки фразе, не дочитав её до конца.

Этот модуль НЕ чинит саму обрезку (это отдельная, большая задача — см.
CLAUDE.md) — только делает её видимой в отчёте эпизода, ничего не меняя
в подборе картинки.
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
    vd._siglip2_text_emb_cache.clear()
    vd._jina_text_emb_cache.clear()
    yield
    vd.reset_text_truncation_report()
    vd._siglip2_text_emb_cache.clear()
    vd._jina_text_emb_cache.clear()


class _FakeTokenizer:
    """text=[...] -> {"input_ids": [[0]*n_tokens]} — контракт, которым
    _report_truncation_if_any реально зовёт токенизатор (см. её докстринг:
    ключевым словом text=, список из одного элемента)."""
    def __init__(self, n_tokens):
        self.n_tokens = n_tokens

    def __call__(self, text=None, **kw):
        assert isinstance(text, list) and len(text) == 1, (
            "ожидался вызов text=[строка] одним элементом")
        return {"input_ids": [[0] * self.n_tokens]}


class TestReportTruncationIfAny:
    def test_records_when_over_limit(self):
        vd._report_truncation_if_any("siglip2", "длинная фраза", _FakeTokenizer(90), 64)
        assert len(vd.TEXT_TRUNCATION_REPORT) == 1
        rec = vd.TEXT_TRUNCATION_REPORT[0]
        assert rec == {"model": "siglip2", "text": "длинная фраза", "tokens": 90, "limit": 64}

    def test_does_not_record_when_within_limit(self):
        vd._report_truncation_if_any("siglip2", "короткая фраза", _FakeTokenizer(10), 64)
        assert vd.TEXT_TRUNCATION_REPORT == []

    def test_does_not_record_exactly_at_limit(self):
        vd._report_truncation_if_any("jina", "фраза", _FakeTokenizer(77), 77)
        assert vd.TEXT_TRUNCATION_REPORT == []

    def test_same_text_reported_once_per_process(self):
        """Дальше эмбеддинг берётся из кэша — повторный замер того же
        (model, text) не даёт новой информации и не должен дублировать
        запись в отчёте."""
        tok = _FakeTokenizer(90)
        vd._report_truncation_if_any("siglip2", "фраза", tok, 64)
        vd._report_truncation_if_any("siglip2", "фраза", tok, 64)
        vd._report_truncation_if_any("siglip2", "фраза", tok, 64)
        assert len(vd.TEXT_TRUNCATION_REPORT) == 1

    def test_same_text_different_model_reported_separately(self):
        vd._report_truncation_if_any("siglip2", "фраза", _FakeTokenizer(90), 64)
        vd._report_truncation_if_any("jina", "фраза", _FakeTokenizer(90), 77)
        assert len(vd.TEXT_TRUNCATION_REPORT) == 2

    def test_fail_open_on_tokenizer_error(self):
        """Диагностика не имеет права ронять подбор картинки."""
        def boom(text=None, **kw):
            raise RuntimeError("токенизатор недоступен")
        vd._report_truncation_if_any("siglip2", "фраза", boom, 64)
        assert vd.TEXT_TRUNCATION_REPORT == []

    def test_reset_clears_both_report_and_dedup_state(self):
        vd._report_truncation_if_any("siglip2", "фраза", _FakeTokenizer(90), 64)
        vd.reset_text_truncation_report()
        assert vd.TEXT_TRUNCATION_REPORT == []
        # После сброса тот же текст снова считается "не виденным".
        vd._report_truncation_if_any("siglip2", "фраза", _FakeTokenizer(90), 64)
        assert len(vd.TEXT_TRUNCATION_REPORT) == 1


class TestWiredIntoRealEmbeddingFunctions:
    """_siglip2_text_emb()/_jina_text_emb()/_jina_text_emb_batch() реально
    зовут _report_truncation_if_any — не просто существующая, но не
    подключённая никуда функция."""

    def test_siglip2_text_emb_reports_truncation(self, monkeypatch, tmp_path):
        import torch

        class _FakeProcessor:
            def __call__(self, images=None, text=None, padding=None,
                         max_length=None, return_tensors=None):
                if images is not None:
                    return {"pixel_values": torch.zeros(1, 3, 2, 2)}
                # Настоящий процессор тоже принимает text=[строка]. Без
                # max_length (замер РЕАЛЬНОЙ длины в _report_truncation_
                # if_any) отдаёт естественную длину — 90 токенов, длиннее
                # лимита; С max_length (сам подсчёт эмбеддинга) — усечённую.
                assert text == ["очень длинная фраза сценария"]
                n = max_length if max_length is not None else 90
                return {"input_ids": torch.zeros(1, n, dtype=torch.long)}

        class _FakeModel:
            def get_text_features(self, **kwargs):
                return torch.tensor([[1.0, 0.0]])

        monkeypatch.setattr(vd, "_get_siglip2_model", lambda: (_FakeModel(), _FakeProcessor()))
        monkeypatch.setattr(vd, "SIGLIP2_MAX_TEXT_LENGTH", 5)
        # Дисковый кэш эмбеддингов живёт в TEMP_FOLDER/emb_cache — общем,
        # НЕ привязанном к tmp_path этого теста пути (temp_smart чистится
        # вместе с эпизодом, не с тестом). Реально пойманный вживую эффект:
        # второй прогон этого же теста подряд получал кэш-хит от первого и
        # выходил из _siglip2_text_emb ДО вызова _report_truncation_if_any,
        # не проверяя вообще ничего. Отключаем диск здесь намеренно —
        # тест обязан быть герметичным независимо от прошлых прогонов.
        monkeypatch.setattr(vd, "_emb_disk_load", lambda kind, key: None)
        monkeypatch.setattr(vd, "_emb_disk_store", lambda kind, key, arr: None)
        # Процессор без .tokenizer -> _siglip2_text_emb должна упасть на сам
        # процессор (см. её код: processor.tokenizer if hasattr(...) else processor).
        vd._siglip2_text_emb("очень длинная фраза сценария")
        assert len(vd.TEXT_TRUNCATION_REPORT) == 1
        assert vd.TEXT_TRUNCATION_REPORT[0]["model"] == "siglip2"

    def test_jina_text_emb_reports_truncation(self, monkeypatch, tmp_path):
        import numpy as np

        class _FakeSession:
            def run(self, output_names, feed):
                return [np.array([[1.0, 0.0]], dtype=np.float32)]

        class _FakeTok:
            def __call__(self, texts=None, text=None, padding=None,
                         truncation=None, max_length=None, return_tensors=None):
                if text is not None:
                    return {"input_ids": [[0] * 90]}
                return {"input_ids": np.zeros((1, max_length), dtype=np.int64)}

        monkeypatch.setattr(vd, "_get_jina_session", lambda: (_FakeSession(), _FakeTok()))
        monkeypatch.setattr(vd, "_JINA_BROKEN", False)
        # См. комментарий в test_siglip2_text_emb_reports_truncation выше —
        # тот же класс проблемы (дисковый кэш эмбеддингов не привязан к
        # tmp_path этого теста).
        monkeypatch.setattr(vd, "_emb_disk_load", lambda kind, key: None)
        monkeypatch.setattr(vd, "_emb_disk_store", lambda kind, key, arr: None)
        vd._jina_text_emb("другая очень длинная фраза")
        assert len(vd.TEXT_TRUNCATION_REPORT) == 1
        assert vd.TEXT_TRUNCATION_REPORT[0]["model"] == "jina"


class TestWiredIntoEpisodeReport:
    """main() (pipeline_smart.py) реально пишет media_plan/
    text_truncation_report.json и печатает предупреждение — не просто
    существующий, но никуда не подключённый отчёт."""

    def test_main_writes_and_prints_the_report(self):
        import pipeline_smart as ps
        src = open(os.path.join(SCRIPTS_DIR, "pipeline_smart.py"), encoding="utf-8").read()
        assert "text_truncation_report.json" in src
        assert "visual_director.TEXT_TRUNCATION_REPORT" in src
        assert "молча обрезаны по лимиту токенов" in src

    def test_main_resets_the_report_when_visual_director_is_imported(self):
        import pipeline_smart as ps
        src = open(os.path.join(SCRIPTS_DIR, "pipeline_smart.py"), encoding="utf-8").read()
        start = src.index('import visual_director as visual_director')
        block = src[start:start + 400]
        assert "reset_text_truncation_report()" in block


class TestSelfReportingConstantsMatchRealModelLimits:
    """Канарейка: если кто-то поменяет лимит модели, не обновив здесь
    ничего дополнительно делать не нужно — _report_truncation_if_any
    получает лимит параметром, а не хардкодит его. Этот тест лишь
    фиксирует, что сами константы — те самые, что реально используются
    для padding/truncation в _siglip2_text_emb/_jina_text_emb (см. их код)."""

    def test_constants_are_the_ones_actually_used_for_encoding(self):
        import inspect
        src_s2 = inspect.getsource(vd._siglip2_text_emb)
        assert "SIGLIP2_MAX_TEXT_LENGTH" in src_s2
        src_jina = inspect.getsource(vd._jina_text_emb)
        assert "JINA_TEXT_MAX_LENGTH" in src_jina
