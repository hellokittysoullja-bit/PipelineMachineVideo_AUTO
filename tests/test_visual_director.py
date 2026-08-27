"""Юнит-тесты Semantic Visual Director v1 (scripts/visual_director.py).
Тот же принцип разделения, что test_look_reference.py уже применяет к
look_reference.py: CLIP/ffprobe-подобные внешние вызовы монkeypatch'ятся на
безопасные значения, тестируется чистая логика (бонусы/штрафы/оркестрация),
без реального torch/файлов на диске."""
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

sys.argv = ["visual_director.py", tempfile.gettempdir()]
import visual_director as vd   # noqa: E402


# ---------- functional_role (тонкая обёртка) ----------

def test_functional_role_delegates_to_classify_shot_function():
    assert vd.functional_role({"stat": True}, False) == "evidence"
    assert vd.functional_role({"is_subcut": True}, False) == "detail"
    assert vd.functional_role({"section": "BLOCK_1"}, True) == "context"
    assert vd.functional_role({"section": "HOOK"}, False) == "hook"
    assert vd.functional_role({"section": "BLOCK_1"}, False) == "narrative"


# ---------- sentence_relevance ----------

def test_sentence_relevance_none_on_empty_text():
    assert vd.sentence_relevance("x.jpg", "") is None
    assert vd.sentence_relevance("x.jpg", None) is None


def test_sentence_relevance_uses_siglip2_not_english_only_clip_relevance(monkeypatch, tmp_path):
    # Реальный, эмпирически подтверждённый баг первой версии: sentence_relevance()
    # передавала русский текст в pipeline_smart.clip_relevance() (одноязычная
    # английская модель) — на реальных фото/фразах канала это давало почти
    # нулевой сигнал (см. комментарий у SENTENCE_RELEVANCE_MODEL_VERSION).
    # Фикс — отдельная модель (SigLIP2), НЕ pipeline_smart.clip_relevance().
    called = {"clip_relevance": False}

    def fail_if_called(path, text):
        called["clip_relevance"] = True
        return 0.99   # заведомо отличается от того, что должен вернуть тест
    monkeypatch.setattr(vd.pipeline_smart, "clip_relevance", fail_if_called)

    import torch

    class _FakeProcessor:
        def __call__(self, images=None, text=None, return_tensors=None, padding=None,
                     max_length=None, truncation=None):
            if images is not None:
                return {"pixel_values": torch.zeros(1, 3, 2, 2)}
            return {"input_ids": torch.zeros(1, 3, dtype=torch.long)}

    class _FakeModel:
        def get_image_features(self, **kwargs):
            return torch.tensor([[1.0, 0.0, 0.0]])

        def get_text_features(self, **kwargs):
            return torch.tensor([[1.0, 0.0, 0.0]])   # параллельно картинке -> cos=1.0

    monkeypatch.setattr(vd, "_get_siglip2_model", lambda: (_FakeModel(), _FakeProcessor()))
    # Jina недоступна в этом тесте (не мокается загрузка ONNX) -> ensemble
    # честно падает на SigLIP2-only (см. test_sentence_relevance_falls_back_*
    # ниже для отдельной проверки самого fail-open пути).
    monkeypatch.setattr(vd, "_jina_relevance", lambda path, text: None)
    img_path = str(tmp_path / "x.jpg")
    from PIL import Image
    Image.new("RGB", (4, 4)).save(img_path)

    result = vd.sentence_relevance(img_path, "полный текст фразы блока")
    assert result == pytest.approx(1.0)
    assert called["clip_relevance"] is False, (
        "sentence_relevance не должна использовать одноязычный "
        "pipeline_smart.clip_relevance() — см. докстринг про баг")


def test_sentence_relevance_none_when_siglip2_model_import_fails(monkeypatch, tmp_path):
    def raise_import_error():
        raise ImportError("transformers недоступен")
    monkeypatch.setattr(vd, "_get_siglip2_model", raise_import_error)
    monkeypatch.setattr(vd, "_SIGLIP2_BROKEN", False)
    img_path = str(tmp_path / "x.jpg")
    from PIL import Image
    Image.new("RGB", (4, 4)).save(img_path)
    assert vd.sentence_relevance(img_path, "текст") is None
    assert vd._SIGLIP2_BROKEN is True


def test_siglip2_truncates_long_block_text(monkeypatch, tmp_path):
    """Регрессия на реальный, воспроизведённый вживую баг: текстовая башня
    so400m держит 64 позиции (max_position_embeddings), SIGLIP2_MAX_TEXT_LENGTH
    тоже 64, а настоящий блок сценария на русском (47 слов) токенизируется в
    80 токенов. Без truncation=True вызов модели падал, except глотал
    исключение, и sentence_relevance() отдавала None ПО ВСЕМ кандидатам сразу —
    то есть доминирующий сигнал Директора (SENTENCE_RELEVANCE_WEIGHT=1.0) на
    длинных блоках молча не участвовал в отборе вообще.

    Фейковый процессор здесь ведёт себя как настоящий: без truncation на
    длинном тексте бросает, с truncation — обрезает."""
    import torch

    seen = {}

    class _StrictProcessor:
        def __call__(self, images=None, text=None, return_tensors=None, padding=None,
                     max_length=None, truncation=None):
            if images is not None:
                return {"pixel_values": torch.zeros(1, 3, 2, 2)}
            seen["truncation"] = truncation
            n_tokens = len(text[0].split()) * 2      # грубая имитация русской токенизации
            if n_tokens > max_length and not truncation:
                raise ValueError("sequence length %d > max_length %d" % (n_tokens, max_length))
            return {"input_ids": torch.zeros(1, min(n_tokens, max_length), dtype=torch.long)}

    class _FakeModel:
        def get_image_features(self, **kwargs):
            return torch.tensor([[1.0, 0.0, 0.0]])

        def get_text_features(self, **kwargs):
            return torch.tensor([[1.0, 0.0, 0.0]])

    monkeypatch.setattr(vd, "_get_siglip2_model", lambda: (_FakeModel(), _StrictProcessor()))
    monkeypatch.setattr(vd, "_jina_relevance", lambda path, text: None)
    img_path = str(tmp_path / "x.jpg")
    from PIL import Image
    Image.new("RGB", (4, 4)).save(img_path)

    long_block = " ".join(["слово"] * 60)          # 120 "токенов" против лимита 64
    result = vd.sentence_relevance(img_path, long_block)

    assert seen["truncation"] is True, (
        "processor() обязан вызываться с truncation=True — иначе длинный блок "
        "роняет модель, except глотает ошибку и сигнал Директора молча исчезает")
    assert result == pytest.approx(1.0), (
        "на длинном блоке sentence_relevance обязана вернуть число, а не None")


# ---------- ensemble SigLIP2 + Jina CLIP v2 (sentence_relevance) ----------
# По прямому запросу пользователя "объединить сильнейшие стороны обеих
# моделей" — 113-позиционный бенчмарк дал измеримый прирост (top-1 81%->85%,
# top-3 93%->95% на SigLIP2=0.7/Jina=0.3, см. докстринг
# ENSEMBLE_WEIGHT_SIGLIP2 в scripts/visual_director.py). Критично: Jina
# ДОЛЖНА быть fail-open — при недоступности (нет onnxruntime, сеть, любая
# ошибка) sentence_relevance() обязана падать на SigLIP2-only, никогда не
# возвращать None только из-за отсутствующей Jina, пока SigLIP2 сам жив.

def test_sentence_relevance_blends_siglip2_and_jina_with_fixed_weights(monkeypatch, tmp_path):
    monkeypatch.setattr(vd, "_siglip2_relevance", lambda path, text: 0.30)
    monkeypatch.setattr(vd, "_jina_relevance", lambda path, text: vd.JINA_SCORE_MEAN + vd.JINA_SCORE_STD)
    img_path = str(tmp_path / "x.jpg")
    result = vd.sentence_relevance(img_path, "текст")
    # jina_raw на 1 стд выше среднего -> rescale даёт SIGLIP2_SCORE_MEAN + SIGLIP2_SCORE_STD
    expected_jina_rescaled = vd.SIGLIP2_SCORE_MEAN + vd.SIGLIP2_SCORE_STD
    expected = vd.ENSEMBLE_WEIGHT_SIGLIP2 * 0.30 + vd.ENSEMBLE_WEIGHT_JINA * expected_jina_rescaled
    assert result == pytest.approx(expected)


def test_sentence_relevance_falls_back_to_siglip2_only_when_jina_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(vd, "_siglip2_relevance", lambda path, text: 0.42)
    monkeypatch.setattr(vd, "_jina_relevance", lambda path, text: None)
    img_path = str(tmp_path / "x.jpg")
    result = vd.sentence_relevance(img_path, "текст")
    assert result == pytest.approx(0.42)   # чистый SigLIP2, без домножения на вес


def test_sentence_relevance_none_when_siglip2_unavailable_even_if_jina_ok(monkeypatch, tmp_path):
    # SigLIP2 — основа, Jina только добавляет; без SigLIP2 весь сигнал None,
    # ensemble НЕ должен подменять базовую модель второстепенной.
    monkeypatch.setattr(vd, "_siglip2_relevance", lambda path, text: None)
    monkeypatch.setattr(vd, "_jina_relevance", lambda path, text: 0.99)
    img_path = str(tmp_path / "x.jpg")
    assert vd.sentence_relevance(img_path, "текст") is None


def test_rescale_jina_to_siglip2_scale_identity_at_mean():
    # На среднем Jina -> среднее SigLIP2 (по построению z-score переноса).
    assert vd._rescale_jina_to_siglip2_scale(vd.JINA_SCORE_MEAN) == pytest.approx(vd.SIGLIP2_SCORE_MEAN)


def test_rescale_jina_to_siglip2_scale_preserves_direction():
    lo = vd._rescale_jina_to_siglip2_scale(vd.JINA_SCORE_MEAN - vd.JINA_SCORE_STD)
    hi = vd._rescale_jina_to_siglip2_scale(vd.JINA_SCORE_MEAN + vd.JINA_SCORE_STD)
    assert hi > lo


def test_jina_relevance_none_on_import_error(monkeypatch):
    def raise_import_error():
        raise ImportError("onnxruntime недоступен")
    monkeypatch.setattr(vd, "_get_jina_session", raise_import_error)
    monkeypatch.setattr(vd, "_JINA_BROKEN", False)
    assert vd._jina_relevance("x.jpg", "текст") is None
    assert vd._JINA_BROKEN is True


def test_jina_relevance_none_on_generic_exception(monkeypatch, tmp_path):
    def raise_runtime_error():
        raise RuntimeError("модель сломалась")
    monkeypatch.setattr(vd, "_get_jina_session", raise_runtime_error)
    monkeypatch.setattr(vd, "_JINA_BROKEN", False)
    assert vd._jina_relevance("x.jpg", "текст") is None
    # generic Exception (не ImportError) НЕ должен взводить permanent-флаг —
    # тот же принцип, что и у _siglip2_relevance (см. except-ветки).
    assert vd._JINA_BROKEN is False


def test_jina_relevance_returns_cosine_from_mocked_onnx_session(monkeypatch, tmp_path):
    import numpy as np

    class _FakeSession:
        def run(self, output_names, feed):
            if output_names == ["l2norm_image_embeddings"]:
                return [np.array([[1.0, 0.0]], dtype=np.float32)]
            return [np.array([[1.0, 0.0]], dtype=np.float32)]

    class _FakeTokenizer:
        def __call__(self, texts, padding=None, truncation=None, max_length=None, return_tensors=None):
            return {"input_ids": np.zeros((1, 3), dtype=np.int64)}

    monkeypatch.setattr(vd, "_get_jina_session", lambda: (_FakeSession(), _FakeTokenizer()))
    monkeypatch.setattr(vd, "_JINA_BROKEN", False)
    img_path = str(tmp_path / "x.jpg")
    from PIL import Image
    Image.new("RGB", (8, 8)).save(img_path)
    result = vd._jina_relevance(img_path, "текст")
    assert result == pytest.approx(1.0)   # параллельные векторы -> cos=1.0


def test_ensemble_model_version_reflects_weights():
    assert str(vd.ENSEMBLE_WEIGHT_SIGLIP2) in vd.SENTENCE_RELEVANCE_MODEL_VERSION
    assert str(vd.ENSEMBLE_WEIGHT_JINA) in vd.SENTENCE_RELEVANCE_MODEL_VERSION
    assert "jina" in vd.SENTENCE_RELEVANCE_MODEL_VERSION.lower()


# ---------- role_shot_size_bonus ----------

def test_role_shot_size_bonus_known_pair():
    assert vd.role_shot_size_bonus("detail", "detail") == pytest.approx(0.20)
    assert vd.role_shot_size_bonus("hook", "wide") == pytest.approx(0.10)


def test_role_shot_size_bonus_unknown_pair_is_zero():
    assert vd.role_shot_size_bonus("narrative", "detail") == 0.0
    assert vd.role_shot_size_bonus("unknown_role", "wide") == 0.0


# ---------- arc_stage_shot_bonus ----------

def test_arc_stage_shot_bonus_known_pair():
    assert vd.arc_stage_shot_bonus("слом", "detail") == pytest.approx(0.12)
    assert vd.arc_stage_shot_bonus("заход-якорь", "wide") == pytest.approx(0.10)


def test_arc_stage_shot_bonus_unknown_pair_is_zero():
    assert vd.arc_stage_shot_bonus("слом", "wide") == 0.0
    assert vd.arc_stage_shot_bonus("совершенно_незнакомая_стадия", "detail") == 0.0


def test_arc_stage_shot_bonus_none_is_zero():
    # Эпизод без Speech Director (нет speech_plan.json) -> arc_stage=None на
    # каждом клипе -> бонус 0.0 везде -> байт-в-байт прежнее поведение.
    assert vd.arc_stage_shot_bonus(None, "detail") == 0.0
    assert vd.arc_stage_shot_bonus(None, "wide") == 0.0


# ---------- domain_match_bonus ----------

def test_domain_match_bonus_when_equal():
    assert vd.domain_match_bonus("snow", "snow") == pytest.approx(vd.DOMAIN_MATCH_BONUS)


def test_domain_match_bonus_zero_when_different_or_missing():
    assert vd.domain_match_bonus("snow", "night") == 0.0
    assert vd.domain_match_bonus(None, "snow") == 0.0
    assert vd.domain_match_bonus("snow", None) == 0.0
    assert vd.domain_match_bonus(None, None) == 0.0


# ---------- repetition_penalty ----------

def test_repetition_penalty_zero_on_empty_history():
    assert vd.repetition_penalty([], "snow", "evidence") == 0.0


def test_repetition_penalty_counts_matches_within_window():
    history = [("snow", "evidence"), ("night", "hook"), ("snow", "evidence")]
    assert vd.repetition_penalty(history, "snow", "evidence") == pytest.approx(2 * vd.REPETITION_PENALTY)


def test_repetition_penalty_ignores_entries_outside_window():
    # REPETITION_WINDOW=3 -> только последние 3 элемента учитываются.
    history = [("snow", "evidence")] * 5 + [("night", "hook"), ("urban", "detail"), ("battle", "context")]
    assert vd.repetition_penalty(history, "snow", "evidence") == 0.0


def test_repetition_penalty_no_match_is_zero():
    history = [("night", "hook"), ("urban", "detail")]
    assert vd.repetition_penalty(history, "snow", "evidence") == 0.0


# ---------- visual_qc_bonus ----------

def test_visual_qc_bonus_combines_sharpness_and_noise(monkeypatch):
    import visual_qc

    monkeypatch.setattr(visual_qc, "_load_gray_normalized", lambda path: "FAKE_GRAY")
    monkeypatch.setattr(visual_qc, "sharpness_score", lambda gray: 200.0)
    monkeypatch.setattr(visual_qc, "noise_score", lambda gray: 0.0)
    bonus = vd.visual_qc_bonus("x.jpg")
    assert bonus == pytest.approx(vd.VISUAL_QC_SHARPNESS_WEIGHT)


def test_visual_qc_bonus_penalizes_noise(monkeypatch):
    import visual_qc

    monkeypatch.setattr(visual_qc, "_load_gray_normalized", lambda path: "FAKE_GRAY")
    monkeypatch.setattr(visual_qc, "sharpness_score", lambda gray: 0.0)
    monkeypatch.setattr(visual_qc, "noise_score", lambda gray: 40.0)
    bonus = vd.visual_qc_bonus("x.jpg")
    assert bonus == pytest.approx(-vd.VISUAL_QC_NOISE_WEIGHT)


def test_visual_qc_bonus_zero_on_failure(monkeypatch):
    import visual_qc

    def boom(path):
        raise RuntimeError("no file")

    monkeypatch.setattr(visual_qc, "_load_gray_normalized", boom)
    assert vd.visual_qc_bonus("missing.jpg") == 0.0


# ---------- _safe_shot_size / candidate_domain_for ----------

def test_safe_shot_size_delegates(monkeypatch):
    monkeypatch.setattr(vd.pipeline_smart, "estimate_shot_size", lambda path: "wide")
    assert vd._safe_shot_size("x.jpg") == "wide"


def test_safe_shot_size_none_on_failure(monkeypatch):
    def boom(path):
        raise RuntimeError("bad file")

    monkeypatch.setattr(vd.pipeline_smart, "estimate_shot_size", boom)
    assert vd._safe_shot_size("x.jpg") is None


def test_candidate_domain_for_delegates(monkeypatch):
    monkeypatch.setattr(vd.lr, "classify_domain", lambda path: ("snow", 0.1))
    assert vd.candidate_domain_for("x.jpg") == "snow"


# ---------- compute_extra_score (оркестрация) ----------

def _patch_all_neutral(monkeypatch):
    """Все под-сигналы на безопасный нейтральный дефолт — тесты ниже
    переопределяют РОВНО одну ось за раз, по тому же принципу, что
    test_visual_qc.py уже применяет к своим составным скорерам."""
    monkeypatch.setattr(vd, "sentence_relevance", lambda path, text: 0.0)
    monkeypatch.setattr(vd, "_safe_shot_size", lambda path: None)
    monkeypatch.setattr(vd, "candidate_domain_for", lambda path: None)
    monkeypatch.setattr(vd, "visual_qc_bonus", lambda path: 0.0)


def test_compute_extra_score_all_neutral_is_zero(monkeypatch):
    _patch_all_neutral(monkeypatch)
    score = vd.compute_extra_score("x.jpg", "narrative", "текст блока", None, [])
    assert score == 0.0


def test_compute_extra_score_uses_sentence_relevance_weight(monkeypatch):
    _patch_all_neutral(monkeypatch)
    monkeypatch.setattr(vd, "sentence_relevance", lambda path, text: 0.3)
    score = vd.compute_extra_score("x.jpg", "narrative", "текст блока", None, [])
    assert score == pytest.approx(0.3 * vd.SENTENCE_RELEVANCE_WEIGHT)


def test_compute_extra_score_adds_role_shot_size_bonus(monkeypatch):
    _patch_all_neutral(monkeypatch)
    monkeypatch.setattr(vd, "_safe_shot_size", lambda path: "detail")
    score = vd.compute_extra_score("x.jpg", "detail", "текст блока", None, [])
    assert score == pytest.approx(vd.ROLE_SHOT_SIZE_BONUS[("detail", "detail")])


def test_compute_extra_score_adds_arc_stage_shot_bonus(monkeypatch):
    _patch_all_neutral(monkeypatch)
    monkeypatch.setattr(vd, "_safe_shot_size", lambda path: "detail")
    score = vd.compute_extra_score("x.jpg", "narrative", "текст блока", None, [], arc_stage="слом")
    assert score == pytest.approx(vd.ARC_STAGE_SHOT_SIZE_BONUS[("слом", "detail")])


def test_compute_extra_score_combines_role_and_arc_stage_bonuses(monkeypatch):
    # Оба бонуса СКЛАДЫВАЮТСЯ (роль отвечает "что это за кадр", arc_stage —
    # "где мы в разоблачении мифа") — независимые сигналы, не взаимоисключающие.
    _patch_all_neutral(monkeypatch)
    monkeypatch.setattr(vd, "_safe_shot_size", lambda path: "detail")
    score = vd.compute_extra_score("x.jpg", "detail", "текст блока", None, [], arc_stage="слом")
    assert score == pytest.approx(vd.ROLE_SHOT_SIZE_BONUS[("detail", "detail")]
                                   + vd.ARC_STAGE_SHOT_SIZE_BONUS[("слом", "detail")])


def test_compute_extra_score_default_arc_stage_is_zero_bonus(monkeypatch):
    # arc_stage не передан (дефолт None) — эпизод без Speech Director,
    # байт-в-байт прежнее поведение (ноль влияния этой оси).
    _patch_all_neutral(monkeypatch)
    monkeypatch.setattr(vd, "_safe_shot_size", lambda path: "detail")
    score = vd.compute_extra_score("x.jpg", "narrative", "текст блока", None, [])
    assert score == 0.0


def test_compute_extra_score_adds_domain_match_bonus(monkeypatch):
    _patch_all_neutral(monkeypatch)
    monkeypatch.setattr(vd, "candidate_domain_for", lambda path: "snow")
    score = vd.compute_extra_score("x.jpg", "narrative", "текст блока", "snow", [])
    assert score == pytest.approx(vd.DOMAIN_MATCH_BONUS)


def test_compute_extra_score_subtracts_repetition_penalty(monkeypatch):
    _patch_all_neutral(monkeypatch)
    monkeypatch.setattr(vd, "candidate_domain_for", lambda path: "snow")
    recent = [("snow", "narrative")]
    score = vd.compute_extra_score("x.jpg", "narrative", "текст блока", None, recent)
    assert score == pytest.approx(-vd.REPETITION_PENALTY)


def test_compute_extra_score_adds_visual_qc_bonus(monkeypatch):
    _patch_all_neutral(monkeypatch)
    monkeypatch.setattr(vd, "visual_qc_bonus", lambda path: 0.05)
    score = vd.compute_extra_score("x.jpg", "narrative", "текст блока", None, [])
    assert score == pytest.approx(0.05)


def test_compute_extra_score_treats_none_relevance_as_zero(monkeypatch):
    _patch_all_neutral(monkeypatch)
    monkeypatch.setattr(vd, "sentence_relevance", lambda path, text: None)
    score = vd.compute_extra_score("x.jpg", "narrative", "текст блока", None, [])
    assert score == 0.0


# ---------- VISUAL_DIRECTOR_MODE — валидация env ----------

def test_invalid_mode_env_value_is_a_known_fallback_state():
    # Модуль уже импортирован при валидном/дефолтном окружении в этой
    # тестовой сессии — здесь только проверяем инвариант, что итоговое
    # значение всегда одно из объявленных режимов (сам fallback при
    # реальном неверном env проверяется вручную, см. docstring модуля).
    assert vd.VISUAL_DIRECTOR_MODE in vd._VISUAL_DIRECTOR_MODES


# ---------- cache_signature (P1-4 форензик-аудита: params_hash раньше не
# знал о VISUAL_DIRECTOR_MODE вообще — off -> assist на уже отрендеренном
# эпизоде не инвалидировал старые клипы temp_smart/) ----------

def test_cache_signature_off_is_stable_string(monkeypatch):
    monkeypatch.setattr(vd, "VISUAL_DIRECTOR_MODE", "off")
    assert vd.cache_signature() == "director:off"


def test_cache_signature_shadow_same_as_off(monkeypatch):
    # shadow никогда не трогает реальный выбор кандидата (см. докстринг
    # cache_signature) — рендер побитово идентичен off, инвалидация кэша
    # бессмысленна и вредна (лишний перерендер без единого визуального
    # отличия).
    monkeypatch.setattr(vd, "VISUAL_DIRECTOR_MODE", "shadow")
    assert vd.cache_signature() == "director:off"


def test_cache_signature_assist_differs_from_off(monkeypatch):
    monkeypatch.setattr(vd, "VISUAL_DIRECTOR_MODE", "assist")
    sig = vd.cache_signature()
    assert sig != "director:off"
    assert sig.startswith("director:assist:")


def test_cache_signature_changes_with_tunable_table(monkeypatch):
    # Регрессия на найденный аудитом класс бага: правка любой из
    # bonus/penalty-констант ассист-скоринга обязана менять сигнатуру,
    # иначе старые закэшированные клипы молча переживут правку рецепта
    # (тот же принцип, что test_domain_grade_cache_signature_changes_with_table
    # в tests/test_parse.py).
    monkeypatch.setattr(vd, "VISUAL_DIRECTOR_MODE", "assist")
    before = vd.cache_signature()
    monkeypatch.setattr(vd, "REPETITION_PENALTY", 0.99)
    after = vd.cache_signature()
    assert before != after


def test_pipeline_smart_visual_director_cache_signature_off_without_module():
    import pipeline_smart
    assert pipeline_smart._visual_director_cache_signature(None) == "director:off"


def test_pipeline_smart_visual_director_cache_signature_delegates_to_module(monkeypatch):
    import pipeline_smart
    monkeypatch.setattr(vd, "VISUAL_DIRECTOR_MODE", "assist")
    sig = pipeline_smart._visual_director_cache_signature(vd)
    assert sig == vd.cache_signature()
    assert sig.startswith("director:assist:")


def test_director_min_pool_stays_in_sync_across_modules():
    # pipeline_smart.DIRECTOR_MIN_POOL реально управляет good_needed в
    # pexels_photo(); vd.DIRECTOR_MIN_POOL — отдельная константа-дубликат
    # (обратный импорт создал бы цикл), используется только для
    # cache_signature()/докстрингов. Расхождение тихо сломало бы
    # инвалидацию кэша при смене значения — см. реальный найденный случай
    # при апгрейде 3->8 вместе с so400m+Jina ensemble.
    import pipeline_smart
    assert pipeline_smart.DIRECTOR_MIN_POOL == vd.DIRECTOR_MIN_POOL


# ---------- sentence_relevance: РЕАЛЬНАЯ модель (SigLIP2) на реальных
# фото + реальной русской фразе сценария (не мок — та же логика, что golden
# CLIP-тесты в test_media_selection_golden.py: сама суть регрессии, которую
# нужно ловить, это ошибка семантики реальной модели, не что-то кодируемое
# в мок-объекте). Фраза — дословно из videos/01_ves-mecha/script.txt. ----------

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

_GOLDEN_FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "golden_media")
_GOLDEN_SWORD = os.path.join(_GOLDEN_FIXTURES, "sword.jpg")
_GOLDEN_PIZZA = os.path.join(_GOLDEN_FIXTURES, "pizza.jpg")


class TestSentenceRelevanceSiglip2RealModel:
    def test_real_sentence_prefers_matching_photo_over_unrelated(self):
        text = "Обычный одноручный рыцарский меч весит от килограмма до полутора."
        r_sword = vd.sentence_relevance(_GOLDEN_SWORD, text)
        r_pizza = vd.sentence_relevance(_GOLDEN_PIZZA, text)
        assert r_sword is not None and r_pizza is not None
        assert r_sword > r_pizza, (
            f"реальная русская фраза сценария про меч должна давать более высокую "
            f"близость к фото меча, чем к фото пиццы: sword={r_sword}, pizza={r_pizza}")

    def test_regression_raw_russian_text_was_near_noise_on_english_only_clip(self):
        # Документирует РЕАЛЬНУЮ причину фикса: та же фраза через ОДНОЯЗЫЧНУЮ
        # (английскую) pipeline_smart.clip_relevance() не различает meч/пиццу
        # на русском тексте — разница в пределах шума. Это не проверка самого
        # sentence_relevance() (уже проверено выше), а живая регрессия на
        # СТАРОЕ поведение, чтобы факт бага не потерялся молча, если кто-то
        # решит "упростить" реализацию обратно.
        text = "Обычный одноручный рыцарский меч весит от килограмма до полутора."
        r_sword = vd.pipeline_smart.clip_relevance(_GOLDEN_SWORD, text)
        r_pizza = vd.pipeline_smart.clip_relevance(_GOLDEN_PIZZA, text)
        assert r_sword is not None and r_pizza is not None
        assert abs(r_sword - r_pizza) < 0.05, (
            "если это упало — одноязычный CLIP неожиданно научился различать "
            "русский текст, sentence_relevance можно упростить обратно; "
            "пока что это ожидаемо мал")
