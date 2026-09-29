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


def _fake_vectors(monkeypatch, img, txt):
    import numpy as np
    monkeypatch.setattr(vd.pipeline_smart, "_gate_image_vec",
                        lambda path: None if img is None else np.asarray(img, dtype="float32"))
    monkeypatch.setattr(vd.pipeline_smart, "_gate_text_vec",
                        lambda text: None if txt is None else np.asarray(txt, dtype="float32"))


def test_sentence_relevance_uses_the_gate_vectors_and_the_legacy_scale(monkeypatch):
    """GPU-ветка (29.09): оценка фразы — та же модель, что у гейтов (общие
    векторы и кэши), сходство переведено на шкалу прежнего ансамбля, на
    которой заданы веса и бонусы режиссёра."""
    _fake_vectors(monkeypatch, [0.6, 0.8], [1.0, 0.0])
    seen = []

    def to_legacy(value, kind):
        seen.append((value, kind))
        return value * 10
    monkeypatch.setattr(vd.vision_model, "to_legacy_level", to_legacy)
    assert vd.sentence_relevance("x.jpg", "фраза блока") == pytest.approx(6.0)
    assert seen == [(pytest.approx(0.6), "sentence")]


def test_sentence_relevance_none_without_calibration(monkeypatch):
    _fake_vectors(monkeypatch, [1.0, 0.0], [1.0, 0.0])

    def no_cal(value, kind):
        raise vd.vision_model.NotCalibrated("нет калибровки")
    monkeypatch.setattr(vd.vision_model, "to_legacy_level", no_cal)
    assert vd.sentence_relevance("x.jpg", "фраза") is None


def test_sentence_relevance_none_when_model_does_not_answer(monkeypatch):
    _fake_vectors(monkeypatch, None, [1.0, 0.0])
    assert vd.sentence_relevance("x.jpg", "фраза") is None


def test_text_text_similarity_is_a_matrix_of_gate_text_vectors(monkeypatch):
    import numpy as np
    vecs = {"а": [1.0, 0.0], "б": [0.0, 1.0], "q": [0.6, 0.8]}
    monkeypatch.setattr(vd.pipeline_smart, "_gate_text_vec",
                        lambda t: np.asarray(vecs[t], dtype="float32"))
    got = vd.text_text_similarity(["а", "б"], ["q"])
    assert got == [[pytest.approx(0.6)], [pytest.approx(0.8)]]
    monkeypatch.setattr(vd.pipeline_smart, "_gate_text_vec", lambda t: None)
    assert vd.text_text_similarity(["а"], ["q"]) is None


def test_model_version_names_qwen_not_siglip_or_jina():
    v = vd.SENTENCE_RELEVANCE_MODEL_VERSION.lower()
    assert "qwen" in v and "siglip" not in v and "jina" not in v


def test_no_siglip2_or_jina_left_in_the_director():
    """Решение владельца 29.09: SigLIP2 убрана из GPU-ветки целиком, Jina —
    некоммерческая лицензия. Ни загрузчиков, ни констант их шкал."""
    for name in ("_get_siglip2_model", "_siglip2_relevance", "_get_jina_session",
                 "_jina_relevance", "SIGLIP2_MODEL_NAME", "JINA_MODEL_REPO",
                 "ENSEMBLE_WEIGHT_JINA"):
        assert not hasattr(vd, name), name


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


# ---------- SAME_QUERY_BONUS (см. её докстринг: слот с чужого запроса
# перетягивал победу у слота, для которого запрос назначен ПО СМЫСЛУ) ----------

def test_compute_extra_score_adds_same_query_bonus_when_matching(monkeypatch):
    _patch_all_neutral(monkeypatch)
    score = vd.compute_extra_score("x.jpg", "narrative", "текст блока", None, [],
                                    own_query="scale", candidate_query="scale")
    assert score == pytest.approx(vd.SAME_QUERY_BONUS)


def test_compute_extra_score_no_bonus_when_query_mismatched(monkeypatch):
    _patch_all_neutral(monkeypatch)
    score = vd.compute_extra_score("x.jpg", "narrative", "текст блока", None, [],
                                    own_query="scale", candidate_query="sword")
    assert score == 0.0


def test_compute_extra_score_no_bonus_when_query_info_missing(monkeypatch):
    # own_query/candidate_query оба None (старые вызовы без этой оси) —
    # ноль изменений поведения.
    _patch_all_neutral(monkeypatch)
    score = vd.compute_extra_score("x.jpg", "narrative", "текст блока", None, [])
    assert score == 0.0


# ---------- OPENING_AESTHETIC_WEIGHT (открывающий кадр — эстетика решает
# base/director выбор ДАЖЕ без VLM-арбитра, см. её блок-комментарий) ----------

def test_compute_extra_score_adds_opening_aesthetic_bonus(monkeypatch):
    _patch_all_neutral(monkeypatch)
    # aesthetic_val=7.0 (верхняя граница документированного диапазона 3-7)
    # -> normalized=(7-3)/4=1.0 -> bonus = 1.0 * OPENING_AESTHETIC_WEIGHT
    score = vd.compute_extra_score("x.jpg", "narrative", "текст блока", None, [],
                                    is_opening=True, aesthetic_val=7.0)
    assert score == pytest.approx(1.0 * vd.OPENING_AESTHETIC_WEIGHT)


def test_compute_extra_score_no_opening_bonus_when_not_opening(monkeypatch):
    _patch_all_neutral(monkeypatch)
    score = vd.compute_extra_score("x.jpg", "narrative", "текст блока", None, [],
                                    is_opening=False, aesthetic_val=7.0)
    assert score == 0.0


def test_compute_extra_score_no_opening_bonus_when_aesthetic_val_missing(monkeypatch):
    _patch_all_neutral(monkeypatch)
    score = vd.compute_extra_score("x.jpg", "narrative", "текст блока", None, [],
                                    is_opening=True, aesthetic_val=None)
    assert score == 0.0


def test_compute_extra_score_opening_bonus_clamped_at_floor(monkeypatch):
    _patch_all_neutral(monkeypatch)
    # aesthetic_val ниже документированного диапазона (3-7) -> normalized
    # клэмпится к 0, не уходит в отрицательное (не штрафует низкую эстетику
    # сверх того, что и так даёт естественная разница нормализованных значений).
    score = vd.compute_extra_score("x.jpg", "narrative", "текст блока", None, [],
                                    is_opening=True, aesthetic_val=0.0)
    assert score == 0.0


def test_compute_extra_score_opening_bonus_clamped_at_ceiling(monkeypatch):
    _patch_all_neutral(monkeypatch)
    # aesthetic_val сильно выше документированного диапазона -> клэмп 1.5,
    # не даёт единичному выдающемуся кадру задавить остальные сигналы бесконечно.
    score = vd.compute_extra_score("x.jpg", "narrative", "текст блока", None, [],
                                    is_opening=True, aesthetic_val=100.0)
    assert score == pytest.approx(1.5 * vd.OPENING_AESTHETIC_WEIGHT)


def test_compute_extra_score_opening_suppresses_same_query_bonus(monkeypatch):
    # 29 августа, реальный найденный конфликт: SAME_QUERY_BONUS награждает
    # буквальную точность запроса, opening-кадр должен предпочитать
    # эффектность точности — бонус НЕ должен применяться при is_opening=True,
    # даже если own_query/candidate_query совпадают.
    _patch_all_neutral(monkeypatch)
    score = vd.compute_extra_score("x.jpg", "narrative", "текст блока", None, [],
                                    own_query="scale", candidate_query="scale",
                                    is_opening=True, aesthetic_val=7.0)
    assert score == pytest.approx(1.0 * vd.OPENING_AESTHETIC_WEIGHT)


def test_compute_extra_score_same_query_bonus_still_applies_when_not_opening(monkeypatch):
    # Регрессия: для всех остальных слотов (is_opening=False, дефолт)
    # SAME_QUERY_BONUS не тронут этим изменением.
    _patch_all_neutral(monkeypatch)
    score = vd.compute_extra_score("x.jpg", "narrative", "текст блока", None, [],
                                    own_query="scale", candidate_query="scale",
                                    is_opening=False, aesthetic_val=7.0)
    assert score == pytest.approx(vd.SAME_QUERY_BONUS)


# ---------- OPENING_DOMAIN_WEIGHT (открывающий кадр — усиленный,
# самостоятельный бонус за жанровое совпадение, заменяет обычный
# DOMAIN_MATCH_BONUS только для is_opening=True, см. её блок-комментарий) ----------

def test_compute_extra_score_opening_domain_weight_when_domains_match(monkeypatch):
    _patch_all_neutral(monkeypatch)
    monkeypatch.setattr(vd, "candidate_domain_for", lambda path: "battle")
    score = vd.compute_extra_score("x.jpg", "narrative", "текст блока", "battle", [],
                                    is_opening=True)
    assert score == pytest.approx(vd.OPENING_DOMAIN_WEIGHT)


def test_compute_extra_score_opening_domain_weight_not_applied_when_domains_differ(monkeypatch):
    # Домены не совпадают -> обычный domain_match_bonus (даёт 0 для
    # несовпадающих непустых доменов), не OPENING_DOMAIN_WEIGHT.
    _patch_all_neutral(monkeypatch)
    monkeypatch.setattr(vd, "candidate_domain_for", lambda path: "snow")
    score = vd.compute_extra_score("x.jpg", "narrative", "текст блока", "battle", [],
                                    is_opening=True)
    assert score == pytest.approx(vd.domain_match_bonus("snow", "battle"))


def test_compute_extra_score_not_opening_uses_regular_domain_match_bonus(monkeypatch):
    # is_opening=False (дефолт) -> прежнее поведение, OPENING_DOMAIN_WEIGHT
    # не участвует вообще, даже при совпадающих доменах.
    _patch_all_neutral(monkeypatch)
    monkeypatch.setattr(vd, "candidate_domain_for", lambda path: "battle")
    score = vd.compute_extra_score("x.jpg", "narrative", "текст блока", "battle", [])
    assert score == pytest.approx(vd.DOMAIN_MATCH_BONUS)
    assert score != pytest.approx(vd.OPENING_DOMAIN_WEIGHT)


def test_compute_extra_score_opening_combines_domain_and_aesthetic_bonuses(monkeypatch):
    # Оба opening-сигнала независимы и складываются вместе.
    _patch_all_neutral(monkeypatch)
    monkeypatch.setattr(vd, "candidate_domain_for", lambda path: "battle")
    score = vd.compute_extra_score("x.jpg", "narrative", "текст блока", "battle", [],
                                    is_opening=True, aesthetic_val=7.0)
    assert score == pytest.approx(vd.OPENING_DOMAIN_WEIGHT + 1.0 * vd.OPENING_AESTHETIC_WEIGHT)


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


# ---------- sentence_relevance: РЕАЛЬНАЯ модель (Qwen3-VL) на реальных фото +
# реальной русской фразе сценария — только там, где модели готовы
# (видеокарта, веса, калибровка). Фраза — дословно из videos/01_ves-mecha. ----------

_GOLDEN_FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "golden_media")
_GOLDEN_SWORD = os.path.join(_GOLDEN_FIXTURES, "sword.jpg")
_GOLDEN_PIZZA = os.path.join(_GOLDEN_FIXTURES, "pizza.jpg")


class TestSentenceRelevanceRealModel:
    def test_real_sentence_prefers_matching_photo_over_unrelated(self):
        problems = vd.vision_model.readiness()
        if problems:
            pytest.skip("модели зрения не готовы: " + "; ".join(problems))
        text = "Обычный одноручный рыцарский меч весит от килограмма до полутора."
        r_sword = vd.sentence_relevance(_GOLDEN_SWORD, text)
        r_pizza = vd.sentence_relevance(_GOLDEN_PIZZA, text)
        assert r_sword is not None and r_pizza is not None
        assert r_sword > r_pizza, f"sword={r_sword}, pizza={r_pizza}"


# ---------- порог режиссёра (DIRECTOR_RELEVANCE_FLOOR) ----------
# Методика — calibrate_relevance_floor(); зовёт её scripts/calibrate_vision.py,
# результат читается из калибровки моделей зрения. Модель мокается.

def _write_pairs(path, pairs):
    import json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"pairs": pairs}, f)


def test_load_calibration_pairs_skips_missing_images(tmp_path, monkeypatch):
    pairs_path = str(tmp_path / "pairs.json")
    real_img = str(tmp_path / "real.jpg")
    open(real_img, "wb").close()
    _write_pairs(pairs_path, [
        {"image": os.path.relpath(real_img, vd.REPO_ROOT), "caption": "x", "label": "good"},
        {"image": "does/not/exist.jpg", "caption": "y", "label": "bad"},
    ])
    monkeypatch.setattr(vd, "CALIBRATION_PAIRS_PATH", pairs_path)
    pairs = vd._load_calibration_pairs()
    assert len(pairs) == 1
    assert pairs[0][2] == "good"


def test_load_calibration_pairs_none_on_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(vd, "CALIBRATION_PAIRS_PATH", str(tmp_path / "nope.json"))
    assert vd._load_calibration_pairs() is None


def test_calibrate_relevance_floor_places_threshold_in_the_gap(monkeypatch, tmp_path):
    pairs_path = str(tmp_path / "pairs.json")
    imgs = {}
    for name in ("g1", "g2", "b1", "b2"):
        p = str(tmp_path / f"{name}.jpg")
        open(p, "wb").close()
        imgs[name] = p
    _write_pairs(pairs_path, [
        {"image": os.path.relpath(imgs["g1"], vd.REPO_ROOT), "caption": "good1", "label": "good"},
        {"image": os.path.relpath(imgs["g2"], vd.REPO_ROOT), "caption": "good2", "label": "good"},
        {"image": os.path.relpath(imgs["b1"], vd.REPO_ROOT), "caption": "bad1", "label": "bad"},
        {"image": os.path.relpath(imgs["b2"], vd.REPO_ROOT), "caption": "bad2", "label": "bad"},
    ])
    monkeypatch.setattr(vd, "CALIBRATION_PAIRS_PATH", pairs_path)
    scores = {"good1": 0.20, "good2": 0.18, "bad1": 0.05, "bad2": 0.02}
    monkeypatch.setattr(vd, "sentence_relevance", lambda path, caption: scores[caption])
    result = vd.calibrate_relevance_floor()
    assert result is not None
    assert result["min_good"] == 0.18
    assert result["max_bad"] == 0.05
    assert abs(result["margin"] - 0.13) < 1e-9
    assert result["max_bad"] < result["floor"] < result["min_good"]


def test_calibrate_relevance_floor_none_when_model_unavailable(monkeypatch, tmp_path):
    pairs_path = str(tmp_path / "pairs.json")
    p = str(tmp_path / "a.jpg")
    open(p, "wb").close()
    _write_pairs(pairs_path, [{"image": os.path.relpath(p, vd.REPO_ROOT), "caption": "x", "label": "good"}])
    monkeypatch.setattr(vd, "CALIBRATION_PAIRS_PATH", pairs_path)
    monkeypatch.setattr(vd, "sentence_relevance", lambda path, caption: None)
    assert vd.calibrate_relevance_floor() is None


def test_calibrate_relevance_floor_none_on_missing_pairs(monkeypatch, tmp_path):
    monkeypatch.setattr(vd, "CALIBRATION_PAIRS_PATH", str(tmp_path / "nope.json"))
    assert vd.calibrate_relevance_floor() is None


def test_calibrate_relevance_floor_takes_an_explicit_score_fn(monkeypatch, tmp_path):
    """Калибровка зовёт методику со своей оценкой (шкалы ещё нет в файле)."""
    pairs_path = str(tmp_path / "pairs.json")
    for name in ("g", "b"):
        open(str(tmp_path / f"{name}.jpg"), "wb").close()
    _write_pairs(pairs_path, [
        {"image": os.path.relpath(str(tmp_path / "g.jpg"), vd.REPO_ROOT), "caption": "good", "label": "good"},
        {"image": os.path.relpath(str(tmp_path / "b.jpg"), vd.REPO_ROOT), "caption": "bad", "label": "bad"},
    ])
    monkeypatch.setattr(vd, "CALIBRATION_PAIRS_PATH", pairs_path)
    monkeypatch.setattr(vd, "sentence_relevance",
                        lambda *a: (_ for _ in ()).throw(AssertionError("не та оценка")))
    got = vd.calibrate_relevance_floor(score_fn=lambda p, c: {"good": 0.3, "bad": 0.1}[c])
    assert got["floor"] == pytest.approx(0.2)


def test_resolve_floor_reads_the_vision_calibration(monkeypatch):
    monkeypatch.setattr(vd.vision_model, "threshold",
                        lambda name, default=None: 0.42 if name == "director_floor" else None)
    assert vd._resolve_director_relevance_floor() == 0.42


def test_resolve_floor_falls_back_when_calibration_has_no_floor(monkeypatch):
    monkeypatch.setattr(vd.vision_model, "threshold", lambda name, default=None: None)
    assert vd._resolve_director_relevance_floor() == vd.DIRECTOR_RELEVANCE_FALLBACK


def test_resolve_floor_falls_back_without_calibration(monkeypatch):
    def no_cal(name, default=None):
        raise vd.vision_model.NotCalibrated("нет файла")
    monkeypatch.setattr(vd.vision_model, "threshold", no_cal)
    assert vd._resolve_director_relevance_floor() == vd.DIRECTOR_RELEVANCE_FALLBACK


def test_real_calibration_pairs_file_is_valid_and_channel_agnostic():
    """Не мок — проверяет реальный committed tests/fixtures/director_
    calibration/calibration_pairs.json: файлы существуют, оба лейбла
    представлены, и набор НЕ состоит из одних мечей/рыцарей (см. ЧАСТЬ 24
    CLAUDE.md — калибровка не должна тащить нишу текущего канала в клон
    репозитория под другую тематику)."""
    pairs = vd._load_calibration_pairs()
    assert pairs is not None and len(pairs) >= 8
    labels = {label for _, _, label in pairs}
    assert labels == {"good", "bad"}
    non_sword_images = {os.path.basename(img) for img, _, _ in pairs
                         if "sword" not in os.path.basename(img) and "katana" not in os.path.basename(img)}
    assert len(non_sword_images) >= 2, "калибровка не должна состоять из одной ниши (ЧАСТЬ 24)"
