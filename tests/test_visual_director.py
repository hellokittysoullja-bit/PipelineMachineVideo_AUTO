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


def test_sentence_relevance_delegates_to_clip_relevance(monkeypatch):
    calls = []

    def fake_clip_relevance(path, text):
        calls.append((path, text))
        return 0.27

    monkeypatch.setattr(vd.pipeline_smart, "clip_relevance", fake_clip_relevance)
    result = vd.sentence_relevance("x.jpg", "полный текст фразы блока")
    assert result == 0.27
    assert calls == [("x.jpg", "полный текст фразы блока")]


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
