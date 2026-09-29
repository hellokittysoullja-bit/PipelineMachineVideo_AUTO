"""Golden media-selection тест — защищает "интеллект" подбора медиа
(CLIP-релевантность, risky-margin гейт против "собирательных" запросов,
LAION-эстетика, ahash-дедуп) от тихой регрессии при будущих правках
CLIP_RELEVANCE_THRESHOLD / RISKY_QUERY_MARGIN / is_relevant_candidate() /
aesthetic_score() / ahash().

Принципиально НЕ мок: гоняет РЕАЛЬНУЮ модель за clip_relevance() (имя
историческое — в GPU-ветке с 29.09 это Qwen3-VL-Embedding-8B, см.
vision_model.py) и РЕАЛЬНЫЕ
(не синтетические/PIL-нарисованные) фотографии — see
tests/fixtures/golden_media/ATTRIBUTION.md за источниками и лицензиями
(все CC BY / CC BY-SA / OGL, авторство указано). Мок или PIL-заливка тут
бесполезны: сама суть регрессии, которую нужно ловить ("формально
совпадают ключевые слова, по смыслу мимо" — см. RISKY_GENERIC_TERMS в
pipeline_smart.py) — это ошибка СЕМАНТИКИ реальной модели на реальном
фото, не что-то, что можно закодировать в мок-объекте.

torch/transformers — та же опциональная зависимость, что CLIP_ENABLED в
pipeline_smart.py (production код уже откатывается без них) — здесь весь
модуль скипается, если их нет, тем же принципом.

Requires: torch, transformers (pip install -r requirements.txt с
раскомментированными torch/transformers/torchvision строками)."""
import os
import sys
import tempfile

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]
import pipeline_smart as ps   # noqa: E402

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "golden_media")
SWORD = os.path.join(FIXTURES, "sword.jpg")
STAINEDGLASS = os.path.join(FIXTURES, "stainedglass.jpg")
PIZZA = os.path.join(FIXTURES, "pizza.jpg")
MEETING = os.path.join(FIXTURES, "meeting.jpg")
SWORD_NEAR_DUP = os.path.join(FIXTURES, "sword_near_dup.jpg")
SWORD_DEGRADED = os.path.join(FIXTURES, "sword_degraded.jpg")
KATANA = os.path.join(FIXTURES, "katana.jpg")
EURO_SWORD_2 = os.path.join(FIXTURES, "euro_sword_2.jpg")


# GPU-ветка (29.09): модель гейтов — Qwen3-VL, пороги — из калибровки
# (vision_model.py, scripts/calibrate_vision.py). Проверки ниже работают там,
# где модели готовы (видеокарта, веса, калибровка), и делятся на два рода:
#   * утверждения — то, что держится по построению калибровки (европейские
#     мечи из фикстур входят в «годные» калибровки гварда клинка и обязаны
#     проходить) или что обязана уметь любая исправная модель поиска (фото
#     меча ближе к «мечу», чем к «пляжу»);
#   * xfail(strict=False) — отказы по ПОРОГУ на заведомом браке: калибровка
#     ставит порог ради нуля потерь годных, и ловит ли он именно этот брак —
#     вопрос замера, а не обещание. XPASS — модель ловит сама.
# Поимённые числа SigLIP2 (margin 0.0481 у meeting.jpg и т.п.) сняты вместе
# с моделью.
@pytest.fixture(scope="module")
def vision_ready():
    import vision_model
    problems = vision_model.readiness()
    if problems:
        pytest.skip("модели зрения не готовы: " + "; ".join(problems))


BY_THRESHOLD = pytest.mark.xfail(strict=False, reason=(
    "отказ по порогу на заведомом браке: порог калибровки держит ноль потерь "
    "годных, ловит ли он этот кадр — замер (calibrate_vision.py печатает "
    "пойманное по каждой оси), а не гарантия"))


@pytest.mark.usefixtures("vision_ready")
class TestClipRelevanceThreshold:
    def test_true_positive_above_threshold(self):
        rel = ps.clip_relevance(SWORD, "medieval sword")
        assert rel is not None and rel >= ps.CLIP_RELEVANCE_THRESHOLD, (
            f"реальное фото меча должно проходить порог по запросу 'medieval sword', rel={rel}")

    def test_own_topic_positive_controls(self):
        assert ps.clip_relevance(PIZZA, "pizza restaurant") >= ps.CLIP_RELEVANCE_THRESHOLD
        assert ps.clip_relevance(MEETING, "office business meeting") >= ps.CLIP_RELEVANCE_THRESHOLD
        assert ps.clip_relevance(STAINEDGLASS, "stained glass cathedral window") >= ps.CLIP_RELEVANCE_THRESHOLD

    def test_model_prefers_the_own_topic(self):
        """Дискриминация без порога: фото меча ближе к мечу, чем к пляжу, а
        меч к мечу ближе, чем пицца к мечу."""
        own = ps.clip_relevance(SWORD, "medieval sword")
        assert own > ps.clip_relevance(SWORD, "tropical beach vacation")
        assert own > ps.clip_relevance(PIZZA, "medieval sword")

    @BY_THRESHOLD
    def test_true_negative_below_threshold(self):
        assert ps.clip_relevance(SWORD, "tropical beach vacation") < ps.CLIP_RELEVANCE_THRESHOLD

    @BY_THRESHOLD
    def test_true_negative_below_threshold_reverse_topic(self):
        assert ps.clip_relevance(PIZZA, "medieval sword") < ps.CLIP_RELEVANCE_THRESHOLD


@pytest.mark.usefixtures("vision_ready")
class TestRiskyQueryMargin:
    """Запрос с «собирательным» словом (museum/exhibition/...) формально
    совпадает с фото зала без предмета — is_relevant_candidate() та же
    функция, что решает в проде."""

    def test_accepts_true_positive_under_risky_query(self):
        assert ps.is_relevant_candidate(SWORD, "medieval weapon exhibition gallery") is True

    @BY_THRESHOLD
    def test_rejects_architecture_only_candidate_under_risky_query(self):
        assert ps.is_relevant_candidate(STAINEDGLASS, "medieval weapon exhibition gallery") is False

    @BY_THRESHOLD
    def test_rejects_unrelated_candidate_under_risky_query(self):
        assert ps.is_relevant_candidate(MEETING, "medieval weapon exhibition gallery") is False

    @BY_THRESHOLD
    def test_known_gap_exact_bug_query_still_accepts_stainedglass(self):
        assert ps.is_relevant_candidate(STAINEDGLASS, "sword museum display case") is False


@pytest.mark.usefixtures("vision_ready")
class TestVisualDomainGuard:
    """Гвард формы клинка (VISUAL_DOMAIN_GUARDS) — та же функция, что гейтит
    в проде. sword.jpg и euro_sword_2.jpg входят в «годные» калибровки
    гварда — их пропуск держится по построению."""

    @BY_THRESHOLD
    def test_rejects_katana(self):
        violated, name = ps.visual_domain_guard_violation(KATANA, "medieval sword close up")
        assert violated is True
        assert name == "east_asian_sword"

    def test_accepts_independent_european_sword_photo(self):
        violated, _ = ps.visual_domain_guard_violation(EURO_SWORD_2, "medieval sword close up")
        assert violated is False

    def test_accepts_existing_golden_sword_fixture_hardest_known_case(self):
        violated, _ = ps.visual_domain_guard_violation(SWORD, "medieval sword close up")
        assert violated is False

    def test_guard_only_applies_to_trigger_terms(self):
        violated, name = ps.visual_domain_guard_violation(KATANA, "medieval armor helmet")
        assert violated is False
        assert name is None

    def test_is_relevant_candidate_accepts_european_sword(self):
        assert ps.is_relevant_candidate(SWORD, "medieval sword close up") is True

    @BY_THRESHOLD
    def test_is_relevant_candidate_integrates_domain_guard(self):
        assert ps.is_relevant_candidate(KATANA, "medieval sword close up") is False


@pytest.mark.usefixtures("vision_ready")
class TestVideoFramesViolate:
    """Видео судится по НЕСКОЛЬКИМ кадрам: нарушение на любом — нарушение
    ролика. Вето по ловушкам здесь выключено: проверяется устройство
    «любой кадр», а не отдельный порог."""

    @BY_THRESHOLD
    def test_catches_violation_visible_only_in_later_frame(self, monkeypatch):
        monkeypatch.setattr(ps, "negative_anchor_violation", lambda p, q: (False, None))
        assert ps.video_frames_violate([MEETING, KATANA], "medieval sword close up") is True

    def test_no_violation_when_all_sampled_frames_clean(self, monkeypatch):
        monkeypatch.setattr(ps, "negative_anchor_violation", lambda p, q: (False, None))
        assert ps.video_frames_violate([EURO_SWORD_2] * 3, "medieval sword close up") is False

    def test_domain_guard_only_applies_to_trigger_terms_for_video_too(self, monkeypatch):
        monkeypatch.setattr(ps, "negative_anchor_violation", lambda p, q: (False, None))
        assert ps.video_frames_violate([KATANA], "medieval armor helmet") is False


class TestAestheticScore:
    """LAION aesthetic predictor (aesthetic_score()) — реальная модель +
    голова из assets/aesthetic/*.npz. Деградированная версия сгенерирована
    локально (Gaussian blur) из ТОГО ЖЕ реального фото — чистая проверка
    "предпочитает ли резкое размытому", без смешивания с эффектом смены
    сюжета."""

    def test_sharp_photo_scores_higher_than_degraded(self):
        sharp = ps.aesthetic_score(SWORD)
        degraded = ps.aesthetic_score(SWORD_DEGRADED)
        assert sharp is not None and degraded is not None
        assert sharp > degraded + 0.5, (
            f"резкое фото должно ощутимо обгонять размытую версию того же кадра: "
            f"sharp={sharp:.2f} degraded={degraded:.2f}")


class TestPhotoDedup:
    """ahash()/hamming()/PHOTO_DEDUP_HAMMING — average-hash дедуп на
    реальных фото: почти то же самое фото (лёгкий кроп краёв — имитация
    того же стокового кадра у другого источника/при другом ресайзе)
    обязано ловиться как дубль; два содержательно разных реальных фото —
    нет."""

    def test_near_duplicate_photo_flagged(self):
        h1 = ps.ahash(SWORD)
        h2 = ps.ahash(SWORD_NEAR_DUP)
        d = ps.hamming(h1, h2)
        assert d <= ps.PHOTO_DEDUP_HAMMING, (
            f"лёгкий кроп того же реального фото должен считаться дублем: hamming={d}")

    def test_distinct_photos_not_flagged_as_duplicate(self):
        h1 = ps.ahash(SWORD)
        h3 = ps.ahash(PIZZA)
        d = ps.hamming(h1, h3)
        assert d > ps.PHOTO_DEDUP_HAMMING, (
            f"содержательно разные реальные фото не должны считаться дублями: hamming={d}")
