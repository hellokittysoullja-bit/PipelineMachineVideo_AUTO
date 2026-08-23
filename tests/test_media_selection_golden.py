"""Golden media-selection тест — защищает "интеллект" подбора медиа
(CLIP-релевантность, risky-margin гейт против "собирательных" запросов,
LAION-эстетика, ahash-дедуп) от тихой регрессии при будущих правках
CLIP_RELEVANCE_THRESHOLD / RISKY_QUERY_MARGIN / is_relevant_candidate() /
aesthetic_score() / ahash().

Принципиально НЕ мок: гоняет РЕАЛЬНУЮ модель openai/clip-vit-base-patch32
и РЕАЛЬНЫЕ (не синтетические/PIL-нарисованные) фотографии — see
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


@pytest.fixture(scope="module", autouse=True)
def _warm_clip_model():
    # Первый вызов clip_relevance() в первом тесте иначе платил бы за
    # загрузку модели — не влияет на корректность, только на то, какой
    # именно тест "внезапно" долгий в выводе pytest -v.
    ps.get_clip_model()


class TestClipRelevanceThreshold:
    """CLIP_RELEVANCE_THRESHOLD=0.19 (см. калибровку в pipeline_smart.py:
    20 верных пар дали 0.217-0.303, 15 посторонних — 0.118-0.201). Живая
    проверка на РЕАЛЬНЫХ независимых фото (не из исходной калибровки),
    что порог всё ещё разделяет верно, а не только на данных, на которых
    его когда-то подобрали."""

    def test_true_positive_above_threshold(self):
        rel = ps.clip_relevance(SWORD, "medieval sword")
        assert rel is not None and rel >= ps.CLIP_RELEVANCE_THRESHOLD, (
            f"реальное фото меча должно проходить порог по запросу 'medieval sword', rel={rel}")

    def test_true_negative_below_threshold(self):
        rel = ps.clip_relevance(SWORD, "pizza restaurant")
        assert rel is not None and rel < ps.CLIP_RELEVANCE_THRESHOLD, (
            f"фото меча НЕ должно проходить порог по посторонней теме 'pizza restaurant', rel={rel}")

    def test_true_negative_below_threshold_reverse_topic(self):
        rel = ps.clip_relevance(PIZZA, "medieval sword")
        assert rel is not None and rel < ps.CLIP_RELEVANCE_THRESHOLD, (
            f"фото пиццы НЕ должно проходить порог по запросу про меч, rel={rel}")

    def test_own_topic_positive_controls(self):
        # Позитивный контроль: каждое фото должно узнавать СВОЮ тему —
        # иначе непонятно, тестируем реальную дискриминацию или просто
        # общий сдвиг скоров вниз/вверх у конкретной модели.
        assert ps.clip_relevance(PIZZA, "pizza restaurant") >= ps.CLIP_RELEVANCE_THRESHOLD
        assert ps.clip_relevance(MEETING, "office business meeting") >= ps.CLIP_RELEVANCE_THRESHOLD
        assert ps.clip_relevance(STAINEDGLASS, "stained glass cathedral window") >= ps.CLIP_RELEVANCE_THRESHOLD


class TestRiskyQueryMargin:
    """Регрессия на РЕАЛЬНЫЙ пойманный вживую баг (см. комментарий у
    RISKY_GENERIC_TERMS/NEGATIVE_ANCHOR_PROMPT в pipeline_smart.py):
    запрос с "собирательным" словом (museum/exhibition/collection/
    display/cabinet/case/gallery) формально совпадает по ключевым словам
    с фото архитектуры/интерьера, где предмета темы вообще нет в кадре —
    "sword museum display case" стабильно возвращал витражные окна
    музейного зала. is_relevant_candidate() — та же функция (вынесена из
    инлайна главного цикла подбора кандидатов специально для этого теста
    в этом же коммите), что реально принимает решение в проде, не копия
    её логики."""

    def test_accepts_true_positive_under_risky_query(self):
        assert ps.is_relevant_candidate(SWORD, "medieval weapon exhibition gallery") is True

    def test_rejects_architecture_only_candidate_under_risky_query(self):
        assert ps.is_relevant_candidate(STAINEDGLASS, "medieval weapon exhibition gallery") is False, (
            "фото витражных окон кафедрального собора БЕЗ меча в кадре не должно "
            "проходить risky-margin гейт по запросу про оружейную выставку")

    def test_rejects_unrelated_candidate_under_risky_query(self):
        assert ps.is_relevant_candidate(MEETING, "medieval weapon exhibition gallery") is False

    @pytest.mark.xfail(reason=(
        "ИЗВЕСТНЫЙ пробел калибровки, найден ИМЕННО ЭТИМ тестом (не гипотеза): "
        "запрос 'sword museum display case' — дословно та формулировка из "
        "документированного бага (см. комментарий у RISKY_GENERIC_TERMS) — "
        "у stainedglass.jpg даёт margin=0.048, ВЫШЕ RISKY_QUERY_MARGIN=0.03, "
        "кандидат формально проходит гейт, хотя меча на кадре нет вообще. "
        "Оригинальная калибровка порога держалась на 5 плохих примерах "
        "(margin от -0.081 до 0.013) — это НЕЗАВИСИМОЕ реальное фото "
        "показывает, что запас 0.03 не покрывает всю дисперсию реальных "
        "плохих кандидатов при этой точной формулировке запроса. Умышленно "
        "НЕ исправлено автоматическим поднятием RISKY_QUERY_MARGIN здесь — "
        "это изменило бы продовую логику отбора без отдельной калибровки на "
        "большем наборе примеров, отдельная задача от golden-теста. xfail "
        "strict=True: если порог когда-нибудь перекалибруют и это начнёт "
        "проходить — тест сломается на XPASS, гэп не потеряется молча."),
        strict=True)
    def test_known_gap_exact_bug_query_still_accepts_stainedglass(self):
        assert ps.is_relevant_candidate(STAINEDGLASS, "sword museum display case") is False


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
