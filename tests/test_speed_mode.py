"""Реальный найденный вживую запрос пользователя (31.08, полный рендер
videos/01_ves-mecha): SigLIP2-so400m ~2.5-3с/кандидат x до 8 кандидатов
на слот даёт ~5-6 мин/клип — на ~165-блочном эпизоде это часы. Первые
FAST_MODE_START_INDEX клипов (хук + немного после — самый важный по
удержанию участок) держат полный пул, дальше пул сознательно уже —
честный компромисс скорость/точность, явно проговорённый с пользователем,
не тихий даунгрейд."""
import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]
import pipeline_smart as ps   # noqa: E402


def test_director_min_pool_full_before_threshold():
    assert ps._director_min_pool_for(0) == ps.DIRECTOR_MIN_POOL
    assert ps._director_min_pool_for(ps.FAST_MODE_START_INDEX - 1) == ps.DIRECTOR_MIN_POOL


def test_director_min_pool_reduced_from_threshold():
    assert ps._director_min_pool_for(ps.FAST_MODE_START_INDEX) == ps.FAST_DIRECTOR_MIN_POOL
    assert ps._director_min_pool_for(ps.FAST_MODE_START_INDEX + 50) == ps.FAST_DIRECTOR_MIN_POOL


def test_photo_dedup_max_tries_full_before_threshold():
    assert ps._photo_dedup_max_tries_for(0) == ps.PHOTO_DEDUP_MAX_TRIES
    assert ps._photo_dedup_max_tries_for(ps.FAST_MODE_START_INDEX - 1) == ps.PHOTO_DEDUP_MAX_TRIES


def test_photo_dedup_max_tries_reduced_from_threshold():
    assert ps._photo_dedup_max_tries_for(ps.FAST_MODE_START_INDEX) == ps.FAST_PHOTO_DEDUP_MAX_TRIES


def test_video_relevance_max_tries_full_before_threshold():
    assert ps._video_relevance_max_tries_for(0) == ps.VIDEO_RELEVANCE_MAX_TRIES
    assert (ps._video_relevance_max_tries_hard_cap_for(0)
            == ps.VIDEO_RELEVANCE_MAX_TRIES_HARD_CAP)


def test_video_relevance_max_tries_reduced_from_threshold():
    assert ps._video_relevance_max_tries_for(ps.FAST_MODE_START_INDEX) == ps.FAST_VIDEO_RELEVANCE_MAX_TRIES
    assert (ps._video_relevance_max_tries_hard_cap_for(ps.FAST_MODE_START_INDEX)
            == ps.FAST_VIDEO_RELEVANCE_MAX_TRIES_HARD_CAP)


def test_fast_mode_values_are_smaller_than_full_quality():
    # Быстрый режим обязан быть ДЕЙСТВИТЕЛЬНО быстрее, не равным/большим —
    # иначе весь смысл переключения теряется.
    assert ps.FAST_DIRECTOR_MIN_POOL < ps.DIRECTOR_MIN_POOL
    assert ps.FAST_PHOTO_DEDUP_MAX_TRIES < ps.PHOTO_DEDUP_MAX_TRIES
    assert ps.FAST_VIDEO_RELEVANCE_MAX_TRIES <= ps.VIDEO_RELEVANCE_MAX_TRIES
    assert ps.FAST_VIDEO_RELEVANCE_MAX_TRIES_HARD_CAP < ps.VIDEO_RELEVANCE_MAX_TRIES_HARD_CAP
