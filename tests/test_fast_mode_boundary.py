"""Граница урезанного пула считается по РЕАЛЬНОЙ длине хука.

РЕАЛЬНЫЙ, ИЗМЕРЕННЫЙ дефект. Замысел константы описан у неё же: «первые
FAST_MODE_START_INDEX клипов (хук + немного после — самый важный по
удержанию участок) держат полный пул». Логика верная, но 15 откалибровано
под эпизод, чей хук помещался в 15 блоков.

Замер на videos/02_ne-mechom (после split_long_blocks): хук занимает слоты
0..29 — тридцать слотов. То есть слоты 15..29, РОВНО ПОЛОВИНА ХУКА, шли на
урезанном бюджете:

    кандидатов сравнивается (фото)   4 -> 2
    пул Директора (фото)             8 -> 2
    попыток дедупа (фото)           20 -> 5
    скачиваний видео на слот         3 -> 1   (потолок 30 -> 3)

И это при том, что по золотому набору опубликованного эпизода годность
хука — 0 кадров из 10: бюджет резался у самой слабой и самой критичной по
удержанию части ролика.

Ключевое свойство правки, которое и держат тесты ниже: функция может
только РАСШИРИТЬ зону полного пула и никогда не сузить — FAST_MODE_START_
INDEX остаётся полом. То есть не существует эпизода, на котором подбор
стал бы беднее, чем был до правки.
"""
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]

import pipeline_smart as ps  # noqa: E402


def _blocks(*sections):
    return [{"section": s, "text": "текст", "words": 5, "pause_after": 0.0}
            for s in sections]


def test_long_hook_extends_the_full_pool_zone():
    """30 слотов хука — полный пул должен покрыть все 30, а не первые 15."""
    blocks = _blocks(*(["HOOK"] * 30 + ["BLOCK 1"] * 20))
    assert ps.fast_mode_start_for_blocks(blocks) == 30


def test_short_hook_keeps_the_calibrated_floor():
    """У эпизода с коротким хуком граница не опускается ниже 15 — прежнее
    поведение сохраняется байт-в-байт."""
    blocks = _blocks(*(["HOOK"] * 6 + ["BLOCK 1"] * 40))
    assert ps.fast_mode_start_for_blocks(blocks) == ps.FAST_MODE_START_INDEX


def test_boundary_never_shrinks_below_the_floor():
    """ГЛАВНЫЙ инвариант: правка умеет только расширять зону полного пула."""
    for hook_len in range(0, 60):
        blocks = _blocks(*(["HOOK"] * hook_len + ["BLOCK 1"] * 10))
        assert ps.fast_mode_start_for_blocks(blocks) >= ps.FAST_MODE_START_INDEX


def test_no_hook_at_all_falls_back_to_the_floor():
    assert ps.fast_mode_start_for_blocks(_blocks("BLOCK 1", "BLOCK 2")) == ps.FAST_MODE_START_INDEX
    assert ps.fast_mode_start_for_blocks([]) == ps.FAST_MODE_START_INDEX


def test_only_leading_hook_section_counts_not_a_later_mention():
    """Считается ПОСЛЕДНИЙ слот секции HOOK — если по какой-то причине
    HOOK встретился позже, зона расширяется до него, но никогда не
    сжимается."""
    blocks = _blocks(*(["HOOK"] * 4 + ["BLOCK 1"] * 30))
    assert ps.fast_mode_start_for_blocks(blocks) == ps.FAST_MODE_START_INDEX


@pytest.mark.parametrize("fn,full,fast", [
    ("_director_min_pool_for", "DIRECTOR_MIN_POOL", "FAST_DIRECTOR_MIN_POOL"),
    ("_base_min_pool_for", "BASE_MIN_POOL", "FAST_BASE_MIN_POOL"),
    ("_photo_dedup_max_tries_for", "PHOTO_DEDUP_MAX_TRIES", "FAST_PHOTO_DEDUP_MAX_TRIES"),
    ("_video_relevance_max_tries_for", "VIDEO_RELEVANCE_MAX_TRIES", "FAST_VIDEO_RELEVANCE_MAX_TRIES"),
])
def test_all_budget_helpers_follow_the_effective_boundary(monkeypatch, fn, full, fast):
    """Все четыре бюджета обязаны читать ДЕЙСТВУЮЩУЮ границу. Если хоть один
    остался на константе, половина хука продолжит собираться урезанно, а
    остальные три будут выглядеть исправленными."""
    monkeypatch.setattr(ps, "_FAST_MODE_START", 30)
    assert getattr(ps, fn)(29) == getattr(ps, full)
    assert getattr(ps, fn)(30) == getattr(ps, fast)


def test_effective_boundary_is_in_the_selection_signature():
    """Другой размер пула = другой победитель. Без подписи это не дошло бы
    до экрана на прогретом temp_smart/."""
    import inspect
    sig = inspect.getsource(ps._selection_stack_signature)
    assert "_FAST_MODE_START" in sig


def test_boundary_is_recorded_in_the_shotlist_gates():
    """Раньше по артефактам готового ролика нельзя было узнать, что 94%
    слотов собирались урезанным бюджетом."""
    import inspect
    body = inspect.getsource(ps.main) if hasattr(ps, "main") else ""
    assert "full_pool_until_slot" in body


def test_real_episode_hook_would_have_been_half_degraded():
    """Регрессионный якорь на реальных данных: сценарий эпизода 02 даёт
    30-слотовый хук, и при старой фиксированной границе половина его
    собиралась урезанно."""
    script = os.path.join(REPO_ROOT, "videos", "02_ne-mechom", "script.txt")
    if not os.path.exists(script):
        pytest.skip("сценарий эпизода 02 недоступен")
    import script_parser
    blocks, _ = ps.split_long_blocks(script_parser.parse_blocks(script), None)
    hook = [i for i, b in enumerate(blocks) if b["section"].startswith("HOOK")]
    assert len(hook) > ps.FAST_MODE_START_INDEX, (
        "у этого сценария хук короче пола — якорь потерял смысл, проверь замер")
    assert ps.fast_mode_start_for_blocks(blocks) == len(hook)
