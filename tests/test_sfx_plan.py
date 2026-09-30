# -*- coding: utf-8 -*-
"""Планировщик расстановки звуковых эффектов.

ЧТО ИМЕННО ЗДЕСЬ ЗАЩИЩЕНО. Замер реального эпизода 02_ne-mechom (27 минут):
четыре звуковых события за весь ролик — 2 акцента кульминации и щелчки на
2 плашках из 11. Девятнадцать границ глав проходили в полной тишине.
Правка добавляет звук туда, где его не было, и главный риск такой правки
ровно один: эффект, поставленный поверх слова. Это самая узнаваемая ошибка
любителя, и она не ловится ни одним автоматическим отчётом — только ушами
на готовом ролике, то есть уже поздно.

Поэтому центральный инвариант всех тестов ниже — ЗВУК ПЕРЕХОДА ЖИВЁТ
ТОЛЬКО В РЕАЛЬНОЙ ТИШИНЕ: он заканчивается до первого слова новой главы и
начинается после последнего слова предыдущей. Не помещается — не ставится
и объясняет причину. Проверяется не на одном удобном примере, а перебором
по сетке реальных длин пауз, которые оставляет fix_pauses.py (0.42..1.35с).
"""
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import sfx_plan  # noqa: E402

SHORT = ("chapter_turn_short.flac", 0.38)
LONG = ("chapter_turn_long.flac", 0.90)
VARIANTS = (SHORT, LONG)


def make_episode(sections, speech=3.0, gap=0.8):
    """Ровный эпизод: каждый блок звучит speech секунд, между блоками gap."""
    blocks = [{"section": s, "words": 10} for s in sections]
    sub_starts, real_weights, t = [], [], 0.0
    for _ in sections:
        sub_starts.append(t)
        real_weights.append(speech)
        t += speech + gap
    return blocks, sub_starts, real_weights, t


def test_boundaries_skip_the_first_block():
    """Нулевой блок не граница — переход из ниоткуда озвучивать нечего."""
    blocks = [{"section": s} for s in ("HOOK", "HOOK", "BLOCK 1", "BLOCK 1", "FINAL")]
    assert sfx_plan.chapter_boundaries(blocks) == [2, 4]


def test_gap_is_measured_from_end_of_speech_not_end_of_clip():
    """Тишина считается от конца РЕЧИ, а не от конца кадра.

    Длительность кадра включает саму паузу — считать от неё значит получить
    тишину нулевой длины на каждой границе и не поставить ни одного звука.
    """
    _, starts, weights, _ = make_episode(["HOOK", "BLOCK 1"], speech=3.0, gap=0.8)
    gap = sfx_plan.speech_gap_before(1, starts, weights)
    assert gap == pytest.approx((3.0, 3.8))


def test_no_alignment_means_no_chapter_sound_at_all():
    """Без посимвольной разметки положение тишины неизвестно — честный отказ.

    Угадывать здесь нельзя: промах ставит свист поверх первого слова главы.
    """
    blocks, starts, _, total = make_episode(["HOOK", "BLOCK 1", "BLOCK 2"])
    accepted, dropped = sfx_plan.plan_sfx_cues(blocks, starts, None, total,
                                               chapter_variants=VARIANTS)
    assert accepted == []
    assert {d["reason"] for d in dropped} == {"no_alignment"}


@pytest.mark.parametrize("gap", [0.42, 0.5, 0.6, 0.75, 0.9, 1.1, 1.35, 2.0])
def test_chapter_sound_never_overlaps_speech(gap):
    """ГЛАВНЫЙ инвариант, по всей сетке реальных длин пауз fix_pauses.py."""
    blocks, starts, weights, total = make_episode(
        ["HOOK", "BLOCK 1", "BLOCK 2", "FINAL"], speech=4.0, gap=gap)
    accepted, _ = sfx_plan.plan_sfx_cues(blocks, starts, weights, total,
                                         chapter_variants=VARIANTS)
    for cue in accepted:
        i = cue["block"]
        speech_end_prev = starts[i - 1] + weights[i - 1]
        assert cue["time"] >= speech_end_prev - 1e-9, "эффект начался поверх предыдущего слова"
        assert cue["time"] + cue["asset_dur"] <= starts[i] + 1e-9, "эффект заехал на новую главу"


def test_longest_fitting_variant_wins():
    """Длина эффекта подбирается под реальную паузу, а не фиксирована."""
    assert sfx_plan.pick_variant(VARIANTS, 2.0) == LONG
    assert sfx_plan.pick_variant(VARIANTS, 0.5) == SHORT
    assert sfx_plan.pick_variant(VARIANTS, 0.2) is None


def test_short_gap_drops_the_cue_with_a_named_reason():
    """Не поместился — тишина и записанная причина, а не 'поставим тише'."""
    blocks, starts, weights, total = make_episode(["HOOK", "BLOCK 1"], speech=4.0, gap=0.2)
    accepted, dropped = sfx_plan.plan_sfx_cues(blocks, starts, weights, total,
                                               chapter_variants=VARIANTS)
    assert accepted == []
    assert dropped[0]["reason"] == "gap_too_short"
    assert dropped[0]["gap_sec"] == pytest.approx(0.2)


def test_long_pause_gets_the_long_variant_short_pause_the_short_one():
    blocks, starts, weights, total = make_episode(["HOOK", "BLOCK 1"], speech=4.0, gap=1.3)
    accepted, _ = sfx_plan.plan_sfx_cues(blocks, starts, weights, total,
                                         chapter_variants=VARIANTS)
    assert accepted[0]["asset"] == LONG[0]
    blocks, starts, weights, total = make_episode(["HOOK", "BLOCK 1"], speech=4.0, gap=0.5)
    accepted, _ = sfx_plan.plan_sfx_cues(blocks, starts, weights, total,
                                         chapter_variants=VARIANTS)
    assert accepted[0]["asset"] == SHORT[0]


def test_climax_window_is_reserved_for_the_reveal():
    """В окне разоблачения не звучит ничего постороннего.

    Музыка там уже проваливается ради одного момента; тик плашки в этом же
    окне разбавляет ровно то, ради чего провал и сделан.
    """
    blocks, starts, weights, total = make_episode(["HOOK", "BLOCK 1"], speech=4.0, gap=1.0)
    plate = [{"time": 6.0, "block": 1, "asset": "plate_tick.flac", "gain_db": -16.0}]
    accepted, dropped = sfx_plan.plan_sfx_cues(
        blocks, starts, weights, total, chapter_variants=VARIANTS,
        plate_cues=plate, reserved_windows=[(5.0, 7.5)])
    assert all(c["kind"] != "plate" for c in accepted)
    assert [d["reason"] for d in dropped if d["kind"] == "plate"] == ["climax_window"]


def test_two_effects_never_land_on_top_of_each_other():
    """Ближе минимального интервала — на слух один сдвоенный удар, брак."""
    blocks, starts, weights, total = make_episode(["HOOK", "BLOCK 1"], speech=4.0, gap=1.0)
    chapter_t = starts[1] - sfx_plan.CHAPTER_HEADROOM_SEC - LONG[1]
    plate = [{"time": chapter_t + 0.3, "block": 1, "asset": "t.flac", "gain_db": -16.0}]
    accepted, dropped = sfx_plan.plan_sfx_cues(
        blocks, starts, weights, total, chapter_variants=VARIANTS, plate_cues=plate)
    assert [c["kind"] for c in accepted] == ["chapter"], "приоритет у перехода, не у тика"
    assert [d["reason"] for d in dropped] == ["too_close"]


def test_density_cap_limits_a_densely_marked_stretch():
    """Потолок плотности — против «игрового автомата» на густом участке."""
    blocks, starts, weights, total = make_episode(["HOOK"], speech=1.0, gap=0.0)
    plate = [{"time": 3.0 * k, "block": 0, "asset": "t.flac", "gain_db": -16.0}
             for k in range(12)]
    accepted, dropped = sfx_plan.plan_sfx_cues(blocks, starts, weights, 120.0,
                                               plate_cues=plate, max_per_min=4)
    assert len(accepted) < len(plate)
    assert "density_cap" in {d["reason"] for d in dropped}


def test_plan_is_deterministic():
    """Тот же вход — тот же план: иначе два прогона дают разный ролик."""
    blocks, starts, weights, total = make_episode(
        ["HOOK", "BLOCK 1", "BLOCK 2", "FINAL"], speech=5.0, gap=1.0)
    plate = [{"time": 7.0, "block": 1, "asset": "t.flac", "gain_db": -16.0}]
    a1, d1 = sfx_plan.plan_sfx_cues(blocks, starts, weights, total,
                                    chapter_variants=VARIANTS, plate_cues=plate)
    a2, d2 = sfx_plan.plan_sfx_cues(blocks, starts, weights, total,
                                    chapter_variants=VARIANTS, plate_cues=plate)
    assert a1 == a2 and d1 == d2


def test_every_boundary_is_accounted_for():
    """Ни одна граница не теряется молча: либо звук, либо причина отказа."""
    blocks, starts, weights, total = make_episode(
        ["HOOK", "BLOCK 1", "BLOCK 2", "BLOCK 3", "FINAL"], speech=4.0, gap=0.3)
    accepted, dropped = sfx_plan.plan_sfx_cues(blocks, starts, weights, total,
                                               chapter_variants=VARIANTS)
    seen = {c["block"] for c in accepted} | {d["block"] for d in dropped if d["kind"] == "chapter"}
    assert seen == set(sfx_plan.chapter_boundaries(blocks))
