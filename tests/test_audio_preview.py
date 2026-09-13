# -*- coding: utf-8 -*-
"""Предпросмотр звука собирает ТОТ ЖЕ звук, что уйдёт в ролик.

Прямая причина существования: проверить звук на слух можно только на
реальном отрывке реального эпизода — с настоящим голосом, настоящими
уровнями и настоящим мастерингом. Отдельная демонстрация («вот так примерно
звучит атмосфера») проверяет не то, что уйдёт в ролик, то есть не проверяет
ничего. Поэтому главный инвариант здесь — ОДНА цепочка на предпросмотр и на
рендер, без копии.
"""
import inspect
import json
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]

import pipeline_smart as ps  # noqa: E402


def test_preview_and_render_share_one_audio_chain():
    """Копия цепочки в предпросмотре означала бы, что человек слушает не то,
    что уйдёт в ролик."""
    assert "build_episode_audio_layers(" in inspect.getsource(ps.write_audio_preview)
    assert "build_episode_audio_layers(" in inspect.getsource(ps.main)
    chain = inspect.getsource(ps.build_episode_audio_layers)
    for step in ("build_music_mix(", "run_ambience(", "add_typewriter_clicks(",
                 "add_reveal_sfx(", "run_sfx_director("):
        assert step in chain, f"{step} выпал из общей цепочки"


def test_preview_uses_the_same_mastering():
    """Уровень — половина вопроса «как звучит». Своя нормализация сделала бы
    отрывок не похожим на то же место в готовом ролике."""
    src = inspect.getsource(ps.write_audio_preview)
    assert "measure_loudnorm_stats(" in src and "build_master_af(" in src


def test_preview_masters_the_whole_episode_then_cuts():
    """loudnorm считает громкость по тому, что ему дали: у сорокасекундного
    куска она своя."""
    src = inspect.getsource(ps.write_audio_preview)
    assert src.index("build_master_af(") < src.index('"-ss"')


def test_preview_runs_before_any_media_is_fetched_or_rendered():
    """Звуку не нужен ни один кадр. Предпросмотр стоит там же, где выходит
    --plan-only, — до подбора стока и до рендера клипов."""
    src = inspect.getsource(ps.main)
    assert "write_audio_preview(" in src
    assert src.index("if PLAN_ONLY:") < src.index("write_audio_preview(")
    assert src.index("write_audio_preview(") < src.index("photo = pexels_photo(")
    assert src.index("write_audio_preview(") < src.index("run_ffmpeg_with_retry")


def test_stat_cues_match_the_variant_rule():
    """Каждая пятая плашка — машинка со своими щелчками, остальные — тик."""
    blocks, durs, starts, base = [], [], [], []
    for i in range(7):
        blocks.append({"section": "BLOCK 1", "text": "x", "words": 10,
                       "stat": f"{i} КГ", "stat_word_pos": 5})
        durs.append(6.0); starts.append(i * 10.0); base.append(6.0)
    clicks, plates = ps.plan_stat_sound_cues(blocks, durs, starts, base)
    assert len(plates) == 6, "тик обязан быть у всех, кроме пятой"
    assert {p["block"] for p in plates} == {0, 1, 2, 3, 5, 6}
    assert clicks, "у пятой плашки — посимвольные щелчки"
    assert all(40.0 <= c <= 50.0 for c in clicks), "щелчки внутри своего блока"


def test_blocks_without_a_stat_make_no_sound():
    blocks = [{"section": "HOOK", "text": "x", "words": 10, "stat": None,
               "stat_word_pos": None}]
    clicks, plates = ps.plan_stat_sound_cues(blocks, [5.0], [0.0], [5.0])
    assert clicks == [] and plates == []


def test_section_bounds_are_on_the_audio_scale():
    blocks = [{"section": "HOOK"}, {"section": "BLOCK 1"}, {"section": "FINAL"}]
    hook_end, final_start = ps.section_audio_bounds(blocks, [0.0, 12.0, 40.0], 50.0)
    assert hook_end == 12.0 and final_start == 40.0


def test_section_bounds_survive_an_episode_without_final():
    blocks = [{"section": "HOOK"}, {"section": "BLOCK 1"}]
    hook_end, final_start = ps.section_audio_bounds(blocks, [0.0, 12.0], 50.0)
    assert hook_end == 12.0 and final_start is None


def test_window_lands_where_the_sounds_are(tmp_path):
    """Случайный кусок ролика запросто не содержит ни границы главы, ни
    плашки — то есть ровно то, что надо услышать, в него не попадёт."""
    os.makedirs(tmp_path / "media_plan")
    with open(tmp_path / "media_plan" / "sfx_plan.json", "w", encoding="utf-8") as f:
        json.dump({"accepted": [{"time": 300.0}, {"time": 305.0}, {"time": 312.0}]}, f)
    start = ps._preview_window(str(tmp_path), 900.0, 30.0)
    assert 282.0 <= start <= 300.0, f"окно {start} промахнулось мимо событий"


def test_window_falls_back_without_a_plan(tmp_path):
    start = ps._preview_window(str(tmp_path), 900.0, 30.0)
    assert 0.0 <= start <= 870.0
