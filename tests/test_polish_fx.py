"""Монтаж второго канала (POLISH_FX): тег [punch], фильтры кадра, звук, флаг."""
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import polish_fx  # noqa: E402
import script_parser  # noqa: E402


def _parse(tmp_path, text):
    p = tmp_path / "script.txt"
    p.write_text(text, encoding="utf-8")
    return script_parser.parse_blocks(str(p))


def test_punch_tag_keeps_text_and_position(tmp_path):
    blocks = _parse(tmp_path, "=== HOOK ===\n[energetic] Раз два. [punch]Три четыре.[pause]Пять.\n")
    assert [b["text"] for b in blocks] == ["Раз два. Три четыре.", "Пять."]
    assert polish_fx.punch_word_positions(blocks[0]) == [2]
    assert polish_fx.punch_word_positions(blocks[1]) == []


def test_punch_tag_never_reaches_tts():
    assert "punch" not in script_parser.strip_pipeline_only_tags("и он [punch]казнил")


def test_punch_is_not_an_ambience_cue():
    import sound_director
    blocks = [{"sound_cues": [{"type": "punch", "name": "hit", "word_pos": 1},
                              {"type": "amb", "name": "rain_mud", "word_pos": 0}]}]
    assert sound_director.inline_cues(blocks) == [(0, "amb", "rain_mud")]


def test_mark_blocks_is_noop_without_flag(monkeypatch):
    monkeypatch.setenv("POLISH_FX", "0")
    blocks = [{"section": "HOOK", "words": 4, "letterbox": "full", "punch_times": [1.0],
               "sound_cues": [{"type": "punch", "word_pos": 2}]}]
    polish_fx.mark_blocks(blocks, [0.0], [4.0])
    assert "letterbox" not in blocks[0] and "punch_times" not in blocks[0]


def test_mark_blocks_with_flag(monkeypatch):
    monkeypatch.setenv("POLISH_FX", "1")
    blocks = [{"section": "HOOK", "words": 4, "sound_cues": [{"type": "punch", "word_pos": 2}]},
              {"section": "BLOCK 1: X", "words": 2, "sound_cues": []}]
    polish_fx.mark_blocks(blocks, [10.0, 14.0], [4.0, 2.0])
    assert blocks[0]["punch_times"] == [12.0] and "letterbox" not in blocks[0]
    assert "punch_times" not in blocks[1]


def test_riser_ends_on_hit():
    hits = [{"time": 30.0, "gain_db": -3.0, "hit": True}]
    cues = polish_fx.riser_cues(hits, ["r.flac"], [2.5], lambda p: (-1.0, "measured"))
    assert cues[0]["time"] == pytest.approx(27.5)
    assert cues[0]["gain_db"] == pytest.approx(-1.0 + polish_fx.RISER_GAIN_OFFSET_DB)


def test_punch_filter_scales_and_crops_back():
    g = polish_fx.punch_filter("0:v", "vpn", [1.0])
    assert "eval=frame" in g and "crop=1920:1080" in g


def test_hook_cut_swooshes_only_on_phrase_changes():
    blocks = [{"section": "HOOK"}, {"section": "HOOK"}, {"section": "HOOK", "is_subcut": True},
              {"section": "HOOK"}, {"section": "BLOCK 1: X"}]
    cues = polish_fx.hook_cut_swooshes(blocks, [0.0, 3.0, 5.0, 7.0, 9.0], ["a", "b"], [0.36, 0.12],
                                       lambda p: (-20.0, "measured"))
    assert [c["block"] for c in cues] == [1, 3]                      # без первой, без под-кадра, без тела
    assert cues[0]["time"] == pytest.approx(3.0 - polish_fx.HOOK_CUT_PEAK_BEFORE_SEC - 0.36)
    assert cues[1]["asset"] == "b"                                    # файлы чередуются


def test_local_media_for_partial_slots_is_not_ignored(tmp_path, monkeypatch):
    """Папка media/ с кадрами только под слоты 2 и 4 раньше игнорировалась
    целиком: use_local проверял слот 0."""
    import inspect
    import pipeline_smart
    src = inspect.getsource(pipeline_smart.main)
    assert "bool(_local_photos())" in src and "bool(local_photo(0))" not in src
