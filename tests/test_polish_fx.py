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
    assert blocks[0]["letterbox"] == "full" and blocks[0]["punch_times"] == [12.0]
    assert "letterbox" not in blocks[1] and "punch_times" not in blocks[1]


def test_riser_ends_on_hit():
    hits = [{"time": 30.0, "gain_db": -3.0, "hit": True}]
    cues = polish_fx.riser_cues(hits, ["r.flac"], [2.5], lambda p: (-1.0, "measured"))
    assert cues[0]["time"] == pytest.approx(27.5)
    assert cues[0]["gain_db"] == pytest.approx(-1.0 + polish_fx.RISER_GAIN_OFFSET_DB)


def _probe_frame(path, t, out):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(t), "-i", path, "-frames:v", "1", out],
                   check=True)
    from PIL import Image
    return Image.open(out).convert("L")


@pytest.mark.skipif(subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0,
                    reason="нет ffmpeg")
def test_letterbox_opens_and_punch_zooms(tmp_path):
    src = str(tmp_path / "src.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "color=white:s=1920x1080:r=24:d=3", "-pix_fmt", "yuv420p", src], check=True)
    g = polish_fx.letterbox_filter("0:v", "vlb", 3.0, "open")
    out = str(tmp_path / "lb.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-filter_complex", g, "-map", "[vlb]",
                    "-pix_fmt", "yuv420p", out], check=True)
    import numpy as np
    early = np.asarray(_probe_frame(out, 0.1, str(tmp_path / "a.png")))
    late = np.asarray(_probe_frame(out, 2.9, str(tmp_path / "b.png")))
    assert early[:100].mean() < 10 and early[-100:].mean() < 10      # полосы на месте
    assert late[:100].mean() > 240 and late[-100:].mean() > 240      # полосы ушли
    g = polish_fx.punch_filter("0:v", "vpn", [1.0])
    assert "eval=frame" in g and "crop=1920:1080" in g
