# -*- coding: utf-8 -*-
"""Заставка главы (chapter_card.py): пауза диктора, размытый первый кадр с
названием, удар. Утверждено владельцем по ручному демо 03.10.

Каждое поведение — отдельная проверка, и каждая падает без правки:
пауза главы в fix_pauses, рез в начало паузы (PHRASE LOCK), переход fade,
запрет стыка чанков перед заставкой, удар вместо взмаха, уровень удара,
CHAPTER_CARD=0 — прежнее поведение байт-в-байт.
"""
import csv
import json
import os
import subprocess
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]

import chapter_card as cc  # noqa: E402
import fix_pauses as fp  # noqa: E402
import sfx_plan  # noqa: E402


def _ps():
    pytest.importorskip("PIL")
    import pipeline_smart as ps
    return ps


SECTIONS = ["HOOK", "HOOK", "BLOCK 1: ПОЛЕ", "BLOCK 1: ПОЛЕ", "BLOCK 2: СЧЁТ", "FINAL"]


def _blocks():
    return [{"section": s, "text": f"фраза {i}", "words": 4} for i, s in enumerate(SECTIONS)]


# ---------------------------------------------------------------- модуль

def test_section_title_matches_pipeline():
    ps = _ps()
    for name in ("HOOK", "FINAL", "BLOCK 3: КАК ПРАВИЛЬНО СДАТЬСЯ", "BLOCK 12:  X ", "BLOCK 4"):
        assert cc.section_title(name) == ps.section_title(name)


def test_card_boundaries_skip_hook_and_final():
    assert cc.card_boundaries(_blocks()) == [2, 4]
    assert cc.card_sections(["HOOK", "BLOCK 1: А", "BLOCK 2: Б", "FINAL"]) == ["BLOCK 1: А", "BLOCK 2: Б"]


def test_flag_and_on_screen_text_both_required(monkeypatch):
    monkeypatch.setenv("CHAPTER_CARD", "1")
    monkeypatch.setenv("ON_SCREEN_TEXT", "1")
    assert cc.enabled()
    monkeypatch.setenv("ON_SCREEN_TEXT", "0")
    assert not cc.enabled()
    monkeypatch.setenv("ON_SCREEN_TEXT", "1")
    monkeypatch.setenv("CHAPTER_CARD", "0")
    assert not cc.enabled()


FONT = os.path.join(REPO_ROOT, "assets", "fonts", "Benzin-ExtraBold.ttf")


def test_short_title_one_line_at_64():
    size, lines, ok = cc.title_layout("ЧЕЛОВЕК, КОТОРЫЙ ВЁЛ СЧЁТ", FONT)
    assert (size, lines, ok) == (64, ["ЧЕЛОВЕК, КОТОРЫЙ ВЁЛ СЧЁТ"], True)


@pytest.mark.parametrize("title", ["ТКАЧИ, КОТОРЫМ НЕ НУЖНЫ БЫЛИ ДЕНЬГИ",
                                   "ГОРЦЫ, КОТОРЫЕ ЗАПРЕТИЛИ ПЛЕННЫХ",
                                   "ОЧЕНЬ ДЛИННОЕ НАЗВАНИЕ ГЛАВЫ КОТОРОЕ НИКАК НЕ ВЛЕЗАЕТ В ОДНУ СТРОКУ ЭКРАНА"])
def test_long_title_never_wider_than_limit(title):
    pytest.importorskip("PIL")
    size, lines, ok = cc.title_layout(title, FONT)
    assert ok and 1 <= len(lines) <= 2
    assert " ".join(lines) == title
    assert max(cc._text_width(x, size, FONT) for x in lines) <= cc.CARD_TITLE_MAX_WIDTH


def test_timing_follows_approved_demo():
    tm = cc.card_timing(lead=1.73, dur=9.8)
    assert tm["title_out"] == pytest.approx(1.73 + 0.4)
    assert tm["dissolve"] == pytest.approx(1.73 + 0.4 + 0.3)
    assert tm["card_end"] == pytest.approx(tm["dissolve"] + 0.5)
    assert tm["title_in"] == pytest.approx(0.25)


def test_timing_clamped_on_short_clip():
    tm = cc.card_timing(lead=1.7, dur=2.6)
    assert tm["card_end"] <= 2.6 - cc.CARD_MIN_SHARP_TAIL + 1e-9
    assert tm["dissolve"] >= 1.7 - 1e-9


def test_filter_has_title_and_line_but_no_chapter_label():
    fc = cc.card_filter("человек, который вёл счёт", 1.7, 9.0, FONT, FONT, lambda s: s)
    assert "ЧЕЛОВЕК, КОТОРЫЙ ВЁЛ СЧЁТ" in fc
    assert "ГЛАВА" not in fc
    assert "drawbox" in fc and "0xC8102E" in fc
    assert "boxblur=24:2" in fc and "brightness=-0.16:saturation=0.55" in fc
    assert "vignette=PI/4" in fc and "overlay=format=yuv420p10" in fc


def test_hit_gain_three_lu_below_voice_and_peak_capped():
    g, src = cc.hit_gain_db(voice_lufs=-16.0, voice_peak_dbfs=-1.0, hit_max_momentary=-10.6,
                            hit_peak_dbfs=-1.5)
    assert (g, src) == (pytest.approx(-8.4), "measured")
    g, src = cc.hit_gain_db(voice_lufs=-16.0, voice_peak_dbfs=-12.0, hit_max_momentary=-10.6,
                            hit_peak_dbfs=-1.5)
    assert g == pytest.approx(-10.5) and src == "measured_peak_capped"
    assert cc.hit_gain_db(None, -1, -10, -1) == (None, "not_measured")


def test_hit_asset_in_library_with_cc0_manifest():
    path = os.path.join(REPO_ROOT, "assets", "library", "sfx", "chapter_hit",
                        "freesound_555245_bright_cinematic_boom.flac")
    assert os.path.exists(path)
    man = json.load(open(os.path.join(REPO_ROOT, "assets", "library", "manifest.json"), encoding="utf-8"))
    item = man["items"][os.path.relpath(path, REPO_ROOT)]
    assert item["license"] == "cc0" and item["creator"] == "cookies+policy"
    assert item["landing"].endswith("/555245")


# ---------------------------------------------------------------- пауза (fix_pauses)

def _write_episode(d, offsets, align):
    os.makedirs(os.path.join(d, "media_plan", "alignment"), exist_ok=True)
    with open(os.path.join(d, "script.txt"), "w", encoding="utf-8") as f:
        f.write("=== HOOK ===\nОдин два три.\n=== BLOCK 1: ПОЛЕ ===\nЧетыре пять.\n=== FINAL ===\nШесть.\n")
    json.dump(offsets, open(os.path.join(d, "media_plan", "section_offsets.json"), "w"))
    for k, rows in enumerate(align):
        with open(os.path.join(d, "media_plan", "alignment", f"{k:02d}.csv"), "w", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["char", "start", "end"])
            w.writerows(rows)


def test_chapter_window_from_alignment(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAPTER_CARD", "1")
    _write_episode(str(tmp_path), {"HOOK": 0.0, "BLOCK 1: ПОЛЕ": 10.0, "FINAL": 20.0},
                   [[["о", 0.0, 0.1], ["!", 9.0, 9.2], [" ", 9.2, 9.6]],
                    [["Ч", 0.3, 0.4], [".", 8.0, 8.1]],
                    [["Ш", 0.1, 0.2]]])
    # перед FINAL заставки нет: только BLOCK 1
    assert fp.chapter_card_windows(str(tmp_path)) == [(9.2, 10.3, cc.CARD_PAUSE_SEC)]
    monkeypatch.setenv("CHAPTER_CARD", "0")
    assert fp.chapter_card_windows(str(tmp_path)) == []


def test_chapter_pause_is_extended_to_target_inside_kept_silence():
    # окно 9.2..10.3, подрезка вырезала 9.6..10.1 -> осталось 0.6с
    segs = [("copy", 0.0, 9.6), ("copy", 10.1, 30.0)]
    out, ins = fp.apply_chapter_pause_targets(list(segs), [(9.2, 10.3, cc.CARD_PAUSE_SEC)])
    assert len(ins) == 1
    pos, sec = ins[0]
    assert sec == pytest.approx(cc.CARD_PAUSE_SEC - 0.6)
    assert 9.2 < pos < 9.6           # в сохранённом куске, не в вырезанном
    assert any(x[0] == "silence" for x in out)


def test_long_enough_chapter_pause_left_alone():
    segs = [("copy", 0.0, 30.0)]
    out, ins = fp.apply_chapter_pause_targets(list(segs), [(9.0, 11.0, cc.CARD_PAUSE_SEC)])
    assert ins == [] and out == segs


def test_save_cuts_without_cards_writes_no_new_keys(tmp_path):
    src = tmp_path / "a.mp3"
    src.write_bytes(b"x")
    fp.save_cuts(str(tmp_path), [(1.0, 3.0)], str(src), str(tmp_path / "none.flac"))
    data = json.load(open(tmp_path / "media_plan" / "pause_cuts.json"))
    assert "chapter_pauses" not in data and "chapter_inserts" not in data
    fp.save_cuts(str(tmp_path), [(1.0, 3.0)], str(src), str(tmp_path / "none.flac"),
                 pause_inserts=[(2.0, 1.0)], chapter_windows=[(1.5, 2.5, 1.8)], chapter_inserts=[(2.0, 1.0)])
    data = json.load(open(tmp_path / "media_plan" / "pause_cuts.json"))
    assert data["chapter_inserts"] == [[2.0, 1.0]] and data["pause_inserts"] == [[2.0, 1.0]]


# ---------------------------------------------------------------- монтаж

def test_card_transition_is_fade_and_flag_free_plan_unchanged():
    ps = _ps()
    blocks = _blocks()
    secs = [b["section"] for b in blocks]
    before = ps.plan_transitions(secs, blocks)
    blocks[2]["chapter_card"] = {"title": "ПОЛЕ", "visual_start": 5.0, "speech_onset": 6.7}
    after = ps.plan_transitions(secs, blocks)
    assert after[1] == ("fade", ps.quantize_dur_to_frame(cc.CARD_FADE_IN_SEC))
    assert before[1] != after[1]
    del blocks[2]["chapter_card"]
    assert ps.plan_transitions(secs, blocks) == before


def test_no_chunk_seam_right_before_a_card():
    ps = _ps()
    secs = ["A"] * 30 + ["B"] * 30
    plain = ps._chunk_bounds(60, secs, 25)
    assert any(a == 30 for a, _b in plain)
    guarded = ps._chunk_bounds(60, secs, 25, no_split={30})
    assert not any(a == 30 for a, _b in guarded)
    assert sum(b - a for a, b in guarded) == 60


def test_card_clip_starts_after_previous_speech_end(monkeypatch):
    ps = _ps()
    monkeypatch.setenv("CHAPTER_CARD", "1")
    monkeypatch.setenv("ON_SCREEN_TEXT", "1")
    blocks = _blocks()
    onsets = [0.0, 3.0, 8.0, 12.0, 17.0, 22.0]
    monkeypatch.setattr(ps, "SPEECH_ENDS", [2.5, 6.0, 11.0, 15.0, 20.0, 25.0])
    marked = ps.mark_chapter_cards(blocks, onsets)
    assert marked == [2, 4]
    assert blocks[2]["chapter_card"]["visual_start"] == pytest.approx(6.0 + cc.CARD_LEAD_SEC)
    secs = [b["section"] for b in blocks]
    plan = ps.effective_transition_plan(ps.plan_transitions(secs, blocks), secs, blocks=blocks)
    durs = ps.phrase_locked_durations(ps.card_visual_onsets(blocks, onsets), 27.0, plan)
    seen = ps.first_visible_starts(blocks, durs)
    assert abs(seen[2] - (6.0 + cc.CARD_LEAD_SEC)) <= 0.5 / ps.FPS + 1e-9
    assert abs(seen[3] - 12.0) <= 0.5 / ps.FPS + 1e-9
    assert sum(durs) - sum(d for _t, d in plan) == pytest.approx(27.0, abs=0.5 / ps.FPS)


def test_card_skipped_when_pause_is_short(monkeypatch):
    ps = _ps()
    monkeypatch.setenv("CHAPTER_CARD", "1")
    blocks = _blocks()
    monkeypatch.setattr(ps, "SPEECH_ENDS", [2.5, 7.6, 11.0, 16.7, 20.0, 25.0])
    assert ps.mark_chapter_cards(blocks, [0.0, 3.0, 8.0, 12.0, 17.0, 22.0]) == []
    assert not any(b.get("chapter_card") for b in blocks)


def test_flag_off_leaves_durations_byte_identical(monkeypatch):
    ps = _ps()
    monkeypatch.setenv("CHAPTER_CARD", "0")
    blocks = _blocks()
    onsets = [0.0, 3.0, 8.0, 12.0, 17.0, 22.0]
    monkeypatch.setattr(ps, "SPEECH_ENDS", [2.5, 6.0, 11.0, 15.0, 20.0, 25.0])
    secs = [b["section"] for b in blocks]
    ref = ps.phrase_locked_durations(onsets, 27.0, ps.effective_transition_plan(
        ps.plan_transitions(secs, blocks), secs))
    assert ps.mark_chapter_cards(blocks, onsets) == []
    assert ps.card_visual_onsets(blocks, onsets) == onsets
    got = ps.phrase_locked_durations(ps.card_visual_onsets(blocks, onsets), 27.0,
                                     ps.effective_transition_plan(ps.plan_transitions(secs, blocks),
                                                                  secs, blocks=blocks))
    assert got == ref


def test_chapter_inserts_do_not_change_slot_split_window(monkeypatch):
    ps = _ps()
    monkeypatch.setattr(ps, "load_chapter_inserts", lambda: [(100.0, 1.3)])
    monkeypatch.setattr(ps, "load_pause_cuts", lambda: [])
    monkeypatch.setattr(ps, "load_pause_inserts", lambda: [(100.0, 1.3)])
    assert ps.chapter_insert_sec_between(99.0, 102.0) == pytest.approx(1.3)
    assert ps.chapter_insert_sec_between(101.0, 102.0) == 0.0


def test_chapter_card_signature_only_in_card_clips():
    ps = _ps()
    sig = ps.chapter_card_recipe_signature()
    assert sig.startswith("card:") and len(sig) > 8
    import inspect
    assert "chapter_card" not in inspect.getsource(ps.render_recipe_signature)


@pytest.mark.skipif(subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0,
                    reason="нет ffmpeg")
def test_apply_card_renders_full_length_clip(tmp_path):
    ps = _ps()
    src = str(tmp_path / "src.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=s=1920x1080:r=24:d=4", "-c:v", "libx264", "-preset", "ultrafast"]
                   + ps.CLIP_PIX_ARGS + ps.COLOR_META_ARGS + [src], check=True)
    out = str(tmp_path / "out.mp4")
    assert ps.apply_chapter_card(src, out, 4.0, {"title": "ПОЛЕ", "lead": 1.0})
    ok, _reason, _d = ps.verify_clip(out, 4.0)
    assert ok
    # заставка темнее и размытее исходника, к концу клип снова чёткий
    import numpy as np
    from PIL import Image

    def frame(path, t):
        p = str(tmp_path / f"f{t}.png")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(t), "-i", path,
                        "-frames:v", "1", p], check=True)
        return np.asarray(Image.open(p).convert("L"), dtype=float)

    assert frame(out, 0.8).mean() < frame(src, 0.8).mean() - 10
    assert np.abs(frame(out, 3.7) - frame(src, 3.7)).mean() < 3


# ---------------------------------------------------------------- звук

def test_hit_replaces_whoosh_and_may_run_under_speech():
    blocks = [{"section": s, "words": 10} for s in ("HOOK", "BLOCK 1: А", "FINAL")]
    sub_starts, rw = [0.0, 5.0, 10.0], [3.0, 3.0, 3.0]
    variants = (("whoosh.flac", 0.5),)
    acc, _ = sfx_plan.plan_sfx_cues(blocks, sub_starts, rw, 20.0, chapter_variants=variants)
    assert [c["asset"] for c in acc if c["kind"] == "chapter"] == ["whoosh.flac", "whoosh.flac"]
    hits = {1: {"time": 3.07, "asset": "hit.flac", "asset_dur": 6.0}}
    acc, _ = sfx_plan.plan_sfx_cues(blocks, sub_starts, rw, 20.0, chapter_variants=variants,
                                    chapter_hits=hits)
    ch = [c for c in acc if c["kind"] == "chapter"]
    assert ch[0]["asset"] == "hit.flac" and ch[0]["time"] == pytest.approx(3.07) and ch[0]["hit"]
    # хвост 6с уходит под голос новой главы (онсет 5.0) — осознанно разрешено
    assert ch[0]["time"] + ch[0]["asset_dur"] > 5.0
    # граница без заставки (FINAL) — прежний взмах
    assert ch[1]["asset"] == "whoosh.flac"


def test_chapter_hit_cues_follow_card_marks(monkeypatch):
    ps = _ps()
    blocks = _blocks()
    assert ps.chapter_hit_cues(blocks) == {}
    blocks[2]["chapter_card"] = {"title": "ПОЛЕ", "visual_start": 6.07, "speech_onset": 8.0}
    path = os.path.join(REPO_ROOT, "assets", "library", "sfx", "chapter_hit",
                        "freesound_555245_bright_cinematic_boom.flac")
    monkeypatch.setattr(ps, "library_sounds", lambda kind, name: [path] if name == "chapter_hit" else [])
    monkeypatch.setattr(ps, "media_duration_or_none", lambda p: 6.048)
    cues = ps.chapter_hit_cues(blocks)
    assert cues == {2: {"time": 6.07, "asset": path, "asset_dur": 6.048,
                        "hit_gap_lu": cc.CARD_HIT_GAP_LU}}


def test_verify_timing_expects_card_cut_at_card_start(tmp_path):
    import verify_timing as vt
    os.makedirs(tmp_path / "media_plan")
    json.dump({"locked": True, "fps": 24, "blocks": [
        {"speech_onset_sec": 0.0}, {"speech_onset_sec": 5.0},
        {"speech_onset_sec": 10.0, "visual_target_sec": 8.3, "chapter_card": True}]},
        open(tmp_path / "media_plan" / "phrase_timeline.json", "w"))
    (tmp_path / "final.mp4").write_bytes(b"")
    seen = {}

    def fake_match(expected, detected):
        seen["expected"] = list(expected)
        return [], list(expected), []
    import pytest as _p
    mp = _p.MonkeyPatch()
    mp.setattr(vt, "detect_cuts", lambda *a, **k: [])
    mp.setattr(vt, "match_cuts", fake_match)
    try:
        vt.verify(str(tmp_path))
    except Exception:
        pass
    finally:
        mp.undo()
    assert seen["expected"] == [5.0, 8.3]
