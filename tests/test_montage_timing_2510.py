"""Монтаж и тайминг, 02.10 (docs/fixes_2510/montage.md).

Каждая проверка падает на коде ДО правки (контрольный прогон — в отчёте).
Правильное поведение закреплено отдельно: блоки и эпизоды, где всё уже было
верно, обязаны дать байт-в-байт прежний результат.
"""
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]

import pipeline_smart as ps  # noqa: E402
import verify_timing as vt  # noqa: E402
import fix_pauses  # noqa: E402
import lumean_tts  # noqa: E402
import script_parser  # noqa: E402

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


# ---------------------------------------------------------------- A. хук

HOOK_PHRASE = ("Звучит как выдумка, но в тысяча сто девятнадцатом году во Франции сошлись "
               "около девятисот рыцарей, и, по словам монаха, который записал эту битву, "
               "погибло из них всего трое. Трое на девятьсот человек в железе и с оружием "
               "в руках, и таких битв тогда было много, настолько много, что один англичанин "
               "на этом по-настоящему разбогател и даже держал при себе отдельного человека, "
               "который вёл счёт, скольких он взял в плен.")


def _bad(words, cuts):
    return [c for c in cuts if not ps.cut_is_clean(words, c)]


def test_hook_punctuation_pass_is_not_skipped_at_radius_zero():
    """Проход «сначала знак препинания» принимал слово-цель без знака (проверка
    стояла под `radius and ...`). 03_plen: «Трое на девятьсот | человек»,
    «который записал | эту битву». Теперь резов посреди словосочетания меньше."""
    words = HOOK_PHRASE.split()
    cuts = ps._hook_split_points(words, 28.56, 0.0, first_slot=False)
    legacy = ps._hook_split_points_search(words, 28.56, 0.0, False, None,
                                          ps._HOOK_SPLIT_LEGACY_PASSES)
    assert len(_bad(words, cuts)) < len(_bad(words, legacy))
    assert len(cuts) == len(legacy), "число кусков то же — меняется только место реза"


def test_hook_block_with_clean_cuts_is_unchanged():
    words = [f"слово{i}," for i in range(24)]   # любой рез — после запятой
    legacy = ps._hook_split_points_search(words, 9.0, 0.0, False, None,
                                          ps._HOOK_SPLIT_LEGACY_PASSES)
    assert legacy and not _bad(words, legacy)
    assert ps._hook_split_points(words, 9.0, 0.0) == legacy


# ---------------------------------------------------------------- B. тело

@pytest.mark.parametrize("text,k,clean", [
    ("знатных французов под охраной оказалось", 3, False),       # после предлога
    ("Двадцать пятое октября тысяча четыреста года", 4, False),  # внутри числа
    ("пятнадцатого года, север Франции", 2, True),               # после запятой
    ("рыцарей, и, по словам монаха", 2, False),                  # «и,» — обрыв
    ("сдаться конкретному человеку и отдавали ему", 3, True),    # перед союзом
    ("велел своим рыцарям привязать своих коней", 5, False),     # после «своих»
    ("Он сделал это. Потом ушёл", 3, True),                      # «это.» — конец фразы
])
def test_cut_rank_vocabulary(text, k, clean):
    assert ps.cut_is_clean(text.split(), k) is clean


def test_body_fallback_avoids_preposition_cut():
    words = ("Англичане набрали столько пленных, что знатных французов под охраной оказалось "
             "чуть ли не больше, чем самих охранников.").split()
    old = [len(words) // 2]
    assert not ps.cut_is_clean(words, old[0])
    new, real = ps._clause_fallback_cut(words, old, len(words) // 2, 8.03, 3.0)
    assert ps.cut_rank(words, new[0]) < ps.cut_rank(words, old[0])
    assert real is False


def test_body_fallback_keeps_clean_cut():
    words = "Раз два три четыре, пять шесть семь восемь девять десять".split()
    assert ps._clause_fallback_cut(words, [4], 5, 10.0, 3.0) == ([4], False)


def test_body_fallback_uses_real_word_times_and_split_keeps_it(monkeypatch):
    """По доле слов вторая половина «…рыцарем, | так король понимал…» — 2.98 с
    (ниже пола 3.0), по реальной речи — больше. Оценочная склейка не должна
    склеивать кусок, проверенный реальным временем."""
    text = ("В Англии четырнадцатого века человек с доходом сорок фунтов в год уже обязан "
            "был стать рыцарем, так король понимал, где заканчивается простой землевладелец.")
    words = text.split()
    n = len(words)
    k = words.index("рыцарем,") + 1
    times = [i * 0.40 for i in range(n)] + [n * 0.40 + 1.0]   # хвост с паузой
    times[k:] = [t + 0.0 for t in times[k:]]
    b = {"text": text, "words": n, "section": "BLOCK 2: X", "pause_after": 0.8,
         "stat": None, "stat_word_pos": None, "is_climax": False, "orig_index": 0}
    monkeypatch.setattr(ps, "_block_word_times", lambda blocks, wanted: {0: times})
    out, _w = ps.split_long_blocks([b], [9.78])
    assert len(out) == 2
    assert out[0]["text"].endswith("рыцарем,")


# ---------------------------------------------------------------- C. рез на онсете

def _xfade_offsets(durs, plan):
    """Ровно арифметика xfade_chain(): offset_i = cum - dur_i."""
    out, cum = [], durs[0]
    for i in range(1, len(durs)):
        d = plan[i - 1][1]
        out.append(max(0.0, cum - d))
        cum = cum + durs[i] - d
    return out


def test_first_frame_of_new_clip_lands_on_onset():
    fps = ps.FPS
    onsets = [0.0, 2.31, 5.02, 7.77, 10.4]
    plan = [("fade", 1 / fps), ("fade", 1 / fps), ("fade", 0.0), ("fade", 10 / fps)]
    durs = ps.phrase_locked_durations(onsets, 13.0, plan, fps=fps)
    for i, off in enumerate(_xfade_offsets(durs, plan), start=1):
        lag = (1 / fps) if plan[i - 1][1] > 0 else 0.0   # xfade: на offset ещё старый клип
        assert abs(off + lag - onsets[i]) <= 0.5 / fps + 1e-6, (i, off, onsets[i])
    assert sum(durs) - sum(d for _t, d in plan) == pytest.approx(13.0, abs=0.5 / fps)


@pytest.mark.skipif(not HAS_FFMPEG, reason="нужен ffmpeg")
def test_rendered_hard_cut_appears_on_onset(tmp_path):
    """Не модель, а пиксели: настоящая xfade-склейка, детектор verify_timing."""
    fps = ps.FPS
    colors = ["red", "0x00A000", "blue", "yellow"]
    onsets = [0.0, 1.5, 3.05, 4.6]
    total = 6.0
    blocks = [{"section": "BLOCK 1: X", "is_subcut": True, "stat": None} for _ in colors]
    plan = ps.plan_transitions([b["section"] for b in blocks], blocks)
    assert all(d == pytest.approx(1 / fps) for _t, d in plan)
    durs = ps.phrase_locked_durations(onsets, total, plan, fps=fps)
    clips = []
    for i, (c, d) in enumerate(zip(colors, durs)):
        p = str(tmp_path / f"c{i}.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                        f"color=c={c}:s=320x180:r={fps}", "-t", f"{d:.6f}",
                        "-pix_fmt", "yuv420p", p], check=True)
        clips.append(p)
    out = str(tmp_path / "merged.mp4")
    ok, _ = ps.xfade_chain(clips, durs, [b["section"] for b in blocks], out, blocks=blocks, plan=plan)
    assert ok
    cuts = vt.detect_cuts(out)
    pairs, unmatched, _extra = vt.match_cuts(onsets[1:], cuts)
    assert not unmatched
    for e, d in pairs:
        assert abs(d - e) <= 0.5 / fps + 1e-3, (e, d)


def test_phrase_timeline_drift_counts_the_frame_xfade_hides():
    blocks = [{"section": "HOOK", "is_subcut": False, "stat": None},
              {"section": "HOOK", "is_subcut": True, "stat": None}]
    durs = [2.0, 2.0]
    origin = ps.hook_visual_starts(blocks, durs)
    seen = ps.first_visible_starts(blocks, durs)
    assert seen[1] - origin[1] == pytest.approx(1 / ps.FPS)
    assert seen[0] == origin[0] == 0.0


# ---------------------------------------------------------------- D. рез не поперёк слова

def _timeline(tmp_path, blocks, words):
    plan = tmp_path / "media_plan"
    plan.mkdir(parents=True, exist_ok=True)
    (plan / "phrase_timeline.json").write_text(json.dumps(
        {"locked": True, "fps": 24, "blocks": blocks, "speech_words": words}), encoding="utf-8")
    (tmp_path / "final.mp4").write_bytes(b"x")


def test_cut_on_onset_is_not_across_speech_even_without_silence(tmp_path, monkeypatch):
    """Под голосом музыка — тишины в final.mp4 нет вообще (03_plen: 0
    интервалов). Прежняя ось давала «cuts_across_speech» любому эпизоду."""
    onsets = [0.0, 2.0, 4.0, 6.0]
    blocks = [{"speech_onset_sec": t, "transition_in_sec": (0.0 if i == 0 else 1 / 24)}
              for i, t in enumerate(onsets)]
    words = [(0.1, 0.9), (1.0, 1.8), (2.0, 2.7), (2.8, 3.8), (4.0, 4.6), (4.7, 5.9), (6.0, 7.0)]
    _timeline(tmp_path, blocks, words)
    monkeypatch.setattr(vt, "detect_cuts", lambda p, r=None: [2.02, 4.02, 6.02])
    rep, code = vt.verify(str(tmp_path))
    assert rep["cuts_across_words"]["across"] == 0
    assert rep["verdict"] == "ok" and code == 0


def test_cut_inside_word_is_caught(tmp_path, monkeypatch):
    onsets = [0.0, 2.0, 4.0, 6.0]
    blocks = [{"speech_onset_sec": t, "transition_in_sec": 1 / 24} for t in onsets]
    words = [(1.0, 1.8), (2.0, 2.7), (4.0, 4.9), (6.0, 7.0)]
    _timeline(tmp_path, blocks, words)
    monkeypatch.setattr(vt, "detect_cuts", lambda p, r=None: [2.3, 4.35, 6.3])
    rep, code = vt.verify(str(tmp_path))
    assert rep["cuts_across_words"]["across"] == 3
    assert rep["verdict"] in ("cuts_across_speech", "drift_median") and code == 2


def test_dissolve_is_not_judged_on_word_axis(tmp_path, monkeypatch):
    blocks = [{"speech_onset_sec": 0.0, "transition_in_sec": 0.0},
              {"speech_onset_sec": 2.0, "transition_in_sec": 10 / 24}]
    _timeline(tmp_path, blocks, [(2.0, 3.0)])
    monkeypatch.setattr(vt, "detect_cuts", lambda p, r=None: [2.2])
    rep, _ = vt.verify(str(tmp_path))
    assert rep["cuts_across_words"]["checked"] == 0


def test_no_words_means_not_measured(tmp_path, monkeypatch):
    blocks = [{"speech_onset_sec": 0.0}, {"speech_onset_sec": 2.0}]
    _timeline(tmp_path, blocks, [])
    monkeypatch.setattr(vt, "detect_cuts", lambda p, r=None: [2.5])
    rep, _ = vt.verify(str(tmp_path))
    assert rep["cuts_across_words"] is None
    assert rep["verdict"] != "cuts_across_speech"


# ---------------------------------------------------------------- E. смещения секций

@pytest.mark.skipif(not HAS_FFMPEG, reason="нужен ffmpeg")
def test_section_start_is_measured_not_summed(tmp_path):
    """Секция в склейке начинается на 25 мс раньше расчётного места — замер
    находит её с точностью до долей миллисекунды."""
    pytest.importorskip("numpy")
    import numpy as np
    sr = 22050
    rng = np.random.default_rng(7)
    voice = (rng.standard_normal(sr // 4 * 32) * 0.2).astype(np.float32)
    voice *= np.repeat(rng.random(32) > 0.3, sr // 4)[:len(voice)].astype(np.float32)
    true_start = 3.000 - 0.025
    full = np.zeros(int(sr * 12), np.float32)
    s0 = int(true_start * sr)
    full[s0:s0 + len(voice)] += voice

    def wav(path, x):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(sr), "-ac", "1",
                        "-i", "-", path], input=x.tobytes(), check=True)
    sec, mix = str(tmp_path / "sec.wav"), str(tmp_path / "full.wav")
    wav(sec, voice)
    wav(mix, full)
    got, corr = lumean_tts.measure_section_start(mix, sec, 3.000)
    assert corr > 0.99
    assert got == pytest.approx(true_start, abs=0.0005)


def test_tag_only_section_does_not_reach_tts(tmp_path):
    p = tmp_path / "script.txt"
    p.write_text("=== HOOK ===\nРаз два три.\n=== BLOCK 1: Пусто ===\n[pause]\n"
                 "=== BLOCK 2: Есть ===\nЧетыре пять.\n", encoding="utf-8")
    names = [n for n, _t in lumean_tts.extract_section_texts(str(p))]
    with contextlib.redirect_stdout(io.StringIO()):
        order = []
        for b in script_parser.parse_blocks(str(p)):
            if b["section"] not in order:
                order.append(b["section"])
    assert names == order, "нумерация alignment/NN.csv совпадает с парсером"


def test_duplicate_section_names_are_reported():
    st = [("HOOK", "a"), ("BLOCK 1: X", "b"), ("BLOCK 1: X", "c")]
    assert lumean_tts.duplicate_section_names(st) == ["BLOCK 1: X"]


# ---------------------------------------------------------------- F/G. смещения по имени

def _csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(["char,start,end"] + [f"{c},{s},{e}" for c, s, e in rows]),
                    encoding="utf-8")


def _chars(text, t0=0.0, step=0.1):
    out, t = [], t0
    for ch in text:
        out.append((ch if ch != "," else ";", round(t, 3), round(t + step, 3)))
        t += step
    return out


def test_planned_pauses_map_csv_by_section_name_with_incomplete_offsets(tmp_path):
    (tmp_path / "script.txt").write_text(
        "=== HOOK ===\nаа.[pause]бб.\n=== BLOCK 1: X ===\nвв.[pause]гг.\n"
        "=== BLOCK 2: Y ===\nдд.[pause]ее.\n", encoding="utf-8")
    al = tmp_path / "media_plan" / "alignment"
    _csv(al / "00.csv", _chars("аа.[pause]бб."))
    _csv(al / "01.csv", _chars("вв.[pause]гг."))
    _csv(al / "02.csv", _chars("дддддд.[pause]ее."))   # тег позже, чем в 01.csv
    (tmp_path / "media_plan" / "section_offsets.json").write_text(
        json.dumps({"HOOK": 0.0, "BLOCK 2: Y": 100.0}), encoding="utf-8")
    with contextlib.redirect_stdout(io.StringIO()):
        got = fix_pauses.planned_tag_pauses(str(tmp_path))
    starts = sorted(round(s, 2) for s, _e, _t, _tag in got)
    # HOOK: тег на 0.3; BLOCK 2: тег на 0.7 своего файла + 100. BLOCK 1 — пропущен.
    assert starts == [0.3, 100.7]


def test_protected_window_of_section_without_offset_is_skipped(tmp_path):
    plan = tmp_path / "media_plan"
    plan.mkdir()
    (plan / "speech_timeline.json").write_text(
        '{"protected_windows": [[1.0, 2.0, 0.9, "BLOCK 1: X#0"], [1.0, 2.0, 0.9, "HOOK#0"]]}',
        encoding="utf-8")
    (plan / "section_offsets.json").write_text('{"BLOCK 2: Y": 50.0}', encoding="utf-8")
    (tmp_path / "script.txt").write_text(
        "=== HOOK ===\nаа.\n=== BLOCK 1: X ===\nбб.\n=== BLOCK 2: Y ===\nвв.\n", encoding="utf-8")
    with contextlib.redirect_stdout(io.StringIO()):
        wins = fix_pauses.load_protected_windows(str(tmp_path))
    assert wins == [(1.0, 2.0, 0.9, "HOOK#0")]


def _point(monkeypatch, tmp_path, offsets):
    plan = tmp_path / "media_plan"
    monkeypatch.setattr(ps, "ALIGNMENT_DIR", str(plan / "alignment"))
    monkeypatch.setattr(ps, "PAUSE_CUTS_PATH", str(plan / "pause_cuts.json"))
    monkeypatch.setattr(ps, "SECTION_OFFSETS_PATH", str(plan / "section_offsets.json"))
    monkeypatch.setattr(ps, "_PAUSE_CUTS_CACHE", None)
    monkeypatch.setattr(ps, "_PAUSE_WINDOWS_CACHE", None)
    monkeypatch.setattr(ps, "_SECTION_OFFSETS_CACHE", None)
    plan.mkdir(parents=True, exist_ok=True)
    if offsets is not None:
        (plan / "section_offsets.json").write_text(json.dumps(offsets), encoding="utf-8")
    return plan / "alignment"


def test_section_without_offset_is_not_treated_as_global(monkeypatch, tmp_path):
    al = _point(monkeypatch, tmp_path, {"HOOK": 0.0})
    _csv(al / "00.csv", _chars("Раз."))
    _csv(al / "01.csv", _chars("Два."))
    blocks = [{"text": "Раз.", "words": 1, "section": "HOOK", "pause_after": 0.0},
              {"text": "Два.", "words": 1, "section": "BLOCK 1: X", "pause_after": 0.0}]
    w = ps.load_alignment_weights(blocks)
    assert w[0] is not None and w[1] is None
    assert ps.load_alignment_onsets(blocks) is None
    assert "смещения" in ps.ALIGNMENT_ONSET_FAILURE["reason"]


# ---------------------------------------------------------------- H. сегменты ↔ блоки

def test_double_pause_does_not_shift_segments(monkeypatch, tmp_path):
    al = _point(monkeypatch, tmp_path, {"HOOK": 0.0})
    text = "Раз.[pause][pause]Два.[pause]Три."
    _csv(al / "00.csv", _chars(text))
    p = tmp_path / "script.txt"
    p.write_text("=== HOOK ===\n" + text + "\n", encoding="utf-8")
    with contextlib.redirect_stdout(io.StringIO()):
        blocks = script_parser.parse_blocks(str(p))
    assert [b["text"] for b in blocks] == ["Раз.", "Два.", "Три."]
    on = ps.load_alignment_onsets(blocks)
    assert on is not None and on[1] == pytest.approx(text.index("Два") * 0.1)


def test_climax_without_pause_gets_own_weight(monkeypatch, tmp_path):
    al = _point(monkeypatch, tmp_path, {"HOOK": 0.0})
    text = "Раз.[climax]Два.[pause]Шесть."
    spoken = "Раз.Два.[pause]Шесть."           # пайплайн-only тег в TTS не уходит
    _csv(al / "00.csv", _chars(spoken))
    p = tmp_path / "script.txt"
    p.write_text("=== HOOK ===\n" + text + "\n", encoding="utf-8")
    with contextlib.redirect_stdout(io.StringIO()):
        blocks = script_parser.parse_blocks(str(p))
    assert [b["text"] for b in blocks] == ["Раз.", "Два.", "Шесть."]
    w = ps.load_alignment_weights(blocks)
    assert w[0] == pytest.approx(0.4) and w[1] == pytest.approx(0.4)
    assert w[2] == pytest.approx(0.6), "третий блок — свой сегмент, а не чужой"


# ---------------------------------------------------------------- I. регистр тегов

def test_pipeline_tags_are_parsed_case_insensitively(tmp_path):
    p = tmp_path / "script.txt"
    p.write_text("=== HOOK ===\nРаз два три.[STAT:3 КГ]Четыре.[pause][Climax]Пять шесть.\n",
                 encoding="utf-8")
    with contextlib.redirect_stdout(io.StringIO()):
        blocks = script_parser.parse_blocks(str(p))
    assert blocks[0]["stat"] == "3 КГ"
    assert blocks[0]["text"] == "Раз два три. Четыре."
    assert blocks[1]["is_climax"] is True


def test_unknown_tag_between_words_does_not_glue_them(tmp_path):
    p = tmp_path / "script.txt"
    p.write_text("=== HOOK ===\nРаз два три.[whisper]Четыре пять [whisper] шесть.\n",
                 encoding="utf-8")
    with contextlib.redirect_stdout(io.StringIO()):
        blocks = script_parser.parse_blocks(str(p))
    assert "три. Четыре" in blocks[0]["text"]          # больше не «три.Четыре»
    assert "пять  шесть" in blocks[0]["text"]          # тег между пробелами — как раньше
