import csv
import json
import os

import frame_timing as ft


def write_alignment(vd, idx, text, t0, step=0.05):
    os.makedirs(os.path.join(vd, "media_plan", "alignment"), exist_ok=True)
    with open(os.path.join(vd, "media_plan", "alignment", f"{idx:02d}.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["char", "start", "end"])
        t = t0
        for c in text:
            w.writerow([c, round(t, 3), round(t + step, 3)])
            t += step


def test_onsets_follow_alignment_across_sections(tmp_path):
    vd = str(tmp_path)
    write_alignment(vd, 0, "Раз два. [pause]Три четыре.", 0.0)
    write_alignment(vd, 1, "Пять шесть.", 0.0)
    json.dump({"HOOK": 0.0, "BLOCK 1": 10.0}, open(os.path.join(vd, "media_plan", "section_offsets.json"), "w"))
    blocks = [{"text": "Раз два."}, {"text": "Три четыре."}, {"text": "Пять шесть."}]
    starts, durs, rep = ft.frame_durations(vd, blocks, 12.0)
    assert rep["source"] == "alignment" and rep["found_in_alignment"] == 3
    # "Три" стоит после "Раз два. [pause]" — 16 символов по 0.05 с
    assert abs(starts[1] - 0.8) < 1e-6
    assert abs(starts[2] - 10.0) < 1e-6
    assert abs(sum(durs) - 12.0) < 1e-6


def test_pause_cuts_shift_time(tmp_path):
    vd = str(tmp_path)
    write_alignment(vd, 0, "Раз два. Три четыре.", 0.0, step=0.3)
    json.dump({"cuts": [[0.5, 0.8]], "pause_inserts": []},
              open(os.path.join(vd, "media_plan", "pause_cuts.json"), "w"))
    starts, _d, _r = ft.frame_durations(vd, [{"text": "Раз два."}, {"text": "Три четыре."}], 5.0,
                                        fixed_audio=True)
    assert abs(starts[1] - (2.7 - 0.3)) < 1e-6


def test_estimate_without_alignment(tmp_path):
    starts, durs, rep = ft.frame_durations(str(tmp_path), [{"text": "а" * 10}, {"text": "б" * 30}], 8.0)
    assert rep["source"] == "estimate" and abs(sum(durs) - 8.0) < 1e-6 and durs[1] > durs[0]


def test_unfound_phrase_is_interpolated_not_zero():
    out = ft.fill_gaps([0.0, None, 4.0], 10.0)
    assert out == [0.0, 2.0, 4.0]


def test_short_last_phrase_never_overflows_audio():
    s = ft.clamp_starts([0.0, 3.5, 3.9], 4.0)
    ends = s[1:] + [4.0]
    durs = [e - a for a, e in zip(s, ends)]
    assert abs(sum(durs) - 4.0) < 1e-9 and min(durs) >= ft.MIN_DUR - 1e-9
