#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Наблюдение за прогоном по ходу, а не в конце (живой прогон 30.09 потерял
под посреди отбора — вместе с ним пропали выбранные кадры и журнал решений):
журнал дописывается в момент решения, копии выбранных кадров появляются
по мере принятия слотов."""
import json
import os
import subprocess
import sys

from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import pipeline_smart as ps  # noqa: E402
import pod_selected_frames as psf  # noqa: E402


def test_journal_record_is_on_disk_the_moment_it_is_made(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "RUN_JOURNAL", [])
    monkeypatch.setattr(ps, "RUN_JOURNAL_LIVE", [None])
    ps.start_live_journal(str(tmp_path))
    live = tmp_path / "media_plan" / "run_journal.live.jsonl"
    ps.journal_record({"record": "slot", "index": 0})
    assert [json.loads(x) for x in live.read_text().splitlines()] == [{"index": 0, "record": "slot"}]
    ps.journal_record({"record": "slot", "index": 1})
    assert len(live.read_text().splitlines()) == 2 and len(ps.RUN_JOURNAL) == 2


def test_live_journal_failure_never_touches_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "RUN_JOURNAL", [])
    monkeypatch.setattr(ps, "RUN_JOURNAL_LIVE", [str(tmp_path / "нет" / "папки" / "j.jsonl")])
    ps.journal_record({"record": "slot", "index": 0})       # не падает
    assert ps.RUN_JOURNAL == [{"record": "slot", "index": 0}]


def _journal(plan, recs, tail=""):
    plan.mkdir(parents=True, exist_ok=True)
    (plan / "run_journal.live.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs) + tail, encoding="utf-8")


def test_selected_frames_appear_per_committed_slot(tmp_path):
    ep = tmp_path / "ep"
    photo = tmp_path / "p.jpg"
    Image.new("RGB", (1600, 900), (200, 50, 50)).save(photo)
    video = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=24:d=2",
                    "-pix_fmt", "yuv420p", str(video)], check=True)
    recs = [{"record": "attempt", "attempt_id": "0-photo-1", "index": 0, "kind": "photo",
             "media": str(photo), "verdicts": [{"kind": "judge", "score": 2}]},
            {"record": "attempt", "attempt_id": "0-video-1", "index": 0, "kind": "video", "media": None},
            {"record": "attempt", "attempt_id": "1-video-1", "index": 1, "kind": "video",
             "media": str(video), "verdicts": []}]
    _journal(ep / "media_plan", recs, tail='{"record": "attem')      # последняя строка дописывается
    done = set()
    assert psf.one_pass(str(ep), done) == 2
    sel = ep / "media_plan" / "selected"
    assert (sel / "00_photo.jpg").exists() and (sel / "01_video.jpg").exists()
    with Image.open(sel / "01_video.jpg") as im:
        assert im.width > im.height * 3, "три кадра ролика в ряд"
    index = json.loads((sel / "index.json").read_text(encoding="utf-8"))
    assert [e["index"] for e in index] == [0, 1] and index[0]["verdicts"][0]["score"] == 2
    assert psf.one_pass(str(ep), done) == 0, "уже сделанные не повторяются"


def test_broken_media_is_recorded_not_fatal(tmp_path):
    ep = tmp_path / "ep"
    _journal(ep / "media_plan", [{"record": "attempt", "attempt_id": "3-photo-1", "index": 3,
                                  "kind": "photo", "media": str(tmp_path / "нет.jpg")}])
    assert psf.one_pass(str(ep), set()) == 1
    index = json.loads((ep / "media_plan" / "selected" / "index.json").read_text(encoding="utf-8"))
    assert index[0]["thumb"] is None and index[0]["error"]


def test_judge_log_snapshot_is_written_when_a_slot_closes(tmp_path, monkeypatch):
    """Прогон 30.09 на L40: под остановлен посреди отбора, и почему прошёл кадр
    фестиваля, по журналу судьи понять было нельзя — он пишется только в конце."""
    mp = tmp_path / "media_plan"
    mp.mkdir()
    monkeypatch.setattr(ps, "RUN_JOURNAL", [])
    monkeypatch.setattr(ps, "RUN_JOURNAL_LIVE", [str(mp / "run_journal.live.jsonl")])
    monkeypatch.setattr(ps, "SHOT_JUDGE_LOG", [{"index": 0, "kind": "video", "scores": [2]}])
    monkeypatch.setitem(ps._SHOT_JUDGE_STATE, "gateway", None)
    ps.journal_record({"record": "attempt", "index": 0, "attempt_id": "0-video-1"})
    assert not (mp / "shot_judge_log.json").exists(), "снимок — по закрытию слота"
    ps.journal_record({"record": "slot", "index": 0, "outcome": "shown"})
    data = json.loads((mp / "shot_judge_log.json").read_text(encoding="utf-8"))
    assert data["calls"] == [{"index": 0, "kind": "video", "scores": [2]}]


def test_judge_log_snapshot_failure_never_touches_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "RUN_JOURNAL", [])
    monkeypatch.setattr(ps, "RUN_JOURNAL_LIVE", [str(tmp_path / "нет" / "j.jsonl")])
    monkeypatch.setattr(ps, "SHOT_JUDGE_LOG", [{"index": 0}])
    ps.journal_record({"record": "slot", "index": 0})
    assert ps.RUN_JOURNAL == [{"record": "slot", "index": 0}]
