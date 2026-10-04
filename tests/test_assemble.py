import json
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="нужен ffmpeg")
def test_end_to_end_assembly_on_old_render_chain(tmp_path):
    from PIL import Image
    (tmp_path / "frames").mkdir()
    (tmp_path / "script.txt").write_text(
        "=== HOOK ===\nПредставь, что тебе семь лет. [pause] Школы нет. [pause] Жив. Полностью.\n", encoding="utf-8")
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from frame_planner import unit_key
    texts = ["Представь, что тебе семь лет.", "Школы нет.", "Жив. Полностью."]
    for i in (1, 3):     # кадр 2 отсутствует — его время должен получить сосед
        Image.new("RGB", (1664, 928), "white").save(tmp_path / "frames" / f"{i:03d}.png")
    (tmp_path / "media_plan").mkdir()
    json.dump({"frames": [{"index": i - 1, "key": unit_key(texts[i - 1]), "status": "ok"} for i in (1, 3)]},
              open(tmp_path / "media_plan" / "frames_report.json", "w"))
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=duration=6",
                    str(tmp_path / "audio.mp3")], check=True)
    env = dict(os.environ, RENDER_WORKERS="2")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "assemble_frames.py"), str(tmp_path)],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 2, r.stdout + r.stderr          # собран, с замечанием (поглощённый кадр)
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                          str(tmp_path / "final.mp4")], capture_output=True, text=True).stdout
    audio = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                            str(tmp_path / "audio.mp3")], capture_output=True, text=True).stdout
    assert abs(float(out) - float(audio)) < 0.1
    m = json.load(open(tmp_path / "media_plan" / "render_manifest.json", encoding="utf-8"))
    assert [c["status"] for c in m["clips"]] == ["ok", "absorbed", "ok"]
    assert (tmp_path / "subtitles.srt").exists() or any(p.suffix == ".srt" for p in tmp_path.iterdir())


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="нужен ffmpeg")
def test_rejected_and_foreign_frames_are_not_shown(tmp_path):
    from PIL import Image
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import assemble_frames
    from frame_planner import unit_key
    (tmp_path / "frames").mkdir()
    (tmp_path / "media_plan").mkdir()
    for i in (1, 2, 3):
        Image.new("RGB", (64, 36), "white").save(tmp_path / "frames" / f"{i:03d}.png")
    blocks = [{"text": "Раз."}, {"text": "Два."}, {"text": "Три."}]
    json.dump({"frames": [{"index": 0, "key": unit_key("Раз."), "status": "ok"},
                          {"index": 1, "key": unit_key("Два."), "status": "rejected"},
                          # кадр 3 нарисован под фразу, которой в сценарии больше нет на этом месте
                          {"index": 2, "key": unit_key("Вставленная фраза."), "status": "ok"}]},
              open(tmp_path / "media_plan" / "frames_report.json", "w"))
    kept, absorbed = assemble_frames.kept_frames(str(tmp_path), blocks)
    assert kept == [0] and absorbed == [{"index": 1, "reason": "rejected"},
                                        {"index": 2, "reason": "frame_for_another_line"}]


def test_frame_source_prefers_clean_drawing_with_timed_labels(tmp_path):
    from PIL import Image
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import assemble_frames as af
    (tmp_path / "media_plan" / "image_cache").mkdir(parents=True)
    (tmp_path / "frames").mkdir()
    Image.new("RGB", (64, 36), "white").save(tmp_path / "media_plan" / "image_cache" / "raw.png")
    rec = {"chosen": "raw.png", "labels_placed": [{"text": "ДОФАМИН", "box": [1, 1, 30, 10]}]}
    src, recs = af.frame_source(str(tmp_path), 0, rec)
    assert src.endswith("raw.png") and recs == rec["labels_placed"]
    # запасная полоса подписей и потерянный исходник — готовый кадр, подписи уже на нём
    band = {"chosen": "raw.png", "labels_placed": [{"text": "А · Б", "box": [0, 0, 5, 5], "fallback": "band"}]}
    assert af.frame_source(str(tmp_path), 0, band) == (str(tmp_path / "frames" / "001.png"), [])
    assert af.frame_source(str(tmp_path), 0, {"chosen": "lost.png"})[1] == []


def test_labels_stay_clear_of_what_the_zoom_crops():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import camera
    import labels
    assert labels.EDGE_SAFE > camera.DRIFT / 2
    assert labels._inside_safe((0, 0, 1920, 1080), 1920, 1080) == (58, 32, 1862, 1048)
    # 3:2 стоит на всю высоту, по бокам — поля: наезд режет только верх и низ
    assert labels._inside_safe((0, 0, 1264, 848), 1264, 848) == (19, 25, 1245, 823)
    assert labels.CAPTION_BAND[3] <= 1 - labels.EDGE_SAFE
