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
    for i in (1, 3):     # кадр 2 отсутствует — его время должен получить сосед
        Image.new("RGB", (1664, 928), "white").save(tmp_path / "frames" / f"{i:03d}.png")
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
def test_rejected_frame_is_not_shown(tmp_path):
    from PIL import Image
    (tmp_path / "frames").mkdir()
    (tmp_path / "media_plan").mkdir()
    (tmp_path / "script.txt").write_text("=== HOOK ===\nРаз. [pause] Два. [pause] Три.\n", encoding="utf-8")
    for i in (1, 2, 3):
        Image.new("RGB", (1664, 928), "white").save(tmp_path / "frames" / f"{i:03d}.png")
    json.dump({"frames": [{"index": 1, "status": "rejected"}]},
              open(tmp_path / "media_plan" / "frames_report.json", "w"))
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import assemble_frames
    kept, absorbed = assemble_frames.kept_frames(str(tmp_path), 3)
    assert kept == [0, 2] and absorbed == [{"index": 1, "reason": "rejected"}]






def test_frame_is_fitted_whole_not_cropped(tmp_path):
    # подпись у самого края кадра 3:2 должна остаться на холсте 16:9
    from PIL import Image
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import assemble_frames as af
    src = Image.new("RGB", (1536, 1024), "white")
    for x in range(1536):
        src.putpixel((x, 3), (255, 0, 0))          # красная полоса у верхнего края
    src.save(tmp_path / "f.png")
    af.fit_canvas(str(tmp_path / "f.png"), str(tmp_path / "c.png"))
    c = Image.open(tmp_path / "c.png").convert("RGB")
    assert c.size == (1920, 1080)
    reds = [y for y in range(1080) if c.getpixel((960, y))[0] > 200 and c.getpixel((960, y))[1] < 80]
    assert reds, "верхний край кадра обрезан"
