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


def test_frame_is_fitted_whole_not_cropped(tmp_path):
    # подпись у самого края кадра 3:2 должна остаться на холсте 16:9
    from PIL import Image
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import assemble_frames as af
    src = Image.new("RGB", (1536, 1024), "white")
    for x in range(1536):
        src.putpixel((x, 20), (255, 0, 0))         # красная полоса у верхнего края (2% высоты)
    src.save(tmp_path / "f.png")
    af.fit_canvas(str(tmp_path / "f.png"), str(tmp_path / "c.png"))
    c = Image.open(tmp_path / "c.png").convert("RGB")
    assert c.size == (1920, 1080)
    reds = [y for y in range(1080) if c.getpixel((960, y))[0] > 200 and c.getpixel((960, y))[1] < 80]
    assert reds, "верхний край кадра обрезан"


def test_margins_are_a_blurred_darker_copy_of_the_frame(tmp_path):
    from PIL import Image, ImageDraw
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import assemble_frames as af
    im = Image.new("RGB", (1536, 1024), (240, 228, 205))
    ImageDraw.Draw(im).rectangle((0, 0, 40, 1024), fill=(20, 60, 200))   # синий левый край рисунка
    im.save(tmp_path / "f.png")
    af.fit_canvas(str(tmp_path / "f.png"), str(tmp_path / "c.png"))
    c = Image.open(tmp_path / "c.png").convert("RGB")
    r, g, b = c.getpixel((5, 540))
    assert b > r + 30, "поле не взято из самого кадра (синий край должен проступать)"
    assert b < 200, "поле не размыто и не затемнено"
    rr, gg, bb = c.getpixel((1915, 540))
    assert rr < 240 * af.MARGIN_DARKEN + 3, "поле не затемнено"


def test_drawing_fills_the_full_height_and_only_the_sides_are_blurred(tmp_path):
    from PIL import Image
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import assemble_frames as af
    Image.new("RGB", (1536, 1024), (240, 228, 205)).save(tmp_path / "f.png")
    af.fit_canvas(str(tmp_path / "f.png"), str(tmp_path / "c.png"))
    c = Image.open(tmp_path / "c.png").convert("RGB")
    for y in (0, 1, 1079):                     # сверху и снизу — сам рисунок, не размытие
        assert c.getpixel((960, y)) == (240, 228, 205), y
    assert c.getpixel((5, 0)) != (240, 228, 205)   # по бокам — затемнённое поле


def test_labels_stay_clear_of_what_the_zoom_crops():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import assemble_frames as af
    import labels
    assert labels.EDGE_SAFE > af.ZOOM / 2
    assert labels._inside_safe((0, 0, 1920, 1080), 1920, 1080) == (58, 32, 1862, 1048)
    assert labels.CAPTION_BAND[3] <= 1 - labels.EDGE_SAFE
