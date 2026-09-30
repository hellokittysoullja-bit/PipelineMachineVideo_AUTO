import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="нужен ffmpeg")
def test_end_to_end_assembly(tmp_path):
    from PIL import Image
    (tmp_path / "frames").mkdir()
    (tmp_path / "script.txt").write_text(
        "=== HOOK ===\nПредставь. [pause] Школы нет. [pause] Жив.\n", encoding="utf-8")
    for i in (1, 3):     # кадр 2 намеренно пропущен — его должен заменить сосед
        Image.new("RGB", (300, 200), "white").save(tmp_path / "frames" / f"{i:03d}.png")
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=duration=4",
                    str(tmp_path / "audio.mp3")], check=True)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "assemble_frames.py"), str(tmp_path)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                          str(tmp_path / "final.mp4")], capture_output=True, text=True).stdout
    assert abs(float(out) - 4.0) < 0.15
    assert "заменены соседом: [2]" in r.stdout
    assert (tmp_path / "final.srt").read_text(encoding="utf-8").count("-->") == 3
