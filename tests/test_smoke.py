"""Дымовой тест полного прохода pipeline_smart.py (реальный FFmpeg, без сети):
синтетические 5с аудио + 2 картинки -> final.mp4 создаётся, длительность
совпадает с аудио (в пределах падинга/xfade, проверяем допуск ±0.5с).
Запуск: .venv/bin/python -m pytest tests/test_smoke.py -v
Требует ffmpeg/ffprobe в PATH — как и весь пайплайн."""
import json
import os
import shutil
import subprocess
import sys

import pytest
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE = os.path.join(REPO_ROOT, "scripts", "pipeline_smart.py")

AUDIO_DURATION = 5.0

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe не найдены в PATH",
)


def _ffprobe_duration(path):
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", path],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(r.stdout)["format"]["duration"])


@pytest.fixture
def video_dir(tmp_path):
    d = tmp_path / "smoke_video"
    media = d / "media"
    media.mkdir(parents=True)

    (d / "script.txt").write_text(
        "=== HOOK === Раз два три четыре пять.[pause]"
        "Шесть семь восемь девять десять одиннадцать.\n",
        encoding="utf-8",
    )

    for i, color in enumerate([(200, 30, 30), (30, 30, 200)]):
        Image.new("RGB", (1280, 720), color).save(media / f"{i:02d}.jpg", quality=90)

    r = subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"sine=frequency=440:duration={AUDIO_DURATION}",
         "-c:a", "libmp3lame", str(d / "audio.mp3")],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"не удалось создать синтетическое аудио: {r.stderr[-500:]}"

    return d


def test_pipeline_produces_final_mp4_matching_audio_length(video_dir):
    env = dict(os.environ, PARALLAX="0")  # без тяжёлых depth-моделей для дымового теста
    r = subprocess.run(
        [sys.executable, PIPELINE, str(video_dir)],
        capture_output=True, text=True, timeout=300, env=env,
    )
    assert r.returncode == 0, f"pipeline_smart.py упал:\nSTDOUT:\n{r.stdout[-2000:]}\nSTDERR:\n{r.stderr[-2000:]}"

    final = video_dir / "final.mp4"
    assert final.exists(), "final.mp4 не создан"
    assert final.stat().st_size > 0

    dur = _ffprobe_duration(str(final))
    assert abs(dur - AUDIO_DURATION) <= 0.5, f"длительность final.mp4={dur:.2f}с, ожидалось ~{AUDIO_DURATION}с"
