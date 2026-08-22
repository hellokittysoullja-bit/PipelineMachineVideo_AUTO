"""Юнит- и дымовые тесты scripts/render_episode.py (Production Orchestrator,
P0-2 форензик-аудита). Логика preflight/strict-refusal тестируется с
подменённым _run() (не гонять реальные section_sync.py/fix_pauses.py на
каждый юнит-тест — они уже покрыты своими файлами); один реальный
end-to-end smoke-тест (как test_smoke.py) проверяет, что оркестратор
реально доводит дело до final.mp4 через настоящие подпроцессы."""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import pytest
from PIL import Image, ImageDraw

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
RENDER_EPISODE = os.path.join(SCRIPTS_DIR, "render_episode.py")
sys.path.insert(0, SCRIPTS_DIR)

sys.argv = ["render_episode.py", tempfile.gettempdir()]
import render_episode as re_mod   # noqa: E402

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe не найдены в PATH",
)


def write_script(path, body):
    path.write_text(body, encoding="utf-8")


# ---------- _section_count / _section_offsets_cover_all (чистая логика) ----------

def test_section_count_single_section(tmp_path):
    write_script(tmp_path / "script.txt", "=== HOOK === Раз два три.\n")
    assert re_mod._section_count(str(tmp_path)) == 1


def test_section_count_multi_section(tmp_path):
    write_script(tmp_path / "script.txt",
                 "=== HOOK === Раз два три.\n=== BLOCK 1: X === Четыре пять шесть.\n")
    assert re_mod._section_count(str(tmp_path)) == 2


def test_section_count_missing_script(tmp_path):
    assert re_mod._section_count(str(tmp_path)) == 0


def test_section_offsets_cover_all_missing_file(tmp_path):
    assert re_mod._section_offsets_cover_all(str(tmp_path), 2) is False


def test_section_offsets_cover_all_partial(tmp_path):
    plan_dir = tmp_path / "media_plan"
    plan_dir.mkdir()
    (plan_dir / "section_offsets.json").write_text('{"HOOK": 0.0}', encoding="utf-8")
    assert re_mod._section_offsets_cover_all(str(tmp_path), 2) is False


def test_section_offsets_cover_all_complete(tmp_path):
    plan_dir = tmp_path / "media_plan"
    plan_dir.mkdir()
    (plan_dir / "section_offsets.json").write_text(
        '{"HOOK": 0.0, "BLOCK 1: X": 12.5}', encoding="utf-8")
    assert re_mod._section_offsets_cover_all(str(tmp_path), 2) is True


# ---------- preflight_and_run: strict-refusal логика (с подменённым _run) ----------

def test_strict_refuses_when_audio_missing(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(re_mod, "_run", lambda *a, **kw: (calls.append(a), 0)[1])
    rc = re_mod.preflight_and_run(str(tmp_path), strict=True, legacy_allow_degraded=False)
    assert rc == 1
    assert calls == []   # ничего не должно было запускаться вообще
    manifest = json.load(open(tmp_path / "media_plan" / "timeline_manifest.json"))
    assert manifest["stages"]["audio"]["status"] == "missing"


def test_strict_refuses_when_section_sync_incomplete(tmp_path, monkeypatch):
    write_script(tmp_path / "script.txt",
                 "=== HOOK === Раз два три.\n=== BLOCK 1: X === Четыре пять шесть.\n")
    (tmp_path / "audio.mp3").write_bytes(b"fake-audio")

    def fake_run(script_name, video_dir, extra_env=None):
        # section_sync.py "запускается", но не пишет section_offsets.json
        # (та же честная ситуация, что и низкая уверенность в реальности).
        return 0
    monkeypatch.setattr(re_mod, "_run", fake_run)
    rc = re_mod.preflight_and_run(str(tmp_path), strict=True, legacy_allow_degraded=False)
    assert rc == 1
    manifest = json.load(open(tmp_path / "media_plan" / "timeline_manifest.json"))
    assert manifest["stages"]["section_sync"]["status"] == "partial"
    assert "pipeline_smart" not in manifest["stages"]   # рендер не запускался


def test_legacy_flag_overrides_strict_refusal(tmp_path, monkeypatch):
    write_script(tmp_path / "script.txt",
                 "=== HOOK === Раз два три.\n=== BLOCK 1: X === Четыре пять шесть.\n")
    (tmp_path / "audio.mp3").write_bytes(b"fake-audio")
    calls = []

    def fake_run(script_name, video_dir, extra_env=None):
        calls.append(script_name)
        return 0
    monkeypatch.setattr(re_mod, "_run", fake_run)
    rc = re_mod.preflight_and_run(str(tmp_path), strict=True, legacy_allow_degraded=True)
    assert rc == 0
    # несмотря на неполный section_sync, --legacy-allow-degraded-timing
    # обязан довести дело до pipeline_smart.py (явно разрешённый компромисс).
    assert "pipeline_smart.py" in calls


def test_default_mode_is_legacy_even_without_flag(tmp_path, monkeypatch):
    # Без --strict-production вообще (strict=False) — то же самое, что
    # --legacy-allow-degraded-timing: обратная совместимость по умолчанию.
    write_script(tmp_path / "script.txt",
                 "=== HOOK === Раз два три.\n=== BLOCK 1: X === Четыре пять шесть.\n")
    (tmp_path / "audio.mp3").write_bytes(b"fake-audio")
    calls = []
    monkeypatch.setattr(re_mod, "_run", lambda script_name, video_dir, extra_env=None:
                         (calls.append(script_name), 0)[1])
    rc = re_mod.preflight_and_run(str(tmp_path), strict=False, legacy_allow_degraded=False)
    assert rc == 0
    assert "pipeline_smart.py" in calls


def test_single_section_skips_section_sync(tmp_path, monkeypatch):
    write_script(tmp_path / "script.txt", "=== HOOK === Раз два три.\n")
    (tmp_path / "audio.mp3").write_bytes(b"fake-audio")
    calls = []
    monkeypatch.setattr(re_mod, "_run", lambda script_name, video_dir, extra_env=None:
                         (calls.append(script_name), 0)[1])
    rc = re_mod.preflight_and_run(str(tmp_path), strict=True, legacy_allow_degraded=False)
    assert rc == 0
    assert "section_sync.py" not in calls   # 1 секция — офсеты не нужны по определению


# ---------- реальный end-to-end (та же схема, что test_smoke.py) ----------

AUDIO_DURATION = 5.0


@pytest.fixture
def video_dir(tmp_path):
    d = tmp_path / "orchestrator_video"
    media = d / "media"
    media.mkdir(parents=True)
    (d / "script.txt").write_text(
        "=== HOOK === Раз два три четыре пять.[pause]"
        "Шесть семь восемь девять десять одиннадцать.\n",
        encoding="utf-8",
    )
    colors = [(200, 30, 30), (30, 30, 200), (30, 200, 30), (200, 200, 30)]
    for i, color in enumerate(colors):
        img = Image.new("RGB", (1280, 720), color)
        draw = ImageDraw.Draw(img)
        accent = colors[(i + 1) % len(colors)]
        box_w = 200 + i * 150
        draw.rectangle([50, 50, 50 + box_w, 50 + box_w], fill=accent)
        img.save(media / f"{i:02d}.jpg", quality=90)
    r = subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"sine=frequency=440:duration={AUDIO_DURATION}",
         "-c:a", "libmp3lame", str(d / "audio.mp3")],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"не удалось создать синтетическое аудио: {r.stderr[-500:]}"
    return d


def test_render_episode_end_to_end_produces_final_mp4(video_dir):
    # timeout=280, не 120: реально занимает ~80-85с изолированно (проверено
    # вживую), но в общем прогоне tests/ (10+ минут, общая CPU-песочница)
    # уже ловил реальный TimeoutExpired на 120с из-за нагрузки от других
    # тестов, выполняющихся ДО этого в той же сессии — не регрессия кода
    # (повторный изолированный запуск сразу же прошёл за 83с), а слишком
    # тесный запас для этой конкретной shared-среды.
    env = dict(os.environ, PARALLAX="0")
    r = subprocess.run([sys.executable, RENDER_EPISODE, str(video_dir), "--strict-production"],
                        capture_output=True, text=True, timeout=280, env=env)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-1000:]
    assert (video_dir / "final.mp4").exists()
    manifest = json.load(open(video_dir / "media_plan" / "timeline_manifest.json"))
    assert manifest["stages"]["audio"]["status"] == "ok"
    assert manifest["stages"]["section_sync"]["status"] == "skipped"   # 1 секция
    assert manifest["stages"]["fix_pauses"]["status"] == "ok"
    assert manifest["stages"]["pipeline_smart"]["status"] == "ok"
