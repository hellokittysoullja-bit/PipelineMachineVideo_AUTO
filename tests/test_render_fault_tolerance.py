"""Стресс-тесты отказоустойчивого рендера (verify_clip/run_ffmpeg_with_retry/
жёсткий финальный гейт) — С РЕАЛЬНЫМИ, НАМЕРЕННО СОЗДАННЫМИ сбоями, не
только happy path (см. test_smoke.py). Цель — не "никогда не упасть", а
что одиночный сбой корректно локализуется: плохой клип не портит готовый
final.mp4 молча, и из системы видно, что именно и почему не удалось.
Требует ffmpeg/ffprobe в PATH."""
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
PIPELINE = os.path.join(SCRIPTS_DIR, "pipeline_smart.py")
sys.path.insert(0, SCRIPTS_DIR)

sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]
import pipeline_smart as ps   # noqa: E402

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe не найдены в PATH",
)


def _make_clip(path, dur=2.0, color="red"):
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c={color}:s=320x180:d={dur}",
                     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "24", path],
                    capture_output=True, check=True)


# ---------- verify_clip: реальные битые/усечённые/некорректные файлы ----------

def test_verify_clip_accepts_good_file(tmp_path):
    p = str(tmp_path / "good.mp4")
    _make_clip(p, dur=2.0)
    ok, reason, dur = ps.verify_clip(p, expected_dur=2.0)
    assert ok, reason
    assert abs(dur - 2.0) < 0.2


def test_verify_clip_rejects_truncated_file(tmp_path):
    p = str(tmp_path / "good.mp4")
    _make_clip(p, dur=2.0)
    # Обрезаем файл до половины — та самая "процесс отчитался успехом, файл
    # усечён" ситуация, которую returncode==0 не ловит.
    truncated = str(tmp_path / "truncated.mp4")
    with open(p, "rb") as src:
        data = src.read()
    with open(truncated, "wb") as dst:
        dst.write(data[: len(data) // 3])
    ok, reason, _ = ps.verify_clip(truncated, expected_dur=2.0)
    assert not ok
    assert reason


def test_verify_clip_rejects_wrong_duration(tmp_path):
    p = str(tmp_path / "short.mp4")
    _make_clip(p, dur=0.5)   # заказали 2.0с, получили 0.5с
    ok, reason, dur = ps.verify_clip(p, expected_dur=2.0)
    assert not ok
    assert "короче" in reason


def test_verify_clip_rejects_non_video_file(tmp_path):
    p = str(tmp_path / "not_a_video.mp4")
    with open(p, "wb") as f:
        f.write(b"this is not an mp4 at all, just text pretending")
    ok, reason, _ = ps.verify_clip(p, expected_dur=2.0)
    assert not ok


def test_verify_clip_rejects_missing_file(tmp_path):
    ok, reason, _ = ps.verify_clip(str(tmp_path / "does_not_exist.mp4"), expected_dur=2.0)
    assert not ok


def test_verify_clip_within_tolerance_is_accepted(tmp_path):
    p = str(tmp_path / "close_enough.mp4")
    _make_clip(p, dur=1.9)   # заказали 2.0с, допуск CLIP_VERIFY_TOLERANCE_SEC=0.25
    ok, reason, _ = ps.verify_clip(p, expected_dur=2.0)
    assert ok, reason


# ---------- run_ffmpeg_with_retry: намеренные транзиентные/постоянные сбои ----------

def test_run_ffmpeg_with_retry_recovers_from_transient_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "RENDER_RETRY_ATTEMPTS", 3)
    monkeypatch.setattr(ps, "RENDER_RETRY_BACKOFF_SEC", 0.01)   # не ждать реальные 1.5с в тесте
    tmp_out = str(tmp_path / "clip.mp4")
    calls = {"n": 0}

    class FakeResult:
        def __init__(self, returncode, stderr=""):
            self.returncode = returncode
            self.stderr = stderr

    def flaky_build():
        calls["n"] += 1
        if calls["n"] < 2:
            return FakeResult(1, "transient failure")   # первая попытка — провал
        _make_clip(tmp_out, dur=1.0)
        return FakeResult(0)   # вторая попытка — успех

    ok, reason = ps.run_ffmpeg_with_retry(flaky_build, tmp_out, expected_dur=1.0, label="test")
    assert ok, reason
    assert calls["n"] == 2   # ровно одна повторная попытка потребовалась, не больше


def test_run_ffmpeg_with_retry_gives_up_after_persistent_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "RENDER_RETRY_ATTEMPTS", 3)
    monkeypatch.setattr(ps, "RENDER_RETRY_BACKOFF_SEC", 0.01)
    tmp_out = str(tmp_path / "clip.mp4")
    calls = {"n": 0}

    class FakeResult:
        returncode = 1
        stderr = "always broken"

    def always_fails():
        calls["n"] += 1
        return FakeResult()

    ok, reason = ps.run_ffmpeg_with_retry(always_fails, tmp_out, expected_dur=1.0, label="test")
    assert not ok
    assert calls["n"] == 3   # исчерпал ВСЕ RENDER_RETRY_ATTEMPTS попытки, не меньше и не больше
    assert "always broken" in reason


def test_run_ffmpeg_with_retry_detects_verified_bad_output_despite_returncode_0(tmp_path, monkeypatch):
    # Систематический класс сбоя, ради которого всё затевалось: ffmpeg
    # каждый раз отчитывается успехом (returncode 0), но реальный файл не
    # проходит verify_clip (например, кодек молча пишет не ту длину) —
    # ретрай не должен слепо доверять коду возврата НИ РАЗУ.
    monkeypatch.setattr(ps, "RENDER_RETRY_ATTEMPTS", 2)
    monkeypatch.setattr(ps, "RENDER_RETRY_BACKOFF_SEC", 0.01)
    tmp_out = str(tmp_path / "clip.mp4")

    class FakeResult:
        returncode = 0
        stderr = ""

    def always_short_but_returncode_0():
        _make_clip(tmp_out, dur=0.3)   # заказано 2.0с
        return FakeResult()

    ok, reason = ps.run_ffmpeg_with_retry(always_short_but_returncode_0, tmp_out,
                                           expected_dur=2.0, label="test")
    assert not ok
    assert "короче" in reason


# ---------- Полный прогон pipeline_smart.py: реальный сбойный клип + жёсткий гейт ----------

@pytest.fixture
def broken_clip_video_dir(tmp_path):
    """3 валидные картинки + 1 НАМЕРЕННО битый файл (не декодируется как
    изображение вообще) — сценарий на 4 sub-cut блока, local_photo()
    циклит по индексу, так что ровно один блок гарантированно достаётся
    битому файлу и его рендер провалится детерминированно, остальные три —
    успешно."""
    d = tmp_path / "broken_video"
    media = d / "media"
    media.mkdir(parents=True)

    (d / "script.txt").write_text(
        "=== HOOK === Раз два три.[pause]Четыре пять шесть.[pause]"
        "Семь восемь девять.[pause]Десять одиннадцать двенадцать.\n",
        encoding="utf-8")

    colors = [(200, 30, 30), (30, 30, 200), (30, 200, 30)]
    for i, color in enumerate(colors):
        img = Image.new("RGB", (1280, 720), color)
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 50 + 200 + i * 150, 50 + 200 + i * 150], fill=(255, 255, 0))
        img.save(media / f"{i:02d}.jpg", quality=90)
    # Четвёртый файл — НЕ картинка (ffmpeg -loop 1 -i на нём гарантированно
    # провалится декодом, детерминированный, воспроизводимый сбой).
    with open(media / "03.jpg", "wb") as f:
        f.write(b"\x00\x01\x02 not a real jpeg, ffmpeg will refuse to decode this " * 20)

    r = subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
         "-c:a", "libmp3lame", str(d / "audio.mp3")],
        capture_output=True, text=True)
    assert r.returncode == 0

    return d


def test_strict_gate_blocks_final_mp4_on_broken_clip(broken_clip_video_dir):
    env = dict(os.environ, PARALLAX="0", RENDER_PARALLEL="0",
               RENDER_RETRY_ATTEMPTS="1", RENDER_STRICT_GATE="1")
    r = subprocess.run([sys.executable, PIPELINE, str(broken_clip_video_dir)],
                        capture_output=True, text=True, timeout=180, env=env)

    assert r.returncode != 0, "гейт обязан вернуть ненулевой код при битом клипе"
    assert not (broken_clip_video_dir / "final.mp4").exists(), \
        "final.mp4 не должен быть создан, если хотя бы один клип не принят"
    assert "СТОП" in r.stdout, f"нет понятного сообщения о причине остановки:\n{r.stdout[-1500:]}"

    manifest_path = broken_clip_video_dir / "media_plan" / "render_manifest.json"
    assert manifest_path.exists(), "манифест обязан быть записан даже при остановке гейтом"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["missing"], "манифест обязан отражать хотя бы один пропущенный клип"
    statuses = {c["index"]: c["status"] for c in manifest["clips"]}
    assert "failed" in statuses.values(), "манифест обязан пометить сбойный клип как failed"
    assert "ok" in statuses.values(), "манифест обязан пометить успешные клипы как ok (не всё подряд failed)"


def test_lenient_gate_still_produces_output_when_disabled(broken_clip_video_dir):
    # RENDER_STRICT_GATE=0 — явный откат к старому поведению "лучше меньше
    # клипов, чем сорванный рендер", для тех, кому это осознанно нужно.
    env = dict(os.environ, PARALLAX="0", RENDER_PARALLEL="0",
               RENDER_RETRY_ATTEMPTS="1", RENDER_STRICT_GATE="0")
    r = subprocess.run([sys.executable, PIPELINE, str(broken_clip_video_dir)],
                        capture_output=True, text=True, timeout=180, env=env)
    assert (broken_clip_video_dir / "final.mp4").exists(), \
        "с RENDER_STRICT_GATE=0 final.mp4 обязан собраться несмотря на пропуск"
    assert r.returncode != 0, "код возврата всё равно обязан честно отражать пропуск"


def test_corrupt_cached_clip_self_heals_on_rerun(tmp_path):
    # Прогоняем простой (без сбойных картинок) сценарий один раз, потом
    # ВРУЧНУЮ бьём один уже готовый закэшированный клип (имитация обрыва
    # процесса посреди записи предыдущего прогона) — повторный запуск
    # обязан обнаружить битый кэш через verify_clip и перерендерить, а не
    # молча принять его или упасть. Картинки — та же схема (разные цвета +
    # пространственная структура), что уже проверена в test_smoke.py, иначе
    # ahash-дедуп (не связанный с этим тестом) сам по себе валит returncode.
    d = tmp_path / "cache_heal_video"
    media = d / "media"
    media.mkdir(parents=True)
    (d / "script.txt").write_text(
        "=== HOOK === Раз два три.[pause]Четыре пять шесть.\n", encoding="utf-8")
    colors = [(200, 30, 30), (30, 30, 200)]
    for i, color in enumerate(colors):
        img = Image.new("RGB", (1280, 720), color)
        draw = ImageDraw.Draw(img)
        accent = colors[(i + 1) % len(colors)]
        draw.rectangle([50, 50, 50 + 200 + i * 150, 50 + 200 + i * 150], fill=accent)
        img.save(media / f"{i:02d}.jpg", quality=90)
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
                     "-c:a", "libmp3lame", str(d / "audio.mp3")], capture_output=True, check=True)

    env = dict(os.environ, PARALLAX="0", RENDER_PARALLEL="0")
    r1 = subprocess.run([sys.executable, PIPELINE, str(d)], capture_output=True, text=True,
                         timeout=180, env=env)
    assert r1.returncode == 0, r1.stdout[-1500:]
    assert (d / "final.mp4").exists()

    temp_smart = d / "temp_smart"
    cached_clips = list(temp_smart.glob("clip_*.mp4"))
    assert cached_clips, "ожидался хотя бы один закэшированный клип после первого прогона"
    # Бьём кэш — обрезаем до трети размера.
    victim = cached_clips[0]
    data = victim.read_bytes()
    victim.write_bytes(data[: len(data) // 3])
    (d / "final.mp4").unlink()

    r2 = subprocess.run([sys.executable, PIPELINE, str(d)], capture_output=True, text=True,
                         timeout=180, env=env)
    assert r2.returncode == 0, r2.stdout[-1500:]
    assert (d / "final.mp4").exists(), "битый кэш должен был самоисцелиться, а не сорвать сборку"
    assert "кэш битый" in r2.stdout, "должно быть видно в логе, что кэш был обнаружен и перерендерен"
