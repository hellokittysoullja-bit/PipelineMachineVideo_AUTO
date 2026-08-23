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


def _make_black_leader_clip(path, black_dur=1.2, content_dur=3.0):
    """Реальный, подтверждённый на живом Pexels-кэше эпизода случай: сток
    иногда начинается с чёрного лидер-кадра/fade-in ДОЛЬШЕ 0.5с (см.
    extract_video_probe_frame()) — этот хелпер воспроизводит именно такой
    клип детерминированно, без сети. "Реальный" сегмент — testsrc (не
    сплошной цвет!), иначе кадр сам по себе однотонный и неотличим от
    чёрного лидера по дисперсии — ложно провалил бы собственную проверку."""
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "lavfi", "-i", f"color=c=black:s=320x180:d={black_dur}:r=24",
         "-f", "lavfi", "-i", f"testsrc=s=320x180:d={content_dur}:r=24",
         "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
         "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", path],
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


# ---------- extract_video_probe_frame/measure_luma: реальный чёрный лидер-кадр
# длиннее 0.5с (найден на живом Pexels-кэше видео эпизода "01_ves-mecha",
# 0008_f62be164.mp4 — t=0.1 и t=0.5 оба чисто чёрные, реальный контент
# начинается только к t≈1.0) ----------

def test_extract_video_probe_frame_skips_black_leader(tmp_path):
    p = str(tmp_path / "black_leader.mp4")
    _make_black_leader_clip(p, black_dur=1.2, content_dur=3.0)
    tmp, cleanup = ps.extract_video_probe_frame(p, base_at=0.5, retry_ats=(1.5, 3.0))
    try:
        assert tmp is not None
        from PIL import Image
        import numpy as np
        arr = np.asarray(Image.open(tmp).convert("L"), dtype=np.float32)
        assert arr.std() >= 0.5, "должен вернуть реальный (не чёрный) кадр после ретрая"
    finally:
        if cleanup and tmp and os.path.exists(tmp):
            os.remove(tmp)


def test_extract_video_probe_frame_no_retry_needed_for_normal_clip(tmp_path):
    p = str(tmp_path / "normal.mp4")
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=s=320x180:d=2.0:r=24",
                     "-c:v", "libx264", "-pix_fmt", "yuv420p", p],
                    capture_output=True, check=True)
    tmp, cleanup = ps.extract_video_probe_frame(p, base_at=0.5, retry_ats=(1.5, 3.0))
    try:
        assert tmp is not None
    finally:
        if cleanup and tmp and os.path.exists(tmp):
            os.remove(tmp)


def test_extract_video_probe_frame_none_when_entirely_black(tmp_path):
    p = str(tmp_path / "all_black.mp4")
    _make_clip(p, dur=5.0, color="black")
    tmp, cleanup = ps.extract_video_probe_frame(p, base_at=0.5, retry_ats=(1.5, 3.0))
    assert tmp is None
    assert cleanup is False


def test_measure_luma_video_not_fooled_by_black_leader(tmp_path):
    p = str(tmp_path / "black_leader.mp4")
    _make_black_leader_clip(p, black_dur=1.2, content_dur=3.0)
    luma = ps.measure_luma(p, is_video=True)
    assert luma is not None
    # чистый чёрный кадр дал бы luma≈0.0 — реальный клип (testsrc после
    # лидера) должен читаться заметно ярче
    assert luma > 0.15, f"ожидали яркость реального кадра, получили {luma}"


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


# ---------- _looks_like_drawtext_failure / captions-fallback (реальная жалоба:
# "субтитров нет вначале, потом резко появляются, рассинхрон") ----------

def test_looks_like_drawtext_failure_true_for_real_signatures():
    assert ps._looks_like_drawtext_failure("ffmpeg вышел с кодом 1: No such filter: 'drawtext'")
    assert ps._looks_like_drawtext_failure("Cannot load default config")
    assert ps._looks_like_drawtext_failure('Unable to parse option value "..." as image size')
    assert ps._looks_like_drawtext_failure("fontfile: could not open font file")
    assert ps._looks_like_drawtext_failure("Unrecognized option 'drawtext'")


def test_looks_like_drawtext_failure_false_for_unrelated_reasons():
    # Реальный кейс, который сломал подписи: таймаут от CPU-контеншна
    # (параллельный процесс) НЕ имеет отношения к drawtext/шрифту.
    assert not ps._looks_like_drawtext_failure("таймаут (106с) — процесс завис")
    assert not ps._looks_like_drawtext_failure("ffmpeg вышел с кодом 1: Cannot allocate memory")
    assert not ps._looks_like_drawtext_failure("длительность 1.20с короче заказанной 2.00с")
    assert not ps._looks_like_drawtext_failure("")
    assert not ps._looks_like_drawtext_failure(None)


def test_kenburns_does_not_silently_drop_captions_on_unrelated_failure(tmp_path, monkeypatch):
    # РЕАЛЬНЫЙ баг, найден по жалобе пользователя на реальном рендере: WITH-
    # captions попытка проваливалась по таймауту (CPU-контеншн от параллельного
    # процесса), код трактовал ЛЮБОЙ сбой как "drawtext не работает" и тихо
    # перерисовывал клип БЕЗ титров/подписей — та версия проще и укладывалась
    # в лимит времени. Итог: подписи пропадали на случайных клипах хука, без
    # единой строчки в логе, объясняющей почему. Теперь — честный провал клипа,
    # НЕ вторая (без подписей) попытка.
    photo = str(tmp_path / "p.jpg")
    Image.new("RGB", (320, 180), (100, 110, 120)).save(photo)
    out = str(tmp_path / "clip_0000.mp4")

    calls = []

    def fake_retry(build_cmd, tmp_out, expected_dur, label=""):
        calls.append(label)
        return False, "таймаут (106с) — процесс завис"

    monkeypatch.setattr(ps, "run_ffmpeg_with_retry", fake_retry)
    ok = ps.kenburns(photo, out, 2.0, section="HOOK", captions=[("СЛОВО", 0.0, 1.0)])
    assert ok is False
    assert len(calls) == 1   # НЕ должно быть второй (без подписей) попытки
    assert not os.path.exists(out)


def test_kenburns_falls_back_without_captions_on_real_drawtext_failure(tmp_path, monkeypatch):
    # Обратный случай — сборка ffmpeg реально без drawtext: откат на версию
    # без подписей ДОЛЖЕН сработать (иначе клип потерялся бы целиком зря).
    photo = str(tmp_path / "p.jpg")
    Image.new("RGB", (320, 180), (100, 110, 120)).save(photo)
    out = str(tmp_path / "clip_0000.mp4")

    calls = []

    def fake_retry(build_cmd, tmp_out, expected_dur, label=""):
        calls.append(1)
        if len(calls) == 1:
            return False, "ffmpeg вышел с кодом 1: No such filter: 'drawtext'"
        _make_clip(tmp_out, dur=expected_dur)
        return True, "ok"

    monkeypatch.setattr(ps, "run_ffmpeg_with_retry", fake_retry)
    ok = ps.kenburns(photo, out, 2.0, section="HOOK", captions=[("СЛОВО", 0.0, 1.0)])
    assert ok is True
    assert len(calls) == 2   # первая (с подписями) провалилась ИМЕННО по drawtext -> вторая (без) удалась
    assert os.path.exists(out)


def test_video_render_does_not_silently_drop_captions_on_unrelated_failure(tmp_path, monkeypatch):
    # Тот же баг/фикс, что у kenburns() выше, но для стокового видео (video_render()).
    vid = str(tmp_path / "src.mp4")
    _make_clip(vid, dur=3.0)
    out = str(tmp_path / "clip_0000.mp4")

    calls = []

    def fake_retry(build_cmd, tmp_out, expected_dur, label=""):
        calls.append(label)
        return False, "таймаут (106с) — процесс завис"

    monkeypatch.setattr(ps, "run_ffmpeg_with_retry", fake_retry)
    ok = ps.video_render(vid, out, 2.0, section="HOOK", captions=[("СЛОВО", 0.0, 1.0)])
    assert ok is False
    assert len(calls) == 1
    assert not os.path.exists(out)


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
