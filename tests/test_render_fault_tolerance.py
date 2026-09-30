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
import render_core as ps   # noqa: E402

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




# ---------- SPEED_RAMP_MAX_SOURCE_FPS (реальный зависший продакшн-рендер) ----------

def _make_clip_at_fps(path, fps, dur=2.0):
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"testsrc=s=320x180:d={dur}:r={fps}",
                     "-c:v", "libx264", "-pix_fmt", "yuv420p", path],
                    capture_output=True, check=True)






# ---------- render_timeout_sec headroom for sequential parallax (реальный найденный пробел) ----------

def test_render_timeout_sec_accounts_for_sequential_parallax_headroom(monkeypatch):
    # РЕАЛЬНЫЙ пробел, пойманный вживую (не гипотеза): формула раньше
    # считала contention ТОЛЬКО от RENDER_POOL_WORKERS, не учитывая, что
    # parallax_kenburns() всегда выполняется ДОПОЛНИТЕЛЬНО, последовательно
    # в главном процессе, поверх пула. +1 к contention — прямая проверка.
    monkeypatch.setattr(ps, "RENDER_POOL_ENABLED", True)
    monkeypatch.setattr(ps, "RENDER_POOL_WORKERS", 3)
    assert ps.render_timeout_sec(10.0) == max(30 * 4, 10.0 * 15 * 4)


def test_render_timeout_sec_pool_disabled_unaffected(monkeypatch):
    # Пул выключен (RENDER_PARALLEL=0, последовательный рендер) -> contention=1
    # как и раньше — headroom применяется только когда пул реально включён.
    monkeypatch.setattr(ps, "RENDER_POOL_ENABLED", False)
    monkeypatch.setattr(ps, "RENDER_POOL_WORKERS", 3)
    assert ps.render_timeout_sec(10.0) == max(30, 10.0 * 15)






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








# ---------- Незащищённые subprocess-вызовы в финальной сборке (реальный аудит-найденный пробел) ----------

def test_xfade_chain_returns_false_on_timeout_instead_of_raising(tmp_path, monkeypatch):
    # РЕАЛЬНЫЙ пробел, найденный аудитом того же класса багов, что и
    # SPEED_RAMP_MAX_SOURCE_FPS/parallax stderr-deadlock: у этого вызова не
    # было timeout вообще — зависание ffmpeg на длинной xfade-цепочке (уже
    # задокументированный в докстринге функции живой кейс "молча роняет
    # кадры и застревает") теряло бы весь уже отрендеренный за часы прогон
    # молча. Должен вернуть (False, 0.0) — тот же контракт, что и на
    # returncode != 0 — а не уронить main() необработанным исключением.
    clip1, clip2 = str(tmp_path / "c1.mp4"), str(tmp_path / "c2.mp4")
    _make_clip(clip1, dur=1.0)
    _make_clip(clip2, dur=1.0)
    out = str(tmp_path / "merged.mp4")

    def fake_run(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw.get("timeout"))

    monkeypatch.setattr(ps.subprocess, "run", fake_run)
    ok, dur = ps.xfade_chain([clip1, clip2], [1.0, 1.0], ["HOOK", "HOOK"], out)
    assert ok is False
    assert dur == 0.0


# ---------- render_sharpness_regression: пост-рендер QC (см. RENDER_SHARPNESS_DROP_RATIO) ----------
# Реальный найденный случай (27 августа, videos/_test20s, слот 3): фикс
# _dof_focus_depth() чинит САМ баг, но ничто до этого коммита не проверяло
# ГОТОВЫЙ рендер против его же источника — эти тесты покрывают ту вторую,
# независимую линию защиты (ловит любой БУДУЩИЙ баг того же класса, не
# только сегодняшний DOF).

def _make_sharp_video(path, dur=1.0):
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"testsrc=s=640x360:d={dur}:r=24",
                     "-c:v", "libx264", "-pix_fmt", "yuv420p", path],
                    capture_output=True, check=True)


def _make_blurred_video(path, dur=1.0):
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"testsrc=s=640x360:d={dur}:r=24",
                     "-vf", "boxblur=20:2", "-c:v", "libx264", "-pix_fmt", "yuv420p", path],
                    capture_output=True, check=True)














# ---------- video_sharpness_ok: реальный найденный случай, videos/_test20s слот 7 ----------
# Видео всадника с занесённым клинком было genuинно смазано (motion-blur
# самой стоковой съёмки) — ни pexels_video(), ни render_qc_report.json
# раньше не проверяли резкость видео-кандидата вообще, только фото.







