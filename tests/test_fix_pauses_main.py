"""Юнит-тесты scripts/fix_pauses.py::main() — сборка сегментов и
filter_complex-графа (atrim/afade/concat), включая вырожденную ветку "все
сегменты короче 0.02с". Раньше этот путь main() (не чистые хелперы вроде
_pause_curve/_keep_sec_for, которые уже покрыты test_parse.py) гонялся
только косвенно, через замоканный subprocess в tests/test_render_episode.py,
и проверялся лишь по status == "ok" — регрессия в самой строке atrim/afade-
графа или в подсчёте сегментов не была бы поймана. Здесь ffmpeg/ffprobe НЕ
нужны — все дорогие хелперы main() (find_audio/duration/detect_silences/
measure_loudness/loudnorm_filter) монкейпатчатся, а сам ffmpeg-вызов
перехватывается и просто создаёт файл-заглушку по месту `out`."""
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

import fix_pauses   # noqa: E402


class _FakeCompleted:
    def __init__(self, returncode, stderr=""):
        self.returncode = returncode
        self.stderr = stderr


def _patch_main_deps(monkeypatch, tmp_path, sil, total, protected_windows=None):
    src = str(tmp_path / "audio.mp3")
    open(src, "wb").write(b"fake source audio")
    monkeypatch.setattr(fix_pauses, "find_audio", lambda video_dir: src)
    monkeypatch.setattr(fix_pauses, "duration", lambda path: total if path == src else 0.0)
    monkeypatch.setattr(fix_pauses, "detect_silences", lambda path: sil)
    monkeypatch.setattr(fix_pauses, "measure_loudness", lambda path: {})
    monkeypatch.setattr(fix_pauses, "loudnorm_filter", lambda stats: "loudnorm=I=-16:TP=-1.5:LRA=11")
    monkeypatch.setattr(fix_pauses, "load_protected_windows", lambda video_dir: protected_windows or [])
    return src


def _run_main_capturing_ffmpeg(monkeypatch, video_dir):
    """Подменяет subprocess.run: любой ffmpeg-вызов "успешен" и создаёт файл
    по последнему аргументу команды (тот же путь, что main() ждёт как out).
    Возвращает список всех вызванных команд (для проверки filter_complex)."""
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        out_path = cmd[-1]
        open(out_path, "wb").write(b"fake flac bytes")
        return _FakeCompleted(0)

    monkeypatch.setattr(fix_pauses.subprocess, "run", fake_run)
    old_argv = sys.argv
    sys.argv = ["fix_pauses.py", video_dir]
    try:
        rc = fix_pauses.main()
    finally:
        sys.argv = old_argv
    return rc, calls


def _filter_complex_arg(cmd):
    return cmd[cmd.index("-filter_complex") + 1]


def test_main_no_silences_normalizes_only(tmp_path, monkeypatch):
    _patch_main_deps(monkeypatch, tmp_path, sil=[], total=10.0)
    rc, calls = _run_main_capturing_ffmpeg(monkeypatch, str(tmp_path))
    assert rc == 0
    assert len(calls) == 1
    assert "-filter_complex" not in calls[0]   # ветка "нечего резать" — простой -af
    assert os.path.exists(tmp_path / "audio_fixed.flac")
    cuts = json.load(open(tmp_path / "media_plan" / "pause_cuts.json", encoding="utf-8"))
    assert cuts["cuts"] == []


def test_main_builds_one_atrim_segment_per_speech_and_pause_chunk(tmp_path, monkeypatch):
    # Одна пауза 2.0-4.0 внутри речи 0-6.0 -> 3 сегмента: речь до паузы,
    # обрезанная пауза, речь после паузы.
    sil = [(2.0, 4.0)]
    _patch_main_deps(monkeypatch, tmp_path, sil=sil, total=6.0)
    rc, calls = _run_main_capturing_ffmpeg(monkeypatch, str(tmp_path))
    assert rc == 0
    assert len(calls) == 1
    filt = _filter_complex_arg(calls[0])
    assert filt.count("atrim=") == 3
    assert filt.count("afade=t=in") == 3
    assert filt.count("afade=t=out") == 3
    assert "concat=n=3:v=0:a=1[c]" in filt
    assert "loudnorm" in filt

    expected_keep = fix_pauses._keep_sec_for(2.0, 4.0, [])
    # Второй сегмент — обрезанная пауза: должен начинаться РОВНО в 2.0 и
    # заканчиваться в 2.0+keep (та же формула, что main() использует для
    # построения segments).
    assert f"atrim=start=2.000000:end={2.0 + expected_keep:.6f}" in filt

    cuts = json.load(open(tmp_path / "media_plan" / "pause_cuts.json", encoding="utf-8"))
    assert len(cuts["cuts"]) == 1
    assert cuts["cuts"][0][0] == round(2.0 + expected_keep, 6)
    assert cuts["cuts"][0][1] == 4.0
    assert cuts["fixed_audio_md5"] is not None   # посчитан ПОСЛЕ успешного ffmpeg, не None


def test_main_degenerate_all_segments_too_short_falls_back_to_plain_normalize(tmp_path, monkeypatch):
    # Регрессия: если ВСЕ сегменты (после вычета tiny speech-огрызков и
    # обрезанных пауз) оказываются короче 0.02с, склеивать filter_complex'ом
    # нечего (ffmpeg упал бы на concat=n=0) — main() обязан честно откатиться
    # на простую нормализацию громкости исходника, не падать.
    sil = [(0.01, 0.015)]   # спич-огрызок (0, 0.01) + обрезанная пауза (0.01, 0.015) — оба < 0.02с
    _patch_main_deps(monkeypatch, tmp_path, sil=sil, total=0.015)
    rc, calls = _run_main_capturing_ffmpeg(monkeypatch, str(tmp_path))
    assert rc == 0
    assert len(calls) == 1
    assert "-filter_complex" not in calls[0]   # откат на -af, не concat=n=0
    assert os.path.exists(tmp_path / "audio_fixed.flac")


def test_main_ffmpeg_failure_returns_1_and_writes_no_output(tmp_path, monkeypatch):
    sil = [(2.0, 4.0)]
    _patch_main_deps(monkeypatch, tmp_path, sil=sil, total=6.0)

    def fake_run_fail(cmd, **kw):
        return _FakeCompleted(1, stderr="ffmpeg exploded")

    monkeypatch.setattr(fix_pauses.subprocess, "run", fake_run_fail)
    old_argv = sys.argv
    sys.argv = ["fix_pauses.py", str(tmp_path)]
    try:
        rc = fix_pauses.main()
    finally:
        sys.argv = old_argv
    assert rc == 1
    assert not os.path.exists(tmp_path / "audio_fixed.flac")


def test_main_respects_protected_window_over_curve(tmp_path, monkeypatch):
    # Speech Director запланировал ДЛИННУЮ паузу (protected) — main() обязан
    # сохранить именно её, а не срезать гладкой кривой/джиттером.
    sil = [(2.0, 4.0)]
    protected = [[2.0, 4.0, 1.8, "BLOCK1#3"]]   # [raw_start, raw_end, target_kept_sec, unit_id]
    _patch_main_deps(monkeypatch, tmp_path, sil=sil, total=6.0, protected_windows=protected)
    rc, calls = _run_main_capturing_ffmpeg(monkeypatch, str(tmp_path))
    assert rc == 0
    filt = _filter_complex_arg(calls[0])
    assert "atrim=start=2.000000:end=3.800000" in filt   # 2.0 + 1.8 из плана, не с кривой

    cuts = json.load(open(tmp_path / "media_plan" / "pause_cuts.json", encoding="utf-8"))
    assert cuts["cuts"][0] == [3.8, 4.0]
