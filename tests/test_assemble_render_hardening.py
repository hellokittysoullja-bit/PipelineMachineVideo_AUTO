"""Регрессионные тесты на отказоустойчивый рендер в scripts/assemble.py:
атомарная запись клипа (tmp -> os.replace) и ffprobe-верификация,
портированные из pipeline_smart.py (см. render_tmp_path/finalize_render/
verify_clip). Без реального ffmpeg — subprocess.run монкейпатчится."""
import json
import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

# pipeline_smart.py читает VIDEO_FOLDER из sys.argv[1] на импорте (см.
# test_parse.py/test_visual_qc.py) — под pytest sys.argv это параметры
# pytest, не путь к видео-папке. test_fps_matches_pipeline_smart() ниже
# импортирует pipeline_smart лениво, но безопасный sys.argv нужен на случай
# запуска этого файла в одиночку (когда никакой другой тестовый модуль ещё
# не успел сделать этот же трюк первым).
sys.argv = ["test_assemble_render_hardening.py", tempfile.gettempdir()]

import assemble   # noqa: E402


class _FakeCompleted:
    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _ffprobe_json(duration, has_video=True):
    streams = [{"codec_type": "video"}] if has_video else [{"codec_type": "audio"}]
    return json.dumps({"format": {"duration": str(duration)}, "streams": streams})


# ---------- verify_clip ----------

def test_verify_clip_accepts_good_duration(tmp_path, monkeypatch):
    p = str(tmp_path / "clip.mp4")
    open(p, "wb").write(b"x")
    monkeypatch.setattr(assemble.subprocess, "run",
                         lambda *a, **kw: _FakeCompleted(0, stdout=_ffprobe_json(5.0)))
    assert assemble.verify_clip(p, expected_dur=5.0) is True


def test_verify_clip_rejects_short_duration(tmp_path, monkeypatch):
    p = str(tmp_path / "clip.mp4")
    open(p, "wb").write(b"x")
    monkeypatch.setattr(assemble.subprocess, "run",
                         lambda *a, **kw: _FakeCompleted(0, stdout=_ffprobe_json(1.2)))
    assert assemble.verify_clip(p, expected_dur=5.0) is False


def test_verify_clip_rejects_missing_video_stream(tmp_path, monkeypatch):
    p = str(tmp_path / "clip.mp4")
    open(p, "wb").write(b"x")
    monkeypatch.setattr(assemble.subprocess, "run",
                         lambda *a, **kw: _FakeCompleted(0, stdout=_ffprobe_json(5.0, has_video=False)))
    assert assemble.verify_clip(p, expected_dur=5.0) is False


def test_verify_clip_rejects_nonzero_ffprobe_returncode(tmp_path, monkeypatch):
    p = str(tmp_path / "clip.mp4")
    open(p, "wb").write(b"x")
    monkeypatch.setattr(assemble.subprocess, "run", lambda *a, **kw: _FakeCompleted(1))
    assert assemble.verify_clip(p, expected_dur=5.0) is False


# ---------- render_tmp_path / finalize_render ----------

def test_finalize_render_moves_tmp_to_out_on_success(tmp_path):
    out = str(tmp_path / "out.mp4")
    tmp = assemble.render_tmp_path(out)
    open(tmp, "wb").write(b"content")
    assert assemble.finalize_render(tmp, out, ok=True) is True
    assert os.path.exists(out)
    assert not os.path.exists(tmp)


def test_finalize_render_removes_tmp_on_failure_and_leaves_no_out(tmp_path):
    out = str(tmp_path / "out.mp4")
    tmp = assemble.render_tmp_path(out)
    open(tmp, "wb").write(b"truncated garbage")
    assert assemble.finalize_render(tmp, out, ok=False) is False
    assert not os.path.exists(out)
    assert not os.path.exists(tmp)


# ---------- kenburns_clip / video_clip: атомарность конца-в-конец ----------

def _fake_ffmpeg_then_ffprobe(monkeypatch, ffmpeg_ok, verify_duration):
    """ffmpeg-вызов пишет байты в свой выходной путь (последний элемент cmd) и
    возвращает ffmpeg_ok; ffprobe-вызов (verify_clip) отдаёт verify_duration."""
    def fake_run(cmd, **kw):
        if cmd[0] == "ffmpeg":
            if ffmpeg_ok:
                open(cmd[-1], "wb").write(b"rendered bytes")
            return _FakeCompleted(0 if ffmpeg_ok else 1)
        if cmd[0] == "ffprobe":
            return _FakeCompleted(0, stdout=_ffprobe_json(verify_duration))
        raise AssertionError(f"unexpected command: {cmd[0]}")
    monkeypatch.setattr(assemble.subprocess, "run", fake_run)


def test_kenburns_clip_success_writes_out_atomically(tmp_path, monkeypatch):
    photo = str(tmp_path / "001_stock.jpg")
    open(photo, "wb").write(b"jpeg bytes")
    out = str(tmp_path / "clip_0001.mp4")
    _fake_ffmpeg_then_ffprobe(monkeypatch, ffmpeg_ok=True, verify_duration=4.0)

    ok = assemble.kenburns_clip(photo, out, d=4.0)

    assert ok is True
    assert os.path.exists(out)
    assert not os.path.exists(assemble.render_tmp_path(out))


def test_kenburns_clip_truncated_output_never_lands_under_final_name(tmp_path, monkeypatch):
    # Регрессия: ffmpeg отчитывается кодом 0, но реально записанный файл
    # короче заказанного (та же ситуация, что убитый посреди записи процесс
    # оставляет на диске). Раньше вывод писался СРАЗУ под именем `out` —
    # следующий прогон main() молча считал огрызок готовым клипом. Теперь
    # верификация должна поймать это ДО переименования в `out`.
    photo = str(tmp_path / "001_stock.jpg")
    open(photo, "wb").write(b"jpeg bytes")
    out = str(tmp_path / "clip_0001.mp4")
    _fake_ffmpeg_then_ffprobe(monkeypatch, ffmpeg_ok=True, verify_duration=0.3)   # заказано 4с

    ok = assemble.kenburns_clip(photo, out, d=4.0)

    assert ok is False
    assert not os.path.exists(out)                              # огрызок НЕ попал под финальное имя
    assert not os.path.exists(assemble.render_tmp_path(out))    # и temp-файл подчищен


def test_video_clip_success_writes_out_atomically(tmp_path, monkeypatch):
    vid = str(tmp_path / "002_stock_video.mp4")
    open(vid, "wb").write(b"mp4 bytes")
    out = str(tmp_path / "clip_0002.mp4")
    monkeypatch.setattr(assemble, "dur", lambda path: 6.0)   # исходная длительность стокового видео
    _fake_ffmpeg_then_ffprobe(monkeypatch, ffmpeg_ok=True, verify_duration=5.0)

    ok = assemble.video_clip(vid, out, d=5.0)

    assert ok is True
    assert os.path.exists(out)
    assert not os.path.exists(assemble.render_tmp_path(out))


def test_video_clip_truncated_output_never_lands_under_final_name(tmp_path, monkeypatch):
    vid = str(tmp_path / "002_stock_video.mp4")
    open(vid, "wb").write(b"mp4 bytes")
    out = str(tmp_path / "clip_0002.mp4")
    monkeypatch.setattr(assemble, "dur", lambda path: 6.0)
    _fake_ffmpeg_then_ffprobe(monkeypatch, ffmpeg_ok=True, verify_duration=0.5)   # заказано 5с

    ok = assemble.video_clip(vid, out, d=5.0)

    assert ok is False
    assert not os.path.exists(out)
    assert not os.path.exists(assemble.render_tmp_path(out))


# ---------- FPS синхронизирован с pipeline_smart.py ----------

def test_fps_matches_pipeline_smart():
    import pipeline_smart
    assert assemble.FPS == pipeline_smart.FPS
