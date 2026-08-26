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


# ---------- _chunk_bounds / xfade_chain_chunked (перенос из pipeline_smart.py) ----------
# До этой правки у слотового сборщика не было защиты от документированного
# бага ffmpeg (на длинной цепочке последовательных xfade в одном
# filter_complex возвращается код 0, но кадры молча роняются, кадр
# застревает с середины ролика — см. проверку реальной длительности в
# xfade_chain()). Единственным ответом было полный откат на голый concat —
# ролик собирался, но терял ВСЕ переходы разом.

def test_chunk_bounds_single_chunk_when_within_size():
    is_hook = [True, True, False, False, False]
    bounds = assemble._chunk_bounds(len(is_hook), is_hook, chunk_size=35)
    assert bounds == [(0, 5)]


def test_chunk_bounds_prefers_hook_body_transition_over_hard_cap():
    # chunk_size=10: цель среза — индекс 10, окно поиска границы [10,20).
    # Реальная граница хук/тело стоит на 15 — резать нужно ИМЕННО там, не
    # на hard_cap=20 (там разрыв читался бы как потеря приёма посреди
    # ровного участка, а не как естественный переход).
    is_hook = [True] * 15 + [False] * 25
    bounds = assemble._chunk_bounds(len(is_hook), is_hook, chunk_size=10)
    assert bounds[0] == (0, 15), f"первый чанк должен закончиться на границе хук/тело, получено {bounds}"


def test_chunk_bounds_covers_all_indices_contiguously_no_gaps():
    is_hook = [True] * 12 + [False] * 140
    n = len(is_hook)
    bounds = assemble._chunk_bounds(n, is_hook, chunk_size=35)
    assert bounds[0][0] == 0
    assert bounds[-1][1] == n
    for (a1, b1), (a2, b2) in zip(bounds, bounds[1:]):
        assert b1 == a2, f"чанки должны идти встык без разрывов/наложений: {bounds}"
    for a, b in bounds:
        assert b - a >= 2, f"orphan-чанк короче 2 клипов должен быть слит с соседом: {(a, b)}"


def test_xfade_chain_chunked_delegates_to_single_call_when_small(monkeypatch):
    calls = []

    def fake_xfade_chain(clips, durs, is_hook, out, xfade_dur=assemble.XFADE_DUR, plan=None):
        calls.append((clips, durs, is_hook, out, plan))
        return True, 12.34
    monkeypatch.setattr(assemble, "xfade_chain", fake_xfade_chain)

    clips = [f"c{i}.mp4" for i in range(5)]
    durs = [1.0] * 5
    is_hook = [True, True, False, False, False]
    ok, total = assemble.xfade_chain_chunked(clips, durs, is_hook, "out.mp4", "/tmp/x",
                                             chunk_size=35)
    assert ok is True and total == 12.34
    assert len(calls) == 1, "маленькая цепочка не должна дробиться на чанки"
    assert calls[0][0] == clips and calls[0][2] == is_hook


def test_xfade_chain_chunked_splits_large_chain_and_concats(monkeypatch, tmp_path):
    xfade_calls = []

    def fake_xfade_chain(clips, durs, is_hook, out, xfade_dur=assemble.XFADE_DUR, plan=None):
        xfade_calls.append((len(clips), out, plan))
        return True, float(len(clips))   # произвольная, но детерминированная длительность
    monkeypatch.setattr(assemble, "xfade_chain", fake_xfade_chain)

    concat_cmds = []

    def fake_run(cmd, capture_output=True, text=True):
        concat_cmds.append(cmd)
        return _FakeCompleted(0)
    monkeypatch.setattr(assemble.subprocess, "run", fake_run)

    n = 80
    is_hook = [True] * 12 + [False] * (n - 12)
    clips = [f"c{i}.mp4" for i in range(n)]
    durs = [1.0] * n
    ok, total = assemble.xfade_chain_chunked(clips, durs, is_hook, str(tmp_path / "merged.mp4"),
                                             str(tmp_path), chunk_size=35)
    assert ok is True
    assert len(xfade_calls) >= 2, "80 клипов при chunk_size=35 обязаны разбиться на несколько чанков"
    assert total == sum(c[0] for c in xfade_calls), "итоговая длительность = сумма по чанкам"
    # Один финальный concat всех чанков поверх индивидуальных xfade-вызовов.
    assert len(concat_cmds) == 1
    assert "-f" in concat_cmds[0] and "concat" in concat_cmds[0]
    # Чанки покрывают все 80 клипов без пропусков/наложений.
    assert sum(c[0] for c in xfade_calls) == n


def test_xfade_chain_chunked_slices_plan_per_chunk(monkeypatch, tmp_path):
    captured_plans = []

    def fake_xfade_chain(clips, durs, is_hook, out, xfade_dur=assemble.XFADE_DUR, plan=None):
        captured_plans.append(plan)
        return True, float(len(clips))
    monkeypatch.setattr(assemble, "xfade_chain", fake_xfade_chain)
    monkeypatch.setattr(assemble.subprocess, "run", lambda *a, **k: _FakeCompleted(0))

    n = 80
    is_hook = [True] * 12 + [False] * (n - 12)
    full_plan = assemble.plan_transitions(is_hook)
    bounds = assemble._chunk_bounds(n, is_hook, chunk_size=35)
    expected_slices = [full_plan[a:b - 1] for a, b in bounds]

    clips = [f"c{i}.mp4" for i in range(n)]
    durs = [1.0] * n
    assemble.xfade_chain_chunked(clips, durs, is_hook, str(tmp_path / "merged.mp4"),
                                 str(tmp_path), chunk_size=35, plan=full_plan)
    assert captured_plans == expected_slices, (
        "каждый чанк должен получить именно свой срез ОБЩЕГО плана, не пересчитывать заново")


def test_xfade_chain_chunked_any_chunk_failure_falls_back_to_full_concat(monkeypatch, tmp_path):
    call_n = [0]

    def flaky_xfade_chain(clips, durs, is_hook, out, xfade_dur=assemble.XFADE_DUR, plan=None):
        call_n[0] += 1
        if call_n[0] == 2:
            return False, 0.0   # второй чанк "застрял" — та самая ffmpeg-бага
        return True, float(len(clips))
    monkeypatch.setattr(assemble, "xfade_chain", flaky_xfade_chain)
    concat_called = []
    monkeypatch.setattr(assemble.subprocess, "run",
                        lambda *a, **k: concat_called.append(1) or _FakeCompleted(0))

    n = 80
    is_hook = [True] * 12 + [False] * (n - 12)
    clips = [f"c{i}.mp4" for i in range(n)]
    durs = [1.0] * n
    ok, total = assemble.xfade_chain_chunked(clips, durs, is_hook, str(tmp_path / "merged.mp4"),
                                             str(tmp_path), chunk_size=35)
    assert (ok, total) == (False, 0.0), (
        "один сорвавшийся чанк должен откатить ВСЮ склейку на concat, как раньше при "
        "единственной цепочке — main() ловит False и уходит на голый concat всех клипов")
    assert not concat_called, "не собирать финальный concat чанков, если хотя бы один чанк не удался"
