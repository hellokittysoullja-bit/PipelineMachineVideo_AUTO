# -*- coding: utf-8 -*-
"""Исправления звука по аудиту готового ролика 03_plen (02.10).

Каждый тест падает на коде до правки (контрольный прогон — снять правку,
снести __pycache__). Числа живого эпизода — docs/fixes_2510/sound.md."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
if "pipeline_smart" not in sys.modules:
    sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]
import pipeline_smart as ps  # noqa: E402
import sound_director as sd  # noqa: E402

HAS_FFMPEG = shutil.which("ffmpeg") is not None
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="нужен ffmpeg")


def _tone(path, freq, dur, lead=0.0, tail=0.0, vol_db=-12.0):
    """Тон с цифровой тишиной в начале (lead) и в конце (tail)."""
    af = f"volume={vol_db}dB"
    if lead:
        af += f",adelay={int(lead * 1000)}|{int(lead * 1000)}"
    if tail:
        af += f",apad=pad_dur={tail}"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    f"sine=frequency={freq}:duration={dur}", "-ac", "2", "-ar", "48000",
                    "-af", af, str(path)], check=True)
    return str(path)


def _pcm(path, ss=None, t=None):
    import numpy as np
    cmd = ["ffmpeg", "-v", "error"]
    if ss is not None:
        cmd += ["-ss", str(ss)]
    if t is not None:
        cmd += ["-t", str(t)]
    cmd += ["-i", str(path), "-f", "f32le", "-ac", "1", "-ar", "48000", "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True).stdout, dtype=np.float32)


def _min_frame_db(path, a, b, frame=0.1):
    import numpy as np
    x = _pcm(path, a, b - a)
    n = int(48000 * frame)
    out = []
    for i in range(0, len(x) - n + 1, n):
        r = float(np.sqrt(np.mean(x[i:i + n].astype(np.float64) ** 2)))
        out.append(-120.0 if r <= 1e-9 else 20 * np.log10(r))
    return min(out)


def _dur(path):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                 "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout)


# ------------------------------------------------------------- 1. подложка
@needs_ffmpeg
def test_bed_loop_never_plays_the_files_trailing_silence(tmp_path):
    """03_plen 216.0-223.25 с: петля шла через 7 с цифровой тишины хвоста
    файла. Теперь петля склеивается по звучащей части кроссфейдом."""
    a = _tone(tmp_path / "a.flac", 220, 12, tail=4.0)          # 12 с звука + 4 с тишины
    bed = ps.build_library_bed([{"start": 0.0, "end": 40.0, "path": a}], 40.0,
                               str(tmp_path / "bed.wav"))
    assert bed
    assert _min_frame_db(bed, 3.0, 34.0) > -40.0


@needs_ffmpeg
def test_bed_chapter_join_does_not_crossfade_into_leading_silence(tmp_path):
    """03_plen 559.75-561.0 с: новая глава входила кроссфейдом в свою же
    вступительную тишину — старый трек гас, новый молчал."""
    a = _tone(tmp_path / "a.flac", 220, 40)
    b = _tone(tmp_path / "b.flac", 330, 40, lead=3.0)
    bed = ps.build_library_bed([{"start": 0.0, "end": 20.0, "path": a},
                                {"start": 20.0, "end": 40.0, "path": b}], 40.0,
                               str(tmp_path / "bed.wav"))
    assert _min_frame_db(bed, 18.0, 26.0) > -30.0


@needs_ffmpeg
def test_bed_without_silence_problems_is_byte_identical_to_the_old_graph(tmp_path):
    """Всё, что работало, не меняется: файл без тишины по краям, участки
    короче файла — те же сэмплы, что даёт прежний граф (кроме добивки
    тишиной до точной длины в самом конце)."""
    a = _tone(tmp_path / "a.flac", 220, 40)
    b = _tone(tmp_path / "b.flac", 330, 40)
    bed = ps.build_library_bed([{"start": 0.0, "end": 20.0, "path": a},
                                {"start": 20.0, "end": 36.0, "path": b}], 36.0,
                               str(tmp_path / "bed.wav"))
    pa = ps._music_file_gain_cached(a, os.path.getmtime(a)) or 0.0
    pb = ps._music_file_gain_cached(b, os.path.getmtime(b)) or 0.0
    old = str(tmp_path / "old.wav")
    fmt = "asetpts=N/SR/TB,aformat=sample_rates=48000:channel_layouts=stereo"
    fc = (f"[0:a]atrim=0:24.000,{fmt},volume={pa:.2f}dB[b0];"
          f"[1:a]atrim=0:16.000,{fmt},volume={pb:.2f}dB[b1];"
          f"[b0][b1]acrossfade=d=4.0:c1=tri:c2=tri[bed];"
          f"[bed]afade=t=in:st=0:d=2,afade=t=out:st=30.000:d=6.000[out]")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-stream_loop", "-1", "-i", a, "-stream_loop", "-1",
                    "-i", b, "-filter_complex", fc, "-map", "[out]", "-t", "36.000", "-ar", "48000",
                    "-ac", "2", old], check=True)
    x, y = _pcm(bed), _pcm(old)
    n = min(len(x), len(y))
    assert n > 48000 * 35
    assert float(abs(x[:n] - y[:n]).max()) == 0.0


@needs_ffmpeg
def test_bed_is_exactly_as_long_as_the_film(tmp_path):
    """03_plen: подложка 1111.954 с при голосе 1111.979 — на 25 мс короче."""
    # mp3, как у Mixkit: поток начинается с 0.025 с, и atrim=0:L отдаёт на
    # 25 мс меньше — ровно расхождение 03_plen.
    a = _tone(tmp_path / "a.mp3", 220, 30)
    b = _tone(tmp_path / "b.mp3", 330, 30)
    c = _tone(tmp_path / "c.mp3", 440, 30)
    bed = ps.build_library_bed([{"start": 0.0, "end": 10.0, "path": a},
                                {"start": 10.0, "end": 20.0, "path": b},
                                {"start": 20.0, "end": 37.3333, "path": c}], 37.3333,
                               str(tmp_path / "bed.wav"))
    assert abs(_dur(bed) - 37.3333) < 0.002
    # Синтетика недостачу не воспроизводит (она есть только на живых mp3
    # Mixkit), поэтому гарантия проверяется и по устройству: дорожка
    # добивается тишиной до -t, а не кончается там, где кончился граф.
    seen = []
    real = ps._run_ok
    ps_run = lambda cmd, out: seen.append(cmd) or real(cmd, out)  # noqa: E731
    try:
        ps._run_ok = ps_run
        ps.build_library_bed([{"start": 0.0, "end": 10.0, "path": a}], 10.0, str(tmp_path / "b2.wav"))
    finally:
        ps._run_ok = real
    fc = seen[-1][seen[-1].index("-filter_complex") + 1]
    assert fc.endswith(",apad[out]")


def test_repeated_bed_track_continues_instead_of_restarting(monkeypatch):
    """03_plen: HOOK и BLOCK 9 оба играли mixkit_712 с начала — одно
    вступление дважды. Повтор трека продолжает пьесу."""
    monkeypatch.setattr(ps, "music_content_window", lambda p: (0.0, 200.0, 200.0))
    first = ps._bed_segment_source(0, 0, "/x/a.mp3", 60.0, 0.0, lead_trim=False, pre=0.0)
    assert "atrim=0:60.000" in first[1]                     # первый раз — прежняя форма
    again = ps._bed_segment_source(5, 4, "/x/a.mp3", 50.0, first[2], lead_trim=True, pre=0.0)
    assert "atrim=60.000:110.000" in again[1]


# ------------------------------------------------------------- 2. врезки
@needs_ffmpeg
def test_cue_level_is_set_by_the_part_that_plays_not_by_the_whole_file(tmp_path):
    """03_plen: первые 15.6 с mixkit_602 тише всего файла на 7.5 LU, врезка
    вышла тише подложки, которую заменяла."""
    quiet = _tone(tmp_path / "q.flac", 300, 12, vol_db=-14.0)
    loud = _tone(tmp_path / "l.flac", 300, 60, vol_db=8.0)
    track = str(tmp_path / "piece.flac")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", quiet, "-i", loud, "-filter_complex",
                    "[0:a][1:a]concat=n=2:v=0:a=1[o]", "-map", "[o]", track], check=True)
    cue = ps.build_music_cue_track([{"start": 0.0, "end": 12.0, "role": "chapter", "path": track,
                                     "seed": 0}], 12.0, str(tmp_path))
    got = ps.measure_integrated_lufs(cue)
    assert got is not None and abs(got - ps.MUSIC_CUE_TARGET_LUFS) < 2.0, got


@needs_ffmpeg
def test_cue_skips_the_files_leading_silence(tmp_path):
    p = _tone(tmp_path / "p.flac", 300, 30, lead=3.0)
    cue = ps.build_music_cue_track([{"start": 5.0, "end": 20.0, "role": "chapter", "path": p,
                                     "seed": 0}], 25.0, str(tmp_path))
    assert _min_frame_db(cue, 7.0, 9.0) > -40.0


def test_cue_without_file_never_falls_back_to_a_niche_folder(monkeypatch, tmp_path):
    """Запасной путь брал library_sounds("music", "medieval") для любой ниши."""
    asked = []
    monkeypatch.setattr(ps, "library_sounds", lambda kind, name: asked.append((kind, name)) or [])
    assert ps.build_music_cue_track([{"start": 0.0, "end": 10.0, "role": "intro", "seed": 0}],
                                    10.0, str(tmp_path)) is None
    assert asked == []


# ------------------------------------------------------------- 5. переходы
@needs_ffmpeg
@pytest.mark.parametrize("kind,files,gain", [
    ("chapter", ("transition/chapter_turn_long.flac", "transition/chapter_turn_short.flac"),
     "SFX_CHAPTER_GAIN_DB"),
    ("plate", ("ui/plate_tick.flac",), "SFX_PLATE_GAIN_DB")])
def test_cue_target_is_the_original_creative_loudness_not_its_peak(kind, files, gain):
    """CLAUDE.md: «переход -12 дБ (пик ок. -22 dBFS)». Цель громкости
    обязана совпасть с громкостью тех ассетов, на которых замысел ставился,
    после их -12/-16 дБ, — а не с числом пика (-22), прочитанным как LUFS."""
    vals = [ps.measure_max_momentary_lufs(os.path.join(ROOT, "assets", "sfx", f)) for f in files]
    assert all(v is not None for v in vals)
    designed = sum(vals) / len(vals) + getattr(ps, gain)
    assert abs(ps.SFX_CUE_TARGET_LUFS[kind] - designed) <= 0.6


# ------------------------------------------------------------- 6/11. мастер
@needs_ffmpeg
def test_final_loudness_is_measured_audio_only(tmp_path, monkeypatch):
    mp4 = str(tmp_path / "f.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25",
                    "-f", "lavfi", "-i", "sine=frequency=440", "-t", "4", "-c:v", "libx264",
                    "-c:a", "aac", mp4], check=True)
    seen = []
    real = subprocess.run

    def spy(cmd, *a, **k):
        seen.append((list(cmd), k.get("timeout")))
        return real(cmd, *a, **k)
    monkeypatch.setattr(ps.subprocess, "run", spy)
    got = ps.measure_final_loudness(mp4)
    assert got and got["I"] < 0 and "TP" in got
    meas = [c for c, _t in seen if "loudnorm" in " ".join(c)]
    assert meas and "-vn" in meas[0]


# ------------------------------------------------------------- 7. отчёт
def test_report_says_music_played_when_the_bed_came_from_the_library(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "MUSIC_ENABLED", False)          # синтезированный дрон снят
    monkeypatch.setattr(ps, "MUSIC_BED_DECISION", {"bed_source": "library", "beds": [{"id": "x"}]})
    path = ps.write_audio_master_report(str(tmp_path), -14.0, -1.7, made_by="audio_preview")
    data = json.load(open(path, encoding="utf-8"))
    assert data["music_enabled"] is True and data["synth_bed_enabled"] is False
    assert data["made_by"] == "audio_preview" and data["final_tp"] == -1.7


def test_audio_preview_rewrites_the_master_report():
    import inspect
    src = inspect.getsource(ps.write_audio_preview)
    assert "write_audio_master_report(" in src


# ------------------------------------------------------------- 8. устаревший план
def test_stale_plan_from_disk_is_announced(tmp_path, capsys):
    (tmp_path / "media_plan").mkdir()
    (tmp_path / "media_plan" / "sound_plan.json").write_text(json.dumps({
        "version": sd.PLAN_VERSION, "signature": "old",
        "cues": [{"section": "HOOK", "text": "Фраза, которой больше нет.", "name": "night"}]}),
        encoding="utf-8")
    cues, src = sd.episode_sound_cues(str(tmp_path), [{"section": "HOOK", "text": "Новая фраза."}],
                                      [0.0], None, ["night"])
    out = capsys.readouterr().out
    assert src == "director" and cues == []
    assert "1 из 1 событий не нашли свою фразу" in out and "другой версии" in out


def test_stale_music_plan_names_missing_chapters():
    note = sd.stale_music_note({"signature": "a", "beds": [{"section": "B9", "id": "x"}]},
                               [{"section": "HOOK"}], expected_sig="b")
    assert "другой версии" in note and "1 врезок/подложек" in note


# ------------------------------------------------------------- 9. нет плана
def test_no_plan_with_director_on_means_silence_not_the_dictionary_bed(tmp_path, monkeypatch):
    import ambience_plan
    called = []
    monkeypatch.setattr(ambience_plan, "plan_ambience", lambda *a, **k: called.append(1) or [])
    out = ps.run_ambience("/x/mix.wav", str(tmp_path), [], [], 60.0, "/x/v.wav",
                          sound={"source": "none", "amb": [], "music": []})
    assert out == "/x/mix.wav" and called == []
    rep = json.load(open(tmp_path / "media_plan" / "ambience_plan.json", encoding="utf-8"))
    assert rep["events"] == [] and rep["summary"]["source"] == "none"


# ------------------------------------------------------------- 10. решения видны
def test_schedule_drops_are_recorded_with_a_reason():
    dropped = []
    ev = sd.ambience_events([(0, "amb", "night"), (1, "amb", "rain_mud")], [10.0, 15.0], 300.0,
                            dropped=dropped)
    assert [e["kind"] for e in ev] == ["night"]
    assert dropped and dropped[0]["kind"] == "rain_mud" and "start_gap" in dropped[0]["reason"]


def test_sting_cut_by_spacing_is_recorded(tmp_path, monkeypatch):
    import music_library
    p = tmp_path / "t.mp3"
    p.write_bytes(b"x")
    tracks = {k: {"id": k, "card": "c", "dur": 100, "path": str(p)} for k in "abc"}
    monkeypatch.setattr(music_library, "track_path", lambda t: t["path"])
    blocks = [{"section": s, "text": "x"} for s in ("HOOK", "B1", "B2", "B3", "FINAL")]
    starts = [0.0, 200.0, 300.0, 600.0, 900.0]
    dropped = []
    cues, _beds = sd.music_plan_cues({"stings": [{"section": "B1", "id": "a"},
                                                 {"section": "B2", "id": "b"}]},
                                     blocks, starts, 1000.0, tracks, dropped=dropped)
    assert [c["id"] for c in cues] == ["a"]
    assert dropped == [{"role": "chapter", "id": "b", "section": "B2",
                        "reason": "within_150s_of_previous_sting"}]


# ------------------------------------------------------------- 14. ниша в профиле
KINDS = ["wind_open", "forest_birds", "night", "stone_hall", "forge_fire", "rain_mud", "river_stream",
         "crowd_market", "sea_waves", "battle_distant", "cavalry_horses", "church_bells", "crows_field",
         "war_drums"]
# Отпечатки заданий модели, снятые кодом ДО переноса ниши в профиль: для
# этого канала тексты обязаны остаться байт-в-байт (они входят в ключ кэша
# ответов модели — изменись хоть символ, план 03_plen перепокупался бы).
FROZEN = {"amb": "cc89cbe1e1ab88b5d6b01dfa20ea96f21e18a8d739daa01f68875862dc5f8b8f",
          "critic": "91dabdbb5751624378b3d505a6a761c4db0802a7f9d27d916924857161afc38e",
          "music": "618e84ede0104ca2cfb11074a6d3b252eab8e98dfc47c18e6eaa389b72e0f0a0",
          "music_critic": "5c54e4855d50789613df0f7acf8ccd30d90c678aadd23bb8dbbcbdd6b1c72f11"}


def _prompts(kinds=KINDS):
    units = [(0, 0.0, "Представь, что ты лежишь в грязи."), (1, 12.0, "Конница пошла в атаку.")]
    return {
        "amb": sd.render_prompt("Плен", "HOOK", units, "конец", kinds, "historical; years 1300-1500"),
        "critic": sd.render_critic("Плен", "HOOK", units, kinds,
                                   [(1, "battle_distant"), (2, "cavalry_horses")],
                                   "historical; years 1300-1500"),
        "music": sd.music_prompt_text(title="Плен", world="w", chapters="1 [00:00] HOOK: a ... b",
                                      cards="x | 10s | c"),
        "music_critic": sd.music_critic_prompt_text(title="Плен", world="w",
                                                    chapters="1 [00:00] HOOK: a ... b",
                                                    options="intro | A: x | c")}


def test_current_channel_prompts_are_byte_identical():
    got = {k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in _prompts().items()}
    assert got == FROZEN
    assert list(sd.kind_descriptions()) == KINDS


def test_without_a_sound_profile_the_prompts_carry_no_niche(monkeypatch):
    import channel_profile
    monkeypatch.setattr(channel_profile, "load", lambda path=None: {})
    p = _prompts(["night", "stone_hall", "cavalry_horses", "church_bells"])
    text = " ".join(p.values()).lower()
    for word in ("medieval", "middle ages", "knight", "castle", "lute"):
        assert word not in text, word
