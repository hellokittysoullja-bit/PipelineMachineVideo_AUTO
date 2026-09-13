"""Акценты на моментах [climax] подключены к сборке звука.

РЕАЛЬНЫЙ пробел, найденный прямой проверкой: scripts/generate_reveal_sfx.py
написан, оба ассета сгенерированы и лежат в assets/sfx/reveal/, докстринг
генератора прямо называет дыру ("момент, когда сценарий говорит «вот
главный ответ», проходил вообще без звукового усиления"), — а

    grep -rn "reveal_hit\\|reveal_riser" scripts/*.py   (вне генератора)

давало ПУСТО. Шаг сборки, на который ссылается тот же докстринг
(scripts/mix_reveal_sfx.py), не был написан никогда.

Измерено на 23-минутном эпизоде: звучал ровно ОДИН тип эффекта (щелчки
машинки) и только на 2 плашках из 11. Кульминация — без звука.

ГЛАВНОЕ, что держат тесты: момент акцента берётся ТОТ ЖЕ, что уже кормит
музыкальный провал — звук и музыка не имеют права говорить о двух слегка
разных моментах.
"""
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]

import feature_flags as ff  # noqa: E402
import pipeline_smart as ps  # noqa: E402


def test_assets_exist_and_are_wired():
    """Тот самый класс пробела: ассет может лежать и не вызываться."""
    assert os.path.exists(ps.REVEAL_RISER_PATH)
    assert os.path.exists(ps.REVEAL_HIT_PATH)
    import inspect
    # Цепочка слоёв вынесена в build_episode_audio_layers() — её же зовёт
    # предпросмотр звука, чтобы человек слушал ровно то, что уйдёт в ролик.
    assert "add_reveal_sfx(" in inspect.getsource(ps.build_episode_audio_layers)
    assert "build_episode_audio_layers(" in inspect.getsource(ps.main)


def test_flag_default_on():
    assert ff.FLAGS["REVEAL_SFX"].default == "1"


def test_no_climax_means_untouched_mix():
    """Эпизод без [climax] — микс возвращается тем же путём, ноль изменений."""
    assert ps.add_reveal_sfx("/tmp/mix.wav", [], 100.0, "/tmp/out.wav") == "/tmp/mix.wav"
    assert ps.add_reveal_sfx("/tmp/mix.wav", None, 100.0, "/tmp/out.wav") == "/tmp/mix.wav"


def test_disabled_flag_is_a_full_noop(monkeypatch):
    monkeypatch.setattr(ps, "REVEAL_SFX_ENABLED", False)
    assert ps.add_reveal_sfx("/tmp/mix.wav", [12.0], 100.0, "/tmp/out.wav") == "/tmp/mix.wav"


def test_uses_the_same_moment_as_the_music_dip():
    """Звук и музыка обязаны говорить об одном моменте: обе ветки кормятся
    из одной переменной climax_times, а не из двух расчётов."""
    import inspect
    body = inspect.getsource(ps.build_episode_audio_layers)
    assert "climax_times = [sub_starts[i]" in body
    assert "add_reveal_sfx(premix, climax_times" in body
    assert "climax_times=climax_times" in body   # тот же список уходит в музыку


def test_gain_keeps_the_accent_under_the_voice():
    """Пик ассета -14 dBFS ≈ уровень программы после мастеринга в -14 LUFS.
    Без ослабления акцент звучал бы вровень с голосом."""
    assert ps.REVEAL_SFX_GAIN_DB < 0
    assert ps.REVEAL_SFX_GAIN_DB <= -6.0


def test_riser_duration_is_read_from_the_file_not_hardcoded():
    """Перегенерация ассета другой длины не должна тихо сдвинуть акцент."""
    import inspect
    body = inspect.getsource(ps.add_reveal_sfx)
    assert "get_media_duration(REVEAL_RISER_PATH)" in body
    assert "1.6" not in body and "1.60" not in body


def test_broken_ffmpeg_falls_back_to_the_original_mix(monkeypatch, tmp_path):
    """Сбой не имеет права испортить уже собранный звук."""
    monkeypatch.setattr(ps, "REVEAL_SFX_ENABLED", True)
    monkeypatch.setattr(ps, "get_media_duration", lambda p: 1.6)

    class _Fail:
        returncode = 1
        stderr = "boom"

    monkeypatch.setattr(ps.subprocess, "run", lambda *a, **kw: _Fail())
    src = str(tmp_path / "mix.wav")
    assert ps.add_reveal_sfx(src, [10.0], 60.0, str(tmp_path / "out.wav")) == src


def test_riser_never_starts_before_zero(monkeypatch, tmp_path):
    """Кульминация в первые секунды ролика не должна давать отрицательный
    сдвиг (ffmpeg adelay его не принимает)."""
    monkeypatch.setattr(ps, "REVEAL_SFX_ENABLED", True)
    monkeypatch.setattr(ps, "get_media_duration", lambda p: 1.6)
    seen = {}

    class _Ok:
        returncode = 0
        stderr = ""

    def _capture(cmd, **kw):
        seen["cmd"] = cmd
        return _Ok()

    monkeypatch.setattr(ps.subprocess, "run", _capture)
    ps.add_reveal_sfx(str(tmp_path / "m.wav"), [0.3], 60.0, str(tmp_path / "o.wav"))
    graph = seen["cmd"][seen["cmd"].index("-filter_complex") + 1]
    assert "adelay=-" not in graph, graph
