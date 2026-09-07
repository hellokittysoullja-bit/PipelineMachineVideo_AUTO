"""Тесты мастер-цепочки звука: уровень подложки, лимитер, QC финала.

До 07.09 тестов на звуковой микс не было НИ ОДНОГО. Обе реальные поломки
музыки в этом проекте нашлись ручными замерами постфактум, а не отчётом и не
тестом — в том числе та, ради которой написан этот файл: усиление подложки
стояло константой -13 дБ, выведенной из пикового уровня ассета (-12 dBFS),
тогда как решает его ГРОМКОСТЬ (-30.0 LUFS). Разрыв с голосом получался
27 LU вместо задуманных 16, музыку в опубликованном ролике было практически
не слышно, и ни один отчёт этого не показывал: loudnorm выравнивает микс
целиком, поэтому итоговые -14 LUFS выглядели идеально.

Ключевая мысль набора: константу, выведенную из свойств файла, обязан
сторожить тест, который этот файл ИЗМЕРЯЕТ. Иначе она снова разойдётся с
реальностью при первой же замене ассета, и снова молча.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]
import pipeline_smart as ps  # noqa: E402

HAS_FFMPEG = shutil.which("ffmpeg") is not None
MUSIC_ASSET = os.path.join(REPO_ROOT, "assets", "music", "ambient_bed.flac")


class TestMusicBedGain:
    """Усиление подложки выводится из ЗАДУМАННОГО разрыва, а не из константы."""

    def test_gain_achieves_the_intended_gap(self, monkeypatch):
        levels = {"voice.wav": -16.0, "music.wav": -30.0}
        monkeypatch.setattr(ps, "measure_integrated_lufs",
                            lambda p: levels[os.path.basename(p)])
        gain, detail = ps.music_bed_gain_db("voice.wav", "music.wav")
        # Подложка после усиления должна оказаться ровно на GAP ниже голоса.
        assert pytest.approx(-30.0 + gain, abs=1e-6) == -16.0 - ps.MUSIC_BED_GAP_LU
        assert detail["source"] == "measured"
        assert not detail["clamped"]

    def test_gain_follows_a_different_asset(self, monkeypatch):
        """Смысл всей правки: заменили музыку — усиление поехало за ней.

        Со старой глухой константой громкий ассет дал бы разрыв 3 LU
        (музыка спорит с голосом), тихий — 27 LU (музыки не слышно), и в
        обоих случаях в коде стояло бы одно и то же число.
        """
        levels = {"voice.wav": -16.0, "loud.wav": -14.0}
        monkeypatch.setattr(ps, "measure_integrated_lufs",
                            lambda p: levels[os.path.basename(p)])
        gain, _ = ps.music_bed_gain_db("voice.wav", "loud.wav")
        assert pytest.approx(-14.0 + gain, abs=1e-6) == -16.0 - ps.MUSIC_BED_GAP_LU
        assert gain < -10, "громкий ассет обязан быть прижат сильно"

    def test_broken_measurement_is_clamped_not_amplified(self, monkeypatch):
        """Битый/пустой ассет меряется как -70 LUFS и попросил бы +38 дБ."""
        levels = {"voice.wav": -16.0, "broken.wav": -70.0}
        monkeypatch.setattr(ps, "measure_integrated_lufs",
                            lambda p: levels[os.path.basename(p)])
        gain, detail = ps.music_bed_gain_db("voice.wav", "broken.wav")
        assert gain == ps.MUSIC_BED_GAIN_MAX_DB
        assert detail["clamped"] is True
        assert detail["raw_gain_db"] > ps.MUSIC_BED_GAIN_MAX_DB

    def test_failed_measurement_falls_back_and_says_so(self, monkeypatch, capsys):
        monkeypatch.setattr(ps, "measure_integrated_lufs", lambda p: None)
        gain, detail = ps.music_bed_gain_db("voice.wav", "music.wav")
        assert gain == ps.MUSIC_BED_GAIN_DB
        assert detail["source"] == "fallback_constant"
        # Молчаливый откат — это ровно то, как прошлая поломка дожила до
        # публикации: в логе обязана остаться строка.
        assert "ВНИМАНИЕ" in capsys.readouterr().out

    @pytest.mark.skipif(not HAS_FFMPEG or not os.path.exists(MUSIC_ASSET),
                        reason="нужен ffmpeg и музыкальный ассет канала")
    def test_fallback_constant_still_matches_the_shipped_asset(self):
        """Сторож ровно того бага, который здесь и случился.

        MUSIC_BED_GAIN_DB — запасное значение на случай сорванного
        измерения, посчитанное под ТЕКУЩИЕ ассеты. Если ассет заменят, а
        константу забудут, откат снова уведёт подложку в неслышимость — и
        снова молча. Тест меряет реальный файл и требует, чтобы константа
        оставалась тем, чем себя называет.
        """
        music_lufs = ps.measure_integrated_lufs(MUSIC_ASSET)
        assert music_lufs is not None, "ebur128 не измерил ассет"
        # Голос канала держится около -16 LUFS (замер audio_fixed.flac
        # эпизода 01) — это и есть предпосылка запасной константы.
        expected = -16.0 - ps.MUSIC_BED_GAP_LU - music_lufs
        assert abs(ps.MUSIC_BED_GAIN_DB - expected) <= 1.5, (
            f"запасное усиление {ps.MUSIC_BED_GAIN_DB} dB разошлось с ассетом "
            f"({music_lufs} LUFS): под задуманный разрыв {ps.MUSIC_BED_GAP_LU} LU "
            f"нужно {expected:.1f} dB")


class TestMasterChain:
    """Порядок ступеней финального прохода — не вкусовщина, см. докстринг."""

    STATS = {"input_i": -20.1, "input_tp": -3.2, "input_lra": 7.0,
             "input_thresh": -30.5, "target_offset": 0.3}

    def test_limiter_sits_after_loudnorm_and_before_fades(self, monkeypatch):
        monkeypatch.setattr(ps, "MASTER_LIMITER_ENABLED", True)
        af = ps.build_master_af(self.STATS, 100.0, 0.05)
        assert af.index("loudnorm") < af.index("alimiter") < af.index("afade")

    def test_limiter_ceiling_equals_the_loudnorm_true_peak_target(self, monkeypatch):
        """Две ступени не должны спорить о потолке."""
        monkeypatch.setattr(ps, "MASTER_LIMITER_ENABLED", True)
        af = ps.build_master_af(self.STATS, 100.0, 0.05)
        assert f"alimiter=limit={ps.LOUDNORM_TARGET_TP}dB" in af

    def test_limiter_never_raises_the_level_back(self, monkeypatch):
        """level=disabled обязателен: авто-уровень обнулил бы loudnorm."""
        monkeypatch.setattr(ps, "MASTER_LIMITER_ENABLED", True)
        assert "level=disabled" in ps.build_master_af(self.STATS, 100.0, 0.05)

    def test_flag_off_restores_the_previous_chain_exactly(self, monkeypatch):
        """Откат по флагу обязан быть побайтово прежним поведением."""
        monkeypatch.setattr(ps, "MASTER_LIMITER_ENABLED", False)
        af = ps.build_master_af(self.STATS, 100.0, 0.05)
        assert "alimiter" not in af
        assert af == (
            f"loudnorm=I={ps.LOUDNORM_TARGET_I}:TP={ps.LOUDNORM_TARGET_TP}:"
            f"LRA={ps.LOUDNORM_TARGET_LRA}:linear=true:"
            f"measured_I=-20.1:measured_TP=-3.2:measured_LRA=7.0:"
            f"measured_thresh=-30.5:offset=0.3,"
            f"afade=t=in:st=0:d=0.05,afade=t=out:st=100.000:d=2")

    def test_single_pass_fallback_still_gets_the_limiter(self, monkeypatch):
        """Первый проход не измерился -> динамический loudnorm.

        Именно там потолок и мягче всего, лимитер нужен тем более.
        """
        monkeypatch.setattr(ps, "MASTER_LIMITER_ENABLED", True)
        af = ps.build_master_af(None, 100.0, 0.05)
        assert "measured_I" not in af and "alimiter" in af

    @pytest.mark.skipif(not HAS_FFMPEG, reason="нужен ffmpeg")
    def test_chain_is_a_valid_ffmpeg_filtergraph(self, monkeypatch, tmp_path):
        """Строка фильтров, которую ffmpeg не примет, сорвала бы весь рендер
        на самом последнем шаге — после часов работы."""
        monkeypatch.setattr(ps, "MASTER_LIMITER_ENABLED", True)
        src = tmp_path / "t.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                        "-i", "sine=frequency=440:duration=3", str(src)], check=True)
        af = ps.build_master_af(self.STATS, 2.0, 0.05)
        r = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(src),
                            "-af", af, "-f", "null", "-"],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-400:]


class TestAudioMasterReport:
    """По готовому ролику должно быть можно ответить, чем собран его звук."""

    def test_report_records_the_bed_decision(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ps, "MUSIC_BED_DECISION",
                            {"voice_lufs": -16.0, "music_lufs": -30.0,
                             "gain_db": -2.0, "source": "measured"})
        path = ps.write_audio_master_report(str(tmp_path), final_lufs=-14.2)
        data = json.load(open(path, encoding="utf-8"))
        assert data["bed"]["gain_db"] == -2.0
        assert data["target_gap_lu"] == ps.MUSIC_BED_GAP_LU
        assert data["final_lufs"] == -14.2

    def test_report_shows_when_the_limiter_was_off(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ps, "MASTER_LIMITER_ENABLED", False)
        path = ps.write_audio_master_report(str(tmp_path))
        assert json.load(open(path, encoding="utf-8"))["limiter"] is None

    def test_report_failure_never_breaks_the_render(self, monkeypatch, capsys):
        """Отчёт вспомогательный: потерять готовый ролик из-за него хуже."""
        assert ps.write_audio_master_report("/proc/nonexistent/nope") is None
        assert "ВНИМАНИЕ" in capsys.readouterr().out


class TestAudioQc:
    """QC обязан проверять то, что услышит зритель, а не только вход."""

    @pytest.mark.skipif(not HAS_FFMPEG, reason="нужен ffmpeg")
    def test_label_distinguishes_input_from_final(self, tmp_path, capsys):
        src = tmp_path / "t.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                        "-i", "sine=frequency=440:duration=2", str(src)], check=True)
        ps.audio_qc(str(src), label="Audio QC финала")
        assert "Audio QC финала" in capsys.readouterr().out

    def test_timeout_scales_with_duration(self, monkeypatch):
        """Фиксированные 120 с — полный декод; на часовом эпизоде QC молча
        не выполнялся вообще (тот же класс, что аудит 04.09 чинил у
        measure_loudnorm_stats)."""
        seen = {}

        def fake_run(cmd, **kw):
            seen["timeout"] = kw.get("timeout")
            raise RuntimeError("stop here")

        monkeypatch.setattr(ps, "_audio_len_for_timeout", lambda p: 3600.0)
        monkeypatch.setattr(ps.subprocess, "run", fake_run)
        ps.audio_qc("whatever.wav")
        assert seen["timeout"] >= 1800
