# -*- coding: utf-8 -*-
"""Атмосферный слой подключён к сборке — и безопасен, пока источников нет.

Отдельно проверяется решение, которое легче всего потерять при следующей
правке: атмосфера НЕ прижимается сайдчейном. Музыка обязана уступать
словам, атмосфера — нет: место действия не выключается, когда человек
говорит. Прижми её тем же 9:1 — и она пропадала бы на 90% ролика и
наплывала в паузах; это качание и есть самый слышимый признак автомата.
"""
import inspect
import json
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]

import ambience_plan as ap  # noqa: E402
import feature_flags as ff  # noqa: E402
import pipeline_smart as ps  # noqa: E402


def test_flag_is_registered():
    assert ff.FLAGS["AMBIENCE_BED"].default == "1"


def test_ambience_is_actually_called_from_the_render():
    assert "run_ambience(" in inspect.getsource(ps.main)


def test_ambience_is_not_ducked_and_not_dipped():
    src = inspect.getsource(ps.add_ambience_bed)
    assert "sidechaincompress" not in src
    assert "_climax_dip_expr" not in src


def test_ambience_goes_in_after_the_music_mix():
    """Порядок важен: слой не должен попасть в сайдчейн музыки."""
    src = inspect.getsource(ps.main)
    assert src.index("build_music_mix(") < src.index("run_ambience(")


def test_level_is_measured_not_a_constant():
    """Константа уровня уже один раз молча разошлась с реальностью на 11 LU.

    И отдельно: не измерилось — слой НЕ добавляется. Фон звучит все 25
    минут, угадывать его громкость нельзя.
    """
    src = inspect.getsource(ps.ambience_gain_db)
    assert "measure_integrated_lufs" in src
    assert "AMBIENCE_GAP_LU" in src
    assert "unmeasured" in src


def test_no_sources_means_exact_no_op():
    """Генератор ещё не запускали — слой обязан быть точным нулём."""
    for bed in ap.AMBIENCE_VOCAB:
        layers = ps.ambience_layers(bed)
        assert layers == [] or len(layers) == len(ap.AMBIENCE_LAYER_SECONDS), \
            "частично записанные источники — хуже, чем никаких: слой звучал бы неполным"
    assert ps.build_ambience_track([{"start": 0.0, "end": 10.0, "bed": "wind_open", "seed": 1}],
                                    10.0, tempfile.mkdtemp()) is None


def test_mixer_is_fail_open(tmp_path):
    mix = str(tmp_path / "mix.wav")
    open(mix, "wb").close()
    out, detail = ps.add_ambience_bed(mix, None, mix, 10.0, str(tmp_path / "o.wav"))
    assert out == mix and detail is None


def test_report_is_written_even_when_nothing_plays(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "TEMP_FOLDER", str(tmp_path))
    blocks = [{"section": "BLOCK 1", "text": "разговор о цифрах и источниках", "words": 40}]
    mix = str(tmp_path / "mix.wav")
    open(mix, "wb").close()
    out = ps.run_ambience(mix, str(tmp_path), blocks, [0.0], 300.0, mix)
    assert out == mix
    report = json.load(open(tmp_path / "media_plan" / "ambience_plan.json", encoding="utf-8"))
    assert report["applied"] is False
    assert report["segments"][0]["reason"] == "low_confidence"


def test_segment_builder_uses_offset_and_drift():
    """Без сдвига вторая глава с той же атмосферой начиналась бы тем же
    звуком, и повтор был бы слышен именно как повтор."""
    src = inspect.getsource(ps._ambience_segment)
    assert "offset" in src and "atrim=" in src
    assert "AMBIENCE_DRIFT_SECONDS" in src and "sin(" in src
    assert "afade=t=in" in src and "afade=t=out" in src


def test_silence_segments_keep_the_track_in_sync():
    """Участок без атмосферы — явная тишина, а не пропуск: иначе всё, что
    после, поедет (тот же класс ошибки, что уже ловили у build_mood_timeline)."""
    src = inspect.getsource(ps.build_ambience_track)
    assert "_silent_segment" in src
    assert "anullsrc" in inspect.getsource(ps._silent_segment)


def test_generator_preview_does_not_arm_the_feature():
    """«Послушать демонстрацию» не должно незаметно включить слой в рендере."""
    src = open(os.path.join(REPO_ROOT, "scripts", "generate_ambience.py"), encoding="utf-8").read()
    assert "preview_only" in src
    assert "_preview" in src
