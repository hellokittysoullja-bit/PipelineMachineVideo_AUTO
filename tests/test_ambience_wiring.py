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
    assert "run_ambience(" in inspect.getsource(ps.build_episode_audio_layers)
    assert "build_episode_audio_layers(" in inspect.getsource(ps.main)


def test_ambience_is_not_ducked_and_not_dipped():
    src = inspect.getsource(ps.add_ambience_bed)
    assert "sidechaincompress" not in src
    assert "_climax_dip_expr" not in src


def test_ambience_goes_in_after_the_music_mix():
    """Порядок важен: слой не должен попасть в сайдчейн музыки."""
    src = inspect.getsource(ps.build_episode_audio_layers)
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


def test_sources_are_one_recording_or_full_synth_set():
    """Два допустимых состояния: ОДНА настоящая запись из библиотеки или
    ПОЛНЫЙ набор из трёх синтезированных слоёв. Частичный синтез хуже, чем
    никакого: слой звучал бы неполным, и заметить это можно было бы только
    ушами."""
    for bed in ap.AMBIENCE_VOCAB:
        layers = ps.ambience_layers(bed, seed=0)
        assert len(layers) in (0, 1, len(ap.AMBIENCE_LAYER_SECONDS))
        for path, dur in layers:
            assert os.path.exists(path) and dur > 0
        if len(layers) == 1:
            assert "/library/ambience/" in layers[0][0].replace(os.sep, "/")


def test_library_recording_varies_by_section_seed(tmp_path, monkeypatch):
    """Две главы с одной атмосферой получают РАЗНЫЕ записи одного места —
    выбор по seed участка среди отобранных."""
    files = [str(tmp_path / f"{n}.flac") for n in ("a", "b", "c")]
    monkeypatch.setattr(ps, "library_sounds", lambda kind, name: files)
    monkeypatch.setattr(ps, "get_media_duration", lambda p: 120.0)
    picks = {ps.ambience_layers("wind_open", seed=s)[0][0] for s in range(6)}
    assert picks == set(files)


def test_long_run_prefers_a_recording_that_does_not_loop_audibly(tmp_path, monkeypatch):
    """Прямое следствие снижения нижней границы приёма в библиотеку 45 -> 15
    (решение владельца 17.09): библиотечный путь крутит ОДИН файл через
    `-stream_loop -1`, то есть период повтора равен его длине. На участке 470
    секунд (замеренное покрытие эпизода 02) пятнадцатисекундная запись
    повторяется ОДИН В ОДИН 31 раз. Выбор обязан учитывать длину участка."""
    short, long_ = str(tmp_path / "s.flac"), str(tmp_path / "l.flac")
    durs = {short: 15.0, long_: 180.0}
    monkeypatch.setattr(ps, "library_sounds", lambda kind, name: [short, long_])
    monkeypatch.setattr(ps, "get_media_duration", lambda p: durs[p])

    # Длинный участок: короткая запись не рассматривается ни при одном seed.
    picks = {ps.ambience_layers("wind_open", seed=s, duration=470.0)[0][0]
             for s in range(6)}
    assert picks == {long_}, "на длинном участке выбрана слышимо зацикленная запись"

    # Короткий участок: обе годятся, разнообразие по seed сохраняется.
    picks = {ps.ambience_layers("wind_open", seed=s, duration=45.0)[0][0]
             for s in range(6)}
    assert picks == {short, long_}


def test_duration_preference_can_never_empty_the_slot(tmp_path, monkeypatch):
    """Правило одностороннее: нет ни одной достаточно длинной записи — берётся
    самая длинная из имеющихся, а не тишина. Участок без фона по причине
    «все записи короткие» был бы регрессом, а не улучшением."""
    a, b = str(tmp_path / "a.flac"), str(tmp_path / "b.flac")
    durs = {a: 15.0, b: 22.0}
    monkeypatch.setattr(ps, "library_sounds", lambda kind, name: [a, b])
    monkeypatch.setattr(ps, "get_media_duration", lambda p: durs[p])
    layers = ps.ambience_layers("wind_open", seed=0, duration=900.0)
    assert layers and layers[0][0] == b


def test_without_duration_the_choice_is_unchanged(tmp_path, monkeypatch):
    """duration=None — прежнее поведение байт-в-байт: тот же выбор по seed
    среди всех записей. Старые вызовы (build_ambience_track, тесты) ничего
    не теряют."""
    files = [str(tmp_path / f"{n}.flac") for n in ("a", "b", "c")]
    durs = {files[0]: 15.0, files[1]: 60.0, files[2]: 200.0}
    monkeypatch.setattr(ps, "library_sounds", lambda kind, name: files)
    monkeypatch.setattr(ps, "get_media_duration", lambda p: durs[p])
    assert {ps.ambience_layers("wind_open", seed=s)[0][0] for s in range(6)} == set(files)


def test_no_sources_means_exact_no_op(monkeypatch):
    """Пока генератор не запускали, слой обязан быть точным нулём — именно
    отсутствие источников и есть выключатель (см. --preview у генератора)."""
    monkeypatch.setattr(ps, "ambience_layers", lambda bed, seed=0: [])
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


def test_ambience_sources_are_true_stereo_not_duplicated_mono():
    """Атмосфера обязана быть широкой, а не моно в два канала.

    Реальная ошибка первой версии, найденная сверкой с соседним
    generate_music_asset.py (тот строит два канала на разных seed): моно-фон
    схлопывает пространство в точку И садится ровно в центр, где идёт речь,
    маскируя её сильнее, чем такой же по громкости широкий фон.

    Точечные эффекты (удар кульминации, тик плашки) — наоборот, обязаны
    остаться моно: их место в центре.
    """
    amb = open(os.path.join(REPO_ROOT, "scripts", "generate_ambience.py"), encoding="utf-8").read()
    assert "def make_layer_stereo" in amb
    assert "np.stack([mono, mono]" not in amb.replace(" ", "")
    assert "_write_flac((left, right)" in amb

    for point_fx in ("generate_sfx_pack.py", "generate_reveal_sfx.py"):
        src = open(os.path.join(REPO_ROOT, "scripts", point_fx), encoding="utf-8").read()
        assert "np.stack([samples_mono, samples_mono]" in src, \
            f"{point_fx}: точечный эффект должен остаться по центру"
