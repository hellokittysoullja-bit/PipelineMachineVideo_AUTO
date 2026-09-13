# -*- coding: utf-8 -*-
"""Планировщик эффектов реально подключён к сборке звука — и безопасен,
пока ассетов нет.

Тот же класс пробела, что уже дважды ловили в этом проекте (Openverse,
Pixabay, сами reveal-акценты): код написан, ассеты лежат, вклад в готовый
ролик — ноль, потому что вызова нет. Здесь это проверяется прямо.

Вторая половина файла — про обратное: ассетов ПОКА нет в репозитории
(их создаёт scripts/generate_sfx_pack.py, и он требует numpy+scipy+ffmpeg),
и до их появления весь слой обязан быть точным no-op, а не источником
ошибок на каждом рендере.
"""
import inspect
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]

import feature_flags as ff  # noqa: E402
import pipeline_smart as ps  # noqa: E402
import sfx_plan  # noqa: E402


def test_flag_is_in_the_registry():
    assert ff.FLAGS["SFX_DIRECTOR"].default == "1"


def test_director_is_actually_called_from_the_render():
    """Цепочка слоёв вынесена в build_episode_audio_layers() — чтобы
    предпросмотр звука собирал ТОТ ЖЕ звук, а не свою копию. Проверяем оба
    звена: планировщик внутри цепочки, цепочка внутри рендера."""
    assert "run_sfx_director(" in inspect.getsource(ps.build_episode_audio_layers), \
        "планировщик написан, но не вызывается из цепочки"
    assert "build_episode_audio_layers(" in inspect.getsource(ps.main), \
        "цепочка написана, но не вызывается из рендера"


def test_plate_cues_and_typewriter_share_one_scale_conversion():
    """Оба звука плашки пересчитывают момент ОДНОЙ функцией.

    Реальная ошибка, найденная у себя при проверке (13.09): первая версия
    тика брала сырой stat_delay, а он посчитан от ДЛИТЕЛЬНОСТИ КЛИПА (в неё
    входит бюджет xfade), тогда как звук живёт на шкале sub_starts —
    той же, что субтитры. Машинка это уже учитывала инлайн, тик — нет:
    тик поехал бы тем сильнее, чем длиннее блок. Отдельная формула для
    звука это готовый рассинхрон, ровно поэтому рядом уже стоит общая
    typewriter_reveal_timing().
    """
    src = inspect.getsource(ps.plan_stat_sound_cues)
    assert src.count("stat_reveal_moment(") == 2, "обе ветки обязаны звать общий пересчёт"
    assert "sub_starts[i] + stat_reveal_moment(" in src


def test_scale_conversion_clamps_into_the_block():
    """Момент не выходит за пределы блока и не уходит в минус."""
    assert ps.stat_reveal_moment(99.0, 5.0, 1.2) == pytest.approx(3.8)
    assert ps.stat_reveal_moment(1.0, 5.0, 1.2) == pytest.approx(1.0)
    assert ps.stat_reveal_moment(2.0, 0.5, 1.2) == 0.0


def test_typewriter_plates_do_not_also_get_a_tick():
    """У варианта с машинкой свой звук — тик поверх был бы сдвоенным."""
    src = inspect.getsource(ps.plan_stat_sound_cues)
    i_type = src.index("if stat_variant % 5 == 4:")
    i_else = src.index("else:", i_type)
    assert i_else > i_type, "тик обязан быть ИНАЧЕ-веткой к машинке, а не независимой"


def test_chapter_sound_needs_the_phrase_locked_timeline():
    """real_weights уходят в планировщик только вместе с онсетами.

    Иначе конец речи (реальная шкала) складывался бы с оценочным стартом
    блока — «пауза», которой нет в аудио, и звук поверх слова.
    """
    src = inspect.getsource(ps.build_episode_audio_layers)
    assert "real_weights if phrase_locked else None" in src


def test_no_assets_means_no_cues_and_no_crash():
    """Ассетов ещё нет — слой обязан быть точным no-op."""
    variants = ps.chapter_sfx_variants()
    for path, dur in variants:
        assert os.path.exists(path) and dur > 0
    if not variants:
        blocks = [{"section": s} for s in ("HOOK", "BLOCK 1")]
        accepted, dropped = sfx_plan.plan_sfx_cues(
            blocks, [0.0, 5.0], [4.0, 4.0], 9.0, chapter_variants=variants)
        assert accepted == []
        assert [d["reason"] for d in dropped] == ["gap_too_short"]


def test_mixer_is_fail_open(tmp_path):
    """Пустой план и выключенный флаг возвращают исходный микс как есть."""
    src = str(tmp_path / "mix.wav")
    open(src, "wb").close()
    assert ps.add_planned_sfx(src, [], 10.0, str(tmp_path / "out.wav")) == src
    cues = [{"time": 1.0, "asset": str(tmp_path / "nope.flac"), "gain_db": -12.0}]
    assert ps.add_planned_sfx(src, cues, 10.0, str(tmp_path / "out.wav")) == src


def test_report_is_written_even_when_nothing_is_placed(tmp_path, monkeypatch):
    """«Почему здесь тихо» обязано быть фактом в файле, а не догадкой."""
    import json
    monkeypatch.setattr(ps, "TEMP_FOLDER", str(tmp_path))
    blocks = [{"section": "HOOK"}, {"section": "BLOCK 1"}]
    mix = str(tmp_path / "mix.wav")
    open(mix, "wb").close()
    out = ps.run_sfx_director(mix, str(tmp_path), blocks, [0.0, 5.0], None, 9.0)
    assert out == mix
    report = json.load(open(tmp_path / "media_plan" / "sfx_plan.json", encoding="utf-8"))
    assert report["summary"]["dropped_by_reason"] == {"no_alignment": 1}
    assert report["gains_db"]["chapter"] == ps.SFX_CHAPTER_GAIN_DB


def test_gains_sit_below_the_programme():
    """Эффект — акцент под речью, а не второй голос.

    Программа мастерится в -14 LUFS; ассеты нормированы к -10 dBFS по пику
    (PEAK_DBFS генератора), значит без ослабления акцент звучал бы вровень
    с голосом. Тик обязан быть тише перехода — это отметка, не удар.
    """
    assert ps.SFX_CHAPTER_GAIN_DB <= -10.0
    assert ps.SFX_PLATE_GAIN_DB < ps.SFX_CHAPTER_GAIN_DB


def test_filter_graph_is_well_formed(tmp_path, monkeypatch):
    """Фильтр-граф проверяется разбором, потому что ffmpeg здесь нет.

    Граф собирается конкатенацией строк, и его нельзя «почти» собрать: одна
    несовпавшая метка или неверное число входов у amix — и весь звук ролика
    молча уходит на fail-open путь, то есть эффектов нет, а рендер зелёный.
    """
    captured = {}

    class R:
        returncode = 0
        stderr = ""

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        return R()

    monkeypatch.setattr(ps.subprocess, "run", fake_run)
    assets = []
    for i in range(3):
        a = tmp_path / f"a{i}.flac"
        a.write_bytes(b"x")
        assets.append(str(a))
    mix = str(tmp_path / "mix.wav")
    open(mix, "wb").close()
    cues = [{"time": 1.5, "asset": assets[0], "gain_db": -12.0},
            {"time": 10.25, "asset": assets[1], "gain_db": -16.0},
            {"time": 30.0, "asset": assets[2], "gain_db": -16.0}]
    out = ps.add_planned_sfx(mix, cues, 60.0, str(tmp_path / "out.wav"))
    assert out == str(tmp_path / "out.wav")

    cmd = captured["cmd"]
    graph = cmd[cmd.index("-filter_complex") + 1]
    # каждый ассет подан отдельным входом, плюс сам микс нулевым
    assert cmd.count("-i") == len(cues) + 1
    assert "amix=inputs=4" in graph, "число входов amix обязано совпадать с числом дорожек"
    # задержка в миллисекундах, а не в секундах — 10.25с это 10250мс
    assert "adelay=10250|10250" in graph
    assert "volume=-12.0dB" in graph and "volume=-16.0dB" in graph
    # все объявленные метки реально потребляются amix, ни одна не висит
    declared = {f"[px{i}]" for i in range(1, len(cues) + 1)}
    consumed = graph.split("amix=")[0].rsplit(";", 1)[-1]
    assert all(lbl in consumed for lbl in declared)
    assert "[0:a]" in consumed
    assert cmd[-1].endswith("out.wav") and "-map" in cmd
