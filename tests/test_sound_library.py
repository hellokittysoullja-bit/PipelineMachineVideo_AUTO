# -*- coding: utf-8 -*-
"""Библиотека настоящих звуков — то, что проверяемо без сети и без моделей.

Сеть, CLAP и AST здесь не трогаются: живой отбор — отдельный прогон
(`python scripts/sound_library.py build`), его результат лежит в
assets/library/manifest.json с числами по каждому файлу. Тут — правила,
которые обязаны держаться независимо от того, что вернул поиск сегодня.
"""
import inspect
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import sound_library as sl  # noqa: E402


def test_license_gate_is_fail_closed():
    """Фильтр в URL запроса — экономия трафика, не гарантия. Решает поле
    результата, и только cc0 — тот же принцип, что у картинок Openverse."""
    assert sl.is_safe_license({"license": "cc0"})
    assert sl.is_safe_license({"license": "CC0"})
    for bad in ("by", "by-sa", "pdm", "by-nc", "", None):
        assert not sl.is_safe_license({"license": bad})
    assert not sl.is_safe_license({})


def test_every_spec_entry_is_complete():
    for kind, entries in sl.LIBRARY_SPEC.items():
        for name, spec in entries.items():
            assert spec["queries"], f"{kind}/{name}: нет запросов"
            assert spec["prompt"], f"{kind}/{name}: нет промпта для CLAP"
            assert spec.get("keep", 0) >= 1
            assert spec.get("min_sec", 0) >= 0


def test_ambience_beds_match_the_planner_vocabulary():
    """Каждый вид атмосферы, который умеет выбирать планировщик, обязан
    иметь рецепт поиска — иначе выбор кончался бы тишиной молча."""
    import ambience_plan as ap
    for bed in ap.AMBIENCE_VOCAB:
        assert bed in sl.LIBRARY_SPEC["ambience"], f"нет рецепта поиска для «{bed}»"


def test_preview_url_swaps_quality_only_for_freesound():
    hq = "https://cdn.freesound.org/previews/172/172666_2213158-hq.mp3"
    assert sl.preview_url(hq, "lq").endswith("-lq.mp3")
    assert sl.preview_url(sl.preview_url(hq, "lq"), "hq") == hq
    other = "https://upload.wikimedia.org/wikipedia/commons/a/ab/x.ogg"
    assert sl.preview_url(other, "lq") == other


def test_clipping_share_counts_full_scale_samples():
    clean = np.sin(np.linspace(0, 200, 48000)).astype(np.float32) * 0.5
    assert sl.clipping_share(clean) == 0.0
    clipped = clean.copy()
    clipped[:100] = 1.0
    assert sl.clipping_share(clipped) > sl.CLIP_SAMPLE_SHARE


def test_hum_detector_flags_mains_tone_and_ignores_broadband():
    sr = 48000
    t = np.arange(sr * 8) / sr
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.05, t.size).astype(np.float32)
    assert sl.hum_prominence_db(noise, sr) < sl.HUM_PROMINENCE_DB
    hum = (noise + 0.3 * np.sin(2 * np.pi * 50.0 * t)).astype(np.float32)
    assert sl.hum_prominence_db(hum, sr) > sl.HUM_PROMINENCE_DB


def test_negative_prompts_cover_the_real_failure_modes():
    joined = " ".join(sl.NEGATIVE_PROMPTS).lower()
    for must in ("speech", "music", "engine", "hum", "distortion"):
        assert must in joined


def test_thresholds_are_conservative_in_the_right_direction():
    """Маржа строго положительная: положительный промпт обязан ПЕРЕБИВАТЬ
    худшую ловушку, а не дотягиваться до неё."""
    assert sl.CLAP_MIN_MARGIN > 0
    assert 0 < sl.AST_VETO["Speech"] < 0.5
    assert sl.AMBIENCE_PEAK_DBFS < sl.SFX_PEAK_DBFS < 0


def test_manifest_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(sl, "LIBRARY_ROOT", str(tmp_path))
    monkeypatch.setattr(sl, "MANIFEST_PATH", str(tmp_path / "manifest.json"))
    assert sl.load_manifest() == {"items": {}}
    m = {"items": {"assets/library/sfx/x/a.flac": {"license": "cc0"}}}
    sl.save_manifest(m)
    assert sl.load_manifest() == m


def test_library_files_lists_only_flac(tmp_path, monkeypatch):
    monkeypatch.setattr(sl, "LIBRARY_ROOT", str(tmp_path))
    d = tmp_path / "sfx" / "plate_tick"
    d.mkdir(parents=True)
    (d / "b.flac").write_bytes(b"x")
    (d / "a.flac").write_bytes(b"x")
    (d / "notes.txt").write_bytes(b"x")
    assert [os.path.basename(p) for p in sl.library_files("sfx", "plate_tick")] == ["a.flac", "b.flac"]
    assert sl.library_files("sfx", "missing") == []


def test_judge_applies_thresholds_on_cached_measurements():
    """Пороги живут в judge(): перенастройка не гоняет модели заново."""
    spec = {"prompt": "wind", "min_sec": 45, "_name": "wind_open"}
    base = {"duration": 100.0, "clipping_share": 0.0, "hum_db": 3.0, "silence_share": 0.01,
            "lufs": -30.0, "lra": 8.0, "true_peak": -6.0,
            "clap_rows": [[0.25, 0.10, 0.05], [0.24, 0.12, 0.04]],
            "neg_names": ["speech", "music"], "ast": {"Speech": 0.01, "Music": 0.02},
            "kind_winner": "wind_open", "kind_gap": 0.1, "kind_scores": {"wind_open": 0.25}}
    ok = sl.judge(base, "ambience", spec)
    assert ok["reasons"] == [] and ok["clap_margin"] == pytest.approx(0.12, abs=1e-3)
    assert ok["clap_worst_neg"] == "speech"
    loud = sl.judge(dict(base, lra=sl.AMB_MAX_LRA + 1), "ambience", spec)
    assert "too_dynamic" in loud["reasons"]
    talk = sl.judge(dict(base, ast={"Speech": 0.9, "Music": 0.0}), "ambience", spec)
    assert "ast_speech" in talk["reasons"]
    lose = sl.judge(dict(base, clap_rows=[[0.10, 0.20, 0.05]]), "ambience", spec)
    assert "clap_negative_wins" in lose["reasons"]
    short = sl.judge(dict(base, duration=10.0), "ambience", spec)
    assert short["reasons"] == ["duration_short"]


def test_title_block_catches_the_real_misses():
    """Реальные промахи первого прогона: колокольчики, прибой, дверь."""
    for title in ("WindChimes1.wav", "Big waves breaking and splashing", "Door_and_whistling_wind.mp3"):
        assert sl.title_blocked("wind_open", title), title
    assert sl.title_blocked("wind_open", "Wind blowing across open marshland") is None


def test_ambience_import_constants_are_sane():
    """Три дефекта, найденных замером на 29 принятых записях: шов петли,
    моно в двух каналах, НЧ-рокот. Константы обработки обязаны остаться
    в рабочем диапазоне."""
    assert 1.0 <= sl.AMBIENCE_LOOP_XFADE_SEC <= 6.0
    assert 40.0 <= sl.AMBIENCE_HIGHPASS_HZ <= 100.0, "выше 100 Гц срежет низ атмосферы"
    assert sl.AMBIENCE_WIDEN_DELAY_SEC > 0.05, "задержка порядка Хааса дала бы гребёнку"
    assert 0.9 < sl.MONO_CORRELATION <= 1.0


def test_channel_correlation_flags_duplicated_mono(tmp_path):
    import subprocess
    mono, stereo = str(tmp_path / "m.wav"), str(tmp_path / "s.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    "anoisesrc=d=3:c=pink:r=48000", "-ac", "2", mono], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "anoisesrc=d=3:c=pink:r=48000:seed=1",
                    "-f", "lavfi", "-i", "anoisesrc=d=3:c=pink:r=48000:seed=2",
                    "-filter_complex", "[0:a][1:a]join=inputs=2:channel_layout=stereo",
                    stereo], check=True)
    assert sl.channel_correlation(mono) >= sl.MONO_CORRELATION
    assert sl.channel_correlation(stereo) < sl.MONO_CORRELATION


def test_lf_share_separates_rumble_from_clean():
    sr = 48000
    t = np.arange(sr * 8) / sr
    rng = np.random.default_rng(3)
    clean = rng.normal(0, 0.1, t.size).astype(np.float32)
    assert sl.lf_share(clean) < 0.05
    rumble = (clean + 2.0 * np.sin(2 * np.pi * 18.0 * t)).astype(np.float32)
    assert sl.lf_share(rumble) > 0.5


def test_kind_decoys_cover_the_acoustic_neighbours():
    """Приманки — то, с чем виды реально путаются по звуку, а не по слову."""
    joined = " ".join(sl.KIND_DECOYS.values()).lower()
    assert "waves" in joined and "traffic" in joined


def test_quiet_kinds_are_not_killed_by_an_absolute_score():
    """Абсолютный порог положительного скора снят как гейт: сырой косинус
    несопоставим между разными текстами, и «тихий гул пустого зала» давал
    0.04-0.05 там, где огонь даёт 0.4 — порог отсекал вид целиком."""
    quiet = {"duration": 100.0, "clipping_share": 0.0, "hum_db": 3.0, "silence_share": 0.01,
             "lufs": -40.0, "lra": 4.0, "clap_rows": [[0.05, 0.005]], "neg_names": ["speech"],
             "ast": {"Speech": 0.01}, "kind_winner": "stone_hall", "kind_gap": 0.02,
             "kind_scores": {"stone_hall": 0.05}}
    v = sl.judge(quiet, "ambience", {"prompt": "hall", "min_sec": 30, "_name": "stone_hall"})
    assert v["reasons"] == [], v["reasons"]


def test_kind_competition_is_a_build_gate_not_only_a_report():
    lost = {"duration": 100.0, "clipping_share": 0.0, "hum_db": 3.0, "silence_share": 0.01,
            "lufs": -30.0, "lra": 8.0, "clap_rows": [[0.30, 0.05]], "neg_names": ["speech"],
            "ast": {"Speech": 0.01}, "kind_winner": "surf", "kind_gap": 0.09,
            "kind_scores": {"surf": 0.34, "wind_open": 0.25}}
    v = sl.judge(lost, "ambience", {"prompt": "wind", "min_sec": 45, "_name": "wind_open"})
    assert "kind_lost_to_surf" in v["reasons"]


# --------------------------------------------------------- прослушивание
def test_audition_takes_only_debatable_reasons():
    """На прослушивание идут только СПОРНЫЕ отказы — те, что про «про то ли
    это». Запись с измеренным дефектом (сетевой гул, клиппинг, речь поверх
    сцены) туда не попадает: слушать там нечего, дефект уже измерен."""
    assert sl._debatable("clap_negative_wins")
    assert sl._debatable("kind_lost_to_surf")
    assert sl._debatable("kind_lost_to_forge_fire")
    for hard in ("mains_hum", "clipping", "too_much_silence", "too_dynamic",
                 "ast_speech", "ast_music", "title:street"):
        assert not sl._debatable(hard), hard


def test_rejected_records_carry_url_so_they_can_be_reheard():
    """Без url отклонённого кандидата физически нечем переслушать, а решение
    «гейт неправ» принимается только ушами — значит url обязателен."""
    src = inspect.getsource(sl.build_kind)
    calls = [c for c in src.split("rejected_log.append(")[1:]]
    assert len(calls) == 2, "веток записи отказа должно быть две: по названию и по измерениям"
    for c in calls:
        head = c.split("\n\n")[0]
        assert 'c["url"]' in head, head[:200]


def test_audition_reuses_the_real_import_chain():
    """Слушается то, что реально ушло бы в ролик (срез рокота, расширение
    моно, бесшовная петля), а не исходник со стока."""
    assert "import_file(" in inspect.getsource(sl.audition)


def test_rejected_log_accumulates_across_builds():
    """Пересборка одного вида не стирает причины отказа у остальных.
    Первая версия заводила пустой список на каждый запуск — и на вопрос
    «почему эта запись не прошла» ответа не оставалось нигде."""
    src = inspect.getsource(sl.main)
    assert "rejected_path" in src
    # старый журнал читается перед записью, а не перезаписывается с нуля
    assert src.index("json.load(f)") < src.index("json.dump(rejected")
    # и чистятся ровно те виды, что пересобираются
    assert "drop = {(k, n) for k, n in wanted}" in src


def test_verify_logs_what_it_deletes():
    """Удалять запись молча нельзя: проигрыш чужому виду — СПОРНЫЙ отказ
    (ветер по высокой траве и правда похож на прибой), а спорное решается
    ушами. Значит удалённое обязано попасть в журнал, откуда его достаёт
    audition."""
    src = inspect.getsource(sl.verify)
    assert "_log_rejection(" in src
    assert src.index("_log_rejection(") < src.index("os.remove(path)")
    assert 'f"kind_lost_to_{winner}"' in src


def test_rejection_log_does_not_clobber_other_entries():
    src = inspect.getsource(sl._log_rejection)
    assert "rows.append(entry)" in src
    assert "json.load(f)" in src


# ------------------------------------------------ расширение моно и петля
def test_mono_widening_happens_after_the_loop_is_built():
    """Реальный измеренный баг. Первая версия ставила adelay ДО сборки
    петли: правый канал получал 2.3 с цифровой тишины в начале, файл
    становился длиннее, а петля резалась по исходной длине — на каждом
    обороте правый канал проваливался в дыру (щелчок на стыке 18.7 и 30.6
    против 1.0 у нормального стыка).
    """
    src = inspect.getsource(sl.import_file)
    assert "adelay" not in src, "adelay удлиняет файл и ломает петлю"
    assert src.index("_seamless_loop(") < src.index("_widen_mono("), \
        "расширение обязано идти ПОСЛЕ сборки петли"


def test_mono_widening_is_a_circular_shift_not_a_delay():
    """Круговой сдвиг длину не меняет и дыры не создаёт: повёрнутая
    бесшовная петля остаётся бесшовной."""
    src = inspect.getsource(sl._widen_mono)
    # только КОД: в докстринге adelay упомянут как описание исправленного бага
    body = src.split('"""')[-1]
    assert "concat=n=2" in body and "atrim=start=" in body
    assert "adelay" not in body
    # слишком короткую запись поворот не делает независимой — не трогаем
    assert "dur <= 3 * shift" in src


def test_seamless_loop_is_built_from_separate_files():
    """Реальный, измеренный баг: прежняя версия делала
    atrim -> acrossfade -> concat в ОДНОМ filter_complex, и acrossfade в
    такой схеме отдавал ПУСТОЙ поток — на выходе оставалось только тело,
    то есть бесшовная петля не собиралась НИ РАЗУ ни у одной записи.
    Код возврата ffmpeg при этом был нулевой; видно было только по
    длительности (174с вместо 177с).
    """
    src = inspect.getsource(sl._seamless_loop)
    body = src.split('"""')[-1]
    # кроссфейд считается двумя ОТДЕЛЬНЫМИ входами-файлами
    assert '"-i", tail, "-i", head' in body
    assert "asplit" not in body and "[0:a]atrim" not in body


def test_seamless_loop_refuses_an_empty_seam():
    """Пустой шов — это и был баг. Молчать про него нельзя."""
    body = inspect.getsource(sl._seamless_loop).split('"""')[-1]
    assert "if not probe_duration(seam):" in body
    assert body.index("probe_duration(seam)") < body.index("concat")


def test_import_falls_back_to_unlooped_audio_if_loop_fails():
    """Не собралась петля — запись всё равно идёт в библиотеку, просто без
    петли: пустой слот хуже слышимого стыка раз в три минуты."""
    src = inspect.getsource(sl.import_file)
    assert "looped = _seamless_loop(" in src
    assert "if looped:" in src


def test_source_sample_rate_is_gated():
    """Библиотека отдаётся в 48 кГц, значит исходник ниже 44.1 — апсемплинг:
    новой информации нет, а потолок слышен. Найдено замером: принятый
    переход главы пришёл с 16 кГц (потолок 8 кГц) и стоял рядом с семью
    вариантами на 44/48; ни один существующий гейт про это не спрашивал."""
    assert sl.MIN_SOURCE_SAMPLE_RATE == 44100
    src = inspect.getsource(sl.judge)
    assert "low_sample_rate" in src
    m = {"duration": 10.0, "clipping_share": 0.0, "source_sample_rate": 16000}
    spec = {"min_sec": 1, "max_sec": 100, "prompt": "x"}
    assert "low_sample_rate" in sl.judge(m, "sfx", spec)["reasons"]


def test_unknown_sample_rate_does_not_reject():
    """Не измерилось — не повод отказывать: тот же fail-open, что и везде,
    где измерение не получилось."""
    m = {"duration": 10.0, "clipping_share": 0.0, "source_sample_rate": 0,
         "clap_rows": [[0.5, 0.1]], "neg_names": ["шум"]}
    spec = {"min_sec": 1, "max_sec": 100, "prompt": "x"}
    assert "low_sample_rate" not in sl.judge(m, "sfx", spec)["reasons"]


def test_loop_length_comes_from_the_real_file_not_the_requested_trim():
    """ffmpeg отдаёт чуть меньше, чем просили (-t 77.662 -> 77.632), и хвост,
    отсчитанный от ЗАПРОШЕННОЙ длины, выходил короче xf. acrossfade с входом
    короче d молча не собирался, и запись оставалась без петли — на замере
    это было видно как стык 11.2 при норме около 1."""
    body = inspect.getsource(sl._seamless_loop).split('"""')[-1]
    assert "body = probe_duration(stage) or body" in body
    assert body.index("probe_duration(stage)") < body.index('f"lh_{h}.wav"')


def test_verify_applies_the_sample_rate_gate_to_all_kinds():
    """Гейт частоты обязан подействовать на УЖЕ принятое, а не только на
    будущий отбор: библиотека уже несла переход главы с исходника 16 кГц.
    И он не про атмосферу — он про любой вид."""
    src = inspect.getsource(sl.verify)
    assert src.index("MIN_SOURCE_SAMPLE_RATE") < src.index('it.get("kind") != "ambience"'), \
        "проверка частоты стоит ДО отсечки не-атмосферы, иначе эффекты её не проходят"
    assert "_log_rejection(" in src


def test_crossfade_duration_comes_from_the_extracted_pieces():
    """Третий заход на тот же класс бага, уже на третьем знаке: ffmpeg
    отдаёт 2.999583 там, где просили 3.000, и acrossfade с d БОЛЬШЕ входа
    снова молча отдаёт пустой поток. d берётся у реально извлечённых кусков
    и округляется ВНИЗ, чтобы гарантированно не превысить вход."""
    body = inspect.getsource(sl._seamless_loop).split('"""')[-1]
    assert "probe_duration(head)" in body and "probe_duration(tail)" in body
    assert "math.floor(d * 1000)" in body
    assert body.index("math.floor(d * 1000)") < body.index("[0:a][1:a]acrossfade")


def test_sfx_peak_is_measured_after_channel_conversion():
    """Реальный найденный перекос: ffmpeg при mono -> stereo применяет -3 дБ
    на канал, и эффект из моно выходил на 3 дБ тише объявленного пика — то
    есть варианты ОДНОГО эффекта звучали врозь случайным образом (замер:
    стерео-исходники ровно -10.0, моно -13.0..-13.2)."""
    src = inspect.getsource(sl.import_file)
    head = src.split('if kind != "ambience":')[1].split("lufs, _, _")[0]
    assert '"-ac", "2", conv' in head, "сначала приведение к стерео"
    assert head.index('"-ac", "2", conv') < head.index("volumedetect")


def test_sample_rate_parsing_survives_the_trailing_comma(monkeypatch):
    """Реальный молчаливый no-op: `-of csv=p=0` на одном поле печатает
    висячую запятую («16000,»), `int()` на ней падает, широкий except
    возвращал 0 — и гейт частоты не отбраковал даже тот файл на 16 кГц,
    ради которого писался. Прогон verify показал «удалено 0».
    """
    class R:
        stdout = "16000,\n"
    monkeypatch.setattr(sl, "_run", lambda cmd: R())
    assert sl.probe_sample_rate("x") == 16000


def test_sample_rate_parsing_returns_zero_when_absent(monkeypatch):
    class R:
        stdout = "\n"
    monkeypatch.setattr(sl, "_run", lambda cmd: R())
    assert sl.probe_sample_rate("x") == 0
