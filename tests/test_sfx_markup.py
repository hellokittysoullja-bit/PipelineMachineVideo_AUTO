# -*- coding: utf-8 -*-
"""Разметка звука в сценарии: [sfx:концепт] и [hush].

Смысл слоя — не разбирать готовый текст, а записать намерение в тот момент,
когда автор этот костёр в текст и вписывает. Обе автоматические схемы были
проверены живьём и отклонены с числами: русский текст главы в текстовую
башню CLAP дал ОДИН И ТОТ ЖЕ вид на всех шести главах эпизода, а мост
«многоязычный эмбеддинг -> английская метка» — 7 верных из 18 на одиночных
словах при случайном выборе 1 из 20. Разметка в сценарии даёт точность по
построению: угадывать нечего.
"""
import os
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import script_parser


def parse(text):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     encoding="utf-8") as f:
        f.write("=== METADATA ===\nTITLE: t\n\n=== HOOK ===\n" + text + "\n")
        path = f.name
    try:
        return script_parser.parse_blocks(path)
    finally:
        os.unlink(path)


def test_sfx_tag_records_name_and_word_anchor():
    b = parse("Кузнец бьёт молотом по раскалённой полосе[sfx:hammer], и это слышно.")
    hit = [x for x in b if x["sfx"]]
    assert len(hit) == 1
    assert hit[0]["sfx"][0]["name"] == "hammer"
    # якорь — сколько слов уже произнесено к моменту тега
    assert hit[0]["sfx"][0]["word_pos"] == 6


def test_sfx_tag_does_not_cut_the_edit():
    """Реальный дефект первой версии: тег разбивает строку на части, и если
    висела несъеденная пауза, следующий огрызок — хоть одна точка — уходил
    в ОТДЕЛЬНЫЙ блок, то есть получал собственный слот под картинку."""
    b = parse("Первая.[pause]Дождь идёт третьи сутки[sfx:rain].")
    assert len(b) == 2, [x["text"] for x in b]
    assert b[1]["text"].startswith("Дождь идёт третьи сутки")
    assert not any(x["text"].strip() in {".", ",", ""} for x in b)


def test_several_sounds_in_one_block_all_survive():
    b = parse("Костёр горит[sfx:fire], и дождь стучит по крыше[sfx:rain].")
    hit = [x for x in b if x["sfx"]][0]
    assert [s["name"] for s in hit["sfx"]] == ["fire", "rain"]
    assert hit["sfx"][0]["word_pos"] < hit["sfx"][1]["word_pos"]


def test_hush_marks_deliberate_silence():
    """[hush] — не «здесь нет звука», а «здесь тишина НУЖНА». Пустое место
    планировщик вправе заполнить, помеченную тишину — нет."""
    b = parse("Первая фраза.[pause]А теперь молчи.[hush][pause]Третья.")
    marked = [x for x in b if x["hush"]]
    assert len(marked) == 1
    assert "молчи" in marked[0]["text"]


def test_sfx_never_reaches_the_tts_text():
    """Тот же принцип, что у [stat:...] и [climax]: пайплайн-only маркер,
    человек копирует ЧИСТЫЙ текст в TTS."""
    b = parse("Костёр догорает[sfx:fire] всю ночь.")
    joined = " ".join(x["text"] for x in b)
    assert "sfx" not in joined and "[" not in joined and "hush" not in joined


def test_unknown_tags_still_warn(capsys):
    """[sfx:]/[hush] не должны заглушить предупреждение о запрещённых тегах
    (ЧАСТЬ 10: [long pause] ломает TTS артефактами)."""
    parse("Фраза[sfx:fire] и ещё[long pause] одна.")
    assert "long pause" in capsys.readouterr().out


@pytest.mark.parametrize("bad", ["[sfx:]", "[sfx: ]"])
def test_empty_sfx_name_is_ignored(bad):
    b = parse("Просто фраза" + bad + " дальше.")
    assert not any(x["sfx"] for x in b)


def test_existing_markup_not_regressed():
    b = parse("Первая.[pause]Вторая[stat:ЧИСЛО].[pause]Третья.[climax][pause]Четвёртая.")
    assert any(x["stat"] == "ЧИСЛО" for x in b)
    assert any(x["is_climax"] for x in b)
    assert all("sfx" in x and "hush" in x for x in b), "новые ключи есть у КАЖДОГО блока"


# ------------------------------------------------ планировщик объектов
import sfx_plan


def _assets(name):
    table = {"fire": ("/x/fire.flac", 8.0, sfx_plan.OBJECT_CLASS_BED),
             "hammer": ("/x/hammer.flac", 0.6, sfx_plan.OBJECT_CLASS_POINT)}
    return table.get(name)


def _episode(marks_at, hush_at=(), n=12, step=10.0, speech=8.0):
    blocks, starts, weights = [], [], []
    for i in range(n):
        b = {"text": "фраза " * 10, "words": 10, "section": "BLOCK 1",
             "sfx": marks_at.get(i, []), "hush": i in hush_at,
             "stat": None, "is_climax": False, "pause_after": 0.8}
        blocks.append(b)
        starts.append(i * step)
        weights.append(speech)
    return blocks, starts, weights


def test_object_sound_arrives_before_the_word_not_on_it():
    """Звук ровно на слове — буквальная иллюстрация речи (Mickey Mousing),
    самый узнаваемый признак любителя. Сначала слышишь, потом понимаешь."""
    blocks, starts, weights = _episode({3: [{"name": "hammer", "word_pos": 1}]})
    acc, _ = sfx_plan.plan_sfx_cues(blocks, starts, weights, 200.0,
                                    object_asset_for=_assets)
    cue = [c for c in acc if c["kind"] == "object"][0]
    # якорь = 30с + (1/10)*8с = 30.8с
    assert abs(cue["anchor"] - 30.8) < 0.01
    # звук ЗАКАНЧИВАЕТСЯ до слова, а не просто начинается раньше
    assert cue["time"] + cue["asset_dur"] <= cue["anchor"]
    # и стоит в РЕАЛЬНОЙ тишине: речь блока 2 кончилась на 28.0,
    # блок 3 начинается на 30.0
    gap = sfx_plan.speech_gap_before(3, starts, weights)
    assert gap[0] <= cue["time"]
    assert cue["time"] + cue["asset_dur"] <= gap[1]


def test_hush_block_is_protected_like_the_climax():
    """[hush] — не «пусто», а «нельзя». Защита тем же механизмом
    зарезервированных окон, что у кульминации."""
    blocks, starts, weights = _episode({4: [{"name": "fire", "word_pos": 1}]},
                                       hush_at=(4,))
    acc, drop = sfx_plan.plan_sfx_cues(blocks, starts, weights, 200.0,
                                       object_asset_for=_assets)
    assert not [c for c in acc if c["kind"] == "object"]
    assert any(c.get("reason") == "climax_window" for c in drop)


def test_missing_asset_is_silence_not_a_substitute():
    """«Похожий» звук под конкретным словом слышен как ошибка, тишина — нет."""
    blocks, starts, weights = _episode({2: [{"name": "нет_такого", "word_pos": 1}]})
    acc, drop = sfx_plan.plan_sfx_cues(blocks, starts, weights, 200.0,
                                       object_asset_for=_assets)
    assert not [c for c in acc if c["kind"] == "object"]
    assert any(c.get("reason") == "no_asset" for c in drop)


def test_object_density_is_stricter_than_service_effects():
    """Главный рычаг слоя — воздержание: ролик, где звучит каждое
    существительное, это озвученный словарь, а не кино."""
    marks = {i: [{"name": "hammer", "word_pos": 1}] for i in range(10)}
    # speech < step — иначе речь блока перекрывает начало следующего и
    # реальной тишины между ними не существует вообще
    blocks, starts, weights = _episode(marks, n=10, step=3.0, speech=2.0)
    acc, drop = sfx_plan.plan_sfx_cues(blocks, starts, weights, 200.0,
                                       object_asset_for=_assets)
    obj = [c for c in acc if c["kind"] == "object"]
    assert len(obj) < 10, "плотность обязана резаться"
    for a, b in zip(obj, obj[1:]):
        assert b["time"] - a["time"] >= sfx_plan.OBJECT_MIN_GAP_SEC - 1e-6
    assert any(c.get("reason", "").startswith("object_") for c in drop)


def test_bed_and_point_get_different_levels():
    """Подзвучник тише точечного удара — но проверяется это по ЦЕЛЕВОМУ
    РАЗРЫВУ, а не по усилению в дБ.

    Первая версия теста требовала OBJECT_BED_GAIN_DB < OBJECT_POINT_GAIN_DB
    и была НЕВЕРНА: у полевой записи фона громкость низкая относительно
    пика, и чтобы попасть в свой (более тихий) коридор, ей нужно БОЛЬШЕ
    усиления, чем резкому удару. Порядок чисел в дБ противоположен порядку
    громкостей — ровно то, из-за чего объявленные уровни и разъехались.
    """
    assert sfx_plan.OBJECT_BED_GAP_LU > sfx_plan.OBJECT_POINT_GAP_LU
    blocks, starts, weights = _episode({1: [{"name": "fire", "word_pos": 2}],
                                        6: [{"name": "hammer", "word_pos": 2}]})
    acc, _ = sfx_plan.plan_sfx_cues(blocks, starts, weights, 200.0,
                                    object_asset_for=_assets)
    by = {c["name"]: c for c in acc if c["kind"] == "object"}
    assert by["fire"]["cls"] == sfx_plan.OBJECT_CLASS_BED
    assert by["hammer"]["cls"] == sfx_plan.OBJECT_CLASS_POINT


def test_no_markup_means_no_object_layer_at_all():
    """Эпизод без разметки — байт-в-байт прежнее поведение."""
    blocks, starts, weights = _episode({})
    acc, drop = sfx_plan.plan_sfx_cues(blocks, starts, weights, 200.0,
                                       object_asset_for=_assets)
    assert not [c for c in acc if c["kind"] == "object"]
    assert not [c for c in drop if c["kind"] == "object"]


# --------------------------------------------------- сведение и резолвер
def test_bed_carries_its_own_envelope_not_the_mixer():
    """Микшер обязан остаться тупым исполнителем плана: длительность и
    фейды несёт САМ кюй. Иначе единственным способом проверить правило
    станет «отрендери ролик и послушай»."""
    blocks, starts, weights = _episode({2: [{"name": "fire", "word_pos": 2}]})
    acc, _ = sfx_plan.plan_sfx_cues(blocks, starts, weights, 200.0,
                                    object_asset_for=_assets)
    bed = [c for c in acc if c["cls"] == sfx_plan.OBJECT_CLASS_BED][0]
    assert bed["trim_sec"] == sfx_plan.OBJECT_BED_SEC
    assert 0 < bed["fade_in_sec"] <= sfx_plan.OBJECT_BED_FADE_IN_SEC
    assert 0 < bed["fade_out_sec"] <= sfx_plan.OBJECT_BED_FADE_OUT_SEC
    # фейды обязаны помещаться в саму длительность
    assert bed["fade_in_sec"] + bed["fade_out_sec"] <= bed["trim_sec"]


def test_point_gets_no_envelope():
    """Удар с полуторасекундным нарастанием смазан — точке фейды не нужны."""
    blocks, starts, weights = _episode({2: [{"name": "hammer", "word_pos": 2}]})
    acc, _ = sfx_plan.plan_sfx_cues(blocks, starts, weights, 200.0,
                                    object_asset_for=_assets)
    pt = [c for c in acc if c["cls"] == sfx_plan.OBJECT_CLASS_POINT][0]
    assert "trim_sec" not in pt and "fade_in_sec" not in pt


def test_bed_trim_never_exceeds_the_asset():
    """Иначе в хвосте окажется тишина с фейдом из ниоткуда."""
    short_bed = lambda n: (("/x/f.flac", 2.0, sfx_plan.OBJECT_CLASS_BED)
                           if n == "fire" else None)
    blocks, starts, weights = _episode({2: [{"name": "fire", "word_pos": 2}]})
    acc, _ = sfx_plan.plan_sfx_cues(blocks, starts, weights, 200.0,
                                    object_asset_for=short_bed)
    bed = [c for c in acc if c["kind"] == "object"][0]
    assert bed["trim_sec"] == 2.0


def test_mixer_applies_the_envelope_the_cue_asks_for():
    import inspect

    import pipeline_smart as ps

    src = inspect.getsource(ps.add_planned_sfx)
    for token in ('c.get("trim_sec")', "atrim=0:", "afade=t=in", "afade=t=out"):
        assert token in src, token


def test_resolver_classifies_by_concept_then_by_duration():
    """Словарь говорит ТОЛЬКО как звук ведёт себя во времени — упомянут ли
    предмет, уже сказал автор тегом. Поэтому список короткий и не растёт с
    нишей: незнакомый концепт классифицируется по длительности файла."""
    import pipeline_smart as ps

    assert "fire" in ps.OBJECT_BED_CONCEPTS
    assert "hammer" not in ps.OBJECT_BED_CONCEPTS
    assert ps.OBJECT_POINT_MAX_SEC > 0


def test_resolver_returns_none_for_unknown_concept():
    import pipeline_smart as ps

    assert ps.object_asset_for("определённо_нет_такого_звука") is None
    assert ps.object_asset_for("") is None


# ------------------------------------------------------------- уровни
def test_levels_are_targets_in_lu_not_declared_decibels():
    """Уровни задаются РАЗРЫВОМ с голосом и выводятся замером. Прямая
    причина, измеренная на реальных ассетах против голоса на -16 LUFS:
    объявленные -20 дБ давали точке 22.0 LU под голосом при цели 26-30
    (на 4 дБ громче — удар выскакивал бы поверх реплики), а -26 дБ давали
    подзвучнику 52.1 LU при цели 30-34 (на 18 дБ тише, не слышно).

    Обе ошибки от одной причины: нормировка библиотеки задаёт ПИК, а не
    громкость. Ровно на этом уже сгорела музыкальная подложка.
    """
    assert sfx_plan.OBJECT_POINT_GAP_LU == 28.0
    assert sfx_plan.OBJECT_BED_GAP_LU == 32.0
    # подзвучник обязан быть тише точки
    assert sfx_plan.OBJECT_BED_GAP_LU > sfx_plan.OBJECT_POINT_GAP_LU
    # и он не громче атмосферы места — это подзвучник под фразой, не сцена
    import ambience_plan
    assert sfx_plan.OBJECT_BED_GAP_LU >= ambience_plan.AMBIENCE_GAP_LU


def test_gain_is_measured_against_this_voice_and_this_asset():
    import inspect

    import pipeline_smart as ps

    src = inspect.getsource(ps.object_gain_db)
    assert "measure_max_momentary_lufs" in src or "_object_gain_cached" in src
    assert "fallback_constant" in src, "не измерилось — честная константа и предупреждение"
    # громче исходного ассета не поднимаем никогда
    assert sfx_plan.OBJECT_GAIN_MAX_DB <= 0.0


def test_momentary_not_integrated_for_transients():
    """Интеграл по 0.6-секундному удару занижает его в разы (замер: -42.0 I
    против -37.7 M), а ухо сравнивает транзиент с речью в момент удара."""
    import inspect

    import pipeline_smart as ps

    assert "M:" in inspect.getsource(ps.measure_max_momentary_lufs)


def test_planner_takes_the_measured_gain_and_does_not_invent_one():
    got = lambda n: ("/x/a.flac", 1.0, sfx_plan.OBJECT_CLASS_POINT, -26.3, "measured")
    blocks, starts, weights = _episode({2: [{"name": "hammer", "word_pos": 2}]})
    acc, _ = sfx_plan.plan_sfx_cues(blocks, starts, weights, 200.0, object_asset_for=got)
    cue = [c for c in acc if c["kind"] == "object"][0]
    assert cue["gain_db"] == -26.3
    assert cue["gain_source"] == "measured"


def test_planner_falls_back_when_resolver_gives_no_gain():
    """Старый контракт из трёх значений обязан продолжать работать."""
    got = lambda n: ("/x/a.flac", 1.0, sfx_plan.OBJECT_CLASS_POINT)
    blocks, starts, weights = _episode({2: [{"name": "hammer", "word_pos": 2}]})
    acc, _ = sfx_plan.plan_sfx_cues(blocks, starts, weights, 200.0, object_asset_for=got)
    cue = [c for c in acc if c["kind"] == "object"][0]
    assert cue["gain_db"] == sfx_plan.OBJECT_POINT_GAIN_DB


# ----------------------------------------- находки глубокого аудита 13.09
def test_object_without_alignment_is_dropped_not_guessed():
    """Замер: с alignment звук вставал на 27.2с, без него — на 20.0с, то
    есть на 7.2с раньше, поверх чужой фразы. Позиция слова внутри блока
    выводится из РЕАЛЬНОЙ длительности речи блока; нет её — позиции нет.

    Переход главы в точно такой же ситуации честно отказывается
    (`no_alignment`), объект угадывал. Одна и та же нехватка данных
    обрабатывалась двумя разными способами — это и есть дефект.
    """
    blocks, starts, weights = _episode({2: [{"name": "hammer", "word_pos": 1}]})
    acc, drop = sfx_plan.plan_sfx_cues(blocks, starts, None, 200.0,
                                       object_asset_for=_assets)
    assert not [c for c in acc if c["kind"] == "object"]
    reasons = {c.get("reason") for c in drop if c.get("kind") == "object"}
    assert reasons == {"no_alignment"}
    # с alignment тот же кюй по-прежнему принимается — отказ не стал глухим
    acc2, _ = sfx_plan.plan_sfx_cues(blocks, starts, weights, 200.0,
                                     object_asset_for=_assets)
    assert [c for c in acc2 if c["kind"] == "object"]


def test_object_and_chapter_refuse_the_same_way_without_alignment():
    """Симметрия — сам инвариант, а не следствие: разные ответы на одну и ту
    же нехватку данных и были корнем находки."""
    blocks, starts, _ = _episode({6: [{"name": "hammer", "word_pos": 1}]})
    for i, b in enumerate(blocks):
        b["section"] = "BLOCK %d" % (i // 4 + 1)
    assert sfx_plan.chapter_boundaries(blocks), "граница глав должна быть"
    acc, drop = sfx_plan.plan_sfx_cues(blocks, starts, None, 200.0,
                                       chapter_variants=(("/x/t.flac", 0.4),),
                                       object_asset_for=_assets)
    kinds = {c.get("kind") for c in acc}
    assert "object" not in kinds and "chapter" not in kinds
    for kind in ("object", "chapter"):
        same = [c for c in drop if c.get("kind") == kind]
        assert same, kind
        assert {c.get("reason") for c in same} == {"no_alignment"}


def test_hush_protection_widens_without_alignment_never_narrows():
    """Без реальной длительности речи окно [hush] строится по расстоянию до
    следующего блока — оно ШИРЕ, а не уже. Направление важно: защита,
    ошибающаяся в сторону «тише», безопасна; в сторону «звук пролез» — нет.
    """
    blocks, starts, weights = _episode({}, hush_at=(2,))
    wide = sfx_plan.hush_windows(blocks, starts, None)
    exact = sfx_plan.hush_windows(blocks, starts, weights)
    assert wide and exact
    assert wide[0][0] == exact[0][0]
    assert wide[0][1] >= exact[0][1]


def test_concept_name_cannot_escape_its_library_folder():
    """Имя концепта приходит из текста сценария и подставляется в путь."""
    import pipeline_smart as ps

    for bad in ("../../etc/passwd", "..", ".", "a/b", "a\\b", "  ", ""):
        assert ps.object_asset_for(bad) is None, bad


def test_reserved_window_is_checked_against_the_whole_cue_not_its_start():
    """Замер 13.09: защищённые окна проверялись по ОДНОМУ моменту старта, а
    кюй звучит секундами. Шестисекундный подзвучник, начавшийся за 2.5с до
    блока `[hush]`, играл **3.55с внутри** тишины, которую сценарий
    потребовал явно — и во всех отчётах числился принятым по правилам.

    То же и с окном кульминации: там музыка проседает ради одного момента,
    а фон спокойно тянулся сквозь него. Правка может только ОТКЛОНИТЬ кюй,
    налезающий на защищённую тишину, и никогда не добавить новый — то есть
    односторонняя по построению.
    """
    blocks, starts, weights = _episode({2: [{"name": "fire", "word_pos": 10}]},
                                       hush_at=(3,))
    bed = lambda n, at=None: ("/x/fire.flac", 12.0, sfx_plan.OBJECT_CLASS_BED,
                              -8.0, "measured")
    acc, drop = sfx_plan.plan_sfx_cues(blocks, starts, weights, 200.0,
                                       object_asset_for=bed)
    assert not [c for c in acc if c["kind"] == "object"]
    assert any(c.get("reason") == "climax_window" for c in drop)

    # Тот же кюй ДАЛЬШЕ от защищённой тишины принимается — отказ не глухой
    blocks2, starts2, weights2 = _episode({0: [{"name": "fire", "word_pos": 1}]},
                                          hush_at=(5,))
    acc2, _ = sfx_plan.plan_sfx_cues(blocks2, starts2, weights2, 200.0,
                                     object_asset_for=bed)
    assert [c for c in acc2 if c["kind"] == "object"]


def test_cue_span_uses_the_trim_the_plan_already_computed():
    """Длительность берётся из того, что план посчитал (`trim_sec` у
    протяжённого, иначе длительность ассета), а не считается заново: вторая
    формула той же величины рано или поздно разойдётся с первой."""
    assert sfx_plan._cue_span({"time": 10.0, "trim_sec": 6.0, "asset_dur": 99.0}) \
        == (10.0, 16.0)
    assert sfx_plan._cue_span({"time": 10.0, "asset_dur": 0.4}) == (10.0, 10.4)
    assert sfx_plan._cue_span({"time": 10.0}) == (10.0, 10.0)
    assert sfx_plan._cue_span({"asset_dur": 1.0}) is None
