"""Юнит-тесты чистой логики (без FFmpeg): парсинг сценария, тайминг блоков,
выбор без повторов, подбор тематического запроса, счётчик слов.
Запуск: .venv/bin/python -m pytest tests/ -v
"""
import os
import re
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

# pipeline_smart на импорте читает VIDEO_FOLDER из sys.argv[1] (ищет audio.mp3,
# грузит themes.json) — под pytest sys.argv это параметры pytest, не путь к
# видео, поэтому подменяем на существующий temp-каталог ДО импорта, иначе
# find_audio() падает на os.listdir() несуществующей директории.
sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]

import pipeline_smart          # noqa: E402
import wordcount                # noqa: E402
import stock_fetch_multisource  # noqa: E402
import fix_pauses               # noqa: E402


def write_script(tmp_path, body):
    p = tmp_path / "script.txt"
    p.write_text(body, encoding="utf-8")
    return str(p)


# ---------- parse_blocks ----------

def test_parse_blocks_basic_sections(tmp_path):
    text = (
        "=== HOOK === Первая фраза хука.[pause]Вторая фраза.\n"
        "=== BLOCK 1: Название === Текст блока один.[pause]Ещё текст.\n"
        "=== FINAL === Финальная фраза.\n"
    )
    blocks = pipeline_smart.parse_blocks(write_script(tmp_path, text))
    sections = [b["section"] for b in blocks]
    assert sections[0] == "HOOK"
    assert any(s.startswith("BLOCK 1") for s in sections)
    assert sections[-1] == "FINAL"


def test_parse_blocks_ignores_service_sections(tmp_path):
    text = (
        "=== METADATA === TITLE: Тест\n"
        "=== HOOK === Хук текст.\n"
        "=== PEXELS QUERIES === HOOK: q1,q2\n"
    )
    blocks = pipeline_smart.parse_blocks(write_script(tmp_path, text))
    assert len(blocks) == 1
    assert blocks[0]["section"] == "HOOK"
    assert "METADATA" not in blocks[0]["text"]
    assert "PEXELS" not in blocks[0]["text"]


def test_parse_blocks_pause_splits_block(tmp_path):
    text = "=== HOOK === Фраза раз.[pause]Фраза два.[pause]Фраза три.\n"
    blocks = pipeline_smart.parse_blocks(write_script(tmp_path, text))
    assert len(blocks) == 3
    assert blocks[0]["pause_after"] == 0.8
    assert blocks[1]["pause_after"] == 0.8
    # у последнего блока нет паузы после
    assert blocks[2]["pause_after"] == 0.0


def test_parse_blocks_short_pause_duration(tmp_path):
    text = "=== HOOK === Раз.[short pause]Два.\n"
    blocks = pipeline_smart.parse_blocks(write_script(tmp_path, text))
    assert blocks[0]["pause_after"] == 0.4


def test_parse_blocks_stat_tag_does_not_split(tmp_path):
    # [stat:...] посреди фразы без соседнего [pause] НЕ должен резать блок
    # (регрессия: раньше stat-маркер сам по себе создавал границу блока).
    text = "=== HOOK === Меч весил [stat:15 КГ — МИФ] почти ничего на самом деле.\n"
    blocks = pipeline_smart.parse_blocks(write_script(tmp_path, text))
    assert len(blocks) == 1
    assert blocks[0]["stat"] == "15 КГ — МИФ"
    assert "STAT" not in blocks[0]["text"]


def test_parse_blocks_stat_attaches_to_correct_block(tmp_path):
    text = "=== HOOK === Первая фраза.[pause][stat:4 КГ]Вторая фраза с цифрой.\n"
    blocks = pipeline_smart.parse_blocks(write_script(tmp_path, text))
    assert len(blocks) == 2
    assert blocks[0]["stat"] is None
    assert blocks[1]["stat"] == "4 КГ"


def test_parse_blocks_no_section_markers_fallback(tmp_path):
    # Сценарий без === берётся как есть (одним блоком/фрагментами по паузам)
    text = "Просто текст без секций.[pause]Ещё кусок.\n"
    blocks = pipeline_smart.parse_blocks(write_script(tmp_path, text))
    assert len(blocks) == 2
    assert all(b["section"] == "BODY" for b in blocks)


# ---------- block_durations ----------

def test_block_durations_sum_matches_total():
    blocks = [
        {"text": "a", "words": 10, "pause_after": 0.0, "section": "HOOK", "stat": None},
        {"text": "b", "words": 20, "pause_after": 0.8, "section": "BODY", "stat": None},
        {"text": "c", "words": 15, "pause_after": 0.0, "section": "FINAL", "stat": None},
    ]
    total = 30.0
    durs = pipeline_smart.block_durations(blocks, total)
    assert len(durs) == 3
    assert abs(sum(durs) - total) < 1e-6
    assert all(d > 0 for d in durs)


def test_block_durations_respects_min_clip_floor():
    # Один блок из одного слова на длинном общем таймлайне не должен уйти
    # ниже MIN_CLIP после масштабирования.
    blocks = [
        {"text": "a", "words": 1, "pause_after": 0.0, "section": "BODY", "stat": None},
        {"text": "b", "words": 500, "pause_after": 0.0, "section": "BODY", "stat": None},
    ]
    durs = pipeline_smart.block_durations(blocks, 200.0)
    assert durs[0] >= pipeline_smart.MIN_CLIP - 1e-6


def test_block_durations_hook_cap_lower_than_body():
    # У HOOK-блока предел короче (HOOK_MAX_CLIP < MAX_CLIP). Финальный масштаб
    # растягивает сумму под total, поэтому пороги видны только когда total
    # уже равен сумме капов (scale=1) — иначе рескейл на единственном блоке
    # просто перекрывает cap. Добираем total маленьким филлер-блоком.
    mc, hc, bc = pipeline_smart.MIN_CLIP, pipeline_smart.HOOK_MAX_CLIP, pipeline_smart.MAX_CLIP

    def durations_with_cap(main_section, cap):
        blocks = [
            {"text": "a", "words": 5000, "pause_after": 0.0, "section": main_section, "stat": None},
            {"text": "b", "words": 1, "pause_after": 0.0, "section": "BODY", "stat": None},
        ]
        total = cap + mc  # ровно сумма ожидаемых капов -> scale == 1
        return pipeline_smart.block_durations(blocks, total)

    hd = durations_with_cap("HOOK", hc)[0]
    bd = durations_with_cap("BODY", bc)[0]
    assert abs(hd - hc) < 1e-6
    assert abs(bd - bc) < 1e-6
    assert hd < bd


# ---------- pick_no_repeat ----------

def test_pick_no_repeat_allows_moderate_repeats():
    history = []
    options = ["a", "b", "c"]
    out = [pipeline_smart.pick_no_repeat(history, "a", options, max_repeat=3) for _ in range(2)]
    assert out == ["a", "a"]


def test_pick_no_repeat_breaks_long_streak():
    history = []
    options = ["a", "b", "c"]
    picks = []
    for _ in range(6):
        picks.append(pipeline_smart.pick_no_repeat(history, "a", options, max_repeat=3))
    # после max_repeat подряд одинаковых кандидат обязан смениться хотя бы раз
    assert len(set(picks)) > 1


def test_pick_no_repeat_history_bounded():
    history = []
    options = ["a", "b"]
    for _ in range(20):
        pipeline_smart.pick_no_repeat(history, "a", options, max_repeat=3)
    # history не должен расти бесконечно (del history[:-(max_repeat+2)])
    assert len(history) <= 5


# ---------- resolve_queries / query_for ----------

def test_resolve_queries_direct_match(monkeypatch):
    monkeypatch.setattr(pipeline_smart, "THEMES", {"меч": "medieval sword close up"})
    blocks = [{"text": "Меч весил немало.", "words": 3, "pause_after": 0.0, "section": "BODY", "stat": None}]
    resolved = pipeline_smart.resolve_queries(blocks)
    assert resolved == ["medieval sword close up"]


def test_resolve_queries_inherits_from_same_section_neighbor(monkeypatch):
    monkeypatch.setattr(pipeline_smart, "THEMES", {"меч": "medieval sword close up"})
    blocks = [
        {"text": "Меч был длинным.", "words": 3, "pause_after": 0.0, "section": "BODY", "stat": None},
        {"text": "Береги себя.", "words": 2, "pause_after": 0.0, "section": "BODY", "stat": None},
    ]
    resolved = pipeline_smart.resolve_queries(blocks)
    assert resolved[1] == "medieval sword close up"


def test_resolve_queries_does_not_leak_across_sections(monkeypatch):
    monkeypatch.setattr(pipeline_smart, "THEMES", {"меч": "medieval sword close up"})
    blocks = [
        {"text": "Меч был длинным.", "words": 3, "pause_after": 0.0, "section": "HOOK", "stat": None},
        {"text": "Береги себя.", "words": 2, "pause_after": 0.0, "section": "FINAL", "stat": None},
    ]
    resolved = pipeline_smart.resolve_queries(blocks)
    assert resolved[1] in pipeline_smart.GENERIC_FALLBACKS


def test_resolve_queries_fallback_cycles(monkeypatch):
    monkeypatch.setattr(pipeline_smart, "THEMES", {})
    blocks = [
        {"text": f"Абстрактная фраза {i}.", "words": 2, "pause_after": 0.0, "section": f"BODY{i}", "stat": None}
        for i in range(len(pipeline_smart.GENERIC_FALLBACKS) + 1)
    ]
    resolved = pipeline_smart.resolve_queries(blocks)
    assert resolved[0] == pipeline_smart.GENERIC_FALLBACKS[0]
    assert resolved[len(pipeline_smart.GENERIC_FALLBACKS)] == pipeline_smart.GENERIC_FALLBACKS[0]


# ---------- wordcount.py ----------

def test_clean_words_strips_tags_and_markers():
    assert wordcount.clean_words("Раз два [pause] три === четыре ===") == 4


def test_clean_words_empty_string():
    assert wordcount.clean_words("") == 0


def test_count_words_only_voiced_sections(tmp_path):
    text = (
        "=== METADATA === TITLE: игнорируется полностью тут\n"
        "=== HOOK === Раз два три.\n"
        "=== BLOCK 1: Тест === Четыре пять.[pause]Шесть.\n"
        "=== PEXELS QUERIES === HOOK: не считается вообще\n"
        "=== FINAL === Семь.\n"
    )
    p = tmp_path / "script.txt"
    p.write_text(text, encoding="utf-8")
    assert wordcount.count_words(str(p)) == 7


def test_count_words_missing_sections_is_zero(tmp_path):
    p = tmp_path / "script.txt"
    p.write_text("=== METADATA === TITLE: x\n=== IMAGE PROMPTS === HOOK_1: y\n", encoding="utf-8")
    assert wordcount.count_words(str(p)) == 0


# ---------- stock_fetch_multisource.build_query / load_themes ----------

def test_build_query_matches_keyword():
    themes = {"меч": "medieval sword close up"}
    assert stock_fetch_multisource.build_query("Тяжёлый меч в руке", themes) == "medieval sword close up"


def test_build_query_default_when_no_match():
    themes = {"меч": "medieval sword close up"}
    assert stock_fetch_multisource.build_query("Что-то совсем другое", themes) == \
        stock_fetch_multisource.DEFAULT_QUERY


def test_load_themes_merges_channel_dictionary_without_episode_file(tmp_path):
    # Регрессия: раньше load_themes() читал ТОЛЬКО media_plan/themes.json
    # эпизода — на свежем эпизоде без него (типичный случай: канальный
    # словарь предполагается общим, эпизодный — необязательной добавкой)
    # build_query() почти на каждом слоте падал в DEFAULT_QUERY, потому что
    # канальный channel_themes.json вообще не подключался.
    merged = stock_fetch_multisource.load_themes(str(tmp_path))
    assert "меч" in merged


def test_load_themes_episode_overrides_channel(tmp_path):
    media_plan = tmp_path / "media_plan"
    media_plan.mkdir()
    (media_plan / "themes.json").write_text('{"меч": "custom episode sword"}', encoding="utf-8")
    merged = stock_fetch_multisource.load_themes(str(tmp_path))
    assert merged["меч"] == "custom episode sword"
    assert "доспех" in merged   # канальный словарь по-прежнему подключён рядом с оверрайдом


def test_build_query_rotates_list_values_across_calls():
    themes = {"меч": ["a sword", "b sword", "c sword"]}
    counts = {}
    picks = [stock_fetch_multisource.build_query("Меч был длинным", themes, counts) for _ in range(4)]
    assert picks == ["a sword", "b sword", "c sword", "a sword"]


def test_build_query_word_boundary_not_mid_word():
    # Левая граница обязана быть началом слова (тот же \b-паттерн, что уже
    # использует query_for() в pipeline_smart.py) — "меч" не должен
    # матчиться, оказавшись подстрокой ВНУТРИ слова, а не в его начале.
    themes = {"меч": "medieval sword close up"}
    # "отмечен" содержит "меч" как подстроку (от-МЕЧ-ен), но НЕ в начале слова.
    assert stock_fetch_multisource.build_query("Этот день отмечен в летописи", themes) == \
        stock_fetch_multisource.DEFAULT_QUERY


# ---------- pipeline_smart._scene_bias / film_look (контент-осознанный грейд) ----------

def test_scene_bias_neutral_without_signal():
    assert pipeline_smart._scene_bias(None, None) == (0.0, 0.0, 0.0)


def test_scene_bias_warm_source_positive_warm_bias():
    warm, _, _ = pipeline_smart._scene_bias(None, (0.55, 0.45, 0.30))
    cool, _, _ = pipeline_smart._scene_bias(None, (0.30, 0.33, 0.42))
    assert warm > 0
    assert cool < 0


def test_scene_bias_saturated_vs_near_mono_chroma():
    _, chroma_vivid, _ = pipeline_smart._scene_bias(None, (0.80, 0.20, 0.15))
    _, chroma_mono, _ = pipeline_smart._scene_bias(None, (0.5, 0.5, 0.5))
    assert chroma_vivid > chroma_mono


def test_scene_bias_dark_vs_bright_key():
    _, _, key_dark = pipeline_smart._scene_bias((0.02, 0.55), None)
    _, _, key_bright = pipeline_smart._scene_bias((0.15, 0.92), None)
    assert key_dark < 0
    assert key_bright > 0


def test_film_look_warm_source_gets_weaker_warm_push_than_cool():
    warm_out = pipeline_smart.film_look(123456789, section="BODY",
                                         levels=(0.05, 0.9), wb=(0.55, 0.45, 0.30))
    cool_out = pipeline_smart.film_look(123456789, section="BODY",
                                         levels=(0.05, 0.9), wb=(0.30, 0.33, 0.42))

    def bs_value(vf):
        m = re.search(r'colorbalance=rs=[^:]+:bs=([\-\d.]+):', vf)
        return float(m.group(1))

    assert bs_value(warm_out) < bs_value(cool_out)


def test_film_look_saturated_source_gets_stronger_selectivecolor():
    vivid = pipeline_smart.film_look(1, section="BODY", levels=(0.05, 0.9), wb=(0.80, 0.20, 0.15))
    mono = pipeline_smart.film_look(1, section="BODY", levels=(0.05, 0.9), wb=(0.5, 0.5, 0.5))

    def greens_first_value(vf):
        m = re.search(r'selectivecolor=greens=([\-\d.]+)', vf)
        return float(m.group(1))

    assert greens_first_value(vivid) > greens_first_value(mono)


def test_film_look_dark_frame_gets_weaker_vignette_than_bright():
    dark = pipeline_smart.film_look(1, section="BODY", levels=(0.02, 0.55), wb=None)
    bright = pipeline_smart.film_look(1, section="BODY", levels=(0.15, 0.92), wb=None)

    def vignette_divisor(vf):
        m = re.search(r'vignette=PI/([\d.]+)', vf)
        return float(m.group(1))

    # Слабее виньетка = БОЛЬШЕ делитель в PI/N.
    assert vignette_divisor(dark) > vignette_divisor(bright)


def test_film_look_neutral_input_matches_unbiased_defaults():
    # Без levels/wb сигнала (None) грейд обязан остаться ровно тем же
    # фиксированным рецептом, что и раньше — content-aware поправка не
    # должна незаметно менять поведение, когда сигнала попросту нет.
    out = pipeline_smart.film_look(1, section="BODY")
    assert "colorbalance=rs=-0.06:bs=0.100:" in out
    assert "selectivecolor=greens=0.0200 0.0400 -0.0200 0.0200:" in out
    assert "vignette=PI/5.000," in out


# ---------- fix_pauses._pause_curve / _keep_sec_for (гладкая кривая пауз) ----------

def test_pause_curve_monotonic_non_decreasing():
    raws = [1.0, 1.2, 1.5, 1.8, 2.2, 2.6, 3.0, 5.0]
    keeps = [fix_pauses._pause_curve(r) for r in raws]
    assert all(a <= b + 1e-9 for a, b in zip(keeps, keeps[1:]))


def test_pause_curve_bounds():
    assert abs(fix_pauses._pause_curve(1.0) - fix_pauses.KEEP_MIN_SEC) < 1e-9
    assert abs(fix_pauses._pause_curve(fix_pauses.KEEP_CURVE_TOP_SEC) - fix_pauses.KEEP_MAX_SEC) < 1e-9
    assert abs(fix_pauses._pause_curve(999.0) - fix_pauses.KEEP_MAX_SEC) < 1e-9


def test_pause_jitter_varies_by_position_but_bounded():
    values = {fix_pauses._pause_jitter(t, t + 1.4) for t in (10.0, 250.0, 900.0, 1500.0)}
    assert len(values) > 1   # не одно и то же значение на всех позициях
    assert all(abs(v) <= fix_pauses.PAUSE_JITTER_SEC + 1e-9 for v in values)


def test_keep_sec_for_idempotent():
    assert fix_pauses._keep_sec_for(123.456, 124.856) == fix_pauses._keep_sec_for(123.456, 124.856)


def test_keep_sec_for_never_exceeds_raw_duration():
    for ss, se in [(0.0, 1.0), (5.0, 6.3), (100.0, 104.0)]:
        assert fix_pauses._keep_sec_for(ss, se) <= se - ss


def test_keep_sec_for_two_pauses_same_raw_duration_are_not_identical():
    # Регрессия против старого поведения: раньше ЛЮБЫЕ две паузы одинаковой
    # сырой длительности давали БИТ-В-БИТ одинаковый keep (ступенька по
    # константе) — джиттер обязан развести их.
    a = fix_pauses._keep_sec_for(10.0, 11.4)
    b = fix_pauses._keep_sec_for(500.0, 501.4)
    assert a != b


# ---------- fix_pauses: protected-паузы из speech_timeline.json (Speech Director) ----------

def test_keep_sec_for_uses_curve_when_no_protected_windows():
    # protected_windows=None (или []) -> поведение НЕ отличается от старого —
    # ноль регресса для эпизодов без Speech Director.
    a = fix_pauses._keep_sec_for(10.0, 11.4)
    b = fix_pauses._keep_sec_for(10.0, 11.4, protected_windows=None)
    c = fix_pauses._keep_sec_for(10.0, 11.4, protected_windows=[])
    assert a == b == c


def test_keep_sec_for_uses_exact_target_for_matched_protected_window():
    windows = [(10.0, 11.4, 1.10, "BLOCK 2: РАЗОБЛАЧЕНИЕ#1")]
    kept = fix_pauses._keep_sec_for(10.05, 11.35, protected_windows=windows)
    assert abs(kept - 1.10) < 1e-9   # ТОЧНАЯ цель плана, не кривая/джиттер


def test_keep_sec_for_protected_target_never_added_beyond_raw_duration():
    # Цель плана длиннее, чем реально есть тишины — не изобретаем лишнее.
    windows = [(10.0, 10.3, 1.40, "u#0")]
    kept = fix_pauses._keep_sec_for(10.0, 10.3, protected_windows=windows)
    assert kept <= 0.3 + 1e-9


def test_keep_sec_for_falls_back_to_curve_for_unmatched_silence():
    # protected_windows заданы, но эта конкретная тишина ни с одним окном не
    # совпадает — обычная кривая+джиттер, как для эпизода без плана.
    windows = [(500.0, 501.0, 1.10, "u#0")]
    curve_only = fix_pauses._keep_sec_for(10.0, 11.4)
    with_unrelated_plan = fix_pauses._keep_sec_for(10.0, 11.4, protected_windows=windows)
    assert curve_only == with_unrelated_plan


def test_match_protected_respects_tolerance_window():
    windows = [(10.0, 11.0, 0.8, "u#0")]
    # Небольшой сдвиг (silencedetect не даёт бит-в-бит те же границы, что
    # посимвольный alignment) — всё ещё должен матчиться.
    kept, unit_id = fix_pauses._match_protected(10.2, 10.9, windows)
    assert unit_id == "u#0"
    assert kept == 0.8


def test_match_protected_ignores_far_away_silence():
    windows = [(10.0, 11.0, 0.8, "u#0")]
    kept, unit_id = fix_pauses._match_protected(500.0, 501.0, windows)
    assert unit_id is None
    assert kept is None


def test_load_protected_windows_missing_file_returns_empty(tmp_path):
    assert fix_pauses.load_protected_windows(str(tmp_path)) == []


def test_load_protected_windows_reads_speech_timeline(tmp_path):
    plan_dir = tmp_path / "media_plan"
    plan_dir.mkdir()
    (plan_dir / "speech_timeline.json").write_text(
        '{"protected_windows": [[1.0, 2.0, 0.9, "HOOK#0"]]}', encoding="utf-8")
    windows = fix_pauses.load_protected_windows(str(tmp_path))
    assert windows == [(1.0, 2.0, 0.9, "HOOK#0")]


def test_load_protected_windows_corrupt_file_returns_empty_not_crash(tmp_path):
    plan_dir = tmp_path / "media_plan"
    plan_dir.mkdir()
    (plan_dir / "speech_timeline.json").write_text("{not valid json", encoding="utf-8")
    assert fix_pauses.load_protected_windows(str(tmp_path)) == []


# ---------- load_section_offsets / offset-коррекция protected_windows ----------
# Регрессия на РЕАЛЬНЫЙ найденный баг (не гипотеза — пойман эмпирической
# проверкой, см. коммит): speech_timeline.json хранит protected_windows в
# ЛОКАЛЬНОМ для каждой секции времени (своя нулевая точка на файл
# alignment.csv), а silencedetect в этом же fix_pauses.py меряет ВЕСЬ
# audio.mp3 ОДНИМ глобальным проходом. Без пересчёта в глобальное время
# через section_offsets.json защита пауз физически совпадала ТОЛЬКО с
# первой секцией эпизода (её локальный ноль случайно равен глобальному) —
# у всех следующих секций совпадения тихо не находилось.

def test_load_section_offsets_missing_file_returns_empty(tmp_path):
    assert fix_pauses.load_section_offsets(str(tmp_path)) == {}


def test_load_section_offsets_reads_file(tmp_path):
    plan_dir = tmp_path / "media_plan"
    plan_dir.mkdir()
    (plan_dir / "section_offsets.json").write_text(
        '{"HOOK": 0.0, "BLOCK 1: ТЕСТ": 6.28}', encoding="utf-8")
    offsets = fix_pauses.load_section_offsets(str(tmp_path))
    assert offsets == {"HOOK": 0.0, "BLOCK 1: ТЕСТ": 6.28}


def test_load_section_offsets_corrupt_file_returns_empty_not_crash(tmp_path):
    plan_dir = tmp_path / "media_plan"
    plan_dir.mkdir()
    (plan_dir / "section_offsets.json").write_text("{not valid", encoding="utf-8")
    assert fix_pauses.load_section_offsets(str(tmp_path)) == {}


def test_load_protected_windows_without_offsets_map_leaves_first_section_correct(tmp_path):
    # Без карты смещений (offset по умолчанию 0.0) окно ПЕРВОЙ секции
    # физически совпадает с глобальным временем случайно — это НЕ
    # регрессия, а известное ограничение (см. докстринг load_protected_windows).
    plan_dir = tmp_path / "media_plan"
    plan_dir.mkdir()
    (plan_dir / "speech_timeline.json").write_text(
        '{"protected_windows": [[1.0, 2.0, 0.9, "HOOK#0"]]}', encoding="utf-8")
    windows = fix_pauses.load_protected_windows(str(tmp_path))
    assert windows == [(1.0, 2.0, 0.9, "HOOK#0")]


def test_load_protected_windows_applies_section_offset_to_non_first_section(tmp_path):
    # ИМЕННО ЭТОТ тест ловит реальный баг: без offset-коррекции окно
    # второй секции осталось бы (2.88, 4.68) — физически не там, где
    # реальная тишина находится в audio.mp3 (9.16, 10.96).
    plan_dir = tmp_path / "media_plan"
    plan_dir.mkdir()
    (plan_dir / "speech_timeline.json").write_text(
        '{"protected_windows": [[2.88, 4.68, 1.4, "BLOCK 1: ТЕСТ#0"]]}', encoding="utf-8")
    (plan_dir / "section_offsets.json").write_text(
        '{"HOOK": 0.0, "BLOCK 1: ТЕСТ": 6.28}', encoding="utf-8")
    windows = fix_pauses.load_protected_windows(str(tmp_path))
    assert windows == [(9.16, 10.96, 1.4, "BLOCK 1: ТЕСТ#0")]


def test_load_protected_windows_unknown_section_defaults_to_zero_offset(tmp_path):
    plan_dir = tmp_path / "media_plan"
    plan_dir.mkdir()
    (plan_dir / "speech_timeline.json").write_text(
        '{"protected_windows": [[1.0, 2.0, 0.9, "MYSTERY#0"]]}', encoding="utf-8")
    (plan_dir / "section_offsets.json").write_text('{"HOOK": 0.0}', encoding="utf-8")
    windows = fix_pauses.load_protected_windows(str(tmp_path))
    assert windows == [(1.0, 2.0, 0.9, "MYSTERY#0")]


def test_load_protected_windows_section_name_split_on_last_hash():
    # unit_id формата "СЕКЦИЯ#N" — секция сама может (гипотетически)
    # содержать "#", берём ПОСЛЕДНИЙ разделитель, не первый.
    assert "A#B".rsplit("#", 1)[0] == "A"
    assert "BLOCK 1: A#B#3".rsplit("#", 1)[0] == "BLOCK 1: A#B"


# ---------- _exclude_climax_overlapping_windows (P0, нейрокогнитивная критика) ----------
# Регрессия на реальный найденный эффект: защита reveal_hold-паузы (см.
# коммит про section_offsets.json) часто делает её ДОСТАТОЧНО длинной,
# чтобы задеть PAUSE_SWELL_MIN_KEEP_SEC — раньше слепая кривая обрезала
# эту же паузу короче порога, swell на ней почти никогда не срабатывал.
# Теперь, без исключения, swell (+3дБ) накладывался бы поверх climax dip
# (-22дБ) на одном и том же участке трека — два последовательных volume-
# фильтра частично гасят друг друга, размывая самый важный провал.

def test_exclude_climax_overlapping_windows_removes_overlapping_pause():
    # Пауза [8.6, 10.0] физически пересекается с dip-окном климакса в
    # t=10.0 (см. _climax_dip_window: [8.9, 10.7] при дефолтных константах).
    kept = pipeline_smart._exclude_climax_overlapping_windows([(8.6, 1.4)], [10.0])
    assert kept == []


def test_exclude_climax_overlapping_windows_keeps_far_away_pause():
    kept = pipeline_smart._exclude_climax_overlapping_windows([(50.0, 1.2)], [10.0])
    assert kept == [(50.0, 1.2)]


def test_exclude_climax_overlapping_windows_mixed_list():
    kept = pipeline_smart._exclude_climax_overlapping_windows(
        [(8.6, 1.4), (50.0, 1.2), (100.0, 1.1)], [10.0])
    assert kept == [(50.0, 1.2), (100.0, 1.1)]


def test_exclude_climax_overlapping_windows_no_climax_is_noop():
    windows = [(8.6, 1.4), (50.0, 1.2)]
    assert pipeline_smart._exclude_climax_overlapping_windows(windows, []) == windows


def test_exclude_climax_overlapping_windows_no_pauses_is_noop():
    assert pipeline_smart._exclude_climax_overlapping_windows([], [10.0]) == []


def test_exclude_climax_overlapping_windows_multiple_climaxes():
    kept = pipeline_smart._exclude_climax_overlapping_windows(
        [(8.6, 1.4), (30.0, 1.1), (70.0, 1.3)], [10.0, 30.5])
    assert (8.6, 1.4) not in kept
    assert (30.0, 1.1) not in kept   # пересекается со вторым климаксом
    assert (70.0, 1.3) in kept


def test_climax_dip_window_matches_dip_expr_bounds():
    # _climax_dip_window() — та же формула d1/seg3_end, что _climax_dip_expr()
    # использует внутри себя (см. докстринг) — сверяем явно, чтобы будущая
    # правка констант не рассинхронила два места молча.
    t_c = 10.0
    start, end = pipeline_smart._climax_dip_window(t_c)
    d1 = max(0.0, t_c - pipeline_smart.CLIMAX_DIP_LEAD_SEC)
    expected_end = d1 + pipeline_smart.CLIMAX_DIP_FADE_SEC + pipeline_smart.CLIMAX_DIP_HOLD_SEC + pipeline_smart.CLIMAX_DIP_FADE_SEC
    assert start == d1
    assert end == expected_end
