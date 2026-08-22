"""Юнит-тесты чистой логики (без FFmpeg): парсинг сценария, тайминг блоков,
выбор без повторов, подбор тематического запроса, счётчик слов.
Запуск: .venv/bin/python -m pytest tests/ -v
"""
import os
import re
import sys
import tempfile

import pytest

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


def _max_run_length(values):
    best = cur = 1
    for a, b in zip(values, values[1:]):
        cur = cur + 1 if b == a else 1
        best = max(best, cur)
    return best


# Реальный, эмпирически найденный случай (жалоба "3 кадра подряд с кино",
# videos/_test_wide): даже когда КАЖДЫЙ отдельный унаследованный запрос
# формально верный по теме, длинная цепочка одинаковых буквальных строк
# подряд визуально монотонна — MAX_CONSECUTIVE_SAME_QUERY режет такие
# цепочки (см. _diversify_repeated_query_runs).

def test_resolve_queries_caps_long_inherited_run(monkeypatch):
    monkeypatch.setattr(pipeline_smart, "THEMES", {"меч": "medieval sword close up"})
    blocks = [{"text": "Меч был длинным.", "words": 3, "pause_after": 0.0, "section": "BODY", "stat": None}]
    blocks += [{"text": f"Абстрактная фраза {i}.", "words": 2, "pause_after": 0.0, "section": "BODY", "stat": None}
               for i in range(8)]
    resolved = pipeline_smart.resolve_queries(blocks)
    assert _max_run_length(resolved) <= pipeline_smart.MAX_CONSECUTIVE_SAME_QUERY


def test_resolve_queries_diversify_converges_no_residual_long_run(monkeypatch):
    # Регрессия на реальный найденный баг: один проход разбивал длинный
    # прогон, но несколько соседних блоков независимо занимали ОДИН и тот
    # же альтернативный запрос у одного внешнего соседа — прогон не
    # исчезал, а просто сдвигался (был 6 подряд "A", стало 3 + НОВЫЙ
    # прогон из 4 "B"). Нужен fixed-point (несколько проходов), не один.
    monkeypatch.setattr(pipeline_smart, "THEMES", {"меч": "medieval sword close up",
                                                     "клинок": "sword blade close up"})
    blocks = [{"text": "Меч был длинным.", "words": 3, "pause_after": 0.0, "section": "HOOK", "stat": None}
              for _ in range(3)]
    blocks += [{"text": f"Абстрактная фраза {i}.", "words": 2, "pause_after": 0.0, "section": "HOOK", "stat": None}
               for i in range(3)]
    blocks += [{"text": "Острый клинок.", "words": 2, "pause_after": 0.0, "section": "HOOK", "stat": None}]
    resolved = pipeline_smart.resolve_queries(blocks)
    assert _max_run_length(resolved) <= pipeline_smart.MAX_CONSECUTIVE_SAME_QUERY


def test_resolve_queries_all_same_in_section_falls_back_to_generic(monkeypatch):
    # Вся секция схлопнулась в ОДИН буквальный запрос (реальное совпадение
    # ключевого слова в каждом блоке, не наследование) — искать "другого
    # соседа той же секции" бессмысленно, никакого другого нет. Диверсификация
    # обязана откатиться на GENERIC_FALLBACKS, а не зациклиться/оставить
    # запрос как есть сверх лимита.
    monkeypatch.setattr(pipeline_smart, "THEMES", {"меч": "medieval sword close up"})
    blocks = [{"text": "Меч меч меч.", "words": 3, "pause_after": 0.0, "section": "BODY", "stat": None}
              for _ in range(6)]
    resolved = pipeline_smart.resolve_queries(blocks)
    assert _max_run_length(resolved) <= pipeline_smart.MAX_CONSECUTIVE_SAME_QUERY
    assert any(q in pipeline_smart.GENERIC_FALLBACKS for q in resolved[pipeline_smart.MAX_CONSECUTIVE_SAME_QUERY:])


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


# ---------- _domain_grade_cache_signature (П.4: temp_smart/ версионирование) ----------

def test_domain_grade_cache_signature_off_when_disabled():
    assert pipeline_smart._domain_grade_cache_signature(None) == "domain:off"


def test_domain_grade_cache_signature_on_when_enabled():
    sig = pipeline_smart._domain_grade_cache_signature(object())
    assert sig.startswith("domain:on:")


def test_domain_grade_cache_signature_changes_with_table(monkeypatch):
    # Регрессия: params_hash (main()) раньше не учитывал DOMAIN_GRADE_MODE/
    # DOMAIN_WARM_PUSH_SCALE вообще — правка таблицы между прогонами молча
    # оставляла старые кэшированные клипы temp_smart/ нетронутыми. Сигнатура
    # должна реально меняться при правке таблицы, не быть константой.
    before = pipeline_smart._domain_grade_cache_signature(object())
    monkeypatch.setattr(pipeline_smart, "DOMAIN_WARM_PUSH_SCALE", {"snow": 0.9})
    after = pipeline_smart._domain_grade_cache_signature(object())
    assert before != after


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


# ---------- channel_profile.json wiring (P3, ЧАСТЬ 24) ----------

def test_channel_profile_loaded_from_repo_root():
    # channel_profile.json (корень репо) должен реально читаться, не тихо
    # игнорироваться — иначе весь смысл выноса MOOD_GRADE/CONTENT_ALT_
    # BLOCKLIST/VOICE_* из кода в конфиг (см. docs/ROADMAP_CHANNEL_PROFILE.md)
    # был бы фиктивным.
    assert pipeline_smart.CHANNEL_PROFILE.get("mood_grade")
    assert pipeline_smart.CHANNEL_PROFILE.get("content_alt_blocklist")
    assert pipeline_smart.CHANNEL_PROFILE.get("voice")


def test_mood_grade_matches_channel_profile_file():
    assert pipeline_smart.MOOD_GRADE == pipeline_smart.CHANNEL_PROFILE["mood_grade"]


def test_content_alt_blocklist_matches_channel_profile_file():
    assert pipeline_smart.CONTENT_ALT_BLOCKLIST == tuple(pipeline_smart.CHANNEL_PROFILE["content_alt_blocklist"])


def test_voice_tuning_matches_channel_profile_file():
    voice = pipeline_smart.CHANNEL_PROFILE["voice"]
    assert pipeline_smart.VOICE_HIGHPASS_HZ == voice["highpass_hz"]
    assert pipeline_smart.VOICE_COMPRESS_THRESHOLD == voice["compress_threshold"]
    assert pipeline_smart.VOICE_COMPRESS_RATIO == voice["compress_ratio"]
    assert pipeline_smart.VOICE_DEESS_INTENSITY == voice["deess_intensity"]


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


# ---------- _face_region_plausible (П.4: "меч всё ещё холодный" — face false positive) ----------

_PARALLAX_SKIP = not pipeline_smart.PARALLAX_LIBS


def _make_bgr(size, color):
    import numpy as np
    h, w = size
    arr = np.zeros((h, w, 3), dtype="uint8")
    arr[:, :] = color   # BGR
    return arr


@pytest.mark.skipif(_PARALLAX_SKIP, reason="cv2 недоступен")
def test_face_region_plausible_rejects_no_skin_tone(monkeypatch):
    # Реальный баг: навершие меча/шлем/кинозал ложно триггерят frontalface-
    # каскад — регион ЯВНО не кожа (холодный металлик), должен отклоняться
    # ДО обращения к eye-каскаду (skin-check первым).
    import cv2
    img = _make_bgr((60, 60), (200, 200, 200))   # серый металлик, не кожа
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert pipeline_smart._face_region_plausible(img, gray, 5, 5, 40, 40) is False


@pytest.mark.skipif(_PARALLAX_SKIP, reason="cv2 недоступен")
def test_face_region_plausible_requires_eyes_even_with_skin_tone(monkeypatch):
    # Камуфляжный геймпад (реальный найденный остаточный false positive) —
    # цвет попадает в skin-tone диапазон, но глаз внутри нет: eye-каскад
    # должен отклонить. Мокаем именно eye-каскад (не сам факт вызова —
    # deterministic поведение теста, не полагается на то, найдёт ли реальный
    # OpenCV-каскад глаза на однотонной заливке).
    import cv2

    class _NoEyes:
        def detectMultiScale(self, *a, **kw):
            return []

    monkeypatch.setattr(pipeline_smart, "_EYE_CASCADE", _NoEyes())
    img = _make_bgr((60, 60), (120, 150, 200))   # BGR внутри YCrCb skin-диапазона
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert pipeline_smart._face_region_plausible(img, gray, 5, 5, 40, 40) is False


@pytest.mark.skipif(_PARALLAX_SKIP, reason="cv2 недоступен")
def test_face_region_plausible_passes_with_skin_and_eyes(monkeypatch):
    import cv2

    class _OneEye:
        def detectMultiScale(self, *a, **kw):
            return [(10, 10, 8, 8)]

    monkeypatch.setattr(pipeline_smart, "_EYE_CASCADE", _OneEye())
    img = _make_bgr((60, 60), (120, 150, 200))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert pipeline_smart._face_region_plausible(img, gray, 5, 5, 40, 40) is True


@pytest.mark.skipif(_PARALLAX_SKIP, reason="cv2 недоступен")
def test_detect_face_anchor_none_for_known_false_positive_sword_photo():
    # Регрессия на реальный, найденный вживую случай (не синтетика): этот
    # конкретный кадр (сток-меч в снегу) годами давал ложное срабатывание
    # каскада лица, что блокировало Look Management assist на нём — см.
    # коммит. Тест — реальный файл, если он ещё в кэше этой рабочей копии
    # (не гарантирован в CI без реального temp_smart/); best-effort skip.
    import glob
    candidates = glob.glob("videos/*/temp_smart/pexels_cache/*1b37a930.jpg")
    if not candidates:
        pytest.skip("реальный тестовый файл не найден в этой рабочей копии")
    assert pipeline_smart.detect_face_anchor(candidates[0]) is None


# ---------- section_offsets.json: локальное -> глобальное время для BLOCK1+ ----------
# Реальный, эмпирически найденный баг (не гипотеза, videos/_test_wide):
# raw_to_real_time() верно вычитает ГЛОБАЛЬНЫЕ обрезки пауз из t, но
# _real_speech_span()/load_alignment_weights() раньше подавали туда
# ЛОКАЛЬНОЕ время секции (свой ноль на файл alignment/NN.csv) НАПРЯМУЮ, как
# будто оно уже глобальное — верно только для первой секции (HOOK, ноль
# совпадает случайно). Для BLOCK1+ это давало неверный вес блока, который
# одинаково портил и субтитры, и тайминг монтажных резов (см.
# scripts/section_sync.py). Фикс — section_offset в _real_speech_span() +
# load_section_offsets() в load_alignment_weights().

def _reset_pipeline_smart_caches(monkeypatch):
    monkeypatch.setattr(pipeline_smart, "_PAUSE_CUTS_CACHE", None)
    monkeypatch.setattr(pipeline_smart, "_PAUSE_WINDOWS_CACHE", None)
    monkeypatch.setattr(pipeline_smart, "_SECTION_OFFSETS_CACHE", None)


def test_load_section_offsets_missing_file_returns_empty(tmp_path, monkeypatch):
    _reset_pipeline_smart_caches(monkeypatch)
    monkeypatch.setattr(pipeline_smart, "SECTION_OFFSETS_PATH", str(tmp_path / "media_plan" / "section_offsets.json"))
    assert pipeline_smart.load_section_offsets() == {}


def test_load_section_offsets_reads_file(tmp_path, monkeypatch):
    _reset_pipeline_smart_caches(monkeypatch)
    plan_dir = tmp_path / "media_plan"
    plan_dir.mkdir()
    (plan_dir / "section_offsets.json").write_text(
        '{"HOOK": 0.0, "BLOCK 1: X": 56.649}', encoding="utf-8")
    monkeypatch.setattr(pipeline_smart, "SECTION_OFFSETS_PATH", str(plan_dir / "section_offsets.json"))
    offsets = pipeline_smart.load_section_offsets()
    assert offsets == {"HOOK": 0.0, "BLOCK 1: X": 56.649}


def test_real_speech_span_ignores_offset_zero_baseline(monkeypatch):
    _reset_pipeline_smart_caches(monkeypatch)
    monkeypatch.setattr(pipeline_smart, "load_pause_cuts", lambda: [])
    segment = [("X", 0.0, 5.0)]
    assert pipeline_smart._real_speech_span(segment) == pytest.approx(5.0)
    assert pipeline_smart._real_speech_span(segment, section_offset=0.0) == pytest.approx(5.0)


def test_real_speech_span_without_offset_misses_real_cut_in_later_section(monkeypatch):
    # Секция реально начинается на 100-й секунде исходного audio.mp3, и
    # ровно внутри неё есть реальная обрезанная пауза (102-103) — БЕЗ
    # смещения (старое поведение) локальный сегмент [0, 5] не пересекается
    # с (102, 103) вообще, обрезка молча теряется -> span завышен.
    _reset_pipeline_smart_caches(monkeypatch)
    cuts = [(102.0, 103.0)]
    monkeypatch.setattr(pipeline_smart, "load_pause_cuts", lambda: cuts)
    segment = [("X", 0.0, 5.0)]
    assert pipeline_smart._real_speech_span(segment, section_offset=0.0) == pytest.approx(5.0)


def test_real_speech_span_with_offset_correctly_applies_real_cut(monkeypatch):
    _reset_pipeline_smart_caches(monkeypatch)
    cuts = [(102.0, 103.0)]
    monkeypatch.setattr(pipeline_smart, "load_pause_cuts", lambda: cuts)
    segment = [("X", 0.0, 5.0)]
    # то же самое локальное время, но с ПРАВИЛЬНЫМ смещением секции (100.0) —
    # обрезка (102-103) реально попадает внутрь [100, 105], вес уменьшается на 1с
    assert pipeline_smart._real_speech_span(segment, section_offset=100.0) == pytest.approx(4.0)


def _write_alignment_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["char,start,end"] + [f"{c},{s},{e}" for c, s, e in rows]
    path.write_text("\n".join(lines), encoding="utf-8")


def test_load_alignment_weights_applies_section_offset_to_non_first_section(tmp_path, monkeypatch):
    _reset_pipeline_smart_caches(monkeypatch)
    plan_dir = tmp_path / "media_plan"
    alignment_dir = plan_dir / "alignment"
    monkeypatch.setattr(pipeline_smart, "ALIGNMENT_DIR", str(alignment_dir))
    monkeypatch.setattr(pipeline_smart, "PAUSE_CUTS_PATH", str(plan_dir / "pause_cuts.json"))
    monkeypatch.setattr(pipeline_smart, "SECTION_OFFSETS_PATH", str(plan_dir / "section_offsets.json"))

    # HOOK: локальный ноль совпадает с глобальным (0.0-0.9), без обрезок рядом
    _write_alignment_csv(alignment_dir / "00.csv",
                         [("Р", 0.0, 0.3), ("а", 0.3, 0.6), ("з", 0.6, 0.9)])
    # BLOCK 1: РЕАЛЬНО начинается на 10.0с исходного audio.mp3 (HOOK кадр
    # длиннее, чем 0.9с локального alignment — тот же случай, что и в
    # реальном эпизоде, где HOOK — не единственный источник глобального
    # смещения). Локально секция сама по себе 0.0-1.5с.
    _write_alignment_csv(alignment_dir / "01.csv",
                         [("Д", 0.0, 0.5), ("в", 0.5, 1.0), ("а", 1.0, 1.5)])

    plan_dir.mkdir(exist_ok=True)
    (plan_dir / "section_offsets.json").write_text(
        '{"BLOCK 1: ТЕСТ": 10.0}', encoding="utf-8")
    # реальная обрезанная пауза внутри ГЛОБАЛЬНОГО окна BLOCK1 (10.0-11.5) —
    # без source_audio_md5, чтобы не требовать реальный audio.mp3 в тесте
    (plan_dir / "pause_cuts.json").write_text(
        '{"cuts": [[10.5, 11.0]]}', encoding="utf-8")

    blocks = [
        {"text": "Раз", "words": 1, "section": "HOOK", "pause_after": 0.0},
        {"text": "Два", "words": 1, "section": "BLOCK 1: ТЕСТ", "pause_after": 0.0},
    ]
    weights = pipeline_smart.load_alignment_weights(blocks)
    assert weights[0] == pytest.approx(0.9)    # HOOK не затронут (обрезка далеко после него)
    assert weights[1] == pytest.approx(1.0)    # BLOCK1: 1.5с локальных - 0.5с реальной обрезки внутри

    # Контроль: без section_offsets.json та же обрезка (10.5-11.0) не
    # пересекает локальный диапазон BLOCK1 [0, 1.5] вообще -> вес остался
    # бы НЕобрезанным (1.5) — именно так ошибка выглядела до фикса.
    (plan_dir / "section_offsets.json").unlink()
    _reset_pipeline_smart_caches(monkeypatch)
    weights_no_offset_file = pipeline_smart.load_alignment_weights(blocks)
    assert weights_no_offset_file[1] == pytest.approx(1.5)


# ---------- audio provenance gate (P0-1 форензик-аудита) ----------
# Реальный, подтверждённый риск: раньше find_audio() безусловно выбирал
# audio_fixed.flac, если он просто СУЩЕСТВОВАЛ на диске — переозвучили
# эпизод (заменили audio.mp3), забыли перезапустить fix_pauses.py, и
# pipeline_smart.py тихо собрал бы ролик СО СТАРЫМ ГОЛОСОМ из
# audio_fixed.flac под НОВЫМ таймингом/субтитрами. Такого сценария не
# было в тестах вообще.

def _write_audio_bytes(path, content):
    path.write_bytes(content)
    return pipeline_smart._md5_file(str(path))


def test_fixed_audio_is_current_no_pause_cuts_file(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_smart, "VIDEO_FOLDER", str(tmp_path))
    fixed = tmp_path / "audio_fixed.flac"
    fixed.write_bytes(b"fixed-bytes")
    assert pipeline_smart._fixed_audio_is_current(str(fixed)) is False


def test_fixed_audio_is_current_true_when_both_hashes_match(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_smart, "VIDEO_FOLDER", str(tmp_path))
    raw = tmp_path / "audio.mp3"
    fixed = tmp_path / "audio_fixed.flac"
    raw_md5 = _write_audio_bytes(raw, b"raw-voice-v1")
    fixed_md5 = _write_audio_bytes(fixed, b"fixed-voice-v1")
    plan_dir = tmp_path / "media_plan"
    plan_dir.mkdir()
    (plan_dir / "pause_cuts.json").write_text(
        f'{{"source_audio_md5": "{raw_md5}", "fixed_audio_md5": "{fixed_md5}", "cuts": []}}',
        encoding="utf-8")
    assert pipeline_smart._fixed_audio_is_current(str(fixed)) is True


def test_fixed_audio_is_current_false_when_raw_audio_changed(tmp_path, monkeypatch):
    # Переозвучили (audio.mp3 другой), fix_pauses.py не перезапускали —
    # source_audio_md5 в pause_cuts.json теперь не совпадает с реальным.
    monkeypatch.setattr(pipeline_smart, "VIDEO_FOLDER", str(tmp_path))
    raw = tmp_path / "audio.mp3"
    fixed = tmp_path / "audio_fixed.flac"
    _write_audio_bytes(raw, b"raw-voice-v2-NEW")
    fixed_md5 = _write_audio_bytes(fixed, b"fixed-voice-v1")
    plan_dir = tmp_path / "media_plan"
    plan_dir.mkdir()
    (plan_dir / "pause_cuts.json").write_text(
        f'{{"source_audio_md5": "old-raw-md5-does-not-match", '
        f'"fixed_audio_md5": "{fixed_md5}", "cuts": []}}', encoding="utf-8")
    assert pipeline_smart._fixed_audio_is_current(str(fixed)) is False


def test_fixed_audio_is_current_false_when_fixed_file_itself_changed(tmp_path, monkeypatch):
    # Сырой audio.mp3 не менялся, но сам audio_fixed.flac подменили/повредили
    # (или это остаток от другого прогона с тем же именем) — раздельная
    # проверка ловит и это, не только рассинхрон raw-источника.
    monkeypatch.setattr(pipeline_smart, "VIDEO_FOLDER", str(tmp_path))
    raw = tmp_path / "audio.mp3"
    fixed = tmp_path / "audio_fixed.flac"
    raw_md5 = _write_audio_bytes(raw, b"raw-voice-v1")
    _write_audio_bytes(fixed, b"SOME OTHER fixed audio entirely")
    plan_dir = tmp_path / "media_plan"
    plan_dir.mkdir()
    (plan_dir / "pause_cuts.json").write_text(
        f'{{"source_audio_md5": "{raw_md5}", '
        f'"fixed_audio_md5": "old-fixed-md5-does-not-match", "cuts": []}}', encoding="utf-8")
    assert pipeline_smart._fixed_audio_is_current(str(fixed)) is False


def test_fixed_audio_is_current_old_format_without_fixed_md5_falls_back_to_source_check(tmp_path, monkeypatch):
    # Старый формат pause_cuts.json (эпизод обработан до этого фикса) — нет
    # ключа fixed_audio_md5 вообще. Не должно стать НОВЫМ ограничением для
    # уже работающих старых эпизодов: тот же уровень доверия, что и раньше
    # (проверка только source_audio_md5).
    monkeypatch.setattr(pipeline_smart, "VIDEO_FOLDER", str(tmp_path))
    raw = tmp_path / "audio.mp3"
    fixed = tmp_path / "audio_fixed.flac"
    raw_md5 = _write_audio_bytes(raw, b"raw-voice-v1")
    _write_audio_bytes(fixed, b"fixed-voice-v1")
    plan_dir = tmp_path / "media_plan"
    plan_dir.mkdir()
    (plan_dir / "pause_cuts.json").write_text(
        f'{{"source_audio_md5": "{raw_md5}", "cuts": []}}', encoding="utf-8")
    assert pipeline_smart._fixed_audio_is_current(str(fixed)) is True


def test_find_audio_falls_back_to_raw_when_fixed_is_stale(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pipeline_smart, "VIDEO_FOLDER", str(tmp_path))
    raw = tmp_path / "audio.mp3"
    fixed = tmp_path / "audio_fixed.flac"
    _write_audio_bytes(raw, b"raw-voice-v2-NEW")   # переозвучили
    _write_audio_bytes(fixed, b"fixed-voice-v1")   # старый FLAC, fix_pauses.py не перезапускали
    plan_dir = tmp_path / "media_plan"
    plan_dir.mkdir()
    (plan_dir / "pause_cuts.json").write_text(
        '{"source_audio_md5": "stale-raw-md5", "fixed_audio_md5": "stale-fixed-md5", "cuts": []}',
        encoding="utf-8")
    result = pipeline_smart.find_audio()
    assert result == str(raw)
    assert "ВНИМАНИЕ" in capsys.readouterr().out


def test_find_audio_uses_fixed_when_current(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_smart, "VIDEO_FOLDER", str(tmp_path))
    raw = tmp_path / "audio.mp3"
    fixed = tmp_path / "audio_fixed.flac"
    raw_md5 = _write_audio_bytes(raw, b"raw-voice-v1")
    fixed_md5 = _write_audio_bytes(fixed, b"fixed-voice-v1")
    plan_dir = tmp_path / "media_plan"
    plan_dir.mkdir()
    (plan_dir / "pause_cuts.json").write_text(
        f'{{"source_audio_md5": "{raw_md5}", "fixed_audio_md5": "{fixed_md5}", "cuts": []}}',
        encoding="utf-8")
    assert pipeline_smart.find_audio() == str(fixed)


# ---------- rescale_hook_words_to_visual_time (P0-3 форензик-аудита) ----------
# Реальный, подтверждённый эмпирически баг: хук-слова кладутся на АУДИО-
# шкалу (sub_starts/sub_baseline), но реально рендерящийся визуальный кадр
# после apply_section_boundary_shift/apply_within_cut_shift/apply_human_jitter/
# snap_hook_cuts_to_energy (все сдвигают ИМЕННО durs) может отличаться по
# длительности/старту — локальная позиция подписи внутри клипа систематически
# уезжает от того, что реально показано на экране.

def test_rescale_hook_words_identity_when_visual_equals_audio_scale():
    # Визуальная шкала бит-в-бит совпадает с аудио (нет сдвигов durs) —
    # rescale обязан быть identity (никакой лишней погрешности из ничего).
    blocks = [{"section": "HOOK"}, {"section": "HOOK"}]
    sub_starts = [0.0, 2.0]
    sub_baseline = [2.0, 3.0]
    visual_starts = [0.0, 2.0]
    durs = [2.0, 3.0]
    hook_words = [("Раз", 0.5, 1.0), ("Два", 3.0, 3.5)]
    out = pipeline_smart.rescale_hook_words_to_visual_time(
        hook_words, blocks, sub_starts, sub_baseline, visual_starts, durs)
    for (w1, s1, e1), (w2, s2, e2) in zip(hook_words, out):
        assert w1 == w2
        assert s1 == pytest.approx(s2)
        assert e1 == pytest.approx(e2)


def test_rescale_hook_words_stretches_into_shifted_visual_window():
    # Блок 1: аудио-окно [2.0, 5.0) (baseline=3.0), визуальное окно после
    # apply_within_cut_shift сдвинулось и стало [2.1, 5.4) (durs=3.3) —
    # слово в середине аудио-окна (50%) должно оказаться в середине
    # ВИЗУАЛЬНОГО окна (50% от 3.3 = 1.65 после старта 2.1 -> 3.75).
    blocks = [{"section": "HOOK"}, {"section": "HOOK"}]
    sub_starts = [0.0, 2.0]
    sub_baseline = [2.0, 3.0]
    visual_starts = [0.0, 2.1]
    durs = [2.0, 3.3]
    hook_words = [("Середина", 3.5, 4.0)]   # 3.5 = 2.0 + 0.5*3.0 (50% блока 1)
    out = pipeline_smart.rescale_hook_words_to_visual_time(
        hook_words, blocks, sub_starts, sub_baseline, visual_starts, durs)
    w, s, e = out[0]
    assert w == "Середина"
    assert s == pytest.approx(2.1 + 0.5 * 3.3)
    assert e == pytest.approx(2.1 + (2.0 / 3.0) * 3.3)


def test_rescale_hook_words_only_rescales_hook_section_windows():
    # Небук-секции игнорируются при построении окон (rescale работает
    # только с hook_idxs), но неспецифичное слово вне всех окон должно
    # просто вернуться как есть, не падать/не подменяться произвольно.
    blocks = [{"section": "HOOK"}, {"section": "BLOCK 1"}]
    sub_starts = [0.0, 2.0]
    sub_baseline = [2.0, 5.0]
    visual_starts = [0.0, 2.2]
    durs = [2.2, 4.5]
    hook_words = [("Вне", 10.0, 10.5)]   # далеко за пределами хук-окна [0, 2.0)
    out = pipeline_smart.rescale_hook_words_to_visual_time(
        hook_words, blocks, sub_starts, sub_baseline, visual_starts, durs)
    assert out == hook_words


def test_rescale_hook_words_empty_input_noop():
    assert pipeline_smart.rescale_hook_words_to_visual_time([], [], [], [], [], []) == []
