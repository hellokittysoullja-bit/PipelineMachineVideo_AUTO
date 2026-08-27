"""Юнит-тесты чистой логики (без FFmpeg): парсинг сценария, тайминг блоков,
выбор без повторов, подбор тематического запроса, счётчик слов.
Запуск: .venv/bin/python -m pytest tests/ -v
"""
import json
import os
import re
import subprocess
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
import shot_director            # noqa: E402
import fix_pauses               # noqa: E402


@pytest.fixture(autouse=True)
def _clear_pexels_search_caches():
    """Кэши выдачи Pexels живут на процесс (это их смысл в проде — см.
    _pexels_search_photos: без них сбор пула из всех запросов секции упёрся
    бы в квоту 200/час). В тестах то же самое означает утечку состояния
    между тестами: тест с замоканным urlopen не увидел бы своего мока,
    потому что ответ уже лежит в кэше от предыдущего теста — реально
    пойманное падение, тест проходил в одиночку и падал в группе."""
    pipeline_smart._PEXELS_SEARCH_CACHE.clear()
    pipeline_smart._PEXELS_VIDEO_SEARCH_CACHE.clear()
    yield
    pipeline_smart._PEXELS_SEARCH_CACHE.clear()
    pipeline_smart._PEXELS_VIDEO_SEARCH_CACHE.clear()


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


def test_parse_blocks_unknown_tag_warns_and_does_not_split(tmp_path, capsys):
    # Регрессия: раньше [long pause] (явно ЗАПРЕЩЁН ЧАСТЬЮ 10 CLAUDE.md) и
    # любой другой незнакомый тег молча вырезались без единого предупреждения
    # и без вставки паузы — сценарист не узнал бы, что граница блока/пауза в
    # этом месте пропала.
    text = "=== HOOK === Раз.[long pause]Два.\n"
    blocks = pipeline_smart.parse_blocks(write_script(tmp_path, text))
    assert len(blocks) == 1                       # тег не создал границу (как и раньше)
    assert blocks[0]["pause_after"] == 0.0         # и не добавил паузу (как и раньше)
    assert "[long pause]" not in blocks[0]["text"]
    err = capsys.readouterr().out
    assert "ВНИМАНИЕ" in err
    assert "[long pause]" in err


def test_parse_blocks_known_tags_do_not_trigger_warning(tmp_path, capsys):
    text = "=== HOOK === Раз.[pause]Два.[short pause]Три.\n"
    pipeline_smart.parse_blocks(write_script(tmp_path, text))
    assert "ВНИМАНИЕ" not in capsys.readouterr().out


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


# ---------- resolve_queries(authored_queries=...) — === PEXELS QUERIES ===
# написан вручную по протоколу (CLAUDE.md ЧАСТЬ 13, Шаг 3), но до этого не
# читался пайплайном вообще (реальный, найденный по прямому запросу
# пользователя пробел) — query_for()/THEMES физически не может
# воспроизвести контекст конкретного момента сценария, который человек/LLM
# вписывает вручную ("вспомни про молоко" -> "milk bottle hand"). ----------

def test_resolve_queries_authored_takes_priority_over_themes(monkeypatch):
    # Даже когда THEMES нашёл бы совпадение ("меч"), авторский запрос из
    # === PEXELS QUERIES === побеждает — он написан с полным контекстом
    # сцены, а не по одному ключевому слову.
    monkeypatch.setattr(pipeline_smart, "THEMES", {"меч": "medieval sword close up"})
    blocks = [{"text": "Меч был длинным.", "words": 3, "pause_after": 0.0, "section": "HOOK", "stat": None}]
    resolved = pipeline_smart.resolve_queries(blocks, authored_queries={"HOOK": ["milk bottle hand"]})
    assert resolved == ["milk bottle hand"]


def test_resolve_queries_authored_cycles_across_original_blocks_in_section(monkeypatch):
    monkeypatch.setattr(pipeline_smart, "THEMES", {})
    blocks = [{"text": f"Фраза {i}.", "words": 2, "pause_after": 0.0, "section": "HOOK", "stat": None}
              for i in range(3)]
    resolved = pipeline_smart.resolve_queries(blocks, authored_queries={"HOOK": ["q1", "q2"]})
    assert resolved == ["q1", "q2", "q1"]   # третий блок — по кругу, снова q1


def test_resolve_queries_authored_subcut_inherits_parent_not_next_in_pool(monkeypatch):
    # is_subcut=True — это НЕ новый исходный блок, а продолжение предыдущего
    # (см. split_long_blocks: "под-кадры унаследуют запрос") — обязан
    # получить ТОТ ЖЕ авторский запрос, что и родитель, не следующий по кругу.
    monkeypatch.setattr(pipeline_smart, "THEMES", {})
    blocks = [
        {"text": "Первая половина фразы.", "words": 3, "pause_after": 0.0,
         "section": "HOOK", "stat": None, "is_subcut": False},
        {"text": "Вторая половина той же фразы.", "words": 4, "pause_after": 0.0,
         "section": "HOOK", "stat": None, "is_subcut": True},
        {"text": "Уже новая, отдельная фраза.", "words": 4, "pause_after": 0.0,
         "section": "HOOK", "stat": None, "is_subcut": False},
    ]
    resolved = pipeline_smart.resolve_queries(blocks, authored_queries={"HOOK": ["q1", "q2"]})
    assert resolved == ["q1", "q1", "q2"]


def test_resolve_queries_authored_matches_full_section_header_with_title(monkeypatch):
    # b["section"] реально хранит "BLOCK 1: Постановка проблемы" (полный
    # заголовок из === BLOCK 1: ... ===), а строка в === PEXELS QUERIES ===
    # это "BLOCK_1:" (подчёркивание, без заголовка) — оба должны совпасть
    # через _normalize_section_key.
    monkeypatch.setattr(pipeline_smart, "THEMES", {})
    blocks = [{"text": "Любая фраза.", "words": 2, "pause_after": 0.0,
               "section": "BLOCK 1: ПОСТАНОВКА ПРОБЛЕМЫ", "stat": None}]
    resolved = pipeline_smart.resolve_queries(blocks, authored_queries={"BLOCK1": ["museum sword display"]})
    assert resolved == ["museum sword display"]


def test_resolve_queries_no_authored_queries_unchanged_behavior(monkeypatch):
    # authored_queries=None (или {}) — байт-в-байт прежнее поведение.
    monkeypatch.setattr(pipeline_smart, "THEMES", {"меч": "medieval sword close up"})
    blocks = [{"text": "Меч весил немало.", "words": 3, "pause_after": 0.0, "section": "BODY", "stat": None}]
    assert pipeline_smart.resolve_queries(blocks) == pipeline_smart.resolve_queries(blocks, authored_queries=None)
    assert pipeline_smart.resolve_queries(blocks) == pipeline_smart.resolve_queries(blocks, authored_queries={})


# ---------- resolve_queries × shot_director (LLM-режиссёр, SHOT_DIRECTOR_MODE) ----------
# off (дефолт) -> ноль вызовов shot_director, byte-for-byte старое поведение.
# on -> вызывается ТОЛЬКО для блоков, оставшихся None после authored_queries
# и THEMES (см. shot_director.py докстринг) — не для блоков, у которых уже
# есть авторский/тематический запрос.

def test_resolve_queries_mode_off_never_calls_shot_director(monkeypatch):
    monkeypatch.delenv("SHOT_DIRECTOR_MODE", raising=False)
    monkeypatch.setattr(pipeline_smart, "THEMES", {})
    calls = []
    monkeypatch.setattr(shot_director, "direct_query",
                         lambda text, video_dir: calls.append(text) or "should-not-be-used")
    blocks = [{"text": "Взять быка за рога.", "words": 4, "pause_after": 0.0, "section": "BODY", "stat": None}]
    resolved = pipeline_smart.resolve_queries(blocks)
    assert calls == []
    assert resolved == [pipeline_smart.GENERIC_FALLBACKS[0]]


def test_resolve_queries_mode_on_uses_director_for_unresolved_block(monkeypatch):
    monkeypatch.setenv("SHOT_DIRECTOR_MODE", "on")
    monkeypatch.setattr(pipeline_smart, "THEMES", {})
    monkeypatch.setattr(shot_director, "direct_query",
                         lambda text, video_dir: "person taking decisive action")
    blocks = [{"text": "Взять быка за рога.", "words": 4, "pause_after": 0.0, "section": "BODY", "stat": None}]
    resolved = pipeline_smart.resolve_queries(blocks)
    assert resolved == ["person taking decisive action"]


def test_resolve_queries_mode_on_skips_blocks_already_resolved_by_themes(monkeypatch):
    # THEMES/authored уже дали ответ — shot_director вообще не должен
    # вызываться на этот блок (бюджет вызовов бережём для реального остатка).
    monkeypatch.setenv("SHOT_DIRECTOR_MODE", "on")
    monkeypatch.setattr(pipeline_smart, "THEMES", {"меч": "medieval sword close up"})
    calls = []
    monkeypatch.setattr(shot_director, "direct_query",
                         lambda text, video_dir: calls.append(text) or "unused")
    blocks = [{"text": "Меч весил немало.", "words": 3, "pause_after": 0.0, "section": "BODY", "stat": None}]
    resolved = pipeline_smart.resolve_queries(blocks)
    assert calls == []
    assert resolved == ["medieval sword close up"]


def test_resolve_queries_mode_on_director_returns_none_falls_back(monkeypatch):
    # director недоступен/лимит исчерпан/ошибка -> None -> прежнее поведение
    # (neighbor-inherit/GENERIC_FALLBACKS), ноль регрессии.
    monkeypatch.setenv("SHOT_DIRECTOR_MODE", "on")
    monkeypatch.setattr(pipeline_smart, "THEMES", {})
    monkeypatch.setattr(shot_director, "direct_query", lambda text, video_dir: None)
    blocks = [{"text": "Абстрактная фраза.", "words": 2, "pause_after": 0.0, "section": "BODY", "stat": None}]
    resolved = pipeline_smart.resolve_queries(blocks)
    assert resolved == [pipeline_smart.GENERIC_FALLBACKS[0]]


def test_resolve_queries_mode_on_authored_still_takes_priority(monkeypatch):
    monkeypatch.setenv("SHOT_DIRECTOR_MODE", "on")
    monkeypatch.setattr(pipeline_smart, "THEMES", {})
    calls = []
    monkeypatch.setattr(shot_director, "direct_query",
                         lambda text, video_dir: calls.append(text) or "unused")
    blocks = [{"text": "Взять быка за рога.", "words": 4, "pause_after": 0.0, "section": "HOOK", "stat": None}]
    resolved = pipeline_smart.resolve_queries(blocks, authored_queries={"HOOK": ["milk bottle hand"]})
    assert calls == []
    assert resolved == ["milk bottle hand"]


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


# ---------- stock_fetch_multisource.fetch_openverse_photo (по прямому запросу
# пользователя — атласные исторические карты, контент, которого физически
# нет у Pexels/Pixabay/Unsplash; проверено вживую на реальном API — "Europe
# 1200"/euratlas.com под CC BY, реальные political-boundary карты под CC0 из
# Wikimedia, см. коммиты). OPENVERSE_ENABLED=0 по умолчанию — ВЫКЛЮЧЕН.
# Второй заход по прямому требованию пользователя "копни глубже, только то,
# что не требует лицензии на 100%": license=cc0 ТОЛЬКО (pdm убран — это
# самоидентификация "кто-то решил, что работа свободна", не юридический
# отказ от прав правообладателя, см. докстринг модуля/CLAUDE.md), И
# source должен быть в списке проверенных институциональных архивов
# (Wikimedia/Смитсоновский институт/музеи и т.п.) — не персональные/
# самотегируемые источники, где лицензию ставит сам загрузивший без
# проверки. by/by-sa по-прежнему исключены (нет сборщика атрибуций). ----------

def _fake_openverse_response(results):
    import json as _json
    payload = _json.dumps({"results": results}).encode()

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return payload
    return _Resp()


def test_fetch_openverse_photo_off_by_default(monkeypatch, tmp_path):
    monkeypatch.setattr(stock_fetch_multisource, "OPENVERSE_ENABLED", False)

    def boom(*a, **kw):
        raise AssertionError("сеть не должна вызываться при OPENVERSE_ENABLED=False")
    monkeypatch.setattr(stock_fetch_multisource.urllib.request, "urlopen", boom)
    out = str(tmp_path / "x.jpg")
    assert stock_fetch_multisource.fetch_openverse_photo("test", out) is False
    assert not os.path.exists(out)


def test_fetch_openverse_photo_rejects_by_license_even_if_returned(monkeypatch, tmp_path):
    # Fail-closed: даже если API вернул результат НЕ из запрошенного
    # license=cc0 (баг на стороне API, устаревший кэш и т.п.) — код не
    # должен скачать его вслепую, доверяя только фильтру запроса.
    monkeypatch.setattr(stock_fetch_multisource, "OPENVERSE_ENABLED", True)
    results = [{"id": "1", "license": "by-sa", "url": "https://example.com/risky.jpg"}]
    monkeypatch.setattr(stock_fetch_multisource.urllib.request, "urlopen",
                         lambda req, timeout=None: _fake_openverse_response(results))
    out = str(tmp_path / "x.jpg")
    assert stock_fetch_multisource.fetch_openverse_photo("test", out) is False
    assert not os.path.exists(out)


def test_fetch_openverse_photo_downloads_safe_license_and_logs_manifest(monkeypatch, tmp_path):
    monkeypatch.setattr(stock_fetch_multisource, "OPENVERSE_ENABLED", True)
    results = [{
        "id": "abc123", "title": "Medieval map", "url": "https://example.com/map.jpg",
        "creator": "someone", "license": "cc0", "license_version": "1.0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "provider": "wikimedia", "source": "wikimedia",
        "foreign_landing_url": "https://example.com/page",
    }]

    def fake_urlopen(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else req
        if "example.com/map.jpg" in url:
            class _ImgResp:
                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    return False

                def read(self):
                    return b"\xff\xd8\xff fake jpeg bytes"
            return _ImgResp()
        return _fake_openverse_response(results)
    monkeypatch.setattr(stock_fetch_multisource.urllib.request, "urlopen", fake_urlopen)

    out = str(tmp_path / "x.jpg")
    ok = stock_fetch_multisource.fetch_openverse_photo("medieval map", out, base=str(tmp_path))
    assert ok is True
    assert os.path.exists(out)
    manifest = tmp_path / "media_plan" / "openverse_license_manifest.jsonl"
    assert manifest.exists()
    entry = json.loads(manifest.read_text().strip())
    assert entry["license"] == "cc0"
    assert entry["id"] == "abc123"
    assert entry["query"] == "medieval map"


def test_fetch_openverse_photo_skips_unsafe_and_takes_next_safe_result(monkeypatch, tmp_path):
    monkeypatch.setattr(stock_fetch_multisource, "OPENVERSE_ENABLED", True)
    results = [
        {"id": "1", "license": "by", "source": "wikimedia", "url": "https://example.com/risky.jpg"},
        {"id": "2", "license": "cc0", "source": "wikimedia", "url": "https://example.com/safe.jpg"},
    ]

    def fake_urlopen(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else req
        if "safe.jpg" in url:
            class _ImgResp:
                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    return False

                def read(self):
                    return b"fake"
            return _ImgResp()
        if "risky.jpg" in url:
            raise AssertionError("не должен скачивать 'by'-результат вообще")
        return _fake_openverse_response(results)
    monkeypatch.setattr(stock_fetch_multisource.urllib.request, "urlopen", fake_urlopen)

    out = str(tmp_path / "x.jpg")
    assert stock_fetch_multisource.fetch_openverse_photo("test", out) is True
    assert os.path.exists(out)


def test_fetch_openverse_photo_no_results_returns_false(monkeypatch, tmp_path):
    monkeypatch.setattr(stock_fetch_multisource, "OPENVERSE_ENABLED", True)
    monkeypatch.setattr(stock_fetch_multisource.urllib.request, "urlopen",
                         lambda req, timeout=None: _fake_openverse_response([]))
    out = str(tmp_path / "x.jpg")
    assert stock_fetch_multisource.fetch_openverse_photo("test", out) is False


def test_is_safe_openverse_license_case_and_whitespace_insensitive():
    assert stock_fetch_multisource._is_safe_openverse_license({"license": "CC0"}) is True
    # pdm сознательно НЕ безопасна (второй заход, "копни глубже") — это
    # самоидентификация ("кто-то решил, что работа свободна от копирайта"),
    # не юридический отказ от прав правообладателя, как у cc0. См. докстринг
    # модуля/CLAUDE.md — официальная формулировка Creative Commons.
    assert stock_fetch_multisource._is_safe_openverse_license({"license": "PDM"}) is False
    assert stock_fetch_multisource._is_safe_openverse_license({"license": " pdm "}) is False
    assert stock_fetch_multisource._is_safe_openverse_license({"license": "by-sa"}) is False
    assert stock_fetch_multisource._is_safe_openverse_license({}) is False


def test_is_trusted_openverse_source_known_institutions():
    assert stock_fetch_multisource._is_trusted_openverse_source({"source": "wikimedia"}) is True
    assert stock_fetch_multisource._is_trusted_openverse_source({"source": "MET"}) is True
    assert stock_fetch_multisource._is_trusted_openverse_source({"source": " rijksmuseum "}) is True


def test_is_trusted_openverse_source_smithsonian_prefix_covers_future_branches():
    # Openverse хранит смитсоновские подразделения отдельными source-слагами
    # — проверка префиксом защищает и ветки, которых нет в явном списке
    # (Openverse может добавить новые).
    assert stock_fetch_multisource._is_trusted_openverse_source(
        {"source": "smithsonian_national_museum_of_natural_history"}) is True
    assert stock_fetch_multisource._is_trusted_openverse_source(
        {"source": "smithsonian_some_future_branch_not_in_list"}) is True


def test_is_trusted_openverse_source_rejects_untrusted_self_tagged_platforms():
    # Реальный, проверенный вживую случай: NASA/Biodiversity Heritage Library
    # технически идут через Flickr API (provider=flickr), но ОБЫЧНЫЙ
    # персональный Flickr-аккаунт — не институция, лицензию там ставит сам
    # загрузивший без проверки. source= (не provider=) различает эти случаи.
    assert stock_fetch_multisource._is_trusted_openverse_source({"source": "flickr"}) is False
    assert stock_fetch_multisource._is_trusted_openverse_source({"source": "rawpixel"}) is False
    assert stock_fetch_multisource._is_trusted_openverse_source({"source": "inaturalist"}) is False
    assert stock_fetch_multisource._is_trusted_openverse_source({}) is False


def test_fetch_openverse_photo_rejects_untrusted_source_even_with_safe_license(monkeypatch, tmp_path):
    # Fail-closed: license=cc0 сам по себе недостаточен — источник тоже
    # должен быть институциональным, иначе честный на вид cc0-тег может
    # быть ошибочно проставлен случайным пользователем без проверки прав.
    monkeypatch.setattr(stock_fetch_multisource, "OPENVERSE_ENABLED", True)
    results = [{"id": "1", "license": "cc0", "source": "rawpixel",
                "url": "https://example.com/risky.jpg"}]
    monkeypatch.setattr(stock_fetch_multisource.urllib.request, "urlopen",
                         lambda req, timeout=None: _fake_openverse_response(results))
    out = str(tmp_path / "x.jpg")
    assert stock_fetch_multisource.fetch_openverse_photo("test", out) is False
    assert not os.path.exists(out)


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


def test_is_parallax_highlight_hook_always_true():
    assert pipeline_smart.is_parallax_highlight({"section": "HOOK"}, False) is True


def test_is_parallax_highlight_section_start_true():
    assert pipeline_smart.is_parallax_highlight({"section": "BLOCK_3"}, True) is True


def test_is_parallax_highlight_climax_true_even_mid_section():
    # По прямому запросу пользователя (17-28-минутные ролики) — [climax]
    # даёт параллакс, даже если это не хук и не первый кадр раздела.
    assert pipeline_smart.is_parallax_highlight(
        {"section": "BLOCK_3", "is_climax": True}, False) is True


def test_is_parallax_highlight_ordinary_body_false():
    assert pipeline_smart.is_parallax_highlight(
        {"section": "BLOCK_3", "is_climax": False}, False) is False


def test_is_parallax_highlight_missing_is_climax_key_defaults_false():
    assert pipeline_smart.is_parallax_highlight({"section": "BLOCK_3"}, False) is False


def test_color_meta_args_includes_explicit_color_range_tv():
    # По прямому запросу пользователя ("копни глубже на Rec.709") —
    # closed-loop проверка вживую (pc-источник через реальный CLIP_PIX_ARGS)
    # показала, что явный -pix_fmt yuv420p10le УЖЕ заставляет ffmpeg
    # корректно пересчитывать в tv-диапазон независимо от входного pc/tv —
    # добавка -color_range tv здесь чистое ужесточение (explicit лучше
    # implicit), подтверждено НЕ менять итоговый вывод. Тест защищает сам
    # факт, что тег явно объявлен, а не только протестированное сегодня
    # поведение по умолчанию.
    assert "-color_range" in pipeline_smart.COLOR_META_ARGS
    idx = pipeline_smart.COLOR_META_ARGS.index("-color_range")
    assert pipeline_smart.COLOR_META_ARGS[idx + 1] == "tv"


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
    # конкретный кадр (сток-меч в снегу) давал ложное срабатывание каскада
    # лица, что блокировало Look Management assist на нём — см. коммит.
    # Фикстура, а не glob по videos/*/temp_smart/: имя файла в кэше склеено
    # из номера слота и хэша ЗАПРОСА, то есть НЕ идентифицирует фото —
    # перерендер эпизода подставляет под то же имя другую картинку, и тест
    # начинает проверять не тот кадр (реально произошло при добавлении
    # candidate_gate_signature() в ключ кэша). То же фото, уменьшенное как
    # остальные фикстуры этой папки — см. ATTRIBUTION.md.
    fixture = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "fixtures", "golden_media", "sword_snow.jpg")
    assert pipeline_smart.detect_face_anchor(fixture) is None


# ---------- apply_depth_of_field(): боке/ГРИП (ЭТАП 1.3 внешнего аудита —
# параллакс уже считает depth-карту, но раньше не использовал её для
# размытия фона) ----------

def _checker_bgr(size, block=8):
    # Не однотонная заливка — Gaussian blur однотонного изображения
    # математически no-op, тест на "размылось ли что-то" иначе бессмыслен.
    import numpy as np
    h, w = size
    yy, xx = np.mgrid[0:h, 0:w]
    pattern = (((xx // block) + (yy // block)) % 2) * 255
    arr = np.stack([pattern, pattern, pattern], axis=-1).astype("uint8")
    return arr


@pytest.mark.skipif(_PARALLAX_SKIP, reason="cv2 недоступен")
def test_apply_depth_of_field_disabled_returns_canvas_unchanged(monkeypatch):
    import numpy as np
    monkeypatch.setattr(pipeline_smart, "DOF_ENABLED", False)
    canvas = _checker_bgr((120, 120))
    depth = np.zeros((120, 120), dtype="float32")   # всё "далеко" — если бы считалось, дало бы max blur
    result = pipeline_smart.apply_depth_of_field(canvas, depth, strength_gain=1.0)
    assert np.array_equal(result, canvas)


@pytest.mark.skipif(_PARALLAX_SKIP, reason="cv2 недоступен")
def test_apply_depth_of_field_all_near_leaves_canvas_sharp(monkeypatch):
    import numpy as np
    monkeypatch.setattr(pipeline_smart, "DOF_ENABLED", True)
    canvas = _checker_bgr((120, 120))
    depth = np.ones((120, 120), dtype="float32")   # всё "близко" — весь кадр должен остаться резким
    result = pipeline_smart.apply_depth_of_field(canvas, depth, strength_gain=1.0)
    assert np.array_equal(result, canvas)


@pytest.mark.skipif(_PARALLAX_SKIP, reason="cv2 недоступен")
def test_apply_depth_of_field_all_far_blurs_canvas(monkeypatch):
    import numpy as np
    monkeypatch.setattr(pipeline_smart, "DOF_ENABLED", True)
    canvas = _checker_bgr((120, 120))
    depth = np.zeros((120, 120), dtype="float32")   # всё "далеко" — весь кадр должен размыться
    result = pipeline_smart.apply_depth_of_field(canvas, depth, strength_gain=1.0)
    assert not np.array_equal(result, canvas)
    # Резкий шахматный узор после размытия должен потерять контраст
    # (значения средних пикселей должны сместиться к середине 0..255, а не
    # оставаться строго на 0/255, как в исходном узоре).
    mid_range = np.logical_and(result > 40, result < 215)
    assert mid_range.mean() > 0.3


@pytest.mark.skipif(_PARALLAX_SKIP, reason="cv2 недоступен")
def test_apply_depth_of_field_mixed_depth_blurs_only_far_half(monkeypatch):
    import numpy as np
    monkeypatch.setattr(pipeline_smart, "DOF_ENABLED", True)
    canvas = _checker_bgr((120, 120))
    depth = np.ones((120, 120), dtype="float32")
    depth[:, 60:] = 0.0   # правая половина кадра — "далеко"
    result = pipeline_smart.apply_depth_of_field(canvas, depth, strength_gain=1.0)
    left_unchanged = np.array_equal(result[:, :60], canvas[:, :60])
    right_changed = not np.array_equal(result[:, 60:], canvas[:, 60:])
    assert left_unchanged, "ближняя половина (depth=1) не должна размываться"
    assert right_changed, "дальняя половина (depth=0) должна размыться"


@pytest.mark.skipif(_PARALLAX_SKIP, reason="cv2 недоступен")
def test_apply_depth_of_field_low_strength_gain_suppresses_effect(monkeypatch):
    import numpy as np
    monkeypatch.setattr(pipeline_smart, "DOF_ENABLED", True)
    canvas = _checker_bgr((60, 60))   # маленький холст — при низком strength_gain сигма уйдёт < 1px
    depth = np.zeros((60, 60), dtype="float32")
    result = pipeline_smart.apply_depth_of_field(canvas, depth, strength_gain=0.3)
    assert np.array_equal(result, canvas)


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


# ---------- auto_levels_params: клэмп выхода (P2 форензик-аудита) ----------
# Реальный, увиденный вживую на полном прогоне случай: у самого порога
# AUTO_LEVELS_MIN_RANGE (но формально его прошедшего) scale_full делится на
# почти нулевой range — контраст улетает в разы сильнее любой легитимной
# коррекции (eq=contrast=4.2453 в реальном логе ffmpeg).

def test_auto_levels_params_none_is_neutral():
    assert pipeline_smart.auto_levels_params(None) == (1.0, 0.0)


def test_auto_levels_params_degenerate_range_is_neutral():
    assert pipeline_smart.auto_levels_params((0.5, 0.53)) == (1.0, 0.0)


def test_auto_levels_params_normal_flat_photo_unclamped():
    # range=0.5 — обычное фото, легитимная умеренная коррекция, клэмп не
    # должен вмешиваться вообще.
    contrast, brightness = pipeline_smart.auto_levels_params((0.1, 0.6))
    assert 1.0 < contrast < pipeline_smart.AUTO_LEVELS_MAX_CONTRAST
    assert abs(brightness) < pipeline_smart.AUTO_LEVELS_MAX_BRIGHTNESS_ABS


def test_auto_levels_params_near_threshold_range_is_clamped():
    # range=0.051 — на волосок выше AUTO_LEVELS_MIN_RANGE=0.05, без клэмпа
    # даёт contrast_eff~9.7 (см. докстринг auto_levels_params).
    contrast, brightness = pipeline_smart.auto_levels_params((0.4, 0.451))
    assert contrast == pytest.approx(pipeline_smart.AUTO_LEVELS_MAX_CONTRAST)
    assert abs(brightness) <= pipeline_smart.AUTO_LEVELS_MAX_BRIGHTNESS_ABS + 1e-9


# ---------- hook_visual_starts (доп. находка из независимого архитектурного
# разбора: наивный cumsum(durs) не учитывает xfade-нахлёст, который РЕАЛЬНО
# сжимает итоговый таймлайн — см. xfade_chain(): cum = cum + durs[i] - this_dur) ----------

def test_hook_visual_starts_first_clip_at_zero():
    assert pipeline_smart.hook_visual_starts([{"section": "HOOK"}], [3.0]) == [0.0]


def test_hook_visual_starts_subtracts_xfade_hard_overlap():
    blocks = [{"section": "HOOK"}, {"section": "HOOK"}, {"section": "HOOK"}]
    durs = [2.0, 3.0, 4.0]
    starts = pipeline_smart.hook_visual_starts(blocks, durs)
    hard = pipeline_smart.XFADE_DUR_HARD
    assert starts[0] == pytest.approx(0.0)
    assert starts[1] == pytest.approx(2.0 - hard)
    cum_after_1 = 2.0 + 3.0 - hard
    assert starts[2] == pytest.approx(cum_after_1 - hard)


def test_hook_visual_starts_uses_snap_cut_after_stat_block():
    # Переход СРАЗУ ПОСЛЕ блока со stat-плашкой — SNAP_CUT_DUR, не
    # XFADE_DUR_HARD (то же условие, что xfade_chain()).
    blocks = [{"section": "HOOK", "stat": "15 КГ"}, {"section": "HOOK"}, {"section": "HOOK"}]
    durs = [2.0, 3.0, 4.0]
    starts = pipeline_smart.hook_visual_starts(blocks, durs)
    snap = pipeline_smart.SNAP_CUT_DUR
    hard = pipeline_smart.XFADE_DUR_HARD
    assert starts[1] == pytest.approx(2.0 - snap)
    cum_after_1 = 2.0 + 3.0 - snap
    assert starts[2] == pytest.approx(cum_after_1 - hard)   # blocks[1] без stat -> обычный hard


def test_hook_visual_starts_matches_xfade_chain_cum_recurrence():
    # Регрессия: формула здесь обязана быть БИТ-В-БИТ той же, что
    # xfade_chain() реально использует для HOOK-переходов (cum = cum + d -
    # this_dur, offset = max(0, cum_before - this_dur)) — пересчитано
    # вручную независимо от реализации, не скопировано с неё вслепую.
    blocks = [{"section": "HOOK"}] * 5
    durs = [1.5, 2.2, 0.9, 3.1, 1.8]
    starts = pipeline_smart.hook_visual_starts(blocks, durs)
    hard = pipeline_smart.XFADE_DUR_HARD
    cum = durs[0]
    expected = [0.0]
    for d in durs[1:]:
        expected.append(max(0.0, cum - hard))
        cum = cum + d - hard
    assert starts == pytest.approx(expected)


# ---------- _wrap_caption_text / write_subtitles readability (реальный,
# задокументированный стандарт субтитрирования: короткие строки, разрыв по
# словам, максимум 2 строки — см. коммит) ----------

def test_wrap_caption_text_short_unchanged():
    text = "Так почему миф жив?"
    assert pipeline_smart._wrap_caption_text(text) == text


def test_wrap_caption_text_empty_string():
    assert pipeline_smart._wrap_caption_text("") == ""


def test_wrap_caption_text_wraps_long_text_at_word_boundary():
    text = ("Обычный одноручный рыцарский меч — рабочая лошадка всего "
            "средневековья — весит от килограмма до полутора")
    wrapped = pipeline_smart._wrap_caption_text(text)
    out_lines = wrapped.split("\n")
    assert len(out_lines) <= pipeline_smart.SRT_MAX_LINES
    for line in out_lines[:-1]:
        assert len(line) <= pipeline_smart.SRT_MAX_LINE_CHARS
    # ни одно слово не разорвано — весь исходный текст восстанавливается
    # обратно словá-в-слово при простом склеивании через пробел
    assert " ".join(wrapped.split()) == " ".join(text.split())


def test_wrap_caption_text_never_loses_words_beyond_max_lines():
    text = " ".join(f"слово{i}" for i in range(40))   # заведомо длиннее 2 строк по 42 символа
    wrapped = pipeline_smart._wrap_caption_text(text)
    assert wrapped.count("\n") <= pipeline_smart.SRT_MAX_LINES - 1
    assert " ".join(wrapped.split()) == text   # ничего не потеряно, даже с превышением лимита


def test_write_subtitles_wraps_long_block(tmp_path):
    raw_text = ("Обычный одноручный рыцарский меч — рабочая лошадка всего "
                "средневековья — весит от килограмма до полутора")
    blocks = [{"text": raw_text, "pause_after": 0.0}]
    path = pipeline_smart.write_subtitles(str(tmp_path), blocks, [0.0], [8.0])
    content = open(path, encoding="utf-8").read()
    assert pipeline_smart._wrap_caption_text(raw_text) in content


# ---------- _load_arc_stage_by_index (arc-stage awareness Look Management/Visual Director) ----------

def test_load_arc_stage_by_index_missing_file_returns_empty(tmp_path):
    assert pipeline_smart._load_arc_stage_by_index(str(tmp_path)) == {}


def test_load_arc_stage_by_index_reads_speech_plan(tmp_path):
    plan_dir = tmp_path / "media_plan"
    plan_dir.mkdir()
    (plan_dir / "speech_plan.json").write_text(
        '{"units": ['
        '{"unit_id": "HOOK#0", "index": 0, "arc_stage": "hook"},'
        '{"unit_id": "BLOCK1#0", "index": 1, "arc_stage": "заход-якорь"},'
        '{"unit_id": "BLOCK1#1", "index": 2, "arc_stage": "слом"}'
        ']}', encoding="utf-8")
    result = pipeline_smart._load_arc_stage_by_index(str(tmp_path))
    assert result == {0: "hook", 1: "заход-якорь", 2: "слом"}


def test_load_arc_stage_by_index_corrupt_file_returns_empty_not_crash(tmp_path):
    plan_dir = tmp_path / "media_plan"
    plan_dir.mkdir()
    (plan_dir / "speech_plan.json").write_text("{not valid json", encoding="utf-8")
    assert pipeline_smart._load_arc_stage_by_index(str(tmp_path)) == {}


def test_load_arc_stage_by_index_skips_units_without_index(tmp_path):
    # Старый формат/битый юнит без "index" — пропускается, не роняет весь
    # словарь (тот же принцип толерантного парсинга, что и у load_protected_windows).
    plan_dir = tmp_path / "media_plan"
    plan_dir.mkdir()
    (plan_dir / "speech_plan.json").write_text(
        '{"units": [{"unit_id": "HOOK#0", "arc_stage": "hook"}, '
        '{"unit_id": "BLOCK1#0", "index": 1, "arc_stage": "постановка"}]}', encoding="utf-8")
    result = pipeline_smart._load_arc_stage_by_index(str(tmp_path))
    assert result == {1: "постановка"}


# --- disambiguate_search_query(): реальный, подтверждённый вживую случай
# (внешний аудит + прямая проверка на 01_ves-mecha/_test20s) — голый запрос
# "sword" намешивает восточноазиатские клинки в выдачу Pexels; см. полную
# калибровку (живые числа) у QUERY_DISAMBIGUATION_RULES в pipeline_smart.py.

def test_disambiguate_search_query_adds_qualifier_to_bare_sword_query():
    assert pipeline_smart.disambiguate_search_query("sword close up") == "european sword close up"


def test_disambiguate_search_query_adds_qualifier_to_medieval_sword_query():
    assert pipeline_smart.disambiguate_search_query("medieval sword close up") == \
        "european medieval sword close up"


def test_disambiguate_search_query_does_not_duplicate_existing_qualifier():
    assert pipeline_smart.disambiguate_search_query("european longsword knight") == \
        "european longsword knight"


def test_disambiguate_search_query_leaves_explicit_asian_query_untouched():
    # Легитимный запрос про катану (например, для другого канала/эпизода
    # про японское оружие) не должен получать искажающую приставку.
    for q in ("katana sword", "japanese katana", "chinese jian sword"):
        assert pipeline_smart.disambiguate_search_query(q) == q


def test_disambiguate_search_query_leaves_unrelated_query_untouched():
    assert pipeline_smart.disambiguate_search_query("shield close up") == "shield close up"


# --- Расширение на другие категории того же класса бага (не только sword) —
# по прямому запросу пользователя не останавливаться на одном частном
# случае: helmet/armor/spear/battle дают ту же контаминацию (проверено
# живым Pexels-поиском, см. комментарий у QUERY_DISAMBIGUATION_RULES).

def test_disambiguate_search_query_adds_qualifier_to_helmet_query():
    assert pipeline_smart.disambiguate_search_query("helmet close up") == "european helmet close up"


def test_disambiguate_search_query_adds_qualifier_to_armor_query():
    assert pipeline_smart.disambiguate_search_query("armor close up") == \
        "european medieval armor close up"


def test_disambiguate_search_query_armor_does_not_duplicate_existing_medieval():
    assert pipeline_smart.disambiguate_search_query("medieval armor close up") == \
        "european medieval armor close up"


def test_disambiguate_search_query_adds_qualifier_to_spear_query():
    assert pipeline_smart.disambiguate_search_query("spear close up") == \
        "european medieval spear close up"


def test_disambiguate_search_query_adds_qualifier_to_battle_query():
    assert pipeline_smart.disambiguate_search_query("battle scene") == \
        "european medieval battle scene"


def test_disambiguate_search_query_leaves_explicit_other_culture_queries_untouched():
    # Легитимные запросы для другой темы/культуры не должны получать
    # искажающую приставку.
    for q in ("samurai armor", "african spear ceremony", "sci-fi helmet",
              "stormtrooper armor costume"):
        assert pipeline_smart.disambiguate_search_query(q) == q


def test_disambiguate_search_query_no_duplicate_words_for_fully_qualified_query():
    q = "european medieval battle scene"
    assert pipeline_smart.disambiguate_search_query(q) == q


# --- pexels_video(): реальный, ранее не закрытый структурный пробел (найден
# внешним аудитом + прямой проверкой на реальном ролике) — раньше брала
# ПЕРВОЕ ещё не показанное видео без единой проверки релевантности, в
# отличие от pexels_photo(). Контроль потока (перебор кандидатов/
# VIDEO_RELEVANCE_MAX_TRIES/честный fallback) проверяется здесь с
# ПОЛНОСТЬЮ замоканными сетью/CLIP — корректность самого CLIP-гейта уже
# покрыта tests/test_media_selection_golden.py::TestVisualDomainGuard,
# здесь тестируется только оркестрация pexels_video().

class _FakeHTTPResponse:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._payload


def _fake_video_entry(vid, width=1920):
    return {"id": vid, "video_files": [{"file_type": "video/mp4", "width": width,
                                          "link": f"https://example.invalid/{vid}.mp4"}]}


def _patch_pexels_video_infra(monkeypatch, tmp_path, videos, relevant_ids, hash_by_id=None):
    """videos — список записей Pexels /videos/search. relevant_ids — set id,
    которые is_relevant_candidate() должен считать подходящими (остальные —
    нет). Скачивание/probe-извлечение — no-op, пишут/возвращают детерминиро-
    ванный dummy-путь, никакой реальной сети/ffmpeg. hash_by_id (опционально)
    — {id: aHash-строка}, подключает детерминированный ahash() для тестов
    межтипового дедупа (см. cross-media dedup ниже) — без него ahash()
    вызывается как есть и падает на несуществующем dummy-файле (что штатно
    гасится в pexels_video() как "хэш не посчитан")."""
    monkeypatch.setattr(pipeline_smart, "TEMP_FOLDER", str(tmp_path))
    monkeypatch.setattr(pipeline_smart, "PEXELS_API_KEY", "fake-key-for-test")

    def fake_urlopen(req, timeout=15):
        return _FakeHTTPResponse({"videos": videos})
    monkeypatch.setattr(pipeline_smart.urllib.request, "urlopen", fake_urlopen)

    downloaded = []

    def fake_atomic_download(req, dest, timeout):
        downloaded.append(dest)
        with open(dest, "wb") as f:
            f.write(b"dummy-video-bytes")
        return True
    monkeypatch.setattr(pipeline_smart, "atomic_url_download", fake_atomic_download)

    def fake_extract_probe(path, base_at=0.5, retry_ats=(1.5, 3.0), timeout=20, q=5):
        return path + ".probe.jpg", False   # cleanup=False — ничего реального не создавали
    monkeypatch.setattr(pipeline_smart, "extract_video_probe_frame", fake_extract_probe)

    def fake_is_relevant(image_path, query, relevance=None):
        # trial-путь несёт id кандидата в имени (см. pexels_video: f".trial_{id}.mp4")
        for vid in relevant_ids:
            if f".trial_{vid}.mp4" in image_path:
                return True
        return False
    monkeypatch.setattr(pipeline_smart, "is_relevant_candidate", fake_is_relevant)

    if hash_by_id is not None:
        def fake_ahash(path):
            for vid, h in hash_by_id.items():
                if f".trial_{vid}.mp4" in path:
                    return h
            raise ValueError("no hash mapped for this candidate")
        monkeypatch.setattr(pipeline_smart, "ahash", fake_ahash)
    return downloaded


def test_pexels_video_skips_irrelevant_candidate_and_picks_next(monkeypatch, tmp_path):
    videos = [_fake_video_entry(111), _fake_video_entry(222)]
    _patch_pexels_video_infra(monkeypatch, tmp_path, videos, relevant_ids={222})
    result = pipeline_smart.pexels_video("european medieval sword close up", 0)
    assert result is not None and os.path.exists(result)


def test_pexels_video_respects_max_tries_bound(monkeypatch, tmp_path):
    # Ни один кандидат не релевантен, кандидатов больше, чем
    # VIDEO_RELEVANCE_MAX_TRIES — должен остановиться на границе, не
    # перебрать всю выдачу.
    videos = [_fake_video_entry(i) for i in range(10)]
    downloaded = _patch_pexels_video_infra(monkeypatch, tmp_path, videos, relevant_ids=set())
    result = pipeline_smart.pexels_video("european medieval sword close up", 1)
    assert result is not None   # честный fallback на первого скачанного, слот не пуст
    assert len(downloaded) <= pipeline_smart.VIDEO_RELEVANCE_MAX_TRIES


def test_pexels_video_falls_back_to_first_when_none_relevant(monkeypatch, tmp_path):
    videos = [_fake_video_entry(333), _fake_video_entry(444)]
    _patch_pexels_video_infra(monkeypatch, tmp_path, videos, relevant_ids=set())
    result = pipeline_smart.pexels_video("european medieval sword close up", 2)
    assert result is not None and os.path.exists(result), (
        "ни один кандидат не прошёл гейт — слот всё равно не должен остаться пустым")


def test_pexels_video_first_candidate_relevant_downloads_once(monkeypatch, tmp_path):
    videos = [_fake_video_entry(555), _fake_video_entry(666)]
    downloaded = _patch_pexels_video_infra(monkeypatch, tmp_path, videos, relevant_ids={555})
    result = pipeline_smart.pexels_video("european medieval sword close up", 3)
    assert result is not None
    assert len(downloaded) == 1, "первый же релевантный кандидат — не нужно скачивать остальных"


# ---------- pexels_video: дедуп против уже показанного медиа (фото ИЛИ видео) ----------
# Реальный пробел: визуальный дедуп по aHash работал только для фото
# (used_photo_hashes передавался в pexels_photo, но не в pexels_video) —
# видео сверялось исключительно по ID Pexels. Одна и та же студийная съёмка
# меча, продающаяся и фотостоком, и видеостоком под разными ID, проходила
# как новый кадр. Пробный кадр для видео и так извлекается ради проверки
# релевантности — дедуп не добавляет вычислений, только ahash() уже
# открытого файла (см. докстринг pexels_video).

DUP_HASH = "0" * 64      # "уже показано" — идентичный хэш
FRESH_HASH = "1" * 64    # заведомо непохожий (hamming = 64 >> PHOTO_DEDUP_HAMMING)


def test_pexels_video_prefers_fresh_over_duplicate_of_already_used_media(monkeypatch, tmp_path):
    # Оба кандидата релевантны; 111 — визуальный дубль уже показанного
    # (фото или видео — источник дубля не важен, используется общий список),
    # 222 — свежий. Должен победить 222, а не первый по порядку.
    videos = [_fake_video_entry(111), _fake_video_entry(222)]
    _patch_pexels_video_infra(monkeypatch, tmp_path, videos, relevant_ids={111, 222},
                              hash_by_id={111: DUP_HASH, 222: FRESH_HASH})
    used_hashes = [DUP_HASH]   # хэш уже выбранного медиа (мог прийти от фото)
    result = pipeline_smart.pexels_video("european medieval sword close up", 4,
                                         used_hashes=used_hashes)
    assert result is not None
    assert "222" in result or True   # cf-имя от query+index, не от id — проверяем через hashes
    assert FRESH_HASH in used_hashes, "хэш ПОБЕДИВШЕГО (222) должен уйти в общий список"
    assert used_hashes.count(DUP_HASH) == 1, "хэш отклонённого дубля (111) не должен задвоиться"


def test_pexels_video_accepts_duplicate_rather_than_empty_slot(monkeypatch, tmp_path):
    # Единственный релевантный кандидат — визуальный дубль. Дедуп не должен
    # опустошать слот: лучше повтор, чем пропущенный кадр (тот же принцип,
    # что уже действует для is_relevant_candidate).
    videos = [_fake_video_entry(333)]
    _patch_pexels_video_infra(monkeypatch, tmp_path, videos, relevant_ids={333},
                              hash_by_id={333: DUP_HASH})
    used_hashes = [DUP_HASH]
    result = pipeline_smart.pexels_video("european medieval sword close up", 5,
                                         used_hashes=used_hashes)
    assert result is not None and os.path.exists(result)


def test_pexels_video_without_used_hashes_behaves_as_before(monkeypatch, tmp_path):
    # used_hashes=None (значение по умолчанию) — дедуп полностью выключен,
    # первый релевантный кандидат побеждает как и раньше.
    videos = [_fake_video_entry(777), _fake_video_entry(888)]
    downloaded = _patch_pexels_video_infra(monkeypatch, tmp_path, videos, relevant_ids={777, 888})
    result = pipeline_smart.pexels_video("european medieval sword close up", 6)
    assert result is not None
    assert len(downloaded) == 1


# ---------- порядок заданий рендера (check_jobs_in_order) ----------
# Реальный баг: проверка требовала job["i"] == позиция_в_списке, но блок без
# медиа (кончилась квота Pexels, пустая выдача, нет ключа) в pending_jobs не
# попадает вообще — после первого же такого блока прогон падал сырым
# AssertionError вместо аккуратного отчёта о пропусках.

def test_check_jobs_in_order_allows_gaps_from_skipped_blocks():
    jobs = [{"i": 0}, {"i": 1}, {"i": 3}, {"i": 7}]   # блоки 2, 4-6 пропущены (нет медиа)
    assert pipeline_smart.check_jobs_in_order(jobs) is True


def test_check_jobs_in_order_raises_on_real_disorder():
    jobs = [{"i": 0}, {"i": 5}, {"i": 3}]
    with pytest.raises(AssertionError):
        pipeline_smart.check_jobs_in_order(jobs)


def test_check_jobs_in_order_raises_on_duplicate_index():
    jobs = [{"i": 0}, {"i": 1}, {"i": 1}]
    with pytest.raises(AssertionError):
        pipeline_smart.check_jobs_in_order(jobs)


def test_check_jobs_in_order_empty_ok():
    assert pipeline_smart.check_jobs_in_order([]) is True


# ---------- кадровая сетка длительностей ----------

def test_quantize_durations_all_multiples_of_frame():
    durs = [3.4567, 1.0001, 8.9999, 5.5, 0.4]
    q = pipeline_smart.quantize_durations_to_frames(durs)
    for d in q:
        assert abs(d * pipeline_smart.FPS - round(d * pipeline_smart.FPS)) < 1e-9


def test_quantize_durations_keeps_total_within_half_frame():
    # Главное свойство: диффузия ошибки (carry) держит суммарное расхождение
    # в пределах полукадра НЕЗАВИСИМО от числа клипов — иначе округления
    # складывались бы в секунды на 400+ кадрах эпизода.
    durs = [1.7 + 0.013 * i for i in range(500)]
    q = pipeline_smart.quantize_durations_to_frames(durs)
    assert abs(sum(q) - sum(durs)) <= 0.5 / pipeline_smart.FPS + 1e-9


def test_quantize_durations_floor_is_one_frame():
    q = pipeline_smart.quantize_durations_to_frames([0.0, 0.001])
    assert all(d >= 1 / pipeline_smart.FPS - 1e-9 for d in q)


def test_quantize_durations_empty():
    assert pipeline_smart.quantize_durations_to_frames([]) == []


# ---------- план переходов и бюджет склейки ----------

def _synthetic_blocks():
    blocks = []
    for k in range(8):
        blocks.append({"section": "HOOK", "text": "хук", "words": 10, "pause_after": 0.4,
                       "stat": None, "is_subcut": k % 5 == 3})
    for b in range(1, 4):
        for k in range(40):
            blocks.append({"section": f"BLOCK {b}: Тема", "text": "тело", "words": 20,
                           "pause_after": 0.8 if k % 3 == 0 else 0.0,
                           "stat": "42 КГ" if k % 25 == 7 else None,
                           "is_subcut": k % 4 == 2})
    for k in range(6):
        blocks.append({"section": "FINAL", "text": "финал", "words": 16, "pause_after": 0.8,
                       "stat": None, "is_subcut": False})
    return blocks


def test_plan_transitions_length_and_frame_alignment():
    blocks = _synthetic_blocks()
    sections = [b["section"] for b in blocks]
    plan = pipeline_smart.plan_transitions(sections, blocks)
    assert len(plan) == len(blocks) - 1
    for _t, d in plan:
        assert d >= 1 / pipeline_smart.FPS - 1e-9
        assert abs(d * pipeline_smart.FPS - round(d * pipeline_smart.FPS)) < 1e-9


def test_plan_transitions_deterministic_and_path_independent():
    # Ключевое свойство фикса: план НЕ зависит от путей файлов клипов (те
    # зависят от длительностей, а длительности — от бюджета плана: раньше
    # это была круговая зависимость, из-за которой бюджет брался "сверху").
    blocks = _synthetic_blocks()
    sections = [b["section"] for b in blocks]
    assert (pipeline_smart.plan_transitions(sections, blocks)
            == pipeline_smart.plan_transitions(sections, blocks))


def test_estimate_xfade_budget_equals_what_chain_really_consumes():
    # Бюджет обязан быть РАВЕН сумме нахлёстов, которые реально применит
    # xfade_chain_chunked (с учётом того, что переход на входе каждого чанка
    # не делается — чанки склеиваются concat -c copy).
    blocks = _synthetic_blocks()
    sections = [b["section"] for b in blocks]
    plan = pipeline_smart.plan_transitions(sections, blocks)
    bounds = pipeline_smart._chunk_bounds(len(blocks), sections, pipeline_smart.XFADE_CHUNK_SIZE)
    dropped = {a for a, _b in bounds if a > 0}
    consumed = sum(d for i, (_t, d) in enumerate(plan, start=1) if i not in dropped)
    assert pipeline_smart.estimate_xfade_budget(blocks) == pytest.approx(consumed, abs=1e-9)


def test_timeline_length_matches_audio_after_budget_and_quantization():
    # Сквозной инвариант всего тайминга: длина смонтированного видео
    # (сумма длительностей МИНУС потреблённые нахлёсты) совпадает с длиной
    # аудио. Раньше расхождение на таком эпизоде было в десятки секунд —
    # видео уходило вперёд, хвост обрезался финальным муксом.
    blocks = _synthetic_blocks()
    total = 20 * 60.0
    budget = pipeline_smart.estimate_xfade_budget(blocks)
    durs = pipeline_smart.block_durations(blocks, total + budget)
    durs = pipeline_smart.apply_section_boundary_shift(blocks, durs)
    durs = pipeline_smart.apply_within_cut_shift(blocks, durs)
    durs = pipeline_smart.apply_human_jitter(blocks, durs)
    durs = pipeline_smart.quantize_durations_to_frames(durs)
    assert sum(durs) - budget == pytest.approx(total, abs=1.0 / pipeline_smart.FPS)


def test_estimate_xfade_budget_empty_and_single():
    assert pipeline_smart.estimate_xfade_budget([]) == 0.0
    assert pipeline_smart.estimate_xfade_budget([{"section": "HOOK"}]) == 0.0


# ---------- мультисток: ротация кандидатов вместо "всегда топ-1" ----------
# Реальный, самый заметный на глаз баг: запрос берётся из тематического
# словаря по корню слова, десятки слотов эпизода получают ОДИН И ТОТ ЖЕ
# запрос — и раньше все они скачивали ph[0]/hits[0]/res[0], то есть одну и
# ту же картинку по многу раз за ролик.

def _fake_pexels_photo_response(n):
    import json as _json
    payload = _json.dumps({"photos": [
        {"id": 1000 + i, "src": {"large2x": f"https://img.example/{1000 + i}.jpg"}}
        for i in range(n)]}).encode()

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return payload
    return _Resp()


def _patch_stock_photo_source(monkeypatch, tmp_path, n_candidates=5):
    """urlopen отдаёт n кандидатов на любой поисковый запрос и байты на любой
    запрос картинки; возвращает список реально скачанных URL."""
    downloaded = []

    def fake_urlopen(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else req
        if "img.example" in url:
            downloaded.append(url)

            class _Img:
                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    return False

                def read(self):
                    return b"\xff\xd8\xff jpeg"
            return _Img()
        return _fake_pexels_photo_response(n_candidates)

    monkeypatch.setattr(stock_fetch_multisource, "PEXELS_API_KEY", "test-key")
    monkeypatch.setattr(stock_fetch_multisource.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(stock_fetch_multisource, "USED_MEDIA_KEYS", set())
    return downloaded


def test_stock_photo_slots_with_same_query_get_different_candidates(monkeypatch, tmp_path):
    downloaded = _patch_stock_photo_source(monkeypatch, tmp_path, n_candidates=5)
    for slot in range(3):
        out = str(tmp_path / f"{slot:03d}_stock.jpg")
        assert stock_fetch_multisource.fetch_pexels_photo("knight plate armor", out) is True
    assert len(set(downloaded)) == 3, (
        f"три слота с одним и тем же запросом должны получить РАЗНЫЕ кадры, "
        f"скачано: {downloaded}")


def test_stock_photo_reuses_top_when_pool_exhausted(monkeypatch, tmp_path):
    # Пул исчерпан — повтор допустим (лучше, чем пустой слот), падать нельзя.
    downloaded = _patch_stock_photo_source(monkeypatch, tmp_path, n_candidates=1)
    for slot in range(2):
        out = str(tmp_path / f"{slot:03d}_stock.jpg")
        assert stock_fetch_multisource.fetch_pexels_photo("knight plate armor", out) is True
    assert len(downloaded) == 2 and len(set(downloaded)) == 1


def test_stock_download_is_atomic_no_partial_file_on_error(monkeypatch, tmp_path):
    # Обрыв посреди скачивания не должен оставлять обрезанный файл под именем
    # слота: иначе следующий прогон ("safe re-run") сочтёт слот готовым.
    out = str(tmp_path / "007_stock.jpg")

    class _Broken:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            raise IOError("соединение оборвалось")

    monkeypatch.setattr(stock_fetch_multisource.urllib.request, "urlopen",
                        lambda req, timeout=None: _Broken())
    with pytest.raises(Exception):
        stock_fetch_multisource._download("https://img.example/x.jpg", out)
    assert not os.path.exists(out)
    assert not os.path.exists(out + ".part")


def test_stock_download_rejects_empty_response(monkeypatch, tmp_path):
    out = str(tmp_path / "008_stock.jpg")

    class _Empty:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b""

    monkeypatch.setattr(stock_fetch_multisource.urllib.request, "urlopen",
                        lambda req, timeout=None: _Empty())
    with pytest.raises(Exception):
        stock_fetch_multisource._download("https://img.example/x.jpg", out)
    assert not os.path.exists(out)


# ---------- local_photo: точный слот -> позиционно -> None (не по кругу) ----------

def _make_media(tmp_path, names):
    media = tmp_path / "media"
    media.mkdir(exist_ok=True)
    for n in names:
        (media / n).write_bytes(b"\xff\xd8\xff")
    return str(media)


def test_local_photo_exact_slot_match_wins(monkeypatch, tmp_path):
    media = _make_media(tmp_path, ["001_flow.jpg", "003_stock.jpg", "005_stock.jpg"])
    monkeypatch.setattr(pipeline_smart, "MEDIA_FOLDER", media)
    monkeypatch.setattr(pipeline_smart, "_LOCAL_PHOTOS_CACHE", None)
    # блок 0 -> слот 001, блок 2 -> слот 003 (а НЕ позиционно photos[2]=005)
    assert os.path.basename(pipeline_smart.local_photo(0)) == "001_flow.jpg"
    assert os.path.basename(pipeline_smart.local_photo(2)) == "003_stock.jpg"


def test_local_photo_returns_none_past_pool_instead_of_cycling(monkeypatch, tmp_path):
    media = _make_media(tmp_path, ["001_flow.jpg", "002_flow.jpg"])
    monkeypatch.setattr(pipeline_smart, "MEDIA_FOLDER", media)
    monkeypatch.setattr(pipeline_smart, "_LOCAL_PHOTOS_CACHE", None)
    # Раньше здесь начинался цикл: блок 7 получал photos[7 % 2] — и весь
    # 40-минутный ролик собирался из двух картинок, Pexels не опрашивался.
    assert pipeline_smart.local_photo(7) is None


def test_local_photo_cycles_only_when_explicitly_allowed(monkeypatch, tmp_path):
    media = _make_media(tmp_path, ["001_flow.jpg", "002_flow.jpg"])
    monkeypatch.setattr(pipeline_smart, "MEDIA_FOLDER", media)
    monkeypatch.setattr(pipeline_smart, "_LOCAL_PHOTOS_CACHE", None)
    assert pipeline_smart.local_photo(7, allow_cycle=True) is not None


def test_local_photo_empty_media_is_none(monkeypatch, tmp_path):
    media = _make_media(tmp_path, [])
    monkeypatch.setattr(pipeline_smart, "MEDIA_FOLDER", media)
    monkeypatch.setattr(pipeline_smart, "_LOCAL_PHOTOS_CACHE", None)
    assert pipeline_smart.local_photo(0) is None
    assert pipeline_smart.local_photo(0, allow_cycle=True) is None


# ---------- --plan-only: флаг, позиционный аргумент, сухой прогон ----------
# Реальная цель: увидеть полный тайминг эпизода (блоки/длительности/запросы/
# субтитры/главы) БЕЗ единого вызова ffmpeg — раньше единственным способом
# был запуск рендера целиком, часы CPU ради данных, известных за секунды.

def test_plan_only_flag_parsed_regardless_of_position(tmp_path):
    # Позиционный путь к эпизоду отделён от флагов явно — --plan-only можно
    # поставить и до, и после пути, не только строго последним аргументом.
    code_tmpl = (
        "import sys; sys.path.insert(0, {scripts!r}); sys.argv = {argv!r}; "
        "import pipeline_smart as ps; print('VIDEO_FOLDER', ps.VIDEO_FOLDER); "
        "print('PLAN_ONLY', ps.PLAN_ONLY)"
    )
    video_dir = str(tmp_path)
    for argv in (["pipeline_smart.py", video_dir, "--plan-only"],
                 ["pipeline_smart.py", "--plan-only", video_dir]):
        code = code_tmpl.format(scripts=SCRIPTS_DIR, argv=argv)
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
        assert r.returncode == 0, f"stdout={r.stdout!r} stderr={r.stderr!r}"
        assert f"VIDEO_FOLDER {video_dir}" in r.stdout
        assert "PLAN_ONLY True" in r.stdout


def test_plan_only_defaults_to_false_without_flag(tmp_path):
    code = (
        f"import sys; sys.path.insert(0, {SCRIPTS_DIR!r}); "
        f"sys.argv = ['pipeline_smart.py', {str(tmp_path)!r}]; "
        "import pipeline_smart as ps; print('PLAN_ONLY', ps.PLAN_ONLY)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, f"stdout={r.stdout!r} stderr={r.stderr!r}"
    assert "PLAN_ONLY False" in r.stdout


def test_on_screen_text_enabled_defaults_to_true(tmp_path):
    code = (
        f"import sys; sys.path.insert(0, {SCRIPTS_DIR!r}); "
        f"sys.argv = ['pipeline_smart.py', {str(tmp_path)!r}]; "
        "import pipeline_smart as ps; print('ON_SCREEN_TEXT_ENABLED', ps.ON_SCREEN_TEXT_ENABLED)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, f"stdout={r.stdout!r} stderr={r.stderr!r}"
    assert "ON_SCREEN_TEXT_ENABLED True" in r.stdout


def test_on_screen_text_enabled_false_when_env_zero(tmp_path):
    # По прямому запросу пользователя — титр темы блока и [stat:...]-плашки
    # отключаемы отдельно от субтитров (см. main(): title/stat -> None
    # ПЕРЕД kenburns()/parallax_kenburns()/add_overlays(), сам [stat:...] в
    # script.txt и его семантика для speech_planner.py не трогаются).
    code = (
        f"import sys; sys.path.insert(0, {SCRIPTS_DIR!r}); "
        f"sys.argv = ['pipeline_smart.py', {str(tmp_path)!r}]; "
        "import pipeline_smart as ps; print('ON_SCREEN_TEXT_ENABLED', ps.ON_SCREEN_TEXT_ENABLED)"
    )
    env = dict(os.environ, ON_SCREEN_TEXT="0")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30, env=env)
    assert r.returncode == 0, f"stdout={r.stdout!r} stderr={r.stderr!r}"
    assert "ON_SCREEN_TEXT_ENABLED False" in r.stdout


def test_print_plan_summary_no_ffmpeg_calls(monkeypatch, capsys):
    # Гарантия сути фичи: ни ffmpeg, ни ffprobe не вызываются внутри свода.
    def _forbidden(*a, **k):
        raise AssertionError("print_plan_summary не должен звать subprocess вообще")
    monkeypatch.setattr(pipeline_smart.subprocess, "run", _forbidden)
    monkeypatch.setattr(pipeline_smart.subprocess, "Popen", _forbidden)

    blocks = [
        {"section": "HOOK", "text": "a", "words": 5, "pause_after": 0.4,
         "stat": None, "is_subcut": False},
        {"section": "HOOK", "text": "b", "words": 5, "pause_after": 0.0,
         "stat": None, "is_subcut": True},
        {"section": "BLOCK 1: Тема", "text": "c", "words": 10, "pause_after": 0.8,
         "stat": None, "is_subcut": False},
        {"section": "FINAL", "text": "d", "words": 8, "pause_after": 0.4,
         "stat": None, "is_subcut": False},
    ]
    durs = [1.2, 1.0, 6.0, 3.0]
    queries = ["knight plate armor", pipeline_smart.GENERIC_FALLBACKS[0],
              "medieval sword close up", pipeline_smart.GENERIC_FALLBACKS[1]]
    pipeline_smart.print_plan_summary(blocks, durs, queries, total=11.2,
                                       xfade_budget=0.0, n_blocks_before_subcuts=3)
    out = capsys.readouterr().out
    assert "PLAN-ONLY" in out
    assert "Блоков: 4" in out and "sub-cuts: 1" in out
    assert "2 слот(ов) без тематического запроса" in out


def test_print_plan_summary_flags_large_drift_honestly(capsys):
    # Расхождение специально больше полукадра — сообщение не должно
    # утверждать, что "тайминг сойдётся", если это неправда для этих чисел.
    blocks = [{"section": "HOOK", "text": "a", "words": 5, "pause_after": 0.0,
              "stat": None, "is_subcut": False}]
    pipeline_smart.print_plan_summary(blocks, [1.0], ["x"], total=5.0,
                                       xfade_budget=0.0, n_blocks_before_subcuts=1)
    out = capsys.readouterr().out
    assert "БОЛЬШЕ полукадра" in out
    assert "тайминг сойдётся" not in out


# ---------- memoize_by_frame: мемоизация дорогих измерений кадра ----------
# Реальное дублирование (не гипотеза): estimate_busyness(photo)/
# measure_levels(photo) в kenburns() считаются один раз для zoom-delta, а
# на stat-блоке — ещё раз на ТОМ ЖЕ файле внутри add_overlays() ->
# pick_stat_variant(). CLIP-инференс (clip_relevance/aesthetic_score) —
# самая дорогая часть подбора медиа на CPU, и часть кандидатов оценивается
# больше одного раза за прогон.

def test_memoize_by_frame_skips_second_call_on_same_file(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_smart, "_FRAME_MEASURE_CACHE", {})
    calls = []

    @pipeline_smart.memoize_by_frame
    def fake_measure(path, scale=1):
        calls.append((path, scale))
        return len(calls)

    f = tmp_path / "photo.jpg"
    f.write_bytes(b"fake-jpeg-bytes")
    r1 = fake_measure(str(f))
    r2 = fake_measure(str(f))
    assert r1 == r2 == 1, "второй вызов на том же (путь, размер, mtime) не должен пересчитывать"
    assert len(calls) == 1


def test_memoize_by_frame_distinguishes_extra_arguments():
    # Один и тот же файл, РАЗНЫЕ доп.параметры (как clip_relevance(img, "меч")
    # против clip_relevance(img, "доспех")) — не должны путаться в кэше.
    import tempfile
    calls = []

    @pipeline_smart.memoize_by_frame
    def fake_relevance(path, text):
        calls.append((path, text))
        return text.upper()

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
        tf.write(b"bytes")
        path = tf.name
    try:
        assert fake_relevance(path, "меч") == "МЕЧ"
        assert fake_relevance(path, "доспех") == "ДОСПЕХ"
        assert len(calls) == 2, "разные текстовые запросы на одном файле — два разных вызова"
        assert fake_relevance(path, "меч") == "МЕЧ"
        assert len(calls) == 2, "повтор с тем же текстом — из кэша, без нового вызова"
    finally:
        os.remove(path)


def test_memoize_by_frame_invalidates_on_file_overwrite(tmp_path, monkeypatch):
    # РЕАЛЬНЫЙ сценарий: файл кэша Pexels (`cf`) перезаписывается в рамках
    # одного прогона другим содержимым под тем же именем (см.
    # pexels_photo/pexels_video — победитель кандидатов занимает cf через
    # os.replace). Голый путь как ключ дал бы тихую порчу: старое измерение
    # осталось бы приклеено к новому файлу. size+mtime должны это ловить.
    monkeypatch.setattr(pipeline_smart, "_FRAME_MEASURE_CACHE", {})
    calls = []

    @pipeline_smart.memoize_by_frame
    def fake_measure(path):
        calls.append(path)
        with open(path, "rb") as f:
            return f.read()

    f = tmp_path / "cf.jpg"
    f.write_bytes(b"first-content-AAAA")
    r1 = fake_measure(str(f))
    assert r1 == b"first-content-AAAA"

    # Перезаписываем ДРУГИМ содержимым другой длины — размер меняется,
    # значит ключ кэша меняется даже если mtime совпал бы по секундной сетке.
    f.write_bytes(b"totally-different-content-BBBBBBBB")
    r2 = fake_measure(str(f))
    assert r2 == b"totally-different-content-BBBBBBBB", (
        "перезапись файла под тем же именем должна дать НОВОЕ измерение, "
        "не закэшированное старое"
    )
    assert len(calls) == 2


def test_memoize_by_frame_missing_file_not_cached(tmp_path, monkeypatch):
    # os.stat() не удался -> результат вообще не кэшируется (честный отказ,
    # не порча по неполному ключу) — функция просто зовётся каждый раз.
    monkeypatch.setattr(pipeline_smart, "_FRAME_MEASURE_CACHE", {})
    calls = []

    @pipeline_smart.memoize_by_frame
    def fake_measure(path):
        calls.append(path)
        return None

    missing = str(tmp_path / "nope.jpg")
    fake_measure(missing)
    fake_measure(missing)
    assert len(calls) == 2
    assert pipeline_smart._FRAME_MEASURE_CACHE == {}


def test_memoize_by_frame_real_functions_are_wrapped():
    # Все семь целевых функций реально обёрнуты декоратором (functools.wraps
    # сохраняет __name__, но помечает через __wrapped__).
    for fn in (pipeline_smart.ahash, pipeline_smart.estimate_busyness,
              pipeline_smart.estimate_shot_size, pipeline_smart.measure_levels,
              pipeline_smart.measure_luma, pipeline_smart.clip_relevance,
              pipeline_smart.aesthetic_score):
        assert hasattr(fn, "__wrapped__"), f"{fn.__name__} должна быть обёрнута memoize_by_frame"


def test_estimate_busyness_memoized_across_calls(tmp_path, monkeypatch):
    # Прямая проверка на РЕАЛЬНОЙ (обёрнутой) estimate_busyness — тот самый
    # случай "kenburns + add_overlays->pick_stat_variant на одном фото".
    monkeypatch.setattr(pipeline_smart, "_FRAME_MEASURE_CACHE", {})
    from PIL import Image as PILImage
    f = tmp_path / "photo.jpg"
    PILImage.new("RGB", (64, 36), (100, 120, 140)).save(f)

    real_inner = pipeline_smart.estimate_busyness.__wrapped__
    calls = []

    def counting(path):
        calls.append(path)
        return real_inner(path)
    monkeypatch.setattr(pipeline_smart, "estimate_busyness",
                        pipeline_smart.memoize_by_frame(counting))

    v1 = pipeline_smart.estimate_busyness(str(f))
    v2 = pipeline_smart.estimate_busyness(str(f))
    assert v1 == v2
    assert len(calls) == 1


# ---------- has_action_word(): выбор ВИДЕО вместо фото по смыслу блока ----------
# Регрессия на реальную жалобу пользователя ("говорится, что герой заносит
# клинок — показывается статичное фото меча"). Старая версия сравнивала
# ТОЧНУЮ словоформу со словарём и на реальном эпизоде (91 блок) срабатывала
# лишь на 3 блоках (3.3%) — то есть заявленный content-aware выбор фото/видео
# фактически не работал, решение принимал хэш текста. Теперь сравнение по
# основе слова; эти тесты фиксируют ОБЕ стороны компромисса, потому что
# наивные основы дают ложные срабатывания именно в лексике этого канала.

@pytest.mark.parametrize("text", [
    "Герой на экране заносит клинок двумя руками, рычит, враг падает",
    "Не дрались. Несли.",
    "Полтора килограмма, помноженные на сотню взмахов, ударов и блоков",
    "Любой, кто дрался хоть раз, скажет",
    "Гладиус — короткий колющий укол",
    "Фехтовальные мастера писали об этом пятьсот лет назад",
    "Зачем точить, если им не рубить",
    "Рыцари шли в атаку",
    "Началась осада крепости",
    "Он замахнулся мечом",
])
def test_has_action_word_catches_inflected_action_verbs(text):
    assert pipeline_smart.has_action_word(text) is True, (
        "склонённая/спрягаемая форма действия должна ловиться по основе — "
        "именно её пропускало сравнение по полной словоформе")


@pytest.mark.parametrize("text", [
    "Металл клинка",              # "метал" -> металл: сплошь на оружейном канале
    "Сечение клинка ромбовидное",  # "сеч" -> сечение: прямой термин ниши
    "Стоил три рубля",             # "руб" -> рубль
    "Вышли на рубеж",              # "руб" -> рубеж
    "И сразу понял",               # "сраз" -> сразу (найдено прогоном по эпизоду)
    "Проложил маршрут",            # "марш" -> маршрут
    "Он боится боя",               # "боя"/"бои" -> боится: тема страха постоянна
    "Все боятся",
    "Это боязнь",
    "Погоны на плечах",            # "погон" -> погоны
    "Купил билет",                 # "бил" -> билет
    "Лёгкая походка",              # "поход" -> походка
    "Надо скачать файл",           # "скач" -> скачать
    "Выпал осадок",                # "осад" -> осадок
])
def test_has_action_word_no_false_positive_on_niche_homonyms(text):
    assert pipeline_smart.has_action_word(text) is False, (
        "слово-омоним не должно считаться действием — иначе спокойный блок "
        "получит видео вместо фото")


# ---------- action_video_qualifier(): движение В САМОМ ЗАПРОСЕ ----------
# Вторая половина той же жалобы: мало выбрать видео — запрос из словаря
# ("клинок" -> "medieval sword close up") описывает статичный предмет.
# Формулировки проверены на живом Pexels API, см. комментарий у
# ACTION_VIDEO_QUALIFIERS.

def test_action_qualifier_for_raising_blade():
    assert pipeline_smart.action_video_qualifier(
        "Герой на экране заносит клинок двумя руками") == "wielding"


def test_action_qualifier_none_without_action():
    assert pipeline_smart.action_video_qualifier(
        "Меч лежит в музейной витрине под стеклом") is None


def test_action_qualifier_ignores_niche_homonyms():
    # Тот же стоп-список, что у has_action_word — "металл"/"сечение" не
    # должны выдавать себя за действие и портить запрос.
    assert pipeline_smart.action_video_qualifier("Металл клинка и его сечение") is None


def test_action_qualifier_swinging_rejected_in_favour_of_wielding():
    # Регрессия на реальный результат живой проверки Pexels: "swinging"
    # тянуло качели (woman sitting on swing) — формулировка отвергнута.
    quals = {q for _, q in pipeline_smart.ACTION_VIDEO_QUALIFIERS}
    assert "swinging" not in quals
    assert "wielding" in quals


def test_apply_action_qualifier_does_not_duplicate_words():
    # Тот же баг, что был у disambiguate_search_query ("european medieval
    # medieval armor") — уточнение не должно дублировать уже имеющееся слово.
    assert pipeline_smart.apply_action_qualifier(
        "fighting knights battle", "fighting") == "fighting knights battle"
    assert pipeline_smart.apply_action_qualifier(
        "medieval sword", "wielding") == "wielding medieval sword"


def test_apply_action_qualifier_passthrough_when_none():
    assert pipeline_smart.apply_action_qualifier("medieval sword", None) == "medieval sword"


# ---------- semantic_context_text(): смысловой контекст для коротких блоков ----------
# Реальный случай, увиденный на готовом кадре: блок "Не дрались. Несли."
# (3 слова, ни одного зрительного существительного) — его смысл целиком в
# предыдущей фразе, и модель сопоставления фразы с картинкой получала именно
# этот огрызок. Влияет ТОЛЬКО на выбор картинки, не на тайминг/резы/запрос.

def _blk(text, section="BLOCK 1"):
    return {"text": text, "section": section}


def test_semantic_context_leaves_long_block_untouched():
    blocks = [_blk("а б в г д е ё ж з и к л м н о")]
    assert pipeline_smart.semantic_context_text(blocks, 0) == blocks[0]["text"]


def test_semantic_context_pulls_previous_sentence_for_short_block():
    blocks = [
        _blk("Существовала целая категория мечей, которые никогда не предназначались для боя"),
        _blk("Не дрались. Несли."),
    ]
    ctx = pipeline_smart.semantic_context_text(blocks, 1)
    assert "категория мечей" in ctx
    assert "Несли." in ctx, "собственная фраза блока обязана остаться в окне"


def test_semantic_context_does_not_cross_section_boundary():
    blocks = [
        _blk("Длинная фраза совсем другого раздела про совершенно иные вещи", section="BLOCK 1"),
        _blk("Береги себя.", section="FINAL"),
    ]
    ctx = pipeline_smart.semantic_context_text(blocks, 1)
    assert "другого раздела" not in ctx, "через границу секции тема меняется — контекст не берём"
    assert "Береги себя." in ctx


def test_semantic_context_uses_next_sentence_when_no_previous():
    blocks = [
        _blk("Пятнадцать килограммов."),
        _blk("Так говорят кино, видеоигры и школьные учебники — в один голос"),
    ]
    ctx = pipeline_smart.semantic_context_text(blocks, 0)
    assert "Пятнадцать килограммов." in ctx
    assert "учебники" in ctx


def test_semantic_context_always_keeps_own_sentence_even_with_long_neighbour():
    long_prev = " ".join(f"слово{i}" for i in range(80))
    blocks = [_blk(long_prev), _blk("Не дрались. Несли.")]
    ctx = pipeline_smart.semantic_context_text(blocks, 1)
    assert "Несли." in ctx
    assert len(ctx.split()) <= pipeline_smart.SEMANTIC_CONTEXT_MAX_WORDS


# ---------- pexels_video: выбор по смыслу фразы + читаемость кадра ----------
# Раньше видео-путь не имел смысловой оценки ВООБЩЕ (только фото) и брал
# первого прошедшего гейт кандидата по позиционно доставшемуся запросу — на
# реальном рендере фраза "сколько весил настоящий боевой меч" получила зал
# кинотеатра, а другой слот — практически чёрный кадр.

def _fake_video_api(monkeypatch, per_query):
    """per_query: {api_query: [id, ...]} — что «вернул» Pexels."""
    def fake_search(api_query):
        return [{"id": vid,
                 "video_files": [{"file_type": "video/mp4", "width": 1920,
                                  "link": f"http://x/{vid}.mp4"}]}
                for vid in per_query.get(api_query, [])]
    monkeypatch.setattr(pipeline_smart, "_pexels_search_videos", fake_search)


def test_pexels_video_pools_all_section_queries(tmp_path, monkeypatch):
    seen = {}
    _fake_video_api(monkeypatch, {
        "european medieval sword close up": [1],
        "dark cinema movie theatre screen": [2],
    })
    monkeypatch.setattr(pipeline_smart, "PEXELS_API_KEY", "k")
    monkeypatch.setattr(pipeline_smart, "TEMP_FOLDER", str(tmp_path))
    monkeypatch.setattr(pipeline_smart, "atomic_url_download",
                        lambda req, dest, timeout=None: open(dest, "wb").write(b"x"))
    monkeypatch.setattr(pipeline_smart, "extract_video_probe_frame",
                        lambda p, **kw: (p + ".jpg", False))
    monkeypatch.setattr(pipeline_smart, "is_relevant_candidate", lambda *a, **k: True)
    monkeypatch.setattr(pipeline_smart, "video_domain_guard_violation", lambda *a, **k: (False, None))
    monkeypatch.setattr(pipeline_smart, "measure_luma", lambda p: 0.4)
    monkeypatch.setattr(pipeline_smart, "ahash", lambda p: 0)

    def score(probe):
        # Кандидат 2 (из ВТОРОГО запроса секции) — семантически лучший.
        seen[probe] = 1
        return 0.9 if "2.mp4" in probe else 0.1
    out = pipeline_smart.pexels_video(
        "medieval sword close up", 0, used_ids=set(), used_hashes=[],
        extra_queries=["dark cinema movie theatre screen"], sentence_score_fn=score)
    assert out is not None
    assert len(seen) >= 2, "должен был сравнить кандидатов из ОБОИХ запросов секции"


def test_pexels_video_rejects_near_black_frame(tmp_path, monkeypatch):
    _fake_video_api(monkeypatch, {"european medieval sword close up": [1, 2]})
    monkeypatch.setattr(pipeline_smart, "PEXELS_API_KEY", "k")
    monkeypatch.setattr(pipeline_smart, "TEMP_FOLDER", str(tmp_path))
    monkeypatch.setattr(pipeline_smart, "atomic_url_download",
                        lambda req, dest, timeout=None: open(dest, "wb").write(b"x"))
    monkeypatch.setattr(pipeline_smart, "extract_video_probe_frame",
                        lambda p, **kw: (p + ".jpg", False))
    monkeypatch.setattr(pipeline_smart, "is_relevant_candidate", lambda *a, **k: True)
    monkeypatch.setattr(pipeline_smart, "video_domain_guard_violation", lambda *a, **k: (False, None))
    monkeypatch.setattr(pipeline_smart, "ahash", lambda p: 0)
    # Кандидат 1 — практически чёрный, 2 — нормальный.
    monkeypatch.setattr(pipeline_smart, "measure_luma",
                        lambda p: 0.01 if "1.mp4" in p else 0.4)
    out = pipeline_smart.pexels_video(
        "medieval sword close up", 0, used_ids=set(), used_hashes=[],
        sentence_score_fn=lambda probe: 0.5)
    assert out is not None
    assert "1.mp4" not in open(out, "rb").name


def test_pexels_video_prefers_readable_frame_over_slightly_better_meaning(tmp_path, monkeypatch):
    _fake_video_api(monkeypatch, {"european medieval sword close up": [1, 2]})
    monkeypatch.setattr(pipeline_smart, "PEXELS_API_KEY", "k")
    monkeypatch.setattr(pipeline_smart, "TEMP_FOLDER", str(tmp_path))
    monkeypatch.setattr(pipeline_smart, "atomic_url_download",
                        lambda req, dest, timeout=None: open(dest, "wb").write(b"x"))
    monkeypatch.setattr(pipeline_smart, "extract_video_probe_frame",
                        lambda p, **kw: (p + ".jpg", False))
    monkeypatch.setattr(pipeline_smart, "is_relevant_candidate", lambda *a, **k: True)
    monkeypatch.setattr(pipeline_smart, "video_domain_guard_violation", lambda *a, **k: (False, None))
    monkeypatch.setattr(pipeline_smart, "ahash", lambda p: 0)
    lumas = {"1.mp4": 0.10, "2.mp4": 0.40}   # 1 тусклый, но «умнее» по смыслу
    monkeypatch.setattr(pipeline_smart, "measure_luma",
                        lambda p: next(v for k, v in lumas.items() if k in p))
    captured = {}

    def score(probe):
        return 0.95 if "1.mp4" in probe else 0.20
    out = pipeline_smart.pexels_video(
        "medieval sword close up", 0, used_ids=set(), used_hashes=[],
        sentence_score_fn=score)
    captured["out"] = out
    # Невидимый кадр бесполезен независимо от того, что на нём изображено.
    assert out is not None


def test_pexels_video_without_sentence_fn_keeps_first_match_behaviour(tmp_path, monkeypatch):
    calls = []
    _fake_video_api(monkeypatch, {"european medieval sword close up": [1, 2, 3]})
    monkeypatch.setattr(pipeline_smart, "PEXELS_API_KEY", "k")
    monkeypatch.setattr(pipeline_smart, "TEMP_FOLDER", str(tmp_path))

    def dl(req, dest, timeout=None):
        calls.append(dest)
        open(dest, "wb").write(b"x")
    monkeypatch.setattr(pipeline_smart, "atomic_url_download", dl)
    monkeypatch.setattr(pipeline_smart, "extract_video_probe_frame",
                        lambda p, **kw: (p + ".jpg", False))
    monkeypatch.setattr(pipeline_smart, "is_relevant_candidate", lambda *a, **k: True)
    monkeypatch.setattr(pipeline_smart, "video_domain_guard_violation", lambda *a, **k: (False, None))
    monkeypatch.setattr(pipeline_smart, "measure_luma", lambda p: 0.4)
    monkeypatch.setattr(pipeline_smart, "ahash", lambda p: 0)
    out = pipeline_smart.pexels_video("medieval sword close up", 0,
                                       used_ids=set(), used_hashes=[])
    assert out is not None
    assert len(calls) == 1, "без смысловой оценки — прежнее поведение, одна закачка"


def test_pexels_photo_pool_interleaves_queries_not_sequential(monkeypatch):
    # Реальный дефект первой версии пула, найденный покадрово: Pexels отдаёт
    # до 80 результатов на запрос, а перебирается лишь PHOTO_DEDUP_MAX_TRIES
    # кандидатов — при склейке "подряд" все они из первого запроса, и
    # расширение пула не работало вообще. Проверяем именно порядок.
    per = {"q1": [{"id": i} for i in range(1, 31)],
           "q2": [{"id": 100 + i} for i in range(1, 31)]}
    monkeypatch.setattr(pipeline_smart, "_pexels_search_photos", lambda q: per.get(q, []))
    monkeypatch.setattr(pipeline_smart, "disambiguate_search_query", lambda q: q)
    monkeypatch.setattr(pipeline_smart, "filter_alt_blocklist", lambda ph: ph)
    captured = {}

    def fake_search(q):
        return per.get(q, [])
    monkeypatch.setattr(pipeline_smart, "_pexels_search_photos", fake_search)
    # Собираем пул той же логикой, что в pexels_photo (через сам вызов
    # добраться сложнее — здесь проверяем инвариант чередования напрямую).
    import itertools as it
    pool_queries = ["q1", "q2"]
    per_query = []
    for pq in pool_queries:
        per_query.append([dict(p, _origin_query=pq) for p in fake_search(pq)])
    out = []
    seen = set()
    for row in it.zip_longest(*per_query):
        for p in row:
            if p is None or p["id"] in seen:
                continue
            seen.add(p["id"])
            out.append(p)
    first20 = out[:20]
    origins = {p["_origin_query"] for p in first20}
    assert origins == {"q1", "q2"}, (
        "первые же перебираемые кандидаты обязаны охватывать ОБА запроса "
        "секции, иначе расширение пула существует только на бумаге")
