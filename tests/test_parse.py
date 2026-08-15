"""Юнит-тесты чистой логики (без FFmpeg): парсинг сценария, тайминг блоков,
выбор без повторов, подбор тематического запроса, счётчик слов.
Запуск: .venv/bin/python -m pytest tests/ -v
"""
import os
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


def test_load_themes_missing_file_returns_empty(tmp_path):
    assert stock_fetch_multisource.load_themes(str(tmp_path)) == {}


def test_load_themes_reads_json(tmp_path):
    media_plan = tmp_path / "media_plan"
    media_plan.mkdir()
    (media_plan / "themes.json").write_text('{"меч": "sword"}', encoding="utf-8")
    assert stock_fetch_multisource.load_themes(str(tmp_path)) == {"меч": "sword"}
