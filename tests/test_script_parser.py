"""script_parser.py — регрессия на реентерабельность (ЧАСТЬ 13 CLAUDE.md /
аудит генератора, пункт 10): parse_blocks() вынесен из pipeline_smart.py в
отдельный модуль БЕЗ побочных эффектов импорта. Полное поведение самой
parse_blocks() уже покрыто test_parse.py через pipeline_smart.parse_blocks
(тот же объект — см. test_pipeline_smart_reexports_same_object_as_script_parser
ниже), здесь — только то, что специфично для факта разделения на модули."""
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")


def test_script_parser_importable_without_any_sys_argv():
    # Регрессия: в отличие от pipeline_smart.py (читает VIDEO_FOLDER из
    # sys.argv[1] на импорте и падает/ведёт себя непредсказуемо без него —
    # см. find_audio()), script_parser.py не имеет побочных эффектов импорта
    # вообще. Проверяем в ЧИСТОМ подпроцессе (не переиспользуя закэшированный
    # sys.modules этой сессии pytest), с sys.argv из ОДНОГО элемента —
    # ровно то, чего раньше не хватало без ручной подмены sys.argv.
    code = ("import sys; sys.path.insert(0, %r); "
            "import script_parser; print('OK', script_parser.PAUSE_DURATIONS['[pause]'])"
            % SCRIPTS_DIR)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, f"stdout={r.stdout!r} stderr={r.stderr!r}"
    assert "OK 0.8" in r.stdout


def test_script_parser_does_not_touch_filesystem_beyond_given_path(tmp_path):
    # НЕ importlib.reload() здесь: script_parser не хранит модульного
    # состояния между вызовами, а reload() создал бы НОВЫЙ объект функции —
    # pipeline_smart (если уже импортирован раньше в этом процессе другим
    # тестовым файлом) продолжал бы держать ссылку на СТАРЫЙ, и
    # test_pipeline_smart_reexports_same_object_as_script_parser ниже
    # ложно упал бы при прогоне всего набора (реально произошло при
    # разработке этого теста).
    sys.path.insert(0, SCRIPTS_DIR)
    import script_parser
    script_path = tmp_path / "script.txt"
    script_path.write_text("=== HOOK === Раз два три.\n", encoding="utf-8")
    blocks = script_parser.parse_blocks(str(script_path))
    assert len(blocks) == 1
    assert blocks[0]["section"] == "HOOK"


def test_pipeline_smart_reexports_same_object_as_script_parser():
    # pipeline_smart.py требует sys.argv[1] на импорте (см. его докстринг) —
    # ставим его перед импортом, как и остальные тесты этого проекта.
    import tempfile
    old_argv = sys.argv
    sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]
    try:
        sys.path.insert(0, SCRIPTS_DIR)
        import pipeline_smart
        import script_parser
        assert pipeline_smart.parse_blocks is script_parser.parse_blocks
        assert pipeline_smart.PAUSE_DURATIONS is script_parser.PAUSE_DURATIONS
    finally:
        sys.argv = old_argv


# ---------- parse_pexels_queries / _normalize_section_key — === PEXELS
# QUERIES === написан вручную по протоколу (CLAUDE.md ЧАСТЬ 13, Шаг 3), но
# до этого коммита ни разу не читался пайплайном (реальный найденный
# пробел) ----------

def _sp():
    sys.path.insert(0, SCRIPTS_DIR)
    import script_parser
    return script_parser


def test_normalize_section_key_matches_full_header_and_underscore_form():
    sp = _sp()
    assert sp._normalize_section_key("BLOCK 1: Постановка проблемы") == "BLOCK1"
    assert sp._normalize_section_key("BLOCK_1") == "BLOCK1"
    assert sp._normalize_section_key("HOOK") == "HOOK"
    assert sp._normalize_section_key("FINAL") == "FINAL"


def test_normalize_section_key_none_for_unknown_prefix():
    sp = _sp()
    assert sp._normalize_section_key("IMAGE PROMPTS") is None
    assert sp._normalize_section_key("") is None


def test_parse_pexels_queries_real_format(tmp_path):
    sp = _sp()
    script_path = tmp_path / "script.txt"
    script_path.write_text(
        "=== HOOK === Текст хука.\n\n"
        "=== BLOCK 1: Название === Текст блока.\n\n"
        "=== PEXELS QUERIES ===\n"
        "HOOK: medieval knight sword battle, milk bottle hand\n"
        "BLOCK_1: medieval sword museum display, knight armor exhibit\n\n"
        "=== IMAGE PROMPTS ===\nНе используется.\n",
        encoding="utf-8")
    result = sp.parse_pexels_queries(str(script_path))
    assert result == {
        "HOOK": ["medieval knight sword battle", "milk bottle hand"],
        "BLOCK1": ["medieval sword museum display", "knight armor exhibit"],
    }


def test_parse_pexels_queries_missing_section_returns_empty(tmp_path):
    sp = _sp()
    script_path = tmp_path / "script.txt"
    script_path.write_text("=== HOOK === Текст хука.\n", encoding="utf-8")
    assert sp.parse_pexels_queries(str(script_path)) == {}


def test_parse_pexels_queries_missing_file_returns_empty():
    sp = _sp()
    assert sp.parse_pexels_queries("/no/such/script.txt") == {}
