"""arc_stage обязан следовать за ФРАЗОЙ, а не за индексом главного цикла.

Фон — N4 из docs/AUDIT_2026-09_DEEP.md, измеренный на реальном эпизоде
videos/01_ves-mecha:

    parse_blocks -> 91 блок
    split_long_blocks -> 198
    merge_short_phrase_locked_blocks -> 165  <- по ним идёт главный цикл main()

а speech_plan.json знает только исходные 91. arc_stage брался по индексу ЦИКЛА,
поэтому:
  * 77 слотов получали стадию ЧУЖОГО блока (слот 9 хука — стадию из BLOCK 1),
  * 74 слота после 90-го не получали ничего,
  * верными оставались 14 из 165 (8%).

Обе надстройки, которые на неё опираются (Look Management — сила коррекции,
Visual Director — бонус за крупность плана), в проде стоят в режиме assist,
то есть реально влияли на кадр, работая на неверных данных.

Фикс: orig_index проставляется сразу после parse_blocks() и переносится сам —
dict(b) в split_long_blocks() и dict(pb) в _merge_two_blocks() копируют ключ.
"""
import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]

import pipeline_smart as ps  # noqa: E402


def _blocks(n=6, sentences=3):
    """Блоки, которые split_long_blocks реально режет.

    Условие резки (см. её тело): либо est >= SUBCUT_MIN_SOURCE_DUR, либо в
    блоке есть внутренние границы ЗАКОНЧЕННЫХ предложений. Берём второе — оно
    не зависит от подобранных весов и потому устойчивее как фикстура.
    """
    sent = "Меч оказался заметно легче ожидаемого веса совсем немного."
    text = " ".join([sent] * sentences)
    words = len(text.split())
    return [{"section": f"BLOCK {i}: тест", "text": text,
             "words": words, "pause_after": 0.5, "stat": None,
             "stat_word_pos": None, "is_climax": False, "orig_index": i}
            for i in range(n)]


def test_orig_index_survives_split_into_subcuts():
    """Под-кадры одной фразы обязаны нести индекс ЭТОЙ фразы."""
    blocks = _blocks()
    weights = [12.0] * len(blocks)
    out, _ = ps.split_long_blocks(blocks, weights)
    assert len(out) > len(blocks), "тест бессмысленен: split ничего не разрезал"
    for b in out:
        assert "orig_index" in b, "orig_index потерян при split_long_blocks()"
    # каждый исходный индекс представлен, чужих не появилось
    assert {b["orig_index"] for b in out} == set(range(len(blocks)))


def test_subcuts_of_one_phrase_share_its_index():
    blocks = _blocks(n=1, sentences=4)
    out, _ = ps.split_long_blocks(blocks, [12.0])
    assert len(out) > 1
    assert all(b["orig_index"] == 0 for b in out), (
        "под-кадры одной фразы разъехались по разным исходным индексам"
    )


def test_merge_keeps_index_of_first_merged_block():
    """Слитый блок начинается там же, где первый из слитых — его индекс и берём."""
    a = {"text": "короткая", "words": 1, "pause_after": 0.1, "stat": None,
         "stat_word_pos": None, "is_climax": False, "orig_index": 7,
         "section": "BLOCK 1"}
    b = {"text": "вторая", "words": 1, "pause_after": 0.2, "stat": None,
         "stat_word_pos": None, "is_climax": False, "orig_index": 8,
         "section": "BLOCK 1"}
    merged = ps._merge_two_blocks(a, b, a["words"])
    assert merged["orig_index"] == 7


def test_cycle_index_would_have_been_wrong_after_split():
    """Фиксируем СУТЬ бага: индекс цикла и индекс фразы расходятся.

    Если этот тест когда-нибудь начнёт падать, значит split/merge перестали
    менять число блоков — и тогда весь класс проблемы исчез сам. Пока они его
    меняют, обращаться к speech_plan по индексу цикла нельзя.
    """
    blocks = _blocks()
    out, _ = ps.split_long_blocks(blocks, [12.0] * len(blocks))
    divergent = [k for k, b in enumerate(out) if k != b["orig_index"]]
    assert divergent, (
        "индексы цикла и фразы совпали — тест перестал проверять то, ради чего "
        "написан; проверить, не изменилась ли механика split_long_blocks()"
    )
