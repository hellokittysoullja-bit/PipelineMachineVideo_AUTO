"""Субтитры: длинный блок больше не показывается одним cue на пол-экрана.

РЕАЛЬНЫЙ, ИЗМЕРЕННЫЙ дефект готового файла, ради которого написан модуль.
Прогон write_subtitles() по videos/02_ne-mechom/script.txt ДО правки: 94
cue из 259 (36%) получали строку длиннее стандарта 42 символа, самая
длинная — 139 символов. _wrap_caption_text() честно признавала это в
докстринге ("остаток дописывается в последнюю строку"), но единицей
нарезки был целый блок после split_long_blocks() — 35+ слов и 14 секунд
окна, что физически не один субтитр.

ПОСЛЕ: 0 строк длиннее 42 символов, 0 нахлёстов cue, скорость чтения не
изменилась (медиана 12.1 симв/с, ровно 2 cue быстрее 20 симв/с — столько
же, сколько было).

Тесты ниже держат ИНВАРИАНТЫ, а не эти конкретные числа: ни один cue не
нарушает стандарт, ни один символ не теряется, cue плотно покрывают окно
блока без щелей и нахлёстов, и текст, который и так влезал, по-прежнему
остаётся ОДНИМ cue (ноль регресса для коротких блоков).
"""
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]

import pipeline_smart  # noqa: E402


def _lines_of(cue_text):
    return pipeline_smart._wrap_caption_text(cue_text).split("\n")


SHORT = "Доспех весил двадцать пять килограммов."
LONG = ("И тут надо сказать вещь, которую обычно пропускают: основная масса "
        "французских латников пришла к месту боя пешком, по вспаханному полю, "
        "в полном доспехе, и это заняло у них почти всё утро.")


def test_short_text_stays_one_cue():
    """Блок, который и так влезал, не дробится — ноль регресса."""
    assert pipeline_smart._split_caption_into_cues(SHORT) == [SHORT]


def test_long_text_is_split():
    cues = pipeline_smart._split_caption_into_cues(LONG)
    assert len(cues) > 1


@pytest.mark.parametrize("text", [SHORT, LONG, LONG + " " + LONG])
def test_every_cue_respects_the_subtitle_standard(text):
    """ГЛАВНЫЙ инвариант: ни одной строки длиннее SRT_MAX_LINE_CHARS и ни
    одного cue выше SRT_MAX_LINES строк. Именно он нарушался в 36% cue."""
    for cue in pipeline_smart._split_caption_into_cues(text):
        lines = _lines_of(cue)
        assert len(lines) <= pipeline_smart.SRT_MAX_LINES
        for line in lines:
            assert len(line) <= pipeline_smart.SRT_MAX_LINE_CHARS, line


@pytest.mark.parametrize("text", [SHORT, LONG, LONG + " " + LONG])
def test_no_word_is_lost_or_reordered(text):
    """Разбивка только переставляет границы, а не редактирует текст."""
    joined = " ".join(pipeline_smart._split_caption_into_cues(text))
    assert joined.split() == text.split()


def test_no_orphan_tail_cue():
    """Жадная набивка оставляла хвост в 4 символа отдельным кадром —
    одинокий обрывок читается как сбой вёрстки. Балансировка это чинит."""
    cues = pipeline_smart._split_caption_into_cues(LONG)
    shortest, longest = min(map(len, cues)), max(map(len, cues))
    assert shortest >= longest * 0.4, cues


def _parse_srt(path):
    out = []
    for chunk in open(path, encoding="utf-8").read().strip().split("\n\n"):
        lines = chunk.split("\n")
        if len(lines) < 3:
            continue
        start, end = lines[1].split(" --> ")
        out.append((_sec(start), _sec(end), lines[2:]))
    return out


def _sec(stamp):
    hh, mm, rest = stamp.split(":")
    ss, ms = rest.split(",")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000


def test_written_file_tiles_the_block_window(tmp_path):
    """cue плотно покрывают окно блока: без нахлёстов (два субтитра разом
    на экране) и без выхода за границы самого блока."""
    blocks = [{"text": LONG, "pause_after": 0.0, "section": "BLOCK 1"},
              {"text": SHORT, "pause_after": 0.0, "section": "BLOCK 1"}]
    starts, durs = [0.0, 14.0], [14.0, 3.0]
    pipeline_smart.write_subtitles(str(tmp_path), blocks, starts, durs)
    cues = _parse_srt(os.path.join(str(tmp_path), "subtitles.srt"))
    assert len(cues) > len(blocks)
    for prev, cur in zip(cues, cues[1:]):
        assert cur[0] >= prev[1] - 1e-6, (prev, cur)
    assert cues[0][0] == pytest.approx(0.0, abs=0.01)
    assert cues[-1][1] <= starts[-1] + durs[-1] + 0.31


def test_written_file_has_no_line_over_the_standard(tmp_path):
    blocks = [{"text": LONG, "pause_after": 0.0, "section": "BLOCK 1"}]
    pipeline_smart.write_subtitles(str(tmp_path), blocks, [0.0], [14.0])
    for _, _, body in _parse_srt(os.path.join(str(tmp_path), "subtitles.srt")):
        assert len(body) <= pipeline_smart.SRT_MAX_LINES
        for line in body:
            assert len(line) <= pipeline_smart.SRT_MAX_LINE_CHARS, line
