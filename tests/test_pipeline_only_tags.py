# -*- coding: utf-8 -*-
"""Словарь пайплайн-only тегов — ОДИН, и обе стороны сравнения чистятся одинаково.

Найдено не чтением кода, а живым платным прогоном 14.09 (videos/_test60s):
в заказ Lumean уехал литеральный "[sfx:armour_clank]". Вслух он прочитан НЕ
был — замер по alignment дал 0.0115 с/символ против 0.049 у настоящей речи, —
но в посимвольный alignment попал. Дальше:

  * script.txt тег содержит, и speech_chars_of_text() снимает его обобщённо;
  * alignment тег содержит, а _clean_timed_chars() чистила ПО СПИСКУ из трёх
    имён — скобки снимал фильтр символов, буквы "sfx:armour_clank" оставались;
  * сходство текста блока с озвученным 1.000 -> 0.889 при пороге 0.9, то есть
    PHRASE LOCK выключился на ВЕСЬ эпизод и кадры поехали по оценочным
    длительностям вместо реальных онсетов речи.

Фикстура — РЕАЛЬНЫЙ alignment того самого оплаченного заказа, а не синтетика:
синтетика не воспроизвела бы ни того, что тег приходит без скобок-разделителей
вокруг, ни его настоящей длительности.
"""
import csv
import difflib
import os
import re
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)
sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]

import pipeline_smart as ps  # noqa: E402
import script_parser as sp  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "fixtures", "alignment_sfx_tag", "01.csv")


class TestOneVocabulary:
    """Копия словаря в lumean_tts.py — это и был корень."""

    @pytest.mark.parametrize("tag", ["[stat:20–30 КГ]", "[climax]",
                                     "[sfx:armour_clank]", "[hush]"])
    def test_pipeline_only_tags_never_reach_tts(self, tag):
        got = sp.strip_pipeline_only_tags(f"до {tag} после")
        assert re.sub(r'\s+', ' ', got).strip() == "до после"

    @pytest.mark.parametrize("tag", ["[pause]", "[short pause]", "[energetic]",
                                     "[slowly]", "[emphasis]"])
    def test_engine_tags_are_kept_verbatim(self, tag):
        """Обратная гарантия, без неё правка легко «почистила» бы лишнее:
        это теги движка (ЧАСТЬ 10), TTS обязан их увидеть."""
        assert tag in sp.strip_pipeline_only_tags(f"до {tag} после")

    def test_lumean_has_no_second_copy(self):
        src = open(os.path.join(SCRIPTS_DIR, "lumean_tts.py"), encoding="utf-8").read()
        assert "strip_pipeline_only_tags" in src
        # Негативный контроль: вернуть сюда свой регексп — и тест падает.
        assert r"re.sub(r'\[stat:" not in src
        assert 'replace("[climax]"' not in src

    def test_parse_blocks_marks_exactly_the_tags_the_vocabulary_knows(self):
        """Гвард от следующего такого же расхождения: parse_blocks() уводит
        пайплайн-only теги в управляющие маркеры \\x01..\\x04. Каждый такой
        тег обязан сниматься и словарём для TTS."""
        for tag in ("[stat:20 КГ]", "[climax]", "[sfx:armour_clank]", "[hush]"):
            assert sp.PIPELINE_ONLY_TAG_RE.search(tag), tag


class TestBothSidesCleanedTheSameWay:
    """Спан вместо списка имён — это и есть то, что вернуло PHRASE LOCK
    на УЖЕ ОПЛАЧЕННОЙ записи, без повторного заказа."""

    def _second_segment(self):
        rows = list(csv.DictReader(open(FIXTURE, encoding="utf-8")))
        chars = [(r["char"], float(r["start"]), float(r["end"])) for r in rows]
        text = "".join(c for c, s, e in chars)
        segs, pos = [], 0
        for m in ps.ALIGNMENT_TAG_RE.finditer(text):
            segs.append(chars[pos:m.start()])
            pos = m.end()
        segs.append(chars[pos:])
        return segs[1]

    def test_unknown_tag_does_not_break_phrase_lock(self):
        """Главный тест. Со старым фильтром (список из трёх имён) got содержит
        "sfx:armour_clank" и ratio=0.889 < ONSET_TEXT_MATCH_MIN_RATIO —
        проверено прогоном этого теста на прежнем коде."""
        clean = ps._clean_timed_chars(self._second_segment())
        want = ps.speech_chars_of_text(
            "Полный боевой доспех пятнадцатого века весит двадцать-тридцать "
            "килограммов.")
        got = "".join(c for c, s, e in clean[:len(want)])
        assert "sfx" not in got, got
        ratio = difflib.SequenceMatcher(None, want.lower(), got.lower()).ratio()
        assert ratio >= ps.ONSET_TEXT_MATCH_MIN_RATIO, (ratio, got)

    def test_engine_tag_letters_are_stripped_too(self):
        """[energetic] покрывался старым списком — спан обязан покрывать его
        по-прежнему, иначе правка была бы разменом, а не апгрейдом."""
        seg = [(c, i * 0.1, i * 0.1 + 0.1) for i, c in enumerate("да[energetic]нет")]
        got = "".join(c for c, s, e in ps._clean_timed_chars(seg))
        assert got == "данет"

    def test_tag_letters_never_become_a_hook_caption_word(self):
        """Тот же корень, третье место: load_hook_word_timings() держала свой
        список, и буквы незнакомого тега стали бы отдельным СЛОВОМ подписи со
        своим временем на экране."""
        text = "".join(c for c, s, e in self._second_segment())
        excluded = set()
        for m in ps.ALIGNMENT_TAG_SPAN_RE.finditer(text):
            excluded.update(range(m.start(), m.end()))
        words, cur = [], []
        for j, c in enumerate(text):
            if j in excluded or c.isspace() or c in "[]":
                if cur:
                    words.append("".join(cur))
                    cur = []
            else:
                cur.append(c)
        if cur:
            words.append("".join(cur))
        assert not [w for w in words if "sfx" in w or "armour_clank" in w], words


def test_no_list_of_tag_names_survives_in_the_alignment_filters():
    """Четыре места чистили alignment, у трёх был свой список имён. Негативный
    контроль: вернуть цикл `for tag in ALIGNMENT_STRIP_TAGS` в любое из них —
    и тест падает."""
    ps_src = open(os.path.join(SCRIPTS_DIR, "pipeline_smart.py"), encoding="utf-8").read()
    ss_src = open(os.path.join(SCRIPTS_DIR, "section_sync.py"), encoding="utf-8").read()
    assert "ALIGNMENT_TAG_SPAN_RE" in ps_src
    assert "for tag in ALIGNMENT_STRIP_TAGS" not in ps_src
    assert "for tag in ps.ALIGNMENT_STRIP_TAGS" not in ss_src
    assert "ps.ALIGNMENT_TAG_SPAN_RE" in ss_src


class TestFfmpegFilterPreflight:
    """Отсутствующий фильтр обязан называться ДО рендера.

    Реальная цена молчания (14.09): ffmpeg из imageio-ffmpeg собран без
    drawtext, каждый клип с плашкой падал 'Filter not found' три попытки
    подряд, имя фильтра тонуло в эхе всего filter_complex, и прогон встал
    на строгом гейте без внятной причины."""

    def test_missing_filter_for_an_enabled_layer_is_named(self, capsys):
        have = ps.available_ffmpeg_filters()
        if not have:
            pytest.skip("ffmpeg не отвечает — гейт и не должен срабатывать")
        saved = ps._FFMPEG_FILTERS_CACHE
        try:
            ps._FFMPEG_FILTERS_CACHE = have - {"drawtext"}
            missing = ps.check_ffmpeg_filters()
            assert ("drawtext", "ON_SCREEN_TEXT") in missing
            assert "drawtext" in capsys.readouterr().out
        finally:
            ps._FFMPEG_FILTERS_CACHE = saved

    def test_disabled_layer_is_not_gated(self):
        """Выключенный слой на сборке без его фильтра — не проблема, и
        ругаться на него значило бы приучать проходить мимо предупреждения."""
        have = ps.available_ffmpeg_filters()
        if not have:
            pytest.skip("ffmpeg не отвечает")
        saved = ps._FFMPEG_FILTERS_CACHE
        try:
            ps._FFMPEG_FILTERS_CACHE = have - {"deflicker"}
            names = [f for f, _ in ps.check_ffmpeg_filters()]
            expected = "deflicker" in names
            assert expected == ps.feature_flags.enabled("DEFLICKER_ENABLED")
        finally:
            ps._FFMPEG_FILTERS_CACHE = saved

    def test_unavailable_ffmpeg_never_blocks(self):
        """Не смогли спросить ffmpeg — не гейтим (тот же fail-open, что у
        остальных проверок окружения)."""
        saved = ps._FFMPEG_FILTERS_CACHE
        try:
            ps._FFMPEG_FILTERS_CACHE = set()
            assert ps.check_ffmpeg_filters() == []
        finally:
            ps._FFMPEG_FILTERS_CACHE = saved
