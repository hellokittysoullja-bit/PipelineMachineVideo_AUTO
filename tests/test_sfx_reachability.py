# -*- coding: utf-8 -*-
"""Звук должен иметь ГДЕ прозвучать — две дыры, найденные готовым роликом.

Замер на videos/_test60s (первый полный прогон с живым звуком): за весь
эпизод принят ОДИН кюй из четырёх. `media_plan/sfx_plan.json`:

    chapter  19.68с  gap_too_short          пауза 0.271с (ассет 0.38с)
    object   31.78с  no_silence_for_object  [sfx:armour_clank]
    object   36.14с  no_silence_for_object  [sfx:armour_clank]

Обе причины — не настройка порогов, а устройство:

1. У тега [sfx:] НЕ БЫЛО рабочей позиции вообще. CLAUDE.md предписывает
   ставить его В НАЧАЛЕ фразы — там merge_next склеивал следующий текст с
   ПРЕДЫДУЩИМ блоком (три блока превращались в два, монтажный рез исчезал,
   звук получал word_pos по чужой фразе). В середине фразы разбор верный,
   но тишины рядом нет, и планировщик честно отбрасывает кюй.

2. [pause] в КОНЦЕ секции движок сворачивает почти в ноль: замер по
   alignment — 0.243с против 1.027с у того же тега в середине текста.
   Плюс склейка секций шла встык. Починить это тегом в тексте нельзя в
   принципе — тег всегда последний в своём заказе.
"""
import csv
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

import script_parser as sp  # noqa: E402

ALIGN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "fixtures", "alignment_sfx_tag")


def _blocks(text, tmp_path):
    p = tmp_path / "s.txt"
    p.write_text("=== HOOK ===\n" + text + "\n", encoding="utf-8")
    return sp.parse_blocks(str(p))


class TestSfxTagHasAWorkingPosition:
    """Негативный контроль: снять flush() в ветке SFX — и первый же тест падает."""

    TEXT_START = ("Первая фраза тут.[pause][sfx:armour_clank]"
                  "Вторая фраза про доспех.[pause]Третья фраза.")
    TEXT_MID = ("Первая фраза тут.[pause]Вторая фраза[sfx:armour_clank]"
                " про доспех.[pause]Третья фраза.")

    def test_tag_at_phrase_start_does_not_eat_the_block_boundary(self, tmp_path):
        """Позиция, которую CLAUDE.md прямо предписывает автору."""
        blocks = _blocks(self.TEXT_START, tmp_path)
        assert len(blocks) == 3, [b["text"] for b in blocks]
        assert blocks[0]["text"] == "Первая фраза тут."
        assert blocks[1]["text"] == "Вторая фраза про доспех."

    def test_tag_at_phrase_start_anchors_to_its_own_block(self, tmp_path):
        blocks = _blocks(self.TEXT_START, tmp_path)
        assert blocks[0]["sfx"] == []
        assert [s["name"] for s in blocks[1]["sfx"]] == ["armour_clank"]
        # word_pos=0 — звук относится к НАЧАЛУ своей фразы, а не к хвосту чужой.
        assert blocks[1]["sfx"][0]["word_pos"] == 0

    def test_the_pause_before_the_tag_survives(self, tmp_path):
        """Тишина, в которой кюй и должен жить, обязана остаться на месте:
        без pause_after у предыдущего блока планировщик снова скажет
        no_silence_for_object."""
        blocks = _blocks(self.TEXT_START, tmp_path)
        assert blocks[0]["pause_after"] == pytest.approx(0.8)

    def test_mid_phrase_position_is_unchanged(self, tmp_path):
        """Правка обязана быть односторонней: середина фразы работала как
        работала (разбор верный), её трогать нельзя."""
        blocks = _blocks(self.TEXT_MID, tmp_path)
        assert len(blocks) == 3
        assert [s["name"] for s in blocks[1]["sfx"]] == ["armour_clank"]
        assert blocks[1]["sfx"][0]["word_pos"] == 2

    def test_tag_without_a_pause_still_merges(self, tmp_path):
        """Без паузы тег стоит ВНУТРИ фразы и рез не создаёт — исходный
        замысел merge_next, он не должен пострадать."""
        blocks = _blocks("Одна сплошная[sfx:armour_clank] фраза без пауз.", tmp_path)
        assert len(blocks) == 1
        assert blocks[0]["text"] == "Одна сплошная фраза без пауз."


class TestSpeechBoundsIgnoreTags:
    def test_trailing_pause_tag_is_silence_not_speech(self):
        chars = [("д", 0.0, 0.2), ("а", 0.2, 0.4)] + \
                [(c, 0.4 + i * 0.03, 0.43 + i * 0.03) for i, c in enumerate("[pause]")]
        assert sp.speech_bounds_from_alignment(chars) == (0.0, 0.4)

    def test_no_real_chars_is_none(self):
        chars = [(c, i * 0.1, i * 0.1 + 0.1) for i, c in enumerate("[pause]")]
        assert sp.speech_bounds_from_alignment(chars) is None


class TestSectionBoundaryGap:
    """Фикстура — РЕАЛЬНЫЙ alignment оплаченного заказа: у синтетики не было
    бы ни хвостового [pause] на 0.243с, ни настоящего стыка 0.271с."""

    def _alignment(self, name):
        path = os.path.join(ALIGN_DIR, name)
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        return [(r["char"], float(r["start"]), float(r["end"])) for r in rows]

    def test_measured_tail_silence_matches_the_real_order(self):
        import lumean_tts as lt
        al = self._alignment("01.csv")
        # Длительность секции берём из самого alignment — файла аудио в git нет.
        dur = al[-1][2]
        head, tail = lt.section_edge_silence(al, dur)
        assert head == pytest.approx(0.0, abs=0.01)
        assert tail == pytest.approx(0.0, abs=0.05)

    def test_target_fits_the_long_asset_and_stays_under_the_trim_threshold(self):
        """Оба конца коридора — не на глаз, а из чужих констант."""
        import lumean_tts as lt
        long_asset = 0.90     # assets/sfx/transition/chapter_turn_long.flac
        headroom = 0.03       # sfx_plan.CHAPTER_HEADROOM_SEC
        assert lt.SECTION_GAP_TARGET_SEC >= long_asset + headroom
        import fix_pauses
        assert lt.SECTION_GAP_TARGET_SEC < fix_pauses.THRESH_SEC

    def test_pad_closes_the_measured_gap_exactly(self, monkeypatch):
        import lumean_tts as lt
        hook = [("а", 0.0, 21.28)] + [(c, 21.28 + i * 0.03, 21.31 + i * 0.03)
                                      for i, c in enumerate("[pause]")]
        blk = [("б", 0.0, 29.776)]
        monkeypatch.setattr(lt, "audio_duration",
                            lambda p: 21.551 if "00" in p else 29.806)
        res = [{"section": "HOOK", "audio_path": "s00.mp3", "alignment": hook},
               {"section": "BLOCK 1", "audio_path": "s01.mp3", "alignment": blk}]
        pads = lt.section_gap_pads(res)
        assert pads[0] == 0.0          # перед первой секцией паузе взяться неоткуда
        tail = lt.section_edge_silence(hook, 21.551)[1]
        head = lt.section_edge_silence(blk, 29.806)[0]
        assert tail + head + pads[1] == pytest.approx(lt.SECTION_GAP_TARGET_SEC, abs=0.001)

    def test_existing_silence_is_counted_not_ignored(self, monkeypatch):
        """Слепая добавка раздула бы уже достаточный стык. Если тишины уже
        хватает — не добавляем ничего."""
        import lumean_tts as lt
        monkeypatch.setattr(lt, "audio_duration", lambda p: 10.0)
        res = [{"section": "A", "audio_path": "a", "alignment": [("а", 0.0, 8.0)]},
               {"section": "B", "audio_path": "b", "alignment": [("б", 0.5, 10.0)]}]
        assert lt.section_gap_pads(res)[1] == 0.0

    def test_offsets_include_the_pads(self):
        """Карта смещений описывает РЕАЛЬНЫЙ audio.mp3. Забыть про паузы
        значило бы сдвинуть весь тайминг после первой секции — ровно тот
        класс, ради которого section_offsets.json и существует."""
        src = open(os.path.join(SCRIPTS_DIR, "lumean_tts.py"), encoding="utf-8").read()
        assert "global_offset += real_pads[idx]" in src
        assert "audio_out, temp_dir, pads)" in src


class TestPadLengthComesFromTheRealFile:
    """Запрошенная длина паузы и реальная — разные числа.

    Замер 14.09: запрошено 0.679с, mp3 отдал 0.705с (кадр 1152 сэмпла ≈26мс,
    короче не бывает). Смещения секций, посчитанные по запрошенной, увели бы
    каждую секцию после первой на эти 26мс — при допуске привязки реза к
    фразе в полкадра, 21мс. Тот же класс, что уже ловили у бесшовной петли."""

    def test_concat_returns_measured_pads(self, tmp_path, monkeypatch):
        import lumean_tts as lt
        made = {}

        def fake_silence(seconds, path, reference):
            made[path] = seconds
            open(path, "wb").write(b"x")
            return path

        monkeypatch.setattr(lt, "make_silence", fake_silence)
        # Реальный файл на 26мс длиннее запрошенного — ровно найденный случай.
        monkeypatch.setattr(lt, "audio_duration",
                            lambda p: (made.get(p, 0.0) + 0.026) if p in made else 10.0)
        monkeypatch.setattr(lt.subprocess, "run",
                            lambda *a, **k: type("R", (), {"returncode": 0, "stderr": ""})())
        monkeypatch.setattr(lt.os, "replace", lambda a, b: None)
        real = lt.concat_audio(["a.mp3", "b.mp3"], str(tmp_path / "out.mp3"),
                               str(tmp_path), [0.0, 0.679])
        assert real[0] == 0.0
        assert real[1] == pytest.approx(0.705, abs=0.001), real

    def test_offsets_are_computed_after_concat_not_before(self):
        """Порядок здесь — часть гарантии: реальную длину паузы знает только
        собранный файл."""
        src = open(os.path.join(SCRIPTS_DIR, "lumean_tts.py"), encoding="utf-8").read()
        assert src.index("real_pads = concat_audio(") < src.index("global_offset += real_pads[idx]")
        assert "global_offset += pads[idx]" not in src
