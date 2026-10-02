# -*- coding: utf-8 -*-
"""Звуковой режиссёр: теги [amb:]/[music:], разбор ответа модели, события
атмосферы и врезки средневековой музыки (scripts/sound_director.py)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import script_parser  # noqa: E402
import sound_director as sd  # noqa: E402


SCRIPT = """=== METADATA ===
TITLE: Тест
=== HOOK === [energetic]
Рыцари пошли в атаку. [pause] Земля дрожала под копытами. [pause] Потом наступила тишина.
=== BLOCK 1: Море ===
Флот вышел из гавани. [pause] Ветер гнал корабли к берегу.
"""


def _parse(text, tmp_path):
    p = tmp_path / "script.txt"
    p.write_text(text, encoding="utf-8")
    return script_parser.parse_blocks(str(p))


def test_tags_do_not_change_blocks(tmp_path):
    """Тег звука не режет фразу и не попадает в её текст: иначе монтаж,
    кэш кадров и PHRASE LOCK поехали бы от одной строки разметки."""
    plain = _parse(SCRIPT, tmp_path)
    tagged = _parse(SCRIPT.replace("Земля", "[amb:cavalry_horses]Земля")
                    .replace("Флот", "[music:medieval]Флот")
                    .replace("гнал", "[amb:sea_waves]гнал"), tmp_path)
    assert [b["text"] for b in tagged] == [b["text"] for b in plain]
    assert [b["section"] for b in tagged] == [b["section"] for b in plain]
    cues = {i: b["sound_cues"] for i, b in enumerate(tagged) if b["sound_cues"]}
    assert cues[1] == [{"type": "amb", "name": "cavalry_horses", "word_pos": 0}]
    assert cues[3] == [{"type": "music", "name": "medieval", "word_pos": 0}]
    assert cues[4] == [{"type": "amb", "name": "sea_waves", "word_pos": 1}]


def test_tags_never_reach_tts_text():
    out = script_parser.strip_pipeline_only_tags("а [amb:sea_waves]б [music:medieval]в", repl="")
    assert "[" not in out and "amb" not in out and "music" not in out


def test_parse_answer_keeps_only_known_kinds_and_valid_lines():
    raw = "1 | cavalry_horses\n**3 | sea_waves**\n2 | dragon_roar\n9 | sea_waves\nnone\n1 | night"
    got = sd.parse_answer(raw, 4, ["cavalry_horses", "sea_waves", "night"])
    assert got == {1: "cavalry_horses", 3: "sea_waves"}
    assert sd.parse_answer("2 | music", 3, []) == {2: "music"}
    assert sd.parse_answer("none", 3, ["night"]) == {}


def test_events_last_about_twenty_seconds_and_start_before_the_line():
    ev = sd.ambience_events([(0, "amb", "sea_waves")], [10.0], 300.0)
    assert len(ev) == 1
    e = ev[0]
    assert e["start"] == 10.0 - sd.AMB_PREROLL_SEC
    assert abs((e["end"] - e["start"]) - sd.AMB_EVENT_SEC) < 1e-6


def test_events_rules_gap_extend_crossfade_and_min_length():
    starts = [10.0, 15.0, 25.0, 60.0, 64.0]
    cues = [(0, "amb", "sea_waves"),      # 8.5
            (1, "amb", "night"),          # 13.5 — ближе 12 с к предыдущему: лишний
            (2, "amb", "sea_waves"),      # 23.5 — тот же вид, ещё звучит: продление
            (3, "amb", "battle_distant"),  # 58.5
            (4, "amb", "crows_field")]    # 62.5 — ближе 12 с: лишний
    ev = sd.ambience_events(cues, starts, 300.0)
    assert [e["kind"] for e in ev] == ["sea_waves", "battle_distant"]
    assert ev[0]["end"] == 23.5 + sd.AMB_EVENT_SEC
    # следующее событие короче 12 с после -> предыдущее обрезается под наплыв
    ev2 = sd.ambience_events([(0, "amb", "sea_waves"), (1, "amb", "night")], [10.0, 24.0], 300.0)
    assert ev2[0]["end"] == ev2[1]["start"] + sd.AMB_CROSSFADE_SEC
    # слишком короткое событие у самого конца ролика выбрасывается
    assert sd.ambience_events([(0, "amb", "night")], [297.0], 300.0) == []


def test_unavailable_kind_is_dropped_and_same_kind_gets_different_recordings():
    cues = [(0, "amb", "sea_waves"), (1, "amb", "sea_waves"), (2, "amb", "dragon")]
    ev = sd.ambience_events(cues, [10.0, 100.0, 200.0], 300.0, available={"sea_waves"})
    assert [e["kind"] for e in ev] == ["sea_waves", "sea_waves"]
    files = 5
    assert ev[0]["seed"] % files != ev[1]["seed"] % files


def test_music_intro_always_outro_and_chapter_cap():
    starts = [float(t) for t in range(0, 1200, 10)]
    cues = [(i, "music", "medieval") for i in range(len(starts))]
    m = sd.music_cues(cues, starts, 1200.0)
    roles = [c["role"] for c in m]
    assert roles[0] == "intro" and m[0]["start"] == 0.0
    assert roles[-1] == "outro" and m[-1]["end"] == 1200.0
    chapters = [c for c in m if c["role"] == "chapter"]
    assert len(chapters) <= sd.MUSIC_MAX_CHAPTER_CUES
    for a, b in zip(chapters, chapters[1:]):
        assert b["start"] - a["start"] >= sd.MUSIC_CHAPTER_MIN_SPACING_SEC
    assert sd.music_cues([], starts, 1200.0, enabled=False) == []


def test_plan_cues_follow_text_not_index():
    """План привязан к тексту фразы: вставка фразы выше не сдвигает звук."""
    blocks = [{"section": "HOOK", "text": "новая фраза"},
              {"section": "HOOK", "text": "Рыцари пошли в атаку."},
              {"section": "B1", "text": "Флот вышел."}]
    plan = {"cues": [{"section": "HOOK", "text": "Рыцари пошли в атаку.", "name": "cavalry_horses"},
                     {"section": "B1", "text": "Флот вышел.", "name": "music"},
                     {"section": "B1", "text": "исчезнувшая", "name": "night"}]}
    assert sd.plan_cues(plan, blocks) == [(1, "amb", "cavalry_horses"), (2, "music", "medieval")]


def test_author_tags_win_over_the_model(tmp_path):
    blocks = [{"section": "HOOK", "text": "x", "sound_cues": [{"type": "amb", "name": "night"}]}]

    class Boom:
        def chat(self, *a, **k):
            raise AssertionError("модель не спрашивается при тегах автора")
    cues, src = sd.episode_sound_cues(str(tmp_path), blocks, [0.0], Boom(), ["night"])
    assert src == "author" and cues == [(0, "amb", "night")]


def test_director_plan_roundtrip_with_fake_gateway(tmp_path):
    blocks = [{"section": "HOOK", "text": "Рыцари пошли в атаку."},
              {"section": "HOOK", "text": "Цена выкупа росла."},
              {"section": "B1", "text": "Флот вышел из гавани."}]

    class Fake:
        def chat(self, model, content, *a, **k):
            text = content[0]["text"]
            if "Флот" in text:
                return "1 | music\n1 | sea_waves", {}, 0
            return "1 | cavalry_horses", {}, 0
    cues, src = sd.episode_sound_cues(str(tmp_path), blocks, [0.0, 5.0, 40.0], Fake(),
                                      ["cavalry_horses", "sea_waves"])
    assert src == "director"
    assert (0, "amb", "cavalry_horses") in cues and (2, "music", "medieval") in cues
    assert os.path.exists(sd.plan_path(str(tmp_path)))


def test_music_cue_track_builds_with_several_pieces(tmp_path, monkeypatch):
    """Регресс 02.10: номер входа ffmpeg считался по длине списка аргументов
    (по два элемента на файл), и дорожка из двух пьес не собиралась НИКОГДА —
    «средневековая музыка не собралась» на живом эпизоде."""
    import shutil
    import subprocess
    if not shutil.which("ffmpeg"):
        import pytest
        pytest.skip("нет ffmpeg")
    import pipeline_smart as ps
    files = []
    for k, f in enumerate((220, 330)):
        p = tmp_path / f"m{k}.flac"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                        f"sine=frequency={f}:duration={30 + 10 * k}", "-ac", "2", "-ar", "48000",
                        str(p)], check=True)
        files.append(str(p))
    monkeypatch.setattr(ps, "library_sounds", lambda kind, name: files)
    cues = [{"start": 0.0, "end": 20.0, "role": "intro", "seed": 0},
            {"start": 40.0, "end": 52.0, "role": "chapter", "seed": 1},
            {"start": 60.0, "end": 80.0, "role": "outro", "seed": 2}]
    out = ps.build_music_cue_track(cues, 80.0, str(tmp_path))
    assert out and os.path.exists(out)
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                "-of", "csv=p=0", out], capture_output=True, text=True).stdout)
    assert abs(dur - 80.0) < 0.2
