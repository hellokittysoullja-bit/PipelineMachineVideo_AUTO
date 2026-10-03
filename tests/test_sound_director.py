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
    assert sd.parse_answer("2 | music", 3, []) == {}
    assert sd.parse_answer("2 | music", 3, [], music=True) == {2: "music"}
    # рассуждение модели не разбирается как ответ
    assert sd.parse_answer("<think>1 | night</think>\n2 | night", 3, ["night"]) == {2: "night"}
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


def test_plan_cues_follow_text_not_index():
    """План привязан к тексту фразы: вставка фразы выше не сдвигает звук."""
    blocks = [{"section": "HOOK", "text": "новая фраза"},
              {"section": "HOOK", "text": "Рыцари пошли в атаку."},
              {"section": "B1", "text": "Флот вышел."}]
    plan = {"cues": [{"section": "HOOK", "text": "Рыцари пошли в атаку.", "name": "cavalry_horses"},
                     {"section": "B1", "text": "Флот вышел.", "name": "sea_waves"},
                     {"section": "B1", "text": "исчезнувшая", "name": "night"}]}
    assert sd.plan_cues(plan, blocks) == [(1, "amb", "cavalry_horses"), (2, "amb", "sea_waves")]


def test_author_tags_win_over_the_model(tmp_path):
    blocks = [{"section": "HOOK", "text": "x", "sound_cues": [{"type": "amb", "name": "night"}]}]

    class Boom:
        def chat(self, *a, **k):
            raise AssertionError("модель не спрашивается при тегах автора")
    cues, src = sd.episode_sound_cues(str(tmp_path), blocks, [0.0], Boom(), ["night"])
    assert src == "author" and cues == [(0, "amb", "night")]


class _FakeGateway:
    """Черновики: первая модель — конница на атаке и рынок на выкупе (брак),
    вторая — море на флоте; проверка оставляет всё, кроме рынка."""

    def __init__(self):
        self.calls = []

    def chat(self, model, content, *a, **k):
        text = content[0]["text"]
        self.calls.append(model)
        if "supervising sound editor" in text:
            out = []
            for line in text.split("Proposed cues")[1].splitlines():
                if ":" in line and "|" in line and line.split(":")[0].strip().isdigit():
                    cid = line.split(":")[0].strip()
                    out.append(f"{cid} | {'drop' if 'crowd_market' in line else 'keep'}")
            return "\n".join(out), {}, 0
        if "Флот" in text:
            return ("1 | sea_waves" if model == sd.DRAFT_MODELS[1] else "none"), {}, 0
        return ("1 | cavalry_horses\n2 | crowd_market" if model == sd.DRAFT_MODELS[0] else "1 | cavalry_horses"), {}, 0


def test_director_plan_roundtrip_with_fake_gateway(tmp_path):
    blocks = [{"section": "HOOK", "text": "Рыцари пошли в атаку."},
              {"section": "HOOK", "text": "Цена выкупа росла."},
              {"section": "B1", "text": "Флот вышел из гавани."}]
    gw = _FakeGateway()
    cues, src = sd.episode_sound_cues(str(tmp_path), blocks, [0.0, 5.0, 40.0], gw,
                                      ["cavalry_horses", "sea_waves", "crowd_market"])
    assert src == "director"
    assert sorted(cues) == [(0, "amb", "cavalry_horses"), (2, "amb", "sea_waves")]
    plan = sd.load_plan(str(tmp_path))
    dropped = [p for p in plan["proposed"] if not p["kept"]]
    assert [(p["name"], p["text"]) for p in dropped] == [("crowd_market", "Цена выкупа росла.")]
    assert set(gw.calls) == set(sd.DRAFT_MODELS) | {sd.CRITIC_MODELS[0]}


def test_critic_only_removes_and_unanswered_cue_is_dropped():
    cands = [(1, "wind_open"), (3, "battle_distant"), (5, "crows_field")]
    raw = "<think>3 | keep</think>\n1: 1 | wind_open | keep\n**2 | drop**"
    assert sd.apply_critic(cands, raw) == {1: "wind_open"}
    # проверка не может добавить: номер, которого не было в предложениях, игнорируется
    assert sd.apply_critic(cands, "9 | keep") == {}


def test_merge_drafts_is_union_without_duplicates_and_thin_repeats_keeps_first():
    got = sd.merge_drafts([{2: "night", 5: "rain_mud"}, {2: "night", 3: "wind_open"}])
    assert got == [(2, "night"), (3, "wind_open"), (5, "rain_mud")]
    assert sd.thin_repeats({4: "night", 1: "night", 2: "rain_mud"}) == {1: "night", 2: "rain_mud"}


def test_critic_failure_leaves_chapter_silent(tmp_path):
    """Проверка не ответила — глава без звуков: лишний звук хуже тишины."""
    class Broken(_FakeGateway):
        def chat(self, model, content, *a, **k):
            if "supervising sound editor" in content[0]["text"]:
                raise RuntimeError("502")
            return super().chat(model, content, *a, **k)
    blocks = [{"section": "HOOK", "text": "Рыцари пошли в атаку."}]
    plan = sd.plan_episode(str(tmp_path), blocks, [0.0], Broken(), ["cavalry_horses"])
    assert plan["cues"] == [] and plan["stats"]["critic_failed"] == 1


def test_world_summary_reads_passport_and_never_assumes_a_niche(tmp_path):
    assert sd.world_summary(str(tmp_path)) == "not specified"
    (tmp_path / "media_plan").mkdir()
    (tmp_path / "media_plan" / "world_card.json").write_text(
        '{"register": "historical", "era": {"from": 1100, "to": 1600}, '
        '"culture": {"include": ["Flemish"]}}', encoding="utf-8")
    w = sd.world_summary(str(tmp_path))
    assert "historical" in w and "1100-1600" in w and "Flemish" in w
    prompt = sd.render_prompt("T", "HOOK", [(0, 0.0, "x")], "", ["night"], w)
    assert f"World of the film: {w}" in prompt
    # в задании нет зашитой ниши: мир берётся только из паспорта
    bare = sd.render_prompt("T", "HOOK", [(0, 0.0, "x")], "", ["night"], "not specified")
    assert not any(word in bare.lower() for word in ("medieval", "history", "knight"))


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
    # Врезки несут свой файл (так их отдаёт план режиссёра); «средневековой»
    # запасной папки для врезки без файла больше нет (аудит 03_plen 02.10).
    cues = [{"start": 0.0, "end": 20.0, "role": "intro", "seed": 0, "path": files[1]},
            {"start": 40.0, "end": 52.0, "role": "chapter", "seed": 1, "path": files[0]},
            {"start": 60.0, "end": 80.0, "role": "outro", "seed": 2, "path": files[0]}]
    out = ps.build_music_cue_track(cues, 80.0, str(tmp_path))
    assert out and os.path.exists(out)
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                "-of", "csv=p=0", out], capture_output=True, text=True).stdout)
    assert abs(dur - 80.0) < 0.2


# ---------------------------------------------------------- музыка
TRACKS = {
    "a": {"id": "a", "card": "film score; moods: dark; no beat", "dur": 120},
    "b": {"id": "b", "card": "folk; sounds: guitar; steady rhythmic pattern", "dur": 90},
    "c": {"id": "c", "card": "ambient; moods: calm; no beat", "dur": 200},
    "d": {"id": "d", "card": "orchestral hybrid; drums or percussion rhythm", "dur": 100},
}


def test_parse_music_answer_and_rules():
    raw = ("intro | b\noutro | b\nsting | 2 | d\nsting | 9 | a\nbed | 1 | a\nbed | 2 | d\n"
           "bed | 3 | zzz\n**bed | 3 | c**")
    got = sd.parse_music_answer(raw, 3, set(TRACKS))
    assert got == {"intro": "b", "outro": "b", ("sting", 2): "d", ("bed", 1): "a",
                   ("bed", 2): "d", ("bed", 3): "c"}
    ruled = sd.enforce_music_rules(got, TRACKS)
    # финал не повторяет трек вступления; подложка с барабанами выброшена
    assert ruled == {"intro": "b", ("sting", 2): "d", ("bed", 1): "a", ("bed", 3): "c"}


def test_music_critic_picks_between_drafts_or_rejects():
    opts = sd.music_options([{"intro": "a", ("bed", 1): "c"}, {"intro": "b", ("bed", 1): "c"}])
    assert opts == [("intro", ["a", "b"]), (("bed", 1), ["c"])]
    assert sd.apply_music_critic(opts, "intro | B\nbed 1 | A") == {"intro": "b", ("bed", 1): "c"}
    assert sd.apply_music_critic(opts, "intro | none\nbed 1 | B") == {}


def test_music_cues_end_on_phrase_boundary_and_beds_cover_the_film(tmp_path, monkeypatch):
    import music_library
    paths = {}
    for k in TRACKS:
        p = tmp_path / f"{k}.mp3"
        p.write_bytes(b"x")
        paths[k] = dict(TRACKS[k], path=str(p))
    monkeypatch.setattr(music_library, "track_path", lambda t: t["path"])
    blocks = [{"section": "HOOK", "text": "x"}] * 3 + [{"section": "B1", "text": "y"}] * 3 + \
             [{"section": "B2", "text": "z"}] * 3 + [{"section": "FINAL", "text": "w"}] * 3
    starts = [0, 12, 33, 100, 140, 190, 260, 300, 340, 520, 560, 575]
    mplan = {"intro": "b", "outro": "a", "stings": [{"section": "B2", "id": "c"}],
             "beds": [{"section": "HOOK", "id": "c"}, {"section": "B1", "id": "c"},
                      {"section": "FINAL", "id": "a"}]}
    cues, beds = sd.music_plan_cues(mplan, blocks, starts, 600.0, paths)
    intro = [c for c in cues if c["role"] == "intro"][0]
    assert intro["end"] == 33.0 and intro["path"].endswith("b.mp3")   # граница фразы рядом с 36 с
    outro = [c for c in cues if c["role"] == "outro"][0]
    assert outro["start"] == 560.0 and outro["end"] == 600.0
    sting = [c for c in cues if c["role"] == "chapter"][0]
    assert sting["start"] == 259.0 and 270.0 <= sting["end"] <= 277.0
    # HOOK и B1 с одной подложкой — один участок без стыка; у B2 подложки нет
    assert [(b["start"], b["end"], b["id"]) for b in beds] == [(0.0, 260.0, "c"), (520.0, 600.0, "a")]


def test_library_bed_and_mix_build_without_synth(tmp_path, monkeypatch):
    import shutil
    import subprocess
    import pytest
    if not shutil.which("ffmpeg"):
        pytest.skip("нет ffmpeg")
    import pipeline_smart as ps

    def tone(name, f, d):
        p = tmp_path / name
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                        f"sine=frequency={f}:duration={d}", "-ac", "2", "-ar", "48000", str(p)], check=True)
        return str(p)
    a, b = tone("a.flac", 220, 20), tone("b.flac", 330, 30)
    voice = tone("v.flac", 500, 60)
    bed = ps.build_library_bed([{"start": 0.0, "end": 25.0, "path": a},
                                {"start": 40.0, "end": 60.0, "path": b}], 60.0, str(tmp_path / "bed.wav"))
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                "csv=p=0", bed], capture_output=True, text=True).stdout)
    assert abs(dur - 60.0) < 0.2
    monkeypatch.setattr(ps, "MUSIC_BED_FLAG", True)
    monkeypatch.setattr(ps, "MUSIC_ENABLED", False)     # синтезированного дрона нет
    out = ps.build_music_mix(voice, 60.0, str(tmp_path / "mix.wav"),
                             music_cues=[{"start": 0.0, "end": 15.0, "role": "intro", "seed": 0,
                                          "path": b}],
                             beds=[{"start": 0.0, "end": 60.0, "path": a}])
    assert out.endswith("mix.wav")
    assert ps.MUSIC_BED_DECISION.get("bed_source") == "library"


def test_repeated_kind_takes_other_recordings_then_non_overlapping_windows(monkeypatch):
    """Одна запись битвы на 208 с, четыре события по 20 с — четыре разных,
    не перекрывающихся момента, а не один кусок четыре раза."""
    import pipeline_smart as ps
    monkeypatch.setattr(ps, "library_sounds", lambda kind, name: ["/x/battle.flac"])
    monkeypatch.setattr(ps, "get_media_duration", lambda p: 208.0)
    picks = [ps.ambience_event_pick("battle_distant", n, 20.0) for n in range(4)]
    offs = sorted(p[1] for p in picks)
    assert all(b - a >= 20.0 for a, b in zip(offs, offs[1:]))
    monkeypatch.setattr(ps, "library_sounds", lambda kind, name: ["/a.flac", "/b.flac"])
    assert [ps.ambience_event_pick("x", n, 20.0)[0] for n in range(3)] == ["/a.flac", "/b.flac", "/a.flac"]
    ev = sd.ambience_events([(0, "amb", "night"), (1, "amb", "night")], [10.0, 100.0], 300.0)
    assert [e["occ"] for e in ev] == [0, 1]


def test_same_inputs_reuse_the_plan_without_calls(tmp_path):
    """Повторный рендер того же эпизода — тот же звук без вызовов модели;
    правка текста — новый вопрос."""
    blocks = [{"section": "HOOK", "text": "Рыцари пошли в атаку."}]
    gw = _FakeGateway()
    import shutil
    cache = tmp_path / "media_plan" / "sound_director_cache"
    sd.plan_episode(str(tmp_path), blocks, [0.0], gw, ["cavalry_horses"])
    n = len(gw.calls)
    shutil.rmtree(cache)     # без кэша вопросов: держит именно отпечаток плана
    sd.plan_episode(str(tmp_path), blocks, [0.0], gw, ["cavalry_horses"])
    assert len(gw.calls) == n
    sd.plan_episode(str(tmp_path), [{"section": "HOOK", "text": "Флот вышел из гавани."}], [0.0], gw,
                    ["cavalry_horses", "sea_waves"])
    assert len(gw.calls) > n


def test_owner_note_outranks_model_mood(tmp_path, monkeypatch):
    """Вердикт человека по слуху попадает в карточку, настроение CLAP — нет
    (02.10: CLAP назвал спокойный Medieval March «lively dance»)."""
    import music_library as ml
    idx = {"tracks": [{"id": "t1", "title": "X", "card": "\"X\" medieval; no beat"}]}
    monkeypatch.setattr(ml, "load_index", lambda: idx)
    monkeypatch.setattr(ml, "save_index", lambda i: None)
    assert ml.note("t1", "calm, atmospheric", log=lambda *a: None)
    assert idx["tracks"][0]["card"].endswith("owner hears: calm, atmospheric")
    ml.note("t1", "too cheerful", log=lambda *a: None)
    assert idx["tracks"][0]["card"].count("owner hears:") == 1
    heard = {"mean": {}, "rhythm": None}
    c = ml.card({"name": "X", "genre": "", "tags": []}, heard, {"lively dance": 3.0})
    assert "feels:" not in c and "lively" not in c
