"""Ритм гарантирован кодом, а не желанием модели: акцент по правилу (число + мера), склейки без
таймингов речи, первый план ролика ≤ 2 с, голова героя — крупный план, детали от судьи."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import frame_planner as fp  # noqa: E402
import shots  # noqa: E402


def test_auto_accent_is_number_with_unit_only():
    assert fp.auto_accent("Ответить на одно письмо — это пять минут. А ты третий день ходишь.") == "пять минут"
    assert fp.auto_accent("Две минуты, и можно бросить.") == "две минуты"
    assert fp.auto_accent("Там было десять человек и одно окно.") == "десять человек"
    assert fp.auto_accent("Открой письмо и всё.") is None
    assert fp.auto_accent("Одно письмо лежит.") is None                    # «одно письмо» — не мера
    frames = [{"text": "Две минуты, и можно бросить.", "kind": "scene"},
              {"text": "Две минуты, и можно бросить.", "kind": "scene", "key_thought": "можно бросить"},
              {"text": "Две минуты.", "kind": "scene", "accent": "своё"}]
    assert fp.accent_pass(frames) == 1 and frames[0]["accent"] == "две минуты" and "accent" not in frames[1]


def _busy_with_subject(SW=1508, SH=848):
    busy = np.zeros((SH, SW), np.float32)
    busy[300:700, 500:1000] = 1.0                       # один большой рисунок посередине
    return busy


def test_cuts_happen_without_word_timings_when_a_view_exists():
    busy = _busy_with_subject()
    objects = [dict(name="cat", box=(500, 300, 1000, 700), role="hero"),
               dict(name="head", box=(600, 300, 900, 480), role="hero_head")]
    p = shots.plan(9.0, busy, [], objects=objects, T0=20.0)
    assert len(p["segments"]) >= 3, p["notes"]                       # 9 с без слов — всё равно резано
    assert all(s["t1"] - s["t0"] <= shots.MAX_VIEW_SEC + 0.05 for s in p["segments"])


def test_first_view_of_the_film_is_at_most_two_seconds():
    busy = _busy_with_subject()
    objects = [dict(name="cat", box=(500, 300, 1000, 700), role="hero"),
               dict(name="head", box=(600, 300, 900, 480), role="hero_head")]
    p = shots.plan(8.25, busy, [], objects=objects, T0=0.0, first_of_film=True)
    assert p["segments"][0]["t1"] <= shots.HOOK_FIRST_VIEW_SEC + 1e-6
    p2 = shots.plan(8.25, busy, [], objects=objects, T0=0.0)                 # без признака — обычный предел
    assert p2["segments"][0]["t1"] > shots.HOOK_FIRST_VIEW_SEC          # не хук — обычный предел


def test_head_alone_gives_a_close_shot_when_no_details():
    busy = _busy_with_subject()
    p0 = shots.plan(9.0, busy, [], objects=[dict(name="cat", box=(500, 300, 1000, 700), role="hero")], T0=20.0)
    p1 = shots.plan(9.0, busy, [], objects=[dict(name="cat", box=(500, 300, 1000, 700), role="hero"),
                                            dict(name="head", box=(600, 300, 900, 480), role="hero_head")], T0=20.0)
    assert len(p1["segments"]) > len(p0["segments"])


def test_objects_for_asks_judge_for_details_when_plan_has_none(tmp_path):
    import objects
    from PIL import Image, ImageDraw
    img = tmp_path / "f.png"; im = Image.new("RGB", (1264, 848), (250, 247, 240))
    ImageDraw.Draw(im).rectangle((400, 300, 800, 600), outline=(0, 0, 0), width=10); im.save(img)
    calls = []

    class GW:
        def chat(self, model, content, max_tokens, est, **kw):
            calls.append(content[0]["text"][:40])
            if "close shot" in content[0]["text"]:
                return '{"parts": ["the left edge of the box", "the top corner"]}', {}, 1
            return '{"boxes": {"1": [320, 350, 640, 700], "2": [330, 360, 500, 450], "3": [620, 350, 640, 420]}}', {}, 1
    frame = {"kind": "scene", "text": "x", "spec": {"subject": "a box"}, "hero": False}
    out = objects.objects_for(GW(), "m", str(img), frame, None, str(tmp_path / "c"))
    roles = [o["role"] for o in out]
    assert roles.count("detail") == 2 and "subject" in roles and len(calls) == 2
    frame2 = dict(frame, details=["the lid"])
    calls.clear(); objects.objects_for(GW(), "m", str(img), frame2, None, str(tmp_path / "c2"))
    assert len(calls) == 1                                               # детали плана — судью не спрашиваем


def test_hook_zone_caps_every_plan_in_the_first_seconds():
    busy = _busy_with_subject()
    objects = [dict(name="cat", box=(500, 300, 1000, 700), role="hero"),
               dict(name="head", box=(600, 300, 900, 480), role="hero_head"),
               dict(name="d", box=(520, 560, 700, 690), role="detail")]
    p = shots.plan(8.25, busy, [], objects=objects, T0=0.0, first_of_film=True)
    early = [s for s in p["segments"] if s["t0"] < shots.HOOK_ZONE_SEC]
    assert early and all(s["t1"] - s["t0"] <= shots.HOOK_FIRST_VIEW_SEC + 1e-6 for s in early)


def test_cut_after_the_key_thought_is_written():
    busy = _busy_with_subject()
    objects = [dict(name="cat", box=(500, 300, 1000, 700), role="hero"),
               dict(name="head", box=(600, 300, 900, 480), role="hero_head")]
    text = "раз два три четыре пять шесть семь восемь девять десять одиннадцать двенадцать тринадцать"
    words = [{"word": w, "start": 0.3 + 0.4*i, "end": 0.6 + 0.4*i} for i, w in enumerate(text.split())]
    p = shots.plan(6.8, busy, words, objects=objects, key="два три", key_dur=2.0, T0=20.0)
    assert p["key_time"] is not None
    cuts = [s["t0"] for s in p["segments"][1:]]
    assert cuts and all(c >= p["key_time"] + 2.0 for c in cuts)        # склейка после дописанной мысли
