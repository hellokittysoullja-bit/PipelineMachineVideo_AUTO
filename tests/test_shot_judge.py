#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Судья кадров: разбор ответа, кэш по содержимому, отказ без частичных оценок."""
import json
import os
import re
import sys
import threading

import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import shot_judge as sj  # noqa: E402


def _img(tmp_path, name, color):
    p = tmp_path / f"{name}.jpg"
    Image.new("RGB", (64, 48), color).save(p)
    return str(p)


class FakeGateway:
    """Отвечает на СВОЙ запрос, а не по очереди: сетки одного слота
    спрашиваются параллельно. answers — {число кадров в сетке: ответ} или
    список (один вызов на ответ, для одиночной сетки)."""

    def __init__(self, answers):
        self.answers = answers
        self.calls = []
        self.lock = threading.Lock()

    def chat(self, model, content, max_tokens, est, **_kw):
        with self.lock:
            self.calls.append(content)
            if isinstance(self.answers, dict):
                n = int(re.search(r"grid of (\d+)", content[0]["text"]).group(1))
                a = self.answers[n]
            else:
                a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a, {}, 7


@pytest.mark.parametrize("text,expect", [
    ('{"scores": {"1": 3, "2": 0}}', {1: 3, 2: 0}),
    ('```json\n{"scores": {"1": 2, "2": 1}}\n```', {1: 2, 2: 1}),
    ('{"scores": {"1": 3}}', None),                 # номер 2 без оценки
    ('{"scores": {"1": 4, "2": 0}}', None),         # вне шкалы
    ('{"scores": {"1": 1.5, "2": 0}}', None),       # не целое
    ('{"scores": {"1": true, "2": 0}}', None),      # bool — не число
    ('не json', None),
    ('{"scores": {"1": "2", "2": "2"}}', {1: 2, 2: 2}),   # строкой — живой ответ judge14
    ('{"scores": {"1": "две", "2": "2"}}', None),
    ('{"scores": {"1": "4", "2": "2"}}', None),           # вне шкалы и строкой
])
def test_parse_demands_a_score_for_every_tile(text, expect):
    assert sj.parse_scores(text, 2) == expect


def test_scores_map_back_to_candidates_and_are_cached(tmp_path):
    cands = [("a", _img(tmp_path, "a", (200, 0, 0))), ("b", _img(tmp_path, "b", (0, 200, 0)))]
    gw = FakeGateway(['{"scores": {"1": 3, "2": 1}}'])
    rep = {}
    s = sj.judge(gw, "m", phrase="Вот кинжал.", brief="a dagger", candidates=cands,
                 cache_dir=str(tmp_path / "c"), report=rep)
    assert s == {"a": 3, "b": 1} and rep["calls"] == 1 and rep["cost"] == 7
    again = sj.judge(FakeGateway([]), "m", phrase="Вот кинжал.", brief="a dagger", candidates=cands,
                     cache_dir=str(tmp_path / "c"), report=rep)
    assert again == s and rep["cache_hits"] == 1, "повторный прогон не платит"


def test_cache_key_follows_image_bytes_not_ids(tmp_path):
    a = _img(tmp_path, "a", (200, 0, 0))
    k1 = sj.cache_key("m", "q", [a])
    Image.new("RGB", (64, 48), (0, 0, 200)).save(a)
    assert sj.cache_key("m", "q", [a]) != k1, "другая картинка под тем же путём — другой ответ"


def test_any_gateway_failure_means_no_judge_at_all(tmp_path):
    cands = [(str(k), _img(tmp_path, str(k), (k * 20, 0, 0))) for k in range(11)]
    gw = FakeGateway({9: '{"scores": {' + ", ".join(f'"{k}": 2' for k in range(1, 10)) + "}}",
                      2: RuntimeError("503")})
    rep = {}
    assert sj.judge(gw, "m", phrase="p", brief="b", candidates=cands, report=rep) is None, \
        "вторая сетка не ответила — оценок нет ни у кого, без смешивания"
    assert "503" in rep["refused"]


def test_grid_is_split_into_nines(tmp_path):
    cands = [(str(k), _img(tmp_path, str(k), (k * 20, 0, 0))) for k in range(11)]
    gw = FakeGateway({9: '{"scores": {' + ", ".join(f'"{k}": 2' for k in range(1, 10)) + "}}",
                      2: '{"scores": {"1": 3, "2": 0}}'})
    s = sj.judge(gw, "m", phrase="p", brief="b", candidates=cands)
    assert len(gw.calls) == 2 and s["9"] == 3 and s["10"] == 0 and s["0"] == 2


def test_prompt_is_the_measured_one():
    """Паспорт целиком ухудшил судью, одна строка мира — улучшила (оба замера
    в докстринге модуля). Вопрос не должен незаметно обрасти ни паспортом,
    ни чем-то ещё без нового замера: без мира он прежний, с миром — ровно
    одна строка при описании кадра."""
    q = sj.question("фраза", "brief", 9)
    assert "setting" not in q and "«brief»" in q and "фраза" in q
    qs = sj.question("фраза", "brief", 9, setting="historical, 1300 AD-1500 AD")
    assert "«brief — setting: historical, 1300 AD-1500 AD»" in qs
    assert qs.replace(" — setting: historical, 1300 AD-1500 AD", "") == q
    assert sj.PROMPT_VERSION == 2


def test_setting_changes_the_cache_key_and_reaches_the_model(tmp_path):
    cands = [("a", _img(tmp_path, "a", (10, 0, 0)))]
    gw = FakeGateway({1: '{"scores": {"1": 2}}'})
    sj.judge(gw, "m", phrase="p", brief="b", candidates=cands, cache_dir=str(tmp_path / "c"),
             setting="historical, 1300 AD-1500 AD")
    sj.judge(gw, "m", phrase="p", brief="b", candidates=cands, cache_dir=str(tmp_path / "c"))
    assert len(gw.calls) == 2, "вопрос с миром и без мира — разные ключи кэша"


def test_transparent_png_is_flattened_not_striped():
    """Цвет прозрачных пикселей произволен: простое convert("RGB") показывал
    его полосами (кинжалы Pixabay «isolated» в сетке судьи)."""
    from PIL import Image
    im = Image.new("RGBA", (4, 1), (255, 0, 255, 0))     # мусорный цвет, прозрачно
    im.putpixel((0, 0), (10, 20, 30, 255))               # один видимый пиксель
    flat = sj.flat_rgb(im)
    assert flat.mode == "RGB" and flat.getpixel((0, 0)) == (10, 20, 30)
    assert flat.getpixel((3, 0)) == (235, 235, 235), "прозрачное — фон, а не мусорный цвет"


class ColourGateway:
    def __init__(self, reply):
        self.reply = reply

    def chat(self, model, content, max_tokens, est, **_kw):
        return self.reply(content), {}, 1


def test_vision_check_catches_a_blind_model():
    """qwen3.8-max молча ставил всем кадрам 0 — такой судья браковал бы всё."""
    ok, why = sj.vision_check(ColourGateway(lambda c: "I do not have vision support."), "m")
    assert not ok and "не видит" in why


def test_vision_check_is_not_passed_by_one_lucky_word():
    ok, _ = sj.vision_check(ColourGateway(lambda c: "red"), "m")
    assert not ok, "два цвета: одно угаданное слово — не зрение"


def test_vision_check_passes_a_seeing_model():
    import base64 as b64
    import io as io_
    from PIL import Image as Im

    def see(content):
        raw = b64.b64decode(content[1]["image_url"]["url"].split(",", 1)[1])
        r, g, b = Im.open(io_.BytesIO(raw)).convert("RGB").getpixel((32, 32))
        return "Red." if r > b else "Blue"
    assert sj.vision_check(ColourGateway(see), "m") == (True, "")


def test_world_asked_once_per_picture_whatever_the_phrase(tmp_path):
    """Вариант «мир отдельно»: вопрос о мире не знает фразы, поэтому одна
    картинка в двух слотах получает ОДИН ответ о мире (кэш), а утверждения
    спрашиваются по фразе каждого слота без ключей мира."""
    path = _img(tmp_path, "a", (120, 90, 60))
    spec = {"focus": "arrow", "claims": [{"id": "c1", "text": "an arrow is visible", "tier": "must"}]}

    class GW:
        def __init__(self):
            self.texts = []

        def chat(self, model, content, max_tokens, est, **_kw):
            t = content[0]["text"]
            self.texts.append(t)
            if "main_in_world: could the MAIN subject of the picture" in t:
                return '{"main_in_world": false, "background_foreign": true, "why": "x"}', {}, 3
            return '{"claims": {"c1": "yes"}, "medium": "photo", "why": "y"}', {}, 5
    gw = GW()
    got = []
    for phrase in ("Стрела летит.", "Стрела падает."):
        ans, _i = sj.verify_claims(gw, "m", phrase=phrase, spec=spec, setting="historical",
                                   path=path, cache_dir=str(tmp_path / "c"), world_separate=True)
        got.append(ans)
    worlds = [t for t in gw.texts if "main_in_world: could the MAIN subject of the picture" in t]
    claims = [t for t in gw.texts if t not in worlds]
    assert len(worlds) == 1 and len(claims) == 2
    assert all("main_in_world" not in t for t in claims)
    assert all(a["main_in_world"] is False and a["background_foreign"] is True
               and a["claims"] == {"c1": "yes"} for a in got)


def test_world_separately_unparsed_means_no_check(tmp_path):
    path = _img(tmp_path, "a", (120, 90, 60))
    spec = {"focus": "arrow", "claims": [{"id": "c1", "text": "an arrow", "tier": "must"}]}

    class GW:
        def chat(self, *a, **k):
            return "no json", {}, 1
    ans, info = sj.verify_claims(GW(), "m", phrase="p", spec=spec, setting="historical",
                                 path=path, world_separate=True)
    assert ans is None and "refused" in info


def test_cache_that_cannot_be_written_does_not_lose_the_paid_answer(tmp_path):
    """24.09: замер упал посреди прогона с OSError «нет места» при записи
    кэша — оплаченный ответ терялся вместе со всем процессом. Кэш — по
    возможности: ответ возвращается, битого файла не остаётся."""
    path = _img(tmp_path, "a", (120, 90, 60))
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")
    spec = {"focus": "arrow", "claims": [{"id": "c1", "text": "an arrow", "tier": "must"}]}

    class GW:
        def chat(self, model, content, *a, **k):
            if "main_in_world: could the MAIN subject of the picture" in content[0]["text"]:
                return '{"main_in_world": true, "background_foreign": false, "why": ""}', {}, 1
            return '{"claims": {"c1": "yes"}, "medium": "photo", "why": ""}', {}, 1
    ans, info = sj.verify_claims(GW(), "m", phrase="p", spec=spec, setting="historical",
                                 path=path, cache_dir=str(blocker / "cache"), world_separate=True)
    assert ans and ans["claims"] == {"c1": "yes"} and ans["main_in_world"] is True
    assert info.get("call")
