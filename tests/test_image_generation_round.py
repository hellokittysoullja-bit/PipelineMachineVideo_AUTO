#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Генерация кадра как ступень лестницы слота: рисованный стиль канала,
описание кадра от мозга с откатом на бриф, сгенерированные варианты —
кандидаты того же судьи (куча только из них), потолок расходов 0 по
умолчанию, отчёт по слоту. Без сети."""
import inspect
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import selection_engine  # noqa: E402
import shot_generator as sg  # noqa: E402

SPEC = {"focus": "a timer set for a short task",
        "claims": [{"id": "core", "text": "a timer is visible", "tier": "must"},
                   {"id": "c1", "text": "it stands on a desk", "tier": "should"}],
        "queries": [{"q": "kitchen timer desk", "for": ["core"], "type": "object"}]}


class Brain:
    def __init__(self, answer="A glass hourglass on a wooden desk next to an open notebook.", exc=None):
        self.calls, self.answer, self.exc = [], answer, exc

    def chat(self, model, content, max_tokens, estimate):
        self.calls.append(content[0]["text"])
        if self.exc:
            raise self.exc
        return self.answer, {}, 7


class Painter:
    """Варианты рисуются одновременно (generation_round) — счётчик под замком."""

    def __init__(self, fail_first=False, delay=0.0):
        import threading
        self.calls, self.fail_first, self.delay = [], fail_first, delay
        self._lock = threading.Lock()
        self.active = self.peak = 0

    def image(self, model, prompt, size):
        import time
        with self._lock:
            self.calls.append(prompt)
            first = len(self.calls) == 1
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(self.delay)
            if self.fail_first and first:
                raise RuntimeError("400 content_filter")
            return [b"\x89PNG fake"], 0
        finally:
            with self._lock:
                self.active -= 1


# ---------------------------------------------------------------- модуль

def test_default_style_is_drawn_in_colour_and_forbids_text():
    s = sg.STYLE_DEFAULT.lower()
    assert "ink" in s and "colored pencil" in s and "photograph" not in s
    assert "no text" in s and "no numbers" in s


def test_channel_profile_overrides_the_style():
    assert sg.style_for({"image_generation": {"style": "  Soft gouache,  no text "}}) == "Soft gouache, no text"
    assert sg.style_for({}) == sg.STYLE_DEFAULT
    assert sg.style_for({"image_generation": {"style": ""}}) == sg.STYLE_DEFAULT


def test_subject_comes_first_style_last():
    p = sg.prompt_for("A glass hourglass on a desk.", None, "STYLE")
    assert p.startswith("A glass hourglass on a desk,") and p.endswith("STYLE")


def test_describe_uses_the_brain_and_caches_its_answer(tmp_path):
    brain = Brain()
    d1, i1 = sg.describe(brain, "m", phrase="Поставь таймер", spec=SPEC, brief="a kitchen timer",
                         card=None, cache_dir=str(tmp_path))
    d2, i2 = sg.describe(brain, "m", phrase="Поставь таймер", spec=SPEC, brief="a kitchen timer",
                         card=None, cache_dir=str(tmp_path))
    assert d1 == d2 == "A glass hourglass on a wooden desk next to an open notebook."
    assert len(brain.calls) == 1 and i2.get("cache_hit")
    q = brain.calls[0]
    assert "a timer is visible" in q and "it stands on a desk" not in q      # только обязательные
    assert "hourglass" in q and "No writing anywhere" in q                   # правила из проб


def test_describe_falls_back_to_the_brief(tmp_path):
    d, info = sg.describe(Brain(exc=RuntimeError("шлюз лёг")), "m", phrase="x", spec=SPEC,
                          brief="a kitchen timer", card=None, cache_dir=str(tmp_path))
    assert d == "a kitchen timer" and info["origin"] == "brief" and "шлюз лёг" in info["error"]
    d, info = sg.describe(None, "m", phrase="x", spec=SPEC, brief=None, card=None)
    assert d == SPEC["focus"] and info["origin"] == "brief"
    d, _ = sg.describe(Brain(answer="   "), "m", phrase="x", spec=SPEC, brief="b", card=None)
    assert d == "b"


def test_variants_are_separate_pictures(tmp_path):
    p = Painter()
    a = sg.generate(p, "an hourglass", None, str(tmp_path), variant=0)
    b = sg.generate(p, "an hourglass", None, str(tmp_path), variant=1)
    assert len(p.calls) == 2 and a["path"] != b["path"] and a["key"] != b["key"]


def test_candidate_is_a_local_pool_candidate_with_provenance(tmp_path):
    meta = sg.generate(Painter(), "an hourglass", None, str(tmp_path))
    c = sg.candidate(meta)
    assert c["id"] == "gen:" + meta["key"] and c["url"] == "" and c["alt"] == ""
    assert c["src"]["large2x"].startswith("file://") and c["src"]["large2x"].endswith(".png")
    assert c["_gen_meta"]["license"] == "Apache-2.0" and "hourglass" in c["_gen_meta"]["prompt"]


# ---------------------------------------------------------------- пайплайн

def _ps():
    sys.argv = ["pipeline_smart.py", REPO]
    import pipeline_smart as ps
    return ps


def _request(query="kitchen timer desk"):
    return selection_engine.SlotRequest(
        index=6, query=query, extra_queries=(), text_key=None,
        shot_brief="a kitchen timer", shot_spec=SPEC, block_text="Поставь таймер.",
        arbiter_text=None, is_opening=False, slot_dur=4.0, action_qualifier=None,
        target_luma=None, director_score_fn=None, director_assist=False, director_report=None,
        video_score_fn=None, used_photo_ids=set(), used_video_ids=set(), used_hashes=[],
        recent_sizes=[])


def test_generated_pool_replaces_every_source(monkeypatch):
    """Пока задана куча генерации, фото-адаптер не ходит ни в один источник:
    поиск этот слот уже прошёл дважды."""
    ps = _ps()
    for f in ("_shelf_search_photos", "_museum_search_photos", "_commons_search_photos",
              "_openverse_search_photos", "_pexels_search_photos",
              "_pixabay_search_photos", "_unsplash_search_photos"):
        monkeypatch.setattr(ps, f, lambda *a, **k: (_ for _ in ()).throw(AssertionError("источник")))
    items = [{"id": "gen:a", "alt": "", "url": "", "src": {"medium": "file:///a.png"}},
             {"id": "gen:b", "alt": "", "url": "", "src": {"medium": "file:///b.png"}}]
    with ps.generated_pool(items):
        jobs = ps.PHOTO_ADAPTER.source_jobs(_request(), "kitchen timer desk")
        pool = selection_engine.build_pool(_request(), ps.PHOTO_ADAPTER)
    assert [n for n, _j in jobs] == ["gen"]
    assert [c["id"] for c in pool] == ["gen:a", "gen:b"]
    assert ps.GENERATED_POOL.get() is None


def test_generated_candidate_is_its_own_source_with_provenance():
    ps = _ps()
    c = {"id": "gen:abc", "_gen_meta": {"model": "am/flux.2-klein-4b", "license": "Apache-2.0",
                                        "prompt": "an hourglass"}}
    assert ps.candidate_source(c) == "gen"
    prov = ps.candidate_provenance(c)
    assert prov["license"] == "Apache-2.0" and prov["channel"] == "gen"


def test_generation_gateway_spends_nothing_by_default(monkeypatch):
    ps = _ps()
    ps._GENERATION_GATEWAY.clear()
    monkeypatch.delenv("IMAGE_GEN_MAX_SPEND", raising=False)
    assert ps._generation_gateway().spend_cap == 0
    ps._GENERATION_GATEWAY.clear()
    monkeypatch.setenv("IMAGE_GEN_MAX_SPEND", "5000")
    assert ps._generation_gateway().spend_cap == 5000
    ps._GENERATION_GATEWAY.clear()


def test_generation_round_off_without_flag_key_or_spec(monkeypatch):
    ps = _ps()
    block = {"text": "Поставь таймер.", "shot_spec": SPEC}
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", "k")
    monkeypatch.setenv("IMAGE_GENERATION", "0")
    assert ps.generation_round(6, block, _request()) is None
    monkeypatch.setenv("IMAGE_GENERATION", "1")
    assert ps.generation_round(6, {"text": "x", "shot_spec": None}, _request()) is None
    monkeypatch.delenv("LLM_GATEWAY_API_KEY")
    assert ps.generation_round(6, block, _request()) is None


def _live_round(ps, monkeypatch, tmp_path, painter):
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", "k")
    monkeypatch.setenv("IMAGE_GENERATION", "1")
    monkeypatch.setattr(ps, "TEMP_FOLDER", str(tmp_path))
    monkeypatch.setattr(ps, "_research_gateway", lambda: Brain())
    monkeypatch.setattr(ps, "_generation_gateway", lambda: painter)
    monkeypatch.setattr(ps, "episode_world_card", lambda video_dir=None: None)
    ps.GENERATION_LOG.clear()
    return ps.generation_round(6, {"text": "Поставь таймер.", "shot_spec": SPEC}, _request(), "failed")


def test_generation_round_builds_the_request_and_candidates(monkeypatch, tmp_path):
    ps = _ps()
    painter = Painter()
    req, items = _live_round(ps, monkeypatch, tmp_path, painter)
    desc = "A glass hourglass on a wooden desk next to an open notebook."
    assert req.query == desc and req.extra_queries == () and req.shot_spec is SPEC
    base = _request()                     # бриф, фраза и слот — те же, меняется только поиск
    assert (req.shot_brief, req.block_text, req.index) == (base.shot_brief, base.block_text, base.index)
    assert [c["id"][:4] for c in items] == ["gen:"] * sg.VARIANTS and len(painter.calls) == sg.VARIANTS
    assert all(p.startswith(desc.rstrip(".")) for p in painter.calls)
    log = ps.GENERATION_LOG[-1]
    assert log["description"] == desc and log["description_origin"] == "model"
    assert len(log["variants"]) == sg.VARIANTS and log["trigger"] == "failed"


def test_a_refused_variant_does_not_lose_the_other(monkeypatch, tmp_path):
    ps = _ps()
    req, items = _live_round(ps, monkeypatch, tmp_path, Painter(fail_first=True))
    assert len(items) == sg.VARIANTS - 1 and "content_filter" in ps.GENERATION_LOG[-1]["errors"][0]


def test_generation_round_with_no_picture_is_none(monkeypatch, tmp_path):
    ps = _ps()

    class Dead:
        def image(self, *a):
            raise RuntimeError("524")
    assert _live_round(ps, monkeypatch, tmp_path, Dead()) is None
    assert ps.GENERATION_LOG[-1]["variants"] == [] and len(ps.GENERATION_LOG[-1]["errors"]) == sg.VARIANTS


def test_no_frame_goes_straight_to_generation_weak_frame_searches_first():
    """Решение владельца 27.09: кадра нет совсем — сразу генерация (секунды),
    второй круг поиска (минуты) — запасным; есть замена без главного —
    сначала второй круг: настоящий кадр ценнее рисунка."""
    ps = _ps()
    assert ps.ladder_steps("failed") == ("generation", "research")
    assert ps.ladder_steps("weak") == ("research", "generation")
    assert ps.ladder_steps(None) == ()


def test_both_last_steps_take_over_by_the_same_rule():
    """Обе ступени встают по research_takes_over: брак — никогда, замену
    вытесняет только строго лучший; после каждой ступени повод
    пересчитывается, и нашедшийся кадр останавливает лестницу."""
    ps = _ps()
    src = inspect.getsource(ps.run_slot_ladder)
    i_loop = src.index("for step in ladder:")
    body = src[i_loop:i_loop + 4000]
    assert "trigger = research_trigger(cur_att)" in body and "break" in body
    assert "research_round_request(i, b, request, trigger)" in body
    assert "generation_round(i, b, request, trigger)" in body
    assert "with generated_pool(items_g):" in body
    assert body.count("research_takes_over(trigger, cur_att, got_att)") == 2


def test_slot_steps_are_timed():
    """Время слота по ступеням (STAGE_TIMER): первый вид, второй вид,
    вторая страница, второй круг, генерация и слот целиком."""
    ps = _ps()
    src = inspect.getsource(ps.main) + inspect.getsource(ps.run_slot_ladder)
    for name in ("slot_first", "slot_other_kind", "slot_page2", "slot_research",
                 "slot_generation", "slot_total"):
        assert f'"{name}"' in src, name


def test_report_is_written_when_generation_ran():
    ps = _ps()
    src = inspect.getsource(ps.main)
    assert "image_generation_report.json" in src and "GENERATION_LOG.clear()" in src


def test_variants_are_painted_at_once_and_kept_in_variant_order(monkeypatch, tmp_path):
    """Варианты независимы — рисуются одновременно (раньше 4 x 35-40 с по
    очереди); порядок кандидатов — по номеру варианта, как раньше."""
    ps = _ps()
    painter = Painter(delay=0.3)
    req, items = _live_round(ps, monkeypatch, tmp_path, painter)
    assert painter.peak == sg.VARIANTS
    prompt = painter.calls[0]
    assert ps.GENERATION_LOG[-1]["variants"] == [sg.cache_key(sg.DEFAULT_MODEL, sg.DEFAULT_SIZE, prompt, v)
                                                 for v in range(sg.VARIANTS)]
