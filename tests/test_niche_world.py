#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Правила КАНАЛА — только для эпизода мира канала (world_card.matches_channel).

Аудит 02.10 прогнал паспорт психологии (tests/fixtures/other_niche/world)
через пайплайн средневекового канала и нашёл, что правила канала действуют в
любом эпизоде: восемь ловушек вето («modern domestic interior, kitchen…»)
отклоняют ожидаемую паспортом раковину с посудой; блоклист («weight loss»,
«katana», «anime»…) режет предметы чужой ниши; домен-гвард «европейский или
азиатский клинок» отклоняет катану в эпизоде про Японию; «astronaut helmet»
уходит в сток как «european astronaut helmet»; «паническая атака» получает
приписку «charging attack»; режиссёр главы требует рыцарей.

Каждый тест ниже падает со снятой правкой (контрольный прогон со снесённым
__pycache__). Отдельно держится обратная сторона: эпизод мира канала и
эпизод без паспорта получают РОВНО прежнее.
"""
import copy
import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import pipeline_smart as ps  # noqa: E402
import shot_brief_director as sbd  # noqa: E402
import shot_planner_llm as spl  # noqa: E402
import world_card as wc  # noqa: E402

FIX = os.path.join(REPO_ROOT, "tests", "fixtures", "other_niche", "world", "world_card.json")


def _psych():
    with open(FIX, encoding="utf-8") as f:
        return json.load(f)


def _card(register="historical", era=(1300, 1500), include=(), exclude=(),
          must=("modern tourists",), anchors=("medieval",)):
    return {"schema_version": 1, "register": register,
            "era": {"from": era[0], "to": era[1]} if era else None,
            "culture": {"include": list(include), "exclude": list(exclude)},
            "must_not_show": list(must), "expected_subjects": ["a thing"],
            "era_anchor_terms": list(anchors)}


# Мир канала: как паспорта реальных эпизодов 02/03/99 (культура включает и
# европейские, и — у 03 — монгольскую; регистр historical или mixed с эпохой).
CHANNEL_LIKE = _card("mixed", (1290, 1480), include=("western european", "english"),
                     exclude=("japanese", "roman"))
CHANNEL_LIKE_MONGOL = _card("historical", (1100, 1600),
                            include=("Western European", "Holy Roman Empire", "Mongol"))
JAPAN = _card("historical", (1467, 1868), include=("Japanese",),
              exclude=("Chinese",), must=("anime character",), anchors=("edo period",))
EGYPT = _card("historical", (-3000, -30), include=("Egyptian",), anchors=("ancient egypt",))


@pytest.fixture
def episode(tmp_path, monkeypatch):
    ps.reset_world_card_cache()
    d = tmp_path / "ep"
    (d / "media_plan").mkdir(parents=True)
    monkeypatch.setattr(ps, "VIDEO_FOLDER", str(d))
    yield d
    ps.reset_world_card_cache()
    getattr(spl, "set_episode", lambda d: None)(None)


def _write(d, card):
    ps.reset_world_card_cache()
    if card is None:
        if os.path.exists(wc.path(str(d))):
            os.remove(wc.path(str(d)))
        return
    with open(wc.path(str(d)), "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False)


# --- одно правило -----------------------------------------------------------

def test_matches_channel_truth_table():
    prof = ps.CHANNEL_PROFILE
    assert wc.matches_channel(None, prof)
    assert wc.matches_channel(CHANNEL_LIKE, prof)
    assert wc.matches_channel(CHANNEL_LIKE_MONGOL, prof)
    # культуры и эпохи паспортов реальных эпизодов 02, 03, 99 (файлы не в git)
    for reg, era, inc in (
            ("mixed", (1290, 1480), ("western european", "english", "french", "flemish",
                                     "scottish", "italian", "german", "burgundian")),
            ("historical", (1100, 1600), ("Western European", "Medieval English",
                                          "Medieval French", "Flemish", "Swiss",
                                          "Holy Roman Empire", "Mongol")),
            ("mixed", (700, 1600), ("Medieval European", "French", "English", "Scandinavian"))):
        assert wc.matches_channel(_card(reg, era, include=inc), prof)
    assert not wc.matches_channel(_psych(), prof)
    assert not wc.matches_channel(JAPAN, prof)
    assert not wc.matches_channel(EGYPT, prof)
    # mixed без эпохи — не прошлое (тот же is_historical, что у отказа по 3D)
    assert not wc.matches_channel(_card("mixed", None), prof)
    # канал, не объявивший эпоху, — его правила и есть его мир
    no_era = {k: v for k, v in prof.items() if k not in ("era_from", "era_to")}
    assert wc.matches_channel(_psych(), no_era)


# --- ловушки вето и домен-гвард --------------------------------------------

def _anchors_used(monkeypatch):
    seen = {}

    def multi(img, texts):
        seen["a"] = list(texts[1:])
        return [1.0] + [0.0] * (len(texts) - 1)
    monkeypatch.setattr(ps, "clip_relevance_multi", multi)
    monkeypatch.setattr(ps, "NEGATIVE_VETO_ENABLED", True)
    ps.negative_anchor_violation("x.jpg", "kitchen sink with unwashed dishes")
    return seen.get("a")


def test_channel_traps_off_in_other_world(episode, monkeypatch):
    _write(episode, _psych())
    used = _anchors_used(monkeypatch)
    assert used == list(wc.forbidden_classes(_psych()))
    assert not any("kitchen" in a for a in used)


def test_channel_traps_kept_in_channel_world(episode, monkeypatch):
    _write(episode, CHANNEL_LIKE)
    used = _anchors_used(monkeypatch)
    assert used == list(ps.CONTENT_NEGATIVE_ANCHORS) + list(CHANNEL_LIKE["must_not_show"])
    _write(episode, None)
    assert _anchors_used(monkeypatch) == list(ps.CONTENT_NEGATIVE_ANCHORS)


def test_blade_guard_off_for_japan(episode, monkeypatch):
    if not ps.VISUAL_DOMAIN_GUARDS:
        pytest.skip("профиль без домен-гварда")
    calls = []
    monkeypatch.setattr(ps, "clip_relevance",
                        lambda img, text: calls.append(text) or (0.0 if "longsword" in text else 1.0))
    _write(episode, JAPAN)
    assert ps.visual_domain_guard_violation("x.jpg", "katana blade") == (False, None)
    assert calls == []
    _write(episode, CHANNEL_LIKE)
    assert ps.visual_domain_guard_violation("x.jpg", "katana blade")[0] is True


# --- блоклист ---------------------------------------------------------------

def test_blocklist_other_world_is_passport_only(episode):
    _write(episode, _psych())
    assert ps.content_blocklist_effective() == ()
    _write(episode, JAPAN)
    assert ps.content_blocklist_effective() == ("chinese",)


def test_blocklist_channel_world_unchanged(episode):
    _write(episode, None)
    assert ps.content_blocklist_effective() == wc.apply_culture(ps.CONTENT_ALT_BLOCKLIST, None)
    _write(episode, CHANNEL_LIKE)
    got = ps.content_blocklist_effective()
    assert got == wc.apply_culture(ps.CONTENT_ALT_BLOCKLIST, CHANNEL_LIKE)
    assert "katana" in got and "weight loss" in got


def test_culture_bound_terms_leave_with_own_culture(episode):
    """Эпизод мира канала, где японская культура своя (сравнение меча с
    катаной): katana/samurai больше не режут предмет разговора, остальное —
    как было."""
    if not ps.CULTURE_BOUND_TERMS:
        pytest.skip("профиль без culture_bound_terms")
    card = copy.deepcopy(CHANNEL_LIKE)
    card["culture"]["include"].append("japanese")
    card["culture"]["exclude"].remove("japanese")
    _write(episode, card)
    assert ps.episode_matches_channel()
    got = ps.content_blocklist_effective()
    for t in ("katana", "samurai", "kimono", "shogun", "ninja"):
        assert t not in got
    assert "hanbok" in got and "weight loss" in got and "reenactment" in got


# --- запросы ---------------------------------------------------------------

def test_no_european_qualifier_in_other_world(episode):
    _write(episode, _psych())
    for q in ("astronaut helmet", "sword fish deep sea"):
        assert "european" not in ps.disambiguate_search_query(q)
    _write(episode, JAPAN)
    assert "european" not in ps.disambiguate_search_query("sword close up")
    _write(episode, CHANNEL_LIKE)
    assert "european" in ps.disambiguate_search_query("sword close up")


def test_action_dictionary_only_in_channel_world(episode):
    if not ps.ACTION_VIDEO_QUALIFIERS:
        pytest.skip("профиль без словаря движения")
    _write(episode, None)
    assert ps.action_video_qualifier("Паническая атака накрывает внезапно") == "charging attack"
    _write(episode, _psych())
    assert ps.action_video_qualifier("Паническая атака накрывает внезапно") is None
    assert ps.action_video_qualifier("Ракета взлетает, двигатели горят") is None


def test_openverse_cascade_keeps_scene_words_in_other_world(episode):
    _write(episode, _psych())
    assert ps._openverse_query_cascade("deep sea water creature") == ["deep sea water creature"]
    _write(episode, None)
    if "water" in ps.OPENVERSE_QUERY_MODIFIERS:
        assert "deep sea creature" in ps._openverse_query_cascade("deep sea water creature")


def test_channel_fallbacks_not_used_in_other_world(episode, monkeypatch):
    card = _psych()
    monkeypatch.setattr(wc, "expected_subjects", lambda c: ())
    _write(episode, card)
    assert ps.generic_fallback_queries_effective() == [ps.NEUTRAL_FALLBACK_QUERY]
    _write(episode, None)
    assert ps.generic_fallback_queries_effective() == (ps.GENERIC_FALLBACKS or [ps.NEUTRAL_FALLBACK_QUERY])


# --- подпись отбора ---------------------------------------------------------

def _gate_sig(monkeypatch):
    monkeypatch.setattr(ps, "_CANDIDATE_GATE_SIG", None)
    monkeypatch.setattr(ps, "selection_code_signature", lambda: "CODE")
    return ps.candidate_gate_signature()


def test_signature_marks_only_other_world(episode, monkeypatch):
    """Строка мира в подписи отбора — только у эпизода другого мира: у
    эпизода мира канала подпись та же, что без правки."""
    _write(episode, CHANNEL_LIKE)
    channel_real = _gate_sig(monkeypatch)
    _write(episode, _psych())
    other_real = _gate_sig(monkeypatch)
    monkeypatch.setattr(ps, "episode_matches_channel", lambda video_dir=None: True)
    _write(episode, CHANNEL_LIKE)
    assert channel_real == _gate_sig(monkeypatch)
    _write(episode, _psych())
    assert other_real != _gate_sig(monkeypatch)


# --- режиссёр главы ---------------------------------------------------------

def test_director_world_from_passport_in_other_world(episode):
    channel_text = sbd.domain_contract()
    _write(episode, _psych())
    spl.set_episode(str(episode))
    text = sbd.domain_contract()
    assert "Средневековье" not in text and text.startswith("МИР КАДРА (паспорт эпизода): modern.")
    assert "sword or dagger" in text
    assert spl.domain_anchor_words() == ()
    assert spl.brief_is_safe("a person's hands resting on a desk", "фраза") == (True, None)
    assert "weight loss" not in spl.channel_blocklist()
    # мир канала и эпизод без паспорта — прежний текст байт-в-байт
    _write(episode, CHANNEL_LIKE)
    assert sbd.domain_contract() == channel_text
    getattr(spl, "set_episode", lambda d: None)(None)
    assert sbd.domain_contract() == channel_text


def test_fill_briefs_reads_episode_world_and_restores(episode):
    _write(episode, _psych())
    text = "Руки лежат на столе."
    plan = {spl.unit_key(text): {"shot_en": "a person's hands resting on a desk"}}
    blocks = [{"text": text}]
    assert spl.fill_briefs([dict(b) for b in blocks], plan) == (0 if spl.domain_anchor_words() else 1)
    assert spl.fill_briefs(blocks, plan, video_dir=str(episode)) == 1
    assert spl._EPISODE["dir"] is None


# --- громкое предупреждение ------------------------------------------------

def test_loud_warning_without_passport(episode, capsys):
    _write(episode, None)
    msg = ps.report_episode_world()
    assert msg and "паспорта мира эпизода НЕТ" in msg
    _write(episode, _psych())
    assert "НЕ совпадает" in ps.report_episode_world()
    _write(episode, CHANNEL_LIKE)
    assert ps.report_episode_world() is None
