#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Отсев откровенного по подписи кандидата — во всех зонах и без отката."""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.argv = ["pipeline_smart.py", REPO]
import pipeline_smart as ps  # noqa: E402

# Настоящая подпись кандидата живого замера 27.09: снимок Commons встал в
# бесплатную пятёрку фразы «Макрофаг размером с пылинку…».
EXPLICIT = {"id": "commons:197696200",
            "alt": "Pubic Hair of an adult male. Pubic hair, also called pubes, develops around the genitals, "
                   "perineum, and sometimes the inner thighs.",
            "url": "https://commons.wikimedia.org/wiki/File:Pubic_Hair_of_an_adult_male.jpg"}

# Законные подписи со словами-соседями: каждое — повод, по которому термин
# НЕ внесён в список (см. комментарий у _SAFETY_CAPTION_TERMS_DEFAULT).
LEGIT = [
    {"id": "commons:34576236", "alt": "Vaginal wet mount with a clue cell. Vaginal wet mount with a NaCl "
                                      "preparation, showing a clue cell at bottom left"},
    {"id": "commons:1", "alt": "Pubic symphysis, anterior view, x-ray"},
    {"id": "commons:2", "alt": "Anglerfish sexual dimorphism, museum specimen"},
    {"id": "commons:3", "alt": "Folio XXX, illuminated initial"},
    {"id": "commons:4", "alt": "Knight with a naked blade, manuscript miniature"},
    {"id": "commons:5", "alt": "Classical nude marble statue"},
    {"id": "commons:6", "alt": "St John's church, Penistone"},
    {"id": "123", "alt": "steel breastplate in a museum", "url": "https://www.pexels.com/photo/steel-breastplate-123/"},
]


def test_explicit_caption_is_dropped_and_neighbours_are_kept():
    assert ps.unsafe_caption(EXPLICIT)
    assert [p["id"] for p in LEGIT if ps.unsafe_caption(p)] == [], "ложное срабатывание на законной подписи"
    kept = ps.filter_alt_blocklist(LEGIT + [EXPLICIT])
    assert EXPLICIT not in kept and len(kept) == len(LEGIT)


def test_no_fallback_to_explicit_items():
    """Жанровый словарь при пустом результате откатывается на исходный список —
    откровенное вернуть нельзя ни ради непустого слота."""
    assert ps.filter_alt_blocklist([EXPLICIT]) == []


def test_paid_zone_drops_instead_of_marking(monkeypatch):
    """В платной зоне словарь жанра не выбрасывает, а помечает кадр для
    проверки мира; откровенное проверка мира не «очищает» — выбрасывается."""
    monkeypatch.setattr(ps, "blocklist_clearable", lambda index=None: True)
    out = ps.filter_pool_by_text(LEGIT + [EXPLICIT], index=0)
    assert [p["id"] for p in out] == [p["id"] for p in LEGIT]


def test_the_channel_can_replace_the_list(monkeypatch):
    monkeypatch.setattr(ps, "SAFETY_CAPTION_TERMS", ())
    assert not ps.unsafe_caption(EXPLICIT)
    assert EXPLICIT in ps.filter_alt_blocklist([EXPLICIT])


def test_the_list_is_in_the_candidate_signature(monkeypatch):
    """Правка списка меняет, кого отсеет отбор, — без неё на прогретом кэше
    прежний выбор отдавался бы как есть."""
    before = ps.candidate_gate_signature()
    monkeypatch.setattr(ps, "_CANDIDATE_GATE_SIG", None)
    monkeypatch.setattr(ps, "SAFETY_CAPTION_TERMS", ps.SAFETY_CAPTION_TERMS + ("zzz",))
    assert ps.candidate_gate_signature() != before
