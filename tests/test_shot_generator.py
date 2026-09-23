#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Кадр по брифу генерацией: промпт без ниши, лицензия закрыта по
умолчанию, кэш не платит дважды, сбой — не исключение. Без сети."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import shot_generator as sg  # noqa: E402

MEDIEVAL = {"register": "historical", "era": {"from": 1300, "to": 1500}}
EGYPT = {"register": "historical", "era": {"from": -2600, "to": -30}}
PSYCHOLOGY = {"register": "abstract", "era": None}


def test_prompt_is_brief_first_then_the_episode_world():
    p = sg.prompt_for("an arrow glancing off a dented steel breastplate.", MEDIEVAL)
    assert p.startswith("an arrow glancing off a dented steel breastplate,")
    assert "set in 1300 AD-1500 AD" in p and "no text" in p


def test_no_world_means_no_frame_not_someone_elses():
    """Эпизод без эпохи не получает чужую эпоху: ни «medieval», ни годов."""
    p = sg.prompt_for("a phone lying face down on a bedside table", PSYCHOLOGY)
    assert "set in" not in p and "medieval" not in p.lower()
    assert "set in" not in sg.prompt_for("a phone on a table", None)


def test_bc_years_are_written_as_bc():
    assert "set in 2600 BC-30 BC" in sg.prompt_for("a wrapped mummy", EGYPT)


def test_module_holds_no_niche_words():
    src = open(sg.__file__, encoding="utf-8").read().lower()
    code = src.split('"""', 2)[2]  # без докстринга модуля, где пример фразы
    for w in ("medieval", "knight", "armour", "sword", "dagger"):
        assert w not in code.replace("# ", "#"), w


class Gw:
    def __init__(self, exc=None):
        self.calls, self.exc = [], exc

    def image(self, model, prompt, size):
        self.calls.append((model, prompt, size))
        if self.exc:
            raise self.exc
        return [b"\xff\xd8jpeg"], 0


def test_unverified_license_is_refused_without_a_call(tmp_path):
    gw = Gw()
    r = sg.generate(gw, "a dagger", MEDIEVAL, str(tmp_path), model="some/unknown-image-model")
    assert "лицензия" in r["error"] and gw.calls == []


def test_second_run_is_served_from_cache(tmp_path):
    gw = Gw()
    a = sg.generate(gw, "a dagger", MEDIEVAL, str(tmp_path))
    b = sg.generate(gw, "a dagger", MEDIEVAL, str(tmp_path))
    assert len(gw.calls) == 1 and a["path"] == b["path"] and b["cached"] and not a["cached"]
    assert b["license"] == "Apache-2.0" and b["prompt"] == a["prompt"]


def test_prompt_change_misses_the_cache(tmp_path):
    gw = Gw()
    sg.generate(gw, "a dagger", MEDIEVAL, str(tmp_path))
    sg.generate(gw, "a dagger", EGYPT, str(tmp_path))
    assert len(gw.calls) == 2


def test_gateway_failure_is_a_reason_not_an_exception(tmp_path):
    r = sg.generate(Gw(exc=RuntimeError("400 content refused")), "a dagger", MEDIEVAL, str(tmp_path))
    assert "content refused" in r["error"] and not os.listdir(tmp_path)


def test_empty_brief_is_not_generated(tmp_path):
    gw = Gw()
    assert sg.generate(gw, "  ", MEDIEVAL, str(tmp_path))["error"] == "нет брифа" and not gw.calls
