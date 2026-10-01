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


def test_prompt_is_description_first_and_forbids_text():
    p = sg.prompt_for("an arrow glancing off a dented steel breastplate.", MEDIEVAL)
    assert "an arrow glancing off a dented steel breastplate." in p
    assert "no text" in p.lower()


def test_no_years_reach_the_image_model():
    """Живой дефект эп.98: «set in 700 AD-2024 AD» в промпте модель рисовала
    надписью «700–2024 AD» на картинке. Годов в промпте нет ни в каком мире."""
    import re
    for card in (MEDIEVAL, EGYPT, PSYCHOLOGY, None):
        p = sg.prompt_for("a wrapped mummy", card)
        assert "set in" not in p and not re.search(r"\d", p), p


def test_describe_is_told_to_name_the_period_in_words():
    assert "never as years or digits" in sg.DESCRIBE_PROMPT and "dates, years" in sg.DESCRIBE_PROMPT


def test_module_holds_no_niche_words():
    src = open(sg.__file__, encoding="utf-8").read().lower()
    code = src.split('"""', 2)[2]  # без докстринга модуля, где пример фразы
    for w in ("medieval", "knight", "armour", "sword", "dagger"):
        assert w not in code.replace("# ", "#"), w


class Gw:
    def __init__(self, exc=None):
        self.calls, self.exc = [], exc

    def image(self, model, prompt, size, quality=None):
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
    assert b["license"] == sg.LICENSES[sg.DEFAULT_MODEL] and b["prompt"] == a["prompt"]


def test_prompt_change_misses_the_cache(tmp_path):
    gw = Gw()
    sg.generate(gw, "a dagger", MEDIEVAL, str(tmp_path))
    sg.generate(gw, "a dagger on a table", MEDIEVAL, str(tmp_path))
    sg.generate(gw, "a dagger on a table", MEDIEVAL, str(tmp_path), style="soft gouache, no text")
    assert len(gw.calls) == 3


def test_gateway_failure_is_a_reason_not_an_exception(tmp_path):
    r = sg.generate(Gw(exc=RuntimeError("400 content refused")), "a dagger", MEDIEVAL, str(tmp_path))
    assert "content refused" in r["error"] and not os.listdir(tmp_path)


def test_empty_brief_is_not_generated(tmp_path):
    gw = Gw()
    assert sg.generate(gw, "  ", MEDIEVAL, str(tmp_path))["error"] == "нет брифа" and not gw.calls
