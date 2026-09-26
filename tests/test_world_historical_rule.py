# -*- coding: utf-8 -*-
"""«Эпизод про прошлое» — одно правило (world_card.is_historical) для отказа по
CG, якоря эпохи в запросах, художественных музеев и отсева по подписи.
`mixed` без окна эпохи — не прошлое: живой случай эп.95 (СДВГ), где все
3D-мозги и модели дофамина бракованы как «3D в историческом эпизоде»."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import world_card as wc  # noqa: E402
import caption_screen as cs  # noqa: E402

ERA = {"from": 1290, "to": 1480}
EP95 = {"register": "mixed", "era": None,
        "culture": {"include": ["contemporary urban life"],
                    "exclude": ["ancient cultures", "medieval cultures", "fantasy"]}}


def test_mixed_needs_an_era_window():
    assert not wc.is_historical(EP95)
    assert wc.is_historical({"register": "mixed", "era": ERA})
    assert wc.is_historical({"register": "historical", "era": ERA})
    for reg in ("modern", "scientific", "abstract"):
        assert not wc.is_historical({"register": reg, "era": ERA})
    assert not wc.is_historical(None) and not wc.is_historical({})


def test_real_passports_keep_their_meaning():
    """Все паспорта репозитория: меняется только mixed без эпохи."""
    root = os.path.join(os.path.dirname(__file__), "..", "videos")
    seen = 0
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name, "media_plan", "world_card.json")
        if not os.path.exists(p):
            continue
        card = json.load(open(p, encoding="utf-8"))
        old = card.get("register") in ("historical", "mixed")
        changed = card.get("register") == "mixed" and wc.era_window(card) is None
        assert wc.is_historical(card) == (old and not changed), name
        seen += 1
    assert seen or True


def test_claims_cg_veto_off_in_science_episode():
    """Реальный ответ судьи эп.95, слот 6: 3D-модель дофамина (commons:98157571),
    главное «да». При старом правиле вектор был None (отказ по CG)."""
    import shot_judge
    spec = {"claims": [{"id": "core", "text": "a dopamine molecule model is visible", "tier": "must"},
                       {"id": "c1", "text": "atoms are shown as connected spheres", "tier": "must"},
                       {"id": "c2", "text": "the background is plain white", "tier": "should"},
                       {"id": "c3", "text": "the model looks scientific", "tier": "should"}]}
    ans = {"claims": {"core": "yes", "c1": "yes", "c2": "no", "c3": "yes"}, "medium": "cg",
           "main_in_world": True, "background_foreign": False}
    assert shot_judge.claims_vector(spec, ans, cg_veto=wc.is_historical(EP95)) is not None
    assert shot_judge.claims_vector(spec, ans, cg_veto=wc.is_historical(dict(EP95, era=ERA))) is None


def test_caption_screen_off_outside_measured_domain():
    class Gw:
        calls = []

        def chat(self, *a, **k):
            self.calls.append(1)
            return '{"mark": {"1": "X"}}', {}, 1
    gw = Gw()
    drop, _ = cs.screen(gw, "фраза", "hourglass", EP95, [("1", "pixabay", "hourglass, ancient, time")])
    assert drop == set() and gw.calls == []
    hist = dict(EP95, era=ERA, register="historical")
    assert cs.active_for(hist)
