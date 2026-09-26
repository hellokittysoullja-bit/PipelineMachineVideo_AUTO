# -*- coding: utf-8 -*-
"""Сетка судьи сказала 0 («не по теме») — одобрение проверки по пунктам не
действует. Живой случай эп.95 (26.09): часы на 10:07 прошли как «часы у
полуночи», игрушечные шарики — как «модель дофамина»; сетка поставила 0,
код поставил на экран, потому что доверял только проверке по пунктам."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

SPEC = {"focus": "a wall clock at midnight",
        "claims": [{"id": "core", "text": "a wall clock shows a few minutes to midnight", "tier": "must"},
                   {"id": "c1", "text": "the clock hands stand just before twelve", "tier": "must"}]}
YES = {"claims": {"core": "yes", "c1": "yes"}, "main_in_world": True, "background_foreign": False,
       "medium": "photo", "why": "a clock"}


def _run(monkeypatch, tmp_path, grid):
    sys.argv = ["pipeline_smart.py", str(tmp_path)]
    import pipeline_smart as ps
    import shot_judge
    monkeypatch.setattr(ps, "TEMP_FOLDER", str(tmp_path))
    monkeypatch.setattr(shot_judge, "verify_claims", lambda *a, **k: (dict(YES), {}))
    c = {"p": {"id": 1}, "path": "x.jpg", "judge": grid, "cascade_rank": 0}
    monkeypatch.setattr(ps, "verify_finalists_of", lambda judged, more=False: [] if more else judged)
    ps._verify_finalists(2, "photo", "И в ночь перед сроком", None, [c], None, "m", None, spec=SPEC)
    return ps, c


def test_grid_zero_cannot_be_approved_by_claims(monkeypatch, tmp_path):
    ps, c = _run(monkeypatch, tmp_path, 0)
    assert ps.judge_rejected(c) and not ps.judge_approved(c)
    assert ps.verify_key(c) < (0,)


def test_grid_two_with_same_answers_stays_approved(monkeypatch, tmp_path):
    ps, c = _run(monkeypatch, tmp_path, 2)
    assert ps.judge_approved(c) and not ps.judge_rejected(c)


def test_render_and_bench_share_one_brak_rule():
    """Рендер и бенч решают «брак по проверке» одной функцией: у бенча своя
    копия не знала про сетку 0, и его «брак принят» не совпадал с тем, что
    рендер ставил на экран."""
    import inspect
    import pipeline_smart as ps
    import pool_recall
    import shot_judge
    assert shot_judge.shows_nothing(SPEC, YES, 0) and not shot_judge.shows_nothing(SPEC, YES, 2)
    assert "shot_judge.shows_nothing(" in inspect.getsource(ps._verify_finalists)
    assert "shot_judge.shows_nothing(" in inspect.getsource(pool_recall.cmd_bench)
    assert "nothing_met(" not in inspect.getsource(pool_recall.cmd_bench)
