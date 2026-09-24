"""Выбор среди равных по смыслу: судья упорядочивает ничью как кадры фильма."""
import os
import sys
import tempfile

import pytest
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]

import pipeline_smart as ps  # noqa: E402
import shot_judge  # noqa: E402


@pytest.mark.parametrize("raw,k,want", [
    ('{"order": [3, 1, 2]}', 3, [2, 0, 1]),
    ('```json\n{"order": [2, 2, 5, 1]}\n```', 3, [1, 0, 2]),   # повтор и номер вне сетки
    ('{"order": [2]}', 3, [1, 0, 2]),                          # недосказанные — в конец
    ('{"order": []}', 3, None), ("не знаю", 3, None),
])
def test_parse_order(raw, k, want):
    assert shot_judge.parse_order(raw, k) == want


class _GW:
    def __init__(self, *answers):
        self.answers, self.calls = list(answers), []

    def chat(self, model, content, *a, **kw):
        self.calls.append(content[0]["text"])
        return self.answers.pop(0), {}, 150


def _img(tmp_path, name, rgb):
    p = tmp_path / name
    Image.new("RGB", (320, 240), rgb).save(p)
    return str(p)


def test_rank_look_is_cached_by_pictures(tmp_path):
    paths = [_img(tmp_path, f"{k}.jpg", (40 * k, 10, 10)) for k in range(3)]
    gw = _GW('{"order": [3, 1, 2]}')
    order, info = shot_judge.rank_look(gw, "m", paths=paths, cache_dir=str(tmp_path / "c"))
    assert order == [2, 0, 1] and info.get("call")
    again, info2 = shot_judge.rank_look(gw, "m", paths=paths, cache_dir=str(tmp_path / "c"))
    assert again == order and info2.get("cache_hit") and len(gw.calls) == 1


def _cand(tmp_path, cid, vec, grid, rgb):
    return {"p": {"id": cid}, "path": _img(tmp_path, f"{cid}.jpg", rgb), "verify": vec, "judge": grid,
            "is_dup_free": 1, "is_readable": 1, "is_relevant": 1, "size_ok": 1, "sharp_ok": 1,
            "relevance": 0.3, "aesthetic_val": 5.0, "luma_score": 1.0, "min_d": 10}


@pytest.fixture
def judge_env(monkeypatch, tmp_path):
    monkeypatch.setenv("LOOK_TIEBREAK", "1")
    monkeypatch.setattr(ps, "SHOT_JUDGE_LOG", [])
    monkeypatch.setattr(ps, "TEMP_FOLDER", str(tmp_path / "temp_smart"))


def test_tie_is_broken_by_the_film_shot_order(tmp_path, judge_env):
    a = _cand(tmp_path, "a", (1.0, 1.0), 3, (200, 0, 0))
    b = _cand(tmp_path, "b", (1.0, 1.0), 3, (0, 200, 0))
    c = _cand(tmp_path, "c", (1.0, 0.0), 3, (0, 0, 200))     # хуже по смыслу — вне ничьей
    gw = _GW('{"order": [2, 1]}')
    ps._rank_look_ties(0, "photo", [a, b, c], gw, "m")
    assert len(gw.calls) == 1 and "AS A FILM SHOT" in gw.calls[0]
    assert b["look_rank"] > a["look_rank"] and "look_rank" not in c
    base, _d = ps._score_and_pick([a, b, c])
    assert base is b, "среди равных по смыслу — лучший кадр по виду"


def test_meaning_still_wins_over_look(tmp_path, judge_env):
    a = _cand(tmp_path, "a", (1.0, 1.0), 3, (200, 0, 0))
    b = _cand(tmp_path, "b", (1.0, 0.0), 3, (0, 200, 0))
    b["look_rank"] = 99
    base, _d = ps._score_and_pick([a, b])
    assert base is a, "вид кадра не перебивает смысл"


def test_no_tie_or_flag_off_means_no_question(tmp_path, judge_env, monkeypatch):
    a = _cand(tmp_path, "a", (1.0, 1.0), 3, (200, 0, 0))
    b = _cand(tmp_path, "b", (1.0, 1.0), 2, (0, 200, 0))    # сетка развела
    gw = _GW()
    ps._rank_look_ties(0, "photo", [a, b], gw, "m")
    assert gw.calls == []
    b["judge"] = 3
    monkeypatch.setenv("LOOK_TIEBREAK", "0")
    ps._rank_look_ties(0, "photo", [a, b], gw, "m")
    assert gw.calls == []


def test_rejected_frames_never_enter_the_tie(tmp_path, judge_env):
    a = _cand(tmp_path, "a", "veto", 3, (200, 0, 0))
    b = _cand(tmp_path, "b", (1.0, 1.0), 3, (0, 200, 0))
    c = _cand(tmp_path, "c", (1.0, 1.0), 3, (0, 0, 200))
    c["verify_nothing"] = True
    gw = _GW()
    ps._rank_look_ties(0, "photo", [a, b, c], gw, "m")
    assert gw.calls == [], "один годный кадр — ничьи нет"
    # все отклонены проверкой — ничья среди брака не решается за деньги
    d = _cand(tmp_path, "d", "veto", 3, (90, 90, 0))
    ps._rank_look_ties(0, "photo", [a, d], gw, "m")
    e = dict(c, p={"id": "e"})
    ps._rank_look_ties(0, "photo", [c, e], gw, "m")
    assert gw.calls == []


def test_gateway_failure_keeps_the_old_order(tmp_path, judge_env):
    a = _cand(tmp_path, "a", (1.0, 1.0), 3, (200, 0, 0))
    b = _cand(tmp_path, "b", (1.0, 1.0), 3, (0, 200, 0))

    class Boom:
        def chat(self, *a, **k):
            raise RuntimeError("шлюз лежит")
    ps._rank_look_ties(0, "photo", [a, b], Boom(), "m")
    assert "look_rank" not in a and "look_rank" not in b
    base, _d = ps._score_and_pick([a, b])
    assert base is a


def test_look_is_judge_code_and_in_the_signature(monkeypatch):
    assert ps._rank_look_ties in ps._judge_code_entries()
    monkeypatch.setenv("SHOT_JUDGE", "1")
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", "test-key")
    monkeypatch.setenv("LOOK_TIEBREAK", "1")
    on = ps.shot_judge_signature(0)
    monkeypatch.setenv("LOOK_TIEBREAK", "0")
    assert ps.shot_judge_signature(0) != on


def test_look_is_not_part_of_the_meaning_key(tmp_path):
    """Повторный выбор «того же смысла» (резкость, скачивание) не должен
    отвергать равного по смыслу кадра из-за вида."""
    a = _cand(tmp_path, "a", (1.0, 1.0), 3, (200, 0, 0))
    b = dict(a, look_rank=5)
    assert ps._meaning_key(a) == ps._meaning_key(b)


def test_both_judge_paths_rank_the_ties():
    """С сеткой и без неё (сетка не ответила — решают утверждения) ничья
    наверху решается одинаково."""
    import inspect
    src = inspect.getsource(ps.judge_candidates)
    assert src.count("_rank_look_ties(index, kind, judged, gw, model)") == 2
    for block in src.split("_rank_look_ties(index, kind, judged, gw, model)")[:2]:
        assert "_verify_finalists(" in block, "порядок — после проверки по утверждениям"
