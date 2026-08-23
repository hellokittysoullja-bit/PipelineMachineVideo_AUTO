"""Regression-тесты _score_and_pick() (scripts/pipeline_smart.py) — чистая
функция ранжирования/выбора победителя, извлечённая из pexels_photo() при
добавлении Semantic Visual Director (scripts/visual_director.py). До этого
рефакторинга pexels_photo() была единственной core-функцией отбора медиа
вообще без тестового покрытия — этот файл в первую очередь доказывает, что
извлечение НИЧЕГО не сломало (base_winner на синтетическом пуле совпадает с
тем же лексикографическим кортежем, что был инлайн раньше), и только потом
проверяет новую director-ветку."""
import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]
import pipeline_smart as ps   # noqa: E402


def _cand(path, is_dup_free=1, size_ok=1, is_relevant=1, aesthetic_val=0.0,
          luma_score=0.0, min_d=99, pid=None):
    return {"path": path, "p": {"id": pid or path}, "is_dup_free": is_dup_free,
            "size_ok": size_ok, "is_relevant": is_relevant,
            "aesthetic_val": aesthetic_val, "luma_score": luma_score, "min_d": min_d}


def test_empty_candidates_returns_none_none():
    base, director = ps._score_and_pick([])
    assert base is None
    assert director is None


def test_single_candidate_wins_trivially():
    c = _cand("a")
    base, director = ps._score_and_pick([c])
    assert base is c
    assert director is None   # director_score_fn не передан


def test_dup_free_beats_everything_else():
    dup = _cand("dup", is_dup_free=0, aesthetic_val=10.0)
    clean = _cand("clean", is_dup_free=1, aesthetic_val=-10.0)
    base, _ = ps._score_and_pick([dup, clean])
    assert base is clean


def test_size_ok_beats_aesthetic():
    bad_size = _cand("bad_size", size_ok=0, aesthetic_val=10.0)
    good_size = _cand("good_size", size_ok=1, aesthetic_val=-10.0)
    base, _ = ps._score_and_pick([bad_size, good_size])
    assert base is good_size


def test_relevance_beats_aesthetic():
    irrelevant = _cand("irrelevant", is_relevant=0, aesthetic_val=10.0)
    relevant = _cand("relevant", is_relevant=1, aesthetic_val=-10.0)
    base, _ = ps._score_and_pick([irrelevant, relevant])
    assert base is relevant


def test_aesthetic_beats_luma():
    pretty = _cand("pretty", aesthetic_val=5.0, luma_score=-10.0)
    dull = _cand("dull", aesthetic_val=0.0, luma_score=0.0)
    base, _ = ps._score_and_pick([pretty, dull])
    assert base is pretty


def test_ties_keep_first_candidate_not_last():
    first = _cand("first")
    second = _cand("second")
    base, _ = ps._score_and_pick([first, second])
    assert base is first   # строгое ">" в сравнении — второй равный не подвинул первого


def test_min_d_is_the_final_tiebreaker():
    near = _cand("near", min_d=5)
    far = _cand("far", min_d=99)
    base, _ = ps._score_and_pick([near, far])
    assert base is far   # больше дистанция до ближайшего used_hash — лучше (меньше похоже на уже выбранное)


def test_director_score_fn_none_leaves_director_winner_none():
    base, director = ps._score_and_pick([_cand("a"), _cand("b")], director_score_fn=None)
    assert director is None
    assert base is not None


def test_director_score_fn_can_diverge_from_base_winner():
    # base предпочтёт "aesthetic_high" по эстетике; extra выбранная для
    # "director_pick" достаточно велика, чтобы перевесить более низкую
    # эстетику — extra стоит СРАЗУ после relevance-гейта, до aesthetic.
    aesthetic_high = _cand("aesthetic_high", aesthetic_val=5.0)
    director_pick = _cand("director_pick", aesthetic_val=0.0)
    scores = {"aesthetic_high": 0.0, "director_pick": 10.0}
    base, director = ps._score_and_pick(
        [aesthetic_high, director_pick], director_score_fn=lambda path: scores[path])
    assert base is aesthetic_high
    assert director is director_pick


def test_director_respects_relevance_gate_even_with_high_extra():
    irrelevant_but_favored = _cand("irrelevant_but_favored", is_relevant=0)
    relevant = _cand("relevant", is_relevant=1)
    scores = {"irrelevant_but_favored": 100.0, "relevant": 0.0}
    base, director = ps._score_and_pick(
        [irrelevant_but_favored, relevant], director_score_fn=lambda path: scores[path])
    assert director is relevant   # is_relevant остаётся гейтом впереди extra, extra не может его перебить
