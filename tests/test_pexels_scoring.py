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


def _cand(path, is_dup_free=1, size_ok=1, is_relevant=1, sharp_ok=1, aesthetic_val=0.0,
          luma_score=0.0, min_d=99, pid=None, origin_query=None):
    p = {"id": pid or path}
    if origin_query is not None:
        p["_origin_query"] = origin_query
    return {"path": path, "p": p, "is_dup_free": is_dup_free,
            "size_ok": size_ok, "is_relevant": is_relevant, "sharp_ok": sharp_ok,
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
        [aesthetic_high, director_pick], director_score_fn=lambda path, candidate_query=None: scores[path])
    assert base is aesthetic_high
    assert director is director_pick


def test_director_respects_relevance_gate_even_with_high_extra():
    irrelevant_but_favored = _cand("irrelevant_but_favored", is_relevant=0)
    relevant = _cand("relevant", is_relevant=1)
    scores = {"irrelevant_but_favored": 100.0, "relevant": 0.0}
    base, director = ps._score_and_pick(
        [irrelevant_but_favored, relevant], director_score_fn=lambda path, candidate_query=None: scores[path])
    assert director is relevant   # is_relevant остаётся гейтом впереди extra, extra не может его перебить


# ---------- sharp_ok: гейт резкости (см. PHOTO_SHARPNESS_REJECT) ----------
# Реальный найденный случай (27 августа, videos/_test20s, слот 7 — размытое
# видео всадника) — ни один кандидат никогда не проверялся на резкость на
# этапе отбора. sharp_ok — тот же класс гейта, что is_relevant: важнее
# эстетики, но не важнее dup/size/relevance.

def test_sharp_beats_aesthetic():
    blurry = _cand("blurry", sharp_ok=0, aesthetic_val=10.0)
    sharp = _cand("sharp", sharp_ok=1, aesthetic_val=-10.0)
    base, _ = ps._score_and_pick([blurry, sharp])
    assert base is sharp


def test_relevance_beats_sharp():
    irrelevant_sharp = _cand("irrelevant_sharp", is_relevant=0, sharp_ok=1)
    relevant_blurry = _cand("relevant_blurry", is_relevant=1, sharp_ok=0)
    base, _ = ps._score_and_pick([irrelevant_sharp, relevant_blurry])
    assert base is relevant_blurry


def test_sharp_ok_defaults_to_one_when_key_missing():
    # Обратная совместимость: candidates_info без ключа "sharp_ok" вообще
    # (старый вызывающий код/тест) — поведение как раньше, никого не режет.
    no_key = {"path": "no_key", "p": {"id": "no_key"}, "is_dup_free": 1, "size_ok": 1,
              "is_relevant": 1, "aesthetic_val": 5.0, "luma_score": 0.0, "min_d": 99}
    with_key = _cand("with_key", sharp_ok=1, aesthetic_val=0.0)
    base, _ = ps._score_and_pick([no_key, with_key])
    assert base is no_key   # выше эстетика побеждает — sharp_ok=1 по умолчанию для обоих


def test_director_sharp_gate_matches_base():
    blurry = _cand("blurry", sharp_ok=0, aesthetic_val=10.0)
    sharp = _cand("sharp", sharp_ok=1, aesthetic_val=-10.0)
    scores = {"blurry": 0.0, "sharp": 0.0}
    base, director = ps._score_and_pick(
        [blurry, sharp], director_score_fn=lambda path, candidate_query=None: scores[path])
    assert director is sharp


# ---------- _pool_cleared_both_gates (STOCK_EXHAUSTED_MISSES ось) ----------

def test_pool_cleared_true_when_one_candidate_passes_both_gates():
    pool = [_cand("a", is_relevant=0, sharp_ok=1), _cand("b", is_relevant=1, sharp_ok=1)]
    assert ps._pool_cleared_both_gates(pool) is True


def test_pool_cleared_false_when_nobody_passes_both():
    pool = [_cand("a", is_relevant=0, sharp_ok=1), _cand("b", is_relevant=1, sharp_ok=0)]
    assert ps._pool_cleared_both_gates(pool) is False


def test_pool_cleared_true_even_if_winner_lost_to_dedup():
    # Реальный случай, который RELEVANCE_GATE_MISSES (победитель-онли) не
    # ловит: "победитель" в _score_and_pick() мог оказаться нерелевантным
    # уникальным кандидатом, хотя релевантный+резкий кандидат в пуле БЫЛ,
    # просто проиграл по is_dup_free (первому элементу кортежа). Пул при
    # этом не исчерпан — сток есть, просто уже использован где-то ещё.
    pool = [
        _cand("dup_but_good", is_dup_free=0, is_relevant=1, sharp_ok=1),
        _cand("unique_but_bad", is_dup_free=1, is_relevant=0, sharp_ok=1),
    ]
    base_winner, _ = ps._score_and_pick(pool)
    assert base_winner["path"] == "unique_but_bad"   # победитель — не relevant
    assert ps._pool_cleared_both_gates(pool) is True   # но пул не исчерпан


def test_pool_cleared_false_on_empty_pool():
    assert ps._pool_cleared_both_gates([]) is False


# ---------- _build_arbiter_shortlist (VLM-арбитр, HOOK-only) ----------

def test_shortlist_includes_both_base_and_director_when_they_differ():
    base = _cand("base_pick")
    director = _cand("director_pick")
    shortlist = ps._build_arbiter_shortlist([base, director], base, director, own_query="q")
    paths = {c["path"] for c in shortlist}
    assert paths == {"base_pick", "director_pick"}


def test_shortlist_dedupes_when_base_equals_director():
    same = _cand("same_pick")
    shortlist = ps._build_arbiter_shortlist([same], same, same, own_query="q")
    assert len(shortlist) == 1
    assert shortlist[0]["path"] == "same_pick"


def test_shortlist_adds_best_same_query_candidate_not_already_present():
    # Реальный найденный случай: победитель (base/director) пришёл из
    # ЧУЖОГО запроса, а лучший кандидат СВОЕГО запроса ("own_query")
    # проиграл — арбитр должен увидеть его тоже, не только победителей.
    base = _cand("foreign_pick", origin_query="other query")
    director = _cand("foreign_pick", origin_query="other query")
    own_good = _cand("own_query_pick", origin_query="scale", aesthetic_val=1.0)
    pool = [base, own_good]
    shortlist = ps._build_arbiter_shortlist(pool, base, director, own_query="scale")
    paths = {c["path"] for c in shortlist}
    assert paths == {"foreign_pick", "own_query_pick"}


def test_shortlist_skips_same_query_candidate_that_fails_gates():
    base = _cand("winner")
    own_bad = _cand("own_bad", origin_query="scale", is_relevant=0)   # провалил гейт
    pool = [base, own_bad]
    shortlist = ps._build_arbiter_shortlist(pool, base, base, own_query="scale")
    assert len(shortlist) == 1
    assert shortlist[0]["path"] == "winner"


def test_shortlist_respects_max_n_cap():
    base = _cand("base_pick")
    director = _cand("director_pick")
    own_good = _cand("own_query_pick", origin_query="scale", aesthetic_val=1.0)
    pool = [base, director, own_good]
    shortlist = ps._build_arbiter_shortlist(pool, base, director, own_query="scale", max_n=2)
    assert len(shortlist) == 2


def test_shortlist_picks_best_of_multiple_same_query_candidates_by_aesthetic():
    base = _cand("winner")
    own_ok = _cand("own_ok", origin_query="scale", aesthetic_val=0.5)
    own_better = _cand("own_better", origin_query="scale", aesthetic_val=2.0)
    pool = [base, own_ok, own_better]
    shortlist = ps._build_arbiter_shortlist(pool, base, base, own_query="scale")
    paths = {c["path"] for c in shortlist}
    assert "own_better" in paths
    assert "own_ok" not in paths


# ---------- _build_video_arbiter_shortlist (VLM-арбитр, видео-путь) ----------
# good — (sent_score, luma_ok, trial_path, id, hash, origin_query), уже
# отсортирован по (luma_ok, sent_score) убыванием (см. pexels_video()).

def _vcand(path, origin_query=None, score=0.0, luma_ok=1, vid=None):
    return (score, luma_ok, path, vid or path, None, origin_query)


def test_video_shortlist_includes_winner_and_own_query_match():
    winner = _vcand("winner", origin_query="other query", score=1.0)
    own_match = _vcand("own_match", origin_query="scale", score=0.5)
    good = [winner, own_match]
    shortlist = ps._build_video_arbiter_shortlist(good, own_query="scale")
    paths = {g[2] for g in shortlist}
    assert paths == {"winner", "own_match"}


def test_video_shortlist_falls_back_to_runner_up_without_own_query_match():
    winner = _vcand("winner", origin_query="other", score=1.0)
    runner_up = _vcand("runner_up", origin_query="another", score=0.5)
    good = [winner, runner_up]
    shortlist = ps._build_video_arbiter_shortlist(good, own_query="scale")
    paths = {g[2] for g in shortlist}
    assert paths == {"winner", "runner_up"}


def test_video_shortlist_dedupes_when_own_query_match_is_the_winner():
    winner = _vcand("winner", origin_query="scale", score=1.0)
    good = [winner]
    shortlist = ps._build_video_arbiter_shortlist(good, own_query="scale")
    assert len(shortlist) == 1
    assert shortlist[0][2] == "winner"


def test_video_shortlist_respects_max_n_cap():
    winner = _vcand("winner", origin_query="other", score=1.0)
    own_match = _vcand("own_match", origin_query="scale", score=0.7)
    runner_up = _vcand("runner_up", origin_query="another", score=0.5)
    good = [winner, own_match, runner_up]
    shortlist = ps._build_video_arbiter_shortlist(good, own_query="scale", max_n=2)
    assert len(shortlist) == 2


# ---------- _build_opening_shortlist (открывающий кадр — критерий
# эффектности, не буквальной точности запроса; см. /goal 29 августа) ----------

def test_opening_shortlist_pulls_best_aesthetic_from_whole_pool_not_own_query():
    # Реальный найденный случай: победитель (base/director) — технически
    # точная, но невыразительная картинка своего запроса; эффектный
    # кросс-опылённый кандидат ЧУЖОГО запроса (высокая эстетика) должен
    # попасть в шорт-лист, а не быть молча отброшен.
    winner = _cand("boring_winner", origin_query="scale", aesthetic_val=0.1)
    dramatic = _cand("dramatic_cross_pollinated", origin_query="warrior on horseback", aesthetic_val=5.0)
    pool = [winner, dramatic]
    shortlist = ps._build_opening_shortlist(pool, winner, winner)
    paths = {c["path"] for c in shortlist}
    assert "dramatic_cross_pollinated" in paths


def test_opening_shortlist_always_includes_base_and_director_winners():
    base = _cand("base_pick", aesthetic_val=0.0)
    director = _cand("director_pick", aesthetic_val=0.0)
    pool = [base, director]
    shortlist = ps._build_opening_shortlist(pool, base, director)
    paths = {c["path"] for c in shortlist}
    assert {"base_pick", "director_pick"} <= paths


def test_opening_shortlist_skips_candidates_failing_gates():
    winner = _cand("winner", aesthetic_val=0.0)
    bad = _cand("gorgeous_but_irrelevant", aesthetic_val=10.0, is_relevant=0)
    pool = [winner, bad]
    shortlist = ps._build_opening_shortlist(pool, winner, winner)
    paths = {c["path"] for c in shortlist}
    assert "gorgeous_but_irrelevant" not in paths


def test_opening_shortlist_respects_max_n_cap():
    base = _cand("base", aesthetic_val=0.0)
    director = _cand("director", aesthetic_val=0.0)
    c1 = _cand("c1", aesthetic_val=3.0)
    c2 = _cand("c2", aesthetic_val=2.0)
    pool = [base, director, c1, c2]
    shortlist = ps._build_opening_shortlist(pool, base, director, max_n=3)
    assert len(shortlist) == 3


# ---------- _build_opening_video_shortlist ----------

def test_opening_video_shortlist_includes_cross_pollinated_candidates():
    winner = _vcand("winner", origin_query="scale", score=1.0)
    cross = _vcand("cross_pollinated", origin_query="warrior on horseback", score=0.8)
    good = [winner, cross]
    shortlist = ps._build_opening_video_shortlist(good)
    paths = {g[2] for g in shortlist}
    assert paths == {"winner", "cross_pollinated"}


def test_opening_video_shortlist_respects_max_n_cap():
    good = [_vcand(f"c{i}", score=1.0 - i * 0.1) for i in range(6)]
    shortlist = ps._build_opening_video_shortlist(good, max_n=4)
    assert len(shortlist) == 4
