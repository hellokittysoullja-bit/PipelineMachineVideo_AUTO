import os
import sys
import types

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import pool_recall  # noqa: E402
import selection_freeze  # noqa: E402


def _rec():
    return {"query": "q0", "pool": [
        {"id": 1, "via": "section a"}, {"id": 2, "via": "q0"},
        {"id": 3, "via": "section b"}, {"id": 4, "via": "phrase 2"},
    ]}


def test_section_last_keeps_phrase_queries_first_in_prior_order():
    out = pool_recall.section_last(_rec(), ["q0", "phrase 2"])
    assert [r["id"] for r in out] == [2, 4, 1, 3]


def test_recall_table_best_label_within_handoff_and_pool():
    rows = [{"id": i} for i in range(5)]
    orders = {"now": rows, "section_last": rows[::-1], "cascade": rows,
              "cascade_noblock": rows + [{"id": 9}]}
    t = pool_recall.recall_table(orders, {"4": 2, "0": 0, "9": 1}, handoff=2)
    assert t["now"] == 0                 # в первых двух — только брак
    assert t["section_last"] == 2        # обратный порядок ставит точный кадр первым
    assert t["pool"] == 2
    assert t["pool_noblock"] == 2


def test_recall_table_unlabeled_is_none_not_zero():
    rows = [{"id": 1}]
    orders = {"now": rows, "section_last": rows, "cascade": rows, "cascade_noblock": rows}
    assert pool_recall.recall_table(orders, {}, 20)["pool"] is None


def test_unknown_ablation_refuses_instead_of_silent_run():
    with pytest.raises(SystemExit):
        selection_freeze.apply_ablation(types.SimpleNamespace(), "nolengthen,typo")


def test_ablations_replace_the_named_layers():
    ps = types.SimpleNamespace(disambiguate_search_query=lambda q: "european " + q,
                               content_blocklist_effective=lambda: ["reenactment"])
    assert selection_freeze.apply_ablation(ps, "nolengthen, noblocklist") == ["nolengthen", "noblocklist"]
    assert ps.disambiguate_search_query("knight") == "knight"
    assert ps.content_blocklist_effective() == []


def test_empty_ablation_changes_nothing():
    ps = types.SimpleNamespace()
    assert selection_freeze.apply_ablation(ps, "") == []
    assert vars(ps) == {}


def test_base_slot_durs_undoes_absorption_carry():
    # режим «только пул»: каждый слот поглощается следующим
    pools = {(0, "photo"): {"slot_dur": 4.0}, (1, "video"): {"slot_dur": 9.0},
             (1, "photo"): {"slot_dur": 9.0}, (2, "photo"): {"slot_dur": 12.5}}
    assert pool_recall.base_slot_durs(pools) == {0: 4.0, 1: 5.0, 2: 3.5}


def test_too_short_matches_pipeline_formula_and_stretch_constant():
    import pipeline_smart as ps
    assert pool_recall.VIDEO_MAX_TIME_STRETCH == ps.VIDEO_MAX_TIME_STRETCH
    for dur, sd in ((2, 4.0), (3, 4.0), (0, 9.0), (None, 9.0), (10, 0)):
        assert pool_recall.too_short({"duration": dur}, sd) == \
            ps._video_candidate_too_short({"duration": dur}, sd)


def test_real_pool_restores_prefilter_order_and_filters():
    rec = {"kind": "video",
           "pool": [{"id": 1, "duration": 20}, {"id": 4, "duration": 20}],
           "removed": [{"id": 2, "duration": 20, "reason": "blocklist:reenactment"},
                       {"id": 3, "duration": 1, "reason": "too_short"},
                       {"id": 5, "duration": 20, "reason": "blocked_id"}],
           "prefilter_order": [1, 2, 3, 4, 5]}
    assert [r["id"] for r in pool_recall.real_pool(rec, 5.0)] == [1, 4]
    assert [r["id"] for r in pool_recall.real_pool(rec, 5.0, blocklist=False)] == [1, 2, 4]
    # ролик длиной 1с годен для слота в 1с: «слишком короткий» — от настоящей длительности
    assert [r["id"] for r in pool_recall.real_pool(rec, 1.0)] == [1, 3, 4]


def test_pool_capture_records_what_the_filter_removed_and_why(tmp_path):
    import json

    class Adapter:
        kind = "photo"

        def filter_pool(self, request, pool):
            return [c for c in pool if "reenactment" not in c["alt"]]

        def choose(self, request, pool, cf):
            return "picked"

    ps = types.SimpleNamespace(
        PHOTO_ADAPTER=Adapter(), VIDEO_ADAPTER=None,
        candidate_probe_url=lambda c: "u", candidate_channel=lambda c: "pexels",
        pexels_candidate_text=lambda c: c["alt"],
        content_blocklist_effective=lambda: ["reenactment"],
        _candidate_block_key=lambda c: str(c["id"]), CONTENT_BLOCKED_CANDIDATE_IDS=set(),
        _video_candidate_too_short=lambda c, d: False)
    path = str(tmp_path / "pools.jsonl")
    assert selection_freeze.install_pool_capture(ps, path)
    req = types.SimpleNamespace(index=3, query="q", extra_queries=(), shot_brief=None,
                                block_text="t", slot_dur=4.0)
    pool = [{"id": 1, "alt": "knight"}, {"id": 2, "alt": "knight reenactment"}]
    kept = ps.PHOTO_ADAPTER.filter_pool(req, pool)
    assert ps.PHOTO_ADAPTER.choose(req, kept, "cf") == "picked"
    rec = json.loads(open(path).read())
    assert [r["id"] for r in rec["pool"]] == [1]
    assert rec["removed"][0]["id"] == 2 and rec["removed"][0]["reason"] == "blocklist:reenactment"
    assert rec["prefilter_order"] == [1, 2] and rec["slot_dur"] == 4.0


def test_claims_vector_background_is_penalty_main_is_veto():
    import shot_judge as sj
    spec = {"claims": [{"id": "core", "text": "a", "tier": "must"},
                       {"id": "c1", "text": "b", "tier": "should"}]}

    def ans(core="yes", c1="yes", main=True, bg=False, medium="photo"):
        return {"claims": {"core": core, "c1": c1}, "medium": medium, "main_in_world": main,
                "background_foreign": bg}
    clean = sj.claims_vector(spec, [ans()], "photo")
    spectators = sj.claims_vector(spec, [ans(bg=True)], "photo")
    no_detail = sj.claims_vector(spec, [ans(c1="no")], "photo")
    no_core = sj.claims_vector(spec, [ans(core="no")], "photo")
    assert clean > no_detail > spectators > no_core, "фон — после must, до should; главное — выше всего"
    assert sj.claims_vector(spec, [ans(main=False)], "photo") is None
    assert sj.claims_vector(spec, [ans(medium="cg")], "photo") is None
    assert sj.claims_vector(spec, [], "photo") is None


def test_verify_question_without_world_asks_no_world_items():
    import shot_judge as sj
    spec = sj.spec_from_brief("фраза", "brief")
    assert "main_in_world" not in sj.claims_question("фраза", spec)
    assert "main_in_world" in sj.claims_question("фраза", spec, setting="historical, 1300 AD")
