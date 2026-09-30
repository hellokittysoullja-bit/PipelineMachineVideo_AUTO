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
    clean = sj.claims_vector(spec, ans())
    no_detail = sj.claims_vector(spec, ans(c1="no"))
    no_core = sj.claims_vector(spec, ans(core="no"))
    # Исторический мир: чужое на фоне — отказ (зрители, бетон, куртка).
    assert sj.claims_vector(spec, ans(bg=True)) is None
    # Под предохранителем мира и вне исторического мира — штраф: после
    # must, до should.
    for kw in ({"world_veto": False}, {"cg_veto": False}):
        spectators = sj.claims_vector(spec, ans(bg=True), **kw)
        assert sj.claims_vector(spec, ans(), **kw) > sj.claims_vector(spec, ans(c1="no"), **kw) \
            > spectators > sj.claims_vector(spec, ans(core="no"), **kw), kw
    assert clean > no_detail > no_core, "главное — выше"
    assert sj.claims_vector(spec, ans(main=False)) is None
    assert sj.claims_vector(spec, ans(medium="cg")) is None
    assert sj.claims_vector(spec, ans(medium="cg"), cg_veto=False) == clean, \
        "3D — брак только в историческом эпизоде"
    assert sj.claims_vector(spec, None) is None


def test_verify_question_without_world_asks_no_world_items():
    import shot_judge as sj
    spec = sj.spec_from_brief("фраза", "brief")
    assert "main_in_world" not in sj.claims_question("фраза", spec)
    assert "main_in_world" in sj.claims_question("фраза", spec, setting="historical, 1300 AD")


def test_embedder_finds_pixabay_vector_by_frame_not_by_expiring_url(tmp_path, monkeypatch):
    """Подписанный адрес превью Pixabay протухает (400) и меняется от выдачи
    к выдаче; прод-каскад кэширует вектор по номеру кадра. Замер обязан
    искать так же — иначе живой кадр уходит в хвост порядка только потому,
    что его адрес устарел, и замер расходится с продом. Мёртвый адрес
    запрашивается один раз, а не на каждом порядке каждого слота."""
    import numpy as np
    import pipeline_smart as ps
    emb = pool_recall.Embedder.__new__(pool_recall.Embedder)
    emb.ps, emb.cache_dirs, emb.write_dir, emb.dead = ps, [], str(tmp_path), set()
    url = "https://pixabay.com/get/g0123abcd_640.jpg"
    cand = pool_recall._cand({"id": 777, "probe_url": url}, "photo")
    key = ps._cascade_key(ps._cascade_ident(cand, url))
    np.save(os.path.join(str(tmp_path), key + ".npy"), np.ones(3, dtype=np.float32))
    calls = []
    monkeypatch.setattr(pool_recall, "fetch", lambda u, h, cand_id=None: calls.append(u))
    v = emb.image_vec(url, None, cand)
    assert v is not None and float(v.sum()) == 3.0
    assert calls == []
    dead = "https://pixabay.com/get/gdead_640.jpg"
    emb.image_vec(dead, None, pool_recall._cand({"id": 1, "probe_url": dead}, "photo"))
    emb.image_vec(dead, None, pool_recall._cand({"id": 1, "probe_url": dead}, "photo"))
    assert calls == [dead]


def test_bench_refuses_episode_without_plan_specs(tmp_path, monkeypatch):
    """26.09: «честная база» бенча шла на плане не той версии, load_specs
    вернул {}, и каждый слот молча оценивался по брифу одним утверждением —
    сравнение планов без плана. Теперь отказ до первого платного вызова."""
    import json
    import llm_gateway
    ep = tmp_path / "ep"
    (ep / "media_plan").mkdir(parents=True)
    (ep / "media_plan" / "stock_queries.json").write_text(
        json.dumps({"version": 1, "units": {}}), encoding="utf-8")
    for n in ("index.json", "labels.json"):
        (tmp_path / n).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(pool_recall, "load_pools", lambda d: {})
    monkeypatch.setattr(pool_recall, "load_phrase_queries", lambda p: {})
    monkeypatch.setattr(pool_recall, "base_slot_durs", lambda p: {})
    monkeypatch.setattr(pool_recall, "_label_rows",
                        lambda *a, **k: pytest.fail("бенч пошёл оценивать без спецификаций"))
    monkeypatch.setattr(llm_gateway, "Gateway", lambda **k: object())
    a = types.SimpleNamespace(
        run_dir=str(tmp_path), phrase_queries=None, index=str(tmp_path / "index.json"),
        labels=str(tmp_path / "labels.json"), emb_cache=[], world_card=None,
        claims_world="exclude", max_spend=1, episode=str(ep), model="m", cache_dir=None)
    with pytest.raises(SystemExit, match="спецификации кадров не загружены"):
        pool_recall.cmd_bench(a)


def test_screen_count_brak_winner_empties_slot_and_counts_missed_good():
    """Победитель-брак опустошает слот; если годный был в первых кадрах —
    это отдельный счёт (так 24.09 по промежуточным счётчикам был снят
    полезный вопрос: пустые слоты при годной замене они не видели)."""
    st = {}
    head = [({"id": 1}, 1, (-5, 2), None, {}), ({"id": 2}, 0, (-5, 1), None, {})]
    pool_recall._screen_count(st, head, 0)
    assert st["screen_empty"] == 1 and st["screen_empty_good"] == 1
    assert st.get("screen_brak", 0) == 0
    st = {}
    head = [({"id": 1}, 0, (1.0, 2), None, {}), ({"id": 2}, 2, (0.0, 3), None, {})]
    pool_recall._screen_count(st, head, 0)
    assert st["screen_brak"] == 1 and st.get("screen_best", 0) == 0
    st = {}
    pool_recall._screen_count(st, [({"id": 3}, 2, (1.0, 3), None, {})], 0)
    assert st["screen_best"] == 1 and st["screen_label_sum"] == 2


def test_bench_finds_spec_of_a_cut_phrase_like_the_render(tmp_path):
    """Рендер привязывает задание к фразе ДО нарезки длинных фраз, кусочки
    его наследуют. Бенч видит только кусочек — и на эп.93 искал задание по
    нему, так что 9 слотов из 14 молча оценивались по брифу."""
    import json
    import shot_planner_llm
    whole = "Когда ей грозит гибель, она делает то, что не умеет никто."
    other = "Глубже двухсот метров темно."
    (tmp_path / "media_plan").mkdir()
    (tmp_path / "media_plan" / "stock_queries.json").write_text(json.dumps(
        {"units": {"a": {"text": whole}, "b": {"text": other}}}), encoding="utf-8")
    spec = {"focus": "x", "claims": []}
    specs = {shot_planner_llm.unit_key(whole): spec}
    texts = pool_recall._plan_unit_texts(str(tmp_path))
    assert pool_recall._spec_for_block(specs, texts, "что не умеет никто.") is spec
    assert pool_recall._spec_for_block(specs, texts, whole) is spec
    assert pool_recall._spec_for_block(specs, texts, "совсем другой текст") is None
    assert pool_recall._spec_for_block(specs, texts, "  ") is None


def test_bench_refuses_to_report_when_gateway_died():
    """Шлюз выключился посреди бенча — числа по оценённой части не
    печатаются как замер (26.09: v6 «оценён» по 1 слоту из 9)."""
    import inspect
    src = inspect.getsource(pool_recall.cmd_bench)
    i, j = src.index('getattr(gw, "dead"'), src.index('print(f"пары верно')
    assert i < j and "НЕДЕЙСТВИТЕЛЕН" in src[i:j]
