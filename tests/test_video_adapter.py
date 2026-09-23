#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Видео в общем ядре отбора (VideoAdapter): оценка по превью-кадрам
источника, скачивание одного победителя, те же проверки и то же
ранжирование, что у фото. Сеть, CLIP и ffmpeg подменены — здесь проверяется
оркестрация; качество самих гейтов покрыто их собственными тестами."""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = ["pipeline_smart.py", REPO]
import pipeline_smart as ps  # noqa: E402
from _media_calls import pick_video  # noqa: E402

from _video_world import DUP_HASH, FRESH_HASH, QUERY, infra, video  # noqa: E402,F401


def test_preview_frames_are_the_same_points_as_the_file_guards():
    v = video(1, frames=15)
    urls = ps.video_preview_urls(v)
    assert urls == [f"https://example.invalid/1/p{k}.jpg" for k in (2, 7, 12)]
    assert ps.VIDEO_PREVIEW_FRACS == ps.VIDEO_DOMAIN_GUARD_SAMPLE_FRACS


def test_pixabay_single_thumbnail_is_used():
    assert ps.video_preview_urls(video(2, pixabay_thumb=True)) == ["https://example.invalid/2/thumb.jpg"]


def test_irrelevant_candidate_loses_and_only_the_winner_is_downloaded(infra):
    infra["videos"] = [video(111), video(222)]
    infra["relevant"] = {222}
    assert pick_video(ps, QUERY, 0) is not None
    assert infra["downloads"] == [222], "скачан ровно победитель"
    assert set(infra["previews"]) == {111, 222}, "оценены оба — по превью"


def test_the_whole_preview_pool_is_compared_not_the_first_passing(infra):
    """Прежний видео-путь на дефолте брал первого прошедшего гейт (дефект 7).
    Теперь сравнивается весь пул: более релевантный побеждает."""
    infra["videos"] = [video(1), video(2)]
    infra["relevant"] = {1, 2}
    infra["rel"] = {1: 0.02, 2: 0.30}
    pick_video(ps, QUERY, 0)
    assert infra["downloads"] == [2]


def test_preview_pool_is_bounded(infra):
    infra["videos"] = [video(i) for i in range(1, 60)]
    pick_video(ps, QUERY, 0)
    assert len(set(infra["previews"])) == ps.VIDEO_PREVIEW_POOL


def test_nothing_relevant_still_fills_the_slot_and_says_so(infra):
    infra["videos"] = [video(333), video(444)]
    att = ps.new_attempt(2, "video")
    with ps.selection_attempt.activate(att):
        res = ps.select_media(ps.build_slot_request(**_req(2)), "video")
    assert res is not None
    kinds = {k for k, _ in att.verdicts}
    assert {"relevance", "stock"} <= kinds, "слот честно помечен как брак"


def test_near_black_frame_is_rejected(infra):
    infra["videos"] = [video(1), video(2)]
    infra["relevant"] = {1, 2}
    infra["luma"] = {1: 0.01}
    infra["rel"] = {1: 0.5, 2: 0.1}
    pick_video(ps, QUERY, 0)
    assert infra["downloads"] == [2]


def test_readable_frame_beats_slightly_better_meaning(infra):
    infra["videos"] = [video(1), video(2)]
    infra["relevant"] = {1, 2}
    infra["luma"] = {1: 0.10}                  # между HARD и PREFER — читается плохо
    infra["rel"] = {1: 0.31, 2: 0.30}
    pick_video(ps, QUERY, 0)
    assert infra["downloads"] == [2]


def test_fresh_beats_duplicate_and_duplicate_beats_empty(infra):
    infra["videos"] = [video(1), video(2)]
    infra["relevant"] = {1, 2}
    infra["hashes"] = {1: DUP_HASH, 2: FRESH_HASH}
    used = [DUP_HASH]
    pick_video(ps, QUERY, 0, used_hashes=used)
    assert infra["downloads"] == [2]
    infra["videos"] = [video(3)]
    infra["relevant"] = {3}
    infra["hashes"] = {3: DUP_HASH}
    infra["downloads"].clear()
    assert pick_video(ps, QUERY, 1, used_hashes=[DUP_HASH]) is not None


def test_failed_download_sharpness_or_veto_hand_the_slot_to_the_next(infra):
    infra["videos"] = [video(1), video(2), video(3), video(4)]
    infra["relevant"] = {1, 2, 3, 4}
    infra["rel"] = {1: 0.4, 2: 0.3, 3: 0.2, 4: 0.1}
    infra["fail_dl"] = {1}
    infra["sharp_bad"] = {2}
    infra["veto"] = {3}
    assert pick_video(ps, QUERY, 0) is not None
    assert infra["downloads"] == [1, 2, 3, 4]


def test_judge_approved_winner_is_not_vetoed(infra, monkeypatch):
    infra["videos"] = [video(1), video(2)]
    infra["relevant"] = {1, 2}
    infra["veto"] = {1}

    def fake_judge(index, kind, phrase, brief, info):
        assert kind == "video" and all("judge_path" in c for c in info)
        for c in info:
            c["judge"] = 3 if c["p"]["id"] == 1 else 1
        return True
    monkeypatch.setattr(ps, "judge_candidates", fake_judge)
    pick_video(ps, QUERY, 0)
    assert infra["downloads"] == [1], "одобренный судьёй кадр не отменяется второй проверкой"


def test_judge_gets_a_strip_of_the_preview_frames(infra, monkeypatch):
    infra["videos"] = [video(1)]
    seen = {}

    def fake_judge(index, kind, phrase, brief, info):
        from PIL import Image as Im
        seen["size"] = Im.open(info[0]["judge_path"]).size
        return False
    monkeypatch.setattr(ps, "judge_candidates", fake_judge)
    pick_video(ps, QUERY, 0)
    w, h = seen["size"]
    assert w > 2.5 * h, "три кадра одного ролика лентой"


def test_cache_hit_rejects_a_visual_duplicate(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "TEMP_FOLDER", str(tmp_path))
    req = ps.build_slot_request(**_req(0, used_hashes=[DUP_HASH]))
    cf = ps.VIDEO_ADAPTER.cache_path(req)
    open(cf, "wb").write(b"v")
    ps.write_media_sidecar(cf, pexels_id=5, query=QUERY, kind="video", ahash_hex=DUP_HASH)
    with ps.selection_attempt.activate(ps.new_attempt(0, "video")):
        assert ps.VIDEO_ADAPTER.cache_hit(req, cf) is None
    req2 = ps.build_slot_request(**_req(0, used_hashes=["1" * 64]))
    att = ps.new_attempt(0, "video")
    with ps.selection_attempt.activate(att):
        assert ps.VIDEO_ADAPTER.cache_hit(req2, cf) == cf
    assert any(kind == "reserve_hash" and args[-1] == DUP_HASH for kind, args in att.effects), \
        "принятый кэш-хит возвращает свой отпечаток в анти-дубль"


def test_sources_are_interleaved_like_photo(infra, monkeypatch):
    infra["videos"] = [video(1), video(2)]
    monkeypatch.setattr(ps, "_pixabay_search_videos",
                        lambda q: [video(3, pixabay_thumb=True), video(4, pixabay_thumb=True)])
    infra["videos"] += []
    order = []
    orig = ps.VIDEO_ADAPTER.choose

    def spy(request, pool, cf):
        order.extend(v["id"] for v in pool)
        return None
    monkeypatch.setattr(ps.VIDEO_ADAPTER, "choose", spy)
    pick_video(ps, QUERY, 0)
    assert order == [1, 3, 2, 4]
    monkeypatch.setattr(ps.VIDEO_ADAPTER, "choose", orig)


def _req(index, used_hashes=None):
    import dataclasses
    import selection_engine
    f = {x.name: None for x in dataclasses.fields(selection_engine.SlotRequest)}
    f.update(index=index, query=QUERY, extra_queries=(), is_opening=False, director_assist=False,
             used_hashes=used_hashes)
    return f


def test_preview_pool_is_not_cut_after_the_hook():
    """Урезание пула после хука было ценой СКАЧИВАНИЯ роликов; превью —
    доли секунды на кандидата, поэтому видео смотрит полный пул везде."""
    assert ps.VIDEO_PREVIEW_POOL == ps.BASE_MIN_POOL
    assert "VIDEO_PREVIEW_POOL" in __import__("inspect").getsource(ps.VideoAdapter.choose)
    assert "_FAST_MODE_START" not in __import__("inspect").getsource(ps.VideoAdapter)


def test_all_section_queries_feed_the_video_pool(infra, monkeypatch):
    seen = []

    def search(q):
        seen.append(q)
        return [video(1)] if "sword" in q else [video(2)]
    monkeypatch.setattr(ps, "_pexels_search_videos", search)
    infra["videos"] = [video(1), video(2)]
    infra["relevant"] = {2}
    pick_video(ps, QUERY, 0, extra_queries=["dark cinema movie theatre screen"])
    assert any("cinema" in q for q in seen) and infra["downloads"] == [2]


def test_text_key_changes_the_cache_path(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "TEMP_FOLDER", str(tmp_path))
    a = ps.VIDEO_ADAPTER.cache_path(ps.build_slot_request(**dict(_req(0), text_key="Не дрались.")))
    b = ps.VIDEO_ADAPTER.cache_path(ps.build_slot_request(**dict(_req(0), text_key="Первое: размер.")))
    assert a != b


def test_director_score_ranks_video_like_photo(infra):
    infra["videos"] = [video(1), video(2)]
    infra["relevant"] = {1, 2}
    calls = []

    def director(path, candidate_query=None, aesthetic_val=None):
        calls.append(path)
        return 0.9 if "_2_" in os.path.basename(path) or "prev_2_" in path else 0.1
    pick_video(ps, QUERY, 0, sentence_score_fn=director, director_assist=True)
    assert calls, "скоринг Режиссёра зовётся для видео тем же _score_and_pick"
    assert infra["downloads"] == [2]


def test_shot_size_rhythm_applies_to_video(infra, monkeypatch):
    infra["videos"] = [video(1), video(2)]
    infra["relevant"] = {1, 2}
    infra["rel"] = {1: 0.30, 2: 0.30}
    monkeypatch.setattr(ps, "estimate_shot_size",
                        lambda p: "wide" if "prev_1_" in p else "detail")
    pick_video(ps, QUERY, 0, recent_sizes=["wide", "wide"])
    assert infra["downloads"] == [2], "повтор крупности уступает другой при равном смысле"


def test_slot_duration_filters_too_short_clips(infra):
    infra["videos"] = [video(1, duration=2), video(2, duration=30)]
    infra["relevant"] = {1, 2}
    infra["rel"] = {1: 0.5, 2: 0.1}
    pick_video(ps, QUERY, 0, slot_dur=10.0)
    assert infra["downloads"] == [2]


def test_arbiter_refusal_is_recorded_for_video(infra, monkeypatch):
    import shot_director
    # Шорт-лист арбитра — победитель и лучший по СВОЕМУ запросу слота: чтобы
    # их было двое, кандидаты приходят из разных запросов пула.
    infra["videos"] = [video(1), video(2)]
    infra["relevant"] = {1, 2}
    infra["rel"] = {1: 0.1, 2: 0.4}
    monkeypatch.setattr(ps, "_pexels_search_videos",
                        lambda q: [video(1)] if "sword" in q else [video(2)])
    monkeypatch.setattr(ps.feature_flags, "mode", lambda n: "on" if n == "VLM_ARBITER_MODE" else "off")
    monkeypatch.setattr(shot_director, "arbitrate_hook_candidates",
                        lambda *a, **k: shot_director.NO_CANDIDATE_FITS)
    att = ps.new_attempt(0, "video")
    with ps.selection_attempt.activate(att):
        out = ps.select_media(ps.build_slot_request(**dict(
            _req(0), arbiter_text="Вот кинжал.", extra_queries=("mounted knight field",))), "video")
    assert out is not None and any(k == "arbiter" for k, _ in att.verdicts)


def test_pixabay_is_asked_from_the_video_pool(infra, monkeypatch):
    asked = []
    monkeypatch.setattr(ps, "_pixabay_search_videos", lambda q: asked.append(q) or [])
    infra["videos"] = [video(1)]
    pick_video(ps, QUERY, 0)
    assert asked
