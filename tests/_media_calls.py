# -*- coding: utf-8 -*-
"""Вызов отбора из тестов прежними именованными аргументами.

В продакшне отбор принимает только SlotRequest без значений по умолчанию
(pipeline_smart.select_media). Тестам удобнее задавать лишь то, что они
проверяют, — эта обёртка живёт ТОЛЬКО в тестах и переводит прежние имена
аргументов pexels_photo/pexels_video в поля запроса: used_ids ->
used_photo_ids или used_video_ids (по виду медиа), is_opening_shot ->
is_opening, sentence_score_fn -> video_score_fn. Незаданное — None/().
"""


def _request(ps, query, index, kind, kw):
    used_ids = kw.pop("used_ids", None)
    fields = dict(
        index=index, query=query,
        extra_queries=kw.pop("extra_queries", None),
        text_key=kw.pop("text_key", None),
        shot_brief=kw.pop("shot_brief", None),
        block_text=kw.pop("block_text", None),
        arbiter_text=kw.pop("arbiter_text", None),
        is_opening=kw.pop("is_opening_shot", False),
        slot_dur=kw.pop("slot_dur", None),
        action_qualifier=kw.pop("action_qualifier", None),
        target_luma=kw.pop("target_luma", None),
        director_score_fn=kw.pop("director_score_fn", None),
        director_assist=kw.pop("director_assist", False),
        director_report=kw.pop("director_report", None),
        video_score_fn=kw.pop("sentence_score_fn", None),
        used_photo_ids=used_ids if kind == "photo" else None,
        used_video_ids=used_ids if kind == "video" else None,
        used_hashes=kw.pop("used_hashes", None),
        recent_sizes=kw.pop("recent_sizes", None),
    )
    if kw:
        raise TypeError(f"неизвестные аргументы: {sorted(kw)}")
    return ps.build_slot_request(**fields)


def pick_photo(ps, query, index, **kw):
    return ps.select_media(_request(ps, query, index, "photo", kw), "photo")


def pick_video(ps, query, index, **kw):
    return ps.select_media(_request(ps, query, index, "video", kw), "video")
