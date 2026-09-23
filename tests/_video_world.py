# -*- coding: utf-8 -*-
"""Подставной мир видео-отбора для тестов: выдача, превью (цветные JPEG),
ролики, гейты. Общий для всех тестов, которые гоняют VideoAdapter, чтобы
каждый файл не собирал свой набор заглушек (раньше тесты видео подменяли
функции добытчика, которых больше нет, и падали не о поведение, а о имена).

Подключение в тесте: `from _video_world import video, infra, QUERY  # noqa`.
"""
import os

import pytest
from PIL import Image

import pipeline_smart as ps

QUERY = "european medieval sword close up"
DUP_HASH, FRESH_HASH = "0" * 64, "1" * 64


def video(vid, *, frames=15, duration=20, pixabay_thumb=False):
    v = {"id": vid, "duration": duration,
         "video_files": [{"file_type": "video/mp4", "width": 1920,
                          "link": f"https://example.invalid/{vid}.mp4"}]}
    if pixabay_thumb:
        v["_preview_frames"] = [f"https://example.invalid/{vid}/thumb.jpg"]
    else:
        v["video_pictures"] = [{"nr": k, "picture": f"https://example.invalid/{vid}/p{k}.jpg"}
                               for k in range(frames)]
    return v


@pytest.fixture
def infra(monkeypatch, tmp_path):
    """Подставной мир: выдача, превью (цветные JPEG), ролики, гейты.
    Возвращает словарь со списком скачанного и настройками кандидатов."""
    st = {"videos": [], "relevant": set(), "hashes": {}, "luma": {}, "rel": {},
          "downloads": [], "previews": [], "sharp_bad": set(), "veto": set(), "fail_dl": set()}
    monkeypatch.setattr(ps, "TEMP_FOLDER", str(tmp_path))
    monkeypatch.setattr(ps, "PEXELS_API_KEY", "fake")
    monkeypatch.setattr(ps, "_pexels_search_videos", lambda q: list(st["videos"]))
    monkeypatch.setattr(ps, "_pixabay_search_videos", lambda q: [])

    def vid_of(path):
        name = os.path.basename(path)
        for v in st["videos"]:
            tok = ps.candidate_path_token(v)
            if f"prev_{tok}_" in name or f"strip_{tok}" in name:
                return v["id"]
        return None

    def fake_download(req, dest, timeout):
        url = req.full_url
        seg = url.split("example.invalid/", 1)[1].split("/", 1)[0].split(".", 1)[0]
        vid = next((v["id"] for v in st["videos"] if str(v["id"]) == seg), None)
        if url.endswith(".mp4"):
            st["downloads"].append(vid)
            if vid in st["fail_dl"]:
                raise OSError("обрыв")
            with open(dest, "wb") as f:
                f.write(b"video")
        else:
            st["previews"].append(vid)
            Image.new("RGB", (64, 36), (int(vid) % 250, 40, 90)).save(dest, "JPEG")
    monkeypatch.setattr(ps, "atomic_url_download", fake_download)
    monkeypatch.setattr(ps, "clip_relevance", lambda p, q: st["rel"].get(vid_of(p), 0.1))
    monkeypatch.setattr(ps, "is_relevant_candidate",
                        lambda p, q, relevance=None: vid_of(p) in st["relevant"])
    monkeypatch.setattr(ps, "visual_domain_guard_violation", lambda p, q: (False, None))
    monkeypatch.setattr(ps, "negative_anchor_violation", lambda p, q: (False, None))
    monkeypatch.setattr(ps, "ahash", lambda p: st["hashes"].get(vid_of(p), FRESH_HASH))
    monkeypatch.setattr(ps, "measure_luma", lambda p, is_video=False: st["luma"].get(vid_of(p), 0.5))
    monkeypatch.setattr(ps, "aesthetic_score", lambda p: 5.0)
    monkeypatch.setattr(ps, "estimate_shot_size", lambda p: "medium")
    monkeypatch.setattr(ps, "video_sharpness_ok",
                        lambda path: False if st["downloads"][-1] in st["sharp_bad"] else True)
    monkeypatch.setattr(ps, "video_smart_relevance_veto",
                        lambda path, q: st["downloads"][-1] in st["veto"])
    return st
