#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Зеркало карточек Мет: кладёт карточки в обычный кэш карточек, продолжает
с места обрыва, при обновлении продлевает неизменённые без запроса."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import museum_sources as ms  # noqa: E402


def _fake_met(monkeypatch, tmp_path, changed=()):
    monkeypatch.setattr(ms, "MUSEUM_CACHE_DIR", str(tmp_path))
    ms._MET_CARD_CACHE.clear()
    calls = []

    def get(url):
        calls.append(url)
        if "/search?" in url:
            assert "q=*" in url and "dateBegin=900" in url and "dateEnd=1600" in url
            return {"objectIDs": [1, 2, 3]}
        if "metadataDate=" in url:
            return {"objectIDs": list(changed)}
        oid = int(url.rsplit("/", 1)[1])
        return {"objectID": oid, "title": f"t{oid}", "rev": len(calls)}
    monkeypatch.setattr(ms, "_met_get", get)
    return calls


def test_mirror_fills_the_card_cache_read_by_selection(monkeypatch, tmp_path):
    calls = _fake_met(monkeypatch, tmp_path)
    st = ms.mirror_met_cards(900, 1600)
    assert st["fetched"] == 3 and st["lost"] == 0
    ms._MET_CARD_CACHE.clear()
    assert ms._met_card_cached(2)["title"] == "t2", "отбор читает зеркало тем же кодом"
    n = len(calls)
    st2 = ms.mirror_met_cards(900, 1600)
    assert st2["fresh"] == 3 and st2["fetched"] == 0 and len(calls) == n + 1, "повтор продолжает, а не качает заново"


def test_refresh_renews_unchanged_and_refetches_changed(monkeypatch, tmp_path):
    _fake_met(monkeypatch, tmp_path)
    ms.mirror_met_cards(900, 1600)
    old = time.time() - ms.MUSEUM_CACHE_TTL_SEC - 10
    for oid in (1, 2, 3):
        os.utime(ms._met_card_path(oid), (old, old))
    calls = _fake_met(monkeypatch, tmp_path, changed=[2])
    st = ms.mirror_met_cards(900, 1600, refresh=True)
    assert st["renewed"] == 2 and st["fetched"] == 1
    assert [u for u in calls if u.endswith("/objects/2")], "изменённая карточка перекачана"
    assert not [u for u in calls if u.endswith("/objects/1")], "неизменённая — без запроса"
    assert ms._met_card_fresh(1) and ms._met_card_fresh(3)
