# -*- coding: utf-8 -*-
"""Библиотека настоящих звуков — то, что проверяемо без сети и без моделей.

Сеть, CLAP и AST здесь не трогаются: живой отбор — отдельный прогон
(`python scripts/sound_library.py build`), его результат лежит в
assets/library/manifest.json с числами по каждому файлу. Тут — правила,
которые обязаны держаться независимо от того, что вернул поиск сегодня.
"""
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import sound_library as sl  # noqa: E402


def test_license_gate_is_fail_closed():
    """Фильтр в URL запроса — экономия трафика, не гарантия. Решает поле
    результата, и только cc0 — тот же принцип, что у картинок Openverse."""
    assert sl.is_safe_license({"license": "cc0"})
    assert sl.is_safe_license({"license": "CC0"})
    for bad in ("by", "by-sa", "pdm", "by-nc", "", None):
        assert not sl.is_safe_license({"license": bad})
    assert not sl.is_safe_license({})


def test_every_spec_entry_is_complete():
    for kind, entries in sl.LIBRARY_SPEC.items():
        for name, spec in entries.items():
            assert spec["queries"], f"{kind}/{name}: нет запросов"
            assert spec["prompt"], f"{kind}/{name}: нет промпта для CLAP"
            assert spec.get("keep", 0) >= 1
            assert spec.get("min_sec", 0) >= 0


def test_ambience_beds_match_the_planner_vocabulary():
    """Каждый вид атмосферы, который умеет выбирать планировщик, обязан
    иметь рецепт поиска — иначе выбор кончался бы тишиной молча."""
    import ambience_plan as ap
    for bed in ap.AMBIENCE_VOCAB:
        assert bed in sl.LIBRARY_SPEC["ambience"], f"нет рецепта поиска для «{bed}»"


def test_preview_url_swaps_quality_only_for_freesound():
    hq = "https://cdn.freesound.org/previews/172/172666_2213158-hq.mp3"
    assert sl.preview_url(hq, "lq").endswith("-lq.mp3")
    assert sl.preview_url(sl.preview_url(hq, "lq"), "hq") == hq
    other = "https://upload.wikimedia.org/wikipedia/commons/a/ab/x.ogg"
    assert sl.preview_url(other, "lq") == other


def test_clipping_share_counts_full_scale_samples():
    clean = np.sin(np.linspace(0, 200, 48000)).astype(np.float32) * 0.5
    assert sl.clipping_share(clean) == 0.0
    clipped = clean.copy()
    clipped[:100] = 1.0
    assert sl.clipping_share(clipped) > sl.CLIP_SAMPLE_SHARE


def test_hum_detector_flags_mains_tone_and_ignores_broadband():
    sr = 48000
    t = np.arange(sr * 8) / sr
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.05, t.size).astype(np.float32)
    assert sl.hum_prominence_db(noise, sr) < sl.HUM_PROMINENCE_DB
    hum = (noise + 0.3 * np.sin(2 * np.pi * 50.0 * t)).astype(np.float32)
    assert sl.hum_prominence_db(hum, sr) > sl.HUM_PROMINENCE_DB


def test_negative_prompts_cover_the_real_failure_modes():
    joined = " ".join(sl.NEGATIVE_PROMPTS).lower()
    for must in ("speech", "music", "engine", "hum", "distortion"):
        assert must in joined


def test_thresholds_are_conservative_in_the_right_direction():
    """Маржа строго положительная: положительный промпт обязан ПЕРЕБИВАТЬ
    худшую ловушку, а не дотягиваться до неё."""
    assert sl.CLAP_MIN_MARGIN > 0
    assert 0 < sl.AST_VETO["Speech"] < 0.5
    assert sl.AMBIENCE_PEAK_DBFS < sl.SFX_PEAK_DBFS < 0


def test_manifest_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(sl, "LIBRARY_ROOT", str(tmp_path))
    monkeypatch.setattr(sl, "MANIFEST_PATH", str(tmp_path / "manifest.json"))
    assert sl.load_manifest() == {"items": {}}
    m = {"items": {"assets/library/sfx/x/a.flac": {"license": "cc0"}}}
    sl.save_manifest(m)
    assert sl.load_manifest() == m


def test_library_files_lists_only_flac(tmp_path, monkeypatch):
    monkeypatch.setattr(sl, "LIBRARY_ROOT", str(tmp_path))
    d = tmp_path / "sfx" / "plate_tick"
    d.mkdir(parents=True)
    (d / "b.flac").write_bytes(b"x")
    (d / "a.flac").write_bytes(b"x")
    (d / "notes.txt").write_bytes(b"x")
    assert [os.path.basename(p) for p in sl.library_files("sfx", "plate_tick")] == ["a.flac", "b.flac"]
    assert sl.library_files("sfx", "missing") == []
