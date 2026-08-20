"""Юнит-тесты Visual QC (scripts/visual_qc.py): резкость/шум на реальных
сгенерированных картинках (без FFmpeg), вердикт-логика через monkeypatch
(без реального CLIP/torch — те же принципы, что test_parse.py уже
использует для pipeline_smart)."""
import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

# visual_qc.py читает VIDEO_DIR из sys.argv[1] на импорте (та же логика,
# что уже требует sys.argv-подмену для pipeline_smart в test_parse.py) —
# под pytest sys.argv это параметры pytest, не путь к видео.
sys.argv = ["visual_qc.py", tempfile.gettempdir()]

import visual_qc as vqc          # noqa: E402
import pipeline_smart            # noqa: E402

from PIL import Image, ImageDraw, ImageFilter   # noqa: E402


def make_sharp_photo(path, size=(480, 270)):
    """Реалистичный (не крайний период-6 чекерборд) резкий кадр — несколько
    контрастных фигур на однородном фоне, с настоящей ГЛАДКОЙ зоной для
    честной проверки noise_score."""
    img = Image.new("RGB", size, (60, 60, 70))
    d = ImageDraw.Draw(img)
    d.rectangle([40, 40, 200, 180], fill=(210, 200, 190))
    d.ellipse([250, 60, 420, 220], fill=(180, 60, 40))
    d.line([0, 0, size[0], size[1]], fill=(255, 255, 255), width=3)
    img.save(path, quality=95)
    return path


def make_blurred_photo(path, size=(480, 270)):
    img = Image.new("RGB", size, (60, 60, 70))
    d = ImageDraw.Draw(img)
    d.rectangle([40, 40, 200, 180], fill=(210, 200, 190))
    d.ellipse([250, 60, 420, 220], fill=(180, 60, 40))
    img = img.filter(ImageFilter.GaussianBlur(radius=12))
    img.save(path, quality=95)
    return path


def make_flat_photo(path, size=(480, 270), color=(128, 128, 128)):
    Image.new("RGB", size, color).save(path, quality=95)
    return path


# ---------- sharpness_score / noise_score ----------

def test_sharpness_higher_for_sharp_than_blurred(tmp_path):
    sharp = make_sharp_photo(str(tmp_path / "sharp.jpg"))
    blurred = make_blurred_photo(str(tmp_path / "blurred.jpg"))
    s_sharp = vqc.sharpness_score(vqc._load_gray_normalized(sharp))
    s_blur = vqc.sharpness_score(vqc._load_gray_normalized(blurred))
    assert s_sharp > s_blur
    assert s_blur < vqc.SHARPNESS_REJECT
    assert s_sharp > vqc.SHARPNESS_REJECT


def test_flat_photo_has_zero_sharpness(tmp_path):
    flat = make_flat_photo(str(tmp_path / "flat.jpg"))
    assert vqc.sharpness_score(vqc._load_gray_normalized(flat)) == 0.0


def test_noise_score_low_on_clean_smooth_region(tmp_path):
    import numpy as np
    clean = np.full((200, 200), 128, dtype=np.uint8)
    img = Image.fromarray(clean, mode="L").convert("RGB")
    p = str(tmp_path / "clean.jpg")
    img.save(p, quality=100)
    n = vqc.noise_score(vqc._load_gray_normalized(p))
    assert n is not None
    assert n < 5.0   # чистое гладкое поле почти без остатка после блюра


def test_noise_score_high_on_noisy_smooth_region(tmp_path):
    import numpy as np
    rng = np.random.default_rng(0)
    noisy = np.clip(128 + rng.normal(0, 30, (200, 200)), 0, 255).astype(np.uint8)
    img = Image.fromarray(noisy, mode="L").convert("RGB")
    p = str(tmp_path / "noisy.jpg")
    img.save(p, quality=100)
    n = vqc.noise_score(vqc._load_gray_normalized(p))
    clean_n = vqc.noise_score(vqc._load_gray_normalized(make_flat_photo(str(tmp_path / "flat2.jpg"))))
    assert n > clean_n


# ---------- qc_verdict (через monkeypatch — без реального CLIP) ----------

def _patch_all_scorers(monkeypatch, luma=0.5, levels=(0.1, 0.9), relevance=0.25,
                        aesthetic=5.0, particle=0.05, ahash_val="0" * 64, motion=0.05):
    monkeypatch.setattr(pipeline_smart, "measure_luma", lambda *a, **kw: luma)
    monkeypatch.setattr(pipeline_smart, "measure_levels", lambda *a, **kw: levels)
    monkeypatch.setattr(pipeline_smart, "clip_relevance", lambda *a, **kw: relevance)
    monkeypatch.setattr(pipeline_smart, "aesthetic_score", lambda *a, **kw: aesthetic)
    monkeypatch.setattr(pipeline_smart, "measure_particle_score", lambda *a, **kw: particle)
    monkeypatch.setattr(pipeline_smart, "measure_motion", lambda *a, **kw: motion)
    monkeypatch.setattr(pipeline_smart, "ahash", lambda *a, **kw: ahash_val)
    monkeypatch.setattr(vqc, "sharpness_score", lambda *a, **kw: 500.0)
    monkeypatch.setattr(vqc, "noise_score", lambda *a, **kw: 5.0)


def _make_photo(tmp_path, name="x.jpg"):
    return make_sharp_photo(str(tmp_path / name))


def test_qc_verdict_pass_on_good_signal(tmp_path, monkeypatch):
    _patch_all_scorers(monkeypatch)
    p = _make_photo(tmp_path)
    info = vqc.qc_verdict(p, is_video=False, query="medieval sword close up",
                           accepted_hashes={}, slot_label="001")
    assert info["verdict"] == "pass"
    assert info["reasons"] == []


def test_qc_verdict_rejects_near_black_frame(tmp_path, monkeypatch):
    _patch_all_scorers(monkeypatch, luma=0.01)
    p = _make_photo(tmp_path)
    info = vqc.qc_verdict(p, is_video=False, query="q", accepted_hashes={}, slot_label="001")
    assert info["verdict"] == "reject"
    assert any("чёрный" in r or "белый" in r for r in info["reasons"])


def test_qc_verdict_rejects_flat_histogram(tmp_path, monkeypatch):
    _patch_all_scorers(monkeypatch, levels=(0.5, 0.51))
    p = _make_photo(tmp_path)
    info = vqc.qc_verdict(p, is_video=False, query="q", accepted_hashes={}, slot_label="001")
    assert info["verdict"] == "reject"
    assert any("гистограмма" in r for r in info["reasons"])


def test_qc_verdict_rejects_low_relevance(tmp_path, monkeypatch):
    _patch_all_scorers(monkeypatch, relevance=0.05)
    p = _make_photo(tmp_path)
    info = vqc.qc_verdict(p, is_video=False, query="unrelated topic", accepted_hashes={}, slot_label="001")
    assert info["verdict"] == "reject"
    assert any("не по теме" in r for r in info["reasons"])


def test_qc_verdict_rejects_duplicate_of_accepted(tmp_path, monkeypatch):
    _patch_all_scorers(monkeypatch, ahash_val="1" * 64)
    p = _make_photo(tmp_path)
    accepted = {"1" * 64: "002"}
    info = vqc.qc_verdict(p, is_video=False, query="q", accepted_hashes=accepted, slot_label="003")
    assert info["verdict"] == "reject"
    assert any("похоже на уже принятый" in r for r in info["reasons"])
    # reject не должен занимать место в пуле дублей
    assert accepted == {"1" * 64: "002"}


def test_qc_verdict_borderline_on_low_aesthetic_not_reject(tmp_path, monkeypatch):
    _patch_all_scorers(monkeypatch, aesthetic=2.0)
    p = _make_photo(tmp_path)
    info = vqc.qc_verdict(p, is_video=False, query="q", accepted_hashes={}, slot_label="001")
    assert info["verdict"] == "borderline"


def test_qc_verdict_missing_file_is_reject():
    info = vqc.qc_verdict("/nonexistent/path/x.jpg", is_video=False, query="q",
                           accepted_hashes={}, slot_label="001")
    assert info["verdict"] == "reject"
    assert info["score"] == -1.0


def test_qc_verdict_accepted_hash_only_updates_on_non_reject(tmp_path, monkeypatch):
    _patch_all_scorers(monkeypatch, ahash_val="abc")
    accepted = {}
    p = _make_photo(tmp_path)
    vqc.qc_verdict(p, is_video=False, query="medieval sword close up", accepted_hashes=accepted, slot_label="001")
    assert accepted == {"abc": "001"}


# ---------- _composite_score ----------

def test_composite_score_prefers_higher_relevance():
    low = vqc._composite_score(sharp=100, noise=5, relevance=0.1, aesthetic=5.0)
    high = vqc._composite_score(sharp=100, noise=5, relevance=0.3, aesthetic=5.0)
    assert high > low


def test_composite_score_penalizes_noise():
    clean = vqc._composite_score(sharp=100, noise=2, relevance=0.2, aesthetic=5.0)
    noisy = vqc._composite_score(sharp=100, noise=35, relevance=0.2, aesthetic=5.0)
    assert clean > noisy


def test_composite_score_handles_all_none():
    assert vqc._composite_score(None, None, None, None) == 0.0


# ---------- load_slot_queries ----------

def test_load_slot_queries_missing_index_returns_empty(tmp_path):
    assert vqc.load_slot_queries(str(tmp_path)) == {}


def test_load_slot_queries_parses_and_builds_query(tmp_path, monkeypatch):
    media_plan = tmp_path / "media_plan"
    media_plan.mkdir()
    (media_plan / "slots_master_index.txt").write_text(
        "1|Тяжёлый меч в руке\n2|Просто атмосфера\n", encoding="utf-8")
    monkeypatch.setattr(vqc.sfm, "load_themes", lambda base: {"меч": "medieval sword close up"})
    out = vqc.load_slot_queries(str(tmp_path))
    assert out[1] == ("Тяжёлый меч в руке", "medieval sword close up")
    assert out[2][1] == vqc.sfm.DEFAULT_QUERY


# ---------- process_slot: AI-файлы никогда не трогаются ----------

def test_process_slot_never_touches_ai_file(tmp_path, monkeypatch):
    media = tmp_path / "media"
    media.mkdir()
    p = make_sharp_photo(str(media / "001_ai.jpg"))
    _patch_all_scorers(monkeypatch)
    entry = vqc.process_slot(str(media), 1, "medieval sword close up", {})
    assert entry["source"] == "ai"
    assert entry["auto_replaced"] is False
    assert os.path.exists(p)


def test_process_slot_never_leaves_slot_empty_when_refetch_unavailable(tmp_path, monkeypatch):
    # Без PEXELS/PIXABAY/UNSPLASH ключей try_sources() всегда возвращает
    # None — тот же путь, что реально происходит без .env в CI. Плохой
    # (тёмный) исходный файл всё равно должен остаться на месте, помеченный
    # accepted_below_threshold, а не исчезнуть/сломать сборку.
    media = tmp_path / "media"
    media.mkdir()
    p = make_sharp_photo(str(media / "001_stock.jpg"))
    monkeypatch.setattr(pipeline_smart, "measure_luma", lambda *a, **kw: 0.01)   # форсируем reject
    monkeypatch.setattr(pipeline_smart, "measure_levels", lambda *a, **kw: (0.1, 0.9))
    monkeypatch.setattr(pipeline_smart, "clip_relevance", lambda *a, **kw: 0.25)
    monkeypatch.setattr(pipeline_smart, "aesthetic_score", lambda *a, **kw: 5.0)
    monkeypatch.setattr(pipeline_smart, "measure_particle_score", lambda *a, **kw: 0.05)
    monkeypatch.setattr(pipeline_smart, "measure_motion", lambda *a, **kw: 0.05)
    monkeypatch.setattr(pipeline_smart, "ahash", lambda *a, **kw: "0" * 64)
    monkeypatch.setattr(vqc.sfm, "try_sources", lambda *a, **kw: None)   # нет доступных источников

    entry = vqc.process_slot(str(media), 1, "medieval sword close up", {})
    assert entry["verdict"] == "accepted_below_threshold"
    assert entry["auto_replaced"] is False
    assert os.path.exists(p)   # файл НИКУДА не делся
