"""GPU-ветка: упреждающий поиск, кодер клипов, устройство моделей.

Главный инвариант упреждения — выбор слота не меняется: упреждение только
прогревает дисковые кэши, а в кэши на процесс и в счётчики, от которых
зависят решения слота, не пишет. Каждый тест ниже держит одну из этих
границ; снятая граница роняет свой тест.
"""
import json
import os
import sys
import tempfile
import threading
import time

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]

import pipeline_smart as ps  # noqa: E402
import slot_prefetch  # noqa: E402
import stock_fetch_multisource as ms  # noqa: E402


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return json.dumps(self._payload).encode()


PIXABAY = {"hits": [{"id": 7, "tags": "knight", "pageURL": "https://pixabay.com/photos/knight-7/",
                     "largeImageURL": "https://cdn.pixabay.com/7_1280.jpg"}]}


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    monkeypatch.setenv("SEARCH_CACHE_DIR", str(tmp_path / "search_cache"))
    for c in (ps._PIXABAY_PHOTO_CACHE, ps._UNSPLASH_PHOTO_CACHE, ps._PEXELS_SEARCH_CACHE):
        c.clear()
    ps._UNSPLASH_CALLS_THIS_RUN[0] = 0
    ps.SOURCE_STATS.clear()
    yield
    for c in (ps._PIXABAY_PHOTO_CACHE, ps._UNSPLASH_PHOTO_CACHE, ps._PEXELS_SEARCH_CACHE):
        c.clear()
    ps._UNSPLASH_CALLS_THIS_RUN[0] = 0


def _in_prefetch(fn, *a, **kw):
    token = ps._PREFETCH_ACTIVE.set(True)
    try:
        return fn(*a, **kw)
    finally:
        ps._PREFETCH_ACTIVE.reset(token)


# ---------- упреждение не пишет в кэши на процесс ----------

def test_prefetch_warms_disk_cache_but_not_process_cache(monkeypatch):
    """Упреждение кладёт выдачу на диск, слот потом читает её без сети — и
    получает ровно то же, что получил бы сам."""
    monkeypatch.setenv("PIXABAY_ENABLED", "1")
    monkeypatch.setattr(ms, "PIXABAY_API_KEY", "k", raising=False)
    calls = []
    monkeypatch.setattr(ps.urllib.request, "urlopen",
                        lambda *a, **kw: calls.append(1) or _Resp(PIXABAY))
    warm = _in_prefetch(ps._pixabay_search_photos, "knight")
    assert "knight" not in ps._PIXABAY_PHOTO_CACHE
    assert len(calls) == 1

    def _boom(*a, **kw):
        raise AssertionError("слот должен прочесть выдачу с диска")
    monkeypatch.setattr(ps.urllib.request, "urlopen", _boom)
    assert ps._pixabay_search_photos("knight") == warm
    assert "knight" in ps._PIXABAY_PHOTO_CACHE


def test_prefetch_failure_does_not_poison_the_slot(monkeypatch):
    """Сбой упреждения (429) не превращается в пустой источник у слота: в
    кэш на процесс пустота после сбоя не пишется, и не считается ошибкой."""
    monkeypatch.setenv("PIXABAY_ENABLED", "1")
    monkeypatch.setattr(ms, "PIXABAY_API_KEY", "k", raising=False)

    def _fail(*a, **kw):
        raise ps.urllib.error.HTTPError("u", 429, "slow down", None, None)
    monkeypatch.setattr(ps.urllib.request, "urlopen", _fail)
    assert _in_prefetch(ps._pixabay_search_photos, "knight") == []
    assert "knight" not in ps._PIXABAY_PHOTO_CACHE
    assert not ps.SOURCE_STATS.get("pixabay", {}).get("search_errors")
    monkeypatch.setattr(ps.urllib.request, "urlopen", lambda *a, **kw: _Resp(PIXABAY))
    assert [c["id"] for c in ps._pixabay_search_photos("knight")] == ["pixabay:7"]


def test_prefetch_skips_unsplash_and_leaves_run_cap_alone(monkeypatch):
    """Потолок Unsplash считает и ответы из кэша — упреждение его не трогает."""
    monkeypatch.setenv("UNSPLASH_ENABLED", "1")
    monkeypatch.setattr(ms, "UNSPLASH_ACCESS_KEY", "k", raising=False)

    def _boom(*a, **kw):
        raise AssertionError("упреждение не ходит в Unsplash")
    monkeypatch.setattr(ps.urllib.request, "urlopen", _boom)
    assert _in_prefetch(ps._unsplash_search_photos, "knight") == []
    assert ps._UNSPLASH_CALLS_THIS_RUN[0] == 0
    assert "knight" not in ps._UNSPLASH_PHOTO_CACHE


def test_prefetch_asks_pexels_only_own_queries(monkeypatch):
    """Дополнительные запросы Pexels решаются остатком квоты в момент слота:
    упреждение их не спрашивает и счётчик пропусков не трогает."""
    monkeypatch.setattr(ps, "PEXELS_QUOTA_LEFT", 0)
    monkeypatch.setattr(ps, "PEXELS_QUOTA_RESERVE", 5)
    before = ps.PEXELS_LOW_PRIORITY_SKIPPED
    assert _in_prefetch(ps.pexels_query_allowed, "q", {}, True) is False
    assert _in_prefetch(ps.pexels_query_allowed, "q", {}, False) is True
    assert ps.PEXELS_LOW_PRIORITY_SKIPPED == before
    # у слота — прежнее правило
    assert ps.pexels_query_allowed("q", {}, True) is False
    assert ps.PEXELS_LOW_PRIORITY_SKIPPED == before + 1


def test_prefetch_source_failure_is_not_reported_to_the_adapter(monkeypatch):
    """Сбой источника в упреждении не взводит серию сбоев Pexels."""
    seen = []
    monkeypatch.setattr(ps, "_note_pexels_failure", lambda *a, **kw: seen.append(a))
    wrapper = ps._PrefetchSources(ps.PHOTO_ADAPTER)
    wrapper.on_source_failure(None, "pexels", RuntimeError("500"))
    assert seen == []


def test_every_process_cache_write_goes_through_cache_put():
    """Прямая запись в кэш поиска на процесс обошла бы границу упреждения."""
    import re
    src = open(os.path.join(REPO_ROOT, "scripts", "pipeline_smart.py"), encoding="utf-8").read()
    direct = re.findall(r"^\s*_(?:PEXELS_SEARCH|MUSEUM_SEARCH|SHELF_SEARCH|PIXABAY_PHOTO|"
                        r"PIXABAY_VIDEO|UNSPLASH_PHOTO|OPENVERSE_SEARCH|COMMONS_SEARCH|"
                        r"PEXELS_VIDEO_SEARCH)_CACHE\[[^\]]+\] = ", src, re.M)
    assert direct == []


# ---------- планировщик упреждения ----------

def _drain(p, timeout=5.0):
    """Дождаться, пока все поставленные задания отработают (close() отменяет
    ещё не начатые — так и задумано в конце цикла слотов)."""
    end = time.time() + timeout
    while time.time() < end:
        st = p.stats
        if st["done"] + st["failed"] >= st["scheduled"]:
            return
        time.sleep(0.01)


def test_prefetcher_keeps_depth_ahead_and_swallows_errors():
    done, lock = [], threading.Lock()

    def job(j):
        if j == 2:
            raise RuntimeError("сбой упреждения")
        with lock:
            done.append(j)
    p = slot_prefetch.SlotPrefetcher(6, job, depth=2, workers=1)
    p.advance(0)
    p.advance(1)
    p.advance(1)
    _drain(p)
    p.close()
    # advance(0) ставит 1 и 2, advance(1) — только новый 3, повтор — ничего
    assert sorted(done) == [1, 3]
    assert p.stats == {"scheduled": 3, "done": 2, "failed": 1}


def test_prefetcher_never_goes_past_the_last_slot():
    got = []
    p = slot_prefetch.SlotPrefetcher(3, got.append, depth=5, workers=1)
    p.advance(0)
    _drain(p)
    p.close()
    assert sorted(got) == [1, 2]


def test_prefetcher_after_close_schedules_nothing():
    got = []
    p = slot_prefetch.SlotPrefetcher(5, got.append, depth=2, workers=1)
    p.close()
    p.advance(0)
    time.sleep(0.05)
    assert got == []


def test_slot_loop_advances_the_prefetcher():
    """Слой без вызывающего — ровно тот класс «код есть, его никто не зовёт»."""
    src = open(os.path.join(REPO_ROOT, "scripts", "pipeline_smart.py"), encoding="utf-8").read()
    loop = src.index("prefetcher = slot_prefetch.SlotPrefetcher(")
    assert "prefetcher.advance(i)" in src[loop:loop + 400]
    assert "prefetcher.close()" in src[loop:]


# ---------- кодер клипов ----------

def test_clip_codec_args_on_cpu_are_the_old_libx264(monkeypatch):
    monkeypatch.delenv("CLIP_ENCODER_RESOLVED", raising=False)
    monkeypatch.setattr(ps, "_NVENC_BROKEN", [False])
    assert ps.clip_codec_args() == ["-c:v", "libx264", "-preset", ps.RENDER_PRESET,
                                     "-crf", ps.RENDER_CRF] + ps.CLIP_PIX_ARGS


def test_nvenc_when_resolved_and_back_to_x264_after_nvenc_failure(monkeypatch):
    monkeypatch.setenv("CLIP_ENCODER_RESOLVED", "nvenc")
    monkeypatch.setattr(ps, "_NVENC_BROKEN", [False])
    args = ps.clip_codec_args()
    assert args[:2] == ["-c:v", "hevc_nvenc"] and "p010le" in args
    ps.note_encoder_failure("unrelated error")
    assert ps.clip_encoder() == "nvenc"
    ps.note_encoder_failure("[hevc_nvenc] OpenEncodeSessionEx failed: out of memory")
    assert ps.clip_encoder() == "x264"


def test_resolve_clip_encoder_falls_back_without_nvenc(monkeypatch):
    monkeypatch.setenv("CLIP_ENCODER", "auto")
    monkeypatch.setattr(ps, "nvenc_works", lambda: False)
    assert ps.resolve_clip_encoder() == "x264"
    assert os.environ["CLIP_ENCODER_RESOLVED"] == "x264"
    monkeypatch.setattr(ps, "nvenc_works", lambda: True)
    assert ps.resolve_clip_encoder() == "nvenc"
    monkeypatch.setenv("CLIP_ENCODER", "x264")
    assert ps.resolve_clip_encoder() == "x264"


def test_recipe_signature_unchanged_on_x264_and_differs_on_nvenc(monkeypatch):
    monkeypatch.setattr(ps, "_NVENC_BROKEN", [False])
    monkeypatch.setenv("CLIP_ENCODER_RESOLVED", "x264")
    cpu = ps.render_recipe_signature()
    monkeypatch.setenv("CLIP_ENCODER_RESOLVED", "nvenc")
    assert ps.render_recipe_signature() != cpu


def test_no_clip_encode_site_bypasses_clip_codec_args():
    src = open(os.path.join(REPO_ROOT, "scripts", "pipeline_smart.py"), encoding="utf-8").read()
    assert src.count('"libx264", "-preset", RENDER_PRESET') == 1   # только в clip_codec_args


# ---------- устройство моделей ----------

def test_cascade_batch_on_cpu_is_the_old_16(monkeypatch):
    import ml_device
    monkeypatch.setattr(ml_device, "device", lambda: "cpu")
    assert ps.cascade_batch() == 16
    monkeypatch.setattr(ml_device, "device", lambda: "cuda")
    assert ps.cascade_batch() == ps.CASCADE_BATCH_GPU


def test_jina_stays_on_cpu_without_cuda(monkeypatch):
    import ml_device
    import visual_director as vd
    vd.jina_providers.cache_clear()
    monkeypatch.setattr(ml_device, "device", lambda: "cpu")
    try:
        assert vd.jina_providers() == ["CPUExecutionProvider"]
        assert vd.jina_device_tag() == ""
    finally:
        vd.jina_providers.cache_clear()


# ---------- сквозной: куча слота та же, что без упреждения ----------

PEXELS = {"photos": [{"id": 501, "alt": "knight armour", "url": "https://www.pexels.com/photo/knight-501/",
                      "src": {"large2x": "https://images.pexels.com/501.jpg",
                              "medium": "https://images.pexels.com/501m.jpg"}}]}


def _fake_net(calls, fail=()):
    def urlopen(req, *a, **kw):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        calls.append(url)
        if any(f in url for f in fail):
            raise ps.urllib.error.HTTPError(url, 429, "slow down", None, None)
        return _Resp(PEXELS if "pexels.com" in url else PIXABAY)
    return urlopen


def _request(index=3, query="medieval knight", extra=("medieval sword",)):
    return ps.build_slot_request(
        index=index, query=query, extra_queries=extra, text_key="Рыцарь поднял меч.",
        shot_brief=None, block_text="Рыцарь поднял меч.", shot_spec=None, arbiter_text=None,
        is_opening=False, slot_dur=4.0, action_qualifier=None, target_luma=None,
        director_score_fn=None, director_assist=False, director_report=None, video_score_fn=None,
        used_photo_ids=set(), used_video_ids=set(), used_hashes=[], recent_sizes=[])


def _ids(pool):
    return [c["id"] for c in pool]


@pytest.mark.parametrize("fail", [(), ("pixabay.com",)])
def test_pool_after_prefetch_is_the_pool_without_it(monkeypatch, tmp_path, fail):
    """Куча слота после упреждения — та же, что без него, и та же при
    сбое источника в упреждении (слот спрашивает сам и получает ответ)."""
    monkeypatch.setenv("PIXABAY_ENABLED", "1")
    monkeypatch.setattr(ms, "PIXABAY_API_KEY", "k", raising=False)
    monkeypatch.setattr(ps, "PEXELS_API_KEY", "k")
    import selection_engine as se

    # без упреждения
    monkeypatch.setenv("SEARCH_CACHE_DIR", str(tmp_path / "a"))
    calls = []
    monkeypatch.setattr(ps.urllib.request, "urlopen", _fake_net(calls))
    baseline = _ids(se.build_pool(_request(), ps.PHOTO_ADAPTER))
    assert baseline

    # с упреждением: сперва прогрев (возможно со сбоем), потом слот
    for c in (ps._PIXABAY_PHOTO_CACHE, ps._PEXELS_SEARCH_CACHE):
        c.clear()
    monkeypatch.setenv("SEARCH_CACHE_DIR", str(tmp_path / "b"))
    calls = []
    monkeypatch.setattr(ps.urllib.request, "urlopen", _fake_net(calls, fail))
    ps.prefetch_slot_inputs(_request(), "photo", cascade=False)
    monkeypatch.setattr(ps.urllib.request, "urlopen", _fake_net(calls))
    live_before = len(calls)
    assert _ids(se.build_pool(_request(), ps.PHOTO_ADAPTER)) == baseline
    # слот сам ходит в сеть только за тем, что упреждение не получило:
    # дополнительный запрос Pexels (его упреждение не спрашивает — квота
    # решается в момент слота) и, при сбое, оба запроса Pixabay
    assert len(calls) - live_before == (3 if fail else 1)


def test_cascade_after_prefetch_same_order_no_recompute(monkeypatch, tmp_path):
    """Каскад слота после упреждения: тот же порядок, ни одной картинки не
    считается заново (эмбеддинг из кэша по адресу превью)."""
    import numpy as np
    from PIL import Image
    monkeypatch.setattr(ps, "TEMP_FOLDER", str(tmp_path / "temp"))
    monkeypatch.setenv("CASCADE_CACHE_DIR", str(tmp_path / "casc"))
    rel = {"a": 0.1, "b": 0.9, "c": 0.5, "d": 0.3}
    ids = sorted(rel)
    calls = {"images": 0}

    def probe(p, dest):
        Image.new("RGB", (8, 8), (ids.index(p["id"]) * 10, 0, 0)).save(dest, "JPEG")

    def embed(images=None, text=None):
        if text is not None:
            return np.ones((1, 1), dtype="float32")
        calls["images"] += len(images)
        return np.array([[rel[ids[round(im.getpixel((4, 4))[0] / 10)]]] for im in images], "float32")
    monkeypatch.setattr(ps, "_gate_embed", embed)
    monkeypatch.setattr(ps, "download_photo_probe", probe)
    cands = [{"id": k, "src": {"large": f"http://x/{k}.jpg"}} for k in ids]

    monkeypatch.setattr(ps, "_CASCADE_EMB", {})
    base = [p["id"] for p in ps.cascade_reorder(cands, "x", str(tmp_path / "a.jpg"), probe, 0)]

    monkeypatch.setattr(ps, "_CASCADE_EMB", {})
    monkeypatch.setenv("CASCADE_CACHE_DIR", str(tmp_path / "casc2"))
    monkeypatch.setattr(ps.selection_engine, "fetch_sources",
                        lambda req, ad, qs: {q: [cands] for q in qs})
    req = _request(query="x", extra=())
    ps.prefetch_slot_inputs(req, "photo", cascade=True)
    assert calls["images"] == 8          # 4 без упреждения + 4 в упреждении
    got = [p["id"] for p in ps.cascade_reorder(cands, "x", str(tmp_path / "b.jpg"), probe, 0)]
    assert got == base
    assert calls["images"] == 8, "слот не пересчитывает то, что посчитало упреждение"


def test_same_search_at_once_goes_to_network_once(monkeypatch, tmp_path):
    """Упреждение и слот спросили одно и то же одновременно — один запрос."""
    monkeypatch.setenv("SEARCH_CACHE_DIR", str(tmp_path / "sc"))
    calls, started = [], threading.Event()

    def fetch():
        calls.append(1)
        started.set()
        time.sleep(0.2)
        return {"hits": [1]}
    out = []
    t = threading.Thread(target=lambda: out.append(ps.cached_search_json("src", "k", fetch)))
    t.start()
    started.wait(2)
    out.append(ps.cached_search_json("src", "k", fetch))
    t.join()
    assert out == [{"hits": [1]}, {"hits": [1]}] and len(calls) == 1


def test_failed_search_is_asked_again_by_the_waiter(monkeypatch, tmp_path):
    monkeypatch.setenv("SEARCH_CACHE_DIR", str(tmp_path / "sc"))

    def boom():
        raise OSError("429")
    with pytest.raises(OSError):
        ps.cached_search_json("src", "k", boom)
    assert ps.cached_search_json("src", "k", lambda: {"ok": 1}) == {"ok": 1}


def test_parallax_redraws_on_x264_when_nvenc_fails(monkeypatch):
    """Аудит 28.09: сбой NVENC в параллаксе (BrokenPipeError на stdin или
    строка про hevc_nvenc раньше последних 200 символов) не отмечал NVENC
    сломанным, и кадр молча терял параллакс — обычный наезд вместо 2.5D."""
    monkeypatch.setenv("CLIP_ENCODER_RESOLVED", "nvenc")
    monkeypatch.setattr(ps, "_NVENC_BROKEN", [False])
    calls = []

    def parallax(i, photo, out, d, **kw):
        calls.append(ps.clip_encoder())
        if len(calls) == 1:
            ps.note_encoder_failure("[hevc_nvenc @ 0x1] OpenEncodeSessionEx failed" + " x" * 400)
            return False
        return True
    monkeypatch.setattr(ps, "_timed_render", lambda fn, i, *a, **kw: parallax(i, *a, **kw)
                        if fn is ps.parallax_kenburns else pytest.fail("ушло в обычный наезд"))
    assert ps.render_highlight_clip(None, 0, "p.jpg", "o.mp4", 3.0, "classic_kb", None, {}) is True
    assert calls == ["nvenc", "x264"]


def test_parallax_exception_path_reads_encoder_error():
    import inspect
    src = inspect.getsource(ps.parallax_kenburns)
    assert src.count("note_encoder_failure(") >= 2
    assert "[-200:]\n            note_encoder_failure" not in src


@pytest.mark.skipif(__import__("shutil").which("ffmpeg") is None, reason="нужен ffmpeg")
def test_fallback_concat_keeps_every_frame_of_mixed_codecs(tmp_path):
    """Аудит 28.09: запасная склейка (xfade не собрался) копировала клипы
    concat -c copy. На видеокарте клипы — HEVC от NVENC, после сбоя NVENC
    часть — H.264; демуксер берёт кодек первого файла, и из 48 кадров смеси
    оставалось 24 с потоком ошибок декодера. Перекодирование одним куском
    давало то же."""
    import subprocess
    enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    if "libx265" not in enc:
        pytest.skip("нужен libx265")
    a, b, out = str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4"), str(tmp_path / "o.mp4")
    for path, src, codec in ((a, "testsrc", ["-c:v", "libx264", "-profile:v", "high10"]),
                             (b, "testsrc2", ["-c:v", "libx265"])):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        f"{src}=s=320x180:d=1:r={ps.FPS}"] + codec + ["-pix_fmt", "yuv420p10le", path],
                       check=True)
    ok, err = ps.concat_reencode([a, b, a], out, str(tmp_path), timeout=120)
    assert ok, err
    frames = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                             "-show_entries", "stream=nb_read_frames,codec_name", "-of", "csv=p=0", out],
                            capture_output=True, text=True).stdout.strip()
    assert frames == f"h264,{3 * ps.FPS}", frames
    errors = subprocess.run(["ffmpeg", "-v", "error", "-i", out, "-f", "null", "-"],
                            capture_output=True, text=True).stderr.strip()
    assert errors == ""
    # Число кадров сходится и при порче: вместо HEVC-участка ffmpeg кладёт
    # испорченные/повторённые кадры. Сравнивается сама картинка середины.
    from PIL import Image, ImageChops, ImageStat

    def frame(path, t):
        dest = str(tmp_path / f"f_{os.path.basename(path)}_{t}.png")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(t), "-i", path,
                        "-frames:v", "1", dest], check=True)
        return Image.open(dest).convert("RGB")

    def diff(x, y):
        return sum(ImageStat.Stat(ImageChops.difference(x, y)).mean)
    mid = frame(out, 1.5)
    assert diff(mid, frame(b, 0.5)) < diff(mid, frame(a, 0.5)) / 3


def test_speculation_queue_is_cancelled_on_abnormal_exit():
    """Аудит 28.09: при падении цикла до close() исполнитель при выходе
    дорабатывал всю очередь — платные лестницы слотов после сбоя."""
    import slot_speculation
    gate = threading.Event()
    ran = []

    def job(j, snap):
        ran.append(j)
        gate.wait(5)
    sp = slot_speculation.SlotSpeculator(10, job, depth=6, workers=1)
    sp.advance(0, lambda: {})
    time.sleep(0.2)
    sp._abandon()
    gate.set()
    time.sleep(0.3)
    assert ran == [1], ran


def test_default_speculation_is_off():
    import feature_flags
    assert feature_flags.FLAGS["SLOT_SPECULATE"].default == "0"
