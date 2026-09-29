#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ускорения поиска, каскада и судьи (26.09): те же кандидаты, тот же
порядок, те же ответы — меньше времени. Каждый тест падает на прежнем коде."""
import concurrent.futures
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.argv = ["pipeline_smart.py", REPO]
import pipeline_smart as ps  # noqa: E402
import museum_sources as ms  # noqa: E402
import met_catalog  # noqa: E402
import shot_judge  # noqa: E402
import llm_gateway as lg  # noqa: E402
import ml_device  # noqa: E402


# ---------------------------------------------------------------- поиск

def test_blocklist_builds_candidate_text_once_per_candidate(monkeypatch):
    calls = []
    real = ps.pexels_candidate_text

    def counted(p):
        calls.append(p["id"])
        return real(p)
    monkeypatch.setattr(ps, "pexels_candidate_text", counted)
    monkeypatch.setattr(ps, "content_blocklist_effective", lambda: [f"t{k}" for k in range(50)] + ["cosplay"])
    items = [{"id": k, "alt": "cosplay knight" if k == 1 else "knight", "url": ""} for k in range(10)]
    out = ps.filter_alt_blocklist(items)
    assert [p["id"] for p in out] == [0, 2, 3, 4, 5, 6, 7, 8, 9]
    assert len(calls) == len(items)


def test_catalog_checks_passport_only_on_word_matches(monkeypatch):
    rows = [{"id": str(k), "dept": "Arms and Armor", "name": "Sword" if k < 3 else "Plate",
             "cls": "", "tags": "", "title": "", "b": 1400, "e": 1450, "culture": "French"}
            for k in range(200)]
    monkeypatch.setattr(met_catalog, "_load", lambda: {"rows": rows})
    seen = []
    real = ms.culture_is_foreign

    def counted(*a):
        seen.append(a)
        return real(*a)
    monkeypatch.setattr(ms, "culture_is_foreign", counted)
    got = met_catalog.search("sword")
    assert [r["id"] for r in got] == ["0", "1", "2"]
    assert len(seen) == 3


def test_three_museums_are_asked_at_once_and_keep_their_order(monkeypatch):
    monkeypatch.setenv("MUSEUM_SOURCES_ENABLED", "1")
    def slow(tag):
        def fn(q, **kw):
            time.sleep(0.4)
            return [{"id": f"{tag}:{k}"} for k in range(2)]
        return fn
    monkeypatch.setattr(ms, "_sources", lambda department=None: (
        ("met", slow("met")), ("cleveland", slow("cleveland")), ("chicago", slow("chicago"))))
    t = time.time()
    out = ms.search_museums("q-order")
    assert time.time() - t < 1.0, "музеи снова опрашиваются по очереди"
    assert [c["id"] for c in out] == ["met:0", "cleveland:0", "chicago:0",
                                      "met:1", "cleveland:1", "chicago:1"]


def test_complete_empty_museum_answer_is_cached(monkeypatch):
    monkeypatch.setenv("MUSEUM_SOURCES_ENABLED", "1")
    calls = []

    def empty(q, **kw):
        calls.append(q)
        return []
    monkeypatch.setattr(ms, "_sources", lambda department=None: (
        ("met", empty), ("cleveland", empty), ("chicago", empty)))
    assert ms.search_museums("nothing here") == []
    ms._SEARCH_CACHE.clear()
    assert ms.search_museums("nothing here") == []
    assert len(calls) == 3, "полный пустой ответ снова спрошен у музеев"


def test_failed_met_search_is_not_cached_as_complete(monkeypatch):
    monkeypatch.setenv("MUSEUM_SOURCES_ENABLED", "1")
    monkeypatch.setattr(ms, "_met_get", lambda url, max_wait=None: None)
    monkeypatch.setattr(ms, "search_cleveland", lambda q, **kw: [{"id": "cleveland:1"}])
    monkeypatch.setattr(ms, "search_chicago", lambda q, **kw: [])
    monkeypatch.setenv("MET_CATALOG", "0")
    out = ms.search_museums("met down")
    assert [c["id"] for c in out] == ["cleveland:1"]
    assert ms._disk_cache_get("met down") is None, "ответ без Мет записан в кэш как полный"


def test_openverse_empty_answer_with_result_count_is_cached(monkeypatch, tmp_path):
    import stock_fetch_multisource as ov
    monkeypatch.setattr(ps, "OPENVERSE_CACHE_DIR", str(tmp_path / "ov"), raising=False)
    monkeypatch.setattr(ps, "_openverse_throttle", lambda auth: None)
    monkeypatch.setattr(ps, "_openverse_bearer", lambda: None)
    hits = []

    class R:
        def __init__(self, body):
            self.body = body

        def read(self, *a):
            return self.body

        def __enter__(self):
            return self

        def __exit__(self, *e):
            return False

    def fake(req, timeout=None):
        hits.append(req.full_url)
        return R(json.dumps({"result_count": 0, "results": []}).encode())
    monkeypatch.setattr(ps.urllib.request, "urlopen", fake)
    assert ps._openverse_fetch_one("medieval exact phrase", ov) == []
    assert ps._openverse_fetch_one("medieval exact phrase", ov) == []
    assert len(hits) == 1


def test_unsplash_answer_comes_from_disk_and_still_counts(monkeypatch, tmp_path):
    import stock_fetch_multisource as sm
    monkeypatch.setenv("SEARCH_CACHE_DIR", str(tmp_path / "sc"))
    monkeypatch.setenv("UNSPLASH_ENABLED", "1")
    monkeypatch.setattr(sm, "UNSPLASH_ACCESS_KEY", "k")
    hits = []

    class R:
        def read(self, *a):
            return json.dumps({"results": [{"id": "a", "urls": {"regular": "http://u/a.jpg"}}]}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *e):
            return False

    monkeypatch.setattr(ps.urllib.request, "urlopen", lambda req, timeout=None: (hits.append(1), R())[1])
    ps._UNSPLASH_CALLS_THIS_RUN[0] = 0
    a = ps._unsplash_search_photos("knight")
    ps._UNSPLASH_PHOTO_CACHE.clear()
    b = ps._unsplash_search_photos("knight")
    assert a == b and a[0]["id"] == "unsplash:a"
    assert len(hits) == 1
    assert ps._UNSPLASH_CALLS_THIS_RUN[0] == 2, "ответ из кэша обязан считаться в потолок прогона"


def test_pacing_sleeps_only_after_live_pexels_requests(monkeypatch):
    slept = []
    monkeypatch.setattr(ps.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(ps, "_PEXELS_PACED_AT", [ps._PEXELS_LIVE_REQUESTS[0]])
    ps._stock_api_pacing(9, True, False)
    assert slept == []

    class Resp:
        headers = {}
    ps._note_pexels_quota(Resp())
    ps._stock_api_pacing(19, True, False)
    assert slept == [0.4]


def test_local_stock_digest_is_read_once_per_file_version(monkeypatch, tmp_path):
    f = tmp_path / "001_stock.jpg"
    Image.new("RGB", (8, 8)).save(f)
    monkeypatch.setattr(ps, "local_photo", lambda i: str(f))
    monkeypatch.setattr(ps, "local_file_is_machine_stock", lambda p: True)
    reads = []
    real = ps._file_digest
    monkeypatch.setattr(ps, "_file_digest", lambda p: (reads.append(p), real(p))[1])
    a = ps.local_stock_candidate(0)
    b = ps.local_stock_candidate(0)
    assert a == b and len(reads) == 1


# ---------------------------------------------------------------- каскад

def _cascade_world(tmp_path, monkeypatch, rel):
    monkeypatch.setattr(ps, "TEMP_FOLDER", str(tmp_path / "temp"))
    monkeypatch.setattr(ps, "_CASCADE_EMB", {})
    ids = sorted(rel)
    got = []

    def probe(p, dest):
        got.append(p["id"])
        Image.new("RGB", (8, 8), (ids.index(p["id"]) * 10, 0, 0)).save(dest, "JPEG")

    def embed(images=None, text=None):
        if text is not None:
            return np.ones((1, 1), dtype="float32")
        return np.array([[rel[ids[round(im.getpixel((4, 4))[0] / 10)]]] for im in images], "float32")
    monkeypatch.setattr(ps, "_gate_embed", embed)
    return probe, got


def _cands(n):
    return [{"id": f"c{k}", "src": {"large": f"http://x/{k}.jpg"}} for k in range(n)]


def test_cascade_cache_is_shared_across_episodes(tmp_path, monkeypatch):
    rel = {"c0": 0.1, "c1": 0.2, "c2": 0.3}
    probe, got = _cascade_world(tmp_path, monkeypatch, rel)
    shared = tmp_path / "shared"
    monkeypatch.setenv("CASCADE_CACHE_DIR", str(shared))
    ps.cascade_reorder(_cands(3), "x", str(tmp_path / "a.jpg"), probe, 0)
    assert len(got) == 3 and len(os.listdir(shared)) == 3
    # Другой эпизод: своя temp_smart, та же общая папка — ни одного скачивания.
    monkeypatch.setattr(ps, "TEMP_FOLDER", str(tmp_path / "other_episode"))
    monkeypatch.setattr(ps, "_CASCADE_EMB", {})
    ps.cascade_reorder(_cands(3), "x", str(tmp_path / "b.jpg"), probe, 0)
    assert len(got) == 3
    assert not [f for f in os.listdir(shared) if f.endswith(".part")]


def test_old_per_episode_cascade_cache_is_still_read(tmp_path, monkeypatch):
    rel = {"c0": 0.1, "c1": 0.2}
    probe, got = _cascade_world(tmp_path, monkeypatch, rel)
    monkeypatch.setenv("CASCADE_CACHE_DIR", str(tmp_path / "shared"))
    old = tmp_path / "temp" / "cascade_embed_cache"
    old.mkdir(parents=True)
    for k in range(2):
        key = ps._cascade_key(ps._cascade_ident(_cands(2)[k], ps.candidate_probe_url(_cands(2)[k])))
        np.save(old / (key + ".npy"), np.array([0.5 + k], "float32"))
    order = ps.cascade_reorder(_cands(2), "x", str(tmp_path / "a.jpg"), probe, 0)
    assert got == [] and [p["id"] for p in order] == ["c1", "c0"]


def test_cascade_hands_over_its_previews_instead_of_deleting(tmp_path, monkeypatch):
    rel = {"c0": 0.1, "c1": 0.9, "c2": 0.3}
    probe, got = _cascade_world(tmp_path, monkeypatch, rel)
    kept = {}
    cands = _cands(3)
    order = ps.cascade_reorder(cands, "x", str(tmp_path / "a.jpg"), probe, 0, keep=kept, keep_top=2)
    assert [p["id"] for p in order] == ["c1", "c2", "c0"]
    assert set(kept) == {id(cands[1]), id(cands[2])}
    assert len([f for f in os.listdir(tmp_path) if "casc_" in f]) == 2
    before = list(got)
    dest = str(tmp_path / "trial.jpg")
    ps.take_kept_preview(kept, cands[1], dest, probe)
    assert got == before and os.path.exists(dest), "превью каскада скачано второй раз"
    ps.take_kept_preview(kept, cands[0], str(tmp_path / "t0.jpg"), probe)
    assert got == before + ["c0"]


def test_cascade_texts_use_the_shared_text_cache(tmp_path, monkeypatch):
    rel = {"c0": 0.1, "c1": 0.2}
    probe, _ = _cascade_world(tmp_path, monkeypatch, rel)
    texts = []
    real = ps._gate_embed

    def counting(images=None, text=None):
        if text is not None:
            texts.append(text)
        return real(images=images, text=text)
    monkeypatch.setattr(ps, "_gate_embed", counting)
    for k in range(3):
        ps.cascade_reorder(_cands(2), ["q1", "q2"], str(tmp_path / f"{k}.jpg"), probe, 0)
    assert texts == ["q1", "q2"]


def test_aesthetic_score_is_cached_by_content(tmp_path, monkeypatch):
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    Image.new("RGB", (16, 16), (40, 50, 60)).save(a)
    b.write_bytes(a.read_bytes())
    calls = []

    class Model:
        def vision_model(self, pixel_values):
            calls.append(1)
            import torch

            class O:
                pooler_output = torch.ones((1, 4))
            return O()

        def visual_projection(self, x):
            return x

    class Proc:
        def __call__(self, images, return_tensors):
            import torch
            return {"pixel_values": torch.zeros((1, 3, 2, 2))}
    monkeypatch.setattr(ps, "get_aesthetic_clip_model", lambda: (Model(), Proc()))
    monkeypatch.setattr(ps, "get_aesthetic_head", lambda: (np.ones(4, "float32"), 1.0))
    monkeypatch.setattr(ps, "AESTHETIC_ENABLED", True)
    monkeypatch.setattr(ps, "CLIP_ENABLED", True)
    s1 = ps.aesthetic_score(str(a))
    s2 = ps.aesthetic_score(str(b))
    assert s1 == s2 == 3.0 and len(calls) == 1
    ps._AESTHETIC_SCORE_CACHE.clear()
    c = tmp_path / "c.jpg"
    c.write_bytes(a.read_bytes())
    assert ps.aesthetic_score(str(c)) == 3.0 and len(calls) == 1, "оценка не взята с диска"


def test_cpu_device_changes_nothing(monkeypatch):
    monkeypatch.setenv("ML_DEVICE", "cpu")
    ml_device.device.cache_clear()
    try:
        assert ml_device.device() == "cpu" and ml_device.tag() == ""
        batch = {"x": object()}
        assert ml_device.inputs(batch) is batch
        m = object()
        assert ml_device.place(m) is m
    finally:
        ml_device.device.cache_clear()


def test_render_workers_run_below_the_selection_priority():
    if not hasattr(os, "nice"):
        return
    code = ("import os,sys; sys.path.insert(0, %r); sys.argv=['p', %r]; "
            "import pipeline_smart as ps; before=os.nice(0); "
            "ps._render_worker_background_priority(); print(os.nice(0) - min(before + 10, 19))"
            % (os.path.join(REPO, "scripts"), REPO))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert out.stdout.strip().splitlines()[-1] == "0", out.stderr[-400:]


# ---------------------------------------------------------------- судья

def test_world_and_claims_are_asked_at_the_same_time(tmp_path, monkeypatch):
    img = tmp_path / "f.jpg"
    Image.new("RGB", (8, 8)).save(img)
    both = threading.Barrier(2, timeout=5)

    def world(*a, **k):
        both.wait()
        return {"main_in_world": True, "background_foreign": False}, {"cost": 1, "call": True}
    real = shot_judge.verify_claims

    def claims(gateway, model, **kw):
        if kw.get("setting") is None:
            both.wait()
            return {"claims": {"c1": "yes"}}, {"cost": 2, "call": True}
        return real(gateway, model, **kw)
    monkeypatch.setattr(shot_judge, "world_of_image", world)
    monkeypatch.setattr(shot_judge, "verify_claims", claims)
    ans, info = real(object(), "m", phrase="p", spec={"claims": []}, setting="medieval",
                     path=str(img), world_separate=True)
    assert ans["main_in_world"] is True and ans["claims"] == {"c1": "yes"}
    assert info["cost"] == 3


def test_gateway_reuses_connections_but_honours_the_recorder(monkeypatch):
    seen = []

    def recorder(req, timeout=None):
        seen.append(req.full_url)
        raise lg.urllib.error.URLError("записано")
    monkeypatch.setattr(lg.urllib.request, "urlopen", recorder)
    try:
        lg.pooled_urlopen(lg.urllib.request.Request("https://example.invalid/x"), timeout=1)
    except lg.urllib.error.URLError:
        pass
    assert seen == ["https://example.invalid/x"], "вызов шлюза прошёл мимо записи сети"
    assert lg.Gateway(api_key="k")._open is lg.pooled_urlopen


def test_gateway_honours_recorder_installed_before_import(tmp_path):
    """Порядок харнесса (selection_freeze.child): запись сети ставится ДО
    импорта шлюза. Снимок urlopen при импорте принимал подмену за оригинал,
    и шлюз шёл в живую сеть мимо записи (прогоны A/B 28.09)."""
    code = ("import sys; sys.path.insert(0, %r); import net_recorder, os; "
            "d=%r; os.makedirs(d, exist_ok=True); open(os.path.join(d,'index.jsonl'),'w').close(); "
            "net_recorder.NetRecorder(d, 'replay').install(); import llm_gateway as lg; "
            "import urllib.error\n"
            "try:\n"
            "    lg.pooled_urlopen(lg.urllib.request.Request('https://example.invalid/x'), timeout=1)\n"
            "except urllib.error.URLError as e:\n"
            "    print('ERR', e)\n"
            % (os.path.join(REPO, "scripts"), str(tmp_path / "net")))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert "freeze: запроса нет в записи" in out.stdout, out.stdout + out.stderr[-600:]


def test_pooled_urlopen_speaks_urllib_errors():
    import http.server

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            code = 429 if self.path == "/busy" else 200
            body = b'{"ok": 1}'
            self.send_response(code)
            if code == 429:
                self.send_header("Retry-After", "3")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    old = {k: os.environ.pop(k, None) for k in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy")}
    try:
        with lg.pooled_urlopen(lg.urllib.request.Request(base + "/ok"), timeout=5) as r:
            assert json.loads(r.read()) == {"ok": 1}
        try:
            lg.pooled_urlopen(lg.urllib.request.Request(base + "/busy"), timeout=5)
            raise AssertionError("429 не стал HTTPError")
        except lg.urllib.error.HTTPError as e:
            assert e.code == 429 and lg.Gateway._retry_after(e, 0) == 3.0
    finally:
        srv.shutdown()
        lg._SESSION = None
        for k, v in old.items():
            if v is not None:
                os.environ[k] = v


def test_caption_screen_keeps_a_paid_answer_when_the_disk_is_full(tmp_path, monkeypatch):
    import caption_screen

    class GW:
        def chat(self, *a, **k):
            return '{"drop": []}', {}, 7
    blocked = tmp_path / "file_not_dir"
    blocked.write_text("x")
    ans, price, cached = caption_screen._ask(GW(), "question", str(blocked / "cache"))
    assert ans == '{"drop": []}' and price == 7 and cached is False


def test_judge_signature_covers_the_grid_prompts():
    src = open(os.path.join(REPO, "scripts", "pipeline_smart.py"), encoding="utf-8").read()
    i = src.index("def shot_judge_signature")
    body = src[i:i + 4000]
    for name in ("shot_judge.PROMPT\n", "shot_judge.PROMPT_VIDEO", "shot_judge.VERIFY_CAPTION",
                 "shot_judge.CLAIMS_VIDEO_NOTE"):
        assert name.strip() in body, name
