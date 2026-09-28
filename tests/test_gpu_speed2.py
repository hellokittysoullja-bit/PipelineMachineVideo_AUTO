#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Скорость на сильной машине без смены результата: куски финальной склейки
кодируются одновременно, декодеры входов не раздувают память, модели
грузятся в фоне и без гонки между потоками упреждения."""
import os
import sys
import threading
import time
import types

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))


def _ps():
    import pipeline_smart as ps
    return ps


class _Done:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


# ---------- финальная склейка ----------

def test_chunk_workers_follow_cores_memory_and_env(monkeypatch):
    ps = _ps()
    monkeypatch.delenv("FINAL_CHUNK_WORKERS", raising=False)
    assert ps.final_chunk_workers(1) == 1
    monkeypatch.setattr(ps.os, "cpu_count", lambda: 4)
    assert ps.final_chunk_workers(5) == 1, "на четырёх ядрах куски идут по очереди, как раньше"
    monkeypatch.setattr(ps.os, "cpu_count", lambda: 32)
    assert 1 <= ps.final_chunk_workers(5) <= 4
    monkeypatch.setenv("FINAL_CHUNK_WORKERS", "3")
    assert ps.final_chunk_workers(5) == 3 and ps.final_chunk_workers(2) == 2


def test_chunks_are_encoded_at_the_same_time_and_glued_in_order(monkeypatch, tmp_path):
    ps = _ps()
    monkeypatch.setenv("FINAL_CHUNK_WORKERS", "4")
    active, peak, seen = [0], [0], []
    lock = threading.Lock()

    def fake_xfade(clips, durs, sections, out, xfade_dur=None, blocks=None, plan=None):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.2)
        with lock:
            active[0] -= 1
            seen.append(out)
        open(out, "wb").write(b"x")
        return True, float(len(clips))
    monkeypatch.setattr(ps, "xfade_chain", fake_xfade)
    lists = []

    def fake_run(cmd, **_kw):
        lists.append(open(cmd[cmd.index("-i") + 1], encoding="utf-8").read())
        return _Done(0)
    monkeypatch.setattr(ps.subprocess, "run", fake_run)
    n = 120
    sections = ["HOOK"] * 10 + [f"B{k // 22}" for k in range(n - 10)]
    ok, total = ps.xfade_chain_chunked([f"c{i}.mp4" for i in range(n)], [1.0] * n, sections,
                                       str(tmp_path / "out.mp4"), str(tmp_path))
    assert ok and total == float(n)
    assert peak[0] > 1, "куски не кодировались одновременно"
    order = [line.split("'")[1] for line in lists[0].splitlines()]
    assert order == sorted(order), "склейка идёт в порядке кусков, а не завершения"


def test_failed_chunk_still_falls_back_and_cleans_up(monkeypatch, tmp_path):
    ps = _ps()
    monkeypatch.setenv("FINAL_CHUNK_WORKERS", "4")

    def fake_xfade(clips, durs, sections, out, xfade_dur=None, blocks=None, plan=None):
        open(out, "wb").write(b"x")
        return (not out.endswith("_001.mp4")), 1.0
    monkeypatch.setattr(ps, "xfade_chain", fake_xfade)
    monkeypatch.setattr(ps.subprocess, "run", lambda *a, **k: pytest.fail("склейка после сбоя"))
    n = 120
    sections = ["HOOK"] * 10 + [f"B{k // 22}" for k in range(n - 10)]
    ok, total = ps.xfade_chain_chunked([f"c{i}.mp4" for i in range(n)], [1.0] * n, sections,
                                       str(tmp_path / "out.mp4"), str(tmp_path))
    assert (ok, total) == (False, 0.0)
    assert not list(tmp_path.glob("_xchunk_*.mp4")), "куски сбойной склейки не убраны"


def test_every_input_of_the_final_pass_gets_bounded_decoder_threads(monkeypatch, tmp_path):
    ps = _ps()
    cmds = []

    def fake_run(cmd, **_kw):
        cmds.append(cmd)
        return _Done(0)
    monkeypatch.setattr(ps.subprocess, "run", fake_run)
    monkeypatch.setattr(ps, "get_media_duration", lambda p: 3.0)
    ps.xfade_chain(["a.mp4", "b.mp4", "c.mp4"], [1.2, 1.2, 1.2], ["A"] * 3, str(tmp_path / "o.mp4"))
    cmd = cmds[0]
    inputs = [k for k, a in enumerate(cmd) if a == "-i"]
    assert len(inputs) == 3
    for k in inputs:
        assert cmd[k - 2:k] == ["-threads", str(ps.XFADE_INPUT_DECODE_THREADS)]


# ---------- загрузка моделей ----------

def _slow_pretrained(counter, delay=0.3):
    class Slow:
        @classmethod
        def from_pretrained(cls, *a, **k):
            counter.append(cls.__name__)
            time.sleep(delay)
            obj = cls()
            obj.eval = lambda: obj
            return obj
    return Slow


def _fake_transformers(monkeypatch, **classes):
    """Ленивый модуль transformers отдаёт настоящие классы мимо setattr —
    подменяется сам модуль, чтобы тест не тянул веса из сети."""
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(**classes))


def _race(fn, n=4):
    out, errs = [], []

    def run():
        try:
            out.append(fn())
        except Exception as e:  # noqa: BLE001
            errs.append(e)
    ts = [threading.Thread(target=run) for _ in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errs, errs
    return out


def test_aesthetic_model_loads_once_and_never_without_processor(monkeypatch):
    ps = _ps()
    loads = []
    _fake_transformers(monkeypatch,
                       CLIPModel=type("CLIPModel", (_slow_pretrained(loads),), {}),
                       CLIPProcessor=type("CLIPProcessor", (_slow_pretrained(loads, 0.5),), {}))
    import ml_device
    monkeypatch.setattr(ml_device, "place", lambda m: m)
    monkeypatch.setattr(ps, "_aesthetic_clip_model", None)
    monkeypatch.setattr(ps, "_aesthetic_clip_processor", None)
    got = _race(ps.get_aesthetic_clip_model)
    assert all(m is not None and p is not None for m, p in got), "поток получил модель без процессора"
    assert loads.count("CLIPModel") == 1


def test_so400m_loads_once_and_never_without_processor(monkeypatch):
    import visual_director as vd
    loads = []
    _fake_transformers(monkeypatch,
                       AutoModel=type("AutoModel", (_slow_pretrained(loads),), {}),
                       AutoProcessor=type("AutoProcessor", (_slow_pretrained(loads, 0.5),), {}))
    import ml_device
    monkeypatch.setattr(ml_device, "place", lambda m: m)
    monkeypatch.setattr(vd, "_siglip2_model", None)
    monkeypatch.setattr(vd, "_siglip2_processor", None)
    got = _race(vd._get_siglip2_model)
    assert all(m is not None and p is not None for m, p in got)
    assert loads.count("AutoModel") == 1


def test_jina_session_loads_once_and_never_without_tokenizer(monkeypatch):
    import visual_director as vd
    loads = []
    fake_ort = types.SimpleNamespace(
        SessionOptions=lambda: types.SimpleNamespace(),
        InferenceSession=lambda *a, **k: (loads.append("session"), time.sleep(0.2), object())[2])
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        types.SimpleNamespace(hf_hub_download=lambda **k: "model.onnx"))
    _fake_transformers(monkeypatch,
                       AutoTokenizer=type("AutoTokenizer", (_slow_pretrained(loads, 0.4),), {}))
    monkeypatch.setattr(vd, "jina_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(vd, "_jina_session", None)
    monkeypatch.setattr(vd, "_jina_tokenizer", None)
    got = _race(vd._get_jina_session)
    assert all(s is not None and t is not None for s, t in got)
    assert loads.count("session") == 1


def test_warmup_loads_in_background_and_swallows_failures(monkeypatch):
    ps = _ps()
    calls = []

    def ok():
        calls.append("ok")

    def bad():
        calls.append("bad")
        raise RuntimeError("нет весов")
    monkeypatch.setenv("MODEL_WARMUP", "1")
    monkeypatch.setattr(ps, "model_warmup_jobs", lambda: [("a", bad), ("b", ok)])
    broken_before = ps.CLIP_BROKEN
    t = ps.start_model_warmup()
    t.join(5)
    assert calls == ["bad", "ok"], "сбой одной модели не мешает следующей"
    assert ps.CLIP_BROKEN == broken_before, "прогрев ничего не помечает сломанным"
    monkeypatch.setenv("MODEL_WARMUP", "0")
    assert ps.start_model_warmup() is None


def test_warmup_loads_only_models_the_run_uses(monkeypatch):
    ps = _ps()
    monkeypatch.setattr(ps, "CLIP_ENABLED", True)
    monkeypatch.setattr(ps, "AESTHETIC_ENABLED", False)
    monkeypatch.setenv("CASCADE_MODEL", "siglip2")
    monkeypatch.setenv("SMART_RELEVANCE_VETO", "0")
    monkeypatch.setenv("VISUAL_DIRECTOR_MODE", "off")
    monkeypatch.setattr(ps, "PARALLAX_ENABLED", False)
    names = [n for n, _ in ps.model_warmup_jobs()]
    assert names == ["SigLIP2 гейта"]
    monkeypatch.setattr(ps, "PARALLAX_ENABLED", True)
    assert [n for n, _ in ps.model_warmup_jobs()] == ["SigLIP2 гейта", "Depth-Anything"]


def test_main_starts_warmup_before_planning():
    ps = _ps()
    import inspect
    src = inspect.getsource(ps.main)
    assert src.index("start_model_warmup()") < src.index("auto_plan_episode(blocks)")


# ---------- звук рядом со склейкой видео ----------

def test_audio_is_mastered_alongside_the_final_video_pass():
    """Звук стартует до склейки видео и ждётся там, где раньше считался."""
    ps = _ps()
    import inspect
    src = inspect.getsource(ps.main)
    start = src.index("audio_pool.submit(")
    assert start < src.index("xfade_chain_chunked(clips"), "звук стартует до склейки"
    assert src.index("audio_future.result()") < src.index("build_master_af(loud_stats")
    assert "process_voice(AUDIO_FILE" not in src and "= measure_loudnorm_stats(" not in src, \
        "звуковая цепочка должна жить в одной функции, без второй копии в main"
    # Каждый ранний выход после старта дожидается звука, а не бросает поток.
    tail = src[start:src.index("audio_future.result()")]
    assert tail.count("return 1") == tail.count("_stop_audio()") - tail.count("def _stop_audio()")


def test_master_audio_premix_runs_the_same_chain(monkeypatch):
    ps = _ps()
    calls = []
    monkeypatch.setattr(ps, "process_voice", lambda a, out: (calls.append("voice"), out)[1])

    def layers(voice, *a, **k):
        calls.append(("layers", k.get("phrase_locked")))
        return "premix.wav"
    monkeypatch.setattr(ps, "build_episode_audio_layers", layers)
    monkeypatch.setattr(ps, "measure_loudnorm_stats", lambda p: (calls.append(("loud", p)), {"i": 1})[1])
    premix, stats = ps.master_audio_premix([], [], [], 10.0, 1.0, 9.0, (), (), True)
    assert (premix, stats) == ("premix.wav", {"i": 1})
    assert calls == ["voice", ("layers", True), ("loud", "premix.wav")]


# ---------- одновременные вопросы упреждения к шлюзу ----------

def test_adaptive_limit_grows_on_success_and_halves_on_429():
    import llm_gateway as lg
    lim = lg.AdaptiveLimit(start=4, maximum=10)
    for _ in range(4):
        lim.acquire()
        lim.release(ok=True)
    assert lim.limit == 5, "после серии без отказа лимит растёт на один"
    lim.throttled()
    assert lim.limit == 2 and lim.stats["throttles"] == 1
    for _ in range(200):
        lim.acquire()
        lim.release(ok=True)
    assert lim.limit == 10, "не выше потолка"


def test_adaptive_limit_bounds_concurrency():
    import llm_gateway as lg
    lim = lg.AdaptiveLimit(start=3, maximum=3)
    active, peak = [0], [0]
    guard = threading.Lock()

    def work():
        lim.acquire()
        with guard:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        with guard:
            active[0] -= 1
        lim.release()
    ts = [threading.Thread(target=work) for _ in range(12)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert peak[0] == 3


def _q(text):
    return [{"type": "text", "text": text}]


def _gateway_with_fake_net(monkeypatch, delay=0.05, fail_429=0):
    import llm_gateway as lg
    monkeypatch.setattr(lg, "_SPEC_LIMITERS", {})
    gw = lg.Gateway(api_key="k", base_url="https://gw.test/v1", spend_cap=10 ** 9)
    monkeypatch.setattr(gw, "cost", lambda model, p, c: 10)
    state = {"active": 0, "peak": 0, "n429": fail_429}
    lock = threading.Lock()

    def fake_request(method, path, body=None, timeout=120, on_lost_body=None):
        with lock:
            if state["n429"] > 0:
                state["n429"] -= 1
                gw.spec_limiter().throttled()
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        time.sleep(delay)
        with lock:
            state["active"] -= 1
        q = body["messages"][0]["content"][0]["text"]
        return {"choices": [{"message": {"content": "ok " + q}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    monkeypatch.setattr(gw, "_request", fake_request)
    return gw, state


def test_speculative_calls_respect_the_limit_real_calls_do_not(monkeypatch):
    import llm_gateway as lg
    gw, state = _gateway_with_fake_net(monkeypatch)
    lim = gw.spec_limiter()
    lim.limit = lim.maximum = 2

    def spec(k):
        with lg.speculation():
            return gw.chat("m", _q(f"q{k}"), 5, 10)
    spec("warm")          # первый вызов модели узнаёт цену в одиночку
    ts = [threading.Thread(target=spec, args=(k,)) for k in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert state["peak"] == 2, "упреждение не выходит за лимит"
    # Настоящий вызов того же вопроса берёт готовый ответ, не спрашивая сеть.
    before = gw.spec_used
    text, _u, _p = gw.chat("m", _q("q3"), 5, 10)
    assert text == "ok q3" and gw.spec_used == before + 1
    # Настоящий вызов нового вопроса идёт мимо ограничителя.
    state["peak"] = 0
    lim.acquire(), lim.acquire()   # лимит занят целиком
    try:
        assert gw.chat("m", _q("fresh"), 5, 10)[0] == "ok fresh"
    finally:
        lim.release(), lim.release()


def test_a_real_429_halves_the_speculative_limit(monkeypatch):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import llm_gateway as lg
    from test_llm_gateway import Opener, http_error, ok
    import source_health
    monkeypatch.setattr(lg.time, "sleep", lambda s: None)
    source_health.reset_all()
    monkeypatch.setattr(lg, "_SPEC_LIMITERS", {})
    gw = lg.Gateway(api_key="k", opener=Opener([http_error(429, headers={"Retry-After": "1"}),
                                                ok("fine")]))
    gw.spec_limiter().limit = 8
    assert gw.chat("m/vision", _q("x"), 50, 1000)[0] == "fine"
    assert gw.spec_limiter().limit == 4 and gw.spec_limiter().stats["throttles"] == 1


# ---------- параллакс в фоне ----------

class _Pool:
    def __init__(self):
        self.calls = []

    def submit(self, fn, *a, **k):
        self.calls.append((fn, a, k))
        import concurrent.futures
        f = concurrent.futures.Future()
        f.set_result("kb-ok")
        return f


_KW = dict(title=None, zoom_in=True, pan_dir=(1, 0), stat=None, section="HOOK", stat_variant=0,
           brightness_bias=0.0, energy_bias=0.0, stat_delay=0.0, levels=None, wb=None,
           grain_scale=1.0, captions=None, look_filter=None, domain=None)


def test_highlight_clip_uses_parallax_when_it_works(monkeypatch):
    ps = _ps()
    monkeypatch.setattr(ps, "_timed_render", lambda fn, i, *a, **k: fn is ps.parallax_kenburns)
    pool = _Pool()
    assert ps.render_highlight_clip(pool, 3, "p.jpg", "o.mp4", 4.0, "classic_kb", "слом", dict(_KW))
    assert pool.calls == []


def test_highlight_clip_falls_back_exactly_like_the_loop_did(monkeypatch):
    ps = _ps()
    monkeypatch.setattr(ps, "_timed_render", lambda fn, *a, **k: False)
    monkeypatch.setattr(ps, "CAMERA_LANGUAGE_STATS",
                        {"modes": {}, "with_stage": 0, "without_stage": 0})
    pool = _Pool()
    got = ps.render_highlight_clip(pool, 3, "p.jpg", "o.mp4", 4.0, "classic_kb", "слом", dict(_KW))
    assert got == "kb-ok"
    fn, a, k = pool.calls[0]
    assert fn is ps._timed_render and a[:5] == (ps.kenburns, 3, "p.jpg", "o.mp4", 4.0)
    assert k["motion_mode"] == "classic_kb" and k["ffmpeg_threads"] == ps.RENDER_FFMPEG_THREADS
    assert {key: k[key] for key in _KW} == _KW, "те же аргументы, что у отката в цикле"
    assert ps.CAMERA_LANGUAGE_STATS == {"modes": {"classic_kb": 1}, "with_stage": 1,
                                        "without_stage": 0}


def test_loop_hands_highlights_to_the_background_thread():
    ps = _ps()
    import inspect
    src = inspect.getsource(ps.main)
    assert "highlight_pool.submit(" in src and "render_highlight_clip" in src
    assert src.index("highlight_pool.shutdown(wait=True)") < src.index("render_pool.shutdown(wait=True)"), \
        "фоновый поток отдаёт откат в пул процессов — пул закрывается после него"


def test_depth_model_goes_to_the_gpu_only_when_there_is_one(monkeypatch):
    ps = _ps()
    import ml_device
    made = []
    _fake_transformers(monkeypatch, pipeline=lambda **k: made.append(k) or object())
    for dev, expect in (("cpu", {}), ("cuda", {"device": "cuda"})):
        monkeypatch.setattr(ml_device, "device", lambda d=dev: d)
        monkeypatch.setattr(ps, "_depth_model", None)
        ps.get_depth_model()
        extra = {k: v for k, v in made[-1].items() if k not in ("task", "model")}
        assert extra == expect
