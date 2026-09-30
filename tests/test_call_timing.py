"""Замер вызовов шлюза и очереди к карте (пункты B4 и B1 прогона 30.09):
строки gateway_call и gpu_run в stage_timings, на ответы не влияют."""
import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import llm_gateway as lg  # noqa: E402
import ml_device  # noqa: E402
import source_health  # noqa: E402
import stage_timer  # noqa: E402
from test_llm_gateway import Opener, _img_answer, http_error, no_sleep, ok  # noqa: E402,F401


@pytest.fixture
def timings(tmp_path, monkeypatch):
    path = tmp_path / "stage_timings.jsonl"
    monkeypatch.setattr(stage_timer, "STAGE_TIMER_ENABLED", True)
    monkeypatch.setattr(stage_timer, "_output_path", str(path))
    monkeypatch.setattr(stage_timer, "SUPPRESS", None)

    def read(stage):
        if not path.exists():
            return []
        return [r for r in map(json.loads, path.read_text().splitlines()) if r["stage"] == stage]
    return read


def ask_judge(gw):
    return gw.chat("m/vision", [{"type": "text", "text": "q"}], 50, 1000)


def test_chat_call_is_timed_with_caller_model_tokens_price(timings):
    gw = lg.Gateway(api_key="k", opener=Opener([ok("A", pt=100, ct=10)]))
    text, _u, price = ask_judge(gw)
    [rec] = timings("gateway_call")
    assert text == "A"
    assert rec["kind"] == "chat" and rec["model"] == "m/vision" and rec["ok"] is True
    assert rec["caller"].endswith(".ask_judge")
    assert rec["prompt_tokens"] == 100 and rec["completion_tokens"] == 10 and rec["price"] == price
    assert rec["from_spec"] is False and rec["t_wall"] >= 0


def test_failed_chat_is_timed_as_failed_and_still_raises(timings):
    gw = lg.Gateway(api_key="k", opener=Opener([http_error(400, {"error": "bad"})]))
    with pytest.raises(lg.GatewayError):
        ask_judge(gw)
    [rec] = timings("gateway_call")
    assert rec["ok"] is False and "GatewayError" in rec["error"]


def test_answer_from_speculation_is_marked(timings):
    gw = lg.Gateway(api_key="k", opener=Opener([ok("A")]))
    stage_timer.SUPPRESS = lambda: lg.speculative()
    with lg.speculation():
        ask_judge(gw)
    assert timings("gateway_call") == [], "упреждение в разрез прогона не входит"
    ask_judge(gw)
    [rec] = timings("gateway_call")
    assert rec["from_spec"] is True and rec["ok"] is True


def test_image_call_is_timed(timings):
    gw = lg.Gateway(api_key="k", opener=Opener([_img_answer(2)]))
    images, price = gw.image("m/img", "p", "1792x1024", n=2)
    [rec] = timings("gateway_call")
    assert rec["kind"] == "image" and rec["got"] == 2 and rec["price"] == price and len(images) == 2


def test_timing_off_writes_nothing_and_answers_the_same(tmp_path, monkeypatch):
    monkeypatch.setattr(stage_timer, "STAGE_TIMER_ENABLED", False)
    monkeypatch.setattr(stage_timer, "_output_path", str(tmp_path / "t.jsonl"))
    gw = lg.Gateway(api_key="k", opener=Opener([ok("A", pt=100, ct=10)]))
    assert ask_judge(gw)[0] == "A"
    assert not (tmp_path / "t.jsonl").exists()


def _fake_cuda(monkeypatch):
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
        OutOfMemoryError=MemoryError, empty_cache=lambda: None,
        mem_get_info=lambda i: (100 * 2 ** 30, 100 * 2 ** 30), current_device=lambda: 0))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(ml_device, "device", lambda: "cuda")


def test_gpu_run_records_wait_and_background_flag(timings, monkeypatch):
    _fake_cuda(monkeypatch)
    monkeypatch.setattr(ml_device, "GPU_RUN_LOG_WAIT_SEC", 0.0)
    monkeypatch.setattr(ml_device, "_RUN_TOTALS", dict.fromkeys(ml_device._RUN_TOTALS, 0))
    assert ml_device.run(lambda: 7) == 7
    tok = source_health.BACKGROUND.set(True)
    try:
        assert ml_device.run(lambda: 8, "cuda:0") == 8
    finally:
        source_health.BACKGROUND.reset(tok)
    fg, bg = timings("gpu_run")
    assert fg["background"] is False and bg["background"] is True
    assert fg["wait"] >= 0 and fg["dev"] == "cuda:0"
    assert bg["totals"]["n_fg"] == 1 and bg["totals"]["n_bg"] == 1


def test_fast_gpu_runs_are_counted_but_not_written(timings, monkeypatch):
    _fake_cuda(monkeypatch)
    monkeypatch.setattr(ml_device, "_RUN_TOTALS", dict.fromkeys(ml_device._RUN_TOTALS, 0))
    for _ in range(50):
        ml_device.run(lambda: 1)
    assert timings("gpu_run") == [], "быстрые прогоны не забивают журнал"
    assert ml_device._RUN_TOTALS["n_fg"] == 50
    assert not ml_device._lock("cuda:0").locked()


def test_gpu_run_releases_the_card_on_error(monkeypatch):
    _fake_cuda(monkeypatch)

    def boom():
        raise ValueError("x")
    with pytest.raises(ValueError):
        ml_device.run(boom)
    assert not ml_device._lock("cuda:0").locked()
