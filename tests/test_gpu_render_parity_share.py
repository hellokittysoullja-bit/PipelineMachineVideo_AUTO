"""Отказ самопроверки GPU-рендера делится между процессами одного прогона."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import gpu_render as gr  # noqa: E402


def test_flag_is_shared_by_run_id_only(tmp_path, monkeypatch):
    out = str(tmp_path / "clip_0001.mp4")
    monkeypatch.delenv("GPU_PARITY_RUN", raising=False)
    gr._parity_share_disabled(out, "почему")
    assert gr._parity_shared_disabled(out) is None, "без идентификатора прогона общего состояния нет"
    monkeypatch.setenv("GPU_PARITY_RUN", "runA")
    assert gr._parity_shared_disabled(out) is None
    gr._parity_share_disabled(out, "яркость 31.9 дБ")
    assert gr._parity_shared_disabled(out) == "яркость 31.9 дБ"
    monkeypatch.setenv("GPU_PARITY_RUN", "runB")
    assert gr._parity_shared_disabled(out) is None, "файл другого прогона не выключает карту"


def test_render_kenburns_stops_on_shared_flag(tmp_path, monkeypatch):
    out = str(tmp_path / "clip_0002.mp4")
    monkeypatch.setenv("GPU_PARITY_RUN", "runC")
    gr._parity_share_disabled(out, "чужая причина")
    monkeypatch.setattr(gr, "device", lambda: "cuda")
    monkeypatch.setitem(gr._PARITY, "disabled", None)
    ok, why = gr._render_kenburns("p.jpg", out, 10, "1", "0", "0", (1, 1, 1, 1, 0, 0), "", [])
    assert not ok and why == "чужая причина"
    assert gr._PARITY["disabled"] == "чужая причина"
    gr._PARITY["disabled"] = None
