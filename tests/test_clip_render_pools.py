# -*- coding: utf-8 -*-
"""Раздатчик клипов (ClipRenderPools): клип наезда — сначала видеокарте,
отказ или сбой карты — тот же вызов процессору; воркеры процессорного пула
карту не трогают никогда (сотня контекстов CUDA не помещается на карту)."""
import concurrent.futures
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import pipeline_smart as ps  # noqa: E402


class _Exec:
    """Синхронный исполнитель с ролью процесса, как у воркера пула."""

    def __init__(self, role, log, broken=False):
        self.role, self.log, self.broken, self.closed = role, log, broken, False

    def submit(self, fn, *a, **k):
        if self.closed:
            raise RuntimeError("пул закрыт")
        self.log.append(self.role)
        f = concurrent.futures.Future()
        prev = ps._WORKER_ROLE[0]
        ps._WORKER_ROLE[0] = self.role
        try:
            if self.broken:
                raise RuntimeError("процесс карты упал")
            f.set_result(fn(*a, **k))
        except Exception as e:  # noqa: BLE001
            f.set_exception(e)
        finally:
            ps._WORKER_ROLE[0] = prev
        return f

    def shutdown(self, wait=True, cancel_futures=False):
        self.closed = True


def _setup(monkeypatch, decline=False):
    def fake_kenburns(photo, *a, **k):
        if ps._WORKER_ROLE[0] == "gpu" and decline:
            return ps.GPU_DECLINED
        return f"{ps._WORKER_ROLE[0]}:{photo}"
    monkeypatch.setattr(ps, "kenburns", fake_kenburns)
    monkeypatch.setattr(ps, "_timed_render", lambda fn, i, *a, **k: fn(*a, **k))


def test_kenburns_goes_to_the_gpu_first(monkeypatch):
    _setup(monkeypatch)
    log = []
    pools = ps.ClipRenderPools(_Exec("cpu", log), _Exec("gpu", log))
    assert pools.submit(ps._timed_render, ps.kenburns, 0, "a.jpg").result() == "gpu:a.jpg"
    assert log == ["gpu"]


def test_declined_clip_is_rendered_by_the_cpu_pool(monkeypatch):
    _setup(monkeypatch, decline=True)
    log = []
    pools = ps.ClipRenderPools(_Exec("cpu", log), _Exec("gpu", log))
    assert pools.submit(ps._timed_render, ps.kenburns, 0, "a.jpg").result() == "cpu:a.jpg"
    assert log == ["gpu", "cpu"]
    assert pools.stats["declined"] == 1


def test_crashed_gpu_pool_is_dropped_and_no_clip_is_lost(monkeypatch):
    _setup(monkeypatch)
    log = []
    pools = ps.ClipRenderPools(_Exec("cpu", log), _Exec("gpu", log, broken=True))
    assert pools.submit(ps._timed_render, ps.kenburns, 0, "a.jpg").result() == "cpu:a.jpg"
    assert pools.gpu is None
    assert pools.submit(ps._timed_render, ps.kenburns, 1, "b.jpg").result() == "cpu:b.jpg"
    assert log == ["gpu", "cpu", "cpu"]


def test_other_jobs_go_straight_to_the_cpu(monkeypatch):
    _setup(monkeypatch)
    log = []
    pools = ps.ClipRenderPools(_Exec("cpu", log), _Exec("gpu", log))
    assert pools.submit(ps._timed_render, lambda p: "video:" + p, 0, "v.mp4").result() == "video:v.mp4"
    assert log == ["cpu"]


def test_cpu_pool_workers_never_touch_the_gpu(monkeypatch):
    monkeypatch.setattr(ps, "_WORKER_ROLE", ["cpu"])
    import gpu_render
    monkeypatch.setattr(gpu_render, "enabled", lambda: True)
    monkeypatch.setattr(gpu_render, "device", lambda: "cuda")
    assert ps.gpu_render_active() is False
    monkeypatch.setattr(ps, "_WORKER_ROLE", ["gpu"])
    assert ps.gpu_render_active() is True


def test_gpu_worker_hands_ineligible_clips_back(monkeypatch, tmp_path):
    """Клип с надписью в воркере карты не рендерится процессором на месте, а
    возвращается раздатчику (процессорных ядер в пуле процессора больше)."""
    monkeypatch.setattr(ps, "_WORKER_ROLE", ["gpu"])
    for name, val in (("gpu_render_active", lambda: True), ("focus_crop", lambda p: p),
                      ("aspect_fit_backdrop", lambda p: p), ("estimate_busyness", lambda p: 0.0),
                      ("resolve_crop_anchor", lambda p: None), ("image_size_as_rendered", lambda p: (1600, 900))):
        monkeypatch.setattr(ps, name, val)
    called = []
    monkeypatch.setattr(ps, "run_ffmpeg_with_retry", lambda *a, **k: (called.append(1), (True, ""))[1])
    r = ps.kenburns("p.jpg", str(tmp_path / "c.mp4"), 1.0, section="BLOCK_1", title="Надпись")
    assert r == ps.GPU_DECLINED and not called


def test_gpu_render_workers_reads_only_existing_flags(monkeypatch):
    """29.09: gpu_render_workers() спрашивала флаг CASCADE_MODEL, которого в
    реестре GPU-ветки нет, — рендер упал бы при создании пула карты. Тест
    зовёт функцию с подменённой картой: без карты в контейнере эта строка не
    выполнялась ни одним тестом."""
    import types
    monkeypatch.delenv("GPU_RENDER_WORKERS", raising=False)
    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(
        get_device_properties=lambda i: types.SimpleNamespace(total_memory=96 * 2 ** 30)))
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert ps.gpu_render_workers() == 4
    fake.cuda.get_device_properties = lambda i: types.SimpleNamespace(total_memory=24 * 2 ** 30)
    assert ps.gpu_render_workers() == 1
    monkeypatch.setenv("GPU_RENDER_WORKERS", "2")
    assert ps.gpu_render_workers() == 2


def test_gpu_render_failure_never_escapes(monkeypatch):
    import gpu_render
    monkeypatch.setattr(gpu_render, "_render_kenburns", lambda *a, **k: (_ for _ in ()).throw(MemoryError("карта")))
    ok, why = gpu_render.render_kenburns("p.jpg", "o.mp4", 3, "", "", "", (1, 1, 1, 1, 0, 0), "", [])
    assert ok is False and "MemoryError" in why
