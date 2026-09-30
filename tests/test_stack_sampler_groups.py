"""Семплер: главный поток виден в группах без номеров строк; журнал процессора пишется."""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import stack_sampler as ss  # noqa: E402


def test_groups_merge_line_numbers_and_thread_numbers(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "SAMPLE_SEC", 0.01)
    monkeypatch.setattr(ss, "FLUSH_SEC", 0.05)
    out = tmp_path / "prof.txt"
    stop = ss.start(str(out))

    def busy():
        end = time.time() + 0.4
        while time.time() < end:
            x = 0
            for i in range(2000):
                x += i
            y = x  # noqa: F841 — другая строка той же функции

    ts = [threading.Thread(target=busy, name=f"prefetch_{i}") for i in range(3)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    stop()
    time.sleep(0.05)
    text = out.read_text(encoding="utf-8")
    assert "ПО ГРУППАМ ПОТОКОВ" in text
    assert "## prefetch:" in text and "prefetch_0" not in text.split("ПО ГРУППАМ")[1]
    assert "busy" in text.split("ПО ГРУППАМ")[1]


def test_cpu_log_has_header_and_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "CPU_SEC", 0.05)
    out = tmp_path / "cpu.csv"
    stop = ss.start_cpu_log(str(out))
    time.sleep(0.4)
    stop()
    time.sleep(0.1)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("timestamp,cpu_busy_pct")
    if ss._cpu_times() is not None:   # Linux
        assert len(lines) >= 3
        assert 0.0 <= float(lines[1].split(",")[1]) <= 100.0
