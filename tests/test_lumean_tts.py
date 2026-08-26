"""Тесты scripts/lumean_tts.py — Lumean как основной путь озвучки (ЧАСТЬ 13,
Шаг 6). Сеть/ffmpeg — только через monkeypatch (никаких реальных вызовов
к api.lumean.app: ключ в .env реальный, платный, тратить его на тесты
нельзя — тот же принцип, что уже применён к speech_generate.py/ElevenLabs).
Сборка audio/ffprobe-хелперы — за skipif на ffmpeg, как test_smoke.py."""
import json
import os
import shutil
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

sys.argv = ["lumean_tts.py"]
import lumean_tts as lt          # noqa: E402
import script_parser             # noqa: E402


def write_script(tmp_path, body):
    p = tmp_path / "script.txt"
    p.write_text(body, encoding="utf-8")
    return str(p)


SAMPLE_SCRIPT = (
    "=== METADATA === (TITLE/SERIES/EPISODE)\n"
    "=== HOOK === [energetic] Представь: три тысячи лет назад.[pause]"
    "Меч весил больше, чем ты думаешь.\n"
    "=== BLOCK 1: Правда о весе === Меч весил около полутора кг.[pause]"
    "[stat:1,5 КГ]Это удивило многих.[climax]Битва была короткой.\n"
    "=== FINAL === Вот и всё на сегодня.\n"
    "=== PEXELS QUERIES === (HOOK: q1,q2 / BLOCK_1: q3)\n"
)


# ---------- extract_section_texts ----------

def test_extract_section_texts_keeps_only_speakable_sections(tmp_path):
    p = write_script(tmp_path, SAMPLE_SCRIPT)
    names = [name for name, _ in lt.extract_section_texts(p)]
    assert names == ["HOOK", "BLOCK 1: ПРАВДА О ВЕСЕ", "FINAL"]


def test_extract_section_texts_matches_script_parser_section_keys(tmp_path):
    # Критично: section_offsets.json/alignment/NN.csv сопоставляются со
    # скриптом ИМЕННО по этой строке (b["section"] в pipeline_smart.py) —
    # расхождение хотя бы в регистре/пробеле молча ломает весь мэппинг.
    p = write_script(tmp_path, SAMPLE_SCRIPT)
    blocks = script_parser.parse_blocks(p)
    pb_order = []
    for b in blocks:
        if not pb_order or pb_order[-1] != b["section"]:
            pb_order.append(b["section"])
    lt_order = [name for name, _ in lt.extract_section_texts(p)]
    assert lt_order == pb_order


def test_extract_section_texts_keeps_tts_tags_strips_pipeline_only_markers(tmp_path):
    p = write_script(tmp_path, SAMPLE_SCRIPT)
    texts = dict(lt.extract_section_texts(p))
    hook = texts["HOOK"]
    assert "[energetic]" in hook and "[pause]" in hook, "теги ElevenLabs v3 должны дойти до TTS как есть"
    block = texts["BLOCK 1: ПРАВДА О ВЕСЕ"]
    assert "[stat:" not in block, "[stat:...] — экранная плашка, TTS её не должен произносить"
    assert "[climax]" not in block, "[climax] — сигнал для музыки/пауз, не для TTS"
    assert "[pause]" in block, "а вот [pause] в том же блоке остаётся"


def test_extract_section_texts_empty_script_returns_empty(tmp_path):
    p = write_script(tmp_path, "=== METADATA === только служебное\n")
    assert lt.extract_section_texts(p) == []


# ---------- wordcount / length gate ----------

def test_wordcount_report_sums_per_section(tmp_path):
    p = write_script(tmp_path, SAMPLE_SCRIPT)
    total, per_section = lt.wordcount_report(lt.extract_section_texts(p))
    assert total == sum(per_section.values())
    assert set(per_section) == {"HOOK", "BLOCK 1: ПРАВДА О ВЕСЕ", "FINAL"}
    assert all(n > 0 for n in per_section.values())


def test_enforce_length_gate_passes_in_corridor(capsys):
    # T=1 мин -> цель ~125 слов, коридор ~119-134
    assert lt.enforce_length_gate(125, 1.0) is True


def test_enforce_length_gate_blocks_out_of_corridor_without_force(capsys):
    assert lt.enforce_length_gate(10, 1.0) is False
    out = capsys.readouterr().out
    assert "СТОП" in out


def test_enforce_length_gate_force_overrides(capsys):
    assert lt.enforce_length_gate(10, 1.0, force=True) is True
    out = capsys.readouterr().out
    assert "--force-length" in out


def test_enforce_length_gate_no_target_skips_check():
    assert lt.enforce_length_gate(3, None) is True


# ---------- try_parse_alignment (best-effort, схема не подтверждена документацией) ----------

def test_try_parse_alignment_recognizes_elevenlabs_native_shape():
    payload = json.dumps({
        "characters": ["П", "р", "и"],
        "character_start_times_seconds": [0.0, 0.1, 0.2],
        "character_end_times_seconds": [0.1, 0.2, 0.3],
    }).encode("utf-8")
    rows = lt.try_parse_alignment(payload)
    assert rows == [("П", 0.0, 0.1), ("р", 0.1, 0.2), ("и", 0.2, 0.3)]


def test_try_parse_alignment_recognizes_flat_list_shape():
    payload = json.dumps([
        {"character": "A", "start": 0.0, "end": 0.05},
        {"char": "B", "start": 0.05, "end": 0.1},
    ]).encode("utf-8")
    rows = lt.try_parse_alignment(payload)
    assert rows == [("A", 0.0, 0.05), ("B", 0.05, 0.1)]


def test_try_parse_alignment_rejects_mismatched_lengths():
    payload = json.dumps({
        "characters": ["A", "B"],
        "character_start_times_seconds": [0.0],
        "character_end_times_seconds": [0.1, 0.2],
    }).encode("utf-8")
    assert lt.try_parse_alignment(payload) is None


def test_try_parse_alignment_rejects_unknown_shape():
    assert lt.try_parse_alignment(b'{"totally": "unrelated"}') is None
    assert lt.try_parse_alignment(b'"just a string"') is None
    assert lt.try_parse_alignment(b'[]') is None


def test_try_parse_alignment_rejects_garbled_bytes():
    assert lt.try_parse_alignment(b'\xff\xfe not json at all') is None


def test_try_parse_alignment_rejects_bad_types_in_flat_list():
    payload = json.dumps([{"character": "A", "start": "oops", "end": 0.1}]).encode("utf-8")
    assert lt.try_parse_alignment(payload) is None


# ---------- api_call: коды ответов Lumean §12 (мок urllib, без сети) ----------

class _FakeHTTPResponse:
    def __init__(self, status, payload):
        self.status = status
        self._payload = json.dumps(payload).encode("utf-8") if payload is not None else b""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._payload


class _FakeHTTPError(Exception):
    def __init__(self, code, payload):
        self.code = code
        self._payload = json.dumps(payload).encode("utf-8") if payload is not None else b""

    def read(self):
        return self._payload


def _patch_urlopen(monkeypatch, responder):
    """responder(request) -> _FakeHTTPResponse (успех) или ("error", code, payload)
    (ошибка — оборачивается в настоящий urllib.error.HTTPError, чтобы except-ветка
    в _http() сработала по-настоящему, не по выдуманному интерфейсу)."""
    import io
    import urllib.error

    def make_http_error(code, payload):
        body = json.dumps(payload).encode("utf-8") if payload is not None else b""
        return urllib.error.HTTPError("https://api.lumean.app/x", code, "err",
                                      hdrs={}, fp=io.BytesIO(body))

    def fake_urlopen(req, timeout=None):
        result = responder(req)
        if isinstance(result, tuple) and result[0] == "error":
            _, code, payload = result
            raise make_http_error(code, payload)
        return result

    monkeypatch.setattr(lt.urllib.request, "urlopen", fake_urlopen)


def test_api_call_unwraps_success_envelope(monkeypatch):
    # РЕАЛЬНЫЙ баг, пойманный живым вызовом --list-voices: успешный ответ
    # Lumean — конверт {success, message, data}, а не голый payload (§4
    # спеки). Мок повторяет ТОЧНУЮ форму реального ответа (проверено
    # вживую на /voices/elevenlabs/library).
    envelope = {"success": True, "message": "Готово", "data": {"id": "abc"}}
    _patch_urlopen(monkeypatch, lambda req: _FakeHTTPResponse(201, envelope))
    data = lt.api_call("POST", "/orders", "key", body={"x": 1})
    assert data == {"id": "abc"}


def test_api_call_success_with_null_data_returns_none(monkeypatch):
    envelope = {"success": True, "message": "ok", "data": None}
    _patch_urlopen(monkeypatch, lambda req: _FakeHTTPResponse(200, envelope))
    assert lt.api_call("GET", "/orders/x", "key") is None


def test_api_call_payg_required_raises_with_body(monkeypatch):
    body = {"success": False, "message": "нужна доплата", "reason": "payg_topup_required",
            "shortfall_lmc": 3.6, "quote_token": "Q1"}
    _patch_urlopen(monkeypatch, lambda req: ("error", 402, body))
    with pytest.raises(lt.LumeanPaymentRequired) as exc:
        lt.api_call("POST", "/orders", "key")
    assert exc.value.body["shortfall_lmc"] == 3.6


def test_api_call_domain_error_raises_lumean_error(monkeypatch):
    body = {"success": False, "message": "заказ не найден"}
    _patch_urlopen(monkeypatch, lambda req: ("error", 404, body))
    with pytest.raises(lt.LumeanError) as exc:
        lt.api_call("GET", "/orders/nope", "key")
    assert exc.value.status == 404


def test_api_call_token_quota_429_waits_retry_after_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def responder(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return ("error", 429, {"success": False, "message": "квота", "reason": "token_quota_exceeded",
                                   "retry_after": 1})
        return _FakeHTTPResponse(200, {"success": True, "message": "ok", "data": {"ok": True}})
    _patch_urlopen(monkeypatch, responder)
    sleeps = []
    monkeypatch.setattr(lt.time, "sleep", lambda s: sleeps.append(s))
    data = lt.api_call("GET", "/usage", "key")
    assert data == {"ok": True}
    assert sleeps == [1.0]


def test_api_call_plain_rate_limit_429_backs_off_without_retry_after(monkeypatch):
    calls = {"n": 0}

    def responder(req):
        calls["n"] += 1
        if calls["n"] <= 2:
            return ("error", 429, {"success": False, "message": "too many requests"})
        return _FakeHTTPResponse(200, {"success": True, "message": "ok", "data": {"ok": True}})
    _patch_urlopen(monkeypatch, responder)
    sleeps = []
    monkeypatch.setattr(lt.time, "sleep", lambda s: sleeps.append(s))
    data = lt.api_call("GET", "/usage", "key")
    assert data == {"ok": True}
    assert len(sleeps) == 2 and all(s > 0 for s in sleeps)


def test_api_call_429_exhausts_retries_and_raises(monkeypatch):
    _patch_urlopen(monkeypatch, lambda req: ("error", 429, {"success": False, "message": "rl"}))
    monkeypatch.setattr(lt.time, "sleep", lambda s: None)
    with pytest.raises(lt.LumeanError):
        lt.api_call("GET", "/usage", "key", max_429_retries=2)


# ---------- download: атомарность ----------

def test_download_atomic_write(tmp_path, monkeypatch):
    dest = str(tmp_path / "out.mp3")

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"fake-mp3-bytes"
    monkeypatch.setattr(lt.urllib.request, "urlopen", lambda req, timeout=None: _Resp())
    lt.download("https://example.invalid/x.mp3", dest)
    assert os.path.exists(dest)
    assert open(dest, "rb").read() == b"fake-mp3-bytes"
    assert not os.path.exists(dest + ".part")


def test_download_leaves_no_partial_file_on_error(tmp_path, monkeypatch):
    dest = str(tmp_path / "out.mp3")

    def raising(req, timeout=None):
        raise IOError("connection reset")
    monkeypatch.setattr(lt.urllib.request, "urlopen", raising)
    with pytest.raises(Exception):
        lt.download("https://example.invalid/x.mp3", dest)
    assert not os.path.exists(dest)
    assert not os.path.exists(dest + ".part")


def test_download_rejects_empty_response(tmp_path, monkeypatch):
    dest = str(tmp_path / "out.mp3")

    class _Empty:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b""
    monkeypatch.setattr(lt.urllib.request, "urlopen", lambda req, timeout=None: _Empty())
    with pytest.raises(Exception):
        lt.download("https://example.invalid/x.mp3", dest)
    assert not os.path.exists(dest)


# ---------- process_section: оркестрация заказа секции (полностью замоканная) ----------

def test_process_section_happy_path(tmp_path, monkeypatch):
    monkeypatch.setattr(lt, "create_order", lambda *a, **k: {"id": "order-1"})
    monkeypatch.setattr(lt, "poll_until_terminal", lambda *a, **k: {
        "status": "completed",
        "result": {"files": ["storage/1/x.mp3"], "service_files": []},
    })
    monkeypatch.setattr(lt, "storage_url", lambda *a, **k: "https://x/audio.mp3")
    monkeypatch.setattr(lt, "download", lambda url, dest: open(dest, "wb").write(b"a"))

    r = lt.process_section("key", "tmpl", str(tmp_path), 0, "HOOK", "text", "episode")
    assert r["audio_path"] and os.path.exists(r["audio_path"])
    assert r["reason"] is None
    assert r["order_id"] == "order-1"


def test_process_section_payg_required_stops_cleanly(tmp_path, monkeypatch):
    def raise_payg(*a, **k):
        raise lt.LumeanPaymentRequired("нужна доплата", body={"shortfall_lmc": 5.0})
    monkeypatch.setattr(lt, "create_order", raise_payg)

    r = lt.process_section("key", "tmpl", str(tmp_path), 0, "HOOK", "text", "episode")
    assert r["audio_path"] is None
    assert "5.0" in r["reason"] or "PAYG" in r["reason"]


def test_process_section_partially_completed_recovers_via_retry_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(lt, "create_order", lambda *a, **k: {"id": "order-2"})
    calls = {"n": 0}

    def fake_poll(api_key, order_id, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"status": "partially_completed", "items": []}
        return {"status": "completed", "result": {"files": ["storage/1/x.mp3"], "service_files": []}}
    monkeypatch.setattr(lt, "poll_until_terminal", fake_poll)
    monkeypatch.setattr(lt, "retry_failed_items", lambda *a, **k: {"queued_count": 1})
    monkeypatch.setattr(lt, "storage_url", lambda *a, **k: "https://x/audio.mp3")
    monkeypatch.setattr(lt, "download", lambda url, dest: open(dest, "wb").write(b"a"))

    r = lt.process_section("key", "tmpl", str(tmp_path), 0, "HOOK", "text", "episode")
    assert r["audio_path"] is not None
    assert calls["n"] == 2


def test_process_section_policy_flagged_reports_reason_but_still_downloads(tmp_path, monkeypatch):
    monkeypatch.setattr(lt, "create_order", lambda *a, **k: {"id": "order-3"})
    monkeypatch.setattr(lt, "poll_until_terminal", lambda *a, **k: {
        "status": "partially_completed",
        "items": [{"chunk_index": 2, "status": "policy_flagged"}],
        "result": {"files": ["storage/1/x.mp3"], "service_files": []},
    })
    monkeypatch.setattr(lt, "retry_failed_items", lambda *a, **k: {"queued_count": 0})
    monkeypatch.setattr(lt, "storage_url", lambda *a, **k: "https://x/audio.mp3")
    monkeypatch.setattr(lt, "download", lambda url, dest: open(dest, "wb").write(b"a"))

    r = lt.process_section("key", "tmpl", str(tmp_path), 0, "BLOCK 1", "text", "episode")
    assert r["audio_path"] is not None, "частичный результат всё равно скачивается"
    assert "policy_flagged" in r["reason"]


def test_process_section_no_files_reports_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(lt, "create_order", lambda *a, **k: {"id": "order-4"})
    monkeypatch.setattr(lt, "poll_until_terminal", lambda *a, **k: {
        "status": "completed", "result": {"files": [], "service_files": []}})

    r = lt.process_section("key", "tmpl", str(tmp_path), 0, "FINAL", "text", "episode")
    assert r["audio_path"] is None
    assert "files" in r["reason"]


def test_process_section_alignment_downloaded_and_parsed(tmp_path, monkeypatch):
    align_payload = json.dumps({
        "characters": ["A"], "character_start_times_seconds": [0.0],
        "character_end_times_seconds": [0.1],
    }).encode("utf-8")

    monkeypatch.setattr(lt, "create_order", lambda *a, **k: {"id": "order-5"})
    monkeypatch.setattr(lt, "poll_until_terminal", lambda *a, **k: {
        "status": "completed",
        "result": {"files": ["storage/1/x.mp3"],
                   "service_files": ["storage/1/service/alignment.json"]},
    })
    monkeypatch.setattr(lt, "storage_url", lambda api_key, path: f"https://x/{os.path.basename(path)}")

    def fake_download(url, dest):
        if "alignment" in dest:
            open(dest, "wb").write(align_payload)
        else:
            open(dest, "wb").write(b"a")
    monkeypatch.setattr(lt, "download", fake_download)

    r = lt.process_section("key", "tmpl", str(tmp_path), 0, "HOOK", "text", "episode")
    assert r["alignment"] == [("A", 0.0, 0.1)]


# ---------- write_alignment_csv / atomic_write_json ----------

def test_write_alignment_csv_roundtrip(tmp_path):
    align_dir = str(tmp_path / "alignment")
    path = lt.write_alignment_csv(align_dir, 0, [("П", 0.0, 0.1), ("р", 0.1, 0.2)])
    assert os.path.exists(path)
    text = open(path, encoding="utf-8").read()
    assert text.splitlines()[0] == "char,start,end"
    assert "П,0.0,0.1" in text
    assert not os.path.exists(path + ".tmp")


def test_atomic_write_json_roundtrip(tmp_path):
    path = str(tmp_path / "sub" / "report.json")
    lt.atomic_write_json(path, {"a": 1, "b": [1, 2, 3]})
    assert json.load(open(path, encoding="utf-8")) == {"a": 1, "b": [1, 2, 3]}
    assert not os.path.exists(path + ".tmp")


# ---------- persist_env_value ----------

def test_persist_env_value_updates_existing_key(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("FOO=old\nBAR=1\n", encoding="utf-8")
    monkeypatch.setattr(lt, "ENV_PATH", str(env_path))
    assert lt.persist_env_value("FOO", "new") is True
    assert env_path.read_text(encoding="utf-8").splitlines() == ["FOO=new", "BAR=1"]


def test_persist_env_value_appends_missing_key(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("BAR=1\n", encoding="utf-8")
    monkeypatch.setattr(lt, "ENV_PATH", str(env_path))
    assert lt.persist_env_value("LUMEAN_TEMPLATE_ID", "uuid-1") is True
    lines = env_path.read_text(encoding="utf-8").splitlines()
    assert "LUMEAN_TEMPLATE_ID=uuid-1" in lines


def test_persist_env_value_noop_when_env_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(lt, "ENV_PATH", str(tmp_path / "does_not_exist.env"))
    assert lt.persist_env_value("FOO", "bar") is False


# ---------- аудио: только со skipif на ffmpeg (как test_smoke.py) ----------

pytestmark_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe не найдены в PATH",
)


@pytestmark_ffmpeg
def test_concat_audio_and_duration(tmp_path):
    import subprocess
    a = str(tmp_path / "a.mp3")
    b = str(tmp_path / "b.mp3")
    for dest, dur in ((a, 1.0), (b, 1.5)):
        subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"sine=frequency=440:duration={dur}",
                        "-c:a", "libmp3lame", dest], capture_output=True, check=True)
    out = str(tmp_path / "out.mp3")
    lt.concat_audio([a, b], out, str(tmp_path))
    assert os.path.exists(out)
    total = lt.audio_duration(out)
    assert abs(total - 2.5) < 0.3
