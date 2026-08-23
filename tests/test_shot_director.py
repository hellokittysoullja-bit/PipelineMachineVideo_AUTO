"""LLM-режиссёр (scripts/shot_director.py) — тесты.

Живой Gemini НЕ вызывается нигде в этом файле (нет ключа в CI, и бюджет
free-tier — по факту ~20 вызовов/день, см. докстринг shot_director.py —
слишком дорог, чтобы тратить его на тесты). urllib.request.urlopen
подменяется тем же паттерном, что test_speech_generate.py использует для
ElevenLabs: без сети, чистый контракт запроса/ответа/кэша/лимита."""
import json
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

import shot_director as sd  # noqa: E402


class _FakeHTTPResponse:
    def __init__(self, payload_bytes):
        self._payload = payload_bytes

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._payload


def _fake_gemini_payload(queries, literal=False):
    inner = json.dumps({"literal": literal, "queries": queries})
    return json.dumps({
        "candidates": [{"content": {"parts": [{"text": inner}]}}]
    }).encode("utf-8")


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    sd.reset_call_counter()
    monkeypatch.setenv("SHOT_DIRECTOR_MODE", "on")
    monkeypatch.setenv("GEMINI_API_KEY", "fake_key_for_test")
    yield
    sd.reset_call_counter()


def test_off_mode_never_touches_network(monkeypatch, tmp_path):
    monkeypatch.setenv("SHOT_DIRECTOR_MODE", "off")

    def _boom(*a, **kw):
        raise AssertionError("network должен быть недостижим в off-режиме")
    monkeypatch.setattr(sd.urllib.request, "urlopen", _boom)
    assert sd.direct_query("Взять быка за рога.", str(tmp_path)) is None


def test_no_api_key_returns_none_without_network(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    def _boom(*a, **kw):
        raise AssertionError("не должно быть сетевого вызова без ключа")
    monkeypatch.setattr(sd.urllib.request, "urlopen", _boom)
    assert sd.direct_query("Взять быка за рога.", str(tmp_path)) is None


def test_successful_call_returns_first_query_and_writes_cache(monkeypatch, tmp_path):
    payload = _fake_gemini_payload(["person taking decisive action", "confident leader close up"])
    monkeypatch.setattr(sd.urllib.request, "urlopen",
                         lambda req, timeout=None: _FakeHTTPResponse(payload))
    result = sd.direct_query("Взять быка за рога.", str(tmp_path))
    assert result == "person taking decisive action"
    cache_file = sd._cache_path(str(tmp_path), "Взять быка за рога.")
    assert os.path.exists(cache_file)
    cached = json.load(open(cache_file, encoding="utf-8"))
    assert cached["queries"][0] == "person taking decisive action"


def test_cache_hit_skips_network_entirely(monkeypatch, tmp_path):
    payload = _fake_gemini_payload(["decisive person action"])
    monkeypatch.setattr(sd.urllib.request, "urlopen",
                         lambda req, timeout=None: _FakeHTTPResponse(payload))
    first = sd.direct_query("Взять быка за рога.", str(tmp_path))

    def _boom(*a, **kw):
        raise AssertionError("кэш-хит не должен трогать сеть")
    monkeypatch.setattr(sd.urllib.request, "urlopen", _boom)
    second = sd.direct_query("Взять быка за рога.", str(tmp_path))
    assert first == second == "decisive person action"


def test_cyrillic_query_is_filtered_out(monkeypatch, tmp_path):
    # Модель иногда всё равно возвращает кириллицу вопреки запрету в промпте —
    # такой запрос бесполезен для Pexels/Pixabay (англоязычный поиск), должен
    # быть отфильтрован, а не уйти в реальный fetch.
    payload = _fake_gemini_payload(["меч в руке", "person taking decisive action"])
    monkeypatch.setattr(sd.urllib.request, "urlopen",
                         lambda req, timeout=None: _FakeHTTPResponse(payload))
    result = sd.direct_query("Взять быка за рога.", str(tmp_path))
    assert result == "person taking decisive action"


def test_all_queries_cyrillic_returns_none(monkeypatch, tmp_path):
    payload = _fake_gemini_payload(["меч в руке"])
    monkeypatch.setattr(sd.urllib.request, "urlopen",
                         lambda req, timeout=None: _FakeHTTPResponse(payload))
    assert sd.direct_query("Взять быка за рога.", str(tmp_path)) is None


def test_malformed_json_response_fails_open(monkeypatch, tmp_path):
    bad_payload = json.dumps({
        "candidates": [{"content": {"parts": [{"text": "не json вообще"}]}}]
    }).encode("utf-8")
    monkeypatch.setattr(sd.urllib.request, "urlopen",
                         lambda req, timeout=None: _FakeHTTPResponse(bad_payload))
    assert sd.direct_query("Абстрактная фраза.", str(tmp_path)) is None


def test_network_error_fails_open(monkeypatch, tmp_path):
    def _raise(req, timeout=None):
        raise OSError("timeout")
    monkeypatch.setattr(sd.urllib.request, "urlopen", _raise)
    assert sd.direct_query("Абстрактная фраза.", str(tmp_path)) is None


def test_json_wrapped_in_markdown_fence_is_still_parsed(monkeypatch, tmp_path):
    # Промпт явно запрещает markdown-обёртку, но responseMimeType=json не
    # железная гарантия для всех моделей/версий — парсер должен выживать,
    # если модель всё равно обернула ответ в ```json ... ```.
    inner = json.dumps({"literal": False, "queries": ["decisive person action"]})
    wrapped = f"```json\n{inner}\n```"
    payload = json.dumps({
        "candidates": [{"content": {"parts": [{"text": wrapped}]}}]
    }).encode("utf-8")
    monkeypatch.setattr(sd.urllib.request, "urlopen",
                         lambda req, timeout=None: _FakeHTTPResponse(payload))
    assert sd.direct_query("Взять быка за рога.", str(tmp_path)) == "decisive person action"


def test_call_budget_hard_cap_per_run(monkeypatch, tmp_path):
    monkeypatch.setenv("SHOT_DIRECTOR_MAX_CALLS_PER_RUN", "2")
    monkeypatch.setattr(sd, "SHOT_DIRECTOR_MAX_CALLS_PER_RUN", 2)
    calls = {"n": 0}

    def _fake_urlopen(req, timeout=None):
        calls["n"] += 1
        return _FakeHTTPResponse(_fake_gemini_payload([f"query {calls['n']}"]))
    monkeypatch.setattr(sd.urllib.request, "urlopen", _fake_urlopen)

    texts = ["Фраза раз.", "Фраза два.", "Фраза три.", "Фраза четыре."]
    results = [sd.direct_query(t, str(tmp_path)) for t in texts]
    assert calls["n"] == 2
    assert results[:2] == ["query 1", "query 2"]
    assert results[2:] == [None, None]


def test_env_cannot_raise_cap_above_hard_ceiling(monkeypatch):
    # SHOT_DIRECTOR_MAX_CALLS_PER_RUN может только СУЗИТЬ лимит (тот же
    # принцип, что SPEECH_GEN_MAX_ATTEMPTS) — попытка задать 999 не должна
    # поднять реальный потолок выше жёстко зашитых 15.
    monkeypatch.setenv("SHOT_DIRECTOR_MAX_CALLS_PER_RUN", "999")
    import importlib
    reloaded = importlib.reload(sd)
    try:
        assert reloaded.SHOT_DIRECTOR_MAX_CALLS_PER_RUN <= 15
    finally:
        monkeypatch.delenv("SHOT_DIRECTOR_MAX_CALLS_PER_RUN", raising=False)
        importlib.reload(sd)
