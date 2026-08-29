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


# ---------- _thinking_config_for_model(): реальное поведение разных моделей ----------
# thinkingBudget=0 обязателен для gemini-2.5-flash (обрыв MAX_TOKENS без
# него), но на gemini-3.6-flash тот же флаг даёт HTTP 400 invalid argument
# (проверено вживую 27.08 на реальном ключе, не гипотеза). Дефолт модуля
# теперь gemini-3.6-flash, потому что gemini-2.5-flash недоступна ключу,
# который в итоге получил этот канал ("no longer available to new users").

def test_thinking_config_zero_budget_for_25_family():
    assert sd._thinking_config_for_model("gemini-2.5-flash") == {"thinkingBudget": 0}
    assert sd._thinking_config_for_model("gemini-2.5-pro") == {"thinkingBudget": 0}


def test_thinking_config_omitted_for_36_flash():
    assert sd._thinking_config_for_model("gemini-3.6-flash") is None


def test_thinking_config_omitted_for_unknown_future_model():
    # Честная деградация на неизвестную модель — не гадаем thinkingBudget,
    # просто не отправляем thinkingConfig вовсе.
    assert sd._thinking_config_for_model("gemini-9.0-nano") is None


def test_call_body_omits_thinking_config_for_default_model(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeHTTPResponse(_fake_gemini_payload(["a query", "b query"]))
    monkeypatch.setattr(sd.urllib.request, "urlopen", fake_urlopen)
    with tempfile.TemporaryDirectory() as d:
        sd.direct_query("человек принимает решение", d)
    assert "thinkingConfig" not in captured["body"]["generationConfig"], (
        "дефолтная модель этого модуля (gemini-3.6-flash) не принимает "
        "thinkingConfig вообще — отправка ломает запрос")


def test_call_body_includes_zero_budget_for_25_flash(monkeypatch):
    monkeypatch.setattr(sd, "SHOT_DIRECTOR_MODEL", "gemini-2.5-flash")
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeHTTPResponse(_fake_gemini_payload(["a query", "b query"]))
    monkeypatch.setattr(sd.urllib.request, "urlopen", fake_urlopen)
    with tempfile.TemporaryDirectory() as d:
        sd.direct_query("человек принимает решение", d)
    assert captured["body"]["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}


# ---------- enrich_atmospheric_queries (атмосферное обогащение хука) ----------
# см. её докстринг в shot_director.py — реальная жалоба пользователя: у блока
# УЖЕ есть технически релевантный запрос, но он взят слишком буквально, в
# отрыве от темы остального ролика.

def test_atmo_off_mode_never_touches_network(monkeypatch, tmp_path):
    monkeypatch.setenv("SHOT_DIRECTOR_MODE", "off")

    def _boom(*a, **kw):
        raise AssertionError("network должен быть недостижим в off-режиме")
    monkeypatch.setattr(sd.urllib.request, "urlopen", _boom)
    assert sd.enrich_atmospheric_queries(
        "Пятнадцать килограммов.", "weighing scale metal object",
        ["medieval european sword blade close up"], str(tmp_path)) is None


def test_atmo_no_api_key_returns_none_without_network(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    def _boom(*a, **kw):
        raise AssertionError("не должно быть сетевого вызова без ключа")
    monkeypatch.setattr(sd.urllib.request, "urlopen", _boom)
    assert sd.enrich_atmospheric_queries(
        "Пятнадцать килограммов.", "weighing scale metal object", [], str(tmp_path)) is None


def test_atmo_successful_call_returns_full_list_and_writes_cache(monkeypatch, tmp_path):
    payload = _fake_gemini_payload(["heavy iron weight medieval sword", "knight holding heavy sword"])
    monkeypatch.setattr(sd.urllib.request, "urlopen",
                         lambda req, timeout=None: _FakeHTTPResponse(payload))
    result = sd.enrich_atmospheric_queries(
        "Пятнадцать килограммов.", "weighing scale metal object",
        ["medieval european sword blade close up"], str(tmp_path))
    assert result == ["heavy iron weight medieval sword", "knight holding heavy sword"]
    cache_file = sd._atmo_cache_path(
        str(tmp_path), "Пятнадцать килограммов.", "weighing scale metal object",
        ["medieval european sword blade close up"])
    assert os.path.exists(cache_file)


def test_atmo_cache_hit_skips_network_entirely(monkeypatch, tmp_path):
    payload = _fake_gemini_payload(["heavy iron weight medieval sword"])
    monkeypatch.setattr(sd.urllib.request, "urlopen",
                         lambda req, timeout=None: _FakeHTTPResponse(payload))
    args = ("Пятнадцать килограммов.", "weighing scale metal object",
            ["medieval european sword blade close up"], str(tmp_path))
    first = sd.enrich_atmospheric_queries(*args)

    def _boom(*a, **kw):
        raise AssertionError("кэш-хит не должен трогать сеть")
    monkeypatch.setattr(sd.urllib.request, "urlopen", _boom)
    second = sd.enrich_atmospheric_queries(*args)
    assert first == second == ["heavy iron weight medieval sword"]


def test_atmo_cache_key_changes_with_context_queries(monkeypatch, tmp_path):
    # Реальный сценарий: человек отредактировал script.txt, состав запросов
    # секции изменился — кэш должен честно промахнуться, не отдать
    # атмосферное обогащение под СТАРУЮ тему раздела.
    payload1 = _fake_gemini_payload(["query for context A"])
    monkeypatch.setattr(sd.urllib.request, "urlopen",
                         lambda req, timeout=None: _FakeHTTPResponse(payload1))
    r1 = sd.enrich_atmospheric_queries("Текст.", "own query", ["context A"], str(tmp_path))
    assert r1 == ["query for context A"]

    payload2 = _fake_gemini_payload(["query for context B"])
    monkeypatch.setattr(sd.urllib.request, "urlopen",
                         lambda req, timeout=None: _FakeHTTPResponse(payload2))
    r2 = sd.enrich_atmospheric_queries("Текст.", "own query", ["context B"], str(tmp_path))
    assert r2 == ["query for context B"]


def test_atmo_shares_call_budget_with_direct_query(monkeypatch, tmp_path):
    monkeypatch.setattr(sd, "SHOT_DIRECTOR_MAX_CALLS_PER_RUN", 1)
    payload = _fake_gemini_payload(["some query"])
    monkeypatch.setattr(sd.urllib.request, "urlopen",
                         lambda req, timeout=None: _FakeHTTPResponse(payload))
    first = sd.enrich_atmospheric_queries("Текст 1.", "q1", ["ctx"], str(tmp_path))
    assert first == ["some query"]
    # Бюджет исчерпан этим ЖЕ вызовом — direct_query() на РАЗНОМ тексте
    # (свежий кэш-промах) обязан молча вернуть None, не звонить в сеть.
    def _boom(*a, **kw):
        raise AssertionError("бюджет должен быть общим со enrich_atmospheric_queries")
    monkeypatch.setattr(sd.urllib.request, "urlopen", _boom)
    assert sd.direct_query("Совсем другой текст.", str(tmp_path)) is None


def test_atmo_network_error_fails_open(monkeypatch, tmp_path):
    def _raise(*a, **kw):
        raise OSError("network down")
    monkeypatch.setattr(sd.urllib.request, "urlopen", _raise)
    assert sd.enrich_atmospheric_queries("Текст.", "q", ["ctx"], str(tmp_path)) is None


def test_atmo_own_query_excluded_from_context(monkeypatch, tmp_path):
    # own_query не должен дублироваться в context — enrich_atmospheric_queries
    # сама фильтрует, вызывающий код (pipeline_smart.py) передаёт "как есть".
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["body"] = req.data.decode("utf-8")
        return _FakeHTTPResponse(_fake_gemini_payload(["q"]))
    monkeypatch.setattr(sd.urllib.request, "urlopen", fake_urlopen)
    sd.enrich_atmospheric_queries("Текст.", "own query", ["own query", "other query"], str(tmp_path))
    assert captured["body"].count("own query") == 1   # только в описании own_query, не в context-списке
