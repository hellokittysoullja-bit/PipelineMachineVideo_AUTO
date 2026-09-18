"""Зрячий гейт кадра: односторонний, платный, кэшируемый.

Три инварианта, каждый из которых уже был в этом репозитории дефектом:
  * «нет вердикта» != «плохо» (NO_CANDIDATE_FITS у VLM-арбитра),
  * платный слой не должен уходить в сеть из тестов (MUSEUM_SOURCES_ENABLED),
  * ключ только из окружения, в коде его нет (шапка CLAUDE.md).
"""
import io
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import frame_verifier as fv  # noqa: E402


@pytest.fixture
def frame(tmp_path):
    from PIL import Image
    p = tmp_path / "f.jpg"
    Image.new("RGB", (320, 200), (40, 90, 140)).save(p)
    return str(p)


def test_disabled_without_key_returns_no_opinion(monkeypatch, frame):
    monkeypatch.setenv("FRAME_VERIFIER", "1")
    monkeypatch.delenv("ANYMODEL_API_KEY", raising=False)
    assert fv.enabled() is False
    assert fv.verify(frame, "коза не могла уснуть") is None


def test_flag_off_returns_no_opinion(monkeypatch, frame):
    monkeypatch.setenv("FRAME_VERIFIER", "0")
    monkeypatch.setenv("ANYMODEL_API_KEY", "sk-test")
    assert fv.enabled() is False
    assert fv.verify(frame, "коза не могла уснуть") is None


def test_network_failure_is_no_opinion_not_rejection(monkeypatch, frame):
    """Недоступный шлюз обязан дать None («мнения нет»), а не «no»: иначе
    обрыв сети молча выбрасывал бы годные кадры из ролика."""
    monkeypatch.setenv("FRAME_VERIFIER", "1")
    monkeypatch.setenv("ANYMODEL_API_KEY", "sk-test")
    fv.reset_stats()

    def boom(*a, **k):
        raise OSError("network down")

    monkeypatch.setattr(fv.urllib.request, "urlopen", boom)
    assert fv.verify(frame, "коза не могла уснуть") is None
    assert fv.STATS["errors"] == 1


def _fake_response(payload):
    class R:
        def __enter__(self_inner):
            return self_inner
        def __exit__(self_inner, *a):
            return False
        def read(self_inner):
            return json.dumps(payload).encode()
    return R()


def _stub(monkeypatch, content, usage=None):
    def urlopen(req, timeout=None):
        return _fake_response({"choices": [{"message": {"content": content}}],
                               "usage": usage or {"total_tokens": 1092}})
    monkeypatch.setattr(fv.urllib.request, "urlopen", urlopen)


def test_verdict_no_is_parsed(monkeypatch, frame):
    monkeypatch.setenv("FRAME_VERIFIER", "1")
    monkeypatch.setenv("ANYMODEL_API_KEY", "sk-test")
    fv.reset_stats()
    _stub(monkeypatch, '{"verdict":"no","seen":"ветка с ягодами","missing":"коза"}')
    out = fv.verify(frame, "А начиналось всё с козы, которая не могла уснуть.")
    assert out["verdict"] == "no"
    assert "коза" in out["missing"]
    assert fv.STATS["tokens"] == 1092


def test_unparseable_answer_is_no_opinion(monkeypatch, frame):
    """Модель ответила прозой — это НЕ отказ кадру. glm/glm-4.6v в живом
    замере 18.09 отвечала именно так, и засчитывать такое за «no» значило бы
    получить 6/8 артефактом подсчёта вместо результата."""
    monkeypatch.setenv("FRAME_VERIFIER", "1")
    monkeypatch.setenv("ANYMODEL_API_KEY", "sk-test")
    fv.reset_stats()
    _stub(monkeypatch, "Кадр в целом подходит по теме, но сложно сказать.")
    assert fv.verify(frame, "коза не могла уснуть") is None
    assert fv.STATS["errors"] == 1


def test_budget_cap_stops_calls(monkeypatch, frame):
    monkeypatch.setenv("FRAME_VERIFIER", "1")
    monkeypatch.setenv("ANYMODEL_API_KEY", "sk-test")
    fv.reset_stats()
    monkeypatch.setattr(fv, "MAX_CALLS_PER_RUN", 1)
    _stub(monkeypatch, '{"verdict":"yes","seen":"кадр","missing":""}')
    assert fv.verify(frame, "фраза один") is not None
    assert fv.verify(frame, "фраза два") is None
    assert fv.STATS["budget_stops"] == 1


def test_cache_hit_does_not_call_again(monkeypatch, frame, tmp_path):
    monkeypatch.setenv("FRAME_VERIFIER", "1")
    monkeypatch.setenv("ANYMODEL_API_KEY", "sk-test")
    fv.reset_stats()
    _stub(monkeypatch, '{"verdict":"no","seen":"кратеры","missing":"парашют"}')
    vd = str(tmp_path)
    a = fv.verify(frame, "раскрывается парашют", vd)
    assert a["verdict"] == "no"
    calls_after_first = fv.STATS["calls"]

    def must_not_call(*a, **k):
        raise AssertionError("повторный платный вызов на том же кадре и фразе")

    monkeypatch.setattr(fv.urllib.request, "urlopen", must_not_call)
    b = fv.verify(frame, "раскрывается парашют", vd)
    assert b["verdict"] == "no"
    assert fv.STATS["calls"] == calls_after_first
    assert fv.STATS["cache_hits"] == 1


def test_cache_key_separates_different_phrases(monkeypatch, frame, tmp_path):
    """Один кадр под РАЗНЫМИ фразами — разные вопросы и разные вердикты."""
    monkeypatch.setenv("FRAME_VERIFIER", "1")
    monkeypatch.setenv("ANYMODEL_API_KEY", "sk-test")
    fv.reset_stats()
    vd = str(tmp_path)
    _stub(monkeypatch, '{"verdict":"no","seen":"кратеры","missing":"парашют"}')
    fv.verify(frame, "раскрывается парашют", vd)
    _stub(monkeypatch, '{"verdict":"yes","seen":"кратеры Марса","missing":""}')
    out = fv.verify(frame, "поверхность планеты вся в кратерах", vd)
    assert out["verdict"] == "yes"
    assert fv.STATS["cache_hits"] == 0


def test_no_key_literal_in_source():
    """Ключ только из окружения — в модуле его нет ни в каком виде."""
    src = open(os.path.join(os.path.dirname(__file__), "..", "scripts",
                            "frame_verifier.py"), encoding="utf-8").read()
    assert "sk-" not in src


def test_browser_user_agent_is_sent(monkeypatch, frame):
    """Cloudflare шлюза отдаёт 403 (error code 1010) на UA питоновского urllib
    и пропускает curl — изолировано перекрёстной проверкой 18.09. Без
    браузерного UA гейт молча не работает вообще."""
    monkeypatch.setenv("FRAME_VERIFIER", "1")
    monkeypatch.setenv("ANYMODEL_API_KEY", "sk-test")
    fv.reset_stats()
    seen = {}

    def urlopen(req, timeout=None):
        seen["ua"] = req.get_header("User-agent") or ""
        return _fake_response({"choices": [{"message": {"content":
                               '{"verdict":"yes","seen":"x","missing":""}'}}],
                               "usage": {"total_tokens": 10}})

    monkeypatch.setattr(fv.urllib.request, "urlopen", urlopen)
    fv.verify(frame, "фраза")
    assert "Mozilla/5.0" in seen["ua"], seen
