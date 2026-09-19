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


# --- МИР КАНАЛА В ПРОМПТЕ: облачный судья видит не только фразу -----------
#
# Прямой ответ на запрос владельца «сделать облачный API умнее не в
# конкретных случаях, а везде»: без мира канала зрячий гейт сравнивает кадр
# ТОЛЬКО с буквальной фразой и пропустит современного туриста на кадре,
# где во фразе нет ни одного предметного слова про эпоху — тот же класс
# промаха, что VISUAL_DOMAIN_GUARDS уже закрывает узким CLIP-анкором формы
# клинка, здесь — общим зрением модели.

def test_empty_world_renders_byte_for_byte_prompt(monkeypatch, frame):
    """Канал без объявленной ниши (SHOT_BRIEF_WORLD=off, новый канал,
    сбой domain_contract()) — промпт ДОЛЖЕН остаться тем же самым текстом,
    что был до появления мира: третий параметр не добавляет ни одного
    лишнего токена, когда сказать нечего."""
    monkeypatch.setenv("FRAME_VERIFIER", "1")
    monkeypatch.setenv("ANYMODEL_API_KEY", "sk-test")
    fv.reset_stats()
    monkeypatch.setattr(fv, "_world_context", lambda video_dir=None: "")
    seen = {}

    def urlopen(req, timeout=None):
        seen["body"] = json.loads(req.data.decode())
        return _fake_response({"choices": [{"message": {"content":
                               '{"verdict":"yes","seen":"x","missing":""}'}}],
                               "usage": {"total_tokens": 10}})

    monkeypatch.setattr(fv.urllib.request, "urlopen", urlopen)
    fv.verify(frame, "коза не могла уснуть")
    text = seen["body"]["messages"][0]["content"][0]["text"]
    assert "Ты монтажёр документального ролика. Тебе дан КАДР" in text
    assert "Канал, для которого сделан ролик" not in text


def test_world_context_reaches_the_prompt_sent_to_the_model(monkeypatch, frame):
    """Непустой мир канала обязан реально дойти до текста, который уходит
    в модель — иначе `_world_context()` был бы ровно тем классом «слой
    есть, и его никто не зовёт», которым этот репозиторий горел семь раз."""
    monkeypatch.setenv("FRAME_VERIFIER", "1")
    monkeypatch.setenv("ANYMODEL_API_KEY", "sk-test")
    fv.reset_stats()
    world = "МИР КАДРА: европейское Средневековье, 900-1600."
    monkeypatch.setattr(fv, "_world_context", lambda video_dir=None: world)
    seen = {}

    def urlopen(req, timeout=None):
        seen["body"] = json.loads(req.data.decode())
        return _fake_response({"choices": [{"message": {"content":
                               '{"verdict":"no","seen":"турист","missing":"рыцарь"}'}}],
                               "usage": {"total_tokens": 10}})

    monkeypatch.setattr(fv.urllib.request, "urlopen", urlopen)
    fv.verify(frame, "рыцарь надевает доспех")
    text = seen["body"]["messages"][0]["content"][0]["text"]
    assert world in text
    assert "чужероден этому миру" in text


def test_world_context_asks_domain_contract_with_this_episode(monkeypatch, tmp_path):
    """`_world_context()` обязана передать video_dir дальше, в
    `domain_contract(video_dir)` — без этого та функция лезет за
    `import pipeline_smart`, а pipeline_smart.py сам импортирует
    frame_verifier: во время реального рендера это заново выполнило бы
    файл целиком под вторым именем модуля."""
    calls = []

    class FakeSBD:
        @staticmethod
        def domain_contract(video_dir=None):
            calls.append(video_dir)
            return "МИР"

    monkeypatch.setitem(sys.modules, "shot_brief_director", FakeSBD)
    out = fv._world_context(str(tmp_path))
    assert out == "МИР"
    assert calls == [str(tmp_path)]


def test_world_context_fails_open_on_any_error(monkeypatch):
    class Boom:
        @staticmethod
        def domain_contract(video_dir=None):
            raise RuntimeError("boom")

    monkeypatch.setitem(sys.modules, "shot_brief_director", Boom)
    assert fv._world_context("videos/01") == ""


def test_cache_key_separates_different_worlds(monkeypatch, frame, tmp_path):
    """Один и тот же кадр и одна и та же фраза, но мир канала поменялся
    (переезд на другую нишу, ЧАСТЬ 24, или включили content_world.json для
    эпизода) — это ДРУГОЙ вопрос модели, и старый вердикт не должен молча
    выжить под ним."""
    monkeypatch.setenv("FRAME_VERIFIER", "1")
    monkeypatch.setenv("ANYMODEL_API_KEY", "sk-test")
    fv.reset_stats()
    vd = str(tmp_path)

    monkeypatch.setattr(fv, "_world_context", lambda video_dir=None: "мир А")
    _stub(monkeypatch, '{"verdict":"yes","seen":"x","missing":""}')
    fv.verify(frame, "фраза", vd)

    monkeypatch.setattr(fv, "_world_context", lambda video_dir=None: "мир Б")
    _stub(monkeypatch, '{"verdict":"no","seen":"y","missing":"z"}')
    out = fv.verify(frame, "фраза", vd)
    assert out["verdict"] == "no"
    assert fv.STATS["cache_hits"] == 0


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
