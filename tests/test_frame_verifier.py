"""Зрячий гейт кадра: односторонний, платный, кэшируемый.

Три инварианта, каждый из которых уже был в этом репозитории дефектом:
  * «нет вердикта» != «плохо» (NO_CANDIDATE_FITS у VLM-арбитра),
  * платный слой не должен уходить в сеть из тестов (MUSEUM_SOURCES_ENABLED),
  * ключ только из окружения, в коде его нет (шапка CLAUDE.md).
"""
import io
import json
import os
import shutil
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
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


# --- РАЗРЕШЁННЫЙ БРИФ АВТОРА В САМОМ ВОПРОСЕ (PROMPT_VERSION 2, 19.09) ----
#
# Реальный, живым прогоном найденный случай, не гипотеза: фраза «Он весил
# меньше, чем ты думаешь — грамм триста» без разрешённого местоимения («он»
# = кинжал, названный двумя фразами раньше в сценарии) заставила гейт
# засчитать кандидата «gauntlet держит МЕЧ» вердиктом «да» — тот же самый
# облачный вызов на том же кадре, ПОЛУЧИВ разрешённый бриф автора
# (`[shot:scene|a gauntleted hand holding THE DAGGER effortlessly]`),
# ответил «нет: gauntleted hands holding sword hilt, missing dagger held
# by the tip». Разрыв был не в модели, а в том, какой вопрос ей задавали:
# CLIP-поиск и сток уже получают разрешённое автором описание, а зрячий
# гейт — только голую, местами неоднозначную фразу диктора.

def test_empty_brief_renders_byte_for_byte_prompt(monkeypatch, frame):
    """Юнит без [shot:] (старый эпизод, режиссёр брифов не гонялся) —
    промпт ДОЛЖЕН остаться тем же самым текстом, что был до PROMPT_VERSION
    2: пустой бриф не добавляет ни одного лишнего токена, когда автор
    ничего не уточнил."""
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
    assert "Уточнение автора сценария" not in text
    assert "а точнее — то, что названо в уточнении автора" not in text
    assert "Вопрос ровно один: показывает ли этот кадр то, о чём говорит фраза? " in text


def test_shot_brief_reaches_the_prompt_sent_to_the_model(monkeypatch, frame):
    """Непустой бриф обязан реально дойти до текста, который уходит в
    модель — иначе это ровно тот класс «слой есть, и его никто не зовёт»,
    которым этот репозиторий уже горел восемь раз."""
    monkeypatch.setenv("FRAME_VERIFIER", "1")
    monkeypatch.setenv("ANYMODEL_API_KEY", "sk-test")
    fv.reset_stats()
    monkeypatch.setattr(fv, "_world_context", lambda video_dir=None: "")
    seen = {}

    def urlopen(req, timeout=None):
        seen["body"] = json.loads(req.data.decode())
        return _fake_response({"choices": [{"message": {"content":
                               '{"verdict":"no","seen":"меч","missing":"кинжал"}'}}],
                               "usage": {"total_tokens": 10}})

    monkeypatch.setattr(fv.urllib.request, "urlopen", urlopen)
    brief = "a gauntleted hand holding the dagger effortlessly by the very tip"
    fv.verify(frame, "Он весил меньше, чем ты думаешь — грамм триста.",
              shot_brief=brief)
    text = seen["body"]["messages"][0]["content"][0]["text"]
    assert brief in text
    assert "а точнее — то, что названо в уточнении автора" in text


def test_cache_key_separates_different_briefs(monkeypatch, frame, tmp_path):
    """Один кадр, одна фраза, но РАЗНЫЙ бриф — это разные вопросы: вердикт
    без брифа не должен молча выжить под брифом (и наоборот)."""
    monkeypatch.setenv("FRAME_VERIFIER", "1")
    monkeypatch.setenv("ANYMODEL_API_KEY", "sk-test")
    fv.reset_stats()
    vd = str(tmp_path)
    _stub(monkeypatch, '{"verdict":"yes","seen":"меч","missing":""}')
    fv.verify(frame, "фраза", vd)   # без брифа

    _stub(monkeypatch, '{"verdict":"no","seen":"меч","missing":"кинжал"}')
    out = fv.verify(frame, "фраза", vd, shot_brief="a dagger held by the tip")
    assert out["verdict"] == "no"
    assert fv.STATS["cache_hits"] == 0


def test_empty_brief_is_a_no_op_for_the_cache_key(monkeypatch, frame, tmp_path):
    """shot_brief=None и shot_brief="" — тот же ключ, что и вовсе без
    параметра (обратная совместимость со старыми вызовами verify())."""
    vd = str(tmp_path)
    k1 = fv._cache_key(frame, "фраза", "")
    k2 = fv._cache_key(frame, "фраза", "", shot_brief=None)
    k3 = fv._cache_key(frame, "фраза", "", shot_brief="")
    assert k1 == k2 == k3


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


# --- СВОЙ CLI-ПРОЦЕСС ОБЯЗАН ВИДЕТЬ КЛЮЧ ИЗ .env, НЕ ТОЛЬКО ИЗ ОКРУЖЕНИЯ --
#
# Найдено 19.09 при живой проверке `python scripts/frame_verifier.py
# balance`: команда печатала `null` на машине, где .env реально содержит
# рабочий ANYMODEL_API_KEY. Причина — .env грузит ТОЛЬКО pipeline_smart.py
# при своём импорте; когда frame_verifier.py запускают своим отдельным
# процессом (а не изнутри pipeline_smart.py), никто .env не читал вообще.
# Поймано живым прогоном, а не рассуждением — см. коммит.

def test_standalone_process_sees_key_from_env_file_alone(tmp_path):
    """Ключ лежит ТОЛЬКО в .env файле (не экспортирован в окружение) —
    отдельный python-процесс, импортирующий копию модуля, обязан увидеть
    его через собственный load_dotenv(), а не молчать."""
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    shutil.copy(os.path.join(REPO, "scripts", "frame_verifier.py"),
                scripts_dir / "frame_verifier.py")
    (tmp_path / ".env").write_text(
        "ANYMODEL_API_KEY=probe-from-dotenv-only\n", encoding="utf-8")

    env = {k: v for k, v in os.environ.items() if k != "ANYMODEL_API_KEY"}
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, sys.argv[1]); import frame_verifier; "
         "import os; print(os.environ.get('ANYMODEL_API_KEY'))",
         str(scripts_dir)],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "probe-from-dotenv-only"


def test_standalone_process_respects_already_exported_key(tmp_path):
    """Переменная, уже заданная в окружении (CI, экспорт руками), не
    должна быть перетёрта файлом `.env` — override=False, не «файл важнее
    оболочки»."""
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    shutil.copy(os.path.join(REPO, "scripts", "frame_verifier.py"),
                scripts_dir / "frame_verifier.py")
    (tmp_path / ".env").write_text(
        "ANYMODEL_API_KEY=from-dotenv-file\n", encoding="utf-8")

    env = dict(os.environ)
    env["ANYMODEL_API_KEY"] = "from-real-shell-export"
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, sys.argv[1]); import frame_verifier; "
         "import os; print(os.environ.get('ANYMODEL_API_KEY'))",
         str(scripts_dir)],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "from-real-shell-export"
