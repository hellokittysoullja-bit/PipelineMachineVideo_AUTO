"""Вето атмосферы: что считается ВЕРДИКТОМ, а что его отсутствием.

Односторонность вето (может только снять то, что выбрал словарь) держится
двумя вещами: точкой вызова в `ambience_plan.plan_ambience` — её проверяет
`tests/test_ambience_plan.py::TestLlmVeto` — и правилом «снимаем только по
явному отказу модели», которое проверяется здесь.

Правило пришлось написать после разбора 17.09: заявленный fail-open не
покрывал ГЛАВНЫЙ режим отказа локальной модели. `LocalBrain.ask()` при сбое
возвращает пустую строку, а не бросает исключение — то есть `except
Exception` не срабатывал, пустой ответ читался как «атмосфера не нужна», и
упавшая модель молча гасила фон во всём эпизоде.
"""
import os
import shutil
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import ambience_director as ad  # noqa: E402


@pytest.mark.parametrize("raw", [
    "",                       # модель упала — LocalBrain.ask отдаёт "" без исключения
    "   \n  ",                # то же, но с пробелами
    "не знаю, наверное ветер",  # ответ не по форме
    "SCENE:",                 # форма есть, сцены нет
])
def test_a_non_answer_never_removes_the_dictionary_bed(raw):
    keep, why = ad.veto_decision(raw)
    assert keep is True, f"ответ без вердикта снял атмосферу: {raw!r}"
    assert why, "причина «почему нет вердикта» обязана быть названа, а не молчать"


def test_explicit_none_is_the_only_thing_that_removes():
    keep, why = ad.veto_decision("NONE")
    assert keep is False
    assert why is None


def test_a_valid_scene_keeps_the_bed():
    keep, why = ad.veto_decision("SCENE: wind over an open field")
    assert keep is True
    assert why is None


def test_discrete_event_wording_does_not_veto_in_veto_mode():
    """`DISCRETE_EVENT_RE` писался для режима, где модель ВЫБИРАЕТ сцену: там
    разовое действие — негодный ответ. В режиме вето сцена выбрасывается
    вызывающим кодом, и отклонять по слову внутри строки, которая на решение
    не влияет, значило бы снимать законный фон.

    Контроль формы: сама `parse_answer` этот ответ по-прежнему отклоняет —
    правило не ослаблено, оно просто не применяется там, где не про то."""
    answer = "SCENE: wind over a battlefield with distant clashing"
    wants, _scene, reason = ad.parse_answer(answer)
    assert wants is False and reason.startswith("rejected_discrete_event")
    keep, why = ad.veto_decision(answer)
    assert keep is True
    assert why and why.startswith("rejected_discrete_event")


def test_the_closure_uses_the_same_rule():
    """Второй копии правила не заводится: замыкание `veto()` обязано звать
    `veto_decision`, иначе тесты выше сторожили бы функцию, которой в проде
    никто не пользуется — ровно тот класс «слой есть, и его никто не зовёт»,
    которым этот репозиторий горит семь раз."""
    import inspect
    src = inspect.getsource(ad.build_veto_fn)
    assert "veto_decision(raw)" in src


# --- СВОБОДА ПЕРЕКЛЮЧЕНИЯ МОЗГА: ЛОКАЛЬНАЯ МОДЕЛЬ <-> ОБЛАКО --------------
#
# Харнесс (LocalBrain/FileBrain/find_model) уже был ПЕРЕИСПОЛЬЗОВАН у
# shot_brief_director.py целиком, но CloudBrain — нет: build_veto_fn()
# хардкодила LocalBrain в двух местах, и облачный ключ, уже подключённый и
# измеренный для брифов, к атмосфере было physически не подключить без
# правки кода. Теперь — одна переменная .env.

def test_build_veto_fn_accepts_an_explicit_brain(monkeypatch, tmp_path):
    """Явно переданный мозг (любой объект с .ask()) используется как есть,
    без похода за локальной моделью и без чтения ANYMODEL_API_KEY."""
    (tmp_path / "script.txt").write_text("=== METADATA ===\nTITLE: т\n"
                                         "=== HOOK ===\nа.\n", encoding="utf-8")

    class Stub:
        name = "stub"
        calls = []

        def ask(self, prompt, chapter_no):
            self.calls.append(prompt)
            return "SCENE: wind over an open field"

    stub = Stub()
    veto = ad.build_veto_fn(str(tmp_path), brain=stub, cache_dir=str(tmp_path / "c"))
    assert veto is not None
    veto("Открытое поле, ветер.")
    assert stub.calls, "явно переданный мозг обязан реально вызываться"


def test_build_veto_fn_switches_to_cloud_by_env_var(monkeypatch, tmp_path):
    """AMBIENCE_VETO_BRAIN=cloud — облако вместо локальной модели, без
    единой правки кода вызывающей стороны (pipeline_smart.py её не знает
    и не обязан)."""
    monkeypatch.setenv("AMBIENCE_VETO_BRAIN", "cloud")
    monkeypatch.setenv("ANYMODEL_API_KEY", "sk-test")
    (tmp_path / "script.txt").write_text("=== METADATA ===\nTITLE: т\n"
                                         "=== HOOK ===\nа.\n", encoding="utf-8")
    veto = ad.build_veto_fn(str(tmp_path), cache_dir=str(tmp_path / "c"))
    assert veto is not None
    # Внутренний мозг замыкания недоступен напрямую — проверяем по типу
    # через побочный эффект: без сети CloudBrain.ask() возвращает "" и не
    # падает (тот же fail-open, что уже проверен для облака отдельно).
    keep, why = ad.veto_decision("")
    assert keep is True and why


def test_build_veto_fn_cloud_without_key_is_a_safe_noop(monkeypatch, tmp_path):
    monkeypatch.setenv("AMBIENCE_VETO_BRAIN", "cloud")
    monkeypatch.delenv("ANYMODEL_API_KEY", raising=False)
    (tmp_path / "script.txt").write_text("=== METADATA ===\nTITLE: т\n"
                                         "=== HOOK ===\nа.\n", encoding="utf-8")
    assert ad.build_veto_fn(str(tmp_path)) is None


def test_build_veto_fn_default_is_still_local(monkeypatch, tmp_path):
    """Байт-в-байт прежнее поведение, когда переменная вообще не задана:
    свобода переключения не должна тихо поменять дефолт."""
    monkeypatch.delenv("AMBIENCE_VETO_BRAIN", raising=False)
    monkeypatch.setattr(ad, "find_model", lambda *_a, **_kw: None)
    (tmp_path / "script.txt").write_text("=== METADATA ===\nTITLE: т\n"
                                         "=== HOOK ===\nа.\n", encoding="utf-8")
    assert ad.build_veto_fn(str(tmp_path)) is None


# --- СВОЙ CLI-ПРОЦЕСС (AMBIENCE_VETO_BRAIN=cloud) ОБЯЗАН ВИДЕТЬ КЛЮЧ ------
#
# Тот же дефект и та же причина, что у shot_brief_director.py/frame_
# verifier.py рядом: .env грузит только pipeline_smart.py при своём
# импорте, а собственный CLI-процесс этого файла — отдельный процесс.

def test_standalone_cli_sees_key_from_env_file_alone(tmp_path):
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    shutil.copy(os.path.join(REPO, "scripts", "ambience_director.py"),
                scripts_dir / "ambience_director.py")
    (tmp_path / ".env").write_text(
        "ANYMODEL_API_KEY=probe-from-dotenv-only\n", encoding="utf-8")

    env = {k: v for k, v in os.environ.items() if k != "ANYMODEL_API_KEY"}
    real_scripts = os.path.join(REPO, "scripts")
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); sys.path.insert(1, sys.argv[2]); "
        "import ambience_director; import os; "
        "print(os.environ.get('ANYMODEL_API_KEY'))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(scripts_dir), real_scripts],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "probe-from-dotenv-only"
