"""Реестр флагов режимов: семантика + сверка с документацией и .env-шаблоном.

Эти тесты существуют из-за ДВУХ реальных, найденных вживую (03.09) поломок,
а не «на всякий случай»:

1. CLAUDE.md месяцами объявлял `VLM_ARBITER_MODE` дефолтом `on` («реализован
   и живьём проверен»), а все три точки чтения в коде подставляли `"off"`.
   Никакой тест не сверял документацию с кодом — расхождение жило молча, и
   рендеры шли без арбитра.
2. CLAUDE.md документировал флаг отката `DEFLICKER_ENABLED`, а код читал
   переменную `DEFLICKER`. Откат ровно по документации не работал.

Оба класса ловятся только сверкой трёх источников между собой, поэтому она
здесь и живёт.
"""
import io
import os
import re
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

import feature_flags as ff   # noqa: E402


def _read(name):
    return io.open(os.path.join(REPO_ROOT, name), encoding="utf-8").read()


# ---------- семантика реестра ----------

def test_mode_returns_registry_default_when_env_unset(monkeypatch):
    monkeypatch.delenv("VLM_ARBITER_MODE", raising=False)
    assert ff.mode("VLM_ARBITER_MODE") == ff.FLAGS["VLM_ARBITER_MODE"].default


def test_mode_normalises_case_and_spaces(monkeypatch):
    monkeypatch.setenv("VISUAL_DIRECTOR_MODE", "  ASSIST ")
    assert ff.mode("VISUAL_DIRECTOR_MODE") == "assist"


def test_garbage_mode_falls_back_to_off_not_to_default(monkeypatch, capsys):
    """Опечатка НЕ должна включать дорогой слой. У VLM_ARBITER_MODE дефолт
    "on" — если бы мусор откатывался на дефолт, `VLM_ARBITER_MODE=noo` тихо
    ВКЛЮЧАЛ бы арбитра. До реестра поведение было именно "off" (сравнение
    == "on" на мусоре), сохраняем его."""
    ff._warned.discard("VLM_ARBITER_MODE")
    monkeypatch.setenv("VLM_ARBITER_MODE", "noo")
    assert ff.mode("VLM_ARBITER_MODE") == "off"
    assert "не входит" in capsys.readouterr().err


def test_enabled_only_zero_disables(monkeypatch):
    """Сознательно НЕ признаём "off"/"false" выключением: сегодня
    RENDER_STRICT_GATE=off означает «включено», и тихо перевернуть это
    значило бы ослабить жёсткий гейт сборки у того, кто уже так написал."""
    monkeypatch.setenv("RENDER_STRICT_GATE", "0")
    assert ff.enabled("RENDER_STRICT_GATE") is False
    monkeypatch.setenv("RENDER_STRICT_GATE", "off")
    assert ff.enabled("RENDER_STRICT_GATE") is True


def test_alias_is_read_only_when_primary_name_unset(monkeypatch):
    """DEFLICKER — историческое имя. Основное имя всегда сильнее."""
    monkeypatch.delenv("DEFLICKER_ENABLED", raising=False)
    monkeypatch.setenv("DEFLICKER", "0")
    assert ff.enabled("DEFLICKER_ENABLED") is False
    monkeypatch.setenv("DEFLICKER_ENABLED", "1")
    assert ff.enabled("DEFLICKER_ENABLED") is True


def test_empty_env_value_means_unset(monkeypatch):
    """`VLM_ARBITER_MODE=` в .env (пустая строка — обычный способ «оставлю
    поле, впишу позже») не должен означать мусор и глушить слой."""
    monkeypatch.setenv("VLM_ARBITER_MODE", "")
    assert ff.mode("VLM_ARBITER_MODE") == ff.FLAGS["VLM_ARBITER_MODE"].default


def test_wrong_accessor_type_raises():
    with pytest.raises(TypeError):
        ff.enabled("VLM_ARBITER_MODE")
    with pytest.raises(TypeError):
        ff.mode("RENDER_STRICT_GATE")


def test_unknown_flag_names_the_fix():
    with pytest.raises(KeyError) as e:
        ff.mode("NO_SUCH_MODE")
    assert "feature_flags.py" in str(e.value)


def test_snapshot_marks_env_override(monkeypatch):
    monkeypatch.setenv("SHOT_DIRECTOR_MODE", "on")
    monkeypatch.delenv("OPENVERSE_ENABLED", raising=False)
    snap = ff.snapshot()
    assert snap["SHOT_DIRECTOR_MODE"] == {"value": "on", "default": "off", "from_env": True}
    assert snap["OPENVERSE_ENABLED"]["from_env"] is False


def test_write_snapshot_lands_in_media_plan(tmp_path):
    import json
    path = ff.write_snapshot(str(tmp_path))
    assert path and os.path.basename(path) == "feature_flags.json"
    assert json.load(open(path, encoding="utf-8"))["VLM_ARBITER_MODE"]["default"] == "on"


# ---------- сверка трёх источников ----------

def _documented_default(doc, name):
    """Дефолт, объявленный в CLAUDE.md: `NAME=варианты`, дефолт `X`."""
    m = re.search(rf"`{name}=[^`]*`.{{0,120}}?дефолт\s+`([^`]+)`", doc, re.S)
    return m.group(1) if m else None


@pytest.mark.parametrize("name", sorted(ff.FLAGS))
def test_registry_default_matches_claude_md(name):
    doc = _read("CLAUDE.md")
    documented = _documented_default(doc, name)
    assert documented is not None, (
        f"CLAUDE.md больше не объявляет дефолт {name} в формате "
        f"«`{name}=варианты`, дефолт `X`» — либо флаг перестали документировать "
        f"(тогда его нельзя выставить осознанно), либо формулировку изменили и "
        f"эта сверка ослепла. Оба случая надо чинить, а не игнорировать.")
    assert documented == ff.FLAGS[name].default, (
        f"{name}: CLAUDE.md обещает дефолт {documented!r}, реестр даёт "
        f"{ff.FLAGS[name].default!r}. Ровно это расхождение (VLM_ARBITER_MODE, "
        f"03.09) месяцами держало арбитра выключенным при документации «дефолт on».")


@pytest.mark.parametrize("name", sorted(ff.FLAGS))
def test_example_env_does_not_contradict_registry(name):
    """config.example.env — то, что пользователь копирует в .env. Значение,
    расходящееся с дефолтом, тихо переопределяет реестр у всех, кто взял
    шаблон (именно так VLM_ARBITER_MODE=off жил рядом с «дефолт on»)."""
    env = _read("config.example.env")
    m = re.search(rf"^{name}=(.*)$", env, re.M)
    if m is None:
        pytest.skip(f"{name} не упомянут в шаблоне — не противоречие")
    assert m.group(1).strip() == ff.FLAGS[name].default, (
        f"{name}: в config.example.env {m.group(1).strip()!r}, дефолт реестра "
        f"{ff.FLAGS[name].default!r}")


@pytest.mark.parametrize("name", sorted(ff.FLAGS))
def test_no_literal_default_left_in_code(name):
    """Ни один скрипт не должен снова читать флаг мимо реестра со своим
    литеральным дефолтом — это и есть механизм, которым код разошёлся с
    документацией. Реестру себя читать можно."""
    import glob
    offenders = []
    for path in glob.glob(os.path.join(SCRIPTS_DIR, "*.py")):
        if os.path.basename(path) == "feature_flags.py":
            continue
        src = io.open(path, encoding="utf-8").read()
        if re.search(rf'os\.(environ\.get|getenv)\(\s*"{name}"', src):
            offenders.append(os.path.basename(path))
    assert not offenders, (
        f"{name} читается мимо реестра в {offenders} — дефолт снова продублирован")


def test_every_registered_flag_is_actually_read_somewhere():
    """Обратная сторона: реестр не должен обрастать флагами, которых код не
    читает (документированный, но мёртвый флаг — это ровно случай
    DEFLICKER_ENABLED до 03.09)."""
    import glob
    src = "".join(io.open(p, encoding="utf-8").read()
                  for p in glob.glob(os.path.join(SCRIPTS_DIR, "*.py"))
                  if os.path.basename(p) != "feature_flags.py")
    for name in ff.FLAGS:
        assert f'"{name}"' in src, f"{name} есть в реестре, но ни один скрипт его не спрашивает"


# ---------- коды возврата ----------

def test_exit_codes_stay_in_sync_across_consumers():
    """pipeline_smart.EXIT_* продублированы в двух потребителях (импорт ради
    трёх чисел тянул бы PIL/numpy/CLIP в оркестратор) — расхождение молча
    вернуло бы ту же путаницу «собрано с замечаниями» = «не собрано»."""
    import render_episode
    import render_with_retry
    pytest.importorskip("PIL")
    sys.argv = ["pipeline_smart.py", os.path.join(REPO_ROOT, "videos", "_none")]
    import pipeline_smart as ps
    for mod in (render_episode, render_with_retry):
        assert mod.PIPELINE_EXIT_OK == ps.EXIT_OK
        assert mod.PIPELINE_EXIT_NOT_BUILT == ps.EXIT_NOT_BUILT
        assert mod.PIPELINE_EXIT_BUILT_WITH_WARNINGS == ps.EXIT_BUILT_WITH_WARNINGS
    assert len({ps.EXIT_OK, ps.EXIT_NOT_BUILT, ps.EXIT_BUILT_WITH_WARNINGS}) == 3


# ---------- находки состязательного аудита собственной правки (03.09) ----------

def test_garbage_falls_back_to_flag_default_when_off_is_not_legal(monkeypatch, capsys):
    """Режимы без "off" в наборе значений (GRAIN_BLEND_MODE, DELIVERY_PROFILE)
    не должны докладывать "off" — рендер в этом случае реально применяет свой
    дефолт (softlight / youtube), и снимок обязан говорить то же самое. До
    фикса snapshot() писал "off", а сводка печатала «Выключено: ...
    DELIVERY_PROFILE=off» на эпизоде, который был закодирован профилем youtube."""
    for name in ("GRAIN_BLEND_MODE", "DELIVERY_PROFILE"):
        if name not in ff.FLAGS:
            pytest.skip(f"{name} ещё не в реестре")
        ff._warned.discard(name)
        monkeypatch.setenv(name, "h265-опечатка")
        assert ff.mode(name) == ff.FLAGS[name].default
        assert ff.snapshot()[name]["value"] == ff.FLAGS[name].default
        assert repr(ff.FLAGS[name].default) in capsys.readouterr().err


def test_garbage_still_falls_back_to_off_where_off_is_legal(monkeypatch):
    """Обратная сторона: у флагов, где "off" легален, поведение не изменилось —
    опечатка НЕ включает дорогой слой (у VLM_ARBITER_MODE дефолт "on")."""
    ff._warned.discard("VLM_ARBITER_MODE")
    monkeypatch.setenv("VLM_ARBITER_MODE", "onn")
    assert ff.mode("VLM_ARBITER_MODE") == "off"


def test_arbiter_participates_in_clip_cache_key(monkeypatch):
    """VLM-арбитр меняет ВЫБОР кадра хука, значит обязан входить в ключ кэша
    клипа — иначе на прогретом temp_smart/ клип берётся из кэша до вызова
    подбора, и включение арбитра остаётся молчаливым no-op (тот же класс бага,
    что уже чинили для VISUAL_DIRECTOR_MODE через director_cache_sig).
    Выключенный арбитр обязан давать ПУСТОЙ суффикс — иначе сама правка
    перерендерила бы весь чужой кэш."""
    pytest.importorskip("PIL")
    sys.argv = ["pipeline_smart.py", os.path.join(REPO_ROOT, "videos", "_none")]
    import pipeline_smart as ps
    monkeypatch.setenv("VLM_ARBITER_MODE", "on")
    assert ps.arbiter_cache_suffix("HOOK") == "|arbiter:on"
    assert ps.arbiter_cache_suffix("BLOCK 1") == ""      # арбитр туда не передаётся
    monkeypatch.setenv("VLM_ARBITER_MODE", "off")
    assert ps.arbiter_cache_suffix("HOOK") == ""
    assert ps.arbiter_cache_suffix("BLOCK 1") == ""
