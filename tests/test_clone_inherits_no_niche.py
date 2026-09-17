"""Клон репозитория под НОВУЮ нишу не должен тащить антианахронизм старой.

ЧАСТЬ 24 CLAUDE.md требует это прямым текстом. Проверки не было ни одной, и
требование не выполнялось — замер 17.09 на запросах чужих ниш:

    'tank battle field'     -> 'european medieval tank battle field'
    'spear hunting savanna' -> 'european medieval spear hunting savanna'
    'infantry helmet mud'   -> 'european infantry helmet mud'
    'sword smith forging'   -> 'european sword smith forging'   (ниша Япония)

Шесть из десяти. Уточнитель уходит в РЕАЛЬНЫЙ вызов API, то есть пул слота
собирался по искажённому запросу — это не косметика.

Причина у всех четырёх найденных мест одна: значение жило КОНСТАНТОЙ В КОДЕ
как «дефолт», а `channel_profile.json` этого канала его не объявлял, поэтому
побеждала константа — и для клона тоже. Проверяется поведение свежего
клона, в котором профиля нет вовсе: именно тот случай, который ЧАСТЬ 24
разрешает («файл можно удалить/оставить пустым»).
"""
import json
import os
import subprocess
import sys
import textwrap

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROBE = textwrap.dedent('''
    import json, os, sys
    sys.path.insert(0, os.path.join(os.environ["CLONE"], "scripts"))
    sys.argv = ["pipeline_smart.py", os.path.join(os.environ["CLONE"], "videos", "ep")]
    os.chdir(os.environ["CLONE"])
    import pipeline_smart as ps
    import museum_sources as ms
    queries = ["stone spear point flint", "tank battle field",
               "spear hunting savanna", "sword smith forging",
               "infantry helmet mud", "surgical blade close up"]
    print(json.dumps({
        "rules": len(ps.QUERY_DISAMBIGUATION_RULES),
        "guards": len(ps.VISUAL_DOMAIN_GUARDS),
        "era_declared": ms.era_window_declared(),
        "cultures_declared": ms.foreign_culture_terms_declared(),
        "distorted": [q for q in queries if ps.disambiguate_search_query(q) != q],
    }))
''')


def _probe_clone(tmp_path, profile=None):
    """Свежий клон: только scripts/, профиль по требованию теста."""
    clone = tmp_path / "clone"
    (clone / "videos" / "ep").mkdir(parents=True)
    (clone / "videos" / "ep" / "script.txt").write_text(
        "=== HOOK ===\n[shot:object|a flint hand axe]Топор. [pause]\n", encoding="utf-8")
    import shutil
    shutil.copytree(os.path.join(REPO_ROOT, "scripts"), str(clone / "scripts"))
    if profile is not None:
        (clone / "channel_profile.json").write_text(
            json.dumps(profile, ensure_ascii=False), encoding="utf-8")
    env = dict(os.environ, CLONE=str(clone))
    # Подпроцесс обязателен: CHANNEL_PROFILE и производные от него списки
    # считаются ОДИН РАЗ на импорте pipeline_smart, и в уже загруженном
    # процессе подмена профиля ничего бы не изменила — тест был бы зелёным
    # по построению.
    out = subprocess.run([sys.executable, "-c", PROBE], capture_output=True,
                         text=True, env=env, cwd=str(clone))
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


class TestFreshCloneInheritsNothing:
    def test_no_query_qualifier_is_injected(self, tmp_path):
        got = _probe_clone(tmp_path)
        assert got["distorted"] == [], got["distorted"]

    def test_no_medieval_rules_or_guards(self, tmp_path):
        got = _probe_clone(tmp_path)
        assert got["rules"] == 0
        assert got["guards"] == 0

    def test_museum_passport_is_not_assumed(self, tmp_path):
        """Окно эпохи и список чужих культур — паспорт предмета. Не объявлены
        — музеи не спрашиваются (см. museum_sources.search_museums), а не
        спрашиваются «под европейское Средневековье 900-1600»."""
        got = _probe_clone(tmp_path)
        assert got["era_declared"] is None
        assert got["cultures_declared"] is None


class TestThisChannelIsUnchanged:
    """Обратная сторона: правка не имеет права ослабить НАСТОЯЩИЙ канал —
    он объявляет те же значения, что раньше были константами."""

    def test_declared_profile_restores_every_list(self, tmp_path):
        profile = json.load(open(os.path.join(REPO_ROOT, "channel_profile.json"),
                                 encoding="utf-8"))
        got = _probe_clone(tmp_path, profile=profile)
        assert got["rules"] == 5
        assert got["guards"] == 1
        assert got["era_declared"] == [900, 1600]
        assert "japanese" in got["cultures_declared"]
        # на своём канале уточнитель обязан работать как раньше
        assert "tank battle field" in got["distorted"]
