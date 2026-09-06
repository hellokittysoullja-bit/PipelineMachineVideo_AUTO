"""Кэш кандидата обязан знать о КОНФИГУРАЦИИ отбора, а не только о функциях гейтов.

Фон (реальный, измеренный симптом, 04.09 — разбор опубликованного эпизода
01_ves-mecha): candidate_gate_signature() хэшировала исходники функций-гейтов и
пороги, но НЕ режимы слоёв, реально выбирающих победителя. Из-за этого включение
VLM-арбитра и Semantic Visual Director (.env: VLM_ARBITER_MODE=on,
VISUAL_DIRECTOR_MODE=assist) не меняло ключ кэша НИ ОДНОГО слота — уже скачанный
кандидат, отобранный до появления этих слоёв голым косинусом эмбеддингов, молча
отдавался как есть при каждом следующем прогоне.

Наблюдаемое следствие на готовом ролике: соседние кадры объективно выбраны
разными алгоритмами, и никакое улучшение отбора не доходило до уже собранных
эпизодов. Это и есть механика жалобы "то работает, то не работает".

Эти тесты падают на коде ДО правки (проверено запуском на предыдущей ревизии) и
защищают от тихого возврата того же класса бага.
"""
import os
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")


def _gate_sig(env_overrides):
    """Подпись гейта в СВЕЖЕМ процессе.

    Отдельный процесс, а не importlib.reload(): candidate_gate_signature()
    мемоизирует результат в модульную глобаль _CANDIDATE_GATE_SIG, и в одном
    процессе вторая конфигурация вернула бы закэшированное значение первой —
    тест бы "проходил" по совершенно неверной причине.
    """
    env = dict(os.environ)
    env.update(env_overrides)
    env["PYTHONPATH"] = SCRIPTS_DIR + os.pathsep + env.get("PYTHONPATH", "")
    code = (
        "import sys, tempfile; "
        "sys.argv = ['pipeline_smart.py', tempfile.gettempdir()]; "
        "import pipeline_smart as ps; "
        "print(ps.candidate_gate_signature())"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True,
        cwd=REPO_ROOT, timeout=300,
    )
    assert out.returncode == 0, f"дочерний процесс упал:\n{out.stderr[-2000:]}"
    return out.stdout.strip().splitlines()[-1]


def test_arbiter_mode_changes_candidate_cache_key():
    """VLM_ARBITER_MODE выбирает победителя напрямую -> обязан менять ключ кэша."""
    on = _gate_sig({"VLM_ARBITER_MODE": "on", "VISUAL_DIRECTOR_MODE": "off"})
    off = _gate_sig({"VLM_ARBITER_MODE": "off", "VISUAL_DIRECTOR_MODE": "off"})
    assert on != off, (
        "включение/выключение VLM-арбитра не меняет ключ кэша кандидата — "
        "значит уже скачанный кадр, выбранный без арбитра, продолжит "
        "отдаваться на кэш-хите, и арбитр физически не повлияет на эпизод"
    )


def test_visual_director_mode_changes_candidate_cache_key():
    """VISUAL_DIRECTOR_MODE=assist подменяет base_winner на director_winner."""
    assist = _gate_sig({"VLM_ARBITER_MODE": "off", "VISUAL_DIRECTOR_MODE": "assist"})
    off = _gate_sig({"VLM_ARBITER_MODE": "off", "VISUAL_DIRECTOR_MODE": "off"})
    assert assist != off, (
        "включение Semantic Visual Director не инвалидирует кэш кандидата"
    )


def test_all_four_mode_combinations_are_distinct():
    """Комбинации режимов не должны коллидировать между собой.

    Проверка именно КОМБИНАЦИЙ, а не только каждого флага по отдельности:
    наивная реализация (например, склейка через сумму или xor булевых) дала бы
    одинаковую подпись для (on, off) и (off, on).
    """
    sigs = {
        (arb, dr): _gate_sig({"VLM_ARBITER_MODE": arb, "VISUAL_DIRECTOR_MODE": dr})
        for arb in ("on", "off")
        for dr in ("assist", "off")
    }
    assert len(set(sigs.values())) == 4, f"подписи коллидируют: {sigs}"


def test_shot_director_mode_deliberately_not_in_signature():
    """SHOT_DIRECTOR_MODE сознательно НЕ входит в подпись — фиксируем решение.

    Он влияет только на ТЕКСТ запроса, а запрос и весь пул запросов секции уже
    входят в qkey/qhash имени кэш-файла (см. pexels_photo() у qkey) — то есть
    инвалидация происходит и без него. Если добавить его в подпись, каждое
    переключение флага гарантированно перекачивало бы весь эпизод, ничего при
    этом не исправляя.

    Тест — защита от "улучшения на автопилоте": тот, кто решит добавить флаг
    сюда, обязан сначала осознанно удалить этот тест и объяснить почему.
    """
    on = _gate_sig({"SHOT_DIRECTOR_MODE": "on", "VLM_ARBITER_MODE": "off",
                    "VISUAL_DIRECTOR_MODE": "off"})
    off = _gate_sig({"SHOT_DIRECTOR_MODE": "off", "VLM_ARBITER_MODE": "off",
                     "VISUAL_DIRECTOR_MODE": "off"})
    assert on == off, (
        "SHOT_DIRECTOR_MODE попал в candidate_gate_signature(); его эффект уже "
        "покрыт qhash запроса, и это лишняя полная перезакачка эпизода"
    )


def test_pool_size_constants_are_in_signature():
    """Размер пула — часть решения, а не деталь производительности.

    Кадр, выбранный из двух кандидатов, и кадр, выбранный из восьми — решения
    разной силы. Переиспользовать первое после расширения пула значит молча
    остаться на более бедном выборе. Проверяем через реальный текст подписи, а
    не через монкейпатч константы: подпись считается в дочернем процессе на
    настоящем модуле.
    """
    sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]
    sys.path.insert(0, SCRIPTS_DIR)
    import pipeline_smart as ps

    raw = ps._selection_stack_signature()
    for const in (ps.DIRECTOR_MIN_POOL, ps.PHOTO_DEDUP_MAX_TRIES,
                  ps.FAST_MODE_START_INDEX, ps.FAST_DIRECTOR_MIN_POOL,
                  ps.FAST_PHOTO_DEDUP_MAX_TRIES):
        assert str(const) in raw, (
            f"константа пула {const} не попала в _selection_stack_signature(): {raw}"
        )
