"""Общая изоляция тестов от РЕАЛЬНОГО .env этого репозитория.

РЕАЛЬНЫЙ баг, найденный вживую (27.08): pipeline_smart.py делает
load_dotenv() на импорте — до этого коммита GEMINI_API_KEY в .env был
пуст, и resolve_queries()/shot_director молчаливо оставались no-op на
любом тесте, который явно не включал SHOT_DIRECTOR_MODE=on сам. Как
только пользователь вписал в .env реальный рабочий ключ, тот же голый
`pytest tests/` начал ДЕЙСТВИТЕЛЬНО дёргать живой Gemini на тестах,
которые никогда не были рассчитаны на сеть (test_resolve_queries_*) —
поймали HTTP 429 (Too Many Requests) прямо в выводе теста и 3 упавших
теста, чья логика проверяет ИМЕННО fallback-путь БЕЗ LLM-режиссёра.

Автоюз-фикстура ниже — тот же принцип, что уже применён к кэшам Pexels
(_clear_pexels_search_caches в test_parse.py): тест не должен зависеть от
того, что лежит в .env рабочей копии на момент запуска. Файлы, которым
реально нужен GEMINI_API_KEY/SHOT_DIRECTOR_MODE=on (test_shot_director.py
и любой другой, кто явно вызывает monkeypatch.setenv), переопределяют эти
переменные ЛОКАЛЬНО своей autouse-фикстурой — та применяется ПОСЛЕ этой
(conftest.py стоит выше по дереву fixtures, локальная fixture файла
накатывается поверх и побеждает — см. документацию pytest про порядок
autouse). Ничего не ломает существующие тесты, которые уже сами
управляют этими переменными."""
import pytest


@pytest.fixture(autouse=True)
def _isolate_from_real_dotenv(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("SHOT_DIRECTOR_MODE", raising=False)
    # VLM_ARBITER_MODE — по той же причине, но с обратным знаком: его дефолт
    # в реестре (scripts/feature_flags.py) "on", и тест, проверяющий
    # поведение ПО УМОЛЧАНИЮ, не должен зависеть от того, выставил ли
    # пользователь в своём .env "off". Живого вызова это не открывает —
    # GEMINI_API_KEY ниже всё равно пуст, арбитр fail-open выходит сразу.
    monkeypatch.delenv("VLM_ARBITER_MODE", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "")
    # OPENVERSE_ENABLED — тот же класс бага, найден живьём 07.09 при
    # подключении Openverse к главному пути отбора. У этого канала в
    # рабочем .env стоит OPENVERSE_ENABLED=1, и тест, который сам не
    # выставляет режим (test_base_min_pool.py), тихо делал ЖИВОЙ сетевой
    # запрос к api.openverse.org внутри юнит-теста — реальный кандидат из
    # архива победил трёх поддельных кандидатов теста, и ассерт на "победил
    # самый эстетичный из ЧЕТЫРЁХ" сломался на пятом, ниоткуда не взявшемся
    # источнике. Дефолт реестра и так "0" — здесь только гасим то, что
    # рабочая копия могла включить поверх дефолта, тем же принципом, что и
    # GEMINI_API_KEY выше.
    monkeypatch.delenv("OPENVERSE_ENABLED", raising=False)
    monkeypatch.setenv("OPENVERSE_ENABLED", "0")
    # MUSEUM_SOURCES_ENABLED — ровно тот же класс бага, и здесь он опаснее:
    # дефолт реестра у него "1" (источник включён в проде), то есть без этой
    # строки КАЖДЫЙ тест, дошедший до сборки пула, ходил бы живьём в три
    # музейных API — а Met на каждый предмет делает отдельный запрос.
    monkeypatch.delenv("MUSEUM_SOURCES_ENABLED", raising=False)
    monkeypatch.setenv("MUSEUM_SOURCES_ENABLED", "0")
    # MET_CATALOG — тот же приём и та же причина, но дефект здесь другого
    # рода и найден сразу пятью упавшими тестами: каталог читает РЕАЛЬНЫЙ
    # индекс с диска (temp_met_catalog/index.json). Тест, который замокал
    # _met_get и ждёт свои три objectID, получал к ним тридцать настоящих
    # и падал — не потому, что код сломан, а потому что на машине, где
    # индекс собран, тесты переставали быть герметичными. Дефолт реестра 1.
    monkeypatch.delenv("MET_CATALOG", raising=False)
    monkeypatch.setenv("MET_CATALOG", "0")
    # SHELF_INDEX — та же причина, что у MET_CATALOG строкой выше, только
    # последствие тяжелее: полка читает с диска матрицу эмбеддингов И грузит
    # SigLIP2, чтобы посчитать запрос. Тест, дошедший до сборки пула, на
    # машине с собранной полкой тянул бы модель на 4.3 ГБ и десятки секунд
    # на каждый вызов. Дефолт реестра 1.
    monkeypatch.delenv("SHELF_INDEX", raising=False)
    monkeypatch.setenv("SHELF_INDEX", "0")
    # Дисковые кэши поиска (музеи, Openverse) — в СВОЮ папку на тест: иначе
    # положительный тест с реалистичным ответом кладёт результат в общий
    # temp_*_cache/, а следующий тест «сеть упала -> пусто» получает из кэша
    # прошлый ответ и падает (поймано 13.09 на test_openverse_live_path).
    monkeypatch.setenv("MUSEUM_CACHE_DIR", str(tmp_path / "museum_cache"))
    monkeypatch.setenv("OPENVERSE_CACHE_DIR", str(tmp_path / "openverse_cache"))
    try:
        import museum_sources as _ms
        monkeypatch.setattr(_ms, "MUSEUM_CACHE_DIR", str(tmp_path / "museum_cache"))
    except Exception:
        pass
    try:
        import pipeline_smart as _ps
        monkeypatch.setattr(_ps, "OPENVERSE_CACHE_DIR", str(tmp_path / "openverse_cache"), raising=False)
    except Exception:
        pass
    # PIXABAY_ENABLED/UNSPLASH_ENABLED — та же причина и тот же дефолт-1, что у
    # музеев: без этих двух строк любой тест, дошедший до сборки пула, ходил бы
    # живьём в Pixabay/Unsplash, если в окружении вдруг оказался ключ.
    for _flag in ("PIXABAY_ENABLED", "UNSPLASH_ENABLED"):
        monkeypatch.delenv(_flag, raising=False)
        monkeypatch.setenv(_flag, "0")


def pytest_configure(config):
    """Регистрация маркера `slow`.

    Метрика по золотому набору (tests/test_golden_set_metric.py) прогоняет
    реальный CLIP по 40 кадрам — это десятки секунд, а не миллисекунды.
    Маркер нужен, чтобы такой прогон можно было исключить одной командой
    (`pytest -m "not slow"`) при быстрой итерации, и чтобы pytest не ругался
    на незарегистрированный маркер.
    """
    config.addinivalue_line(
        "markers", "slow: живые ML-прогоны, десятки секунд и дольше")
