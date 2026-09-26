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
import os

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
    # COMMONS_ENABLED — тот же класс: дефолт реестра 1, тест, дошедший до
    # сборки пула, ходил бы живьём в Wikimedia.
    monkeypatch.setenv("COMMONS_ENABLED", "0")
    # RESEARCH_ROUND — второй круг поиска зовёт модель через шлюз; тест,
    # которому он нужен, включает его сам.
    monkeypatch.setenv("RESEARCH_ROUND", "0")
    # CAPTION_SCREEN — отсев по подписи зовёт DeepSeek через шлюз; тест,
    # которому он нужен, включает его сам.
    monkeypatch.setenv("CAPTION_SCREEN", "0")
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
    monkeypatch.setenv("COMMONS_CACHE_DIR", str(tmp_path / "commons_cache"))
    try:
        import commons_source as _cs
        monkeypatch.setattr(_cs, "CACHE_DIR", str(tmp_path / "commons_cache"))
    except Exception:
        pass
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
    # SMART_RELEVANCE_VETO — ТОТ ЖЕ класс, найден живьём 17.09 при установке
    # torch/transformers в СЕССИЮ (не в постоянное окружение): дефолт флага
    # 1, и без этой строки любой тест, дошедший до pexels_photo()/
    # pexels_video() на машине, где эти пакеты УЖЕ стоят (например, для
    # другой работы в этой же сессии), реально гонял бы SigLIP2+Jina по
    # синтетическим тестовым фикстурам — те никогда не были рассчитаны на
    # настоящую семантическую проверку и получали бы честный отказ модели,
    # ломая тесты, которые проверяют совсем другое. Ровно так и произошло:
    # 11 тестов упали в первом же прогоне после установки torch в контейнер.
    monkeypatch.delenv("SMART_RELEVANCE_VETO", raising=False)
    monkeypatch.setenv("SMART_RELEVANCE_VETO", "0")
    # SHOT_JUDGE — платный судья кадров через шлюз. Дефолт реестра 1, и на
    # машине с LLM_GATEWAY_API_KEY в окружении любой тест, дошедший до
    # выбора победителя, ходил бы в сеть и тратил деньги владельца.
    monkeypatch.delenv("SHOT_JUDGE", raising=False)
    monkeypatch.setenv("SHOT_JUDGE", "0")
    # Пустая строка, а не удаление — тот же приём, что у GEMINI_API_KEY выше.
    # Удалённую переменную load_dotenv() в дочернем процессе рендера (тесты
    # запускают pipeline_smart.py подпроцессом) возвращает из рабочего .env:
    # 24.09 test_select_only так ходил в шлюз за паспортом мира и
    # спецификациями кадров — платно, и результат теста зависел от ответа
    # модели. Существующую пустую переменную load_dotenv() не перезаписывает.
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", "")


@pytest.fixture(scope="session")
def _suite_working_root(tmp_path_factory):
    return tmp_path_factory.mktemp("suite_work")


@pytest.fixture(autouse=True)
def _private_working_folders(monkeypatch, _suite_working_root):
    """Рабочая папка эпизода задаётся ФИКСТУРОЙ, а не подменой sys.argv.

    РЕАЛЬНЫЙ, измеренный 15.09 дефект, а не гигиена: `VIDEO_FOLDER` и
    `TEMP_FOLDER` вычисляются из `sys.argv` ОДИН РАЗ на импорте
    pipeline_smart, и 69 тест-файлов подменяют `sys.argv` перед импортом,
    чтобы увести кэш во временную папку. Работает ровно ОДНА подмена — та,
    что случилась первой; у остальных 68 строка не делает ничего, и
    рабочей папкой всей суиты становится argv первого по алфавиту файла,
    то есть путь к чужому `.py`. Симптом: `tests/test_thumbnail_first.py`
    в одиночку зелёный, а после соседнего файла — четыре падения с
    `NotADirectoryError: tests/test_brief_stock_query.py/temp_smart`.
    Суита, результат которой зависит от порядка файлов, не защищает ни от
    чего — и именно так четыре красных теста доехали до HEAD незамеченными.

    Фикстура не отменяет ничьих настроек: тест, который задаёт папку сам,
    накатывает свой `monkeypatch` ПОВЕРХ этого и побеждает. Заодно ни один
    тест больше не может писать кэш в рабочую копию репозитория.

    Папка ОДНА НА СЕССИЮ, а не на каждый тест, и это измеренный выбор, а не
    небрежность. Дефект, ради которого фикстура существует, — в том, ЧТО
    это за путь (чужой `.py` вместо папки), и он закрывается общей
    временной папкой полностью: порядок файлов больше ни на что не влияет.
    Отдельная папка НА КАЖДЫЙ тест добавила бы сверх этого только защиту
    от утечки через ТЁПЛЫЙ КЭШ между тестами — утечки, ни одного случая
    которой замер не показал, — и стоила бы реального времени: кэш клипов
    перестаёт переиспользоваться, и каждый тест рендера зовёт ffmpeg
    заново (прогон замедлялся примерно вдвое). Ставить дорогую защиту от
    отказа, которого не наблюдали, — ровно то, за что этот файл уже
    критиковал сам себя (User-Agent для Викимедиа, троттлинг Europeana).
    Появится измеренный случай утечки — фикстуре достаточно сменить
    scope."""
    import sys
    ps = sys.modules.get("pipeline_smart")
    if ps is not None:
        root = str(_suite_working_root)
        monkeypatch.setattr(ps, "VIDEO_FOLDER", root, raising=False)
        monkeypatch.setattr(ps, "TEMP_FOLDER", os.path.join(root, "temp_smart"),
                            raising=False)
    yield


@pytest.fixture(autouse=True)
def _clear_process_level_search_caches():
    """Кэши поиска живут НА ПРОЦЕСС — значит переживают границу теста.

    ЧЕСТНО О ПРОИСХОЖДЕНИИ: это была ГИПОТЕЗА о причине падений
    `tests/test_thumbnail_first.py` после соседнего файла, и замер её НЕ
    подтвердил — очистка кэшей не починила ни одного из четырёх падений,
    настоящей причиной оказался `TEMP_FOLDER`, указывавший на чужой `.py`
    (см. фикстуру выше). Фикстура оставлена не «на всякий случай», а
    потому что сам факт проверяем и от результата гипотезы не зависит:
    `_MUSEUM_SEARCH_CACHE`/`_SHELF_SEARCH_CACHE`/`_PEXELS_VIDEO_SEARCH_CACHE`
    действительно живут на процесс, и заполненный одним файлом кэш
    действительно отдаёт готовый ответ другому — тогда
    `monkeypatch.setattr(ps, "_museum_search_photos", ...)` следующего
    теста не вызывается ВООБЩЕ, и тест молча проверяет чужие данные.
    Наблюдённого случая такой утечки пока нет; стоит она ноль.

    Список кэшей НЕ перечисляется руками: ровно такой список (в фикстуре
    одного файла чистился только `_PEXELS_SEARCH_CACHE` из девяти) и
    отстал. Берётся по ФОРМЕ имени — правило переживает добавление
    следующего источника, а перечисление не переживало."""
    import sys
    ps = sys.modules.get("pipeline_smart")
    if ps is None:
        # Тест, который pipeline_smart вообще не импортировал, не должен
        # платить за его импорт (там тяжёлый стек и load_dotenv).
        yield
        return
    for mod in (ps, sys.modules.get("museum_sources")):
        for name in dir(mod or ()):
            if name.startswith("_") and name.endswith("CACHE"):
                obj = getattr(mod, name, None)
                if isinstance(obj, dict):
                    obj.clear()
    try:
        ps.reset_source_stats()
    except Exception:
        pass
    yield


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
