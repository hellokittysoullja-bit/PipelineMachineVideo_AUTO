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
def _isolate_from_real_dotenv(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("SHOT_DIRECTOR_MODE", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "")
