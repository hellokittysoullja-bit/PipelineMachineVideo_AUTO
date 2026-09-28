"""Вложенные пулы на пути отбора обязаны передавать контекст упреждения.

Аудит 28.09: обычный ThreadPoolExecutor не передаёт contextvars, и внутри
упреждающего отбора сетка судьи, проверка финалистов, вопрос о мире, отсев
по подписи, скачивание превью и музеи работали как живые вызовы — оплата
сразу, запись в кэш ответов, мимо ограничителя параллельности.
"""
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import ctx_pool  # noqa: E402
import llm_gateway  # noqa: E402


def test_task_sees_submitters_speculation():
    seen = []
    with llm_gateway.speculation():
        with ctx_pool.ContextThreadPoolExecutor(2) as ex:
            seen += list(ex.map(lambda _: llm_gateway.speculative(), range(4)))
    with ctx_pool.ContextThreadPoolExecutor(2) as ex:
        seen.append(ex.submit(llm_gateway.speculative).result())
    assert seen == [True, True, True, True, False]


def test_plain_pool_would_lose_it():
    """Контроль: без передачи контекста задача не видит упреждения —
    ровно поломка, которую закрывает ctx_pool."""
    import concurrent.futures
    with llm_gateway.speculation():
        with concurrent.futures.ThreadPoolExecutor(1) as ex:
            assert ex.submit(llm_gateway.speculative).result() is False


ALLOWED_PLAIN = {
    # внешние пулы самого упреждения и фоновые работы вне отбора: контекст
    # им задают явно или он не нужен
    "slot_prefetch.py": 1, "slot_speculation.py": 1,
    "pipeline_smart.py": 2,   # параллакс-поток и премикс звука
    "selection_engine.py": 1,  # передаёт контекст сам: submit(copy_context().run, ...)
}


def test_no_plain_thread_pool_on_the_selection_path():
    for name in ("pipeline_smart.py", "shot_judge.py", "caption_screen.py", "museum_sources.py",
                 "stock_query_planner.py", "shot_research.py", "shot_generator.py",
                 "commons_source.py", "selection_engine.py", "slot_prefetch.py",
                 "slot_speculation.py"):
        src = open(os.path.join(REPO_ROOT, "scripts", name), encoding="utf-8").read()
        plain = len(re.findall(r"concurrent\.futures\.ThreadPoolExecutor\(", src))
        assert plain <= ALLOWED_PLAIN.get(name, 0), (name, plain)
