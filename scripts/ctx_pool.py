#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Пул потоков, который передаёт задаче контекст того, кто её отправил.

Обычный ThreadPoolExecutor запускает задачу в контексте рабочего потока, и
переменные contextvars отправителя туда не доходят. Упреждающий отбор
(slot_speculation) держит в них всё, что отличает его от настоящего цикла:
«этот вызов спекулятивный» у шлюза (llm_gateway._SPECULATIVE), «идёт
упреждение» у отбора (_SPECULATING), журналы и резерв квоты. Аудит 28.09:
одиннадцать вложенных пулов на пути отбора (сетка судьи, проверка
финалистов, мир и пункты, отсев по подписи, скачивание превью, музеи)
теряли их, и внутри упреждения вызовы шли как живые — оплата сразу, запись
в кэш ответов, мимо ограничителя параллельности. Здесь каждая задача
выполняется в копии контекста отправителя, как в selection_engine.
"""
import concurrent.futures
import contextvars


class ContextThreadPoolExecutor(concurrent.futures.ThreadPoolExecutor):
    def submit(self, fn, /, *args, **kwargs):
        return super().submit(contextvars.copy_context().run, fn, *args, **kwargs)
