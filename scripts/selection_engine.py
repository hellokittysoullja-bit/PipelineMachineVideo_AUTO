#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ядро отбора кадра: одна последовательность шагов для любого вида медиа.

ПОЧЕМУ ОТДЕЛЬНО. Фото и видео годами отбирались двумя рукописными копиями
одного алгоритма (pexels_photo и pexels_video), и каждая правка одной копии
рисковала не доехать до другой. CLAUDE.md документирует семь таких случаев
(«починили фото, забыли видео»): жанровый фильтр, скоринг Режиссёра, бриф
в пуле, пол пула, проверка анти-дубля на кэш-хите, чередование источников,
провенанс в sidecar. Ядро — это шаги, общие по построению; различия видов
медиа живут в адаптере (MediaAdapter), и ядро НЕ ЗНАЕТ, какой вид медиа
перед ним: в этом модуле нет ни одной ветки по виду и ни одного импорта
pipeline_smart или адаптеров (tests/test_selection_engine.py держит оба
правила по AST).

ЗАПРОС СЛОТА. Всё, что отбор знает о слоте, приходит одним неизменяемым
SlotRequest без значений по умолчанию. Раньше добытчик принимал полтора
десятка именованных аргументов со значениями по умолчанию, и забытый
аргумент молча становился None: спасающий вызов фото после негодного
видео годами шёл БЕЗ брифа фразы (5 слотов из 8 в эпизоде 94), и никто
этого не видел. Запрос строится в main() один раз на слот и передаётся
всем попыткам слота как есть.

ШАГИ:
  1. путь в кэше; готовый файл -> адаптер решает, годен ли кэш-хит;
  2. путь переводится в стейджинг попытки (selection_attempt): в кэш файл
     попадёт только коммитом;
  3. пул: запрос из брифа первым, затем запрос слота и запросы секции;
     по каждому — кандидаты источников ПО КРУГУ, затем запросы по кругу;
     дубль по id отбрасывается при первом появлении; жанровый фильтр по
     тексту кандидата; каждому оставшемуся — «предложен» в счёт источника;
  4. пустой пул — честное «кандидатов нет», а не сбой источника;
  5. порядок, перебор, ранжирование и материализация победителя — через
     адаптер (этап 2 перестройки; этапы 3-4 переносят их в ядро).
"""
import dataclasses
import itertools
import os

import selection_attempt


@dataclasses.dataclass(frozen=True)
class SlotRequest:
    """Всё, что отбор знает о слоте. Значений по умолчанию нет намеренно:
    забытое поле — ошибка конструирования, а не тихий None.

    Контейнеры анти-дубля и ритма (used_*, recent_sizes) — общее состояние
    эпизода: отбор их ЧИТАЕТ, меняет только коммит (close_slot)."""
    index: int
    query: str
    extra_queries: tuple
    text_key: object
    shot_brief: object
    # Спецификация кадра фразы (stock_query_planner v3): фокус, утверждения
    # по убыванию важности, запросы с целями. Проверка финалиста спрашивает
    # её утверждения, ранжирование сравнивает кадры по ним. Нет плана — None.
    shot_spec: object
    block_text: object
    arbiter_text: object
    is_opening: bool
    slot_dur: object
    action_qualifier: object
    target_luma: object
    director_score_fn: object
    director_assist: bool
    director_report: object
    video_score_fn: object
    used_photo_ids: object
    used_video_ids: object
    used_hashes: object
    recent_sizes: object


class MediaAdapter:
    """Различия вида медиа. Ядро зовёт только эти методы."""

    kind = None

    def cache_path(self, request):
        raise NotImplementedError

    def cache_hit(self, request, path):
        """Файл уже в кэше. Вернуть путь (кэш годен) или None (ставим как
        промах и отбираем заново)."""
        raise NotImplementedError

    def brief_query(self, request):
        raise NotImplementedError

    def sources(self, request, pool_query):
        """Список списков кандидатов, по одному на источник, для одного
        запроса пула; кандидаты уже помечены _origin_query."""
        raise NotImplementedError

    def filter_pool(self, request, pool):
        raise NotImplementedError

    def note_offered(self, pool):
        raise NotImplementedError

    def choose(self, request, pool, cf):
        """Порядок, перебор, ранжирование, материализация победителя в cf,
        эффекты попытки. Возвращает путь (cf) или None."""
        raise NotImplementedError

    def on_failure(self, request, exc):
        raise NotImplementedError


def pool_queries(request, brief_query):
    """Запросы пула по порядку: бриф фразы (если есть и не совпадает),
    запрос слота, запросы секции без повторов."""
    queries = [request.query] + [q for q in request.extra_queries
                                 if q and q != request.query]
    if brief_query and brief_query not in queries:
        queries = [brief_query] + queries
    return queries


def round_robin(lists):
    """Элементы списков по кругу; None-заполнители zip_longest пропускаются."""
    out = []
    for row in itertools.zip_longest(*lists):
        out.extend(x for x in row if x is not None)
    return out


def unique_by_id(candidates):
    """Первое появление каждого id; порядок сохраняется."""
    seen, out = set(), []
    for c in candidates:
        cid = c.get("id")
        if cid in seen:
            continue
        seen.add(cid)
        out.append(c)
    return out


def query_tiers(request, queries):
    """Запросы пула по ярусам: сначала запросы спецификации кадра этой фразы
    (stock_query_planner v3) в её порядке, затем — всё остальное: перевод
    брифа, общие запросы секции. Внутри яруса — по кругу, ярусы — друг за другом:
    без этого запросы секции, общие на 6-9 слотов, занимали 75-85% пула
    (docs/quality/POOL_RECALL_EP94.md) и вытесняли кадры фокуса фразы.
    Нет спецификации — один ярус, порядок прежний."""
    spec = getattr(request, "shot_spec", None)
    if not spec:
        return [queries]
    own = [x["q"] for x in spec.get("queries") or []]
    first = [q for q in own if q in queries] or [request.query]
    rest = [q for q in queries if q not in first]
    return [t for t in (first, rest) if t]


def build_pool(request, adapter):
    tiers = query_tiers(request, pool_queries(request, adapter.brief_query(request)))
    pool = unique_by_id([c for tier in tiers
                         for c in round_robin([round_robin(adapter.sources(request, pq))
                                               for pq in tier])])
    if not pool:
        return []
    pool = adapter.filter_pool(request, pool)
    adapter.note_offered(pool)
    return pool


def select(request, adapter):
    """Отобрать медиа слота. Работает внутри попытки selection_attempt:
    состояние эпизода не меняет, только записывает эффекты."""
    final = adapter.cache_path(request)
    if os.path.exists(final) and os.path.getsize(final) > 0:
        hit = adapter.cache_hit(request, final)
        if hit is not None:
            return hit
    cf = selection_attempt.stage_path(final)
    try:
        pool = build_pool(request, adapter)
        if not pool:
            return None
        return adapter.choose(request, pool, cf)
    except Exception as exc:  # noqa: BLE001 — адаптер решает, чей это сбой
        adapter.on_failure(request, exc)
        return None
