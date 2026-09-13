#!/usr/bin/env python3
"""Единый реестр флагов режимов пайплайна (источник истины для дефолтов).

ЗАЧЕМ ОТДЕЛЬНЫЙ МОДУЛЬ. До него дефолт каждого режима был вписан ЛИТЕРАЛОМ
в КАЖДОЙ точке чтения — девять мест в пяти файлах (pipeline_smart.py x6,
shot_director.py x3, look_reference.py, speech_planner.py,
stock_fetch_multisource.py). Две РЕАЛЬНЫЕ, найденные вживую (03.09) поломки
ровно этого устройства — не гипотетические риски:

1. **VLM_ARBITER_MODE.** CLAUDE.md (ЧАСТЬ 14) объявляет «дефолт `on`,
   реализован и живьём проверен 29 августа», а все три точки чтения в коде
   подставляли `"off"` — и `config.example.env` тоже. То есть на любом
   рендере, где переменная не выставлена руками, VLM-арбитра не было вообще,
   при том что документация канала считала его работающим. Документация и код
   разошлись молча: ни один тест не сверял их между собой.

2. **DEFLICKER_ENABLED.** CLAUDE.md документирует этот флаг как «быстрый
   откат без редеплоя», а код читал переменную с ДРУГИМ именем —
   `DEFLICKER`. Человек, выставивший `DEFLICKER_ENABLED=0` ровно по
   документации, деффликер не отключал: флаг отката не работал по своему же
   документированному имени.

Поэтому: дефолт объявляется здесь ОДИН раз, точки чтения спрашивают реестр,
а `tests/test_feature_flags.py` сверяет реестр с тем, что обещает CLAUDE.md,
и с `config.example.env`. Расхождение теперь падает тестом, а не тихо живёт
в проде месяцами.

СЕМАНТИКА (сохранена байт-в-байт с тем, что было до реестра — модуль не
меняет поведение ни одного флага, только убирает дублирование дефолта):

* Значение читается из окружения при КАЖДОМ вызове, не кэшируется на импорте
  — иначе monkeypatch в тестах и любой код, выставляющий режим после импорта
  модуля, видели бы застывший снимок (та же причина, по которой
  shot_director.py уже проверяет `os.environ` внутри функций, а не
  модуль-константой, см. комментарий у direct_query()).
* Нормализация — `.strip().lower()`, как во всех прежних точках чтения.
* Значение ВНЕ списка допустимых → громкое предупреждение и откат на "off"
  (не на дефолт!). Именно так вело себя всё до реестра: сравнение вида
  `... == "on"` на мусорном значении давало «выключено», а look_reference.py
  делал этот откат явно. Опечатка не должна ВКЛЮЧАТЬ дорогой слой.
* Булевы флаги (`_ENABLED`/`_GATE`) — «включено, если значение не "0"», ровно
  как `os.environ.get(..., "1") != "0"` раньше. Список синонимов вроде
  "off"/"false" сюда СОЗНАТЕЛЬНО не добавлен: сегодня `RENDER_STRICT_GATE=off`
  означает «включено», и тихо перевернуть это значило бы ослабить жёсткий
  гейт сборки у того, кто уже так написал.
"""
import os
import sys


class Flag:
    """Описание одного флага. `aliases` — исторические имена переменной,
    которые продолжают работать (см. DEFLICKER ниже): читаются ПОСЛЕ
    основного имени, только если основное не выставлено."""

    def __init__(self, name, default, allowed=None, aliases=(), summary=""):
        self.name = name
        self.default = default
        self.allowed = tuple(allowed) if allowed else None
        self.aliases = tuple(aliases)
        self.summary = summary

    @property
    def is_boolean(self):
        return self.allowed is None


FLAGS = {f.name: f for f in (
    # --- режимы (строковые) ---
    Flag("VLM_ARBITER_MODE", "on", ("off", "on"),
         summary="VLM-арбитр шорт-листа хука (Gemini, только HOOK, fail-open без ключа)"),
    Flag("SHOT_DIRECTOR_MODE", "off", ("off", "on"),
         summary="LLM-режиссёр запросов для блоков без своего запроса и без словаря"),
    Flag("VISUAL_DIRECTOR_MODE", "off", ("off", "shadow", "assist"),
         summary="Semantic Visual Director — реранк пула кандидатов по смыслу фразы"),
    Flag("LOOK_MANAGEMENT_MODE", "off", ("off", "shadow", "assist"),
         summary="Reference-Guided Look Management — коррекция кадра к эталону канала"),
    Flag("GRAIN_BLEND_MODE", "softlight", ("softlight", "grainmerge", "expr"),
         summary="Наложение зерна: softlight (нативный, быстрый) / grainmerge / expr (прежняя формула, медленно)"),
    Flag("DELIVERY_PROFILE", "youtube", ("youtube", "archive", "hevc"),
         summary="Финальный проход: youtube (VBV-потолок 12 Мбит/с) / archive (без потолка) / hevc (libx265)"),
    Flag("DOMAIN_GRADE_MODE", "on", ("off", "on"),
         summary="Доменная модуляция грейда (DOMAIN_WARM_PUSH_SCALE) — теплота по содержанию кадра"),
    # --- булевы ---
    Flag("RENDER_STRICT_GATE", "1", aliases=(),
         summary="Не собирать final.mp4, если хоть один клип не принят"),
    Flag("DEFLICKER_ENABLED", "1", aliases=("DEFLICKER",),
         summary="Деффликер стокового ВИДЕО перед творческим грейдом"),
    # Тот же случай, что DEFLICKER, найден продолжением аудита 03.09: выключатель
    # зерна жил под именем GRAIN и не был документирован в CLAUDE.md ВООБЩЕ — при
    # том что рядом задокументированы GRAIN_OPACITY и GRAIN_BLEND_MODE, и что
    # зерно даёт +50-60% к весу клипа (замер 03.09). Человек, ищущий в
    # документации, как его выключить, не находил ничего. Старое имя — псевдоним.
    Flag("GRAIN_ENABLED", "1", aliases=("GRAIN",),
         summary="Плёночное зерно на каждом кадре (ассет assets/grain/grain_loop.mp4)"),
    Flag("OPENVERSE_ENABLED", "0",
         summary="Openverse (CC0 + институциональные источники) в ротации фото-источников"),
    # Прямые API музеев вместо агрегатора: паспорт предмета (дата + культура)
    # приходит вместе со снимком, поэтому анахронизм отсекается ЗНАНИЕМ, а не
    # догадкой по пикселям — см. докстринг museum_sources.py.
    Flag("MUSEUM_SOURCES_ENABLED", "1",
         summary="Прямые открытые API музеев (Met/Cleveland/Chicago), фильтр по эпохе и культуре"),
    # Ровно тот же класс пробела, что уже находили у Openverse 07.09 ("код
    # существовал, был включён, и не давал ролику ничего"): Pixabay и Unsplash
    # реализованы в stock_fetch_multisource.py полностью, их ключи стоят в
    # config.example.env, CLAUDE.md описывает мультисток как дефолт заполнения
    # — а в pipeline_smart.py (реальный путь отбора эпизода) не было НИ ОДНОГО
    # упоминания обоих. Вклад в ролик — ноль.
    # Дефолт 1, потому что без ключа обе функции возвращают пустой список и
    # не меняют ничего вообще; с ключом кандидаты идут в ТОТ ЖЕ пул под ТЕ ЖЕ
    # гейты, что и Pexels. Квоты у них СВОИ (Pixabay ~100 запросов/мин,
    # Unsplash demo 50/час) — то есть пул растёт, не тратя лимит Pexels
    # (200/час), который и есть реальное ограничение этого пайплайна.
    # Шаг 4 протокола (stock_fetch_multisource.py) кладёт в media/ файлы
    # {idx:03d}_stock.jpg, а local_photo() отдаёт их слоту ПО НОМЕРУ — раньше
    # Pexels. В самом stock_fetch_multisource.py нет НИ ОДНОГО упоминания
    # relevance/CLIP/домен-гварда/негативного вето: то есть выполнение
    # документированного Шага 4 отключало весь стек гейтов для каждого
    # закрытого им слота. Гейт применяется ТОЛЬКО к машинно-подобранным
    # *_stock.* — AI-картинки (_flow/_grok/_fastgen/_ai) курирует человек и
    # по ЧАСТИ 14 они не заменяются никогда.
    # Музеи/архивы дали пулу портретный материал (страницы кодексов, эффигии,
    # доспехи в рост), а increase+crop срезает у него 58-68% высоты — то есть
    # подлинник показывался узкой полосой из середины. Подложка включается
    # только на источниках уже 4:3; всё, что шире (3:2 Pexels, 16:9), идёт
    # прежним путём байт-в-байт.
    # Случайная (по хэшу имени файла) вариация контраста/насыщенности/яркости
    # в film_look(). Замер на 40 кадрах опубликованного эпизода: разброс
    # яркости 169.3 -> 174.3, средний скачок насыщенности между соседними
    # кадрами 16.5 -> 17.5. То есть джиттер РАСШИРЯЕТ ровно тот разнобой
    # источников, который остальная цепочка сводит (насыщенность: 51.2 -> 16.5,
    # втрое). Задачу «не гнать один рецепт на каждое фото» и без него решают
    # три СОДЕРЖАТЕЛЬНЫХ модулятора — _scene_bias, DOMAIN_GRADE_MODE и Look
    # Management; хэш-джиттер был единственным, у которого нет связи с кадром.
    # Дефолт 0: выключен по измерению, но код оставлен под флагом.
    Flag("GRADE_HASH_JITTER", "0",
         summary="Случайная вариация контраста/сатурации/яркости по хэшу файла в film_look()"),
    Flag("ASPECT_FIT_BACKDROP", "1",
         summary="Портретный источник вписывается целиком на размытую тёмную подложку вместо кропа"),
    Flag("LOCAL_STOCK_GATE", "1",
         summary="Файлы media/*_stock.* из Шага 4 проходят тот же relevance-гейт, что и Pexels"),
    Flag("PIXABAY_ENABLED", "1",
         summary="Pixabay (фото и видео) в общем пуле кандидатов, своя квота"),
    Flag("UNSPLASH_ENABLED", "1",
         summary="Unsplash (фото) в общем пуле кандидатов, своя квота 50/час"),
    # Среди видео-слотов брак 63% против 23% среди фото (замер эпизода 02):
    # видео-корпус стока на исторические темы тоньше, а музеи видео не дают.
    Flag("VIDEO_PHOTO_RESCUE", "1",
         summary="Заведомо негодное видео слота заменяется фотографией до карточки-фолбэка"),
    Flag("STRESS_HINTS_ENABLED", "0",
         summary="Подсказки по ударению омографов в speech_plan_annotated.txt"),
    # Флаги ниже жили как "магические" os.environ.get() мимо реестра
    # (найдено разбором 04.09). Последствие было не косметическим: они не
    # попадали в snapshot() -> media_plan/feature_flags.json, и по артефактам
    # готового ролика нельзя было ответить на вопросы "была ли в нём музыка",
    # "с каким грейдом он собран", "работал ли CLIP-гейт вообще". Ровно тот
    # класс расхождения код-документация, ради которого реестр и заводился.
    # ВАЖНО: у части из них к переменной окружения добавлено ещё и условие
    # наличия ассета на диске (зерно, музыка, щелчки) — реестр отвечает только
    # за окружение, условие файла остаётся на месте вызова.
    Flag("DOF_BLUR", "1",
         summary="Имитация глубины резкости в parallax_kenburns (только фото-параллакс)"),
    Flag("CLIP_RELEVANCE", "1",
         summary="CLIP-гейт релевантности/анахронизмов кандидата; off -> кадры не проверяются"),
    Flag("MUSIC_BED", "1",
         summary="Музыкальная подложка (нужен assets/music/ambient_bed*.flac)"),
    Flag("MASTER_LIMITER", "1",
         summary="Лимитер в конце мастер-цепочки, после loudnorm (страховка по пикам)"),
    Flag("FALLBACK_CARD", "1",
         summary="Процедурная карточка вместо заведомо плохого кадра (отказ арбитра / пустой сток)"),
    Flag("NEGATIVE_VETO", "1",
         summary="Контрастивное вето по ловушкам-негативам (современный спорт/толпа/улица в историческом кадре)"),
    Flag("VOICE_PROCESS", "1",
         summary="Обработка голоса: highpass/EQ/де-эссер/компрессор перед миксом"),
    Flag("TYPEWRITER_CLICKS", "1",
         summary="Щелчки печатной машинки под stat-плашкой варианта 5 (нужны assets/sfx/keyboard_clicks/)"),
    Flag("ON_SCREEN_TEXT", "1",
         summary="Отрисовка титров/stat-плашек поверх кадра (текст сценария не трогается)"),
)}

_warned = set()


def _spec(name):
    try:
        return FLAGS[name]
    except KeyError:
        raise KeyError(
            f"{name} нет в реестре scripts/feature_flags.py. Новый флаг режима "
            f"добавляется СЮДА (и в CLAUDE.md, и в config.example.env — "
            f"tests/test_feature_flags.py сверяет все три)") from None


def raw(name):
    """Сырое значение из окружения (с учётом исторических имён) или None."""
    spec = _spec(name)
    for key in (spec.name,) + spec.aliases:
        val = os.environ.get(key)
        if val is not None and val.strip() != "":
            return val
    return None


def mode(name):
    """Нормализованное значение строкового режима. Мусорное значение ->
    безопасный откат (см. докстринг модуля) с однократным предупреждением.

    Безопасный откат — "off", ЕСЛИ он вообще легален для этого флага, иначе
    его собственный дефолт. Первая версия возвращала литеральное "off" всегда,
    и это было верно ровно до тех пор, пока все флаги были вкл/выкл. Как только
    в реестр приехали режимы БЕЗ "off" (GRAIN_BLEND_MODE=softlight/grainmerge/
    expr, DELIVERY_PROFILE=youtube/archive/hevc), опечатка вроде
    DELIVERY_PROFILE=h265 давала расхождение отчёта с реальностью: рендер
    честно падал на свой дефолт (оба потребителя защищаются сами —
    DELIVERY_PROFILES.get(...) / else-ветка softlight), а snapshot() писал в
    media_plan/feature_flags.json значение "off" и сводка печатала
    «Выключено: DELIVERY_PROFILE=off» — при том, что зерно накладывалось и
    эпизод кодировался. Файл, заведённый ради ответа «каким пайплайном собран
    этот ролик», врал именно там, где нужен. Найдено состязательным аудитом
    (03.09). Для четырёх исходных флагов "off" в allowed есть — их поведение
    не изменилось."""
    spec = _spec(name)
    if spec.is_boolean:
        raise TypeError(f"{name} — булев флаг, используй enabled({name!r})")
    val = raw(name)
    val = spec.default if val is None else val.strip().lower()
    if val not in spec.allowed:
        fallback = "off" if "off" in spec.allowed else spec.default
        if name not in _warned:
            _warned.add(name)
            print(f"  ВНИМАНИЕ: {name}={val!r} не входит в {spec.allowed} — "
                  f"откатываюсь на {fallback!r}.", file=sys.stderr)
        return fallback
    return val


def enabled(name):
    """True/False для булева флага. Выключает ТОЛЬКО значение "0" — см.
    докстринг модуля про сознательно не добавленные синонимы."""
    spec = _spec(name)
    if not spec.is_boolean:
        raise TypeError(f"{name} — режим со списком значений, используй mode({name!r})")
    val = raw(name)
    return (spec.default if val is None else val.strip()) != "0"


def value(name):
    """Текущее значение как строка — для сводки/снимка, без ветвления по типу."""
    spec = _spec(name)
    return mode(name) if not spec.is_boolean else ("1" if enabled(name) else "0")


def is_default(name):
    return raw(name) is None


def snapshot():
    """Снимок всех флагов — что РЕАЛЬНО исполнялось в этом прогоне.
    Пишется рядом с рендером (media_plan/feature_flags.json), чтобы «какой
    пайплайн собрал этот ролик» был проверяемым фактом, а не памятью."""
    return {name: {"value": value(name),
                   "default": FLAGS[name].default,
                   "from_env": not is_default(name)}
            for name in FLAGS}


def write_snapshot(video_dir):
    """Кладёт снимок флагов в <video_dir>/media_plan/feature_flags.json.
    Нужен затем же, зачем render_manifest.json: через неделю вопрос «а этот
    ролик собран с арбитром или без» должен иметь ответ в файле, а не в
    памяти. Любая ошибка записи — не повод ронять рендер (fail-open)."""
    import json
    try:
        plan_dir = os.path.join(video_dir, "media_plan")
        os.makedirs(plan_dir, exist_ok=True)
        path = os.path.join(plan_dir, "feature_flags.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(snapshot(), fh, ensure_ascii=False, indent=2)
        return path
    except Exception:
        return None


def format_summary():
    """Человекочитаемая сводка активных слоёв. Печатается в начале рендера:
    самый дешёвый способ увидеть, что дорогой слой выключен, ДО того как
    потрачены часы CPU (та же дисциплина, что ЧАСТЬ 1 CLAUDE.md)."""
    on, off = [], []
    for name in FLAGS:
        val = value(name)
        mark = "" if is_default(name) else " (из .env)"
        (off if val in ("off", "0") else on).append(f"{name}={val}{mark}")
    lines = [f"  Активные слои: {', '.join(on) if on else '— (все выключены)'}"]
    if off:
        lines.append(f"  Выключено: {', '.join(off)}")
    return "\n".join(lines)


def print_summary():
    print(format_summary())


if __name__ == "__main__":
    print_summary()
