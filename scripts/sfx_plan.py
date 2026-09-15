#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Планировщик расстановки звуковых эффектов — РЕШАЕТ, ЧТО И КОГДА звучит.

ЗАЧЕМ ОТДЕЛЬНЫЙ МОДУЛЬ (реальный, измеренный пробел 13.09). До него звук
в готовом ролике ставился двумя независимыми кусками кода, каждый из
которых знал только про себя:

  * `add_typewriter_clicks()` — щелчки, но ТОЛЬКО под плашкой варианта 4
    (`stat_variant % 5 == 4`). Замер на реальном эпизоде `02_ne-mechom`
    (27 минут, 11 плашек): звучали 2 из 11, остальные 9 цифр выскакивали
    на экран в полной тишине.
  * `add_reveal_sfx()` — акцент на `[climax]`. В том же эпизоде тегов
    ровно 2.

Итого ЧЕТЫРЕ звуковых события за 27 минут, и ни одно из них не знало о
существовании другого: столкнись они на одной секунде — наложились бы, и
никто бы этого не заметил. Девятнадцать границ глав не озвучены вообще.

ЧТО ДЕЛАЕТ ЭТОТ МОДУЛЬ. Собирает ВСЕ звуковые события эпизода в один
список, проверяет каждое против реального тайминга речи и разрешает
конфликты по приоритету — то есть ровно то, что делает звукорежиссёр,
раскладывая эффекты по таймлайну, а не «каждый эффект сам по себе».

ТРИ ПРАВИЛА, КОТОРЫЕ ОТЛИЧАЮТ ГРАМОТНУЮ РАССТАНОВКУ ОТ СЛУЧАЙНОЙ:

1. **Звук перехода живёт ТОЛЬКО в реальной тишине.** Граница главы —
   это пауза между двумя фразами, и её длина известна точно: конец речи
   предыдущего блока берётся из посимвольного alignment (`real_weights`),
   начало следующего — из онсетов (`sub_starts`). Эффект заканчивается
   РОВНО на первом слове новой главы и не начинается раньше, чем
   закончилось последнее слово предыдущей. Не поместился — не ставится
   вообще (и об этом пишется причина), а не «поставим, авось не помешает».
   Свист поверх слова — самая узнаваемая ошибка любителя, и единственный
   способ гарантированно её не совершить — не оставлять себе такой
   возможности.
   Нет alignment (старый эпизод без посимвольной разметки) — переходы не
   ставятся СОВСЕМ: без него не существует данных о том, где тишина, и
   любое размещение было бы угадыванием.

2. **Длительность эффекта подбирается под реальный размер паузы.**
   `fix_pauses.py` подрезает тишину по кривой в диапазоне
   KEEP_MIN_SEC=0.42 .. KEEP_MAX_SEC=1.35 секунды — то есть на одном и том
   же эпизоде паузы на границах глав РАЗНЫЕ. Один эффект фиксированной
   длины либо не влез бы в короткие, либо звучал бы огрызком в длинных.
   Планировщику передаётся НЕСКОЛЬКО вариантов одного эффекта, он берёт
   самый длинный из помещающихся.

3. **Окно кульминации зарезервировано.** Музыка уже проваливается вокруг
   `[climax]` (CLIMAX_DIP_* в pipeline_smart.py), и туда же приходит
   акцент разоблачения. Любой другой звук в этом окне разбавляет ровно тот
   момент, ради которого провал и сделан, — поэтому окно отдаётся
   кульминации целиком, а всё остальное из него выбрасывается.

Плюс два общих ограничителя: минимальный интервал между любыми двумя
эффектами и потолок плотности на минуту — чтобы слой не превратился в
игровой автомат на густо размеченном участке сценария.

Модуль — чистый python без зависимостей: ни ffmpeg, ни ML, ни сети.
Он ничего не сводит и не рендерит, только считает список «что, когда, как
громко и почему». Сведение — `add_planned_sfx()` в pipeline_smart.py,
аудит-трейл — `media_plan/sfx_plan.json`.
"""
import inspect


# Приоритет при конфликте. Кульминация сюда не входит — она обрабатывается
# как ЗАРЕЗЕРВИРОВАННОЕ ОКНО (см. reserved_windows): её акцент ставит
# add_reveal_sfx() по уже проверенной в проде конвенции, а планировщик
# обязан вокруг неё расступиться, а не переставлять её заново.
CUE_PRIORITY = {"chapter": 3, "object": 2, "plate": 1}

# --- ОБЪЕКТНЫЙ СЛОЙ (звук предмета, о котором говорит текст) -------------
# Опережение: звук приходит РАНЬШЕ слова. Постановка ровно на слове —
# буквальная иллюстрация речи (в монтаже это называют Mickey Mousing) и
# самый узнаваемый признак любителя. Профессионально сначала слышишь,
# потом понимаешь: сознание успевает принять звук как часть сцены, а не
# как подпись к реплике.
OBJECT_PRE_LAP_SEC = 0.45

# Два класса, а не один: молот — это удар (точка), костёр — это состояние
# (протяжённость). Один механизм «положить сэмпл в точку» на них не
# работает: удар с полуторасекундным нарастанием смазан, а костёр,
# оборванный через 0.7с, читается как щелчок.
OBJECT_CLASS_POINT = "point"
OBJECT_CLASS_BED = "bed"
OBJECT_BED_SEC = 6.0           # сколько звучит протяжённый объект
OBJECT_BED_FADE_IN_SEC = 1.2
OBJECT_BED_FADE_OUT_SEC = 2.0

# Уровни задаются РАЗРЫВОМ с голосом в LU, а не усилением в дБ, и
# выводятся замером — константы ниже только запасные.
#
# Причина измерена, а не предположена. Первая версия объявляла -20 дБ точке
# и -26 дБ подзвучнику поверх ассета, нормированного к пику -10 dBFS.
# Замер против голоса на -16 LUFS дал: точка 22.0 LU под голосом (цель
# 26-30, то есть на 4 дБ ГРОМЧЕ задуманного — удар молота выскакивал бы
# поверх реплики), подзвучник 52.1 LU (цель 30-34, то есть на 18 дБ тише,
# практически не слышен).
#
# Обе ошибки от одной причины: -10 dBFS — это ПИК ассета, а не его
# громкость. У короткого удара и у шестисекундного фона с одним и тем же
# пиком громкость разная, поэтому одинаковые «дБ» для них несопоставимы.
# Ровно на этом уже сгорела музыкальная подложка (MUSIC_BED_GAIN_DB был
# -13 дБ, задуманный разрыв 16 LU, в опубликованном ролике вышло 27).
#
# Разрыв считается по МАКСИМАЛЬНОЙ МГНОВЕННОЙ громкости, не по
# интегральной: ухо сравнивает транзиент с речью именно в момент удара, а
# интеграл по 0.6-секундному удару занижает его в разы. На стационарном
# материале обе меры совпадают (замер подзвучника: -68.5 против -67.8).
OBJECT_POINT_GAP_LU = 28.0     # середина коридора 26-30
OBJECT_BED_GAP_LU = 32.0       # середина коридора 30-34
OBJECT_GAIN_MIN_DB = -40.0
OBJECT_GAIN_MAX_DB = 0.0       # громче исходного ассета не поднимаем никогда
OBJECT_POINT_GAIN_DB = -24.0   # запасные: пересчитаны по замеру выше
OBJECT_BED_GAIN_DB = -8.0

# Плотность. Главный рычаг объектного слоя — воздержание: три звука за
# главу дороже пятнадцати. Ролик, где звучит каждое существительное, это
# озвученный словарь, а не кино.
# Окно смысловой связи: насколько раньше своего слова звук может ЗАКОНЧИТЬСЯ
# и всё ещё читаться как относящийся к нему. Меряется от КОНЦА кюя, а не от
# начала — иначе длинный ассет штрафовался бы за собственную длительность,
# хотя заканчивается он ровно там же, где короткий.
#
# Ограничение обязательно, и это не вкус: без него кюй для слова в середине
# десятисекундного блока уезжал бы в паузу ПЕРЕД блоком, то есть на 7+ секунд
# раньше своего слова, поверх чужой фразы — ровно измеренный 13.09 промах
# (20.0с вместо 27.2). Практический смысл для автора: тег [sfx:] работает,
# когда стоит В НАЧАЛЕ фразы, а не в её середине.
OBJECT_MAX_LEAD_SEC = 2.2
OBJECT_MIN_GAP_SEC = 6.0
OBJECT_MAX_PER_MIN = 3

# Минимальный интервал между двумя ЛЮБЫМИ принятыми эффектами. Два
# акцента ближе этого на слух сливаются в один сдвоенный удар — ошибка
# читается как сбой, а не как приём.
SFX_MIN_GAP_SEC = 2.5

# Потолок плотности в скользящем окне 60с. Не цель, а именно потолок:
# на реальном эпизоде (19 границ глав + 11 плашек за 27 минут) плотность
# выходит около 1.1/мин, то есть в норме ограничитель не срабатывает
# вообще — он существует ради участка, где сценарий вдруг размечен густо.
SFX_MAX_PER_MIN = 6
SFX_DENSITY_WINDOW_SEC = 60.0

# Зазор между концом эффекта перехода и первым словом новой главы. Ноль
# означал бы стык впритык — на слух это «съеденный» первый согласный.
CHAPTER_HEADROOM_SEC = 0.03


def _section_of(block):
    return str((block or {}).get("section", ""))


def chapter_boundaries(blocks):
    """Индексы блоков, с которых начинается НОВАЯ секция сценария.

    Нулевой блок не граница: ролик и так начинается с этого места, звук
    перехода там озвучивал бы переход из ниоткуда.
    """
    out = []
    prev = None
    for i, b in enumerate(blocks or []):
        sec = _section_of(b)
        if prev is not None and sec != prev:
            out.append(i)
        prev = sec
    return out


def speech_gap_before(index, sub_starts, real_weights):
    """(начало_тишины, конец_тишины) перед блоком index — или None.

    Конец речи предыдущего блока — его онсет плюс РЕАЛЬНАЯ длительность
    речи из alignment (`real_weights`, см. `_real_speech_span` в
    pipeline_smart.py: это время, которое блок реально звучит в
    audio_fixed, уже после подрезки пауз). Начало тишины именно там, а не
    «онсет предыдущего плюс его длительность КАДРА» — длительность кадра
    включает паузу и дала бы тишину нулевой длины на каждом блоке.

    None, если alignment для нужного блока отсутствует: тогда данных о
    положении тишины нет, и звать их «нулём» опаснее, чем честно отказаться.
    """
    if index <= 0 or index >= len(sub_starts):
        return None
    prev = index - 1
    if not real_weights or prev >= len(real_weights):
        return None
    w = real_weights[prev]
    if not w or w <= 0:
        return None
    start = float(sub_starts[prev]) + float(w)
    end = float(sub_starts[index])
    if end <= start:
        return None
    return start, end


def pick_variant(variants, available_sec):
    """Самый ДЛИННЫЙ вариант эффекта, помещающийся в available_sec.

    variants — [(путь, длительность_сек), ...] в любом порядке. Ни один не
    помещается -> None (вызывающий код обязан отказаться от эффекта, а не
    подставить самый короткий «хоть что-то»).
    """
    fits = [v for v in variants or [] if v[1] is not None and v[1] <= available_sec]
    if not fits:
        return None
    return max(fits, key=lambda v: v[1])


def _cue_span(cue):
    """(начало, конец) звучания кюя. Не точка — ОТРЕЗОК.

    Реальный дефект, найденный замером 13.09: защищённые окна проверялись
    по ОДНОМУ моменту старта, а кюй звучит сколько-то секунд. Шестисекундный
    подзвучник, стартовавший за 2.5с до блока `[hush]`, играл **3.55с внутри
    тишины**, которую сценарий потребовал явно, — и по всем отчётам проходил
    как принятый по правилам. То же самое и с окном кульминации: там музыка
    проседает ради одного момента, а фон спокойно тянулся сквозь него.

    Длительность берётся из того, что план уже посчитал: обрезка
    (`trim_sec`) у протяжённого, иначе длительность самого ассета.
    """
    t = cue.get("time")
    if t is None:
        return None
    dur = cue.get("trim_sec") or cue.get("asset_dur") or 0.0
    return float(t), float(t) + float(dur)


def _span_hits_window(cue, windows):
    span = _cue_span(cue)
    if span is None:
        return False
    a, b = span
    for w0, w1 in windows or ():
        if a < w1 and b > w0:
            return True
    return False


def _in_any_window(t, windows):
    for a, b in windows or ():
        if a <= t <= b:
            return True
    return False


def hush_windows(blocks, sub_starts, real_weights):
    """Окна, помеченные в сценарии как [hush] — «здесь тишина НУЖНА».

    Возвращается как зарезервированное окно, то есть тем же механизмом, что
    защищает кульминацию: разница между «сюда ничего не поместилось» и
    «сюда ничего нельзя» должна быть в данных, а не в удаче планировщика.
    """
    out = []
    for i, b in enumerate(blocks or []):
        if not (b or {}).get("hush"):
            continue
        if i >= len(sub_starts or ()):
            continue
        start = float(sub_starts[i])
        dur = None
        if real_weights and i < len(real_weights) and real_weights[i]:
            dur = float(real_weights[i])
        elif i + 1 < len(sub_starts or ()):
            dur = float(sub_starts[i + 1]) - start
        if dur and dur > 0:
            out.append((start, start + dur))
    return out


def word_anchor_time(block, word_pos, start, speech_dur):
    """Момент слова №word_pos внутри блока.

    ЧЕСТНЫЙ ПРЕДЕЛ: это линейная интерполяция внутри блока, а не позиция
    конкретного слова из alignment. Обе величины берутся из ОДНОЙ шкалы
    (реальный онсет блока + реальная длительность его речи), поэтому
    смешения шкал — того класса бага, что уже ловили у protected_windows и
    у веса блока, — здесь нет. Но слово в середине длинной фразы может
    приехать на пару десятых от истины, и опережение (pre-lap) именно
    поэтому берётся с запасом: ошибка в сторону «раньше» безобидна, в
    сторону «позже» даёт звук поверх уже сказанного слова.
    """
    words = max(1, int((block or {}).get("words") or 1))
    pos = max(0, min(int(word_pos or 0), words))
    if not speech_dur or speech_dur <= 0:
        return float(start)
    return float(start) + (pos / float(words)) * float(speech_dur)


def _call_asset_for(fn, name, at, max_sec=None):
    """Вызвать резолвер ассета, поддерживая обе арности.

    Арность проверяется явно, а не ловится через TypeError: перехват
    TypeError поймал бы и настоящую ошибку ВНУТРИ резолвера и молча
    повторил бы вызов без момента — то есть замаскировал бы дефект под
    «старый контракт».
    """
    try:
        params = inspect.signature(fn).parameters
        n = sum(1 for prm in params.values()
                if prm.kind in (prm.POSITIONAL_ONLY, prm.POSITIONAL_OR_KEYWORD))
        if any(prm.kind == prm.VAR_POSITIONAL for prm in params.values()):
            n = max(n, 2)
    except (TypeError, ValueError):
        n = 1
    # max_sec передаётся ТОЛЬКО резолверу, который его объявил — тем же
    # способом (разбор сигнатуры), что и момент выше, и по той же причине:
    # ловить TypeError значило бы замаскировать настоящую ошибку внутри
    # резолвера под «старый контракт».
    if max_sec is not None:
        try:
            if "max_sec" in inspect.signature(fn).parameters:
                return fn(name, at, max_sec=max_sec) if n >= 2 else fn(name, max_sec=max_sec)
        except (TypeError, ValueError):
            pass
        return None
    return fn(name, at) if n >= 2 else fn(name)


def _silence_slot_for(index, sub_starts, real_weights, asset_dur, anchor):
    """Старт точечного кюя в РЕАЛЬНОЙ тишине перед блоком — или None.

    Тишина берётся тем же `speech_gap_before()`, что уже обслуживает
    переходы глав: конец речи предыдущего блока из alignment, начало
    следующего из онсетов. Второй формулы того же промежутка не заводится.

    Звук заканчивается за CHAPTER_HEADROOM_SEC до первого слова блока —
    та же конвенция, что у перехода: сначала слышишь, потом понимаешь.
    Не поместился в паузу или уехал бы дальше OBJECT_MAX_LEAD_SEC от своего
    слова — кюй НЕ ставится вообще, с записанной причиной. Это тот же
    выбор, что уже сделан для переходов («не поместился — не ставится»), и
    он же защищает от промаха на всю длину блока.
    """
    gap = speech_gap_before(index, sub_starts, real_weights)
    if not gap:
        return None
    g0, g1 = gap
    end = g1 - CHAPTER_HEADROOM_SEC
    start = end - float(asset_dur or 0.0)
    if start < g0:
        return None
    if anchor - end > OBJECT_MAX_LEAD_SEC:
        return None
    return start


def object_cues(blocks, sub_starts, real_weights, asset_for=None,
                pre_lap=OBJECT_PRE_LAP_SEC):
    """Кандидаты объектного слоя из разметки [sfx:...] сценария.

    asset_for(name[, момент]) -> (путь, длительность, класс[, усиление_дБ,
    источник[, референс_LUFS, чем_обоснован]]) либо None, если под этот
    концепт в библиотеке ничего нет. Усиление приходит ИЗМЕРЕННЫМ (см.
    pipeline_smart.object_gain_db) — планировщик его не выдумывает; без него
    берётся запасная константа класса. Нет ассета — кандидат честно уходит в
    отклонённые с причиной, а не подменяется похожим: «похожий» звук под
    конкретным словом слышен как ошибка, а тишина — нет.

    Момент передаётся ВТОРЫМ аргументом, потому что уровень кюя считается
    от громкости речи ВОКРУГ него, а не от средней по эпизоду. Резолвер с
    одним параметром продолжает работать — арность проверяется, а не
    ловится через TypeError: тот поймал бы и настоящую ошибку внутри
    резолвера, выдав её за «старый контракт».
    """
    cands, dropped = [], []
    for i, b in enumerate(blocks or []):
        marks = (b or {}).get("sfx") or []
        if not marks:
            continue
        if i >= len(sub_starts or ()):
            continue
        start = float(sub_starts[i])
        speech = float(real_weights[i]) if (real_weights and i < len(real_weights)
                                            and real_weights[i]) else 0.0
        if not speech:
            # Без РЕАЛЬНОЙ длительности речи блока позиция слова внутри него
            # неизвестна, и звук встал бы в начало блока — замер показал
            # промах на 7.2с при десятисекундном блоке. Переход главы в
            # точно такой же ситуации честно не ставится (`no_alignment`);
            # объект обязан вести себя так же, а не угадывать. Это та самая
            # асимметрия, из-за которой одна и та же нехватка данных
            # обрабатывалась двумя разными способами.
            for m in marks:
                nm = str((m or {}).get("name") or "").strip()
                if nm:
                    dropped.append({"kind": "object", "block": i, "name": nm,
                                    "section": _section_of(b),
                                    "reason": "no_alignment"})
            continue
        for m in marks:
            name = str((m or {}).get("name") or "").strip()
            if not name:
                continue
            base = {"kind": "object", "block": i, "name": name,
                    "section": _section_of(b)}
            # Якорь считается ДО резолвера: уровень зависит от того, какая
            # речь звучит вокруг этого момента, значит момент должен быть
            # известен раньше уровня.
            anchor = word_anchor_time(b, m.get("word_pos"), start, speech)
            got = _call_asset_for(asset_for, name, anchor) if asset_for else None
            if not got:
                dropped.append(dict(base, reason="no_asset"))
                continue
            path, asset_dur, cls = got[0], got[1], got[2]
            # ДЛИНА АССЕТА ПОД ДОСТУПНУЮ ТИШИНУ — там, где точечному кюю
            # иначе негде прозвучать. Замер 14.09: концепт armour_clank
            # имеет записи 0.60/1.93/2.00/2.27с, ротация по имени выдавала
            # 2.27с, и кюй отбрасывался с no_silence_for_object, хотя
            # подходящая запись лежала в той же папке. Тот же приём, что
            # уже работает у переходов глав (pick_variant).
            #
            # Спрашиваем ПОВТОРНО и только когда природный выбор не влез:
            # правка строго добавляющая — она может лишь найти
            # помещающийся вариант, но никогда не отнять прежний.
            if cls == OBJECT_CLASS_POINT and asset_for:
                gap = speech_gap_before(i, sub_starts, real_weights)
                room = (gap[1] - CHAPTER_HEADROOM_SEC - gap[0]) if gap else None
                if room and room > 0 and asset_dur > room:
                    fitting = _call_asset_for(asset_for, name, anchor, max_sec=room)
                    if fitting:
                        got = fitting
                        path, asset_dur, cls = got[0], got[1], got[2]
            gain_db = got[3] if len(got) > 3 else None
            gain_src = got[4] if len(got) > 4 else "constant"
            ref_lufs = got[5] if len(got) > 5 else None
            ref_kind = got[6] if len(got) > 6 else None
            # Точечный звук живёт ТОЛЬКО в реальной тишине — то же правило,
            # по которому уже живут переходы глав, и то же, что записано в
            # докстринге local_voice_lufs(). Раньше оно было только словами:
            # кюй ставился на `anchor - pre_lap` и попадал поверх речи.
            #
            # Почему это не придирка, а условие осмысленности всего слоя —
            # замер на собранной ленте: три версии с разрывом 24/28/32 LU
            # (то есть усиления, отличающиеся на 8 дБ) различались НЕ БОЛЕЕ
            # чем на 0.133 дБ в любом 100мс окне. Речь маскирует кюй на
            # -28 LU полностью: менять его уровень под речью бессмысленно,
            # слышно его не становится ни при каком числе.
            # ТОЛЬКО для точечного. Протяжённый подзвучник по замыслу и
            # звучит ПОД речью — это текстура под фразой, ближайший родич
            # атмосферного слоя (тот тоже идёт под голосом и намеренно не
            # прижимается сайдчейном). Требовать, чтобы шестисекундный фон
            # уместился в паузу, значило бы не ставить его никогда.
            # Отсюда и разные коридоры: точка 28 LU — это слышимость В
            # ПАУЗЕ, фон 32 LU — маскировка ПОД речью.
            if cls == OBJECT_CLASS_POINT:
                slot = _silence_slot_for(i, sub_starts, real_weights,
                                         float(asset_dur or 0.0), anchor)
                if slot is None:
                    # Причина без числа не говорит автору НИЧЕГО. Реальный
                    # случай 14.09: тег стоял после [short pause], которому
                    # движок дал 0.169с (замер по alignment), — туда не
                    # влезает ни один ассет, и починка это правка сценария
                    # на полный [pause], а не настройка порогов. Раньше
                    # отчёт писал только «нет тишины», и понять, что дело в
                    # выборе тега, было неоткуда.
                    gap = speech_gap_before(i, sub_starts, real_weights)
                    have = round(gap[1] - gap[0], 3) if gap else None
                    dropped.append(dict(
                        base, reason="no_silence_for_object",
                        anchor=round(anchor, 3),
                        gap_sec=have,
                        needed_sec=round(float(asset_dur or 0.0) + CHAPTER_HEADROOM_SEC, 3),
                        hint=("перед тегом [sfx:] нужна полная пауза [pause]: "
                              "[short pause] движок сводит почти в ноль")
                              if have is not None else "нет сигнала о тишине"))
                    continue
                t = slot
            else:
                t = anchor - float(pre_lap)
                if t < 0:
                    t = 0.0
            if gain_db is None:
                gain_db = (OBJECT_BED_GAIN_DB if cls == OBJECT_CLASS_BED
                           else OBJECT_POINT_GAIN_DB)
            cue = dict(base, time=t, anchor=anchor, asset=path,
                       asset_dur=float(asset_dur or 0.0), cls=cls,
                       gain_db=float(gain_db), gain_source=gain_src)
            if ref_kind:
                # Вторая сторона разрыва едет ВМЕСТЕ с кюем: без неё по
                # отчёту нельзя отличить «уровень выведен из речи рядом» от
                # «из средней по эпизоду», а это разные числа.
                cue["voice_ref"] = ref_kind
                if ref_lufs is not None:
                    cue["voice_ref_lufs"] = round(float(ref_lufs), 2)
            if cls == OBJECT_CLASS_BED:
                # Длительность и фейды НЕСЁТ САМ КЮЙ, а не логика сведения:
                # микшер обязан остаться тупым исполнителем плана, иначе
                # единственным способом проверить правило станет «отрендери
                # ролик и послушай». Обрезка не длиннее самого ассета —
                # иначе в хвосте окажется тишина с фейдом из ниоткуда.
                keep = min(OBJECT_BED_SEC, float(asset_dur or OBJECT_BED_SEC))
                cue["trim_sec"] = keep
                cue["fade_in_sec"] = min(OBJECT_BED_FADE_IN_SEC, keep / 3.0)
                cue["fade_out_sec"] = min(OBJECT_BED_FADE_OUT_SEC, keep / 2.0)
            cands.append(cue)
    return cands, dropped


def plan_sfx_cues(blocks, sub_starts, real_weights, total_dur,
                  chapter_variants=(), plate_cues=(), reserved_windows=(),
                  min_gap=SFX_MIN_GAP_SEC, max_per_min=SFX_MAX_PER_MIN,
                  object_asset_for=None):
    """Итоговый список эффектов эпизода: (принятые, отклонённые).

    blocks/sub_starts/real_weights/total_dur — ровно те же объекты, что уже
    посчитал main() для субтитров и монтажных резов. Планировщик НИЧЕГО не
    вычисляет заново: любая своя оценка времени рано или поздно разошлась бы
    с той, по которой реально смонтирован ролик.

    chapter_variants — [(путь, длительность), ...] варианты звука перехода.
    plate_cues — [{"time", "block", "section", "asset", "gain_db"}, ...]
        моменты появления плашек на экране; время считает вызывающий код ТОЙ
        ЖЕ формулой, что задаёт появление текста в кадре, иначе звук и
        картинка разъедутся.
    reserved_windows — [(начало, конец), ...] окна кульминации.

    Каждый отклонённый cue сохраняет причину — «почему на этой границе
    главы нет звука» обязано быть проверяемым фактом, а не догадкой при
    следующем разборе готового ролика.
    """
    accepted, dropped = [], []
    candidates = []

    for i in chapter_boundaries(blocks):
        section = _section_of(blocks[i])
        base = {"kind": "chapter", "block": i, "section": section,
                "anchor": float(sub_starts[i]) if i < len(sub_starts) else None}
        gap = speech_gap_before(i, sub_starts, real_weights)
        if gap is None:
            dropped.append(dict(base, reason="no_alignment"))
            continue
        gap_start, gap_end = gap
        room = (gap_end - gap_start) - CHAPTER_HEADROOM_SEC
        variant = pick_variant(chapter_variants, room)
        if variant is None:
            dropped.append(dict(base, reason="gap_too_short",
                                gap_sec=round(gap_end - gap_start, 3)))
            continue
        path, dur = variant
        # Эффект ЗАКАНЧИВАЕТСЯ на первом слове новой главы (минус зазор) —
        # та же конвенция, что уже у нарастания перед кульминацией.
        start = gap_end - CHAPTER_HEADROOM_SEC - dur
        candidates.append(dict(base, time=start, asset=path, asset_dur=dur,
                               gap_sec=round(gap_end - gap_start, 3)))

    for cue in plate_cues or ():
        candidates.append(dict(cue, kind="plate"))

    obj_cands, obj_dropped = object_cues(blocks, sub_starts, real_weights,
                                         asset_for=object_asset_for)
    dropped.extend(obj_dropped)
    candidates.extend(obj_cands)

    # Помеченная тишина защищается ТЕМ ЖЕ механизмом, что кульминация.
    reserved_windows = list(reserved_windows or ()) + \
        hush_windows(blocks, sub_starts, real_weights)

    # Сортировка по приоритету, потом по времени: при конфликте побеждает
    # более важный эффект независимо от того, кто раньше на таймлайне.
    candidates.sort(key=lambda c: (-CUE_PRIORITY.get(c["kind"], 0), c.get("time") or 0.0))

    taken = []
    taken_obj = []          # у объектного слоя свои, более строгие лимиты
    for c in candidates:
        t = c.get("time")
        anchor = c.get("anchor", t)
        if t is None or t < 0 or (total_dur and t > total_dur):
            dropped.append(dict(c, reason="out_of_range"))
            continue
        # Проверяется ВЕСЬ отрезок звучания, а не только его старт, плюс
        # якорь (у объекта звук начинается раньше слова, и попасть в
        # защищённое окно может любой из двух концов).
        if _in_any_window(anchor if anchor is not None else t, reserved_windows) \
                or _span_hits_window(c, reserved_windows):
            dropped.append(dict(c, reason="climax_window"))
            continue
        if any(abs(t - o) < min_gap for o in taken):
            dropped.append(dict(c, reason="too_close"))
            continue
        if max_per_min:
            near = sum(1 for o in taken if abs(t - o) <= SFX_DENSITY_WINDOW_SEC / 2.0)
            if near >= max_per_min:
                dropped.append(dict(c, reason="density_cap"))
                continue
        if c["kind"] == "object":
            # Свой, более жёсткий бюджет: главный рычаг этого слоя —
            # воздержание. Общий потолок здесь не помогает, он рассчитан на
            # служебную фурнитуру, которой за минуту бывает и шесть.
            if any(abs(t - o) < OBJECT_MIN_GAP_SEC for o in taken_obj):
                dropped.append(dict(c, reason="object_too_close"))
                continue
            near_obj = sum(1 for o in taken_obj
                           if abs(t - o) <= SFX_DENSITY_WINDOW_SEC / 2.0)
            if near_obj >= OBJECT_MAX_PER_MIN:
                dropped.append(dict(c, reason="object_density_cap"))
                continue
            taken_obj.append(t)
        taken.append(t)
        accepted.append(c)

    accepted.sort(key=lambda c: c["time"])
    dropped.sort(key=lambda c: (c.get("time") is None, c.get("time") or 0.0))
    return accepted, dropped


def summarize(accepted, dropped):
    """Короткая сводка по видам — для консоли и для sfx_plan.json."""
    out = {"accepted_total": len(accepted), "dropped_total": len(dropped),
           "accepted_by_kind": {}, "dropped_by_reason": {}}
    for c in accepted:
        k = c.get("kind", "?")
        out["accepted_by_kind"][k] = out["accepted_by_kind"].get(k, 0) + 1
    for c in dropped:
        r = c.get("reason", "?")
        out["dropped_by_reason"][r] = out["dropped_by_reason"].get(r, 0) + 1
    return out
