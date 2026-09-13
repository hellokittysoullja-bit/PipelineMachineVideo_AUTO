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

# Приоритет при конфликте. Кульминация сюда не входит — она обрабатывается
# как ЗАРЕЗЕРВИРОВАННОЕ ОКНО (см. reserved_windows): её акцент ставит
# add_reveal_sfx() по уже проверенной в проде конвенции, а планировщик
# обязан вокруг неё расступиться, а не переставлять её заново.
CUE_PRIORITY = {"chapter": 2, "plate": 1}

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


def _in_any_window(t, windows):
    for a, b in windows or ():
        if a <= t <= b:
            return True
    return False


def plan_sfx_cues(blocks, sub_starts, real_weights, total_dur,
                  chapter_variants=(), plate_cues=(), reserved_windows=(),
                  min_gap=SFX_MIN_GAP_SEC, max_per_min=SFX_MAX_PER_MIN):
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

    # Сортировка по приоритету, потом по времени: при конфликте побеждает
    # более важный эффект независимо от того, кто раньше на таймлайне.
    candidates.sort(key=lambda c: (-CUE_PRIORITY.get(c["kind"], 0), c.get("time") or 0.0))

    taken = []
    for c in candidates:
        t = c.get("time")
        anchor = c.get("anchor", t)
        if t is None or t < 0 or (total_dur and t > total_dur):
            dropped.append(dict(c, reason="out_of_range"))
            continue
        if _in_any_window(anchor if anchor is not None else t, reserved_windows) \
                or _in_any_window(t, reserved_windows):
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
