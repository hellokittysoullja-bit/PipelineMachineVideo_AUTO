#!/usr/bin/env python3
"""Субтитры (SRT: до 2 строк по ~42 символа, длинный блок — несколькими
подряд) и главы для описания YouTube — из уже посчитанного тайминга.

Перенесено из pipeline_smart.py (PipelineMachineVideo_AUTO) без изменения
логики."""
import math
import os
import re


def section_title(name):
    """BLOCK N: Название -> 'Название'. HOOK/FINAL/безымянные BLOCK — без титра."""
    m = re.match(r'BLOCK\s+\d+\s*:\s*(.+)', name, re.I)
    return m.group(1).strip() if m else None


def _srt_timestamp(t):
    t = max(0.0, t)
    ms_total = int(round(t * 1000))
    h, rem = divmod(ms_total, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


SRT_MAX_LINE_CHARS = 42   # реальный, задокументированный стандарт индустрии субтитров


                           # (Netflix/BBC-класс гайдлайнов, ~40-42 символа/строка — читаемо
                           # без движения глаз поперёк всего экрана)
SRT_MAX_LINES = 2   # тот же стандарт — больше 2 строк одновременно закрывает кадр


def _wrap_caption_text(text, max_line_chars=SRT_MAX_LINE_CHARS, max_lines=SRT_MAX_LINES):
    """Перенос ОДНОГО SRT-cue на строки по РЕАЛЬНЫМ границам слов, не более
    max_lines строк — задокументированный стандарт субтитрирования (см.
    CLAUDE.md/коммит про GSA/CU Boulder captioning guidelines): короткие
    читаемые строки, разрыв по словам, никогда посередине слова.

    Не пытается влезть в max_lines любой ценой — блок, для которого даже
    после max_lines строк остаётся текст, честно дописывает остаток В
    ПОСЛЕДНЮЮ строку (длиннее стандарта, но ничего не обрезается и не
    теряется молча) — слишком длинный для 2 строк блок сценария сам по
    себе повод пересмотреть длину блока выше по пайплайну, не что чинить
    здесь угадыванием, где резать смысл."""
    words = text.split()
    if not words:
        return text
    lines, idx = [], 0
    while idx < len(words) and len(lines) < max_lines:
        cur = words[idx]
        idx += 1
        while idx < len(words):
            candidate = f"{cur} {words[idx]}"
            if len(candidate) > max_line_chars:
                break
            cur = candidate
            idx += 1
        lines.append(cur)
    if idx < len(words):
        lines[-1] = lines[-1] + " " + " ".join(words[idx:])
    return "\n".join(lines)


# Сколько текста реально помещается в один субтитр-кадр по тому же стандарту
# (SRT_MAX_LINES строк по SRT_MAX_LINE_CHARS символов). Всё, что длиннее,
# физически не может быть показано за раз — см. _split_caption_into_cues().
SRT_MAX_CUE_CHARS = SRT_MAX_LINE_CHARS * SRT_MAX_LINES


def _split_caption_into_cues(text, max_chars=SRT_MAX_CUE_CHARS,
                             max_line_chars=SRT_MAX_LINE_CHARS,
                             max_lines=SRT_MAX_LINES):
    """Разбить текст ОДНОГО блока на несколько последовательных субтитр-cue.

    РЕАЛЬНЫЙ, ИЗМЕРЕННЫЙ дефект готового файла (не гипотеза). До этой
    функции write_subtitles() писала строго один cue на блок, а
    _wrap_caption_text() честно признавала в своём докстринге, что остаток,
    не влезший в SRT_MAX_LINES строк, дописывается В ПОСЛЕДНЮЮ строку
    ("длиннее стандарта, но ничего не теряется"). Прямой прогон по
    videos/02_ne-mechom/script.txt: 94 cue из 259 (36%) получали строку
    длиннее стандарта 42 символа, САМАЯ ДЛИННАЯ — 139 символов. На
    YouTube это не "чуть длиннее": плеер переносит такую строку сам и
    закрывает текстом треть кадра поверх картинки, ради которой весь
    остальной пайплайн и работает.

    Причина дефекта была не в переносе, а в единице нарезки: у блока после
    split_long_blocks() может быть 35+ слов и 14 секунд окна — это не один
    субтитр, это три. Показывать их одновременно нельзя, а резать было
    нечем.

    Разбивка жадная по границам слов (никогда внутри слова) с ВЫРАВНИВАНИЕМ:
    сначала считаем, сколько cue нужно минимум (n = ceil(len/max_chars)), и
    наполняем до len/n, а не до max_chars. Без выравнивания жадность даёт
    хвост вида "и всё." отдельным кадром — одинокий обрывок на экране
    читается как сбой вёрстки, ровно тот же класс претензии, из-за которого
    fallback_card.py выбирает ЗАКОНЧЕННУЮ клаузу, а не обрезанную первую.

    Текст, который и так влезает в один субтитр-кадр, возвращается ОДНИМ
    элементом — для таких блоков вывод write_subtitles() остаётся
    байт-в-байт прежним. Проверка «влезает» — не по длине в символах,
    а прогоном через сам _wrap_caption_text() (цикл ниже): 11 блоков
    того же эпизода короче max_chars и всё равно не укладывались в две
    строки, потому что ломаются по словам неудачно — ранний выход по
    длине их и пропускал.
    """
    words = (text or "").split()
    if not words:
        return [text or ""]

    def _fits(chunk):
        wrapped = _wrap_caption_text(chunk).split("\n")
        return len(wrapped) <= max_lines and all(len(l) <= max_line_chars for l in wrapped)

    def _fill(limit_chars):
        """Жадно набирает cue, пока следующий шаг не ломает стандарт.
        limit_chars — мягкая цель по длине (для балансировки); жёсткое
        условие всегда одно и то же — _fits(), то есть РЕАЛЬНЫЙ перенос."""
        out, cur = [], ""
        for w in words:
            cand = f"{cur} {w}".strip()
            if cur and (len(cand) > limit_chars or not _fits(cand)):
                out.append(cur)
                cur = w
            else:
                cur = cand
        if cur:
            out.append(cur)
        return out

    # Шаг 1 — максимально плотная набивка: даёт МИНИМАЛЬНО возможное число
    # cue. Минимальное здесь важно не из экономии: каждый лишний cue режет
    # окно блока на более короткие куски, а cue короче ~0.7с читается как
    # мигание, а не как субтитр (реальный регресс первой версии этой
    # функции — она дробила на n частей "по формуле" и на 259 блоках
    # эпизода 02 дала 44 cue короче 0.7с, которых до неё не было ни одного).
    packed = _fill(max_chars)
    # Шаг 2 — балансировка на ТО ЖЕ число cue: жадная набивка оставляет
    # хвост вида "и всё." отдельным кадром, а одинокий обрывок на экране
    # читается как сбой вёрстки (тот же класс претензии, из-за которого
    # fallback_card.py берёт ЗАКОНЧЕННУЮ клаузу, а не обрезанную первую).
    # Берём балансировку ТОЛЬКО если она не увеличила число cue и каждый
    # кусок по-прежнему проходит _fits(); иначе остаётся плотный вариант.
    if len(packed) > 1:
        total = len(" ".join(words))
        # Перебор мягкой цели от самой ровной (total/n) до плотной
        # (max_chars): первая, что укладывается в то же число cue и
        # проходит _fits() — самая ровная из возможных. Одной попытки
        # (ровно total/n) не хватает: на неудачной границе слова она
        # ломается, и код откатывался на плотный вариант с хвостом в 4
        # символа — 14 таких сирот на эпизоде 02. Перебор конечен и
        # заведомо результативен: на limit == max_chars он вырождается
        # ровно в packed.
        for limit in range(math.ceil(total / len(packed)), max_chars + 1):
            balanced = _fill(limit)
            if len(balanced) <= len(packed) and all(_fits(c) for c in balanced):
                return balanced
    return packed


def write_subtitles(video_dir, blocks, starts, durs, real_weights=None):
    """SRT — бесплатный побочный продукт уже посчитанного тайминга: реальный
    посимвольный alignment.csv (см. load_alignment_weights) уже участвует в
    расчёте durs/starts, отдельно парсить CSV второй раз не нужно. starts —
    РЕАЛЬНОЕ аудио-время (block_durations по total, не по раздутой под xfade
    target — та же модель, что уже использует energy_pace_multipliers для
    сэмплинга кривой громкости, см. main()), иначе субтитры разъехались бы
    с озвучкой на сумму xfade-нахлёстов к концу ролика.

    Один блок = один субтитр-кадр: текст уже чистый (теги вырезаны в
    parse_blocks), деления и так на границах пауз/фраз — не нужно заново
    резать по 5-7 слов, разбивка уже смысловая.

    Видимое окно субтитра НЕ включает хвостовую паузу блока (2.7, тот же
    класс бага, что уже чинили в load_hook_word_timings): durs[i] —
    ПОЛНОЕ окно блока, block_durations() прибавляет b["pause_after"] к весу
    ДО floor/cap/глобального scale, так что субтитр держался бы на экране
    почти всю следующую паузу — визуально текст "отстаёт" от голоса на
    тишине. real_weights (если есть) даёт долю паузы в исходном сыром весе
    блока — она инвариантна к равномерному масштабированию, обрезаем ею
    durs[i] перед показом. Без real_weights (старый вызов/фоллбэк) — как
    раньше, полное окно."""
    path = os.path.join(video_dir, "subtitles.srt")
    lines = []
    n = 0
    for i, (b, s, d) in enumerate(zip(blocks, starts, durs)):
        text = b["text"].strip()
        if not text:
            continue
        pause_after = b.get("pause_after") or 0.0
        w = real_weights[i] if (real_weights and i < len(real_weights) and real_weights[i]) else None
        visible_d = d * (1.0 - pause_after / (w + pause_after)) if (pause_after > 0 and w) else d
        # Блок длиннее одного субтитр-кадра показывается НЕСКОЛЬКИМИ cue
        # подряд (см. _split_caption_into_cues): окно блока делится между
        # ними ПРОПОРЦИОНАЛЬНО числу символов. Это оценка внутри блока, но
        # строго более точная, чем то, что было: раньше весь текст блока
        # висел на экране целиком всё его окно, то есть последняя фраза
        # блока показывалась с самого начала — за секунды до того, как её
        # произнесут. Границы cue вычисляются нарастающим итогом от одного
        # и того же старта, чтобы они плотно покрыли окно без щелей и
        # нахлёстов (та же дисциплина "квантуем границы, а не длительности",
        # что и в phrase_locked_durations).
        cues = _split_caption_into_cues(text)
        chars_total = sum(len(c) for c in cues) or 1
        acc_chars = 0
        for c in cues:
            cue_start = s + visible_d * (acc_chars / chars_total)
            acc_chars += len(c)
            cue_end = s + visible_d * (acc_chars / chars_total)
            n += 1
            lines.append(str(n))
            lines.append(f"{_srt_timestamp(cue_start)} --> "
                         f"{_srt_timestamp(max(cue_end, cue_start + 0.3))}")
            lines.append(_wrap_caption_text(c))
            lines.append("")
    open(path, "w", encoding="utf-8").write("\n".join(lines))
    return path


def write_chapters(video_dir, blocks, starts):
    """Список глав для описания YouTube — из уже известных границ секций
    (section_title()) и их реального старта в аудио (starts, см.
    write_subtitles). YouTube требует первую главу строго с 00:00 и минимум
    3 главы от 10с каждая, иначе не активирует таймлайн — это проверяется
    руками при вставке в описание, здесь только честный расчёт таймингов."""
    path = os.path.join(video_dir, "chapters.txt")
    seen, lines = set(), []
    for b, s in zip(blocks, starts):
        if b["section"] in seen:
            continue
        seen.add(b["section"])
        label = section_title(b["section"])
        if label is None:
            label = "Хук" if b["section"].startswith("HOOK") else (
                "Итоги" if b["section"].startswith("FINAL") else b["section"])
        # Заголовки блоков в script.txt пишутся КАПСОМ для видимости внутри
        # сценария (не для показа зрителю) — в реальном описании YouTube это
        # читается как крик. Трогаем только то, что ЦЕЛИКОМ капс — намеренно
        # смешанный регистр не портим.
        if label.isupper():
            label = label[0] + label[1:].lower()
        t = max(0.0, s)
        h, rem = divmod(int(t), 3600)
        m, sec = divmod(rem, 60)
        ts = f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"
        lines.append(f"{ts} {label}")
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    return path
