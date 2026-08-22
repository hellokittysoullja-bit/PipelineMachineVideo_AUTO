#!/usr/bin/env python3
"""Section Offset Detector — section_offsets.json для эпизодов БЕЗ Stage B
(ручной Шаг 6, вариант А — подавляющее большинство существующих эпизодов).

ROOT CAUSE (эмпирически подтверждено на videos/_test_wide — реальный
рассинхрон субтитров и монтажных резов с голосом на BLOCK1/BLOCK2, картинки
совпадают с субтитрами, но оба разъехались с голосом): raw_to_real_time()
в pipeline_smart.py верно вычитает обрезанные fix_pauses.py паузы ИЗ
ГЛОБАЛЬНОГО времени, но получает на вход ЛОКАЛЬНОЕ время (свой ноль на
файл media_plan/alignment/NN.csv) для каждой секции — кроме первой (HOOK),
где локальный ноль случайно совпадает с глобальным. Для BLOCK1+ это давало
неверный вес блока, который одинаково портил и тайминг субтитров
(write_subtitles), и тайминг монтажных резов (block_durations) — отсюда
"субтитры совпадают с картинками, но оба разъехались с голосом". Раньше
точную карту смещений писал только Stage B (scripts/speech_generate.py,
сам конкатенирует фрагменты — знает точно). Для ручной озвучки (вариант А)
карты не было вообще.

МЕТОД (полностью детерминированный, без калибровки на живом выводе):
1. Из локального alignment/NN.csv секции берём РЕАЛЬНЫЕ паузы TTS — и
   тег-паузы ([pause]/[short pause], та же техника, что уже использует
   load_alignment_weights() для точного сегментирования — ALIGNMENT_TAG_RE),
   и натуральные (без тега) разрывы между соседними НЕ-тег символами
   (TTS делает вдохи и микропаузы, которых нет в тексте как тега — см.
   docstring fix_pauses.py: на реальном эпизоде их даже больше, чем
   тег-пауз). Это уже точное, ЛОКАЛЬНОЕ время (посимвольный alignment —
   не приближение).
2. Отдельно, один раз на весь audio.mp3 — низкопороговый silencedetect
   (короче и чувствительнее, чем THRESH_SEC=1.0 у fix_pauses.py, который
   специально режет только ДОЛГИЕ паузы для аудио-монтажа; для сверки
   паттернов нужны и короткие [short pause]-паузы тоже).
3. Голосование (Hough-подобная кросс-корреляция): для каждой пары
   (локальный разрыв, реальная глобальная тишина) в разумном окне поиска
   (структурная нижняя граница — конец предыдущей уже определённой секции)
   считаем implied_offset = глобальная_тишина − локальный_разрыв,
   кластеризуем implied_offset'ы, берём центр самого большого кластера —
   это и есть оценка глобального смещения секции. confidence = доля
   СОБСТВЕННЫХ локальных разрывов секции, подтвердивших этот кластер.
4. Sanity-проверка №2 (независимая от паттерна пауз): ожидаемая
   длительность секции по числу слов / эмпирической скорости чтения ЭТОГО
   эпизода (та же формула wps, что уже использует block_durations() в
   pipeline_smart.py — измеренная НА ЭТОМ аудио, не универсальная
   константа 125 слов/мин из CLAUDE.md) должна быть в разумных пределах
   от детектированного интервала между смещениями. Ловит "патологически
   регулярный" паттерн пауз, который мог бы обмануть чистую
   кросс-корреляцию.
5. Sanity-проверка №3, ОПЦИОНАЛЬНАЯ (только если есть chapters.txt, и
   число глав совпадает с числом секций — по ПОРЯДКУ, не по фаззи-сверке
   заголовков, см. write_chapters(): главы пишутся строго в том же
   порядке, что и секции script.txt): грубая (не точная — chapters.txt в
   ВРЕМЕНИ ПОСЛЕ обрезки пауз, наши смещения — ДО) проверка направления и
   порядка величины — реальный интервал между главами не может быть
   БОЛЬШЕ сырого (обрезка только убирает время, никогда не добавляет).

ЧЕСТНЫЙ ПОТОЛОК (проговорено с пользователем явно, не скрывается):
это не 100%-гарантия. Патологически регулярный паттерн пауз теоретически
может обмануть кросс-корреляцию даже с двумя доп. проверками. Поэтому —
жёсткий confidence-гейт с безопасным откатом: секция, не набравшая
уверенности, просто НЕ попадает в section_offsets.json (тот же эффект,
что "файла вообще нет" — весь даунстрим-код уже умеет в это безопасно
откатываться, см. load_section_offsets() в fix_pauses.py и
pipeline_smart.py — отсутствие ключа для секции трактуется как offset=0.0,
т.е. ровно текущее, уже привычное поведение для необслуженных секций, не
хуже статус-кво). Профессиональный стандарт — честная уверенность плюс
обязательный человеческий контроль (ЧАСТЬ 13, Шаг 7.5), а не обещание
нуля ошибок.

Usage: python scripts/section_sync.py <video_dir>
Вход:  script.txt, audio.mp3, media_plan/alignment/NN.csv (по одному на
       секцию, см. load_alignment_weights в pipeline_smart.py).
Выход: media_plan/section_offsets.json (тот же потребитель и тот же
       формат, что уже пишет Stage B — fix_pauses.py и pipeline_smart.py
       читают его одинаково, не зная, кто именно его написал) +
       media_plan/section_sync_report.json (метод/уверенность/причина
       отказа по каждой секции — честный аудит-трейл, тот же принцип,
       что pause_decisions.json у Pause Intelligence).

Не трогает существующий НЕПУСТОЙ section_offsets.json — если он уже есть
(Stage B писал сам, знает точно по построению), детектору тут делать
нечего и он может только всё испортить перезаписью на оценку."""
import csv
import json
import os
import re
import subprocess
import sys


def _load_pipeline_smart(video_dir):
    """pipeline_smart.py читает VIDEO_FOLDER из sys.argv[1] на импорте —
    тот же обходной приём, что уже применяет speech_generate.py."""
    saved_argv = sys.argv
    sys.argv = ["pipeline_smart.py", video_dir]
    try:
        import pipeline_smart
    finally:
        sys.argv = saved_argv
    return pipeline_smart


GAP_MIN_SEC = 0.25        # локальный разрыв в alignment.csv, который считаем паузой-кандидатом
FINE_SILENCE_D = 0.25     # длительность тишины для низкопорогового silencedetect на мастер-файле
FINE_NOISE_DB = "-30dB"
VOTE_CLUSTER_TOL = 0.5    # секунд — окно кластеризации implied_offset при голосовании
MIN_CONFIDENCE = 0.6      # доля локальных разрывов секции, подтверждённых консенсус-смещением
SANITY_RATIO_MIN = 0.5
SANITY_RATIO_MAX = 2.0
SEARCH_SLACK_SEC = 6.0    # окно поиска вокруг структурного (последовательного) предположения


def detect_fine_silences(audio_path):
    r = subprocess.run(["ffmpeg", "-i", audio_path, "-af",
                        f"silencedetect=noise={FINE_NOISE_DB}:d={FINE_SILENCE_D}",
                        "-f", "null", "-"], capture_output=True, text=True)
    log = r.stderr
    starts = [float(x) for x in re.findall(r'silence_start:\s*([\d.]+)', log)]
    ends = [float(x) for x in re.findall(r'silence_end:\s*([\d.]+)', log)]
    return list(zip(starts, ends))


def local_pause_events(chars, ps):
    """[(local_start, local_end), ...] — тег-паузы (ALIGNMENT_TAG_RE, та же
    техника, что load_alignment_weights уже использует для точного
    сегментирования) ОБЪЕДИНЁННЫЕ с натуральными (без тега) разрывами
    >= GAP_MIN_SEC между соседними РЕАЛЬНЫМИ (не тег-символами) буквами —
    тег-паузы дают точные, надёжные опорные точки (TTS реально держит их
    заявленную длину), натуральные разрывы дают ДОПОЛНИТЕЛЬНЫЕ точки там,
    где TTS сделал незапланированный вдох (их обычно даже больше, чем
    тег-пауз, см. docstring fix_pauses.py)."""
    text = "".join(c for c, s, e in chars)
    events = []
    tag_positions = set()
    for m in ps.ALIGNMENT_TAG_RE.finditer(text):
        seg = chars[m.start():m.end()]
        if seg:
            events.append((seg[0][1], seg[-1][2]))
        tag_positions.update(range(m.start(), m.end()))
    for tag in ps.ALIGNMENT_STRIP_TAGS:
        idx = text.find(tag)
        while idx != -1:
            tag_positions.update(range(idx, idx + len(tag)))
            idx = text.find(tag, idx + 1)
    real_chars = [(s, e) for j, (c, s, e) in enumerate(chars) if j not in tag_positions]
    for (s1, e1), (s2, e2) in zip(real_chars, real_chars[1:]):
        if s2 - e1 >= GAP_MIN_SEC:
            events.append((e1, s2))
    events.sort()
    return events


def _cluster_center(values, tol):
    """Центр самого большого кластера значений в пределах tol друг от друга
    (сортировка + скользящее окно). (None, 0), если values пуст."""
    if not values:
        return None, 0
    vs = sorted(values)
    best_lo, best_len, lo = 0, 1, 0
    for hi in range(len(vs)):
        while vs[hi] - vs[lo] > tol:
            lo += 1
        if hi - lo + 1 > best_len:
            best_len, best_lo = hi - lo + 1, lo
    cluster = vs[best_lo:best_lo + best_len]
    return sum(cluster) / len(cluster), best_len


def estimate_section_offset(local_events, global_silences, search_lo, search_hi):
    """Голосование: implied_offset = global_mid - local_mid для каждой пары
    (локальное событие, глобальная тишина), чей implied_offset попадает в
    [search_lo, search_hi]. confidence — доля СОБСТВЕННЫХ локальных
    событий секции, подтвердивших итоговый консенсус-кластер (не число
    implied-точек — одно локальное событие может дать несколько соседних
    implied-точек, если мастер-тишина рядом разбита на несколько
    интервалов)."""
    if not local_events:
        return None, 0.0
    implied = []
    for ls, le in local_events:
        lm = (ls + le) / 2
        for gs, ge in global_silences:
            gm = (gs + ge) / 2
            off = gm - lm
            if search_lo <= off <= search_hi:
                implied.append(off)
    center, _ = _cluster_center(implied, VOTE_CLUSTER_TOL)
    if center is None:
        return None, 0.0
    confirmed = 0
    for ls, le in local_events:
        lm = (ls + le) / 2
        hit = any(search_lo <= (gm - lm) <= search_hi and abs((gm - lm) - center) <= VOTE_CLUSTER_TOL
                  for gs, ge in global_silences for gm in ((gs + ge) / 2,))
        if hit:
            confirmed += 1
    return center, confirmed / len(local_events)


def _load_chapters_seconds(video_dir):
    path = os.path.join(video_dir, "chapters.txt")
    if not os.path.exists(path):
        return None
    out = []
    for line in open(path, encoding="utf-8"):
        m = re.match(r'\s*(\d+):(\d{2})(?::(\d{2}))?\s+\S', line)
        if not m:
            continue
        g = [x for x in m.groups()[:3] if x is not None]
        parts = [int(x) for x in g]
        sec = (parts[0] * 60 + parts[1]) if len(parts) == 2 else (parts[0] * 3600 + parts[1] * 60 + parts[2])
        out.append(float(sec))
    return out or None


def detect_section_offsets(video_dir):
    """Возвращает (offsets, report). offsets — {section: raw_global_offset_sec}
    ГОТОВ к прямой записи в section_offsets.json (тот же формат, что и у
    Stage B). report — подробности решения по каждой секции (для
    section_sync_report.json и консоли)."""
    ps = _load_pipeline_smart(video_dir)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import fix_pauses

    script_path = os.path.join(video_dir, "script.txt")
    if not os.path.exists(script_path):
        return {}, {"error": "script.txt не найден"}
    blocks = ps.parse_blocks(script_path)
    section_order = []
    for b in blocks:
        if not section_order or section_order[-1] != b["section"]:
            section_order.append(b["section"])
    if not section_order:
        return {}, {"error": "в script.txt не найдено ни одной секции HOOK/BLOCK/FINAL"}

    existing_path = os.path.join(video_dir, "media_plan", "section_offsets.json")
    if os.path.exists(existing_path):
        try:
            existing = json.load(open(existing_path, encoding="utf-8"))
            if existing:
                return existing, {"_skipped": "section_offsets.json уже существует и непуст "
                                               "(вероятно, Stage B) — не перезаписываю точное знание оценкой"}
        except Exception:
            pass

    audio_path = fix_pauses.find_audio(video_dir)
    if not audio_path or not os.path.exists(audio_path):
        return {}, {"error": "audio.mp3 не найден"}

    alignment_dir = os.path.join(video_dir, "media_plan", "alignment")
    section_chars = {}
    for i, name in enumerate(section_order):
        p = os.path.join(alignment_dir, f"{i:02d}.csv")
        if not os.path.exists(p):
            continue
        rows = list(csv.DictReader(open(p, encoding="utf-8")))
        section_chars[name] = [(r["char"], float(r["start"]), float(r["end"])) for r in rows]

    if len(section_chars) < 2:
        return {}, {"_skipped": "меньше двух секций с alignment.csv — детектор не нужен "
                                 "(первая секция офсет 0.0 по определению)"}

    global_silences = detect_fine_silences(audio_path)

    tw = sum(b["words"] for b in blocks)
    tp = sum(b["pause_after"] for b in blocks)
    total_audio_dur = ps.get_media_duration(audio_path)
    wps = tw / max(total_audio_dur - tp, 1.0)
    section_words, section_pause = {}, {}
    for b in blocks:
        section_words[b["section"]] = section_words.get(b["section"], 0) + b["words"]
        section_pause[b["section"]] = section_pause.get(b["section"], 0.0) + b["pause_after"]

    chapters = _load_chapters_seconds(video_dir)
    chapters_usable = bool(chapters) and len(chapters) == len(section_order)

    offsets = {section_order[0]: 0.0}
    report = {section_order[0]: {"confidence": 1.0, "method": "first_section_is_zero_by_definition"}}

    first_chars = section_chars.get(section_order[0])
    cursor = first_chars[-1][2] if first_chars else 0.0

    for i in range(1, len(section_order)):
        name = section_order[i]
        chars = section_chars.get(name)
        expected_dur = section_words.get(name, 0) / max(wps, 0.01) + section_pause.get(name, 0.0)
        if not chars:
            report[name] = {"confidence": 0.0, "method": "rejected", "reason": "no_alignment_file"}
            cursor += max(expected_dur, SEARCH_SLACK_SEC)
            continue

        events = local_pause_events(chars, ps)
        lo, hi = cursor - SEARCH_SLACK_SEC, cursor + SEARCH_SLACK_SEC
        offset, conf = estimate_section_offset(events, global_silences, lo, hi)
        local_span = chars[-1][2] - chars[0][1]

        sanity_ok = True
        if offset is not None:
            observed_gap = offset - offsets.get(section_order[i - 1], 0.0)
            ratio = observed_gap / max(expected_dur, 0.1)
            sanity_ok = SANITY_RATIO_MIN <= ratio <= SANITY_RATIO_MAX

        chapter_ok, chapter_note = True, None
        if offset is not None and chapters_usable:
            real_gap = chapters[i] - chapters[i - 1]
            raw_gap = offset - offsets.get(section_order[i - 1], 0.0)
            chapter_note = {"real_gap": real_gap, "raw_gap": raw_gap}
            if raw_gap > 0:
                chapter_ok = 0.0 <= real_gap <= raw_gap * 1.05

        if offset is not None and conf >= MIN_CONFIDENCE and sanity_ok and chapter_ok:
            offsets[name] = round(offset, 3)
            report[name] = {"confidence": round(conf, 3), "method": "pause_cross_correlation",
                             "gaps_total": len(events), "expected_dur_sec": round(expected_dur, 2),
                             "chapter_check": chapter_note}
            cursor = offset + local_span
        else:
            reason = ("no_cluster" if offset is None else
                       "low_confidence" if conf < MIN_CONFIDENCE else
                       "sanity_failed" if not sanity_ok else "chapter_mismatch")
            report[name] = {"confidence": round(conf, 3) if offset is not None else 0.0,
                             "method": "rejected", "reason": reason,
                             "gaps_total": len(events), "expected_dur_sec": round(expected_dur, 2),
                             "chapter_check": chapter_note}
            # Безопасный откат: офсет для этой секции НЕ пишем (см. докстринг
            # модуля — тот же эффект, что "файла вообще нет"). Структурный
            # курсор всё равно продвигаем лучшей доступной оценкой, чтобы
            # поиск СЛЕДУЮЩЕЙ секции не потерял окно целиком из-за одного отказа.
            cursor += max(local_span, expected_dur)

    return offsets, report


def save_results(video_dir, offsets, report):
    media_plan = os.path.join(video_dir, "media_plan")
    os.makedirs(media_plan, exist_ok=True)
    offsets_path = os.path.join(media_plan, "section_offsets.json")
    tmp = offsets_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(offsets, f, ensure_ascii=False, indent=2)
    os.replace(tmp, offsets_path)
    report_path = os.path.join(media_plan, "section_sync_report.json")
    tmp = report_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    os.replace(tmp, report_path)
    return offsets_path, report_path


def main():
    video_dir = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    offsets, report = detect_section_offsets(video_dir)
    if "error" in report:
        print(f"  Section Sync: {report['error']} — section_offsets.json не создан.")
        return 1
    if "_skipped" in report:
        print(f"  Section Sync: {report['_skipped']}")
        return 0
    offsets_path, report_path = save_results(video_dir, offsets, report)
    total_sections = len(report)
    resolved = sum(1 for v in report.values() if v.get("method") != "rejected")
    print(f"  Section Sync: {resolved}/{total_sections} секций получили офсет "
          f"({offsets_path}), отчёт по каждой — {report_path}")
    for name, r in report.items():
        if r.get("method") == "rejected":
            print(f"    ОТКАЗ [{name}]: reason={r.get('reason')} confidence={r.get('confidence')} "
                  f"— офсет не записан, даунстрим-код откатится на текущее поведение (offset=0.0)")
    unresolved = total_sections - resolved
    return 1 if unresolved else 0


if __name__ == "__main__":
    sys.exit(main())
