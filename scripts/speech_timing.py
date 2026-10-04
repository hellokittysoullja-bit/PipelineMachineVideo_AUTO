#!/usr/bin/env python3
"""PHRASE LOCK: когда в готовой озвучке начинается и кончается каждая фраза.

Источник — посимвольный alignment озвучки (media_plan/alignment/NN.csv, по
файлу на секцию), карта вырезанных пауз fix_pauses.py (pause_cuts.json) и
глобальные смещения секций (section_offsets.json, пишет lumean_tts.py).

Логика перенесена из pipeline_smart.py (PipelineMachineVideo_AUTO) без
изменений; отличие одно — состояние. Там папка ролика читалась из sys.argv
при импорте, а кэши и итоги (концы речи, причина отказа) жили глобальными
переменными модуля. Здесь всё это — поля SpeechTiming(video_dir): два
ролика в одном процессе не делят кэш, а импорт модуля ничего не читает."""
import csv
import difflib
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Словарь тегов живёт в ОДНОМ месте (script_parser): копии уже стоили
# эпизоду PHRASE LOCK. Имя переэкспортируется — на него ссылаются тесты.
from script_parser import ALIGNMENT_TAG_SPAN_RE  # noqa: E402

ALIGNMENT_TAG_RE = re.compile(r'\[short pause\]|\[pause\]')
ONSET_TEXT_MATCH_MIN_RATIO = 0.9   # ниже — текст блока разошёлся с озвученным, привязке нельзя доверять
WEIGHT_TEXT_MATCH_MIN_RATIO = 0.55  # вес блока терпимее онсета: длина речи переживает правку слова-двух


def raw_to_real_time(t, cuts):
    """Точный (не приближённый по тегам) пересчёт сырого времени alignment.csv
        (до обрезки пауз в fix_pauses.py) в реальное время audio_fixed.mp3 —
        вычитает из t суммарно всё, что реально вырезано ДО t (частично, если t
        попадает внутрь самого вырезанного интервала). cuts — из load_pause_cuts(),
        отсортированы по возрастанию (естественный порядок silencedetect)."""
    removed = 0.0
    for a, b in cuts:
        if a >= t:
            break
        removed += min(t, b) - a
    return t - removed


def _clean_timed_chars(segment):
    """Только РЕАЛЬНО озвученные символы сегмента с их временами
        [(char, start, end), ...] — без пробелов, скобок и служебных тегов вроде
        [energetic]/[slowly] (они не читаются TTS вслух, но формально попадают в
        поток символов alignment.csv). Общая основа для _real_speech_bounds()
        (границы сегмента) и load_alignment_onsets() (посимвольная привязка
        ПОД-блоков внутри сегмента) — одна реализация фильтра на оба места,
        иначе они молча разошлись бы при следующей правке."""
    text = "".join(c for c, s, e in segment)
    excluded = set()
    for m in ALIGNMENT_TAG_SPAN_RE.finditer(text):
        excluded.update(range(m.start(), m.end()))
    return [(c, s, e) for j, (c, s, e) in enumerate(segment)
            if j not in excluded and re.match(r'[^\s\[\]]', c or "")]


def speech_chars_of_text(text):
    """Та же нормализация, что _clean_timed_chars(), но для ТЕКСТА блока —
        строка из одних озвучиваемых символов. Позволяет сопоставить блок с его
        участком посимвольного alignment один-в-один: sub-cut'ы (см.
        split_long_blocks) делят слова исходного блока БЕЗ остатка, поэтому их
        очищенные тексты, склеенные подряд, посимвольно равны очищенному тексту
        сегмента — это и даёт точную, а не пропорциональную привязку под-кадра
        ко времени озвучки."""
    text = re.sub(r'\[[^\]]*\]', '', text or "")
    return re.sub(r'[\s\[\]]', '', text)


def _word_times(text, want, timed, to_real):
    """Слова блока с моментами их звучания: [{"word", "start", "end"}, ...].

    want — озвучиваемые символы блока (speech_chars_of_text), timed — символы
    alignment на той же позиции; они совпадают не буквально (ratio >= 0.9 —
    TTS мог прочитать «2025» словами), поэтому позиция каждого символа слова
    переносится через совпавшие куски SequenceMatcher, а не по индексу. Слово,
    ни один символ которого не совпал, получает время соседей по положению —
    честная оценка внутри уже измеренного блока, а не догадка по всему ролику."""
    got = "".join(c for c, s, e in timed).lower()
    m = difflib.SequenceMatcher(None, want.lower(), got, autojunk=False)
    pos_map = {}
    for a, b_, n in m.get_matching_blocks():
        for k in range(n):
            pos_map[a + k] = b_ + k
    words, i = [], 0
    for w in re.sub(r'\[[^\]]*\]', ' ', text or "").split():
        n = len(re.sub(r'[\s\[\]]', '', w))
        idx = [pos_map[j] for j in range(i, i + n) if j in pos_map]
        frac = (i / max(1, len(want)), (i + n) / max(1, len(want)))
        i += n
        if idx:
            st, en = timed[min(idx)][1], timed[max(idx)][2]
        else:
            lo, hi = timed[0][1], timed[-1][2]
            st, en = lo + (hi - lo)*frac[0], lo + (hi - lo)*frac[1]
        words.append({"word": w, "start": to_real(st), "end": to_real(en)})
    return words


def _real_speech_bounds(segment):
    """(старт, конец) реально озвученного текста в сегменте символов
        (index,char,start,end) — по первому и последнему озвученному символу
        (см. _clean_timed_chars). None, если в сегменте нет ни одного реального
        символа."""
    clean = _clean_timed_chars(segment)
    if not clean:
        return None
    return clean[0][1], clean[-1][2]


class SpeechTiming:
    """Тайминг речи одного ролика. Файлы читаются лениво и один раз:
    за время сборки они не меняются.

    После onsets(): speech_ends — ИЗМЕРЕННЫЙ конец речи каждого блока (та же
    шкала), failure — почему PHRASE LOCK не включился (None — включился или
    onsets() ещё не вызывался)."""

    def __init__(self, video_dir):
        self.video_dir = os.path.abspath(video_dir)
        plan = os.path.join(self.video_dir, "media_plan")
        self.alignment_dir = os.path.join(plan, "alignment")
        self.pause_cuts_path = os.path.join(plan, "pause_cuts.json")
        self.section_offsets_path = os.path.join(plan, "section_offsets.json")
        self._cuts = None
        self._offsets = None
        self.speech_ends = []
        self.word_times = []
        self.failure = None

    def pause_cuts(self):
        """[[вырезанный_старт, вырезанный_конец], ...] в СЫРОМ времени audio.mp3
            (см. fix_pauses.save_cuts) — реальные интервалы, которые
            silencedetect+atrim вычистили из аудио. Нет файла (старый эпизод, ещё
            не пересчитанный fix_pauses.py) -> [] (тихий откат на identity-маппинг,
            raw_to_real_time тогда просто возвращает t без изменений).

            Проверка отпечатка (source_audio_md5): pause_cuts.json описывает
            порезки КОНКРЕТНОГО audio.mp3 на момент запуска fix_pauses.py. Если
            audio.mp3 потом перезаписали (новая озвучка, правка) без повторного
            fix_pauses.py — порезки больше не соответствуют реальности, а
            raw_to_real_time() молча применил бы их к чужому аудио, воспроизводя
            ровно тот класс рассинхрона, который этот файл должен был исправить.
            Несовпадение -> предупреждение в консоль + откат на [] (честно "без
            точной подгонки", а не тихая порча по устаревшим данным)."""
        if self._cuts is not None:
            return self._cuts
        try:
            with open(self.pause_cuts_path, encoding="utf-8") as f:
                data = json.load(f)
            raw_audio_path = os.path.join(self.video_dir, "audio.mp3")
            expected_md5 = data.get("source_audio_md5")
            if expected_md5 and os.path.exists(raw_audio_path):
                h = hashlib.md5()
                with open(raw_audio_path, "rb") as af:
                    for chunk in iter(lambda: af.read(1 << 20), b""):
                        h.update(chunk)
                if h.hexdigest() != expected_md5:
                    print("  ВНИМАНИЕ: pause_cuts.json не соответствует текущему audio.mp3 "
                          "(перезаписали озвучку без повторного fix_pauses.py?) — "
                          "порезки игнорирую, подписи хука пересчитаются без точной подгонки паузы.")
                    self._cuts = []
                    return self._cuts
            self._cuts = [(float(a), float(b)) for a, b in data.get("cuts", [])]
        except Exception:
            self._cuts = []
        return self._cuts

    def section_offsets(self):
        """{section_name: raw_global_offset_sec} из media_plan/section_offsets.json —
            пишет scripts/speech_generate.py Stage B (сам конкатенирует фрагменты,
            знает точно) ИЛИ scripts/section_sync.py (кросс-корреляция паттерна
            пауз против реального аудио — для эпизодов без Stage B, см. докстринг
            section_sync.py). Нет файла -> {} (тихий откат: offsets.get(name, 0.0)
            везде ниже даёт РОВНО прежнее поведение — локальное время секции
            трактуется как глобальное, корректно только для первой секции по
            определению, как и было до этой карты)."""
        if self._offsets is not None:
            return self._offsets
        try:
            with open(self.section_offsets_path, encoding="utf-8") as f:
                data = json.load(f)
            self._offsets = {str(k): float(v) for k, v in data.items()}
        except Exception:
            self._offsets = {}
        return self._offsets

    def real_speech_span(self, segment, section_offset=0.0):
        """РЕАЛЬНАЯ (после обрезки пауз fix_pauses.py, см. raw_to_real_time)
            длительность озвученного текста в сегменте — см. _real_speech_bounds
            (общая логика, тег-агностичная граница). Раньше возвращала СЫРУЮ
            длительность (до обрезки) — а именно эта величина кормит вес блока в
            load_alignment_weights()/block_durations(), то есть раньше ДЛИТЕЛЬНОСТЬ
            КАДРА блока могла быть завышена, если внутри блока TTS сделал
            незапланированную (не тегом [pause]) паузу/вдох длиннее секунды —
            silencedetect в fix_pauses.py режет ЛЮБУЮ такую тишину, независимо от
            тегов в тексте (пойман вживую: 41 реальная порезка на 90 тегов паузы в
            эпизоде — то есть часть порезок вообще не в тех местах, где текст
            формально ожидал паузу). Теперь вес блока — точно то время, что он
            реально звучит в audio_fixed.mp3, а не в сыром audio.mp3.

            section_offset — сдвиг ЛОКАЛЬНОГО времени сегмента (свой ноль на файл
            alignment/NN.csv) в ГЛОБАЛЬНОЕ (по всему audio.mp3, та же шкала, что и
            cuts из load_pause_cuts() — см. load_section_offsets()). Раньше bounds
            (локальные) подавались в raw_to_real_time() НАПРЯМУЮ, как будто уже
            глобальные — для первой секции (HOOK) это случайно верно (локальный
            ноль совпадает с глобальным), для всех следующих секций cuts из ДРУГИХ
            участков ролика применялись не туда, давая неверный вес блока —
            реальный, эмпирически подтверждённый баг (см. section_sync.py)."""
        bounds = _real_speech_bounds(segment)
        if not bounds:
            return 0.0
        cuts = self.pause_cuts()
        real_start = raw_to_real_time(bounds[0] + section_offset, cuts)
        real_end = raw_to_real_time(bounds[1] + section_offset, cuts)
        return real_end - real_start

    def section_segments(self, blocks):
        """{секция: [сегмент, ...]} — разборка media_plan/alignment/NN.csv на
            сегменты по тегам пауз. Общая основа для load_alignment_weights() (вес
            блока) и load_alignment_onsets() (момент НАЧАЛА блока) — одна реализация
            на оба места. None, если папки alignment нет вообще."""
        if not os.path.isdir(self.alignment_dir):
            return None
        section_order = []
        for b in blocks:
            if not section_order or section_order[-1] != b["section"]:
                section_order.append(b["section"])
        section_segments = {}
        for i, name in enumerate(section_order):
            path = os.path.join(self.alignment_dir, f"{i:02d}.csv")
            if not os.path.exists(path):
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                chars = [(r["char"], float(r["start"]), float(r["end"])) for r in rows]
            except Exception as e:
                print(f"  alignment {path} битый, пропускаю: {e}")
                continue
            text = "".join(c for c, s, e in chars)
            segs, pos = [], 0
            for m in ALIGNMENT_TAG_RE.finditer(text):
                segs.append(chars[pos:m.start()])
                pos = m.end()
            segs.append(chars[pos:])
            section_segments[name] = segs
        return section_segments

    def onsets(self, blocks):
        """РЕАЛЬНЫЙ момент НАЧАЛА речи каждого блока (сек, в шкале финального
            audio_fixed — после обрезки пауз fix_pauses.py). Список длиной len(blocks)
            или None, если хотя бы для одного блока привязка не подтвердилась
            (fail-open целиком, а не частично: половина точных онсетов и половина
            угаданных дала бы рассинхрон хуже, чем честная старая оценка).

            ЗАЧЕМ ЭТО ЕСТЬ (реальная, измеренная причина — 28 августа, разбор жалобы
            «видео показывается позже, когда фраза уже сказана»): длительности кадров
            считались по load_alignment_weights() — то есть по ДЛИНЕ речи блока, а не
            по её ПОЛОЖЕНИЮ. Дальше эта оценка проходила через семь преобразований
            (floor/cap-клэмп и rescale в block_durations, energy_mults, apply_section_
            boundary_shift ±250-400мс, apply_within_cut_shift ±80-180мс,
            apply_human_jitter ±300мс, snap_hook_cuts_to_energy), каждое из которых
            двигает границу кадра ОТНОСИТЕЛЬНО фразы, а ошибки складываются по
            накоплению — к концу хука рез уезжает от своей фразы на доли секунды и
            больше. Зритель видит ровно то, на что жаловался пользователь: кадр
            «не под озвучку».

            Длина речи не даёт положения — его даёт только ОНСЕТ. Поэтому здесь
            возвращается именно момент начала, и по нему (см. phrase_locked_durations)
            рез ставится ТОЧНО на начало следующей фразы.

            ТОЧНОСТЬ ДЛЯ ПОД-КАДРОВ: sub-cut'ы (split_long_blocks) делят слова
            исходного блока без остатка, поэтому очищенные тексты под-блоков,
            склеенные подряд, посимвольно равны очищенному тексту сегмента —
            привязка идёт по РЕАЛЬНОМУ символу alignment, а не пропорционально
            числу слов (как считается вес в load_alignment_weights).

            БЕЗОПАСНОСТЬ: текст каждого блока сверяется с реально озвученными
            символами на его позиции (ONSET_TEXT_MATCH_MIN_RATIO). Сценарий,
            поправленный ПОСЛЕ записи озвучки, не получит чужой тайминг молча —
            вернётся None, и сборка честно откатится на прежнее поведение."""
        self.failure = None
        self.speech_ends = []
        self.word_times = []

        def _give_up(reason, **detail):
            """Запомнить ПРИЧИНУ отказа, а не просто вернуть None.

                    N5 (docs/AUDIT_2026-09_DEEP.md:281): у этой функции пять разных
                    точек отказа, и каждая молча выключала PHRASE LOCK на ВЕСЬ эпизод —
                    в консоли при этом не появлялось ни строчки. Самый вероятный случай
                    (нормализация произношения в speech_generate.py: TTS получает
                    "1,5 килограмма", а script.txt содержит "1,5 кг") выглядел как
                    обычный рендер, но кадры переставали держаться за фразы.
                    """
            self.failure = {"reason": reason, **detail}
            return None

        section_segments = self.section_segments(blocks)
        if not section_segments:
            return _give_up("нет ни одного сегмента alignment (media_plan/alignment/*.csv)")
        section_offsets = self.section_offsets()
        cuts = self.pause_cuts()
        onsets, ends = [], []
        seg_idx, char_pos = {}, {}
        for bi, b in enumerate(blocks):
            section = b["section"]
            segs = section_segments.get(section)
            if segs is None:
                return _give_up("для секции нет alignment", section=section, block_index=bi)
            k = seg_idx.get(section, 0)
            if k >= len(segs):
                return _give_up("блоков больше, чем сегментов записи — доверять нечему",
                                section=section, block_index=bi)
            clean = _clean_timed_chars(segs[k])
            pos = char_pos.get(section, 0)
            want = speech_chars_of_text(b["text"])
            if not want or pos + len(want) > len(clean):
                return _give_up("текст блока не помещается в оставшийся alignment",
                                section=section, block_index=bi,
                                text=b["text"][:60], want_chars=len(want),
                                available_chars=len(clean) - pos)
            got = "".join(c for c, s, e in clean[pos:pos + len(want)])
            ratio = difflib.SequenceMatcher(None, want.lower(), got.lower()).ratio()
            if ratio < ONSET_TEXT_MATCH_MIN_RATIO:
                return _give_up("текст блока разошёлся с озвученным",
                                section=section, block_index=bi, ratio=round(ratio, 3),
                                threshold=ONSET_TEXT_MATCH_MIN_RATIO,
                                script_text=b["text"][:60], spoken_text=got[:60])
            offset = section_offsets.get(section, 0.0)
            onsets.append(raw_to_real_time(clean[pos][1] + offset, cuts))
            # КОНЕЦ речи блока берётся ОТТУДА ЖЕ, где и начало — из символов
            # alignment. Раньше его нигде не было, и потребителям (sfx_plan)
            # приходилось складывать точный онсет с ВЕСОМ блока, а вес после
            # split_long_blocks делится между кусками ПРОПОРЦИОНАЛЬНО словам,
            # то есть является оценкой. Сложение измерения с оценкой — ровно тот
            # класс, который этот файл запрещает, и цена измерена (14.09,
            # videos/_test60s): у 3 границ из 9 «конец речи» оказывался ПОЗЖЕ
            # онсета следующего блока, тишина выходила отрицательной, и
            # speech_gap_before() честно возвращала «нет сигнала». Молча
            # пропадали и переход главы, и объектный кюй — на трети границ.
            ends.append(raw_to_real_time(clean[pos + len(want) - 1][2] + offset, cuts))
            self.word_times.append(_word_times(b["text"], want, clean[pos:pos + len(want)],
                                               lambda t: raw_to_real_time(t + offset, cuts)))
            pos += len(want)
            if pos >= len(clean):
                seg_idx[section] = k + 1
                char_pos[section] = 0
            else:
                char_pos[section] = pos
        self.speech_ends = ends
        return onsets

    def weights(self, blocks):
        """Реальные веса длительности блоков из посимвольного alignment.csv
            (ElevenLabs/Lumean отдают его бесплатно вместе с каждым TTS-заказом —
            раньше просто не забирался). Раньше длительность блока считалась ЧИСТО
            по числу слов и общей скорости слов/сек — то есть каждое слово "весило"
            одинаково. На деле темп TTS живой: короткая ударная фраза читается
            быстрее длинной со сложными словами, [slowly]-блок — заметно медленнее.
            Реальный посимвольный тайминг ловит эту вариацию, а не усредняет её.

            Файлы лежат в media_plan/alignment/00.csv, 01.csv... — по одному на
            КАЖДУЮ секцию (HOOK, BLOCK 1, ..., FINAL) в порядке их первого появления
            в script.txt, разрезаны по буквальным вхождениям [pause]/[short pause]
            на сегменты 1:1 с под-блоками этой секции в том же порядке.

            Возвращает список той же длины, что blocks, — вес (сек) реальной речи
            на блок или None для блоков без данных (эпизод без сохранённого
            alignment, или новый блок, добавленный после записи — сборка тогда
            просто откатывается на word-count оценку для этих блоков, как раньше)."""
        section_segments = self.section_segments(blocks)
        if section_segments is None:
            return None
        section_offsets = self.section_offsets()   # {} без смещений -> 0.0 для всех секций
        weights = []
        seg_cursor = {}
        stale = 0
        for b in blocks:
            segs = section_segments.get(b["section"])
            if segs is None:
                weights.append(None)
                continue
            offset = section_offsets.get(b["section"], 0.0)
            k = seg_cursor.get(b["section"], 0)
            if k >= len(segs):
                # Число под-блоков в текущем script.txt разошлось с числом
                # сегментов в сохранённом alignment (текст правили после
                # записи) — честно откатываемся на word-count для остатка
                # секции, а не подставляем чужой сегмент не по месту.
                weights.append(None)
            else:
                # Число сегментов может совпасть, а ТЕКСТ — разойтись (слово-два
                # поправили в script.txt после записи озвучки, не тронув счётчик
                # [pause]) — тогда старый код молча применял чужой тайминг к
                # новому тексту. Fuzzy-сверка (difflib, без новых зависимостей)
                # блока против реально озвученного сегмента: разошлись сильнее
                # порога — честный откат на word-count вместо тихой подмены.
                seg_text = re.sub(r'\s+', ' ', "".join(c for c, s, e in segs[k])).strip().lower()
                block_text = re.sub(r'\s+', ' ', b["text"]).strip().lower()
                ratio = (difflib.SequenceMatcher(None, block_text, seg_text).ratio()
                         if (block_text and seg_text) else 0.0)
                if ratio < WEIGHT_TEXT_MATCH_MIN_RATIO:
                    weights.append(None)
                    stale += 1
                else:
                    span = self.real_speech_span(segs[k], offset)
                    weights.append(span if span > 0.05 else None)
            seg_cursor[b["section"]] = k + 1
        if stale:
            print(f"  Alignment: {stale} блок(ов) текстом разошлись с сохранённым таймингом "
                  f"(script.txt правили после записи?) — откат на word-count для них")
        return weights
