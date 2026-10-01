#!/usr/bin/env python3
"""Привязка кадров к фразам и мастеринг звука — функции pipeline_smart.py
(PipelineMachineVideo_AUTO), перенесённые ДОСЛОВНО и только они.

Корни отбора (всё остальное подтянуто деревом ссылок):
  * load_alignment_onsets / load_alignment_weights — PHRASE LOCK: начало
    каждой фразы по посимвольному alignment озвучки, с картой вырезанных
    пауз fix_pauses.py и смещениями секций;
  * process_voice — срез низов, EQ, де-эссер, компрессор голоса;
  * music_bed_gain_db — уровень музыки по ЗАМЕРУ громкости (разрыв 16 LU);
  * measure_loudnorm_stats / build_master_af — двухпроходный loudnorm
    -14 LUFS + лимитер; audio_qc — клиппинг и аномальная громкость;
  * write_subtitles (2 строки, ~42 символа) и write_chapters.

Отличия от исходника ровно два, оба — среда, а не логика:
  * папку ролика задаёт configure(video_dir), а не sys.argv на импорте;
  * вместо реестра флагов на 50 режимов — прослойка на два, что здесь
    читаются (VOICE_PROCESS, MASTER_LIMITER; дефолты те же: 1)."""
import csv
import difflib
import hashlib
import json
import math
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env  # noqa: E402
from script_parser import ALIGNMENT_TAG_SPAN_RE, PAUSE_DURATIONS, parse_blocks  # noqa: E402,F401


class feature_flags:  # noqa: N801 — имя модуля исходника, чтобы код ниже остался дословным
    DEFAULTS = {"VOICE_PROCESS": "1", "MASTER_LIMITER": "1"}

    @staticmethod
    def enabled(name):
        return os.environ.get(name, feature_flags.DEFAULTS.get(name, "0")).strip().lower() in ("1", "true", "on", "yes")


_ARGV_POSITIONAL = [a for a in sys.argv[1:] if not a.startswith("--")]


VIDEO_FOLDER = _ARGV_POSITIONAL[0] if _ARGV_POSITIONAL else os.getcwd()


# Насколько подложка должна быть тише голоса ДО сайдчейн-дакинга (дакинг —
# страховка на пиках речи, не единственная линия обороны против "музыка
# спорит с закадром"). Это ЗАДУМАННЫЙ разрыв, а не готовое усиление:
# реальное усиление считается под конкретный ассет, см. music_bed_gain_db().
#
# РЕАЛЬНЫЙ найденный баг (07.09, измерено ebur128 на живых файлах эпизода
# 01_ves-mecha, не гипотеза). Здесь стояло глухое MUSIC_BED_GAIN_DB=-13.0, а
# комментарий рядом обосновывал его "собственными -12dBFS ассета". -12 dBFS —
# это ПИК ассета, а не его громкость: интегральная громкость
# assets/music/ambient_bed.flac — -30.0 LUFS, голос эпизода — -16.0 LUFS.
# То есть разрыв составлял 14 LU ЕЩЁ ДО применения -13 дБ, а после —
# 27 LU вместо задуманных 16. Музыку в опубликованном ролике практически
# не слышно, и никакой отчёт этого не показывал: loudnorm выравнивает микс
# ЦЕЛИКОМ, поэтому итоговые -14 LUFS выглядели идеально при неслышимой
# подложке. Число -13 было не "слишком тихо на вкус", а двойным учётом.
MUSIC_BED_GAP_LU = 16.0


# Запасное значение, если измерить громкость не удалось. Посчитано под
# ТЕКУЩИЕ ассеты канала (-30.0 LUFS против -16.0 LUFS голоса): для другого
# набора музыки оно снова будет неверным — ровно поэтому основной путь
# меряет, а не берёт константу.
MUSIC_BED_GAIN_DB = -2.0


# Клэмп на вычисленное усиление: сорванное измерение (пустой/битый ассет
# даёт -inf или -70 LUFS) иначе попросило бы +50 дБ и разнесло бы микс.
# Границы намеренно широкие — они защищают от абсурда, а не подменяют расчёт.
MUSIC_BED_GAIN_MIN_DB, MUSIC_BED_GAIN_MAX_DB = -30.0, 6.0


# Порог в ЛИНЕЙНОЙ амплитуде (ffmpeg sidechaincompress, не дБ): реальный
# голос в проекте — mean -16dB/max -1.7dB (см. volumedetect на audio_fixed.mp3),
# 0.06 ~= -24dB — уверенно ловит речь, не ловит фоновую тишину/дыхание между
# фразами (там подложка может чуть подняться — естественное "дыхание" мастера,
# как в живом монтаже, а не плоская постоянная громкость).
MUSIC_DUCK_THRESHOLD = 0.06


MUSIC_DUCK_RATIO = 9.0


MUSIC_DUCK_ATTACK_MS = 15.0    # быстро прижать к началу фразы


MUSIC_DUCK_RELEASE_MS = 450.0  # плавно отпустить после — без эффекта "накачки"


# A6: цепочка обработки самого голоса (сырой TTS -> вещательного качества),
# ДО подмешивания музыки/дакинга — то, что обычно делает звукорежиссёр с
# дорожкой диктора: срез суб-баса (в голосе его быть не должно, только гул),
# лёгкая тональная коррекция (тепло внизу, разборчивость наверху — TTS
# нейтрален "из коробки", не спроектирован под смесь с музыкой), деэссер
# (TTS-движки нередко дают резкие "с/ш" на определённых голосах/скоростях),
# мягкая компрессия РОВНО голоса (не путать с loudnorm — тот выравнивает
# ГРОМКОСТЬ ролика целиком, это — микро-динамика внутри фраз, чтобы тихие
# слова не тонули под подложкой ДО дакинга).
VOICE_PROCESS_ENABLED = feature_flags.enabled("VOICE_PROCESS")


VOICE_HIGHPASS_HZ = 80


VOICE_EQ_WARMTH_HZ, VOICE_EQ_WARMTH_GAIN = 200, 1.5


VOICE_EQ_PRESENCE_HZ, VOICE_EQ_PRESENCE_GAIN = 3000, 2.0


VOICE_DEESS_INTENSITY = 0.3


VOICE_COMPRESS_THRESHOLD = 0.15


VOICE_COMPRESS_RATIO = 2.5


# Лимитер в самом конце мастер-цепочки — последняя ступень, которую ставит
# любой звукорежиссёр перед выдачей. Раньше её не было вообще: цепочка
# заканчивалась loudnorm + afade, и единственной защитой от пиков был
# TP-режим самого loudnorm. Он честно держит true peak, ПОКА линейного
# усиления хватает; в остальных случаях ffmpeg уходит в динамический режим
# и гарантия становится мягкой. Лимитер стоит копейки и делает потолок
# безусловным — особенно теперь, когда подложка стала на 11 дБ громче
# (см. MUSIC_BED_GAP_LU) и суммарные пики микса выросли.
# level=disabled — критично: включённый auto-level поднял бы громкость
# обратно и обнулил только что выставленный loudnorm.
MASTER_LIMITER_ENABLED = feature_flags.enabled("MASTER_LIMITER")


MASTER_LIMITER_ATTACK_MS = 5


MASTER_LIMITER_RELEASE_MS = 50


def section_title(name):
    """BLOCK N: Название -> 'Название'. HOOK/FINAL/безымянные BLOCK — без титра."""
    m = re.match(r'BLOCK\s+\d+\s*:\s*(.+)', name, re.I)
    return m.group(1).strip() if m else None


# Единый источник целевой громкости — ЕДИНСТВЕННОЕ место, которое трогать
# при смене цели. Раньше -14 (было -16) была рассыпана литералом по пяти
# местам вручную (дефолт measure_loudnorm_stats, два af= в финальном
# миксе, print_format=json первого прохода, sanity-проверка готового
# файла) — РЕАЛЬНЫЙ найденный баг (02.09): при переходе -16 -> -14 пятое
# место (sanity-проверка) осталось на старом значении, и честный рендер
# на -14.3 (в пределах цели) печатал ложную тревогу "разошлось с целью
# -16". Четыре места нашли и поправили сразу, пятое — нет, потому что
# ничего в коде не требовало держать их согласованными. Константа не
# устраняет риск рассинхрона полностью (использования всё ещё нужно
# найти и подставить руками), но делает следующую смену цели ОДНИМ
# изменением вместо пяти grep'ов по литералу.
LOUDNORM_TARGET_I = -14.0


LOUDNORM_TARGET_TP = -1.5


LOUDNORM_TARGET_LRA = 11


def _audio_len_for_timeout(path):
    try:
        return get_media_duration(path)
    except Exception:
        return 600.0


def measure_loudnorm_stats(audio_path, target_i=LOUDNORM_TARGET_I,
                            target_tp=LOUDNORM_TARGET_TP, target_lra=LOUDNORM_TARGET_LRA):
    """Первый проход двух-проходного loudnorm — только измерение, звук не
    трогаем. Однопроходный (динамический) режим подстраивает усиление на
    лету по ходу файла и даёт неровную громкость внутри ролика и неточный
    итоговый integrated — это ffmpeg-документация прямо называет compromise-
    режимом, не тем, что стоит использовать для финального мастеринга.
    None при любой ошибке — main() откатывается на старый однопроходный af."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-i", audio_path, "-af",
             f"loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}:print_format=json",
             "-f", "null", "-"], capture_output=True, text=True, encoding="utf-8", errors="replace",
            # Аудит 04.09: измерение — полный декод + ebur128 (~34x realtime на
            # 4 ядрах): при фиксированных 60с любой эпизод длиннее ~34 минут
            # молча уходил в однопроходный loudnorm ("compromise-режим").
            timeout=max(120, int(_audio_len_for_timeout(audio_path) * 0.5)))
        start, end = r.stderr.rindex("{"), r.stderr.rindex("}") + 1
        return json.loads(r.stderr[start:end])
    except Exception as e:
        print(f"  ВНИМАНИЕ: двухпроходный loudnorm не измерился ({type(e).__name__}) — "
              f"финал пойдёт через однопроходный режим (громкость внутри ролика ровнее не будет)")
        return None


def measure_integrated_lufs(audio_path):
    """Интегральная громкость дорожки в LUFS (ebur128), None при сбое.

    Отдельно от measure_loudnorm_stats(): та печатает сообщение про откат
    финального мастеринга в однопроходный режим, которое к измерению уровня
    подложки отношения не имеет и только путало бы лог. Таймаут — по той же
    формуле от длительности, что и там (аудит 04.09: фиксированные 60 с
    молча резали любой эпизод длиннее ~34 минут).
    """
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostats", "-i", audio_path,
             "-af", "ebur128=framelog=quiet", "-f", "null", "-"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=max(120, int(_audio_len_for_timeout(audio_path) * 0.5)))
        m = re.findall(r"I:\s*(-?[\d.]+)\s*LUFS", r.stderr)
        return float(m[-1]) if m else None
    except Exception:
        return None


def music_bed_gain_db(voice_path, music_path):
    """Усиление подложки под ЭТОТ голос и ЭТОТ ассет: (дБ, чем обосновано).

    Считается, а не берётся константой, потому что константа уже один раз
    молча разошлась с реальностью (см. MUSIC_BED_GAP_LU выше: задумано 16 LU,
    в опубликованном ролике вышло 27). Замена музыкального ассета или смена
    движка озвучки сдвигает обе громкости независимо — единственный способ
    удержать ЗАДУМАННЫЙ разрыв — измерить обе стороны на месте.

    Разрыв считается по интегральной громкости (LUFS), а не по пикам:
    слышимость подложки под речью определяется именно средней громкостью,
    пик у тихого дрона может быть каким угодно.
    """
    voice_lufs = measure_integrated_lufs(voice_path)
    music_lufs = measure_integrated_lufs(music_path)
    detail = {"voice_lufs": voice_lufs, "music_lufs": music_lufs,
              "target_gap_lu": MUSIC_BED_GAP_LU}
    if voice_lufs is None or music_lufs is None:
        detail.update(gain_db=MUSIC_BED_GAIN_DB, source="fallback_constant")
        print(f"  ВНИМАНИЕ: громкость дорожек не измерилась — подложка идёт по "
              f"запасной константе {MUSIC_BED_GAIN_DB} dB, задуманный разрыв "
              f"{MUSIC_BED_GAP_LU:.0f} LU НЕ гарантирован")
        return MUSIC_BED_GAIN_DB, detail
    raw = voice_lufs - MUSIC_BED_GAP_LU - music_lufs
    gain = max(MUSIC_BED_GAIN_MIN_DB, min(MUSIC_BED_GAIN_MAX_DB, raw))
    detail.update(gain_db=round(gain, 2), raw_gain_db=round(raw, 2),
                  source="measured", clamped=abs(gain - raw) > 1e-6)
    if detail["clamped"]:
        print(f"  ВНИМАНИЕ: расчётное усиление подложки {raw:+.1f} dB вышло за "
              f"[{MUSIC_BED_GAIN_MIN_DB}, {MUSIC_BED_GAIN_MAX_DB}] — обрезано до "
              f"{gain:+.1f} dB, разрыв будет не {MUSIC_BED_GAP_LU:.0f} LU")
    else:
        print(f"  Подложка: голос {voice_lufs:.1f} LUFS, музыка {music_lufs:.1f} LUFS "
              f"-> усиление {gain:+.1f} dB (разрыв {MUSIC_BED_GAP_LU:.0f} LU)")
    return gain, detail


def process_voice(voice_path, out_path):
    """A6: обработка сырой TTS-дорожки перед миксом с музыкой — срез
    суб-баса, лёгкая тональная коррекция, деэссер, мягкая компрессия
    микро-динамики фраз. VOICE_PROCESS_ENABLED=False -> просто копия
    исходника (безопасный откат, тот же принцип, что MUSIC_ENABLED)."""
    if not VOICE_PROCESS_ENABLED:
        r = subprocess.run(["ffmpeg", "-y", "-i", voice_path, "-ar", "48000", "-ac", "2", out_path],
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
        return out_path if r.returncode == 0 else voice_path
    af = (
        f"highpass=f={VOICE_HIGHPASS_HZ},"
        f"equalizer=f={VOICE_EQ_WARMTH_HZ}:width_type=o:width=1.5:g={VOICE_EQ_WARMTH_GAIN},"
        f"equalizer=f={VOICE_EQ_PRESENCE_HZ}:width_type=o:width=1.2:g={VOICE_EQ_PRESENCE_GAIN},"
        f"deesser=i={VOICE_DEESS_INTENSITY}:m=0.4,"
        f"acompressor=threshold={VOICE_COMPRESS_THRESHOLD}:ratio={VOICE_COMPRESS_RATIO}:"
        f"attack=8:release=120:makeup=1.15"
    )
    r = subprocess.run(["ffmpeg", "-y", "-i", voice_path, "-af", af, "-ar", "48000", "-ac", "2", out_path],
                        capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(f"  ВНИМАНИЕ: обработка голоса не собралась, голос идёт как есть: {r.stderr[-200:].strip()}")
        r2 = subprocess.run(["ffmpeg", "-y", "-i", voice_path, "-ar", "48000", "-ac", "2", out_path],
                             capture_output=True, text=True, encoding="utf-8", errors="replace")
        return out_path if r2.returncode == 0 else voice_path
    return out_path


ALIGNMENT_DIR = os.path.join(VIDEO_FOLDER, "media_plan", "alignment")


ALIGNMENT_TAG_RE = re.compile(r'\[short pause\]|\[pause\]')


# ALIGNMENT_TAG_SPAN_RE импортируется из script_parser (см. его докстринг:
# словарь тегов живёт в ОДНОМ месте — копии уже стоили эпизоду PHRASE LOCK).
# Имя переэкспортируется здесь, потому что на pipeline_smart.ALIGNMENT_TAG_SPAN_RE
# ссылается section_sync.py и тесты.
PAUSE_CUTS_PATH = os.path.join(VIDEO_FOLDER, "media_plan", "pause_cuts.json")


_PAUSE_CUTS_CACHE = None   # ленивый кэш на процесс — файл не меняется за время рендера


def load_pause_cuts():
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
    global _PAUSE_CUTS_CACHE
    if _PAUSE_CUTS_CACHE is not None:
        return _PAUSE_CUTS_CACHE
    try:
        with open(PAUSE_CUTS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        raw_audio_path = os.path.join(VIDEO_FOLDER, "audio.mp3")
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
                _PAUSE_CUTS_CACHE = []
                return _PAUSE_CUTS_CACHE
        _PAUSE_CUTS_CACHE = [(float(a), float(b)) for a, b in data.get("cuts", [])]
    except Exception:
        _PAUSE_CUTS_CACHE = []
    return _PAUSE_CUTS_CACHE


_PAUSE_INSERTS_CACHE = None


def load_pause_inserts():
    """[(сырая_позиция, секунд_вставлено), ...] из того же pause_cuts.json.

    Отдельный ключ, а не третий элемент cuts: raw_to_real_time()
    распаковывает cuts строго как пары (a, b) по всему файлу, и менять эту
    форму ради нового потребителя — тот же класс, что уже ловили у
    pause_windows. Нет ключа (эпизод старше правки) -> [] и прежнее
    поведение."""
    global _PAUSE_INSERTS_CACHE
    if _PAUSE_INSERTS_CACHE is not None:
        return _PAUSE_INSERTS_CACHE
    try:
        with open(PAUSE_CUTS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        items = [(float(a), float(b)) for a, b in (data.get("pause_inserts") or [])]
        _PAUSE_INSERTS_CACHE = sorted(items)
    except Exception:
        _PAUSE_INSERTS_CACHE = []
    return _PAUSE_INSERTS_CACHE


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
    # ВСТАВЛЕННАЯ тишина — вторая половина той же карты, и читается она
    # ЗДЕСЬ, а не у шести вызывающих. Причина ровно та, что уже дважды
    # стоила этому репозиторию сломанного тайминга: карту времени, которую
    # надо передавать руками, рано или поздно кто-нибудь не передаст.
    # Вставки появляются, когда fix_pauses.py доводит тег-паузу до её
    # документированной длины (ЧАСТЬ 10): движок отдаёт [short pause] как
    # 0.169с вместо 0.4 (замер 14.09), и без доведения звук в такую щель не
    # влезает никогда. Старый эпизод без ключа pause_inserts даёт пустой
    # список — поведение байт-в-байт прежнее.
    added = 0.0
    for pos, sec in load_pause_inserts():
        if pos >= t:
            break
        added += float(sec)
    return t - removed + added


_SECTION_OFFSETS_CACHE = None   # ленивый кэш на процесс, как и _PAUSE_CUTS_CACHE


SECTION_OFFSETS_PATH = os.path.join(VIDEO_FOLDER, "media_plan", "section_offsets.json")


def load_section_offsets():
    """{section_name: raw_global_offset_sec} из media_plan/section_offsets.json —
    пишет scripts/speech_generate.py Stage B (сам конкатенирует фрагменты,
    знает точно) ИЛИ scripts/section_sync.py (кросс-корреляция паттерна
    пауз против реального аудио — для эпизодов без Stage B, см. докстринг
    section_sync.py). Нет файла -> {} (тихий откат: offsets.get(name, 0.0)
    везде ниже даёт РОВНО прежнее поведение — локальное время секции
    трактуется как глобальное, корректно только для первой секции по
    определению, как и было до этой карты)."""
    global _SECTION_OFFSETS_CACHE
    if _SECTION_OFFSETS_CACHE is not None:
        return _SECTION_OFFSETS_CACHE
    try:
        with open(SECTION_OFFSETS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        _SECTION_OFFSETS_CACHE = {str(k): float(v) for k, v in data.items()}
    except Exception:
        _SECTION_OFFSETS_CACHE = {}
    return _SECTION_OFFSETS_CACHE


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


def _real_speech_bounds(segment):
    """(старт, конец) реально озвученного текста в сегменте символов
    (index,char,start,end) — по первому и последнему озвученному символу
    (см. _clean_timed_chars). None, если в сегменте нет ни одного реального
    символа."""
    clean = _clean_timed_chars(segment)
    if not clean:
        return None
    return clean[0][1], clean[-1][2]


def _real_speech_span(segment, section_offset=0.0):
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
    cuts = load_pause_cuts()
    real_start = raw_to_real_time(bounds[0] + section_offset, cuts)
    real_end = raw_to_real_time(bounds[1] + section_offset, cuts)
    return real_end - real_start


def _alignment_section_segments(blocks):
    """{секция: [сегмент, ...]} — разборка media_plan/alignment/NN.csv на
    сегменты по тегам пауз. Общая основа для load_alignment_weights() (вес
    блока) и load_alignment_onsets() (момент НАЧАЛА блока) — одна реализация
    на оба места. None, если папки alignment нет вообще."""
    if not os.path.isdir(ALIGNMENT_DIR):
        return None
    section_order = []
    for b in blocks:
        if not section_order or section_order[-1] != b["section"]:
            section_order.append(b["section"])
    section_segments = {}
    for i, name in enumerate(section_order):
        path = os.path.join(ALIGNMENT_DIR, f"{i:02d}.csv")
        if not os.path.exists(path):
            continue
        try:
            rows = list(csv.DictReader(open(path, encoding="utf-8")))
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


ONSET_TEXT_MATCH_MIN_RATIO = 0.9   # ниже — текст блока разошёлся с озвученным, привязке нельзя доверять


def load_alignment_onsets(blocks):
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
    global ALIGNMENT_ONSET_FAILURE
    ALIGNMENT_ONSET_FAILURE = None
    del SPEECH_ENDS[:]

    def _give_up(reason, **detail):
        """Запомнить ПРИЧИНУ отказа, а не просто вернуть None.

        N5 (docs/AUDIT_2026-09_DEEP.md:281): у этой функции пять разных
        точек отказа, и каждая молча выключала PHRASE LOCK на ВЕСЬ эпизод —
        в консоли при этом не появлялось ни строчки. Самый вероятный случай
        (нормализация произношения в speech_generate.py: TTS получает
        "1,5 килограмма", а script.txt содержит "1,5 кг") выглядел как
        обычный рендер, но кадры переставали держаться за фразы.
        """
        global ALIGNMENT_ONSET_FAILURE
        ALIGNMENT_ONSET_FAILURE = {"reason": reason, **detail}
        return None

    section_segments = _alignment_section_segments(blocks)
    if not section_segments:
        return _give_up("нет ни одного сегмента alignment (media_plan/alignment/*.csv)")
    section_offsets = load_section_offsets()
    cuts = load_pause_cuts()
    onsets = []
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
        SPEECH_ENDS.append(raw_to_real_time(clean[pos + len(want) - 1][2] + offset, cuts))
        pos += len(want)
        if pos >= len(clean):
            seg_idx[section] = k + 1
            char_pos[section] = 0
        else:
            char_pos[section] = pos
    return onsets


# ИЗМЕРЕННЫЙ конец речи каждого блока (реальная шкала), заполняется
# load_alignment_onsets() тем же посимвольным проходом, что и онсеты.
# Существует ровно затем, чтобы потребителям тишины не приходилось
# складывать точный онсет с пропорциональной оценкой веса — см. комментарий
# у самого заполнения.
SPEECH_ENDS = []


# Почему PHRASE LOCK не включился в этом прогоне (см. load_alignment_onsets()).
# None = либо всё в порядке, либо функция ещё не вызывалась.
ALIGNMENT_ONSET_FAILURE = None


def load_alignment_weights(blocks):
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
    section_segments = _alignment_section_segments(blocks)
    if section_segments is None:
        return None

    section_offsets = load_section_offsets()   # {} без Stage B/section_sync.py -> offset 0.0 для всех, как раньше
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
            ratio = difflib.SequenceMatcher(None, block_text, seg_text).ratio() if (block_text and seg_text) else 0.0
            if ratio < 0.55:
                weights.append(None)
                stale += 1
            else:
                span = _real_speech_span(segs[k], offset)
                weights.append(span if span > 0.05 else None)
        seg_cursor[b["section"]] = k + 1
    if stale:
        print(f"  Alignment: {stale} блок(ов) текстом разошлись с сохранённым таймингом "
              f"(script.txt правили после записи?) — откат на word-count для них")
    return weights


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


def get_media_duration(path):
    r = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json",
                        "-show_format", path], capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    return float(json.loads(r.stdout)["format"]["duration"])


def audio_qc(path, label="Audio QC"):
    """Технический QC дорожки — тот же принцип, что qc_report() для
    картинок: не блокирует, только честно докладывает брак, который иначе
    заметили бы только на слух постфактум (клиппинг, слишком тихо/громко,
    длинные участки мёртвой тишины помимо самих [pause] — забытый обрыв
    записи). Бесплатно — штатные ffmpeg-фильтры astats/silencedetect,
    никакого нового API.

    Вызывается ДВАЖДЫ: на входном голосе (что нам дали) и на ГОТОВОМ
    final.mp4 (что услышит зритель) — см. label. Долгое время проверялся
    только вход, и вся мастер-цепочка (обработка голоса, подложка, дакинг,
    loudnorm, лимитер) уходила к зрителю вообще без технической проверки:
    обе реальные поломки музыки в этом проекте нашлись ручными замерами,
    а не отчётом.
    """
    try:
        r = subprocess.run(
            ["ffmpeg", "-nostats", "-i", path, "-af",
             "astats=reset=0,silencedetect=noise=-40dB:d=1.5", "-f", "null", "-"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            # Тот же класс, что чинил аудит 04.09 у measure_loudnorm_stats():
            # фиксированные 120 с — это полный декод, на часовом эпизоде он
            # не укладывается, и QC молча не выполнялся вообще.
            timeout=max(120, int(_audio_len_for_timeout(path) * 0.5)))
        err = r.stderr
    except Exception as e:
        print(f"  {label}: не удалось проверить ({e})")
        return
    warns = []
    m = re.search(r"Peak level dB:\s*(-?[\d.]+)", err)
    if m and float(m.group(1)) > -0.1:
        warns.append(f"пик {m.group(1)} dBFS — на грани клиппинга")
    m = re.search(r"RMS level dB:\s*(-?[\d.]+)", err)
    rms = float(m.group(1)) if m else None
    if rms is not None and rms < -35:
        warns.append(f"RMS {rms:.1f} dBFS — очень тихо")
    elif rms is not None and rms > -12:
        warns.append(f"RMS {rms:.1f} dBFS — очень громко/пережато")
    silences = [float(x) for x in re.findall(r"silence_duration:\s*([\d.]+)", err)]
    if silences:
        try:
            dur = get_media_duration(path)
            ratio = sum(silences) / dur
            if ratio > 0.15:
                warns.append(f"{ratio*100:.0f}% дорожки — тишина длиннее 1.5с (мёртвый воздух?)")
        except Exception:
            pass
    print(f"{label}: {'; '.join(warns)}" if warns
          else f"{label}: клиппинга/аномальной громкости не найдено")


def build_master_af(loud_stats, fade_out_st, fade_in_sec):
    """Мастер-цепочка финального прохода: loudnorm -> лимитер -> фейды.

    Вынесено из main() отдельной функцией, чтобы порядок ступеней можно было
    проверить тестом, а не только прочитать. Порядок здесь — не вкусовщина:

    * лимитер ПОСЛЕ loudnorm — иначе loudnorm поднял бы уровень уже после
      того, как потолок выставлен, и потолок перестал бы быть потолком;
    * лимитер ПЕРЕД afade — afade только уменьшает уровень, так что на
      результат он не влияет, но обратный порядок заставил бы лимитер
      работать на затухающих хвостах, где он бесполезен;
    * порог лимитера — та же константа LOUDNORM_TARGET_TP, что цель true
      peak у loudnorm: две ступени не должны спорить о потолке.

    loud_stats=None — первый проход не измерился, идём однопроходным
    (динамическим) loudnorm; это уже существующий откат, лимитер в нём
    тем более уместен.
    """
    base = f"loudnorm=I={LOUDNORM_TARGET_I}:TP={LOUDNORM_TARGET_TP}:LRA={LOUDNORM_TARGET_LRA}"
    if loud_stats:
        base += (f":linear=true:measured_I={loud_stats['input_i']}:"
                 f"measured_TP={loud_stats['input_tp']}:"
                 f"measured_LRA={loud_stats['input_lra']}:"
                 f"measured_thresh={loud_stats['input_thresh']}:"
                 f"offset={loud_stats['target_offset']}")
    stages = [base]
    if MASTER_LIMITER_ENABLED:
        stages.append(f"alimiter=limit={LOUDNORM_TARGET_TP}dB:"
                      f"attack={MASTER_LIMITER_ATTACK_MS}:"
                      f"release={MASTER_LIMITER_RELEASE_MS}:level=disabled")
    stages.append(f"afade=t=in:st=0:d={fade_in_sec}")
    stages.append(f"afade=t=out:st={fade_out_st:.3f}:d=2")
    return ",".join(stages)




def configure(video_dir):
    """Папка ролика (в исходнике — sys.argv[1] на импорте). Сбрасывает кэши."""
    global VIDEO_FOLDER, ALIGNMENT_DIR, PAUSE_CUTS_PATH, SECTION_OFFSETS_PATH
    global _PAUSE_CUTS_CACHE, _PAUSE_INSERTS_CACHE, _PAUSE_WINDOWS_CACHE, _SECTION_OFFSETS_CACHE
    global SPEECH_ENDS, ALIGNMENT_ONSET_FAILURE, MUSIC_BED_DECISION
    VIDEO_FOLDER = os.path.abspath(video_dir)
    ALIGNMENT_DIR = os.path.join(VIDEO_FOLDER, "media_plan", "alignment")
    PAUSE_CUTS_PATH = os.path.join(VIDEO_FOLDER, "media_plan", "pause_cuts.json")
    SECTION_OFFSETS_PATH = os.path.join(VIDEO_FOLDER, "media_plan", "section_offsets.json")
    _PAUSE_CUTS_CACHE = _PAUSE_INSERTS_CACHE = _PAUSE_WINDOWS_CACHE = None
    _SECTION_OFFSETS_CACHE = None
    SPEECH_ENDS = []
    ALIGNMENT_ONSET_FAILURE = None
    MUSIC_BED_DECISION = None

MUSIC_BED_DECISION = None
