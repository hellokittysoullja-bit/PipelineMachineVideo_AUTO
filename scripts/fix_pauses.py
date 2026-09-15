#!/usr/bin/env python3
"""Подрезка длинных пауз в озвучке + нормализация громкости (ЧАСТЬ 9, п.7).
Находит тишину длиннее THRESH_SEC и укорачивает её по гладкой кривой
длительности (см. _pause_curve/_keep_sec_for) — короче исходная пауза,
короче держим, длиннее — дольше, без скачка между двумя фиксированными
значениями.
Двухпроходный loudnorm (EBU R128) поверх — гуляющая громкость между
блоками/эпизодами звучит непрофессионально и это бесплатно чинится.
Usage: python scripts/fix_pauses.py <video_dir>
Вход: audio.mp3 (или первый *.mp3). Выход: audio_fixed.flac.

P0-4 (аудит звукового пайплайна): раньше выход был audio_fixed.mp3 — вторая
lossy-перекодировка поверх уже сжатого TTS-исходника, слышимая на
"с"/"ш"/"ч" (та же ошибка, что уже поймана и исправлена для музыки — см.
generate_music_asset.py). FLAC lossless — та же причина, тот же фикс.

Реальный баг, пойманный вживую (жалоба на подписи хука, "убегающие" от
голоса): silencedetect режет ЛЮБУЮ акустическую тишину длиннее THRESH_SEC —
включая естественные вдохи/микропаузы TTS, которых НЕТ в тексте как тега
[pause]/[short pause] (на реальном эпизоде — 41 реальная порезка на 90
тегов паузы в тексте, т.е. большинство тегов даже не доросли до порога, а
часть порезок вообще не привязана ни к одному тегу). Даунстрим-код в
pipeline_smart.py (load_hook_word_timings/load_alignment_weights) раньше
восстанавливал реальную шкалу ПРИБЛИЖЁННО — по границам ТЕГОВ, не по
факту, — что и давало нарастающий рассинхрон. Теперь порезки сохраняются
1:1 сюда же (media_plan/pause_cuts.json), и pipeline_smart.py умеет
пересчитывать alignment.csv В ТОЧНОСТИ на реальную (обрезанную) шкалу
вместо приближения."""
import csv
import hashlib
import json
import os
import re
import subprocess
import sys

THRESH_SEC = 1.0     # тишина длиннее этого — подрезается
NOISE_DB = "-30dB"   # порог тишины
LOUDNORM_TARGET = "I=-16:TP=-1.5:LRA=11"   # стандарт для закадрового голоса

# Пауза как драматургический приём (продакшн-разбор: "1-2 сек тишины после
# сильного факта" — реальный приём режиссёра, не брак). РАНЬШЕ keep был
# ЖЁСТКОЙ СТУПЕНЬКОЙ — ровно ОДНО из двух чисел (KEEP_SEC или
# LONG_HOLD_KEEP_SEC) на КАЖДУЮ паузу всего ролика. На реальном 20-40-
# минутном эпизоде это сотни пауз, и почти все из них бьют РОВНО одну и ту
# же длину бит-в-бит — живая речь так не звучит: диктор не держит паузу
# секундомером, короче исходная заминка — короче держим, длиннее — дольше,
# ГЛАДКО, без скачка на пороге (тот же принцип, что smoothstep уже даёт
# зуму Ken Burns вместо линейного роста — резкий шаг между двумя
# значениями читается механически что в движении камеры, что в паузах
# голоса). KEEP_MIN/KEEP_MAX — теперь границы диапазона кривой
# (_pause_curve), не два дискретных значения.
KEEP_MIN_SEC = 0.42        # у самой границы детекции (сырая тишина ~THRESH_SEC) —
                            # короткая разговорная связка между фразами
KEEP_MAX_SEC = 1.35        # у длинных, явно осознанных пауз (сырая ≥ KEEP_CURVE_TOP_SEC)
KEEP_CURVE_TOP_SEC = 3.0   # сырая тишина такой длины (или длиннее) -> keep = KEEP_MAX_SEC
# Органический разброс ПОВЕРХ кривой — без него две паузы с одинаковой сырой
# длительностью всё равно получали бы ИДЕНТИЧНЫЙ keep (кривая гладкая, но
# детерминированная только по длительности) — в реальной речи два похожих
# вдоха никогда не звучат бит-в-бит одинаково. Детерминированный (по хэшу
# границ самой паузы), не псевдослучайный на каждый прогон — идемпотентно,
# как остальные hash-джиттеры пайплайна (см. kb_hash_choices и т.п.).
PAUSE_JITTER_SEC = 0.05
LONG_HOLD_REPORT_SEC = 1.0   # только для отчёта в консоль — "сколько пауз держались дольше секунды"


def measure_loudness(path):
    """Первый проход loudnorm: только измерение, ничего не меняет в файле."""
    r = subprocess.run(["ffmpeg", "-i", path, "-af",
                        f"loudnorm={LOUDNORM_TARGET}:print_format=json",
                        "-f", "null", "-"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    m = re.search(r'\{[^{}]*"input_i"[^{}]*\}', r.stderr, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def loudnorm_filter(stats):
    """Второй проход: если есть замер с первого — линейная точная нормализация,
    иначе одиночный проход loudnorm (менее точно, но не падает)."""
    if not stats:
        return f"loudnorm={LOUDNORM_TARGET}"
    return (f"loudnorm={LOUDNORM_TARGET}:"
            f"measured_I={stats['input_i']}:measured_TP={stats['input_tp']}:"
            f"measured_LRA={stats['input_lra']}:measured_thresh={stats['input_thresh']}:"
            f"offset={stats.get('target_offset', 0)}:linear=true:print_format=summary")


def find_audio(video_dir):
    for name in ("audio.mp3",):
        p = os.path.join(video_dir, name)
        if os.path.exists(p):
            return p
    if not os.path.isdir(video_dir):
        return None
    mp3s = [f for f in os.listdir(video_dir) if f.lower().endswith(".mp3")
            and "fixed" not in f.lower()]
    return os.path.join(video_dir, mp3s[0]) if mp3s else None


def duration(path):
    r = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries",
                        "format=duration", "-of", "csv=p=0", path],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    return float(r.stdout.strip())


def detect_fine_silences(path):
    """Тишина ЛЮБОЙ длины от FINE_SILENCE_MIN_SEC — отдельный, более мелкий
    проход, чем detect_silences().

    Нужен ровно для одного: измерить, сколько тишины РЕАЛЬНО есть вокруг
    тег-паузы. detect_silences() по устройству не показывает ничего короче
    THRESH_SEC=1.0с, и это не придирка — замер 14.09: на границе секций
    лежало 0.95с (0.243 сам тег плюс 0.705, вставленных склейкой lumean),
    то есть чуть НИЖЕ порога. Код её не видел, считал паузу равной 0.243с и
    добавлял сверху ещё 0.557 — двойной счёт, пойманный замером до рендера.

    Тот же приём, что уже применён в section_sync.py («низкопороговый
    silencedetect по всему audio.mp3»), и та же причина: подрезка и ИЗМЕРЕНИЕ
    отвечают на разные вопросы и не обязаны делить один порог."""
    r = subprocess.run(["ffmpeg", "-i", path, "-af",
                        f"silencedetect=noise={NOISE_DB}:d={FINE_SILENCE_MIN_SEC}",
                        "-f", "null", "-"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    starts = [float(x) for x in re.findall(r'silence_start:\s*([\d.]+)', r.stderr)]
    ends = [float(x) for x in re.findall(r'silence_end:\s*([\d.]+)', r.stderr)]
    return list(zip(starts, ends))


def detect_silences(path, total=None):
    """Пары (начало, конец) тишины длиннее THRESH_SEC.

    total (длина файла) нужен для ХВОСТОВОЙ тишины: если запись кончается
    молчанием, часть сборок ffmpeg не печатает для него закрывающий
    silence_end вообще — тогда zip() просто отбрасывал последнюю пару, и
    хвостовая тишина не подрезалась (мёртвый воздух в конце ролика, который
    этот скрипт как раз и должен убирать). Если закрывающая метка есть,
    длины совпадают и ветка ниже не срабатывает — поведение то же."""
    r = subprocess.run(["ffmpeg", "-i", path, "-af",
                        f"silencedetect=noise={NOISE_DB}:d={THRESH_SEC}",
                        "-f", "null", "-"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    log = r.stderr
    starts = [float(x) for x in re.findall(r'silence_start:\s*([\d.]+)', log)]
    ends = [float(x) for x in re.findall(r'silence_end:\s*([\d.]+)', log)]
    pairs = list(zip(starts, ends))
    if total is not None and len(starts) == len(ends) + 1 and starts[-1] < total:
        pairs.append((starts[-1], total))
    return pairs


def _audio_fingerprint(path):
    """md5 содержимого исходного audio.mp3 — записывается вместе с
    порезками, чтобы pipeline_smart.py мог обнаружить, что audio.mp3
    перезаписали (новая озвучка/правка) БЕЗ повторного запуска
    fix_pauses.py, и не применить молча устаревшие порезки к чужому
    аудио (реальный риск: пользователь перезаписывает эпизод озвучкой,
    забывает про этот шаг — тогда pause_cuts.json тихо соврёт про то,
    где резать)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pause_curve(raw_dur):
    """Гладкая (smoothstep) кривая сырая-длительность -> keep: короче сырая
    тишина — короче держим, длиннее — дольше, БЕЗ скачка на пороге. t=0 у
    самой границы детекции (raw_dur == THRESH_SEC), t=1 у KEEP_CURVE_TOP_SEC
    и длиннее. raw_dur всегда >= THRESH_SEC — так гарантирует сам порог
    silencedetect, вызвавший эту паузу."""
    t = max(0.0, min(1.0, (raw_dur - THRESH_SEC) / (KEEP_CURVE_TOP_SEC - THRESH_SEC)))
    eased = 3 * t * t - 2 * t * t * t
    return KEEP_MIN_SEC + (KEEP_MAX_SEC - KEEP_MIN_SEC) * eased


def _pause_jitter(ss, se):
    """Детерминированный микро-разброс поверх _pause_curve — см. комментарий
    у PAUSE_JITTER_SEC. Хэш от границ самой паузы (не индекса по счёту) —
    стабилен между прогонами при неизменном audio.mp3, даже если раньше по
    треку что-то нашлось/пропало (индекс "N-я пауза по счёту" сдвинулся бы,
    хэш от собственных границ — нет)."""
    h = int(hashlib.md5(f"{ss:.3f}|{se:.3f}".encode()).hexdigest()[:8], 16)
    return (h % 1000) / 1000.0 * 2 * PAUSE_JITTER_SEC - PAUSE_JITTER_SEC


SPEECH_TIMELINE_PATH_NAME = "speech_timeline.json"
PROTECTED_OVERLAP_TOLERANCE = 0.35   # секунд — alignment (Cadence Validator) и
                                       # ffmpeg silencedetect видят одну и ту же
                                       # физическую тишину чуть по-разному
                                       # (порог шума -30dB против точных
                                       # посимвольных таймкодов), окна нужно
                                       # сопоставлять с запасом, не 1-в-1


SECTION_OFFSETS_PATH_NAME = "section_offsets.json"


def load_section_offsets(video_dir):
    """{section_name: global_start_offset_sec} из media_plan/section_offsets.json
    (пишет scripts/speech_generate.py Stage B при сборке — единственное
    место, которое РЕАЛЬНО знает, где начинается аудио каждой секции в
    итоговом audio.mp3, потому что само его туда конкатенировало). Нет
    файла (ручной Шаг 6, вариант А — секции склеены человеком, карты
    смещений ни от кого не осталось) -> {} — см. load_protected_windows()
    про честное ограничение без этой карты."""
    path = os.path.join(video_dir, "media_plan", SECTION_OFFSETS_PATH_NAME)
    if not os.path.exists(path):
        return {}
    try:
        data = json.load(open(path, encoding="utf-8"))
        return {str(k): float(v) for k, v in data.items()}
    except Exception as e:
        print(f"  ВНИМАНИЕ: {path} битый/нечитаемый, смещения секций игнорирую: {e}")
        return {}


def load_protected_windows(video_dir):
    """[(raw_start, raw_end, target_kept_sec, unit_id), ...] из
    media_plan/speech_timeline.json (см. scripts/speech_validator.py) —
    паузы, которые Speech Planner поставил ОСОЗНАННО (reveal_hold перед
    [climax], evidence_beat после [stat:...], закрывающий hold финальной
    мысли и т.д.) и Cadence Validator подтвердил реальным alignment.
    Нет файла (эпизод без Speech Director — большинство существующих
    эпизодов) -> [] (тихий откат: вся логика ниже отрабатывает РОВНО как
    раньше, ни одна пауза не считается protected, тот же принцип
    безопасного отката, что и у остальных опциональных источников данных
    пайплайна).

    РЕАЛЬНЫЙ БАГ, найден эмпирической проверкой (не гипотетический):
    время в speech_timeline.json ЛОКАЛЬНОЕ для каждой секции (своя
    нулевая точка на файл alignment.csv — та же конвенция, что и у
    ручного Шага 6), а silencedetect ниже в этом же скрипте меряет ВЕСЬ
    audio.mp3 ОДНИМ глобальным проходом. Без пересчёта в глобальное время
    protected-окно физически совпадает со своей silencedetect-паузой
    ТОЛЬКО у первой секции эпизода (её локальный ноль случайно совпадает
    с глобальным) — у всех следующих секций _match_protected() молча не
    находил совпадения, и защита пауз тихо переставала работать после
    первой секции у КАЖДОГО эпизода. Чинится смещением из
    load_section_offsets() (см. unit_id -> секция через 'СЕКЦИЯ#N',
    та же нумерация, что speech_planner.build_units())."""
    path = os.path.join(video_dir, "media_plan", SPEECH_TIMELINE_PATH_NAME)
    if not os.path.exists(path):
        return []
    offsets = load_section_offsets(video_dir)
    try:
        data = json.load(open(path, encoding="utf-8"))
        raw_windows = data.get("protected_windows", [])
    except Exception as e:
        print(f"  ВНИМАНИЕ: {path} битый/нечитаемый, protected-паузы игнорирую: {e}")
        return []
    if raw_windows and not offsets:
        print(f"  ВНИМАНИЕ: {SECTION_OFFSETS_PATH_NAME} не найден — время в "
              f"{SPEECH_TIMELINE_PATH_NAME} локальное для каждой секции, а silencedetect "
              f"здесь меряет весь audio.mp3 одним проходом. Без карты смещений "
              f"protected-паузы физически совпадут ТОЛЬКО в первой секции эпизода "
              f"(Stage B — scripts/speech_generate.py — пишет карту сам; при ручном "
              f"Шаге 6, вариант А, карты пока нет).")
    out = []
    for w in raw_windows:
        raw_start, raw_end, kept, unit_id = float(w[0]), float(w[1]), float(w[2]), str(w[3])
        section = unit_id.rsplit("#", 1)[0]
        offset = offsets.get(section, 0.0)
        out.append((raw_start + offset, raw_end + offset, kept, unit_id))
    return out


def _match_protected(ss, se, protected_windows):
    """Находит protected-окно, физически совпадающее с обнаруженной
    silencedetect-паузой (ss, se) — по перекрытию интервалов с допуском
    PROTECTED_OVERLAP_TOLERANCE, а не точным равенством границ (alignment
    и silencedetect двумя разными методами меряют одну и ту же тишину,
    границы почти никогда не совпадают бит-в-бит). Первое совпадение по
    порядку (список уже хронологический — юниты плана идут по тексту
    сценария) — соседние protected-окна на практике не перекрываются
    между собой, второго кандидата в реальном сценарии не бывает."""
    for w_start, w_end, kept, unit_id in protected_windows:
        if ss < w_end + PROTECTED_OVERLAP_TOLERANCE and se > w_start - PROTECTED_OVERLAP_TOLERANCE:
            return kept, unit_id
    return None, None


def _keep_sec_for(ss, se, protected_windows=None):
    """Сколько секунд тишины (ss, se) реально оставляем.

    Если (ss, se) физически совпадает с protected-окном из
    speech_timeline.json (Speech Director распорядился этой конкретной
    паузой осознанно) — берём ТОЧНУЮ цель плана, БЕЗ гладкой кривой и БЕЗ
    джиттера: оба существуют именно для пауз, у которых НЕТ режиссёрского
    решения, и наложение джиттера поверх точно спланированной длины не
    "оживляет" её, а портит точность (см. ЧАСТЬ протокола Speech Director).

    Иначе — прежнее поведение: гладкая кривая от сырой длительности
    (_pause_curve) плюс небольшой детерминированный джиттер (_pause_jitter),
    safety net для ВСЕХ пауз, которые Speech Planner не запланировал (не
    только "эпизод без Speech Director вообще", но и обычные connective-
    паузы внутри эпизода С планом — тег есть, роль нейтральная, тут план
    целиком доверяет прежней кривой).

    ЕДИНСТВЕННОЕ место, где считается keep — и main() (реальная обрезка
    atrim), и save_cuts() (что записать как вырезанное) обязаны звать
    именно эту функцию с ОДНИМ И ТЕМ ЖЕ protected_windows: иначе
    pause_cuts.json/speech_timeline.json тихо разойдутся по тому, что
    реально вырезано — и вся синхронизация подписей хука снова разъедется,
    только уже на новых данных."""
    raw_dur = se - ss
    if protected_windows:
        target, _unit_id = _match_protected(ss, se, protected_windows)
        if target is not None:
            return max(0.15, min(raw_dur, target))
    keep = _pause_curve(raw_dur) + _pause_jitter(ss, se)
    return max(0.15, min(raw_dur, keep))


# Доведение тег-паузы до её ДОКУМЕНТИРОВАННОЙ длины (ЧАСТЬ 10 CLAUDE.md).
#
# Почему это нужно системно, а не как заплатка под один эпизод. Замер по
# alignment реального оплаченного заказа (14.09) показал, что движок держит
# длину тега как придётся:
#     [pause]        в середине текста   1.027с   (документировано 0.8)
#     [pause]        в конце заказа      0.243с
#     [short pause]  в середине текста   0.169с   (документировано 0.4)
# То есть расхождение до шести раз, и в ОБЕ стороны. Весь звуковой монтаж
# при этом считает тишину по факту: звук перехода главы и объектный кюй
# живут ТОЛЬКО в реальной паузе (см. sfx_plan.py), и в 0.169с не влезает ни
# один ассет — кюй честно отбрасывается с причиной no_silence_for_object.
#
# Правка убирает зависимость от добросовестности движка целиком: тишину
# задаёт ПЛАН, речь — движок. Она ОДНОСТОРОННЯЯ по построению — только
# доводит короткую паузу до её же документированной длины и никогда не
# укорачивает: подрезка живёт выше THRESH_SEC=1.0с, оба целевых значения
# (0.8 и 0.4) ниже, поэтому две логики физически не пересекаются.
# Формат вставляемой тишины. Считывается у исходника в main() — жёстко
# зашитые 44100/mono дали бы несовпадение с потоком и падение concat.
# Порог ИЗМЕРЕНИЯ тишины (не подрезки): достаточно мелкий, чтобы увидеть
# паузу в 0.17с, и достаточно крупный, чтобы не считать паузой смычку между
# словами.
FINE_SILENCE_MIN_SEC = 0.08

SILENCE_RATE = 48000
SILENCE_LAYOUT = "mono"

TAG_PAUSE_TARGETS = {"[pause]": 0.8, "[short pause]": 0.4}
TAG_PAUSE_TOLERANCE = 0.05     # ближе этого к цели — не трогаем вообще
_ALIGNMENT_TAG_RE = re.compile(r'\[(?:short\s+)?pause\]', re.I)


def planned_tag_pauses(video_dir):
    """[(сырой_старт, сырой_конец, цель_сек, тег), ...] по alignment секций.

    Источник истины — посимвольный alignment (где тег РЕАЛЬНО звучал) плюс
    section_offsets.json (глобальное смещение секции). Нет alignment или нет
    карты смещений для секции — эта секция просто не попадает в список:
    угадывать положение паузы нельзя, а молча промахнуться мимо неё хуже,
    чем не трогать."""
    plan_dir = os.path.join(video_dir, "media_plan")
    align_dir = os.path.join(plan_dir, "alignment")
    if not os.path.isdir(align_dir):
        return []
    try:
        with open(os.path.join(plan_dir, "section_offsets.json"), encoding="utf-8") as f:
            offsets = [float(v) for v in json.load(f).values()]
    except Exception:
        return []
    out = []
    for idx, off in enumerate(sorted(offsets)):
        path = os.path.join(align_dir, f"{idx:02d}.csv")
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            chars = [(r["char"], float(r["start"]), float(r["end"])) for r in rows]
        except Exception:
            continue
        text = "".join(c for c, _s, _e in chars)
        for m in _ALIGNMENT_TAG_RE.finditer(text):
            seg = chars[m.start():m.end()]
            if not seg:
                continue
            tag = "[short pause]" if "short" in m.group(0).lower() else "[pause]"
            out.append((seg[0][1] + off, seg[-1][2] + off, TAG_PAUSE_TARGETS[tag], tag))
    return sorted(out)


def real_silence_at(ps, pe, sil, protected_windows=None, fine=None):
    """Сколько тишины РЕАЛЬНО останется в готовом файле вокруг тег-паузы.

    Считать по длине самого тега в alignment нельзя, и это не теория:
    на границе секций lumean_tts.py уже вставляет свою паузу при склейке
    (SECTION_GAP_TARGET_SEC), а тег при этом остаётся коротким. Первая
    версия этой функции сравнивала цель с длиной ТЕГА и добавила 0.557с
    поверх уже вставленных 0.705с — двойной счёт, пойманный замером до
    рендера. Поэтому берём объединение тега с реально найденными тишинами
    вокруг него и вычитаем то, что подрезка из этого куска уносит."""
    s0, s1 = ps, pe
    # Растём по МЕЛКИМ тишинам, пока есть что присоединить: тег и
    # вставленная склейкой пауза стоят встык, а не внахлёст, и одного
    # прохода объединения не хватило бы.
    for _ in range(8):
        grew = False
        for ss, se in (fine if fine is not None else (sil or [])):
            if se > s0 - 0.05 and ss < s1 + 0.05 and (ss < s0 or se > s1):
                s0, s1 = min(s0, ss), max(s1, se)
                grew = True
        if not grew:
            break
    kept = s1 - s0
    for ss, se in sil or []:
        if se <= s0 or ss >= s1:
            continue
        keep = _keep_sec_for(ss, se, protected_windows)
        removed_start = min(se, ss + keep)
        kept -= max(0.0, min(s1, se) - max(s0, removed_start))
    return max(0.0, kept)


def apply_tag_pause_targets(segments, planned, sil=None, protected_windows=None, fine=None):
    """Довести короткие тег-паузы до цели, вставив тишину в разрез сегмента.

    Возвращает (новые_сегменты, вставки), где сегмент — это либо
    ("copy", a, b) кусок исходника, либо ("silence", секунды, сырая_позиция).
    Вставки — [(сырая_позиция, секунды), ...] для pause_cuts.json."""
    segs = [("copy", a, b) for a, b in segments]
    inserts = []
    for ps, pe, target, _tag in planned:
        have = real_silence_at(ps, pe, sil, protected_windows, fine)
        need = target - have
        if need <= TAG_PAUSE_TOLERANCE:
            continue
        mid = (ps + pe) / 2.0
        for i, item in enumerate(segs):
            if item[0] != "copy":
                continue
            _k, a, b = item
            if not (a < mid < b):
                continue
            segs[i:i + 1] = [("copy", a, mid), ("silence", need, mid), ("copy", mid, b)]
            inserts.append((round(mid, 6), round(need, 6)))
            break
    return segs, sorted(inserts)


def save_cuts(video_dir, sil, src, out, protected_windows=None, pause_inserts=None):
    """Сохраняет РЕАЛЬНО вырезанные интервалы (сырое время audio.mp3) —
    только ту часть каждой тишины, что реально ушла (see _keep_sec_for —
    protected-паузы, hold-паузы и короткие тишины теряют разную долю) +
    отпечаток исходного audio.mp3 (см. _audio_fingerprint). pipeline_smart.py
    читает этот файл, чтобы ТОЧНО (не приближённо по тегам) пересчитать
    alignment.csv на реальную обрезанную шкалу — см. raw_to_real_time().
    protected_windows — тот же список из load_protected_windows(), что и
    в main(): ОБЯЗАН быть тем же самым, иначе pause_cuts.json запишет keep,
    отличный от того, что реально вырезал atrim в main().

    fixed_audio_md5 (P0-1 форензик-аудита) — отпечаток УЖЕ ГОТОВОГО out
    (audio_fixed.flac), не только исходного src. Раньше pipeline_smart.py
    проверял ТОЛЬКО source_audio_md5 (что audio.mp3 не переозвучили молча) —
    но не то, что сам audio_fixed.flac реально соответствует ЭТОМУ прогону
    fix_pauses.py, а не остался от предыдущего (перезаписали озвучку,
    забыли перезапустить fix_pauses.py — старый FLAC со старым голосом
    тихо подставлялся бы под новые alignment/субтитры/тайминг). out может
    ещё не существовать (вызов ДО ffmpeg в веток "нечего резать" — тогда
    дважды не считаем: этот случай сейчас всегда вызывается ПОСЛЕ успешного
    ffmpeg, см. main())."""
    # P2-9 (аудит звукового пайплайна): раньше округляли до 3 знаков (1мс) —
    # совпадало с :.3f в atrim/afade ниже, но обе точности были ГРУБЕЕ
    # семпла (при 48000Hz 1мс = 48 семплов). Подняли до 6 знаков (мкс) в
    # ОБОИХ местах разом (см. main() ниже) — записанное в pause_cuts.json
    # снова точно совпадает с тем, что реально режет ffmpeg, просто на
    # семпл-уровне, а не мс-уровне.
    cuts = [[round(min(se, ss + _keep_sec_for(ss, se, protected_windows)), 6), round(se, 6)]
            for ss, se in sil if se - min(se, ss + _keep_sec_for(ss, se, protected_windows)) > 0.001]
    # P1-15 (аудит звукового пайплайна): отдельно от cuts (вырезанное) —
    # СОХРАНЁННЫЕ окна тишины [сырой_старт, сколько_оставлено], нужны
    # pipeline_smart.py, чтобы дать подложке лёгкий "вздох" именно на
    # паузах, реально удержанных надолго (плановых protected или кривой
    # _pause_curve — см. PAUSE_SWELL_MIN_KEEP_SEC в pipeline_smart.py,
    # фильтрует по keep, а не по сырой длительности), не на каждом обычном
    # вдохе TTS. Отдельный ключ, а не третий элемент
    # в cuts — raw_to_real_time() распаковывает cuts строго как (a, b) пары
    # по всему файлу, менять эту форму не нужно ради нового потребителя.
    pause_windows = [[round(ss, 6), round(_keep_sec_for(ss, se, protected_windows), 6)] for ss, se in sil]
    plan_dir = os.path.join(video_dir, "media_plan")
    os.makedirs(plan_dir, exist_ok=True)
    with open(os.path.join(plan_dir, "pause_cuts.json"), "w", encoding="utf-8") as f:
        json.dump({"source_audio_md5": _audio_fingerprint(src),
                    "fixed_audio_md5": _audio_fingerprint(out) if os.path.exists(out) else None,
                    "cuts": cuts, "pause_windows": pause_windows,
                    "pause_inserts": [[round(p, 6), round(sec, 6)]
                                      for p, sec in (pause_inserts or [])]}, f)


def main():
    video_dir = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    src = find_audio(video_dir)
    if not src:
        print("Аудио не найдено (audio.mp3)")
        return 1
    out = os.path.join(video_dir, "audio_fixed.flac")
    total = duration(src)
    sil = detect_silences(src, total=total)
    loud = loudnorm_filter(measure_loudness(src))
    protected_windows = load_protected_windows(video_dir)
    if protected_windows:
        print(f"  Speech Director: {len(protected_windows)} запланированных пауз защищено "
              f"от гладкой кривой/джиттера (media_plan/speech_timeline.json)")
    if not sil:
        print("Длинных пауз не найдено — нормализую громкость.")
        r = subprocess.run(["ffmpeg", "-y", "-i", src, "-af", loud,
                            "-c:a", "flac", out],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0 or not os.path.exists(out):
            print("Ошибка ffmpeg:", r.stderr[-400:])
            return 1
        # save_cuts ПОСЛЕ успешного ffmpeg (не до) — fixed_audio_md5 обязан
        # быть отпечатком уже ГОТОВОГО out, иначе всегда записывался бы None.
        save_cuts(video_dir, [], src, out, protected_windows)
        print(f"Готово: {out}")
        return 0

    # Строим сегменты: речь целиком + каждая тишина обрезана по гладкой
    # кривой длительности (см. _keep_sec_for/_pause_curve)
    segments = []
    prev = 0.0
    long_holds = 0
    protected_used = 0
    for ss, se in sil:
        target, unit_id = _match_protected(ss, se, protected_windows) if protected_windows else (None, None)
        keep = _keep_sec_for(ss, se, protected_windows)
        if unit_id is not None:
            protected_used += 1
        if keep >= LONG_HOLD_REPORT_SEC:
            long_holds += 1
        if ss > prev:
            segments.append((prev, ss))
        segments.append((ss, min(se, ss + keep)))
        prev = se
    if prev < total:
        segments.append((prev, total))

    # Аудит звукового пайплайна (P0-1): concat встык на стыке двух кусков с
    # ненулевым уровнем сигнала даёт слышимый щелчок/ступеньку волны — на
    # речи это "цок". Короткий (8мс) fade-in/fade-out на КАЖДОМ сегменте
    # перед concat убирает разрыв амплитуды на стыке, не трогая тайминг
    # (fade — это огибающая внутри уже вырезанных границ, не сдвиг границ) —
    # безопасно для pause_cuts.json/raw_to_real_time, которые считаются по
    # границам сегментов, не по амплитуде.
    # Тег-паузы доводятся до документированной длины ПОСЛЕ построения
    # сегментов: подрезка уже отработала, а две логики не пересекаются по
    # построению (см. TAG_PAUSE_TARGETS).
    global SILENCE_RATE, SILENCE_LAYOUT
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0",
                            "-show_entries", "stream=sample_rate,channels",
                            "-of", "csv=p=0", src],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace").stdout.strip()
    if probe:
        bits = (probe.split(",") + ["", ""])[:2]
        if bits[0].strip().isdigit():
            SILENCE_RATE = int(bits[0].strip())
        SILENCE_LAYOUT = "stereo" if bits[1].strip() == "2" else "mono"
    planned_pauses = planned_tag_pauses(video_dir)
    segments, pause_inserts = apply_tag_pause_targets(
        segments, planned_pauses, sil, protected_windows, detect_fine_silences(src))
    if pause_inserts:
        print(f"  Тег-паузы доведены до документированной длины: {len(pause_inserts)} "
              f"(+{sum(sec for _p, sec in pause_inserts):.2f}с) — движок отдал их короче")

    SPLICE_FADE_SEC = 0.008
    parts, filt = [], ""
    for i, item in enumerate(segments):
        if item[0] == "silence":
            _k, sec, _pos = item
            if sec <= 0.001:
                continue
            # Тишина В ТОМ ЖЕ формате, что и куски исходника — иначе concat
            # упрётся в несовпадение частоты/каналов.
            filt += (f"anullsrc=r={SILENCE_RATE}:cl={SILENCE_LAYOUT},"
                     f"atrim=end={sec:.6f},asetpts=PTS-STARTPTS[a{i}];")
            parts.append(f"[a{i}]")
            continue
        _k, a, b = item
        dur = b - a
        if dur <= 0.02:
            continue
        fade = min(SPLICE_FADE_SEC, dur / 3)
        fade_out_start = max(0.0, dur - fade)
        filt += (f"[0:a]atrim=start={a:.6f}:end={b:.6f},asetpts=PTS-STARTPTS,"
                 f"afade=t=in:d={fade:.6f},afade=t=out:st={fade_out_start:.6f}:d={fade:.6f}[a{i}];")
        parts.append(f"[a{i}]")
    if not parts:
        # все сегменты оказались короче 0.02с — склеивать нечего, ffmpeg бы
        # упал на concat=n=0; отдаём исходник без изменений (кроме громкости)
        print("Нечего склеивать — нормализую громкость исходника.")
        r = subprocess.run(["ffmpeg", "-y", "-i", src, "-af", loud,
                            "-c:a", "flac", out],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0 or not os.path.exists(out):
            print("Ошибка ffmpeg:", r.stderr[-400:])
            return 1
        save_cuts(video_dir, [], src, out, protected_windows)
        print(f"Готово: {out}")
        return 0
    filt += "".join(parts) + f"concat=n={len(parts)}:v=0:a=1[c];[c]{loud}[out]"

    # Граф фильтров уходит в ФАЙЛ (-filter_complex_script), а не в аргумент
    # командной строки. РЕАЛЬНОЕ ограничение платформы, не стилистика: на
    # каждую найденную паузу здесь два сегмента по ~150 символов, и на
    # 40-60-минутном эпизоде с сотнями пауз (ЧАСТЬ 9 CLAUDE.md прямо
    # предполагает до ~90 тегов паузы плюс естественные вдохи TTS — на
    # реальном эпизоде их было 41 при 90 тегах) граф разрастается до
    # десятков тысяч символов. Лимит командной строки Windows (основная
    # платформа по CLAUDE.md ЧАСТЬ 3) — 32767 символов на ВСЮ строку: за
    # ним ffmpeg просто не запускается (WinError 206), и шаг подрезки пауз
    # отваливается целиком именно на длинных роликах, ради которых он и
    # нужен. Linux тоже не безграничен (MAX_ARG_STRLEN 128КБ на аргумент).
    filt_path = os.path.join(video_dir, "_fix_pauses_filter.txt")
    with open(filt_path, "w", encoding="utf-8") as f:
        f.write(filt)
    cmd = ["ffmpeg", "-y", "-i", src, "-filter_complex_script", filt_path,
           "-map", "[out]", "-c:a", "flac", out]
    # Файл графа НЕ удаляем: он маленький (десятки КБ), а при разборе
    # ошибки ffmpeg на длинной цепочке это единственный способ увидеть, что
    # именно ушло в фильтр.
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0 or not os.path.exists(out):
        print("Ошибка ffmpeg:", r.stderr[-400:])
        return 1
    save_cuts(video_dir, sil, src, out, protected_windows, pause_inserts)
    protected_note = f", из них по плану Speech Director: {protected_used}" if protected_windows else ""
    print(f"Готово: {out} | подрезано пауз: {len(sil)} (из них длинных hold-пауз: {long_holds}"
          f"{protected_note}) | было {total:.1f}с → стало {duration(out):.1f}с")
    return 0


if __name__ == "__main__":
    sys.exit(main())
