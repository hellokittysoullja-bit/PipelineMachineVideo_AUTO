#!/usr/bin/env python3
"""Главная сборка по паузам (ЧАСТЬ 6, 9, 13-шаг7).
Разбивает script.txt по [pause] на смысловые блоки, считает тайминг
(слова+паузы, масштаб под реальную длину аудио), Ken Burns на всю длину
кадра, чередование in/out. Медиа: локальная папка media/ по порядку,
fallback — Pexels по тематическому запросу.
Usage: python scripts/pipeline_smart.py <video_dir>"""
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    import numpy as np
except ImportError:
    np = None   # аудио-ритм по громкости — опциональная фича, без numpy просто выключена

from PIL import Image as PILImage   # Pillow уже обязательная зависимость (requirements.txt)

try:
    import cv2
    PARALLAX_LIBS = True
except ImportError:
    PARALLAX_LIBS = False   # 2.5D-параллакс — опциональная фича (torch/transformers/opencv тяжёлые)

PARALLAX_ENABLED = os.environ.get("PARALLAX", "1") != "0" and PARALLAX_LIBS
PARALLAX_BROKEN = False   # взводится только на системном сбое (модель/сеть), не на одной плохой картинке

VIDEO_FOLDER = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
SCRIPT_FILE = os.path.join(VIDEO_FOLDER, "script.txt")
MEDIA_FOLDER = os.path.join(VIDEO_FOLDER, "media")
OUTPUT_FILE = os.path.join(VIDEO_FOLDER, "final.mp4")
TEMP_FOLDER = os.path.join(VIDEO_FOLDER, "temp_smart")

PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")
FPS, WIDTH, HEIGHT = 25, 1920, 1080
# Каждый клип рендерится независимым процессом ffmpeg, и все решения (зум,
# пан, переход, выбор медиа) принимаются ДО запуска рендера — общего
# изменяемого состояния между клипами нет. Значит параллельный запуск N из
# них не меняет результат, только время сборки. -threads на каждый процесс
# клэмпится так, чтобы WORKERS процессов суммарно не забирали больше ядер,
# чем есть физически — без клэмпа N параллельных full-core энкодов душат
# друг друга и выигрыша по времени почти нет.
def resolve_workers(env_value, cpu_count):
    if env_value:
        try:
            v = int(env_value)
            if v > 0:
                return v
        except ValueError:
            pass
    return max(1, (cpu_count or 4) - 1)


def resolve_ffmpeg_threads(workers, cpu_count):
    return max(1, (cpu_count or 4) // max(1, workers))


WORKERS = resolve_workers(os.environ.get("PIPELINE_WORKERS", ""), os.cpu_count())
FFMPEG_THREADS = resolve_ffmpeg_threads(WORKERS, os.cpu_count())

# CRF интермедиатов поднят с 23: клип кодируется раз, потом ЕЩЁ РАЗ поверх
# кодируется вся цепочка xfade — то есть в финал уходит второе поколение
# потерь. 17 — стандартный порог "визуально без потерь" для x264, цена —
# только временное место на диске в temp_smart/. Финальный проход
# (xfade_chain, единственный полноразмерный энкод на весь ролик, а не N раз,
# как per-clip) может себе позволить -preset slow.
CLIP_CRF = "17"
FINAL_CRF = "18"
FINAL_PRESET = "slow"
# ZOOM_FLOOR — минимальный зум держится ВЕСЬ клип (не 1.0). Раньше offset пана
# был обязан = 0 ровно в момент zoom=1.0 (иначе край вылезет за картинку), и на
# каждом втором клипе (zoom-out) кадр половину времени стоял мёртвым по центру.
# С постоянным полом запас под пан есть на любом кадре клипа, а не только к
# концу движения зума.
ZOOM_FLOOR = 1.04
# Скорость зума в долях/сек, а не фиксированным приростом на клип — иначе
# 4-секундный и 20-секундный кадр "дышат" с заметно разной скоростью.
ZOOM_RATE_BASE = 0.010
ZOOM_DELTA_MIN, ZOOM_DELTA_MAX = 0.05, 0.22
MIN_CLIP, MAX_CLIP = 4.0, 20.0
# Панорамирование считается от РЕАЛЬНОГО zoom в каждый момент кадра — (1-1/zoom)/2
# это точный геометрический запас смещения без вылета за картинку, безопасно по
# построению на любом кадре, поэтому PAN_SAFETY можно брать ближе к пределу, чем
# раньше (когда запас считался один раз заранее от конечного zoom клипа).
PAN_SAFETY = 0.9
PAN_JITTER_MIN, PAN_JITTER_MAX = 0.6, 1.0   # органический разброс силы пана по клипам
PAN_DIRECTIONS = [(0, 0), (1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1)]
# Один и тот же грейд на все 200+ кадров — тоже штамп, если приглядеться.
# Лёгкий hash-джиттер контраста/сатурации/яркости на каждый клип убирает
# эту одинаковость, оставаясь в узком диапазоне (не превращается в разнобой).
# Базовый грейд — не нейтральный "как есть у стока": сатурация чуть прибрана,
# тени уводим в холод, света чуть тёплые (лёгкий teal-orange) — читается как
# военно-историческая документалка, а не разноцветный слайд-шоу из Pexels.
# MOOD_GRADE — тот же язык грейда, но с сдвигом под секцию: хук резче и
# холоднее (цепляет), тело — базовая "стальная" документалка, финал теплее
# и мягче (спад напряжения). LUT-переключение по секции без внешних .cube
# файлов — тот же eq/colorbalance, просто другие опорные точки.
MOOD_GRADE = {
    "HOOK":  {"c0": 1.10, "s0": 0.82, "vign": "PI/4.2", "bs": 0.13, "rh": 0.02},
    "FINAL": {"c0": 1.02, "s0": 0.90, "vign": "PI/6",   "bs": 0.06, "rh": 0.08},
    "BODY":  {"c0": 1.06, "s0": 0.86, "vign": "PI/5",   "bs": 0.10, "rh": 0.03},
}


def film_look(photo_hash, section=""):
    mood = MOOD_GRADE["HOOK"] if section.startswith("HOOK") else (
           MOOD_GRADE["FINAL"] if section.startswith("FINAL") else MOOD_GRADE["BODY"])
    c = mood["c0"] + (photo_hash % 100) / 100 * 0.05
    s = mood["s0"] + ((photo_hash >> 7) % 100) / 100 * 0.08
    b = ((photo_hash >> 14) % 100) / 100 * 0.02
    return (f"eq=contrast={c:.3f}:saturation={s:.3f}:brightness={b:.3f},"
            f"colorbalance=rs=-0.06:bs={mood['bs']:.3f}:rm=-0.02:bm=0.04:rh={mood['rh']:.3f}:bh=-0.02,"
            f"vignette={mood['vign']},noise=alls=2:allf=t+u")


XFADE_DUR = 0.4        # диссолв на границах секций и часть обычных склеек
XFADE_DUR_HARD = 0.06  # почти мгновенный переход — читается как жёсткий cut
# hblur (смаз в движении — читается как whip pan) и zoomin (панч-переход)
# добавлены в пул обычных склеек; fadewhite — вспышка светом на границах
# разделов вперемешку с dissolve/fadeblack, без кастомных текстур-ассетов.
XFADE_TRANSITIONS = ["fade", "dissolve", "smoothleft", "smoothright",
                      "smoothup", "smoothdown", "hblur", "zoomin"]
BOUNDARY_TRANSITIONS = ["dissolve", "fadeblack", "fadewhite"]
HOOK_MAX_CLIP = 5.0     # в хуке кадры короче и чаще — критично для удержания первых секунд
PAUSE_DURATIONS = {"[pause]": 0.8, "[short pause]": 0.4,
                   "[slowly]": 0.0, "[emphasis]": 0.0, "[energetic]": 0.0}

# Russo One — фирменный "рубленый" дисплейный шрифт (CHANNEL.md house
# style), не системный DejaVu. OFL, бесплатно (Google Fonts / google/fonts
# на GitHub). Anton (первый выбор) не подошёл — в нём НЕТ кириллицы вообще
# (проверено вживую: буквы рисуются квадратами-тофу). Russo One — с родной
# кириллицей от российской студии, того же плакатного характера.
# Держим DejaVu запасным на случай отсутствия assets/fonts/.
FONT_CANDIDATES = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "fonts", "RussoOne-Regular.ttf"),
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
]
FONT_PATH = next((p for p in FONT_CANDIDATES if os.path.exists(p)), None)
FONT_IS_DISPLAY = bool(FONT_PATH) and "RussoOne" in FONT_PATH   # дисплейный шрифт уже капслочный и очень широкий по кеглю


def section_title(name):
    """BLOCK N: Название -> 'Название'. HOOK/FINAL/безымянные BLOCK — без титра."""
    m = re.match(r'BLOCK\s+\d+\s*:\s*(.+)', name, re.I)
    return m.group(1).strip() if m else None


def escape_drawtext(s):
    # ВАЖНО: % НЕ удваивается. Проверено вживую на ffmpeg 6.1: при дефолтном
    # expansion=normal текст с процентом в ЛЮБОМ виде ("%", "%%", "\\%") даёт
    # "Stray %" и drawtext не рисует ВООБЩЕ НИЧЕГО — при этом код возврата 0,
    # то есть откат на версию без титра/плашки не срабатывает и плашка просто
    # молча исчезает с экрана. Лечится только expansion=none на самом фильтре
    # (см. DRAWTEXT_OPTS), и тогда % надо оставлять как есть — иначе на экран
    # выйдет буквальное "40%%".
    return (s.replace("\\", "\\\\").replace(":", "\\:")
             .replace("'", "\u2019"))


# expansion=none — выключает %{...}/strftime-подстановки в drawtext. Нам они не
# нужны ни в титрах, ни в плашках, а без них любой % в тексте убивает весь
# слой целиком (см. escape_drawtext).
DRAWTEXT_OPTS = "expansion=none:"


def pick_no_repeat(history, candidate, options, max_repeat):
    """\u0425\u044d\u0448 \u0434\u0430\u0451\u0442 \u0440\u0430\u0437\u043d\u043e\u043e\u0431\u0440\u0430\u0437\u0438\u0435 "\u0432 \u0441\u0440\u0435\u0434\u043d\u0435\u043c", \u043d\u043e \u043d\u0435 \u043c\u0435\u0448\u0430\u0435\u0442 3-4 \u043e\u0434\u0438\u043d\u0430\u043a\u043e\u0432\u044b\u043c \u043f\u043e\u0434\u0440\u044f\u0434
    \u0441\u043b\u0443\u0447\u0430\u0439\u043d\u044b\u043c \u0441\u043e\u0432\u043f\u0430\u0434\u0435\u043d\u0438\u0435\u043c. \u0415\u0441\u043b\u0438 \u043f\u043e\u0441\u043b\u0435\u0434\u043d\u0438\u0435 max_repeat \u0440\u0435\u0448\u0435\u043d\u0438\u0439 \u0441\u043e\u0432\u043f\u0430\u0434\u0430\u044e\u0442 \u0441
    \u043a\u0430\u043d\u0434\u0438\u0434\u0430\u0442\u043e\u043c \u2014 \u0434\u0435\u0442\u0435\u0440\u043c\u0438\u043d\u0438\u0440\u043e\u0432\u0430\u043d\u043d\u043e \u0431\u0435\u0440\u0451\u043c \u0441\u043b\u0435\u0434\u0443\u044e\u0449\u0438\u0439 \u0432\u0430\u0440\u0438\u0430\u043d\u0442 \u043f\u043e \u043a\u0440\u0443\u0433\u0443 \u0432\u043c\u0435\u0441\u0442\u043e
    \u043d\u0435\u0433\u043e. history \u043c\u0443\u0442\u0438\u0440\u0443\u0435\u0442\u0441\u044f \u043d\u0430 \u043c\u0435\u0441\u0442\u0435 (\u0441\u043f\u0438\u0441\u043e\u043a \u043f\u043e\u0441\u043b\u0435\u0434\u043d\u0438\u0445 \u0440\u0435\u0448\u0435\u043d\u0438\u0439)."""
    if len(history) >= max_repeat and all(x == candidate for x in history[-max_repeat:]):
        idx = options.index(candidate) if candidate in options else 0
        candidate = options[(idx + 1) % len(options)]
    history.append(candidate)
    del history[:-(max_repeat + 2)]
    return candidate


def audio_energy_curve(audio_path, window_sec=1.0):
    """RMS-\u0433\u0440\u043e\u043c\u043a\u043e\u0441\u0442\u044c \u0430\u0443\u0434\u0438\u043e \u043f\u043e \u043e\u043a\u043d\u0430\u043c \u2014 \u0431\u0435\u0437 Whisper/librosa, \u0447\u0438\u0441\u0442\u044b\u0439 PCM+numpy.
    \u0412\u043e\u0437\u0432\u0440\u0430\u0449\u0430\u0435\u0442 (rms_array, window_sec) \u0438\u043b\u0438 None, \u0435\u0441\u043b\u0438 numpy \u043d\u0435\u0434\u043e\u0441\u0442\u0443\u043f\u0435\u043d/\u0447\u0442\u043e-\u0442\u043e
    \u043f\u043e\u0448\u043b\u043e \u043d\u0435 \u0442\u0430\u043a (\u0444\u0438\u0447\u0430 \u043e\u043f\u0446\u0438\u043e\u043d\u0430\u043b\u044c\u043d\u0430\u044f, \u043f\u0430\u0439\u043f\u043b\u0430\u0439\u043d \u043d\u0435 \u0434\u043e\u043b\u0436\u0435\u043d \u043f\u0430\u0434\u0430\u0442\u044c \u0431\u0435\u0437 \u043d\u0435\u0451)."""
    if np is None:
        return None
    sr = 8000   # \u043d\u0443\u0436\u043d\u0430 \u0442\u043e\u043b\u044c\u043a\u043e \u043e\u0433\u0438\u0431\u0430\u044e\u0449\u0430\u044f \u0433\u0440\u043e\u043c\u043a\u043e\u0441\u0442\u0438, \u043d\u0435 \u0437\u0432\u0443\u043a \u2014 \u043d\u0438\u0437\u043a\u0438\u0439 sr \u0434\u043e\u0441\u0442\u0430\u0442\u043e\u0447\u043d\u043e \u0438 \u0431\u044b\u0441\u0442\u0440\u043e
    try:
        r = subprocess.run(["ffmpeg", "-v", "quiet", "-i", audio_path, "-f", "s16le",
                            "-ac", "1", "-ar", str(sr), "-"], capture_output=True, timeout=120)
        if r.returncode != 0 or not r.stdout:
            return None
        raw = r.stdout[:len(r.stdout) - len(r.stdout) % 2]
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
        if len(samples) < sr:
            return None
        win = max(1, int(sr * window_sec))
        n_win = max(1, len(samples) // win)
        trimmed = samples[:n_win * win].reshape(n_win, win)
        rms = np.sqrt(np.mean(trimmed ** 2, axis=1))
        return rms, window_sec
    except Exception as e:
        print(f"  \u0410\u043d\u0430\u043b\u0438\u0437 \u0433\u0440\u043e\u043c\u043a\u043e\u0441\u0442\u0438 \u043d\u0435 \u0443\u0434\u0430\u043b\u0441\u044f (\u043f\u0440\u043e\u043f\u0443\u0441\u043a\u0430\u044e): {e}")
        return None


def energy_pace_multipliers(curve, starts, durs, lo=0.8, hi=1.25):
    """\u0413\u0440\u043e\u043c\u0447\u0435 \u0443\u0447\u0430\u0441\u0442\u043e\u043a \u2014 \u043a\u043e\u0440\u043e\u0447\u0435 \u043a\u0430\u0434\u0440\u044b (\u043c\u043d\u043e\u0436\u0438\u0442\u0435\u043b\u044c <1), \u0442\u0438\u0448\u0435 \u2014 \u0434\u043b\u0438\u043d\u043d\u0435\u0435 (>1).
    \u041c\u043d\u043e\u0436\u0438\u0442\u0435\u043b\u044c \u0441\u0447\u0438\u0442\u0430\u0435\u0442\u0441\u044f \u043e\u0442 \u043e\u0442\u043d\u043e\u0448\u0435\u043d\u0438\u044f \u043a \u041c\u0415\u0414\u0418\u0410\u041d\u0415 \u0433\u0440\u043e\u043c\u043a\u043e\u0441\u0442\u0438 \u0432\u0441\u0435\u0439 \u0434\u043e\u0440\u043e\u0436\u043a\u0438,
    \u0447\u0442\u043e\u0431\u044b \u0442\u0438\u0445\u0438\u0439 \u0440\u043e\u043b\u0438\u043a \u0446\u0435\u043b\u0438\u043a\u043e\u043c \u043d\u0435 \u0440\u0430\u0441\u0442\u044f\u0433\u0438\u0432\u0430\u043b \u0432\u0441\u0435 \u043a\u0430\u0434\u0440\u044b \u043e\u0434\u0438\u043d\u0430\u043a\u043e\u0432\u043e."""
    if curve is None:
        return [1.0] * len(durs)
    rms, window_sec = curve
    med = float(np.median(rms))
    if med <= 0:
        return [1.0] * len(durs)
    mults = []
    for t0, d in zip(starts, durs):
        i0 = max(0, int(t0 / window_sec))
        i1 = min(len(rms), max(i0 + 1, int((t0 + d) / window_sec)))
        seg = rms[i0:i1] if i0 < len(rms) else rms[-1:]
        e = float(seg.mean()) if len(seg) else med
        ratio = max(0.5, min(2.0, e / med))
        mults.append(max(lo, min(hi, 1.0 / ratio)))
    return mults


# --- Точная привязка кадров к речи (опционально, нужен faster-whisper) ---
# Без этого длительность блока — ОЦЕНКА (слова/скорость), а не факт: граница
# кадра никогда не совпадает с реальным концом фразы, только статистически
# близко к нему. faster-whisper даёт пословные тайм-коды по РЕАЛЬНОМУ аудио;
# порядок слов в озвучке известен заранее (TTS читает сценарий как есть),
# поэтому слово сценария сопоставляется слову распознавания не по смыслу, а
# по позиции — пропорциональным индексом в общем счётчике слов. Это не
# полноценный forced-aligner (Whisper иногда путает 1-2 слова), но для границ
# БЛОКОВ (не отдельных слов) точности достаточно, а зависимостей на порядок
# легче, чем у настоящего forced-alignment пакета.
try:
    from faster_whisper import WhisperModel
    WHISPER_LIBS = True
except ImportError:
    WHISPER_LIBS = False   # опциональная фича — без пакета работает прежняя оценка по словам

WHISPER_ENABLED = os.environ.get("WHISPER_ALIGN", "0") == "1" and WHISPER_LIBS
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")
WHISPER_LANG = os.environ.get("WHISPER_LANG", "ru")
_whisper_model = None
_whisper_lock = threading.Lock()


def get_whisper_model():
    global _whisper_model
    with _whisper_lock:
        if _whisper_model is None:
            _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
        return _whisper_model


def transcribe_words(audio_path):
    """[(start_sec, end_sec, word), ...] по всему аудио, в порядке звучания."""
    model = get_whisper_model()
    segments, _ = model.transcribe(audio_path, language=WHISPER_LANG, word_timestamps=True)
    words = []
    for seg in segments:
        for w in (seg.words or []):
            words.append((float(w.start), float(w.end), w.word))
    return words


def _map_index(k, n_total, m_total):
    """Пропорциональный индекс: k-е слово сценария (из n_total) -> позиция в
    последовательности из m_total распознанных слов. Монотонно неубывающая
    "линейная деформация времени" — без полного форс-алайнмента (DTW/edit
    distance), которого для границ блоков не требуется: несколько неверно
    распознанных слов внутри блока сдвигают индекс на доли слова, границы
    блоков это не портит заметно."""
    if n_total <= 0 or m_total <= 0:
        return 0
    return min(m_total - 1, max(0, round(k * (m_total - 1) / n_total)))


def whisper_breakpoints(blocks, audio_path):
    """[t0, t1, ..., tN] (N=len(blocks)) — граница времени речи между блоками
    по факту звучания audio_path. t0=0.0, tN=длина аудио. None при
    недоступности/сбое/подозрительном результате — тогда вызывающий код
    остаётся на прежней оценке по словам, тайминг не должен падать из-за
    опциональной фичи."""
    if not WHISPER_ENABLED:
        return None
    try:
        words = transcribe_words(audio_path)
    except Exception as e:
        print(f"  Whisper-выравнивание недоступно ({type(e).__name__}: {e}), оценка по словам.")
        return None
    script_word_counts = [b["words"] for b in blocks]
    n_total = sum(script_word_counts)
    m_total = len(words)
    if n_total <= 0 or m_total < n_total * 0.5:
        # Меньше половины ожидаемых слов распознано — модель не справилась с
        # этим аудио (шум, музыка, не тот язык), а не сценарий пустой.
        # Доверять таким тайм-кодам опаснее, чем прежней оценке.
        print(f"  Whisper распознал {m_total} слов на {n_total} в сценарии — "
              f"расхождение слишком большое, оценка по словам.")
        return None
    try:
        audio_len = get_media_duration(audio_path)
    except Exception:
        return None
    cum = 0
    breakpoints = [0.0]
    for count in script_word_counts:
        cum += count
        breakpoints.append(audio_len if cum >= n_total else words[_map_index(cum, n_total, m_total)][0])
    return breakpoints


def _format_srt_timestamp(t):
    ms = round(max(0.0, t) * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    sec, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def write_srt(blocks, breakpoints, path):
    """Экспорт субтитров по тем же тайм-кодам, что дал Whisper для тайминга
    монтажа — бесплатный побочный продукт точной привязки к речи. Внутри
    блока делим текст на предложения и распределяем время пропорционально
    числу слов: без этого длинный блок превращался бы в одну нечитаемую
    plaque на 15-20 секунд."""
    cues = []
    for b, t0, t1 in zip(blocks, breakpoints[:-1], breakpoints[1:]):
        text = b["text"].strip()
        if not text or t1 <= t0:
            continue
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if not sentences:
            continue
        counts = [max(1, len(s.split())) for s in sentences]
        total_words = sum(counts)
        cur = t0
        for sent, cnt in zip(sentences, counts):
            share = (t1 - t0) * cnt / total_words
            cues.append((cur, cur + share, sent))
            cur += share
    if not cues:
        return
    with open(path, "w", encoding="utf-8") as f:
        for n, (a, bnd, text) in enumerate(cues, 1):
            f.write(f"{n}\n{_format_srt_timestamp(a)} --> {_format_srt_timestamp(bnd)}\n{text}\n\n")
    print(f"Субтитры: {path} ({len(cues)} реплик)")


def find_audio():
    # flac/wav — новый lossless-выход fix_pauses.py (убирает второе поколение
    # lossy-сжатия между trim/loudnorm и финальным AAC-миксом); audio_fixed.mp3
    # остаётся резервом для роликов, сделанных до этого изменения.
    for name in ("audio_fixed.flac", "audio_fixed.wav", "audio_fixed.mp3", "audio.mp3"):
        p = os.path.join(VIDEO_FOLDER, name)
        if os.path.exists(p):
            return p
    # sorted(): listdir отдаёт файлы в порядке файловой системы, и на папке с
    # несколькими mp3 выбор "первого" менялся от прогона к прогону.
    mp3s = sorted(f for f in os.listdir(VIDEO_FOLDER) if f.lower().endswith(".mp3"))
    return os.path.join(VIDEO_FOLDER, mp3s[0]) if mp3s else os.path.join(VIDEO_FOLDER, "audio.mp3")


AUDIO_FILE = find_audio()


def load_json_dict(path):
    if os.path.exists(path):
        try:
            return json.load(open(path, encoding="utf-8"))
        except Exception as e:
            print(f"  Битый {path}, пропускаю: {e}")
    return {}


def load_themes():
    """Два уровня: канальный словарь (channel_themes.json в корне репо —
    общие для ниши слова вроде "меч"/"доспех"/"музей", не меняются от
    ролика к ролику) + эпизодный (media_plan/themes.json конкретного
    видео — только то, что специфично именно этой теме). Раньше весь
    словарь собирался заново руками под каждый новый сценарий; теперь
    новому эпизоду нужно всего несколько строк добавки поверх базы."""
    base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "channel_themes.json")
    episode_path = os.path.join(VIDEO_FOLDER, "media_plan", "themes.json")
    merged = load_json_dict(base_path)
    merged.update(load_json_dict(episode_path))
    return merged


THEMES = load_themes()


def get_audio_duration():
    r = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json",
                        "-show_format", AUDIO_FILE], capture_output=True, text=True, check=True)
    return float(json.loads(r.stdout)["format"]["duration"])


def parse_blocks(path):
    raw = open(path, encoding="utf-8").read()
    # Берём ТОЛЬКО озвучиваемые секции (HOOK / BLOCK* / FINAL). Всё служебное —
    # METADATA, PEXELS QUERIES, IMAGE PROMPTS, COMPETITOR ANALYSIS, TITLE/THUMBNAIL
    # OPTIONS — отбрасывается по имени секции, а не по списку известных заголовков:
    # раньше METADATA просачивалась в озвучку и съедала первый кадр.
    parts = re.split(r'===\s*(.*?)\s*===', raw)
    kept = []
    for i in range(1, len(parts), 2):
        name = parts[i].upper()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        if name.startswith(("HOOK", "BLOCK", "FINAL")):
            kept.append((name, body))
    if kept:
        # Маркер секции — ещё один спецтокен в общем потоке (не просто "\n"-склейка
        # как раньше), чтобы граница === не рвала перенос паузы через границу блока,
        # а секция при этом была известна для каждого получившегося блока.
        # БЕЗ .strip("\x00") — иначе снимается ведущий \x00 у маркера самой первой
        # секции (обычно HOOK), split() перестаёт его узнавать, и "SECTION:HOOK"
        # утекает в текст как отдельный псевдо-блок из одного слова — под первый
        # кадр хука уходит мусорный клип с generic-запросом вместо реального текста.
        content = "".join(f"\x00SECTION:{name}\x00{body}" for name, body in kept)
    else:
        content = re.sub(r'===.*?===', '', raw).strip()   # сценарий без === — берём как есть
    # [stat:TEXT] — цифра-плашка на экран (ЧАСТЬ "Монтаж под удержание": цифра
    # без плашки не запоминается). Вытаскиваем ДО общего вырезания [...],
    # иначе текст плашки пропадает вместе со всеми остальными тегами.
    content = re.sub(r'\[stat:(.*?)\]', lambda m: f"\x01STAT:{m.group(1)}\x01", content)
    processed = content
    for tag in sorted(PAUSE_DURATIONS, key=len, reverse=True):
        processed = processed.replace(tag, f"__PAUSE_{PAUSE_DURATIONS[tag]}__")
    # Пробел, а не пустая строка: теги в сценарии стоят СПЛОШНЯКОМ с текстом
    # ("грамма.[emphasis]Так" — ЧАСТЬ 10). Вырезание в пустоту склеивало
    # соседние слова в одно и занижало words у блока, то есть кадру
    # доставалось меньше времени, чем реально звучит текст. Ровно этот же
    # баг уже был найден и починен в wordcount.py — здесь остался.
    processed = re.sub(r'\[.*?\]', ' ', processed)
    parts = re.split(r'(__PAUSE_[\d.]+__|\x00SECTION:.*?\x00|\x01STAT:.*?\x01)', processed)
    blocks, cur, pause, stat = [], "", 0.0, None
    section = "BODY"

    def flush():
        nonlocal cur, pause, stat
        if cur:
            blocks.append({"text": cur, "pause_after": pause,
                           "words": len(cur.split()), "section": section, "stat": stat})
        cur, pause, stat = "", 0.0, None

    for part in parts:
        mp = re.match(r'__PAUSE_([\d.]+)__', part)
        ms = re.match(r'\x00SECTION:(.*?)\x00', part)
        mst = re.match(r'\x01STAT:(.*?)\x01', part)
        if mp:
            pause += float(mp.group(1))
        elif ms:
            flush()             # смена секции — всегда граница блока
            section = ms.group(1)
        elif mst:
            # Плашка НЕ режет монтаж — [stat:...] может стоять посреди фразы
            # без соседнего [pause], и это не повод обрывать клип на ровном
            # месте. Но если пауза уже открыта (граница блока ждёт следующий
            # текст), плашка относится к ЕЩЁ НЕ начатому следующему блоку —
            # без явного flush() она утекала бы в текущий cur (баг: плашка
            # после [pause] подписывалась под предыдущий, а не следующий кадр).
            if pause > 0 and cur:
                flush()
            stat = mst.group(1)
        else:
            t = part.strip()
            if t:
                if pause > 0 and cur:
                    flush()      # реальная пауза — вот это настоящая граница блока
                    cur = t
                else:
                    cur = f"{cur} {t}".strip()
    flush()
    print(f"Блоков: {len(blocks)}")
    return blocks


def fit_to_total(raw, mins, maxs, total, iters=60):
    """Подогнать длительности под ровно total, НЕ вылезая за [min, max].

    Раньше здесь был один общий scale = total / sum(d) поверх уже обрезанных
    по капу значений — и этот множитель капы просто перечёркивал. На реальном
    сценарии (18 мин, ~135 блоков) HOOK-кадры вместо предела 5с выходили на
    9.6с, а обычные вместо 20с — на 38с: то есть главное правило удержания
    ("в хуке кадры короче и чаще") в готовом ролике не работало вообще, при
    том что в коде оно записано.

    Вместо этого — water-filling: обрезали по границам, посчитали недобор/
    перебор и разложили остаток ТОЛЬКО по тем блокам, которым ещё есть куда
    двигаться. Сумма сходится к total и капы держатся одновременно.
    Если задача физически неразрешима (аудио длиннее суммы всех капов или
    короче суммы всех минимумов), синхрон важнее капа — честно говорим об
    этом и раскладываем пропорционально."""
    n = len(raw)
    if n == 0:
        return []
    lo_sum, hi_sum = sum(mins), sum(maxs)
    if total <= lo_sum or total >= hi_sum:
        # Границы недостижимы — держим общую длину (иначе поедет синхрон с
        # аудио), но предупреждаем: это значит, что блоков реально мало/много.
        base = mins if total <= lo_sum else maxs
        ssum = sum(base) or 1.0
        print(f"  Кадров {n} на {total:.0f}с — уложиться в пределы "
              f"{MIN_CLIP:g}-{MAX_CLIP:g}с невозможно "
              f"(коридор {lo_sum:.0f}-{hi_sum:.0f}с). Держу синхрон, капы не соблюдаю: "
              f"{'дроби блоки [pause]-ами' if total >= hi_sum else 'блоков слишком много'}.")
        return [x * total / ssum for x in base]
    d = [min(mx, max(mn, r)) for r, mn, mx in zip(raw, mins, maxs)]
    for _ in range(iters):
        diff = total - sum(d)
        if abs(diff) < 1e-6:
            break
        free = [i for i in range(n)
                if (diff > 0 and d[i] < maxs[i] - 1e-9) or (diff < 0 and d[i] > mins[i] + 1e-9)]
        if not free:
            break
        pool = sum(d[i] for i in free)
        if pool <= 1e-9:
            break
        k = (pool + diff) / pool
        for i in free:
            d[i] = min(maxs[i], max(mins[i], d[i] * k))
    return d


def block_durations(blocks, total, energy_mults=None, raw_override=None):
    """raw_override — реальные длительности по факту речи (Whisper), а не
    оценка по словам. Всё остальное (энергетический множитель, капы,
    water-filling под total) применяется одинаково независимо от источника
    сырых значений."""
    if raw_override is not None:
        raw = list(raw_override)
    else:
        tw = sum(b["words"] for b in blocks)
        tp = sum(b["pause_after"] for b in blocks)
        wps = tw / max(total - tp, 1)
        raw = [b["words"] / wps + b["pause_after"] for b in blocks]
    if energy_mults:
        raw = [r * m for r, m in zip(raw, energy_mults)]
    # В хуке кадры короче и чаще — первые секунды решают, останется ли зритель.
    mins = [MIN_CLIP] * len(blocks)
    maxs = [HOOK_MAX_CLIP if b["section"].startswith("HOOK") else MAX_CLIP for b in blocks]
    return fit_to_total(raw, mins, maxs, total)


PEXELS_BROKEN = False       # взводится только на реальном отказе API, не на пустой выдаче


def download_atomic(url, dest, timeout=20):
    """Скачать во ВРЕМЕННЫЙ файл и переименовать только после полной записи.

    Раньше писали сразу в файл кэша. Обрыв связи или Ctrl-C посреди загрузки
    оставлял в кэше обрезанный jpg/mp4 — и он лежал там НАВСЕГДА: на каждом
    следующем прогоне кэш считался готовым, ffmpeg на битом файле падал, блок
    молча оставался без кадра, а причина ниоткуда не была видна. Плюс
    отбрасываем пустой ответ (0 байт — тоже "успех" на уровне HTTP)."""
    tmp = f"{dest}.part"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
            data = r.read()
            if not data:
                raise ValueError("пустой ответ")
            f.write(data)
        os.replace(tmp, dest)
        return True
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


DUPE_HAMMING_THRESHOLD = 6   # тот же порог, что у qc_report — эмпирически "заметно похоже"


def pexels_photo(query, index, used_ids=None, avoid_hashes=None):
    """used_ids — множество ID уже показанных в этом ролике фото (мутируется на
    месте). Разные блоки часто ловят один и тот же тематический запрос — без
    этого им всем доставался бы top-1 результат, то есть одна и та же картинка
    по нескольку раз за ролик.

    avoid_hashes — список ahash уже ПРИНЯТЫХ в этом ролике кадров (мутируется
    на месте). qc_report уже находит визуально похожие кадры (Pexels ID разные,
    а по факту похожий кроп того же сюжета) — но делает это ПОСЛЕ сборки и
    только печатает список. Здесь тот же хэш применяется ДО рендера: похожий
    результат отбрасывается и пробуется следующий из выдачи — предотвращение,
    а не диагностика постфактум.

    Перебираем выдачу (per_page=15) и берём первый ID, которого ещё не было И
    чей ahash не похож на уже принятое; если вся выдача занята/похожа — берём
    последний скачанный кандидат всё равно (лучше повтор, чем сорванная
    сборка)."""
    global PEXELS_BROKEN
    cache = os.path.join(TEMP_FOLDER, "pexels_cache")
    os.makedirs(cache, exist_ok=True)
    # Хэш запроса в имени файла — иначе смена themes.json без чистки temp_smart/
    # молча оставляет картинку под старый запрос (кэш бил только по номеру блока).
    qhash = hashlib.md5(query.encode()).hexdigest()[:8]
    cf = os.path.join(cache, f"{index:04d}_{qhash}.jpg")
    if os.path.exists(cf):
        if avoid_hashes is not None:
            # Кэш-хит на повторном прогоне никогда не прогонялся через хэш в
            # ЭТОМ процессе — без этого добавления resume-прогон не видел бы,
            # что этот кадр уже "занят", и дедуп для новых кадров того же
            # запуска был бы неполным.
            try:
                avoid_hashes.append(ahash(cf))
            except Exception:
                pass
        return cf
    if not PEXELS_API_KEY:
        return None
    try:
        q = urllib.parse.quote(query)
        req = urllib.request.Request(
            f"https://api.pexels.com/v1/search?query={q}&per_page=15&orientation=landscape",
            headers={"Authorization": PEXELS_API_KEY, "User-Agent": UA})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.load(r)
    except Exception as e:
        PEXELS_BROKEN = True
        print(f"  Pexels [{query}]: {e}")
        return None
    photos = data.get("photos") or []
    if not photos:
        return None
    candidates = [p for p in photos if used_ids is None or p.get("id") not in used_ids] or photos
    got_any = False
    for pick in candidates:
        url = pick["src"].get("large2x") or pick["src"].get("large") or pick["src"].get("original")
        if not url:
            continue
        try:
            download_atomic(url, cf, timeout=20)
        except Exception as e:
            print(f"  Pexels [{query}] кандидат id={pick.get('id')}: {e}")
            continue
        got_any = True
        if used_ids is not None:
            used_ids.add(pick.get("id"))
        if avoid_hashes is None:
            break
        try:
            h = ahash(cf)
        except Exception:
            break   # хэш не считается — картинка всё равно валидна, отдаём как есть
        if not any(hamming(h, prev) <= DUPE_HAMMING_THRESHOLD for prev in avoid_hashes):
            avoid_hashes.append(h)
            break
        # похоже на уже принятое — cf уже перезапишется следующим кандидатом
    return cf if got_any else None


PHOTO_EXT = (".jpg", ".jpeg", ".png", ".webp")
VIDEO_EXT = (".mp4", ".mov", ".m4v", ".webm", ".mkv")


def scan_local_media():
    """Всё содержимое media/ по порядку слота: и фото, И ВИДЕО.

    Раньше сюда попадали только .jpg/.png. Но stock_fetch_multisource.py по
    схеме ЧАСТИ 14 кладёт в ту же папку чётные слоты как NNN_stock_video.mp4 —
    то есть ПОЛОВИНА скачанного стока не использовалась вообще, а оставшиеся
    фотографии прокручивались по кругу и повторялись по 2 раза за ролик.
    Читаем каталог ОДИН раз (раньше listdir дёргался на каждый блок).
    Сортировка — по числовому префиксу имени, как и раньше, чтобы порядок
    слотов из media_plan сохранялся."""
    if not os.path.isdir(MEDIA_FOLDER):
        return []
    items = []
    for f in os.listdir(MEDIA_FOLDER):
        low = f.lower()
        if low.startswith((".", "_")):
            continue
        kind = "photo" if low.endswith(PHOTO_EXT) else ("video" if low.endswith(VIDEO_EXT) else None)
        if not kind:
            continue
        nums = re.findall(r'\d+', f)
        items.append((int(nums[0]) if nums else 0, f, kind))
    items.sort(key=lambda x: (x[0], x[1]))
    return [(kind, os.path.join(MEDIA_FOLDER, f)) for _, f, kind in items]


LOCAL_MEDIA = scan_local_media()


def local_media(index):
    """(kind, path) для слота или None. Циклический перебор — как и раньше."""
    if not LOCAL_MEDIA:
        return None
    return LOCAL_MEDIA[index % len(LOCAL_MEDIA)]


def local_photo(index):
    """Совместимость: только фото из media/ (используется как последний
    резерв, когда для блока не нашлось вообще ничего другого)."""
    photos = [p for k, p in LOCAL_MEDIA if k == "photo"]
    return photos[index % len(photos)] if photos else None


GENERIC_FALLBACKS = [
    "medieval sword still life", "knight armor moody light",
    "medieval castle atmosphere", "old manuscript parchment history",
    "cinematic dark fantasy weapon",
]


def query_for(text):
    tl = text.lower()
    for kw, q in THEMES.items():
        if kw in tl:
            return q
    return None


def normalize_section_key(name):
    """"BLOCK 1: Название" / "BLOCK_1" / "HOOK" -> единый ключ ("HOOK",
    "BLOCK 1", "FINAL"), не зависящий ни от разметки, ни от произвольного
    человеческого титра после двоеточия."""
    if not name:
        return None
    n = name.strip().upper().replace("_", " ")
    if n.startswith("HOOK"):
        return "HOOK"
    if n.startswith("FINAL"):
        return "FINAL"
    m = re.match(r'BLOCK\s*(\d+)', n)
    return f"BLOCK {m.group(1)}" if m else None


def parse_pexels_queries(raw_text):
    """Разбирает === PEXELS QUERIES === script.txt в словарь
    {"HOOK": [q1, q2, ...], "BLOCK 1": [...], "FINAL": [...]}.

    Формат по ЧАСТИ 9: "HOOK: q1,q2,q3 / BLOCK_1: qa,qb / ...". Группы могут
    стоять и через "/", и по отдельным строкам — сценарий пишет LLM, а не
    парсер, оформление плывёт.

    Это ПРИОРИТЕТНЫЙ источник запросов (см. resolve_queries): протокол
    (ЧАСТЬ 13, шаг 3) требует финализировать эту секцию под конкретный
    сценарий ДО подбора стока — раньше её парсили только чтобы выбросить, и
    картинки подбирались исключительно по общему словарю канала. Самая
    точная информация о содержимом кадра писалась в файл и терялась."""
    parts = re.split(r'===\s*(.*?)\s*===', raw_text)
    body = ""
    for i in range(1, len(parts), 2):
        if parts[i].upper().startswith("PEXELS QUERIES"):
            body = parts[i + 1] if i + 1 < len(parts) else ""
            break
    if not body.strip():
        return {}
    out = {}
    for group in re.split(r'[/\n]+', body):
        group = group.strip().strip("()").strip()
        if not group or ":" not in group:
            continue
        key_raw, _, vals = group.partition(":")
        key = normalize_section_key(key_raw)
        qs = [v.strip() for v in vals.split(",") if v.strip()]
        if key and qs:
            out.setdefault(key, []).extend(qs)
    return out


def _read_script_queries():
    try:
        with open(SCRIPT_FILE, encoding="utf-8") as f:
            return parse_pexels_queries(f.read())
    except OSError:
        return {}


def resolve_queries(blocks):
    """Прямой поиск по themes.json ловит не все блоки — короткие связки
    ("Не дрались. Несли.", "Береги себя.") и абстрактные куски без предметных
    слов улетают в generic-заглушку, которая никак не привязана к теме ролика.
    Вместо одной фиксированной строки на все такие блоки: сперва пробуем
    унаследовать запрос соседнего блока той же секции (тема раздела обычно
    не меняется от предложения к предложению), и только если во всей секции
    вообще ничего не нашлось — берём по кругу из GENERIC_FALLBACKS (не один
    и тот же текст на всё, иначе Pexels отдаёт одну и ту же жалкую пятёрку).

    ПРИОРИТЕТ: если в === PEXELS QUERIES === script.txt для секции блока
    расписан список запросов — используем его, циклически меняя запрос
    внутри секции (несколько разных картинок на один раздел, а не одна и та
    же строка на все его кадры). Канальный словарь THEMES и вся резервная
    цепочка ниже остаются на случай, если раздела в PEXELS QUERIES нет или
    список пуст."""
    script_queries = _read_script_queries()
    section_cursor = {}
    raw = []
    for b in blocks:
        key = normalize_section_key(b["section"])
        qs = script_queries.get(key) if key else None
        if qs:
            idx = section_cursor.get(key, 0)
            raw.append(qs[idx % len(qs)])
            section_cursor[key] = idx + 1
        else:
            raw.append(query_for(b["text"]))
    resolved = list(raw)
    for i, q in enumerate(resolved):
        if q is not None:
            continue
        for j in range(i - 1, -1, -1):
            if blocks[j]["section"] != blocks[i]["section"]:
                break
            if raw[j] is not None:
                resolved[i] = raw[j]
                break
        if resolved[i] is not None:
            continue
        for j in range(i + 1, len(blocks)):
            if blocks[j]["section"] != blocks[i]["section"]:
                break
            if raw[j] is not None:
                resolved[i] = raw[j]
                break
    fallback_n = 0
    for i, q in enumerate(resolved):
        if q is None:
            resolved[i] = GENERIC_FALLBACKS[fallback_n % len(GENERIC_FALLBACKS)]
            fallback_n += 1
    return resolved


def add_overlays(vf_base, dur, title=None, stat=None):
    """Титр секции (низ кадра, слайд+fade первые ~2.5с, фирменный дисплейный
    шрифт) + цифровая плашка (верх кадра, красный баннер — акцентный цвет
    CHANNEL.md house style, держится почти весь клип: цифру без плашки не
    запоминают, см. производственные пометки сценария). Общий код для фото
    и видео-стока. Обычный fade раньше был самым узнаваемым штампом
    автослайдшоу — слайд добавляет "кто-то это анимировал руками"."""
    vf = vf_base
    if title and FONT_PATH:
        hold = min(2.5, dur * 0.6)
        fin = max(0.3, hold * 0.3)
        text = title.upper() if FONT_IS_DISPLAY else title
        safe = escape_drawtext(text)
        fs = 46 if FONT_IS_DISPLAY else 54
        vf += (f",drawtext=fontfile='{FONT_PATH}':{DRAWTEXT_OPTS}text='{safe}':"
               f"fontcolor=white:fontsize={fs}:borderw=3:bordercolor=black@0.8:"
               f"x=(w-text_w)/2:"
               f"y='h-180+(1-min(t/{fin:.2f}\\,1))*36':"
               f"alpha='if(lt(t\\,{fin:.2f})\\,t/{fin:.2f}\\,"
               f"if(lt(t\\,{hold:.2f})\\,1\\,max(0\\,1-(t-{hold:.2f})/0.4)))'")
    if stat and FONT_PATH:
        fin = 0.25
        hold = max(fin, dur - 0.5)
        text = stat.upper() if FONT_IS_DISPLAY else stat
        safe = escape_drawtext(text)
        fs = 58 if FONT_IS_DISPLAY else 64
        vf += (f",drawtext=fontfile='{FONT_PATH}':{DRAWTEXT_OPTS}text='{safe}':"
               f"fontcolor=white:fontsize={fs}:"
               f"box=1:boxcolor=0xC8102E@0.92:boxborderw=18:"
               f"x=(w-text_w)/2:"
               f"y='110-(1-min(t/{fin:.2f}\\,1))*26':"
               f"alpha='if(lt(t\\,{fin:.2f})\\,t/{fin:.2f}\\,"
               f"if(lt(t\\,{hold:.2f})\\,1\\,max(0\\,1-(t-{hold:.2f})/0.4)))'")
    return vf


def kb_hash_choices(photo):
    """Кандидаты зума/пана по хэшу файла — детерминированно, но БЕЗ памяти о
    соседних клипах (это даёт anti-repetition в main(), см. pick_no_repeat)."""
    h = int(hashlib.md5(photo.encode()).hexdigest()[:8], 16)
    zoom_in = bool(h & 1)
    pan_dir = PAN_DIRECTIONS[(h >> 1) % len(PAN_DIRECTIONS)]
    return h, zoom_in, pan_dir


def estimate_busyness(photo_path):
    """Грубая мера "тесноты" кадра без depth-модели — просто плотность
    перепадов яркости на уменьшенной копии. Не настоящая детекция крупности
    плана, но дешёвый прокси: у тесного/плотного кадра (уже крупный план)
    перепадов много на каждый пиксель, у просторного — мало. Используется
    только чтобы НЕ зумить ещё сильнее то, что и так уже крупный план."""
    try:
        img = PILImage.open(photo_path).convert("L").resize((160, 90))
        arr = np.asarray(img, dtype=np.float32) if np is not None else None
        if arr is None:
            return 0.5
        gx = np.abs(np.diff(arr, axis=1)).mean()
        gy = np.abs(np.diff(arr, axis=0)).mean()
        return float(min(1.0, (gx + gy) / 40.0))
    except Exception:
        return 0.5


def kenburns(photo, out, dur, title=None, zoom_in=None, pan_dir=None, stat=None, section=""):
    frames = max(1, round(dur * FPS))
    h, zoom_in_default, pan_dir_default = kb_hash_choices(photo)
    if zoom_in is None:
        zoom_in = zoom_in_default
    dx, dy = pan_dir if pan_dir is not None else pan_dir_default
    rate_jit = ((h >> 5) % 1000) / 1000.0
    pan_jit = ((h >> 15) % 1000) / 1000.0
    delta = max(ZOOM_DELTA_MIN, min(ZOOM_DELTA_MAX,
                ZOOM_RATE_BASE * dur * (0.75 + rate_jit * 0.5)))
    # Уже тесный/плотный кадр (крупный план) не нуждается в таком же зуме,
    # как просторный — иначе на кадре и так крупного меча к концу клипа
    # остаётся одна текстура металла, некуда уже приближаться со смыслом.
    busy = estimate_busyness(photo)
    if busy > 0.6:
        delta *= 0.7
    max_zoom = ZOOM_FLOOR + delta
    # PAN_SAFETY * jitter всегда < 1.0 (проверено численно) — офсет остаётся
    # строго внутри геометрического запаса (1-1/zoom)/2 на любом кадре клипа.
    pan_amt = PAN_SAFETY * (PAN_JITTER_MIN + pan_jit * (PAN_JITTER_MAX - PAN_JITTER_MIN))

    t = f"(on/{frames})"
    # smoothstep вместо линейного роста: старт/финиш без рывка — линейный зум
    # это самый узнаваемый штамп шаблонных слайд-шоу.
    eased = f"(3*pow({t},2)-2*pow({t},3))"
    z = (f"'{ZOOM_FLOOR}+{delta:.5f}*{eased}'" if zoom_in
         else f"'{max_zoom:.5f}-{delta:.5f}*{eased}'")
    # margin = (1-1/zoom)/2 считается от РЕАЛЬНОГО zoom в текущем кадре (переменная
    # zoompan даёт его сама) — офсет безопасен на любом кадре клипа по построению,
    # а не только к концу движения, как раньше при завязке на delta.
    x = (f"'iw/2-(iw/zoom/2){dx * pan_amt:+.5f}*(1-1/zoom)/2*iw'" if dx
         else "'iw/2-(iw/zoom/2)'")
    y = (f"'ih/2-(ih/zoom/2){dy * pan_amt:+.5f}*(1-1/zoom)/2*ih'" if dy
         else "'ih/2-(ih/zoom/2)'")
    # increase+crop (не decrease+pad) — исходные фото редко ровно 16:9, "вписать
    # в рамку" оставляло чёрные поля по краям на большинстве кадров (проверено:
    # 3 из 8 тестовых кадров с полосами по обеим сторонам). Залить кадр целиком
    # и обрезать лишнее — тот же приём, что уже применялся к видео-стоку ниже.
    vf_base = (f"scale=8000:4500:force_original_aspect_ratio=increase,"
               f"crop=8000:4500,setsar=1,"
               f"zoompan=z={z}:x={x}:y={y}:"
               f"d={frames}:s={WIDTH}x{HEIGHT}:fps={FPS},"
               f"{film_look(h, section)}")
    vf_overlay = add_overlays(vf_base, dur, title, stat) if (title or stat) else None

    def render(vf):
        cmd = ["ffmpeg", "-y", "-loop", "1", "-i", photo, "-vf", vf,
               "-t", str(dur), "-c:v", "libx264", "-preset", "fast",
               "-crf", CLIP_CRF, "-threads", str(FFMPEG_THREADS),
               "-pix_fmt", "yuv420p", "-r", str(FPS), out]
        return subprocess.run(cmd, capture_output=True, text=True)

    if vf_overlay:
        r = render(vf_overlay)
        if r.returncode == 0:
            return True
        # Титр/плашка (drawtext/шрифт) — не повод терять весь кадр: некоторые
        # сборки ffmpeg собраны без drawtext вообще. Откат на версию без них.
        print(f"  титр/плашка не встали ({os.path.basename(out)}), рисую без них: {r.stderr[-200:]}")
    return render(vf_base).returncode == 0


# --- 2.5D-параллакс (опционально, нужны torch/transformers/opencv) ---
# Depth-Anything-V2-Small с Hugging Face: по одной фотке строит карту глубины
# (белое — близко, чёрное — далеко), дальше "ближние" пиксели двигаются
# сильнее "дальних" при панорамировании — настоящее ощущение объёма вместо
# плоского зума. GitHub codeload (откуда обычно тянут MiDaS) закрыт сетевой
# политикой (проверено: 403) — huggingface.co открыт, поэтому источник модели
# именно оттуда.
PARALLAX_MARGIN = 1.5     # рабочий холст на 50% больше кадра — запас под
                           # зум+пан+параллакс-смещение без чёрных дыр по краям
PARALLAX_PX_BASE = 55.0   # максимальный доп.разброс между ближним/дальним слоем, px
_depth_model = None


def get_depth_model():
    global _depth_model
    if _depth_model is None:
        from transformers import pipeline as hf_pipeline
        _depth_model = hf_pipeline(task="depth-estimation",
                                    model="depth-anything/Depth-Anything-V2-Small-hf")
    return _depth_model


def estimate_depth(photo_path, work_w, work_h):
    """Карта глубины (float32, 0..1, 1=близко) под размер рабочего холста."""
    model = get_depth_model()
    img = PILImage.open(photo_path).convert("RGB")
    out = model(img)
    depth = np.array(out["predicted_depth"], dtype=np.float32)
    depth = cv2.resize(depth, (work_w, work_h), interpolation=cv2.INTER_LINEAR)
    d_min, d_max = float(depth.min()), float(depth.max())
    if d_max - d_min < 1e-6:
        return np.full((work_h, work_w), 0.5, dtype=np.float32)
    return (depth - d_min) / (d_max - d_min)


def fill_crop_canvas(photo_path, cw, ch):
    """Тот же increase+crop, что и в ffmpeg-пути: заливаем холст целиком,
    обрезаем лишнее — без чёрных полос по краям (BGR для cv2)."""
    img = PILImage.open(photo_path).convert("RGB")
    iw, ih = img.size
    scale = max(cw / iw, ch / ih)
    nw, nh = max(1, round(iw * scale)), max(1, round(ih * scale))
    img = img.resize((nw, nh), PILImage.LANCZOS)
    x0, y0 = (nw - cw) // 2, (nh - ch) // 2
    img = img.crop((x0, y0, x0 + cw, y0 + ch))
    arr = np.array(img)[:, :, ::-1].copy()   # RGB -> BGR
    return arr


def parallax_kenburns(photo, out, dur, title=None, zoom_in=None, pan_dir=None, stat=None, section=""):
    """2.5D-версия kenburns(): собственный покадровый рендер (OpenCV remap)
    вместо ffmpeg zoompan — только так можно сделать смещение, зависящее от
    глубины пикселя. При любой накладке (модель не встала, ffmpeg-пайп упал)
    возвращает False — вызывающий код откатывается на обычный kenburns()."""
    global PARALLAX_BROKEN
    if PARALLAX_BROKEN:
        return False
    proc = None
    try:
        frames = max(1, round(dur * FPS))
        cw, ch = round(WIDTH * PARALLAX_MARGIN), round(HEIGHT * PARALLAX_MARGIN)
        canvas = fill_crop_canvas(photo, cw, ch)
        try:
            depth = estimate_depth(photo, cw, ch)
        except Exception as e:
            # Сбой именно тут почти всегда системный (модель не встала /
            # сеть до HF Hub отвалилась), не проблема конкретной картинки —
            # дальше пытаться на каждом из оставшихся хайлайтов бессмысленно,
            # только тратим время на повторную (медленную) загрузку модели.
            print(f"  depth-модель недоступна, параллакс выключается на остаток ролика: "
                  f"{type(e).__name__} {e}")
            PARALLAX_BROKEN = True
            return False

        h, zoom_in_default, pan_dir_default = kb_hash_choices(photo)
        if zoom_in is None:
            zoom_in = zoom_in_default
        dx, dy = pan_dir if pan_dir is not None else pan_dir_default
        rate_jit = ((h >> 5) % 1000) / 1000.0
        pan_jit = ((h >> 15) % 1000) / 1000.0
        delta = max(ZOOM_DELTA_MIN, min(ZOOM_DELTA_MAX,
                    ZOOM_RATE_BASE * dur * (0.75 + rate_jit * 0.5)))
        max_zoom = ZOOM_FLOOR + delta
        pan_amt_frac = PAN_SAFETY * (PAN_JITTER_MIN + pan_jit * (PAN_JITTER_MAX - PAN_JITTER_MIN))
        parallax_px = PARALLAX_PX_BASE * (0.7 + rate_jit * 0.6)

        # Шаг сэмплирования по холсту. РАНЬШЕ здесь стояло просто /zoom, то есть
        # на кадр брались WIDTH/zoom пикселей холста из cw = WIDTH*1.5 — 64%
        # ширины картинки против 96% у обычного zoompan-пути. Хук и открывашки
        # разделов (единственные места, куда идёт параллакс) выходили заметно
        # крупнее остального ролика, да ещё и мягче: 1846 пикселей растягивались
        # до 1920. С множителем PARALLAX_MARGIN кадрирование совпадает с
        # zoompan-версией, а холст работает как запас под пан и параллакс —
        # ровно то, ради чего он и заводился.
        cx, cy = cw / 2.0, ch / 2.0
        ox_grid, oy_grid = np.meshgrid(np.arange(WIDTH, dtype=np.float32),
                                        np.arange(HEIGHT, dtype=np.float32))

        cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-",
               "-frames:v", str(frames)]
        vf = film_look(h, section)
        vf = add_overlays(vf, dur, title, stat) if (title or stat) else vf
        cmd += ["-vf", vf, "-c:v", "libx264", "-preset", "fast", "-crf", CLIP_CRF,
                "-threads", str(FFMPEG_THREADS), "-pix_fmt", "yuv420p", "-r", str(FPS), out]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        for frame_i in range(frames):
            t = frame_i / frames
            eased = 3 * t ** 2 - 2 * t ** 3   # smoothstep — тот же профиль, что у zoompan-версии
            zoom = (ZOOM_FLOOR + delta * eased) if zoom_in else (max_zoom - delta * eased)
            # Шаг сэмплирования по холсту. Раньше здесь было просто /zoom: на кадр
            # бралось WIDTH/zoom пикселей холста шириной cw = WIDTH*PARALLAX_MARGIN,
            # то есть 64% ширины картинки против 96% у обычного zoompan-пути. Хук и
            # открывашки разделов (единственные места, куда идёт параллакс) выходили
            # заметно крупнее остального ролика и мягче — 1846 пикселей растягивались
            # до 1920. С множителем PARALLAX_MARGIN кадрирование совпадает с
            # zoompan-версией, а холст остаётся тем, чем задумывался: запасом под пан
            # и параллакс-смещение.
            sx, sy = PARALLAX_MARGIN / zoom, PARALLAX_MARGIN / zoom
            # Геометрический запас на текущем зуме, минус резерв под параллакс —
            # так суммарное смещение гарантированно не вылезает за холст, и по краям
            # кадра не появляется размазанный BORDER_REPLICATE.
            avail_x = max(0.0, (cw - cw / zoom) / 2.0 - parallax_px / 2.0)
            avail_y = max(0.0, (ch - ch / zoom) / 2.0 - parallax_px / 2.0)

            map_x0 = cx + (ox_grid - WIDTH / 2) * sx + dx * pan_amt_frac * avail_x
            map_y0 = cy + (oy_grid - HEIGHT / 2) * sy + dy * pan_amt_frac * avail_y
            map_x0 = np.clip(map_x0, 0, cw - 1)
            map_y0 = np.clip(map_y0, 0, ch - 1)

            d_here = cv2.remap(depth, map_x0, map_y0, interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
            extra = (d_here - 0.5) * parallax_px * eased   # центрировано: ближе/дальше среднего
            map_x = np.clip(map_x0 + dx * extra, 0, cw - 1)
            map_y = np.clip(map_y0 + dy * extra, 0, ch - 1)

            frame = cv2.remap(canvas, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
            proc.stdin.write(frame.tobytes())

        proc.stdin.close()
        proc.wait(timeout=60)
        if proc.returncode != 0:
            print(f"  параллакс-рендер не встал ({os.path.basename(out)}): "
                  f"{proc.stderr.read().decode(errors='replace')[-200:]}")
            return False
        return True
    except Exception as e:
        print(f"  параллакс сорвался ({os.path.basename(out)}): {type(e).__name__} {e}")
        return False
    finally:
        # Раньше при сбое ПОСРЕДИ покадрового цикла (BrokenPipeError на
        # stdin.write, таймаут на wait) дочерний ffmpeg не убивался вообще —
        # завис бы висячим процессом до конца сессии. Гасим его в любом
        # случае, если он ещё жив к выходу из функции.
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass


# Не динамический ramp внутри клипа (это отдельная и намного более сложная
# задача — переменная PTS-кривая), а постоянная скорость по контексту клипа:
# хук чуть быстрее — энергичнее, финал чуть медленнее — снимает напряжение.
SPEED_BIAS = {"HOOK": 1.12, "FINAL": 0.92}


def video_render(vid, out, dur, title=None, stat=None, section=""):
    """Аналог kenburns(), но для стокового видео: без zoompan (движение уже
    есть в кадре), заливка кадра целиком + обрезка (не letterbox), растяжение
    по времени, если исходный ролик короче нужной длительности."""
    try:
        actual = get_media_duration(vid)
    except Exception:
        actual = dur
    h = int(hashlib.md5(vid.encode()).hexdigest()[:8], 16)
    vf_base = f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,crop={WIDTH}:{HEIGHT},setsar=1"
    bias = next((v for k, v in SPEED_BIAS.items() if section.startswith(k)), 1.0)
    setpts_factor = None
    if actual < dur - 0.05:
        setpts_factor = dur / max(actual, 0.1)   # уже растягиваем нехватку — bias не добавляем поверх
    elif bias != 1.0 and actual >= dur * bias:
        setpts_factor = 1.0 / bias
    if setpts_factor is not None:
        vf_base += f",setpts={setpts_factor:.5f}*PTS"
    vf_base += f",{film_look(h, section)}"
    vf_overlay = add_overlays(vf_base, dur, title, stat) if (title or stat) else None

    def render(vf):
        cmd = ["ffmpeg", "-y", "-i", vid, "-vf", vf, "-t", str(dur), "-an",
               "-c:v", "libx264", "-preset", "fast", "-crf", CLIP_CRF,
               "-threads", str(FFMPEG_THREADS), "-pix_fmt", "yuv420p", "-r", str(FPS), out]
        return subprocess.run(cmd, capture_output=True, text=True)

    if vf_overlay:
        r = render(vf_overlay)
        if r.returncode == 0:
            return True
        print(f"  титр/плашка не встали на видео ({os.path.basename(out)}), рисую без них: {r.stderr[-200:]}")
    return render(vf_base).returncode == 0


def pexels_video(query, index, used_ids=None):
    """Тот же принцип, что pexels_photo(): перебираем выдачу и берём первое
    ещё не показанное видео. Из доступных video_files берём ближайшее по
    ширине к целевому 1920 — не тянем 4K ради 1080p-выхода."""
    global PEXELS_BROKEN
    cache = os.path.join(TEMP_FOLDER, "pexels_video_cache")
    os.makedirs(cache, exist_ok=True)
    qhash = hashlib.md5(query.encode()).hexdigest()[:8]
    cf = os.path.join(cache, f"{index:04d}_{qhash}.mp4")
    if os.path.exists(cf):
        return cf
    if not PEXELS_API_KEY:
        return None
    try:
        q = urllib.parse.quote(query)
        req = urllib.request.Request(
            f"https://api.pexels.com/videos/search?query={q}&per_page=10&orientation=landscape",
            headers={"Authorization": PEXELS_API_KEY, "User-Agent": UA})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.load(r)
        videos = data.get("videos") or []
        if not videos:
            return None
        pick = videos[0]
        if used_ids is not None:
            for v in videos:
                if v.get("id") not in used_ids:
                    pick = v
                    break
        if used_ids is not None:
            used_ids.add(pick.get("id"))
        files = [f for f in (pick.get("video_files") or [])
                 if f.get("file_type") == "video/mp4" and f.get("width")]
        if not files:
            return None
        best = min(files, key=lambda f: abs(f["width"] - WIDTH))
        download_atomic(best["link"], cf, timeout=40)
        return cf
    except Exception as e:
        PEXELS_BROKEN = True
        print(f"  Pexels video [{query}]: {e}")
        return None


def plan_transitions(sections, xfade_dur=XFADE_DUR):
    """Заранее и детерминированно решить, КАКОЙ переход и КАКОЙ длины стоит
    на каждой склейке. Возвращает список (transition, duration) длиной n-1.

    Ключ хэша — номер стыка и имена секций, а НЕ имена файлов клипов. Это
    принципиально: имя файла клипа содержит хэш его длительности, а
    длительность считается от бюджета переходов — то есть от результата этой
    самой функции. Завязка на имена файлов делала бюджет невычислимым заранее,
    и main() был вынужден закладывать XFADE_DUR на КАЖДУЮ склейку, хотя ~65%
    из них — почти мгновенный hardcut на 0.06с. Разница копилась: на 135
    кадрах видео получалось на ~25с длиннее аудио, финальный -t срезал хвост,
    а картинка на всём протяжении ролика всё сильнее отставала от слов."""
    n = len(sections)
    plan = []
    cut_hist, boundary_hist = [], []
    for i in range(1, n):
        h = int(hashlib.md5(f"xfade:{i}|{sections[i-1]}|{sections[i]}".encode()).hexdigest()[:8], 16)
        if sections[i] != sections[i - 1]:
            # Смена темы — заметный переход, не обычная склейка. dissolve/fadeblack/
            # fadewhite (вспышка светом — читается как "новая глава начинается ярко")
            # вперемешку.
            candidate = BOUNDARY_TRANSITIONS[h % len(BOUNDARY_TRANSITIONS)]
            plan.append((pick_no_repeat(boundary_hist, candidate, BOUNDARY_TRANSITIONS, 1), xfade_dur))
        else:
            # Большинство склеек в реальном монтаже — жёсткий cut, не dissolve;
            # заметный переход — редкость, не норма. ~65% hard cut / ~35% вариация.
            candidate = "hardcut" if (h % 3 != 0) else XFADE_TRANSITIONS[(h >> 8) % len(XFADE_TRANSITIONS)]
            choice = pick_no_repeat(cut_hist, candidate, ["hardcut"] + XFADE_TRANSITIONS, max_repeat=3)
            plan.append(("fade" if choice == "hardcut" else choice,
                         XFADE_DUR_HARD if choice == "hardcut" else xfade_dur))
    return plan


def transitions_budget(sections, xfade_dur=XFADE_DUR):
    """Сколько секунд суммарно съедят склейки — точно, а не по верхней оценке."""
    return sum(d for _, d in plan_transitions(sections, xfade_dur))


def xfade_chain(clips, durs, sections, out, xfade_dur=XFADE_DUR, plan=None):
    """Один проход filter_complex с цепочкой xfade между ВСЕМИ соседними
    кадрами — вместо жёсткой склейки. Тип перехода и длительность варьируются
    (разнообразие + иногда почти жёсткий cut), на границе секций — заметный
    dissolve. Возвращает (True, итоговая_длительность) или (False, 0.0), если
    что-то пошло не так (тогда main() откатывается на обычный concat -c copy)."""
    n = len(clips)
    if n < 2:
        return False, 0.0
    if plan is None:
        plan = plan_transitions(sections, xfade_dur)
    parts, prev_label, cum = [], "0:v", durs[0]
    for i in range(1, n):
        transition, this_dur = plan[i - 1]
        offset = max(0.0, cum - this_dur)
        out_label = f"vx{i}" if i < n - 1 else "vout"
        parts.append(f"[{prev_label}][{i}:v]xfade=transition={transition}:"
                     f"duration={this_dur:.3f}:offset={offset:.3f}[{out_label}]")
        cum = cum + durs[i] - this_dur
        prev_label = out_label
    cmd = ["ffmpeg", "-y"]
    for c in clips:
        cmd += ["-i", c]
    cmd += ["-filter_complex", ";".join(parts), "-map", "[vout]",
            "-c:v", "libx264", "-preset", FINAL_PRESET, "-crf", FINAL_CRF,
            "-pix_fmt", "yuv420p", "-r", str(FPS), out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("  xfade-склейка не удалась, откат на concat:", r.stderr[-300:])
        return False, 0.0
    return True, max(cum, 0.1)


def pad_to_length(video, target, temp_dir):
    """Достраивает видео до нужной длины заморозкой последнего кадра — нужно
    после xfade_chain(), которая суммарно укорачивает ролик на (n-1)*XFADE_DUR
    относительно суммы длительностей кадров."""
    cur = get_media_duration(video)
    gap = target - cur
    if gap <= 0.05:
        return video
    lastframe = os.path.join(temp_dir, "_lastframe.jpg")
    padclip = os.path.join(temp_dir, "_pad.mp4")
    padded = os.path.join(temp_dir, "_padded.mp4")
    subprocess.run(["ffmpeg", "-y", "-sseof", "-0.3", "-i", video,
                    "-vframes", "1", lastframe], capture_output=True)
    r = subprocess.run(["ffmpeg", "-y", "-loop", "1", "-i", lastframe, "-t", f"{gap:.3f}",
                        "-c:v", "libx264", "-preset", "fast", "-crf", CLIP_CRF,
                        "-pix_fmt", "yuv420p", "-r", str(FPS), padclip],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return video
    lst = os.path.join(temp_dir, "_pad_concat.txt")
    open(lst, "w", encoding="utf-8").write(
        f"file '{os.path.abspath(video)}'\nfile '{os.path.abspath(padclip)}'\n")
    r = subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                        "-i", lst, "-c", "copy", padded], capture_output=True, text=True)
    return padded if r.returncode == 0 else video


def get_media_duration(path):
    r = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json",
                        "-show_format", path], capture_output=True, text=True, check=True)
    return float(json.loads(r.stdout)["format"]["duration"])


def ahash(photo_path, size=8):
    """Средний хэш (average hash) картинки — 64-битная строка, дешёвая
    замена imagehash-библиотеке (Pillow уже обязательная зависимость, лишний
    пакет не нужен). Похожие по контенту кадры дают близкий хэш даже если
    это разные файлы с разных источников."""
    img = PILImage.open(photo_path).convert("L").resize((size, size), PILImage.LANCZOS)
    pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)
    return "".join("1" if p > avg else "0" for p in pixels)


def hamming(a, b):
    return sum(c1 != c2 for c1, c2 in zip(a, b))


def qc_report(media_log):
    """Не блокирует сборку — просто честно докладывает, если несколько
    кадров ролика визуально почти одинаковые (Pexels ID разные, а по факту
    похожий кроп/пересъёмка того же сюжета — то, что ловится глазом, но не
    ловится дедупом по ID). Порог 6 бит из 64 — эмпирически "заметно похоже"."""
    if len(media_log) < 2:
        return
    hashes = []
    for i, path in media_log:
        try:
            hashes.append((i, ahash(path)))
        except Exception:
            pass
    dupes = []
    for a in range(len(hashes)):
        for bb in range(a + 1, len(hashes)):
            i1, h1 = hashes[a]
            i2, h2 = hashes[bb]
            d = hamming(h1, h2)
            if d <= 6:
                dupes.append((i1 + 1, i2 + 1, d))
    if dupes:
        print(f"QC: похожие кадры (проверь глазами) — {dupes}")
    else:
        print("QC: явных повторов по картинке не найдено")


def main():
    if not os.path.exists(AUDIO_FILE):
        print(f"Аудио не найдено: {AUDIO_FILE}")
        return 1
    total = get_audio_duration()
    print(f"Аудио: {total:.1f}с ({total/60:.1f} мин)")
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    blocks = parse_blocks(SCRIPT_FILE) if os.path.exists(SCRIPT_FILE) else []
    if not blocks:
        print("Сценарий не найден/пуст")
        return 1
    # Кроссфейд между КАЖДОЙ парой кадров "съедает" время — закладываем это в
    # целевую длительность заранее, чтобы после склейки общая длина видео снова
    # совпала с аудио. Бюджет считается ТОЧНО, по заранее построенному плану
    # переходов (plan_transitions), а не по верхней оценке (n-1)*XFADE_DUR:
    # большинство склеек — hardcut на 0.06с, и завышенный бюджет раздувал
    # кадры так, что видео выходило на десятки секунд длиннее аудио, хвост
    # срезался, а картинка ползла всё дальше от слов к концу ролика.
    block_sections = [b["section"] for b in blocks]
    xfade_plan = plan_transitions(block_sections)
    xfade_budget = sum(d for _, d in xfade_plan)
    target = total + xfade_budget

    # Whisper даёт РЕАЛЬНЫЕ границы речи вместо оценки по словам (опционально,
    # WHISPER_ALIGN=1). Когда он сработал — энергетический ритм ниже не нужен
    # и не запускается: тот эвристический проход был попыткой приблизить
    # оценку к реальному темпу речи, а тут уже сам реальный темп.
    whisper_bp = whisper_breakpoints(blocks, AUDIO_FILE)
    if whisper_bp:
        raw = [whisper_bp[i + 1] - whisper_bp[i] for i in range(len(blocks))]
        durs = block_durations(blocks, target, raw_override=raw)
        print("Тайминг блоков: по факту речи (Whisper)")
        write_srt(blocks, whisper_bp, os.path.join(VIDEO_FOLDER, "subtitles.srt"))
    else:
        durs = block_durations(blocks, target)
        # Второй проход: ритм по громкости поверх word-count-базы (не вместо
        # неё) — громкие места режутся чаще, тихие держатся дольше.
        # Опционально (нужен numpy). Стартовые точки для сэмплинга энергии
        # считаем от РЕАЛЬНОЙ длины аудио (total), не от раздутой под
        # кроссфейды target — иначе поздние блоки на длинном ролике со
        # множеством склеек сэмплили бы энергию не в том месте.
        curve = audio_energy_curve(AUDIO_FILE)
        if curve:
            baseline = block_durations(blocks, total)
            starts, acc = [], 0.0
            for d in baseline:
                starts.append(acc)
                acc += d
            mults = energy_pace_multipliers(curve, starts, baseline)
            durs = block_durations(blocks, target, energy_mults=mults)
            print("Ритм по громкости: включён")
    print(f"Средний кадр: {sum(durs)/len(durs):.1f}с")

    missing = []   # индексы блоков, для которых не нашлось ни фото, ни видео
    zoom_hist, pan_hist = [], []
    use_local = bool(LOCAL_MEDIA)
    use_pexels = bool(PEXELS_API_KEY)
    queries = resolve_queries(blocks)
    used_photo_ids = set()   # общий на весь ролик — не даём одной фотке всплыть дважды
    used_video_ids = set()   # то же самое, отдельно для видео (разные ID-пространства)
    avoid_hashes = []        # ahash принятых фото — не даём визуально похожему кадру пройти
    last_media = None        # чем закрыть блок, для которого вообще ничего не нашлось
    reused = []              # такие блоки — чтобы честно сказать о них в конце

    # --- Фаза A: все решения по порядку — медиа (сеть + общий дедуп), зум/пан
    # (анти-повтор по истории), кэш-хиты уже готовых клипов. Последовательно,
    # потому что мутирует общее состояние (used_ids/avoid_hashes/zoom_hist/
    # last_media), которое должно видеть каждое предыдущее решение.
    jobs = [None] * len(blocks)
    for i, (b, d) in enumerate(zip(blocks, durs)):
        # Титр темы — только на ПЕРВОМ кадре новой секции (BLOCK N: Название).
        is_section_start = i == 0 or blocks[i]["section"] != blocks[i - 1]["section"]
        title = section_title(b["section"]) if is_section_start else None
        stat = b.get("stat")
        # Медиа разрешается ДО проверки кэша клипа: иначе имя клипа не знает,
        # из какой картинки он собран, и подмена файла в media/ или правка
        # themes.json молча оставляли старый кадр (тот же класс бага, что уже
        # правили для pexels_cache и для параметров рендера).
        photo = video = None
        local = local_media(i) if use_local else None
        if local:
            kind, path = local
            if kind == "video":
                video = path
            else:
                photo = path
        if not photo and not video and use_pexels:
            # Чередуем фото/видео через один — живое движение вперемешку со
            # статикой вместо чистого слайд-шоу (ЧАСТЬ 14: нечётные фото,
            # чётные видео). Слишком короткому кадру видео не заказываем —
            # не тянуть ролик ради 4 секунд. Если у предпочтённого типа для
            # этой темы пусто — откатываемся на другой тип, а не теряем кадр.
            prefer_video = (i % 2 == 1) and d >= MIN_CLIP + 1.0
            if prefer_video:
                video = pexels_video(queries[i], i, used_ids=used_video_ids)
                if not video:
                    photo = pexels_photo(queries[i], i, used_ids=used_photo_ids, avoid_hashes=avoid_hashes)
            else:
                photo = pexels_photo(queries[i], i, used_ids=used_photo_ids, avoid_hashes=avoid_hashes)
                if not photo and d >= MIN_CLIP + 1.0:
                    video = pexels_video(queries[i], i, used_ids=used_video_ids)
            # Раньше Pexels отключался навсегда после ЛЮБОГО промаха, включая
            # обычную пустую выдачу по одному неудачному запросу. Гасим источник
            # только если API реально отвалился.
            if not photo and not video and PEXELS_BROKEN:
                use_pexels = False
        if not photo and not video:
            photo = local_photo(i)
        reuse = False
        if not photo and not video and last_media:
            # Выбросить блок нельзя: его длительность заложена в общий тайминг,
            # и без кадра ВСЁ, что идёт дальше, уезжает относительно звука (на
            # длинном ролике это десятки секунд рассинхрона к финалу). Повторяем
            # предыдущее медиа — один повторный кадр заметно дешевле сбитого
            # синхрона на весь остаток ролика. Зум разворачиваем в обратную
            # сторону, чтобы повтор не читался как зависшая картинка.
            kind, path = last_media
            video, photo = (path, None) if kind == "video" else (None, path)
            reuse = True
            reused.append(i + 1)
        if not photo and not video:
            print(f"  [{i+1}] нет медиа")
            missing.append(i + 1)
            continue
        last_media = ("video", video) if video else ("photo", photo)
        src = video or photo
        params_hash = hashlib.md5(
            f"{d:.3f}|{title}|{stat}|{b['section']}|{os.path.basename(src)}|{reuse}".encode()
        ).hexdigest()[:8]
        out = os.path.join(TEMP_FOLDER, f"clip_{i:04d}_{params_hash}.mp4")
        if os.path.exists(out):
            jobs[i] = {"cached": True, "out": out, "d": d, "section": b["section"],
                       "photo_for_log": photo}
            continue
        if video:
            jobs[i] = {"cached": False, "kind": "video", "src": video, "out": out, "d": d,
                       "title": title, "stat": stat, "section": b["section"]}
        else:
            # anti-repetition: хэш сам по себе не мешает 3 зумам подряд случайно
            # совпасть — держим окно последних решений и форсируем смену при повторе.
            _, zi_cand, pd_cand = kb_hash_choices(photo)
            if reuse:
                zi_cand = not zi_cand   # повтор кадра — хотя бы в другую сторону
            zoom_in = pick_no_repeat(zoom_hist, zi_cand, [True, False], max_repeat=2)
            pan_dir = pick_no_repeat(pan_hist, pd_cand, PAN_DIRECTIONS, max_repeat=2)
            # Параллакс — только на самые заметные точки ролика (хук целиком +
            # первый кадр каждого раздела), не на все фото: покадровый рендер
            # с depth-моделью в разы дороже по времени zoompan-версии.
            is_highlight = b["section"].startswith("HOOK") or is_section_start
            jobs[i] = {"cached": False, "kind": "photo", "src": photo, "out": out, "d": d,
                       "title": title, "stat": stat, "section": b["section"],
                       "zoom_in": zoom_in, "pan_dir": pan_dir, "is_highlight": is_highlight}
        if use_pexels and not use_local and i % 10 == 9:
            time.sleep(0.4)

    # --- Фаза B: параллельный рендер того, чего не нашлось в кэше готовых
    # клипов. Каждый job уже несёт все параметры — рендер одного клипа не
    # зависит ни от чего вне себя, поэтому N параллельных процессов ffmpeg
    # дают тот же результат, что и последовательный перебор, только быстрее.
    def render_job(job):
        if job["kind"] == "video":
            return video_render(job["src"], job["out"], job["d"], title=job["title"],
                                 stat=job["stat"], section=job["section"])
        ok = False
        if PARALLAX_ENABLED and job["is_highlight"]:
            ok = parallax_kenburns(job["src"], job["out"], job["d"], title=job["title"],
                                    zoom_in=job["zoom_in"], pan_dir=job["pan_dir"],
                                    stat=job["stat"], section=job["section"])
        if not ok:
            ok = kenburns(job["src"], job["out"], job["d"], title=job["title"],
                          zoom_in=job["zoom_in"], pan_dir=job["pan_dir"],
                          stat=job["stat"], section=job["section"])
        return ok

    to_render = [(i, j) for i, j in enumerate(jobs) if j is not None and not j["cached"]]
    results = {}
    if to_render:
        print(f"Рендер {len(to_render)} кадров, {WORKERS} параллельно "
              f"({FFMPEG_THREADS} потока ffmpeg на кадр)...")
        print_lock = threading.Lock()
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futures = {ex.submit(render_job, job): i for i, job in to_render}
            done_n = 0
            for fut in concurrent.futures.as_completed(futures):
                i = futures[fut]
                try:
                    ok = fut.result()
                except Exception as e:
                    print(f"  [{i+1}] рендер упал: {type(e).__name__} {e}")
                    ok = False
                results[i] = ok
                done_n += 1
                if not ok or done_n % 20 == 0 or done_n <= 3:
                    with print_lock:
                        print(f"  [{done_n}/{len(to_render)}] блок {i+1}/{len(blocks)} "
                              f"{'OK' if ok else 'ОШИБКА'}")

    # --- Сборка результата в ИСХОДНОМ порядке блоков — порядок завершения
    # потоков произвольный, xfade_chain требует клипы строго по таймлайну.
    clips, clip_durs, clip_sections = [], [], []
    media_log = []   # (индекс, путь_к_фото) — для QC-проверки на похожие кадры в конце
    for i, job in enumerate(jobs):
        if job is None:
            continue
        ok = True if job["cached"] else results.get(i, False)
        if ok:
            clips.append(job["out"])
            clip_durs.append(job["d"])
            clip_sections.append(job["section"])
            photo_for_log = job.get("photo_for_log") if job["cached"] else (
                job["src"] if job["kind"] == "photo" else None)
            if photo_for_log:
                media_log.append((i, photo_for_log))
        else:
            missing.append(i + 1)

    if not clips:
        print("Нет клипов")
        return 1
    merged = os.path.join(TEMP_FOLDER, "merged.mp4")
    # План переходов должен совпадать с тем, по которому считался бюджет. Если
    # все блоки доехали (обычный случай) — берём его как есть; если какой-то
    # блок всё же выпал, честно пересчитываем по фактической цепочке клипов.
    plan = xfade_plan if len(clips) == len(blocks) else plan_transitions(clip_sections)
    ok, xfade_total = xfade_chain(clips, clip_durs, clip_sections, merged, plan=plan)
    if not ok:
        concat = os.path.join(TEMP_FOLDER, "concat.txt")
        # Пути ТОЛЬКО абсолютные: concat-демуксер ffmpeg резолвит относительные
        # пути от папки самого concat.txt, а не от cwd — иначе сборка падает.
        open(concat, "w", encoding="utf-8").write(
            "".join(f"file '{os.path.abspath(c)}'\n" for c in clips))
        r = subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                            "-i", concat, "-c", "copy", merged], capture_output=True, text=True)
        if r.returncode != 0:
            print("Склейка:", r.stderr[-300:])
            return 1
    merged = pad_to_length(merged, total, TEMP_FOLDER)
    # -shortest САМ ПО СЕБЕ недостаточен с -c:v copy: копирование пакетов
    # режет только по границам GOP исходного клипа, а не по факту конца
    # аудио — на реальном 17-минутном ролике это давало +6с видео без
    # звука сверху (проверено вживую). Явный -t с точной длительностью
    # аудио режет ровно там, где надо, независимо от размера GOP.
    r = subprocess.run(["ffmpeg", "-y", "-i", merged, "-i", AUDIO_FILE,
                        "-t", f"{total:.3f}",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                        "-movflags", "+faststart",
                        "-shortest", OUTPUT_FILE], capture_output=True, text=True)
    if r.returncode != 0:
        print("Аудио:", r.stderr[-300:])
        return 1
    mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    # Пропущенные кадры и раньше не останавливали сборку (стратегия
    # "лучше меньше клипов, чем сорванный рендер") — но раньше это тонуло
    # в середине лога, а ГОТОВО выглядело как чистый успех. Теперь пропуски
    # видны в итоговой строке, и код возврата честно отражает, что не всё
    # доехало (не 0, если что-то пропущено).
    status = f" | ПРОПУЩЕНО {len(missing)} блоков: {missing}" if missing else ""
    if reused:
        status += f" | ПОВТОР медиа (не нашлось своего) в блоках: {reused}"
    print(f"\nГОТОВО: {OUTPUT_FILE} ({mb:.0f} MB, {total/60:.1f} мин, {len(clips)} кадров){status}")
    qc_report(media_log)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
