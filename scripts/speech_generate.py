#!/usr/bin/env python3
"""Speech Direction Engine — Stage B (генерация): встраивает ElevenLabs
ПРЯМО в генератор вместо ручного Шага 6. Читает media_plan/speech_plan.json
(Stage A — scripts/speech_planner.py, обязателен как предыдущий шаг: план
по главам/риторическим ролям уже посчитан там бесплатно, Stage B его не
пересчитывает), группирует юниты в СВЯЗНЫЕ ФРАГМЕНТЫ (несколько
предложений одной секции и одной риторической стадии — не одна фраза за
вызов), озвучивает каждый через ElevenLabs (audio + посимвольный
alignment за один вызов, endpoint with-timestamps), оценивает результат
(паузы, темп, энергия — не только длина тишины) и решает: принять,
исправить только границу паузы или перегенерировать ИМЕННО этот фрагмент
(максимум 2 попытки, жёстко — см. MAX_ATTEMPTS_HARD_CAP). Кэш по хэшу
текста — одинаковый фрагмент дважды не оплачивается. Жёсткий лимит числа
живых вызовов за прогон (SPEECH_GEN_MAX_CALLS_PER_RUN) — сеть/фрагментов
может внезапно оказаться намного больше, чем ожидалось (баг сегментации,
не-финализированный сценарий), лимит не даёт прогону молча слить
непредсказуемо много кредитов ElevenLabs за один запуск.

ЧЕСТНЫЕ ГРАНИЦЫ ЭТОЙ ВЕРСИИ (не скрыто, не преувеличено):
- [slowly]/[emphasis], если они вручную расставлены в script.txt, СЕЙЧАС
  не долетают до TTS-текста — parse_blocks() схлопывает оба тега (и
  [energetic]) в нулевую паузу ДО того, как текст доходит до этого модуля
  (PAUSE_DURATIONS содержит их с длительностью 0.0), позиция и сам факт
  тега теряются безвозвратно на уровне общего парсера. Разворачивать это
  значило бы трогать parse_blocks() — общий, тяжело нагруженный код
  рендера (см. коммит с этим модулем: были live-verified эквивалентные
  попытки, обе отклонены как слишком рискованные ради редкого тега).
  [energetic] воспроизводится ОТДЕЛЬНО и детерминированно (см.
  fragment_text_for_tts) — ЧАСТЬ 10 CLAUDE.md требует его РОВНО один раз
  в начале хука, это не нужно вычитывать из script.txt позиционно.
- "Разборчивость" (intelligibility) НЕ оценивается в этой версии —
  честная оценка требует ASR (Whisper и т.п.) со сверкой текста, это
  отдельная тяжёлая опциональная зависимость, которой здесь нет. evaluate_fragment()
  возвращает intelligibility=None с явной причиной, НЕ подменяет её
  дешёвой метрикой темпа/энергии под чужим именем.
- Живой HTTP-контракт ElevenLabs (call_elevenlabs_with_timestamps)
  проверен ПРОТИВ ДОКУМЕНТИРОВАННОГО формата API и юнит-тестами с
  моком — НЕ прогнан на реальных кредитах (в этой среде разработки нет
  ELEVENLABS_API_KEY, и тратить чужие деньги на "проверку кода" — не
  та цена, которую стоит платить). Рекомендация: один дешёвый ручной
  прогон на самом коротком фрагменте перед первым использованием на
  реальном эпизоде.

Usage: python scripts/speech_generate.py <video_dir>
Вход: media_plan/speech_plan.json (см. speech_planner.py), script.txt.
Выход: audio.mp3, media_plan/alignment/NN.csv (тот же формат, что и
ручной Шаг 6 — downstream fix_pauses.py/pipeline_smart.py/
speech_validator.py потребляют его БЕЗ ИЗМЕНЕНИЙ), media_plan/final_timeline.json
(единый таймлайн: голос + куда садятся субтитры/музыка/монтаж — сводка,
не новый обязательный вход для рендера), media_plan/speech_generation_report.json
(что сгенерировано, сколько попыток, кэш-хиты, потрачено символов).

.env: ELEVENLABS_API_KEY (обязателен), ELEVENLABS_VOICE_ID (обязателен),
TTS_MODEL (опц., по умолчанию eleven_v3 — ЧАСТЬ 10 CLAUDE.md).
SPEECH_GEN_MAX_CALLS_PER_RUN (опц., умолч. 200) — жёсткий потолок числа
ЖИВЫХ (не кэш-хитов) вызовов ElevenLabs за один прогон."""
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

try:
    import numpy as np
except ImportError:
    np = None

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

VIDEO_DIR = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()

# pipeline_smart/speech_planner читают VIDEO_FOLDER/VIDEO_DIR из sys.argv[1]
# НА ИМПОРТЕ — та же подмена, что уже использует весь остальной пайплайн
# (test_parse.py, visual_qc.py, speech_planner.py сам).
_saved_argv = sys.argv
sys.argv = ["pipeline_smart.py", VIDEO_DIR]
import pipeline_smart   # noqa: E402
import speech_planner   # noqa: E402
sys.argv = _saved_argv

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(REPO_ROOT, ".env"))
except ImportError:
    pass


# --- Константы/лимиты ---
ELEVENLABS_API_BASE = "https://api.elevenlabs.io/v1"
DEFAULT_TTS_MODEL = "eleven_v3"   # ЧАСТЬ 10 CLAUDE.md — модель по умолчанию канала
# Спецификация задачи: "максимум 2 попытки" — ЖЁСТКИЙ потолок, ENV может
# только СУЗИТЬ (1), никогда не расширить сверх 2 — иначе одна опечатка в
# .env превращает "максимум 2" в "максимум сколько угодно".
MAX_ATTEMPTS_HARD_CAP = 2
SPEECH_GEN_MAX_ATTEMPTS = min(
    MAX_ATTEMPTS_HARD_CAP,
    max(1, int(os.environ.get("SPEECH_GEN_MAX_ATTEMPTS", str(MAX_ATTEMPTS_HARD_CAP)))))
SPEECH_GEN_MAX_CALLS_PER_RUN = max(1, int(os.environ.get("SPEECH_GEN_MAX_CALLS_PER_RUN", "200")))
FRAGMENT_MAX_UNITS = 6
FRAGMENT_MAX_WORDS = 45
REQUEST_TIMEOUT_SEC = 60
RETRY_BACKOFF_SEC = 2.0

# Допуски оценки (ПРИНЦИПИАЛЬНЫЕ дефолты, как и tempo_mult/energy_target в
# speech_planner.py — НЕ откалиброваны на измеренном живом ElevenLabs-выводе,
# см. докстринг модуля про честные границы).
TEMPO_TOLERANCE_RATIO = 0.30      # доля отклонения от target_wpm юнита
ENERGY_LOW_RMS_DB = -35.0         # ниже — "провал энергии" независимо от цели
JOIN_RMS_JUMP_DB = 12.0           # скачок RMS на стыке фрагментов — подозрительно резкий шов
PAUSE_BOUNDARY_TOLERANCE_SEC = 0.20

CACHE_DIR_NAME = "speech_cache"


class SpeechGenerationLimitError(Exception):
    """SPEECH_GEN_MAX_CALLS_PER_RUN исчерпан — прогон останавливается
    явно и с понятной причиной, а не тратит непредсказуемо много
    ElevenLabs-кредитов молча."""


class ElevenLabsAPIError(Exception):
    pass


# --- Фрагментация: группировка юнитов speech_plan.json в связные куски
# для ОДНОГО TTS-вызова ---

def segment_fragments(units, max_units=FRAGMENT_MAX_UNITS, max_words=FRAGMENT_MAX_WORDS):
    """Группирует units (см. speech_planner.build_units()+assign_chapter_arcs())
    в связные фрагменты. Фрагмент НИКОГДА не пересекает границу секции (та
    же граница, что media_plan/alignment/NN.csv уже использует ниже по
    пайплайну) и не смешивает две разные arc_stage — иначе один TTS-вызов
    получил бы противоречивую цель подачи (например, конец "слома" и
    начало спокойного "доказательства" в одном дубле, где нужен разный
    темп/энергия). Внутри этих границ — жадная упаковка до max_units/
    max_words, чтобы дать модели больше контекста на вызов, чем одна
    фраза (ровно то, чего просила задача)."""
    fragments = []
    cur = []
    cur_words = 0
    prev_key = None

    def flush():
        nonlocal cur, cur_words
        if cur:
            fragments.append(cur)
        cur, cur_words = [], 0

    for u in units:
        key = (u["section"], u.get("arc_stage"))
        if (key != prev_key or len(cur) >= max_units or cur_words + u["words"] > max_words):
            flush()
        cur.append(u)
        cur_words += u["words"]
        prev_key = key
    flush()
    return fragments


def fragment_id_for(fragment_units):
    return fragment_units[0]["unit_id"] + ".." + fragment_units[-1]["unit_id"]


def fragment_text_for_tts(fragment_units, prefix_energetic=False):
    """Текст, который реально уходит в TTS — озвучиваемые слова юнитов +
    восстановленные литеральные [pause]/[short pause] между ними (см.
    честные границы в докстринге модуля насчёт [slowly]/[emphasis]).
    Тег ПОСЛЕ юнита включается ВСЕГДА, когда он есть (в т.ч. у
    последнего юнита фрагмента) — это реальная пауза, которую нужно
    озвучить/получить в alignment, не только "внутренний" разделитель
    между юнитами одного вызова; на сборке она просто остаётся частью
    аудио этого фрагмента, следующий фрагмент клеится встык без
    дополнительной вставленной тишины."""
    parts = []
    if prefix_energetic:
        parts.append("[energetic]")
    for u in fragment_units:
        parts.append(u["text"])
        if u.get("tag"):
            parts.append(u["tag"])
    return " ".join(parts)


def is_first_hook_fragment(fragment_units, fragments, idx):
    """ЧАСТЬ 10 CLAUDE.md: [energetic] РОВНО один раз, в начале хука —
    детерминированно на первом фрагменте секции HOOK, независимо от
    того, что было (или не было) в оригинальном script.txt на этом
    месте (см. докстринг модуля)."""
    if not fragment_units[0]["section"].startswith("HOOK"):
        return False
    for prev in fragments[:idx]:
        if prev[0]["section"].startswith("HOOK"):
            return False
    return True


# --- Кэш: по хэшу (текст+voice+model) — одинаковый фрагмент дважды не
# оплачивается, тот же принцип атомарности, что atomic_url_download в
# pipeline_smart.py. ---

def cache_key(text, voice_id, model, attempt):
    h = hashlib.sha1(f"{text}|{voice_id}|{model}|attempt={attempt}".encode("utf-8")).hexdigest()
    return h[:24]


def cache_paths(video_dir, key):
    cache_dir = os.path.join(video_dir, "media_plan", CACHE_DIR_NAME)
    return (os.path.join(cache_dir, f"{key}.mp3"),
            os.path.join(cache_dir, f"{key}.json"))


def load_from_cache(video_dir, key):
    audio_path, meta_path = cache_paths(video_dir, key)
    if os.path.exists(audio_path) and os.path.exists(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            return audio_path, meta["alignment"]
        except Exception:
            return None
    return None


def save_to_cache(video_dir, key, audio_bytes, alignment):
    audio_path, meta_path = cache_paths(video_dir, key)
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    tmp_audio = audio_path + ".tmp"
    tmp_meta = meta_path + ".tmp"
    with open(tmp_audio, "wb") as f:
        f.write(audio_bytes)
    with open(tmp_meta, "w", encoding="utf-8") as f:
        json.dump({"alignment": alignment}, f)
    os.replace(tmp_audio, audio_path)
    os.replace(tmp_meta, meta_path)
    return audio_path


# --- ElevenLabs HTTP-клиент. Никогда не логирует/не печатает api_key —
# см. _redact() и то, что ключ читается только из os.environ и передаётся
# ТОЛЬКО в заголовке запроса, никогда в URL/логах/media_plan/*.json. ---

def _redact(text, api_key):
    if not api_key:
        return text
    return text.replace(api_key, "***REDACTED***")


def call_elevenlabs_with_timestamps(text, voice_id, api_key, model, previous_text=None, next_text=None):
    """POST /v1/text-to-speech/{voice_id}/with-timestamps — документированный
    endpoint ElevenLabs, отдаёт audio (base64) + посимвольный alignment за
    ОДИН вызов (то, что раньше человек получал вручную и клал в
    media_plan/alignment/*.csv, см. ЧАСТЬ 13 CLAUDE.md Шаг 6). previous_text/
    next_text — контекст соседних фрагментов ДЛЯ ПРОСОДИИ (уменьшает, но не
    убирает риск шва на стыке — см. докстринг модуля).
    Возвращает (audio_bytes, alignment) где alignment — список (char, start, end).
    Бросает ElevenLabsAPIError с текстом, где api_key уже вычищен, на любой
    ошибке (сеть/HTTP/формат ответа)."""
    url = f"{ELEVENLABS_API_BASE}/text-to-speech/{voice_id}/with-timestamps"
    body = {"text": text, "model_id": model}
    if previous_text:
        body["previous_text"] = previous_text
    if next_text:
        body["next_text"] = next_text
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST", headers={
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SEC) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        raise ElevenLabsAPIError(_redact(f"HTTP {e.code}: {err_body}", api_key)) from None
    except urllib.error.URLError as e:
        raise ElevenLabsAPIError(_redact(f"сетевая ошибка: {e.reason}", api_key)) from None
    except Exception as e:
        raise ElevenLabsAPIError(_redact(f"{type(e).__name__}: {e}", api_key)) from None

    try:
        import base64
        audio_bytes = base64.b64decode(payload["audio_base64"])
        al = payload["alignment"]
        chars = al["characters"]
        starts = al["character_start_times_seconds"]
        ends = al["character_end_times_seconds"]
        alignment = list(zip(chars, starts, ends))
    except (KeyError, TypeError, ValueError) as e:
        raise ElevenLabsAPIError(
            _redact(f"неожиданный формат ответа ElevenLabs: {type(e).__name__}: {e}", api_key)) from None
    return audio_bytes, alignment


def call_elevenlabs_with_retry_on_transient_error(text, voice_id, api_key, model,
                                                    previous_text=None, next_text=None, attempts=2):
    """Ретрай ТОЛЬКО транспортного/HTTP-5xx сбоя (ElevenLabs недоступен на
    секунду) — НЕ тот же ретрай, что решает decide_fragment_action() ниже
    (тот перегенерирует из-за КАЧЕСТВА речи, этот — из-за того, что запрос
    вообще не доехал). Раздельные счётчики: транспортный ретрай не тратит
    из лимита в 2 попытки по качеству."""
    last_err = None
    for i in range(attempts):
        try:
            return call_elevenlabs_with_timestamps(text, voice_id, api_key, model,
                                                     previous_text, next_text)
        except ElevenLabsAPIError as e:
            last_err = e
            if i < attempts - 1:
                time.sleep(RETRY_BACKOFF_SEC * (i + 1))
    raise last_err


# --- Локальная оценка (бесплатно, без сети): паузы + темп + энергия. Стыки
# оцениваются ОТДЕЛЬНО, после сборки (см. evaluate_joins) — их не видно,
# пока фрагмент существует в одиночестве. Разборчивость — см. докстринг
# модуля, честно не оценивается в этой версии. ---

def decode_audio_to_array(path, sr=22050):
    """ffmpeg-декод в моно float32 numpy — тот же принцип (subprocess+ffmpeg,
    без новых тяжёлых аудио-библиотек), что уже использует audio_qc() в
    pipeline_smart.py, только с сырыми сэмплами вместо текстового вывода
    astats — нужна огибающая ПО ВРЕМЕНИ (энергия), не один агрегат на
    весь файл."""
    if np is None:
        return None, sr
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"]
    r = subprocess.run(cmd, capture_output=True, timeout=60)
    if r.returncode != 0 or not r.stdout:
        return None, sr
    arr = np.frombuffer(r.stdout, dtype=np.float32)
    return arr, sr


def rms_envelope_db(arr, sr, window_sec=0.25):
    """RMS в dBFS по окнам window_sec — огибающая энергии по времени."""
    if arr is None or len(arr) == 0:
        return np.array([]) if np is not None else []
    win = max(1, int(sr * window_sec))
    n_windows = max(1, len(arr) // win)
    trimmed = arr[:n_windows * win]
    windows = trimmed.reshape(n_windows, win)
    rms = np.sqrt(np.mean(windows.astype(np.float64) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(np.maximum(rms, 1e-6))
    return db


def _unit_char_spans(fragment_units, fragment_text):
    """Символьный диапазон [start_char, end_char) КАЖДОГО unit в
    fragment_text (см. fragment_text_for_tts) — нужен, чтобы сопоставить
    юнит с его срезом alignment. Строится той же конкатенацией, что и
    fragment_text_for_tts(), НЕ повторным поиском текста в строке (поиск
    подстрокой ломался бы на повторяющихся словах)."""
    spans = []
    pos = 0
    for i, u in enumerate(fragment_units):
        if i > 0:
            pos += 1   # разделяющий пробел
        start = pos
        pos += len(u["text"])
        end = pos
        spans.append((start, end))
        if u.get("tag"):
            pos += 1 + len(u["tag"])
    return spans


def evaluate_fragment(fragment_units, fragment_text, audio_path, alignment):
    """Возвращает dict с оценкой фрагмента: per-unit паузы (сверка с
    target_range_sec из speech_plan), темп (слов/мин из alignment против
    target_wpm), энергия (RMS-огибающая против energy_target). Не сетевой
    вызов — работает с уже полученными audio_path/alignment."""
    full_text_from_alignment = "".join(c for c, s, e in alignment)
    spans = _unit_char_spans(fragment_units, fragment_text)

    unit_evals = []
    for u, (cs, ce) in zip(fragment_units, spans):
        # Защита от рассинхрона alignment vs наш реконструированный текст
        # (ElevenLabs теоретически может вернуть alignment другой длины,
        # напр. если модель что-то нормализовала) — не падаем, честно
        # помечаем unit как "нет сигнала" вместо угадывания по смещённым
        # индексам.
        if ce > len(alignment):
            unit_evals.append({"unit_id": u["unit_id"], "signal": False,
                               "reason": "alignment короче реконструированного текста"})
            continue
        seg = alignment[cs:ce]
        real = [(c, s, e) for c, s, e in seg if re.match(r'[^\s\[\]]', c or "")]
        if not real:
            unit_evals.append({"unit_id": u["unit_id"], "signal": False, "reason": "пустой сегмент"})
            continue
        spoken_start, spoken_end = real[0][1], real[-1][2]
        spoken_dur = max(1e-6, spoken_end - spoken_start)
        wpm = u["words"] / spoken_dur * 60.0
        target_wpm = u.get("target_wpm", speech_planner.BASE_WPM)
        tempo_ratio = wpm / target_wpm if target_wpm else 1.0
        tempo_ok = abs(tempo_ratio - 1.0) <= TEMPO_TOLERANCE_RATIO

        # Пауза ПОСЛЕ юнита (внутри фрагмента, если не последний — берём
        # разрыв до начала следующего unit'а alignment; для последнего
        # юнита фрагмента пауза оценивается на границе стыка отдельно,
        # см. evaluate_joins).
        pause_ok = True
        observed_pause = None
        rng = u.get("target_range_sec")
        if rng and u is not fragment_units[-1]:
            next_idx = fragment_units.index(u) + 1
            next_cs = spans[next_idx][0]
            next_seg = alignment[next_cs:spans[next_idx][1]] if next_cs < len(alignment) else []
            next_real = [(c, s, e) for c, s, e in next_seg if re.match(r'[^\s\[\]]', c or "")]
            if next_real:
                observed_pause = next_real[0][1] - spoken_end
                pause_ok = (rng[0] - PAUSE_BOUNDARY_TOLERANCE_SEC <= observed_pause
                            <= rng[1] + PAUSE_BOUNDARY_TOLERANCE_SEC)

        unit_evals.append({
            "unit_id": u["unit_id"], "signal": True,
            "wpm": round(wpm, 1), "target_wpm": target_wpm,
            "tempo_ratio": round(tempo_ratio, 3), "tempo_ok": tempo_ok,
            "observed_pause_sec": round(observed_pause, 3) if observed_pause is not None else None,
            "target_pause_range_sec": rng, "pause_ok": pause_ok,
        })

    energy_eval = {"scored": False, "reason": "numpy недоступен"}
    if np is not None and os.path.exists(audio_path):
        arr, sr = decode_audio_to_array(audio_path)
        if arr is not None and len(arr) > 0:
            db = rms_envelope_db(arr, sr)
            mean_db = float(np.mean(db)) if len(db) else None
            peak_db = float(np.max(db)) if len(db) else None
            energy_ok = mean_db is not None and mean_db > ENERGY_LOW_RMS_DB
            energy_eval = {"scored": True, "mean_rms_db": round(mean_db, 1) if mean_db is not None else None,
                          "peak_rms_db": round(peak_db, 1) if peak_db is not None else None,
                          "energy_ok": energy_ok}
        else:
            energy_eval = {"scored": False, "reason": "не удалось декодировать аудио"}

    return {
        "units": unit_evals,
        "energy": energy_eval,
        "intelligibility": {"scored": False, "reason": "требует ASR — не подключён в этой версии (см. докстринг модуля)"},
    }


def evaluate_joins(assembled_fragments_audio):
    """Оценка ШВОВ между УЖЕ принятыми соседними фрагментами (вызывается
    на сборке — до неё швов не существует). Резкий скачок RMS на границе
    (последнее окно фрагмента N против первого окна фрагмента N+1) —
    грубый, но честный, бесплатный сигнал "здесь может быть заметный
    шов". Не гарантия неслышимости шва человеком (см. докстринг модуля)."""
    joins = []
    if np is None:
        return joins
    for i in range(len(assembled_fragments_audio) - 1):
        path_a = assembled_fragments_audio[i]
        path_b = assembled_fragments_audio[i + 1]
        arr_a, sr = decode_audio_to_array(path_a)
        arr_b, _ = decode_audio_to_array(path_b)
        if arr_a is None or arr_b is None or len(arr_a) == 0 or len(arr_b) == 0:
            joins.append({"index": i, "signal": False})
            continue
        tail = arr_a[-int(sr * 0.15):]
        head = arr_b[:int(sr * 0.15)]
        tail_db = 20 * math.log10(max(1e-6, math.sqrt(float(np.mean(tail.astype(np.float64) ** 2)))))
        head_db = 20 * math.log10(max(1e-6, math.sqrt(float(np.mean(head.astype(np.float64) ** 2)))))
        jump = abs(tail_db - head_db)
        joins.append({"index": i, "signal": True, "tail_db": round(tail_db, 1),
                      "head_db": round(head_db, 1), "jump_db": round(jump, 1),
                      "join_ok": jump <= JOIN_RMS_JUMP_DB})
    return joins


# --- Решение: принять / исправить только границу паузы / перегенерировать.
# Максимум SPEECH_GEN_MAX_ATTEMPTS попыток (жёстко <=2) — после исчерпания
# ПРИНИМАЕТСЯ лучшая из попыток с явным предупреждением, не бесконечный
# цикл (тот же принцип "локализовать сбой, не тратить деньги без конца",
# что и в отказоустойчивом рендере клипов, см. pipeline_smart.py). ---

def decide_fragment_action(evaluation, attempt):
    """Возвращает ("accept"|"fix_boundary"|"regenerate", reason)."""
    units = evaluation["units"]
    signal_units = [u for u in units if u.get("signal")]
    if not signal_units:
        return ("regenerate" if attempt < SPEECH_GEN_MAX_ATTEMPTS else "accept",
                "нет сигнала alignment ни на одном юните")

    tempo_bad = [u for u in signal_units if not u["tempo_ok"]]
    pause_bad = [u for u in signal_units if not u.get("pause_ok", True)]
    energy_ok = evaluation["energy"].get("energy_ok", True) if evaluation["energy"].get("scored") else True

    if not tempo_bad and not pause_bad and energy_ok:
        return "accept", "темп/паузы/энергия в допуске"

    if not tempo_bad and energy_ok and pause_bad:
        # ТОЛЬКО пауза не в допуске — дешёвый путь: не тратим вторую
        # генерацию, отдаём границу на исправление тем же механизмом,
        # что уже правит паузы (fix_pauses.py/protected_windows), не
        # перегенерируем ради тишины, которую можно просто подрезать.
        return "fix_boundary", f"{len(pause_bad)} пауз(ы) вне диапазона, темп/энергия в норме"

    if attempt < SPEECH_GEN_MAX_ATTEMPTS:
        reasons = []
        if tempo_bad:
            reasons.append(f"{len(tempo_bad)} юнит(ов) вне темпа")
        if not energy_ok:
            reasons.append("энергия ниже порога")
        return "regenerate", "; ".join(reasons)

    return "accept", f"лимит попыток ({SPEECH_GEN_MAX_ATTEMPTS}) исчерпан — принимаю лучшую попытку с предупреждением"


def _eval_problem_count(evaluation):
    """Меньше — лучше. Используется ТОЛЬКО чтобы выбрать "менее плохую"
    попытку после исчерпания лимита регенераций — не влияет на решение
    accept/fix_boundary/regenerate само по себе (то принимает
    decide_fragment_action по конкретным порогам, не по этому счётчику)."""
    units = evaluation["units"]
    n = sum(1 for u in units if u.get("signal") and not u.get("tempo_ok", True))
    n += sum(1 for u in units if u.get("signal") and not u.get("pause_ok", True))
    if evaluation["energy"].get("scored") and not evaluation["energy"].get("energy_ok", True):
        n += 1
    n += sum(1 for u in units if not u.get("signal"))
    return n


# --- Оркестрация: кэш -> генерация с ретраями -> оценка -> решение, для
# каждого фрагмента по очереди. "fix_boundary" НЕ реализует коррекцию
# паузы САМ — эта коррекция уже существует (speech_validator.py считает
# protected_windows из ЛЮБОГО audio+alignment против того же
# speech_plan.json, fix_pauses.py их применяет, см. ЧАСТЬ 13 CLAUDE.md
# Шаги 6.5/7) — fix_boundary здесь означает "эта попытка принимается,
# регенерация не нужна, а точную границу паузы доведёт до ума уже
# существующий последующий шаг", не дублирует его логику заново. ---

def generate_all_fragments(video_dir, fragments, api_key, voice_id, model, on_progress=None):
    results = []
    live_calls = 0
    cache_hits = 0
    for idx, frag_units in enumerate(fragments):
        prefix_energetic = is_first_hook_fragment(frag_units, fragments, idx)
        text = fragment_text_for_tts(frag_units, prefix_energetic=prefix_energetic)
        prev_text = fragment_text_for_tts(fragments[idx - 1])[-200:] if idx > 0 else None
        next_text = fragment_text_for_tts(fragments[idx + 1])[:200] if idx + 1 < len(fragments) else None

        decision_log = []
        best = None
        attempt = 1
        chosen = None
        while True:
            key = cache_key(text, voice_id, model, attempt)
            cached = load_from_cache(video_dir, key)
            from_cache = cached is not None
            if cached:
                audio_path, alignment = cached
                cache_hits += 1
            else:
                if live_calls >= SPEECH_GEN_MAX_CALLS_PER_RUN:
                    raise SpeechGenerationLimitError(
                        f"SPEECH_GEN_MAX_CALLS_PER_RUN={SPEECH_GEN_MAX_CALLS_PER_RUN} исчерпан "
                        f"на фрагменте {fragment_id_for(frag_units)} (попытка {attempt}); "
                        f"уже сгенерированные фрагменты остались в кэше — повторный "
                        f"запуск переиспользует их и продолжит с этого места")
                audio_bytes, alignment = call_elevenlabs_with_retry_on_transient_error(
                    text, voice_id, api_key, model, previous_text=prev_text, next_text=next_text)
                live_calls += 1
                audio_path = save_to_cache(video_dir, key, audio_bytes, alignment)

            evaluation = evaluate_fragment(frag_units, text, audio_path, alignment)
            action, reason = decide_fragment_action(evaluation, attempt)
            decision_log.append({"attempt": attempt, "from_cache": from_cache,
                                 "action": action, "reason": reason})
            if best is None or _eval_problem_count(evaluation) < _eval_problem_count(best[2]):
                best = (audio_path, alignment, evaluation)

            if action in ("accept", "fix_boundary"):
                chosen = (audio_path, alignment, evaluation)
                break
            if attempt >= SPEECH_GEN_MAX_ATTEMPTS:
                chosen = best
                break
            attempt += 1

        if on_progress:
            on_progress(idx + 1, len(fragments), fragment_id_for(frag_units), decision_log[-1])

        results.append({
            "fragment_id": fragment_id_for(frag_units), "units": frag_units, "text": text,
            "audio_path": chosen[0], "alignment": chosen[1], "evaluation": chosen[2],
            "decision_log": decision_log, "attempts": attempt,
        })
    return results, live_calls, cache_hits


# --- Сборка: конкатенация audio.mp3, восстановление per-секционных
# alignment/NN.csv (тот же формат char,start,end, что и ручной Шаг 6 —
# downstream БЕЗ ИЗМЕНЕНИЙ), единый final_timeline.json. ---

def concat_fragment_audio(fragment_results, out_path, temp_dir):
    """ffmpeg concat demuxer (stream copy — все фрагменты из одного
    аккаунта/модели ElevenLabs, формат совпадает) с откатом на re-encode
    concat при несовпадении форматов — тот же принцип отказоустойчивости,
    что xfade_chain_chunked() в pipeline_smart.py уже применяет к видео."""
    concat_list = os.path.join(temp_dir, "speech_concat.txt")
    with open(concat_list, "w", encoding="utf-8") as f:
        for r in fragment_results:
            f.write(f"file '{os.path.abspath(r['audio_path'])}'\n")
    tmp_out = out_path + ".tmp.mp3"
    r = subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list,
                        "-c", "copy", tmp_out], capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        r2 = subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list,
                             "-c:a", "libmp3lame", "-q:a", "2", tmp_out],
                            capture_output=True, text=True, timeout=180)
        if r2.returncode != 0:
            raise RuntimeError(f"склейка аудио не удалась (copy и re-encode): {r2.stderr[-300:]}")
    os.replace(tmp_out, out_path)


def write_section_alignment_csvs(fragment_results, alignment_dir):
    """Формат — ТОТ ЖЕ, что ручной Шаг 6 (см. load_alignment_weights() в
    pipeline_smart.py: 'char,start,end', по файлу на секцию в порядке
    первого появления, время — ЛОКАЛЬНОЕ для каждой секции (см. докстринг
    модуля про то, почему это осознанно, не глобальный сдвиг по всему
    audio.mp3 — так уже устроен весь downstream, включая speech_validator.py's
    'нет сигнала' на границах секций)."""
    os.makedirs(alignment_dir, exist_ok=True)
    section_order = []
    section_fragments = {}
    for r in fragment_results:
        sec = r["units"][0]["section"]
        if sec not in section_fragments:
            section_order.append(sec)
            section_fragments[sec] = []
        section_fragments[sec].append(r)

    for i, sec in enumerate(section_order):
        rows = []
        local_offset = 0.0
        for r in section_fragments[sec]:
            for char, start, end in r["alignment"]:
                rows.append((char, round(start + local_offset, 5), round(end + local_offset, 5)))
            frag_dur = pipeline_smart.get_media_duration(r["audio_path"])
            local_offset += frag_dur
        path = os.path.join(alignment_dir, f"{i:02d}.csv")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["char", "start", "end"])
            w.writerows(rows)
        os.replace(tmp, path)


def build_final_timeline(fragment_results, join_evals):
    """media_plan/final_timeline.json — единая сводка (голос + куда садятся
    субтитры/музыка/монтаж), собранная из уже посчитанных данных: real
    время каждого юнита В ГЛОБАЛЬНОМ (по всему собранному audio.mp3)
    времени + глава/риторическая роль/цель темпа-энергии + фактический
    результат оценки + попытки/кэш. ДОПОЛНИТЕЛЬНЫЙ артефакт для
    человека/будущих потребителей — рендер по-прежнему читает
    audio.mp3+alignment/*.csv+pause_cuts.json напрямую, этот файл ничего
    из существующего пути не заменяет и не обязателен для него."""
    entries = []
    global_offset = 0.0
    for r in fragment_results:
        frag_dur = pipeline_smart.get_media_duration(r["audio_path"])
        for u_eval in r["evaluation"]["units"]:
            u = next(uu for uu in r["units"] if uu["unit_id"] == u_eval["unit_id"])
            entries.append({
                "unit_id": u["unit_id"], "section": u["section"],
                "chapter_id": u.get("chapter_id"), "arc_stage": u.get("arc_stage"),
                "rhetorical_kind": u.get("rhetorical_kind"),
                "target_wpm": u.get("target_wpm"), "observed_wpm": u_eval.get("wpm"),
                "tempo_ok": u_eval.get("tempo_ok"),
                "target_pause_range_sec": u_eval.get("target_pause_range_sec"),
                "observed_pause_sec": u_eval.get("observed_pause_sec"),
                "pause_ok": u_eval.get("pause_ok"),
                "fragment_id": r["fragment_id"], "attempts": r["attempts"],
            })
        global_offset += frag_dur
    return {"units": entries, "joins": join_evals, "total_duration_sec": round(global_offset, 3)}


def atomic_write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main():
    video_dir = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()

    api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "").strip()
    if not api_key or not voice_id:
        print("Speech Generate: нужны ELEVENLABS_API_KEY и ELEVENLABS_VOICE_ID в .env "
              "(защита ключа: читается только из окружения, никогда не печатается/не "
              "пишется в media_plan/*.json) — не запускаю.")
        return 1
    model = os.environ.get("TTS_MODEL", "").strip() or DEFAULT_TTS_MODEL

    audio_out = os.path.join(video_dir, "audio.mp3")
    if os.path.exists(audio_out):
        print(f"Speech Generate: {audio_out} уже существует — не перезаписываю молча "
              f"(могла быть уже готовая ручная озвучка). Удали/переименуй файл, "
              f"если действительно нужна автогенерация поверх.")
        return 1

    plan_path = os.path.join(video_dir, "media_plan", "speech_plan.json")
    if not os.path.exists(plan_path):
        print(f"Speech Generate: {plan_path} не найден — сначала запусти "
              f"scripts/speech_planner.py {video_dir} (Stage A, бесплатно).")
        return 1
    with open(plan_path, encoding="utf-8") as f:
        plan = json.load(f)
    try:
        speech_planner.validate_plan(plan)
    except ValueError as e:
        print(f"Speech Generate: speech_plan.json не прошёл валидацию — {e}")
        return 1
    units = plan["units"]
    if not units or "arc_stage" not in units[0]:
        print("Speech Generate: speech_plan.json без chapter/arc-разметки — "
              "перезапусти scripts/speech_planner.py (текущая версия), это Stage B "
              "требует план по главам, не только по юнитам.")
        return 1

    fragments = segment_fragments(units)
    total_chars = sum(len(fragment_text_for_tts(f)) for f in fragments)
    print(f"Speech Generate: {len(units)} юнитов -> {len(fragments)} связных фрагментов "
          f"(~{total_chars} символов на озвучку). Лимит живых вызовов за прогон: "
          f"{SPEECH_GEN_MAX_CALLS_PER_RUN}, лимит попыток на фрагмент: {SPEECH_GEN_MAX_ATTEMPTS}.")

    def on_progress(i, n, frag_id, last_decision):
        print(f"  [{i}/{n}] {frag_id}: {last_decision['action']} "
              f"({'кэш' if last_decision['from_cache'] else 'live'}) — {last_decision['reason']}")

    try:
        fragment_results, live_calls, cache_hits = generate_all_fragments(
            video_dir, fragments, api_key, voice_id, model, on_progress=on_progress)
    except SpeechGenerationLimitError as e:
        print(f"Speech Generate: СТОП — {e}")
        return 1
    except ElevenLabsAPIError as e:
        print(f"Speech Generate: ElevenLabs API — {e}")
        return 1

    temp_dir = os.path.join(video_dir, "media_plan", "speech_temp")
    os.makedirs(temp_dir, exist_ok=True)
    try:
        concat_fragment_audio(fragment_results, audio_out, temp_dir)
    except Exception as e:
        print(f"Speech Generate: сборка audio.mp3 не удалась — {e}")
        return 1

    alignment_dir = os.path.join(video_dir, "media_plan", "alignment")
    write_section_alignment_csvs(fragment_results, alignment_dir)

    join_evals = evaluate_joins([r["audio_path"] for r in fragment_results])
    timeline = build_final_timeline(fragment_results, join_evals)
    atomic_write_json(os.path.join(video_dir, "media_plan", "final_timeline.json"), timeline)

    exhausted = [r for r in fragment_results
                if r["decision_log"][-1]["action"] == "accept"
                and "лимит попыток" in r["decision_log"][-1]["reason"]]
    bad_joins = [j for j in join_evals if j.get("signal") and not j.get("join_ok", True)]
    report = {
        "fragments": len(fragment_results), "live_calls": live_calls, "cache_hits": cache_hits,
        "total_chars_sent": total_chars, "model": model,
        "exhausted_attempts_fragments": [r["fragment_id"] for r in exhausted],
        "suspicious_joins": bad_joins,
    }
    atomic_write_json(os.path.join(video_dir, "media_plan", "speech_generation_report.json"), report)

    print(f"\nГенерация: {live_calls} живых вызовов, {cache_hits} из кэша, "
          f"~{total_chars} символов отправлено.")
    if exhausted:
        print(f"  ВНИМАНИЕ: {len(exhausted)} фрагмент(ов) приняты после исчерпания лимита "
              f"попыток ({SPEECH_GEN_MAX_ATTEMPTS}) — стоит переслушать: "
              + ", ".join(r["fragment_id"] for r in exhausted))
    if bad_joins:
        print(f"  ВНИМАНИЕ: {len(bad_joins)} подозрительных стыков между фрагментами "
              f"(резкий перепад RMS) — стоит переслушать границы.")
    print("  Дальше по протоколу: speech_validator.py (Шаг 6.5) -> fix_pauses.py (Шаг 7) "
          "— та же цепочка, что и после ручной озвучки, без изменений.")
    print(f"Готово: {audio_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
