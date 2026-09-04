#!/usr/bin/env python3
"""Lumean — основной путь озвучки (ЧАСТЬ 13, Шаг 6), вместо ручного/прямого
ElevenLabs. Один заказ (POST /orders) на КАЖДУЮ секцию script.txt (HOOK,
BLOCK N, FINAL) по готовому шаблону (голос уже зашит в шаблон — у Lumean
нет orders.voice_id, см. Lumean Public API §1/§13), опрос до готовности,
скачивание результата, склейка всех секций в один audio.mp3 — тот же
итоговый файл, что раньше делался вручную через веб ElevenLabs.

ПОЧЕМУ ПО СЕКЦИЯМ, А НЕ ПО МЕЛКИМ ФРАГМЕНТАМ (как scripts/speech_generate.py
для прямого ElevenLabs): та схема дробит текст на связные фрагменты (6 юнитов/
45 слов) ради previous_text/next_text — параметров ElevenLabs, которых нет в
публичном контуре Lumean заказов (§8.0 — полный список полей POST /orders,
previous_text/next_text там нет). Взамен Lumean сам режет заказ на чанки и
сшивает результат в ОДНУ дорожку на сервере (§5: "1 (склеенная дорожка)") —
секция целиком как единица заказа не теряет ничего, что реально доступно по
API, и естественно ложится на существующую по-секционную схему alignment/
section_offsets (см. ниже).

ЧЕСТНЫЕ ГРАНИЦЫ ЭТОЙ ВЕРСИИ (не скрыто, не преувеличено):
- Голос НЕ подбирается автоматически. CHANNEL.md сам требует ручного выбора
  голоса на слух ("ищи спокойный низкий мужской голос... впиши после
  выбора") — то же самое здесь: --list-voices печатает кандидатов с
  preview_url, голос слушает и выбирает человек, шаблон создаётся ОДИН РАЗ
  (см. --create-template), UUID уходит в LUMEAN_TEMPLATE_ID .env. Без него
  генерация не запускается — гадать чужим фирменным голосом канала нельзя.
- Посимвольный alignment.csv — ЛУЧШЕЕ УСИЛИЕ (best-effort), не гарантия.
  Формат сервисного файла alignment.*/result.json в публичной спеке Lumean
  описан только функционально ("пословное/посимвольное выравнивание"), без
  точной JSON-схемы. Пробуем распознать ТРИ вероятные формы (нативную форму
  ElevenLabs with-timestamps — её уже парсит speech_generate.py напрямую —
  плоский список {character,start,end}, и — найдено живым прогоном на
  реальном аккаунте, не гипотеза — готовый CSV-текст с заголовком
  "index,char,start,end"); ЛЮБАЯ форма, не прошедшая строгую проверку
  структуры, отбрасывается целиком для этой секции — НЕ подсовывается как
  правдоподобная. Даунстрим-код (pipeline_smart.py) уже штатно откатывается
  на word-count тайминг для секций без alignment.csv — тот же путь, что и у
  ручной озвучки без сохранённого alignment, не деградация.
- Фрагментный quality-gated retry (темп/энергия/паузы, есть у прямого
  ElevenLabs-пути) здесь не реализован — Lumean-регенерация чанка ПЛАТНАЯ
  (§7.11: "regenerate — платно, precheck -> 429"), автоматический цикл
  "не понравилось - попробуй ещё" на этом контуре стоил бы реальных денег
  на каждой итерации без разрешения пользователя. Технический сбой чанка
  (failed) обрабатывается один раз бесплатно (POST .../items/retry-failed);
  контентный отказ (policy_flagged) требует правки текста человеком — см.
  playbook в самом отчёте прогона.
- 402 payg_topup_required (не хватает квоты подписки) НЕ подтверждается
  автоматически — реальные деньги. Секция останавливается с ценой доплаты,
  повторный запуск с --confirm-payg подтверждает явно (Рецепт G спеки).

Usage:
  python scripts/lumean_tts.py --list-voices [поисковый запрос] [--gender male|female]
  python scripts/lumean_tts.py --create-template <VOICE_ID> [--name "Имя шаблона"]
  python scripts/lumean_tts.py <video_dir> [T_минут] [--force-length] [--confirm-payg]

.env: LUMEAN_API_KEY (обязателен для всех режимов), LUMEAN_TEMPLATE_ID
(обязателен для генерации — создаётся один раз через --create-template).

Вход: script.txt.
Выход: audio.mp3, media_plan/alignment/NN.csv (best-effort, ТОТ ЖЕ формат,
что и Шаг 6/speech_generate.py — downstream fix_pauses.py/pipeline_smart.py
потребляют его без изменений), media_plan/section_offsets.json,
media_plan/lumean_generation_report.json."""
import csv
import io
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
ENV_PATH = os.path.join(REPO_ROOT, ".env")

try:
    from dotenv import load_dotenv
    load_dotenv(ENV_PATH)
except ImportError:
    pass

import wordcount  # noqa: E402  (clean_words — тот же счётчик, что ЧАСТЬ 9 CLAUDE.md)

BASE = "https://api.lumean.app/api/public"
DEFAULT_MODEL_ID = "eleven_v3"      # ЧАСТЬ 10 CLAUDE.md — теги [pause]/[energetic]/... это теги v3
DEFAULT_LANGUAGE_CODE = "ru"
REQUEST_TIMEOUT_SEC = 30
DOWNLOAD_TIMEOUT_SEC = 120
POLL_INTERVAL_SEC = 4.0
POLL_TIMEOUT_SEC = 20 * 60   # на секцию; секции этого пайплайна короткие (минуты аудио)
TERMINAL_STATUSES = {"completed", "result_delivered", "failed", "compensated", "cancelled"}
WPM = wordcount.WPM   # 125 слов/мин — ЧАСТЬ 9 CLAUDE.md, одна база на весь пайплайн

SECTION_SPLIT_RE = re.compile(r'===\s*(.*?)\s*===')   # ТОТ ЖЕ regex, что script_parser.parse_blocks()


class LumeanError(Exception):
    """Доменная ошибка Lumean с машинным reason (если он был в теле ответа)."""

    def __init__(self, message, status=None, reason=None, body=None):
        super().__init__(message)
        self.status = status
        self.reason = reason
        self.body = body or {}


class LumeanPaymentRequired(LumeanError):
    pass


# ---------- текст: секции script.txt для TTS (теги ОСТАЮТСЯ, не как script_parser) ----------

def extract_section_texts(path):
    """[(SECTION_NAME, text), ...] — только HOOK/BLOCK*/FINAL, в порядке
    первого появления. SECTION_NAME — ПОЛНОСТЬЮ УППЕРКЕЙСНЫЙ заголовок,
    БИТ-В-БИТ то же значение, что кладёт в b["section"] script_parser.
    parse_blocks() (та же regex, тот же .upper() без .strip() — раздельное
    совпадение имени секции критично: downstream section_offsets.json/
    alignment/NN.csv сопоставляются со скриптом ИМЕННО по этой строке).

    В отличие от parse_blocks() (тот стирает [...] теги ради видео-тайминга
    и словосчёта), здесь теги ОСТАЮТСЯ буква-в-букву — это теги ElevenLabs
    v3 (ЧАСТЬ 10 CLAUDE.md), TTS должен их увидеть, как увидел бы при ручном
    копировании текста в веб ElevenLabs. Вырезаются только ДВА
    пайплайн-only маркера, которых TTS никогда не должен произнести:
    [stat:...] (цифра-плашка на экран) и [climax] (сигнал для музыки/пауз,
    добавляется отдельно scripts/script_parser.py при разборе)."""
    raw = open(path, encoding="utf-8").read()
    parts = SECTION_SPLIT_RE.split(raw)
    out = []
    for i in range(1, len(parts), 2):
        name = parts[i].upper()
        if not name.startswith(("HOOK", "BLOCK", "FINAL")):
            continue
        body = parts[i + 1] if i + 1 < len(parts) else ""
        body = re.sub(r'\[stat:.*?\]', ' ', body)
        body = body.replace("[climax]", " ")
        body = re.sub(r'[ \t]+', ' ', body).strip()
        if body:
            out.append((name, body))
    return out


def wordcount_report(section_texts):
    """(всего_слов, {секция: слов}) — тот же clean_words, что wordcount.py
    (ЧАСТЬ 9: теги -> пробел, не пустота, иначе соседние слова слипаются и
    счётчик занижает длину — критический класс отказа из ЧАСТИ 1)."""
    per_section = {name: wordcount.clean_words(text) for name, text in section_texts}
    return sum(per_section.values()), per_section


def enforce_length_gate(total_words, target_minutes, force=False):
    """Тот же коридор, что wordcount.py (0.95x..1.07x от T*WPM), но здесь —
    ЖЁСТКИЙ гейт перед платным вызовом (ЧАСТЬ 1: "перед любым платным
    действием проверь дважды"), не просто консольная подсказка: без
    --force выход за коридор останавливает прогон ДО первого заказа."""
    if not target_minutes:
        print(f"  (T не задан — пропускаю проверку коридора длины; слов: {total_words})")
        return True
    lo, hi = target_minutes * WPM * 0.95, target_minutes * WPM * 1.07
    if lo <= total_words <= hi:
        print(f"  Длина в коридоре: {total_words} слов (цель {target_minutes:g} мин -> {lo:.0f}-{hi:.0f}).")
        return True
    direction = "МАЛО" if total_words < lo else "МНОГО"
    print(f"  ВНИМАНИЕ: {direction} слов — {total_words}, коридор {lo:.0f}-{hi:.0f} "
          f"(цель {target_minutes:g} мин).")
    if force:
        print("  --force-length: продолжаю вопреки коридору (осознанное решение).")
        return True
    print("  СТОП: сценарий не в коридоре длины — не трачу заказы Lumean на "
          "недописанный/раздутый текст. Правь текст или перезапусти с --force-length.")
    return False


# ---------- HTTP ----------

def _http(method, path, api_key, body=None, timeout=REQUEST_TIMEOUT_SEC):
    """Один запрос к Lumean. Возвращает (status_code, parsed_json_or_None).
    Не ретраит сама — решение о повторе принимает вызывающий код (разные
    случаи 429 требуют разного лечения, см. Lumean Public API §6)."""
    url = BASE + path
    headers = {"X-API-KEY": api_key}
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            payload = json.loads(raw) if raw else None
        except Exception:
            payload = None
        return e.code, payload
    except urllib.error.URLError as e:
        raise LumeanError(f"сетевая ошибка: {e.reason}") from None


def api_call(method, path, api_key, body=None, max_429_retries=3):
    """_http() + машинная обработка кодов по Lumean Public API §12:
    - 402 payg_topup_required -> LumeanPaymentRequired (реальные деньги,
      решение только у вызывающего кода/пользователя, не ретраится вслепую);
    - 429 token_quota_exceeded -> спит Retry-After и повторяет (ограниченно);
      429 без reason (rate-limit) -> экспоненциальный backoff (в теле нет
      Retry-After, см. §6.1) и повтор;
    - остальные некритичные для нас коды (403/404/409/422/5xx) -> LumeanError
      с телом ответа для диагностики, не глотаются молча."""
    attempt = 0
    while True:
        status, payload = _http(method, path, api_key, body=body)
        if status in (200, 201, 202):
            # Конверт успеха — {success, message, data} (ЧАСТЬ §4 спеки).
            # РЕАЛЬНЫЙ БАГ, пойманный живым вызовом --list-voices (не по
            # догадке — voices/elevenlabs/library вернул total>0, но пустой
            # список на выходе этой функции): все call sites этого модуля
            # написаны в расчёте на УЖЕ развёрнутый payload (order.get("id"),
            # data.get("url") и т.п.), а _http() возвращал ВЕСЬ конверт
            # целиком. Разворачиваем здесь один раз — единая точка, не по
            # call site'у, ошибиться негде. Ошибочные ответы (см. ветки
            # ниже) конверта "data" НЕ несут — там плоский {success:false,
            # message, reason,...} по построению (§12), разворачивать нечего.
            return (payload or {}).get("data")
        payload = payload or {}
        reason = payload.get("reason")
        message = payload.get("message", f"HTTP {status}")
        if status == 402 and reason == "payg_topup_required":
            raise LumeanPaymentRequired(message, status=status, reason=reason, body=payload)
        if status == 429:
            attempt += 1
            if attempt > max_429_retries:
                raise LumeanError(f"429 после {max_429_retries} повторов: {message}",
                                  status=status, reason=reason, body=payload)
            retry_after = payload.get("retry_after")
            wait = float(retry_after) if retry_after else min(30.0, 2.0 ** attempt)
            print(f"    429 ({reason or 'rate-limit'}) — жду {wait:.0f}с и повторяю "
                  f"({attempt}/{max_429_retries})...")
            time.sleep(wait)
            continue
        raise LumeanError(f"{method} {path} -> {status}: {message}",
                          status=status, reason=reason, body=payload)


def download(url, dest, timeout=DOWNLOAD_TIMEOUT_SEC):
    """Атомарная запись (tmp -> os.replace) — тот же принцип, что
    pipeline_smart.atomic_url_download()/stock_fetch_multisource._download():
    обрыв сети не должен оставлять обрезанный файл под финальным именем."""
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "lumean-tts-integration/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
        if not data:
            raise IOError("скачан 0-байтный файл")
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, dest)
    except Exception:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


# ---------- голоса / шаблон (ручной выбор — см. докстринг модуля) ----------

def list_elevenlabs_voices(api_key, search=None, gender=None, page=0, page_size=20):
    q = {"page": page, "page_size": page_size}
    if search:
        q["search"] = search
    if gender:
        q["gender"] = gender
    data = api_call("GET", f"/voices/elevenlabs/library?{urllib.parse.urlencode(q)}", api_key)
    return (data or {}).get("voices") or []


def create_elevenlabs_template(api_key, voice_id, name, language_code=DEFAULT_LANGUAGE_CODE,
                                model_id=DEFAULT_MODEL_ID):
    """Минимальный валидный config для сервиса elevenlabs — те же поля и
    значения, что в собственном примере спеки (§1 TL;DR / §8.2): stability
    обязателен, остальное — тонкая настройка под advanced_voice_settings.
    stability=0.5 валиден и на v3-семействе (там допустимы только 0.0/0.5/1.0,
    см. §8.2), и на прочих моделях (диапазон 0..1) — единое безопасное значение."""
    body = {
        "service_key": "elevenlabs",
        "name": name,
        "config": {
            "tts_settings": {
                "mode": "mode_v1",
                "model_id": model_id,
                "voice_id": voice_id,
                "language_code": language_code,
                "advanced_voice_settings": True,
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                    "use_speaker_boost": True,
                    "speed": 1.0,
                },
            }
        },
    }
    data = api_call("POST", "/templates", api_key, body=body)
    return (data or {}).get("id")


def persist_env_value(key, value):
    """Дописывает/обновляет KEY=value в .env на месте (тот же однократный
    bootstrap-паттерн, что PART 0 CLAUDE.md уже делает для остальных
    ключей) — без этого шаблон, созданный --create-template, пришлось бы
    вписывать в .env руками каждый раз."""
    if not os.path.exists(ENV_PATH):
        return False
    lines = open(ENV_PATH, encoding="utf-8").read().splitlines()
    pattern = re.compile(rf'^{re.escape(key)}=')
    found = False
    for i, line in enumerate(lines):
        if pattern.match(line):
            lines[i] = f"{key}={value}"
            found = True
            break
    if not found:
        lines.append(f"{key}={value}")
    with open(ENV_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return True


# ---------- заказы ----------

def create_order(api_key, template_id, text, name, confirm_payg=False, quote_token=None):
    body = {"template_id": template_id, "input_text": text, "name": name}
    if confirm_payg and quote_token:
        body["confirm_payg_topup"] = True
        body["quote_token"] = quote_token
    return api_call("POST", "/orders", api_key, body=body)


def get_order(api_key, order_id):
    return api_call("GET", f"/orders/{order_id}", api_key)


def retry_failed_items(api_key, order_id):
    return api_call("POST", f"/orders/{order_id}/items/retry-failed", api_key)


def poll_until_terminal(api_key, order_id, timeout=POLL_TIMEOUT_SEC, interval=POLL_INTERVAL_SEC):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = get_order(api_key, order_id)
        status = last.get("status")
        if status in TERMINAL_STATUSES or status == "partially_completed":
            return last
        time.sleep(interval)
    raise LumeanError(f"заказ {order_id} не завершился за {timeout:.0f}с "
                      f"(последний статус: {(last or {}).get('status')})")


def storage_url(api_key, path):
    data = api_call("POST", "/storage/url", api_key, body={"path": path})
    return (data or {}).get("url")


# ---------- alignment: best-effort парсинг неподтверждённой схемы ----------

def _try_parse_alignment_csv(raw_bytes):
    """Форма C — найдена живым прогоном (не гипотеза): у этого аккаунта
    alignment-сервисный файл приходит уже готовым CSV-текстом с заголовком
    "index,char,start,end" (одна строка — один символ, "," как символ сам
    честно квотируется CSV-модулем, а не ломает разбор наивным split(",")).
    Строгая та же дисциплина, что Форма A/B: любое отклонение от ожидаемых
    колонок/типов -> None целиком, не частичный/угаданный результат."""
    try:
        text = raw_bytes.decode("utf-8")
    except (UnicodeDecodeError, AttributeError):
        return None
    try:
        reader = csv.DictReader(io.StringIO(text))
    except Exception:
        return None
    if not reader.fieldnames or not {"char", "start", "end"} <= set(reader.fieldnames):
        return None
    rows = []
    for row in reader:
        c, s, e = row.get("char"), row.get("start"), row.get("end")
        if c is None or s is None or e is None:
            return None
        try:
            rows.append((str(c), float(s), float(e)))
        except (TypeError, ValueError):
            return None
    return rows if rows else None


def try_parse_alignment(raw_bytes):
    """Список [(char, start, end), ...] или None, если данные не подошли ни
    под одну из ДВУХ вероятных форм (см. докстринг модуля — схема
    alignment.*/result.json не задокументирована полем-в-поле в публичной
    спеке Lumean). Строгая проверка структуры ДО того, как строки уйдут в
    alignment.csv: пропущенный/неверно понятый формат должен превратиться в
    честный None (секция без alignment, word-count fallback), а не в
    молчаливо неверные тайминги, скормленные дальше в рендер."""
    try:
        data = json.loads(raw_bytes)
    except Exception:
        return _try_parse_alignment_csv(raw_bytes)

    # Форма A: нативный ElevenLabs with-timestamps (её же напрямую парсит
    # scripts/speech_generate.py call_elevenlabs_with_timestamps) —
    # правдоподобно, что Lumean прокси отдаёт ЕЁ ЖЕ для service_key=elevenlabs.
    if isinstance(data, dict) and {"characters", "character_start_times_seconds",
                                    "character_end_times_seconds"} <= data.keys():
        chars = data["characters"]
        starts = data["character_start_times_seconds"]
        ends = data["character_end_times_seconds"]
        if (isinstance(chars, list) and isinstance(starts, list) and isinstance(ends, list)
                and len(chars) == len(starts) == len(ends) and len(chars) > 0):
            try:
                return [(str(c), float(s), float(e)) for c, s, e in zip(chars, starts, ends)]
            except (TypeError, ValueError):
                return None
        return None

    # Форма B: плоский список объектов {character|char, start, end}.
    if isinstance(data, list) and data:
        rows = []
        for item in data:
            if not isinstance(item, dict):
                return None
            c = item.get("character", item.get("char"))
            s, e = item.get("start"), item.get("end")
            if c is None or s is None or e is None:
                return None
            try:
                rows.append((str(c), float(s), float(e)))
            except (TypeError, ValueError):
                return None
        return rows if rows else None

    return None


# ---------- аудио: длительность/склейка (без импорта pipeline_smart — см. script_parser.py
# про то, почему остальные самостоятельные скрипты его тоже избегают) ----------

def audio_duration(path):
    r = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", path],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    return float(json.loads(r.stdout)["format"]["duration"])


def concat_audio(paths, out_path, temp_dir):
    """ffmpeg concat demuxer (stream copy, откат на re-encode) — тот же
    принцип, что concat_fragment_audio() в speech_generate.py."""
    concat_list = os.path.join(temp_dir, "lumean_concat.txt")
    with open(concat_list, "w", encoding="utf-8") as f:
        for p in paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
    tmp_out = out_path + ".tmp.mp3"
    r = subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list,
                        "-c", "copy", tmp_out], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    if r.returncode != 0:
        r2 = subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list,
                             "-c:a", "libmp3lame", "-q:a", "2", tmp_out],
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
        if r2.returncode != 0:
            raise RuntimeError(f"склейка секций не удалась (copy и re-encode): {r2.stderr[-300:]}")
    os.replace(tmp_out, out_path)


# ---------- одна секция целиком: заказ -> опрос -> скачивание ----------

def process_section(api_key, template_id, temp_dir, idx, name, text, episode_name,
                    confirm_payg=False):
    """Возвращает dict с audio_path (или None при неисправимом сбое),
    alignment (список строк или None), status, order_id, reason."""
    result = {"section": name, "order_id": None, "status": None,
              "audio_path": None, "alignment": None, "reason": None}
    try:
        order = create_order(api_key, template_id, text,
                             name=f"{episode_name} — {name}"[:120],
                             confirm_payg=confirm_payg)
    except LumeanPaymentRequired as e:
        shortfall = e.body.get("shortfall_lmc")
        result["reason"] = (f"нужна доплата PAYG {shortfall} LMC (квоты подписки не хватило) — "
                            f"перезапусти с --confirm-payg, если согласен: {e}")
        print(f"    [{name}] СТОП: {result['reason']}")
        return result
    except LumeanError as e:
        result["reason"] = str(e)
        print(f"    [{name}] заказ не создан: {e}")
        return result

    order_id = order.get("id")
    result["order_id"] = order_id
    print(f"    [{name}] заказ {order_id} создан, жду готовности...")

    try:
        order = poll_until_terminal(api_key, order_id)
    except LumeanError as e:
        result["reason"] = str(e)
        print(f"    [{name}] {e}")
        return result

    status = order.get("status")
    if status == "partially_completed":
        print(f"    [{name}] partially_completed — пробую retry-failed (бесплатно, техника)...")
        try:
            retry_failed_items(api_key, order_id)
            order = poll_until_terminal(api_key, order_id)
            status = order.get("status")
        except LumeanError as e:
            print(f"    [{name}] retry-failed не помог: {e}")

    result["status"] = status
    if status not in ("completed", "result_delivered", "partially_completed"):
        result["reason"] = f"статус {status} — заказ не дал результата (см. items вручную)"
        print(f"    [{name}] {result['reason']}")
        return result
    if status == "partially_completed":
        flagged = [it.get("chunk_index") for it in order.get("items", [])
                  if it.get("status") == "policy_flagged"]
        result["reason"] = (f"часть чанков policy_flagged ({flagged}) — нужна правка текста "
                            f"человеком, см. §7.11 Lumean Public API")
        print(f"    [{name}] ВНИМАНИЕ: {result['reason']}")

    files = (order.get("result") or {}).get("files") or []
    if not files:
        result["reason"] = result["reason"] or "result.files пуст — скачивать нечего"
        print(f"    [{name}] {result['reason']}")
        return result

    audio_url = storage_url(api_key, files[0])
    if not audio_url:
        result["reason"] = "storage/url не вернул ссылку на аудио"
        print(f"    [{name}] {result['reason']}")
        return result
    audio_path = os.path.join(temp_dir, f"section_{idx:02d}.mp3")
    download(audio_url, audio_path)
    result["audio_path"] = audio_path
    print(f"    [{name}] аудио скачано: {audio_path}")

    for svc_path in (order.get("result") or {}).get("service_files") or []:
        base = os.path.basename(svc_path).lower()
        if "alignment" in base or base.startswith("result.json"):
            try:
                url = storage_url(api_key, svc_path)
                raw_dest = os.path.join(temp_dir, f"section_{idx:02d}_{base}")
                download(url, raw_dest)
                rows = try_parse_alignment(open(raw_dest, "rb").read())
                if rows:
                    result["alignment"] = rows
                    print(f"    [{name}] alignment распознан ({len(rows)} символов)")
                else:
                    print(f"    [{name}] alignment.* скачан, но схема не распознана — "
                          f"секция пойдёт по word-count таймингу")
            except Exception as e:
                print(f"    [{name}] не удалось скачать/разобрать alignment: {e}")
            break
    return result


# ---------- вывод: alignment CSV / section_offsets.json (тот же формат, что Шаг 6) ----------

def write_alignment_csv(alignment_dir, idx, rows):
    os.makedirs(alignment_dir, exist_ok=True)
    path = os.path.join(alignment_dir, f"{idx:02d}.csv")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["char", "start", "end"])
        w.writerows(rows)
    os.replace(tmp, path)
    return path


def atomic_write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ---------- CLI ----------

def cmd_list_voices(argv):
    api_key = os.environ.get("LUMEAN_API_KEY", "").strip()
    if not api_key:
        print("LUMEAN_API_KEY не задан в .env")
        return 1
    gender = None
    rest = []
    i = 0
    while i < len(argv):
        if argv[i] == "--gender" and i + 1 < len(argv):
            gender = argv[i + 1]
            i += 2
            continue
        rest.append(argv[i])
        i += 1
    search = " ".join(rest) if rest else "calm documentary narrator"
    try:
        voices = list_elevenlabs_voices(api_key, search=search, gender=gender)
    except LumeanError as e:
        print(f"Ошибка запроса к Lumean: {e}")
        return 1
    if not voices:
        print(f"Ничего не найдено по запросу {search!r}. Попробуй другой запрос "
              f"(например, на английском — библиотека ElevenLabs международная).")
        return 0
    print(f"Найдено {len(voices)} голос(ов) по запросу {search!r}"
         f"{f' (gender={gender})' if gender else ''}:\n")
    for v in voices:
        vid = v.get("voice_id") or v.get("id")
        vname = v.get("name") or v.get("display_name") or "?"
        preview = v.get("preview_url") or "(нет превью)"
        # Реальная форма ответа (проверено вживую, не по документации — там
        # эти поля не расписаны): gender/age/accent/description/language —
        # плоские top-level ключи, НЕ вложенный labels{}.
        meta = ", ".join(f"{k}={v[k]}" for k in ("gender", "age", "accent", "language")
                         if v.get(k))
        desc = v.get("description") or ""
        print(f"  voice_id={vid}  name={vname!r}  {meta}\n"
             f"    {desc}\n    preview: {preview}\n")
    print(f"Прослушай preview у выбранного голоса, затем:\n"
         f"  python scripts/lumean_tts.py --create-template <VOICE_ID>")
    return 0


def cmd_create_template(argv):
    api_key = os.environ.get("LUMEAN_API_KEY", "").strip()
    if not api_key:
        print("LUMEAN_API_KEY не задан в .env")
        return 1
    if not argv:
        print("Usage: lumean_tts.py --create-template <VOICE_ID> [--name \"Имя\"]")
        return 1
    voice_id = argv[0]
    name = "TTS"
    if "--name" in argv:
        idx = argv.index("--name")
        if idx + 1 < len(argv):
            name = argv[idx + 1]
    try:
        template_id = create_elevenlabs_template(api_key, voice_id, name)
    except LumeanError as e:
        print(f"Не удалось создать шаблон: {e}")
        return 1
    if not template_id:
        print("Lumean вернул ответ без data.id — шаблон не создан (см. лог выше).")
        return 1
    persist_env_value("LUMEAN_TEMPLATE_ID", template_id)
    print(f"Шаблон создан: {template_id}\nЗаписан в .env как LUMEAN_TEMPLATE_ID.")
    return 0


def main():
    argv = sys.argv[1:]
    if argv and argv[0] == "--list-voices":
        return cmd_list_voices(argv[1:])
    if argv and argv[0] == "--create-template":
        return cmd_create_template(argv[1:])

    api_key = os.environ.get("LUMEAN_API_KEY", "").strip()
    template_id = os.environ.get("LUMEAN_TEMPLATE_ID", "").strip()
    if not api_key:
        print("Lumean TTS: LUMEAN_API_KEY не задан в .env — не запускаю.")
        return 1
    if not template_id:
        print("Lumean TTS: LUMEAN_TEMPLATE_ID не задан — голос ещё не выбран.\n"
             "  1) python scripts/lumean_tts.py --list-voices [запрос]\n"
             "  2) прослушай preview_url, выбери голос\n"
             "  3) python scripts/lumean_tts.py --create-template <VOICE_ID>")
        return 1

    positional = [a for a in argv if not a.startswith("--")]
    force_length = "--force-length" in argv
    confirm_payg = "--confirm-payg" in argv
    video_dir = positional[0] if positional else os.getcwd()
    target_minutes = float(positional[1]) if len(positional) > 1 else None

    script_path = os.path.join(video_dir, "script.txt")
    audio_out = os.path.join(video_dir, "audio.mp3")
    if not os.path.exists(script_path):
        print(f"Lumean TTS: {script_path} не найден.")
        return 1
    if os.path.exists(audio_out):
        print(f"Lumean TTS: {audio_out} уже существует — не перезаписываю молча "
             f"(могла быть уже готовая озвучка). Удали/переименуй файл для повторной генерации.")
        return 1

    section_texts = extract_section_texts(script_path)
    if not section_texts:
        print("Lumean TTS: в script.txt не найдено ни одной секции HOOK/BLOCK/FINAL.")
        return 1

    total_words, per_section = wordcount_report(section_texts)
    print(f"Lumean TTS: {len(section_texts)} секци(й), {total_words} слов "
         f"(~{total_words / WPM:.1f} мин при {WPM:.0f} слов/мин).")
    for name, n in per_section.items():
        print(f"  {name}: {n} слов")
    if not enforce_length_gate(total_words, target_minutes, force=force_length):
        return 1

    temp_dir = os.path.join(video_dir, "media_plan", "lumean_temp")
    os.makedirs(temp_dir, exist_ok=True)
    episode_name = os.path.basename(os.path.normpath(video_dir))

    section_results = []
    for idx, (name, text) in enumerate(section_texts):
        print(f"\n[{idx + 1}/{len(section_texts)}] {name}")
        section_results.append(
            process_section(api_key, template_id, temp_dir, idx, name, text,
                            episode_name, confirm_payg=confirm_payg))

    failed = [r for r in section_results if not r["audio_path"]]
    if failed:
        print(f"\nСТОП: {len(failed)}/{len(section_results)} секци(й) не дали аудио — "
             f"audio.mp3 не собран:")
        for r in failed:
            print(f"  [{r['section']}] {r['reason']}")
        report_path = os.path.join(video_dir, "media_plan", "lumean_generation_report.json")
        atomic_write_json(report_path, {"ok": False, "sections": section_results})
        print(f"Отчёт: {report_path}\nПерезапусти после правки текста/доплаты — "
             f"уже готовые секции переиспользуются не будут (каждый прогон создаёт новые "
             f"заказы; см. докстринг модуля про честные границы retry).")
        return 1

    section_offsets = {}
    alignment_written = []
    global_offset = 0.0
    for idx, r in enumerate(section_results):
        section_offsets[r["section"]] = round(global_offset, 5)
        if r["alignment"]:
            path = write_alignment_csv(os.path.join(video_dir, "media_plan", "alignment"),
                                       idx, [(c, round(s, 5), round(e, 5)) for c, s, e in r["alignment"]])
            alignment_written.append(r["section"])
        global_offset += audio_duration(r["audio_path"])

    concat_audio([r["audio_path"] for r in section_results], audio_out, temp_dir)
    total_duration = audio_duration(audio_out)

    atomic_write_json(os.path.join(video_dir, "media_plan", "section_offsets.json"), section_offsets)

    estimated_minutes = total_words / WPM
    actual_minutes = total_duration / 60.0
    drift_pct = (abs(actual_minutes - estimated_minutes) / estimated_minutes * 100
                if estimated_minutes else 0.0)
    report = {
        "ok": True,
        "audio_path": audio_out,
        "total_duration_sec": round(total_duration, 2),
        "total_words": total_words,
        "estimated_minutes_by_wordcount": round(estimated_minutes, 2),
        "actual_minutes": round(actual_minutes, 2),
        "drift_percent": round(drift_pct, 1),
        "sections_with_alignment": alignment_written,
        "sections": section_results,
    }
    atomic_write_json(os.path.join(video_dir, "media_plan", "lumean_generation_report.json"), report)

    print(f"\nГотово: {audio_out} ({total_duration:.1f}с, {actual_minutes:.1f} мин)")
    print(f"Оценка по словам: {estimated_minutes:.1f} мин | факт: {actual_minutes:.1f} мин "
         f"| расхождение {drift_pct:.1f}%")
    if drift_pct > 15:
        print("  ВНИМАНИЕ: расхождение больше 15% — TTS прочитал заметно быстрее/медленнее "
             "расчётной скорости (ЧАСТЬ 9: 125 слов/мин — ориентир, не гарантия конкретного "
             "голоса); проверь темп голоса на слух перед сборкой видео.")
    if alignment_written:
        print(f"Alignment распознан для: {', '.join(alignment_written)}")
    if len(alignment_written) < len(section_results):
        missing = len(section_results) - len(alignment_written)
        print(f"  {missing} секци(й) без alignment.csv — pipeline_smart.py откатится на "
             f"word-count тайминг для их блоков (тот же путь, что у ручной озвучки без "
             f"сохранённого alignment — не деградация, штатный fallback).")
    print("Дальше по протоколу: scripts/fix_pauses.py (Шаг 7) — та же цепочка, что и после "
         "ручной/ElevenLabs-озвучки, без изменений.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
