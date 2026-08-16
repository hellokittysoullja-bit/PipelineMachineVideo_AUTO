#!/usr/bin/env python3
"""Бесплатный автоматический AI-генератор картинок (Pollinations.ai, без ключа,
без браузера, без ручных кликов) — для house-style каналов с маскотом, где
честный сток не подходит по стилю (ЧАСТЬ 14 CLAUDE.md).

Нужны два файла в <video_dir>/media_plan/:
  mascot_lock.txt    — первая строка "SEED=<число>" (фиксирует персонажа
                        между кадрами), дальше — промпт-замок персонажа/стиля
                        на английском (взять из CHANNEL.md §6 "Промпты")
  image_prompts.txt  — по одной строке на слот: "idx|английский промпт сцены"
                        (переносится из === IMAGE PROMPTS === сценария)

Пишет <video_dir>/media/{idx:03d}_ai.jpg — этот суффикс уже приоритетный
в assemble.py (AI > сток), никаких других изменений пайплайна не нужно.

Пруф-концепт (проверено вручную 16.08.2026): одинаковый SEED + один и тот же
промпт-замок персонажа дают устойчиво узнаваемого персонажа между разными
промптами сцены/позы — этого достаточно для консистентности мультяшного
маскота на 50-60 кадров видео.

Ограничения (честно): бесплатный публичный сервис без SLA — без гарантий
аптайма и скорости (кадр может занять от 2 до ~90 сек), может быть недоступен.
Не line-art/строгий flat-vector, а мягкая мультяшная 2D-иллюстрация — под это
и калибровать промпт-замок. При отказе слот остаётся пустым — фоллбэк неудачных
слотов на честный сток через stock_fetch_multisource.py (он их не тронет, если
_ai.jpg уже есть, и заполнит только то, что осталось пустым).

ВАЖНО про seed: не любой seed одинаково хорош. На практике (проверено вручную)
один и тот же промпт-замок на разных seed давал то чистого цветного персонажа,
то деградат (блёклый, без цвета, в паразитной овальной рамке) — это обычная
особенность диффузионных моделей, не баг сервиса. Перед первым роликом ОБЯЗАТЕЛЬНО
подобрать рабочий seed через --pick-seed, не брать первый попавшийся.

Safe re-run (пропуск уже готовых слотов, если не передан --force).
Usage:
  python scripts/ai_gen_pollinations.py <video_dir> --pick-seed [N]
      Сгенерировать N (по умолчанию 6) кандидатов seed на одном референсном
      промпте (только замок персонажа, без сцены) в media_plan/seed_candidates/,
      посмотреть глазами, вписать удачный SEED= в mascot_lock.txt вручную.
  python scripts/ai_gen_pollinations.py <video_dir> [--force]
      Обычный прогон: генерация всех слотов из image_prompts.txt.
"""
import os
import sys
import time
import urllib.parse
import urllib.request

BASE_URL = "https://image.pollinations.ai/prompt/{}"
WIDTH, HEIGHT = 1920, 1080
MODEL = "flux"
TIMEOUT = 90            # публичный сервис иногда отвечает медленно под нагрузкой
MAX_RETRIES = 3
DELAY_BETWEEN = 1.0     # вежливая пауза между запросами, не долбить сервис пачкой
MIN_BYTES = 2000        # подозрительно маленький ответ = вероятная заглушка/ошибка


def load_lock(video_dir):
    p = os.path.join(video_dir, "media_plan", "mascot_lock.txt")
    if not os.path.exists(p):
        print(f"Не найден {p}\n"
              f"  Создай файл: первая строка 'SEED=<число>', дальше — промпт-замок\n"
              f"  персонажа/стиля на английском из CHANNEL.md §6 'Промпты'.")
        return None, None
    seed = 42
    body_lines = []
    for line in open(p, encoding="utf-8").read().splitlines():
        s = line.strip()
        if not s:
            continue
        if s.upper().startswith("SEED="):
            try:
                seed = int(s.split("=", 1)[1].strip())
            except ValueError:
                print(f"  Не разобрал SEED в '{s}', беру дефолт {seed}")
        else:
            body_lines.append(s)
    if not body_lines:
        print(f"  {p}: нет текста промпта-замка после строки SEED=")
        return None, None
    return seed, " ".join(body_lines)


def load_prompts(video_dir):
    p = os.path.join(video_dir, "media_plan", "image_prompts.txt")
    if not os.path.exists(p):
        print(f"Не найден {p}\n"
              f"  Формат: 'idx|английский промпт сцены', по одной строке на слот.")
        return []
    rows, bad = [], []
    for n, line in enumerate(open(p, encoding="utf-8"), 1):
        line = line.rstrip("\n")
        if not line.strip():
            continue
        if "|" not in line:
            bad.append(n)
            continue
        i, text = line.split("|", 1)
        try:
            rows.append((int(i), text.strip()))
        except ValueError:
            bad.append(n)
    if bad:
        print(f"  Пропущены битые строки (ожидался формат idx|промпт): {bad}")
    return rows


def fetch(prompt, seed, out):
    url = (BASE_URL.format(urllib.parse.quote(prompt)) +
           f"?width={WIDTH}&height={HEIGHT}&nologo=true&seed={seed}&model={MODEL}")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                data = r.read()
            if len(data) < MIN_BYTES:
                raise ValueError(f"ответ подозрительно маленький ({len(data)} байт)")
            open(out, "wb").write(data)
            return True
        except Exception as e:
            print(f"    попытка {attempt}/{MAX_RETRIES}: {type(e).__name__}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(3 * attempt)
    return False


def pick_seed(video_dir, n):
    """Не читает SEED= из mascot_lock.txt (его как раз предстоит выбрать) —
    берёт только текст промпта-замока, игнорируя первую строку SEED=..."""
    p = os.path.join(video_dir, "media_plan", "mascot_lock.txt")
    if not os.path.exists(p):
        print(f"Не найден {p} — сначала создай его (промпт-замок из CHANNEL.md §6,\n"
              f"первая строка может быть любой заглушкой SEED=0, она не используется в этом режиме).")
        return 1
    body_lines = [ln.strip() for ln in open(p, encoding="utf-8").read().splitlines()
                  if ln.strip() and not ln.strip().upper().startswith("SEED=")]
    lock = " ".join(body_lines)
    if not lock:
        print(f"{p}: пусто после строки SEED=")
        return 1
    outdir = os.path.join(video_dir, "media_plan", "seed_candidates")
    os.makedirs(outdir, exist_ok=True)
    seeds = [11, 42, 123, 777, 1001, 2024, 3141, 5000, 8080, 9999][:max(1, n)]
    print(f"Генерирую {len(seeds)} кандидатов в {outdir}/ — посмотри глазами, выбери лучший,\n"
          f"впиши 'SEED=<число>' первой строкой в mascot_lock.txt.")
    for s in seeds:
        out = os.path.join(outdir, f"seed_{s}.jpg")
        if os.path.exists(out):
            continue
        print(f"  seed={s}...", flush=True)
        if not fetch(lock, s, out):
            print(f"    seed={s}: не удалось сгенерировать")
        time.sleep(DELAY_BETWEEN)
    return 0


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    video_dir = sys.argv[1]
    rest = sys.argv[2:]

    if "--pick-seed" in rest:
        i = rest.index("--pick-seed")
        n = 6
        if i + 1 < len(rest) and rest[i + 1].isdigit():
            n = int(rest[i + 1])
        return pick_seed(video_dir, n)

    force = "--force" in rest
    outdir = os.path.join(video_dir, "media")
    os.makedirs(outdir, exist_ok=True)

    seed, lock = load_lock(video_dir)
    if lock is None:
        return 1
    rows = load_prompts(video_dir)
    if not rows:
        return 1

    ok = fail = skip = 0
    for idx, scene_prompt in rows:
        out = os.path.join(outdir, f"{idx:03d}_ai.jpg")
        if os.path.exists(out) and not force:
            skip += 1
            continue
        full_prompt = f"{lock}, {scene_prompt}"
        print(f"[{idx:03d}] {scene_prompt[:70]}", flush=True)
        if fetch(full_prompt, seed, out):
            ok += 1
        else:
            fail += 1
            print(f"[{idx:03d}] ПРОВАЛ — слот пуст, добьёт stock_fetch_multisource.py")
        time.sleep(DELAY_BETWEEN)

    print(f"\nИтого: OK={ok} FAIL={fail} SKIP={skip}", flush=True)
    if ok == 0 and fail > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
