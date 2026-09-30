#!/usr/bin/env python3
"""Кадры по плану (media_plan/frame_plan.json) -> frames/NNN.png.

Модель картинок — IMAGE_MODEL из .env (сильная модель, которая пишет
кириллицу). Русские подписи из плана уходят в промпт ДОСЛОВНО.

Проверка текста (TEXT_CHECK_MODEL, модель со зрением): модель читает всё,
что написано на картинке, код сверяет прочитанное с подписями плана.
Не совпало — ещё одна попытка (до IMAGE_MAX_ATTEMPTS). Ни одна не прошла —
остаётся лучшая, кадр попадает в отчёт «проверить глазами», а не молча в
ролик. Сцена без текста проверяется на обратное: букв быть не должно.

ДЕНЬГИ. Сильная модель платная. Перед первым вызовом печатается верхняя
граница цены (кадры × попытки × цена картинки из каталога шлюза) и
генерация НЕ начинается без --confirm-spend. Потолок прогона —
IMAGE_MAX_SPEND (обязателен для платной модели): шлюз откажет ДО вызова,
а не спишет лишнее. Готовые кадры кэшируются: повторный прогон не платит.

Usage: python scripts/frame_generator.py <video_dir> [--confirm-spend] [--only 3,7] [--force]"""
import argparse
import base64
import difflib
import hashlib
import io
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import channel  # noqa: E402

GEN_VERSION = 1
# Подпись принимается только при ТОЧНОМ совпадении букв (после снятия
# пунктуации и пробелов, Ё=Е). Похожесть 0.9 пропускала «ПОЛНОСТЮ» вместо
# «ПОЛНОСТЬЮ» (0.957) — одна буква на крупной подписи видна каждому зрителю.
TEXT_MATCH_MIN = 1.0
READ_PROMPT = ("Transcribe every piece of text visible in this image exactly as written, "
               "letter by letter, one text fragment per line. Keep the original language and "
               "letters, do not translate or correct spelling. If there is no text, answer exactly: NONE")


def build_prompt(frame, profile):
    st = profile["style"]
    parts = [st["base"], st["backdrops"].get(frame["backdrop"], st["backdrops"]["paper"])]
    if frame.get("mascot"):
        parts.append("The main character appears: " + profile["mascot"]["description"])
    parts.append("Picture: " + frame["picture"])
    labels = frame.get("labels") or []
    if labels:
        quoted = "; ".join(f'label {i}: "{lab}"' for i, lab in enumerate(labels, 1))
        parts.append(f"Russian text in the image, {st['lettering']}, spelled exactly as given, "
                     f"Cyrillic letters: {quoted}. No other text, letters or numbers anywhere.")
    else:
        parts.append("No text, no letters, no numbers, no signs anywhere in the image.")
    return ". ".join(p.rstrip(". ") for p in parts) + "."


def normalize(s):
    s = s.upper().replace("Ё", "Е")
    return re.sub(r"[^0-9A-ZА-Я]+", "", s)


def text_score(expected, transcript):
    """(0..1, детали). Для каждой ожидаемой подписи — лучшее совпадение среди
    прочитанных строк и их склеек соседей (модель может разбить подпись на
    две строки). Итог — минимум по подписям: одна кривая подпись = кривой кадр.
    Без подписей: 1.0, если букв не прочитано, иначе 0.0."""
    lines = [ln.strip() for ln in (transcript or "").splitlines() if ln.strip()]
    if len(lines) == 1 and lines[0].strip(" .").upper() == "NONE":
        lines = []
    if not expected:
        letters = sum(len(re.sub(r"[^A-Za-zА-Яа-яЁё]", "", ln)) for ln in lines)
        return (1.0 if letters == 0 else 0.0), {"unexpected_text": lines}
    pool = [normalize(ln) for ln in lines]
    pool += [pool[i] + pool[i + 1] for i in range(len(pool) - 1)]
    pool.append("".join(pool[:len(lines)]))
    details, worst = {}, 1.0
    for lab in expected:
        want = normalize(lab)
        best = 0.0
        for got in pool:
            if not got:
                continue
            if want and want in got:
                best = 1.0
                break
            best = max(best, difflib.SequenceMatcher(None, want, got).ratio())
        details[lab] = round(best, 3)
        worst = min(worst, best)
    return worst, details


def read_text(gateway, model, png_bytes):
    from PIL import Image
    im = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    im.thumbnail((1280, 1280))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    content = [{"type": "text", "text": READ_PROMPT},
               {"type": "image_url", "image_url": {
                   "url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}}]
    text, _u, price = gateway.chat(model, content, 300, 1200, reasoning=False)
    return text, price


def cache_key(model, size, quality, prompt, attempt):
    return hashlib.sha1("|".join([str(GEN_VERSION), model, size, str(quality), prompt,
                                  str(attempt)]).encode("utf-8")).hexdigest()[:20]


class Generator:
    def __init__(self, gateway, video_dir, profile, model, check_model=None, attempts=2,
                 check_gateway=None):
        self.gw, self.video_dir, self.profile = gateway, video_dir, profile
        self.model, self.check_model, self.attempts = model, check_model, max(1, attempts)
        self.check_gw = check_gateway or gateway
        self.size = profile["image"]["size"]
        self.quality = profile["image"].get("quality")
        self.cache_dir = os.path.join(video_dir, "media_plan", "image_cache")
        self.out_dir = os.path.join(video_dir, "frames")
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.out_dir, exist_ok=True)
        self.lock = threading.Lock()
        self.spent = 0

    def _image(self, prompt, attempt):
        key = cache_key(self.model, self.size, self.quality, prompt, attempt)
        path = os.path.join(self.cache_dir, key + ".png")
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return open(path, "rb").read(), True
        images, price = self.gw.image(self.model, prompt, self.size, self.quality, n=1)
        with self.lock:
            self.spent += price
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(images[0])
        os.replace(tmp, path)
        return images[0], False

    def _check(self, frame, img_bytes, attempt, prompt):
        if not self.check_model:
            return None, None
        key = cache_key(self.check_model, "read", None, prompt, attempt)
        path = os.path.join(self.cache_dir, key + ".read.txt")
        if os.path.exists(path):
            transcript = open(path, encoding="utf-8").read()
        else:
            transcript, price = read_text(self.check_gw, self.check_model, img_bytes)
            with self.lock:
                self.spent += price
            with open(path, "w", encoding="utf-8") as f:
                f.write(transcript)
        score, details = text_score(frame.get("labels") or [], transcript)
        return score, {"transcript": transcript, "match": details}

    def frame(self, frame):
        prompt = build_prompt(frame, self.profile)
        out = os.path.join(self.out_dir, f"{frame['index'] + 1:03d}.png")
        tries, best = [], None
        for attempt in range(self.attempts):
            try:
                img, cached = self._image(prompt, attempt)
            except Exception as e:   # сбой одной попытки — не сбой кадра
                tries.append({"attempt": attempt, "error": f"{type(e).__name__}: {e}"})
                if type(e).__name__ in ("BudgetExhausted", "PaymentRequired"):
                    break
                continue
            try:
                score, check = self._check(frame, img, attempt, prompt)
            except Exception as e:
                score, check = None, {"error": f"{type(e).__name__}: {e}"}
            tries.append({"attempt": attempt, "cached": cached, "score": score, "check": check})
            rank = -1 if score is None else score
            if best is None or rank > best[0]:
                best = (rank, img)
            if score is None or score >= TEXT_MATCH_MIN:
                break      # проверки нет (нечем проверить) или текст верный
        rec = {"index": frame["index"], "kind": frame["kind"], "labels": frame.get("labels"),
               "prompt": prompt, "tries": tries, "path": None}
        if best is None:
            rec["status"] = "failed"
            return rec
        with open(out, "wb") as f:
            f.write(best[1])
        rec["path"] = os.path.relpath(out, self.video_dir)
        if best[0] < 0:
            rec["status"] = "unchecked"
        elif best[0] >= TEXT_MATCH_MIN:
            rec["status"] = "ok"
        else:
            rec["status"] = "text_mismatch"
        return rec


def estimate(gateway, model, size, quality, n_frames, attempts):
    per = gateway.image_cost(model, size, quality, 1)
    return per, per * n_frames * attempts


def main():
    channel.load_env()
    import llm_gateway
    ap = argparse.ArgumentParser()
    ap.add_argument("video_dir")
    ap.add_argument("--confirm-spend", action="store_true",
                    help="разрешить платную генерацию (без флага только печатается цена)")
    ap.add_argument("--only", default="", help="номера кадров через запятую (1-based)")
    ap.add_argument("--force", action="store_true", help="перерисовать даже готовые frames/NNN.png")
    a = ap.parse_args()

    model = os.environ.get("IMAGE_MODEL", "").strip()
    if not model:
        sys.exit("В .env не задан IMAGE_MODEL (модель картинок, которая пишет кириллицу). "
                 "Список моделей шлюза: python scripts/list_models.py image")
    check_model = os.environ.get("TEXT_CHECK_MODEL", "").strip() or None
    attempts = int(os.environ.get("IMAGE_MAX_ATTEMPTS", "2"))
    workers = int(os.environ.get("IMAGE_WORKERS", "3"))
    cap = os.environ.get("IMAGE_MAX_SPEND", "").strip()
    profile = channel.load_profile()
    plan = json.load(open(os.path.join(a.video_dir, "media_plan", "frame_plan.json"), encoding="utf-8"))
    frames = plan["frames"]
    only = {int(x) - 1 for x in a.only.split(",") if x.strip()}
    todo = [f for f in frames if (not only or f["index"] in only) and
            (a.force or not os.path.exists(os.path.join(a.video_dir, "frames", f"{f['index'] + 1:03d}.png")))]
    if not todo:
        print("Все кадры уже нарисованы.")
        return

    gw = llm_gateway.Gateway(spend_cap=int(cap) if cap else None)
    if not gw.configured:
        sys.exit("Нет LLM_GATEWAY_API_KEY в .env")
    size, quality = profile["image"]["size"], profile["image"].get("quality")
    per, worst = estimate(gw, model, size, quality, len(todo), attempts)
    print(f"Кадров к генерации: {len(todo)}, модель {model} {size}, цена картинки {per}, "
          f"верхняя граница {worst} токенов баланса (попыток до {attempts}; готовое из кэша бесплатно)")
    if per > 0 and not a.confirm_spend:
        sys.exit("Платная генерация: запустите с --confirm-spend, когда цена устраивает.")
    if per > 0 and not cap:
        sys.exit("Для платной модели задайте потолок IMAGE_MAX_SPEND в .env.")
    if not check_model:
        print("TEXT_CHECK_MODEL не задан: подписи не проверяются, сверяйте кадры глазами.")

    gen = Generator(gw, a.video_dir, profile, model, check_model, attempts)
    with ThreadPoolExecutor(max(1, workers)) as ex:
        recs = list(ex.map(gen.frame, todo))

    report_path = os.path.join(a.video_dir, "media_plan", "frames_report.json")
    old = {}
    try:
        old = {r["index"]: r for r in json.load(open(report_path, encoding="utf-8"))["frames"]}
    except (OSError, ValueError, KeyError):
        pass
    for r in recs:
        old[r["index"]] = r
    merged = [old[k] for k in sorted(old)]
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"model": model, "check_model": check_model, "frames": merged,
                   "spent_this_run": gen.spent, "gateway": gw.summary()}, f, ensure_ascii=False, indent=1)
    by = {}
    for r in recs:
        by[r["status"]] = by.get(r["status"], 0) + 1
    print(f"Готово: {by}. Потрачено: {gen.spent}. Отчёт: {report_path}")
    bad = [r for r in recs if r["status"] in ("text_mismatch", "failed")]
    for r in bad:
        print(f"  кадр {r['index'] + 1}: {r['status']} {r.get('labels') or ''}")
    if bad:
        sys.exit(2)


if __name__ == "__main__":
    main()
