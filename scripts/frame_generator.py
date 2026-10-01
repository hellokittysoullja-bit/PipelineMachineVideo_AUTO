#!/usr/bin/env python3
"""Кадры по плану (media_plan/frame_plan.json) -> frames/NNN.png.

КАК ВЫБИРАЕТСЯ КАДР — связка старого генератора, не новая:
  1. На кадр рисуется IMAGE_VARIANTS вариантов (по умолчанию 1: картинка
     платная, и описание кадра должно попадать с первого раза).
  2. Текст: нейросеть букв НЕ пишет никогда. Модель со зрением читает каждый
     вариант — на сыром кадре не должно быть ни одной буквы (псевдонадписи
     модели — брак). Русские подписи кладёт код (labels.py): Shantell Sans
     ExtraBold, запасной Balsamiq Sans, без обводки; место — пустая нижняя
     полоса или рамка от модели со зрением, пустота проверяется по пикселям.
  3. Сетка судьи (shot_judge.judge): все варианты одной картинкой, оценка
     0-3 по фразе и описанию кадра.
  4. Проверка по утверждениям (shot_judge.verify_claims) лучших по сетке:
     главное фразы, must/should из плана. Кадры
     сравниваются вектором shot_judge.claims_vector в порядке спецификации.
     Запрет «мультфильм/3D = брак» здесь выключен (cg_veto=False): канал
     рисованный по замыслу.
  5. Ничья по смыслу — shot_judge.rank_look («лучший как кадр фильма»).
  6. Ни один вариант не годен — ещё раунд (IMAGE_ROUNDS). Не вышло и после
     него — кадр помечается rejected и на экран НЕ идёт: сборщик отдаёт его
     время соседнему проверенному кадру (NEVER_SHOW_KNOWN_BAD старого
     генератора — «ни карточек, ни повторов», решение владельца).

КУДА РИСОВАТЬ: модель картинок шлюза (IMAGE_MODEL, IMAGE_SIZE, IMAGE_QUALITY
в .env). На каждый кадр в модель уходят образцы стиля look/style/, на кадры
с героем — ещё look/hero.* (look.py). Цена печатается до вызова, без
--confirm-spend генерации нет, потолок IMAGE_MAX_SPEND обязателен.

Судья и чтение текста — через шлюз (SHOT_JUDGE_MODEL, по умолчанию та же
Qwen 3.7 Plus, что в старом генераторе), с проверкой зрения в начале прогона.
Без ключа шлюза кадр берётся первым вариантом и помечается unchecked.

Usage: python scripts/frame_generator.py <video_dir> --confirm-spend [--only 3,7] [--force]"""
import argparse
import base64
import hashlib
import io
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env  # noqa: E402
import labels  # noqa: E402

GEN_VERSION = 3
TEXT_MATCH_MIN = 1.0        # только точное совпадение букв (см. докстринг, п.2)
VERIFY_TOP = 3              # сколько лучших по сетке проверять по утверждениям
READ_PROMPT = ("Transcribe every piece of text visible in this image exactly as written, "
               "letter by letter, one text fragment per line. Keep the original language and "
               "letters, do not translate or correct spelling. If there is no text, answer exactly: NONE")


# ------------------------------------------------------------------ промпт

def build_prompt(frame, n_style, with_hero):
    """Задание модели картинок. Главное первым (описание кадра от
    планировщика), служебное после, ссылки на референсы — по их порядку в
    запросе: сначала n_style образцов стиля, герой последним (look.refs)."""
    parts = [frame["picture"]]
    if with_hero:
        parts.append(f"The main character is the person in reference image {n_style + 1}: keep exactly their "
                     "head, face, body proportions and clothes; change only pose, action and expression")
    # Буквы нейросеть не пишет никогда: русские подписи кладёт код (labels.py).
    labs = frame.get("labels") or []
    if frame.get("kind") == "caption" and labs:
        parts.append("Leave the bottom fifth of the image as plain empty background for a caption added later")
    elif labs:
        parts.append(f"Leave {len(labs)} empty patches of plain background, one next to each labelled part, "
                     "each with a short hand-drawn arrow to its part, and nothing drawn around the patches")
    parts.append("No text, letters, numbers, digits, symbols, logos or signs anywhere in the image")
    # Камера наезжает на кадр до ~10% и вписывает его в 16:9 — главное у
    # самого края срезалось бы.
    parts.append("Keep the main subject well inside the frame, away from the edges")
    imgs = "reference image 1" if n_style == 1 else f"reference images 1-{n_style}"
    parts.append(f"Draw it in exactly the drawing style of {imgs}: the same line work, colors, shading and "
                 "simplicity. Take only the style from them, not their content or characters")
    return ". ".join(p.rstrip(". ") for p in parts) + "."


def normalize(s):
    s = s.upper().replace("Ё", "Е")
    return re.sub(r"[^0-9A-ZА-Я]+", "", s)


def text_score(expected, transcript):
    """(0..1, детали). Подпись найдена целиком в прочитанном (строки и склейки
    соседних строк: модель может разбить подпись на две) — 1.0, иначе 0.
    Без подписей: 1.0, если букв не прочитано."""
    lines = [ln.strip() for ln in (transcript or "").splitlines() if ln.strip()]
    if len(lines) == 1 and lines[0].strip(" .").upper() == "NONE":
        lines = []
    if not expected:
        letters = sum(len(re.sub(r"[^A-Za-zА-Яа-яЁё]", "", ln)) for ln in lines)
        return (1.0 if letters == 0 else 0.0), {"unexpected_text": lines}
    pool = [normalize(ln) for ln in lines]
    pool += [pool[i] + pool[i + 1] for i in range(len(pool) - 1)]
    pool.append("".join(pool[:len(lines)]))
    details = {lab: float(any(normalize(lab) and normalize(lab) in got for got in pool)) for lab in expected}
    return min(details.values()), details


# ------------------------------------------------------------------ бэкенды картинок

class GatewayBackend:
    """Модель картинок шлюза — через llm_gateway (потолок, цена из каталога)."""

    def __init__(self, gateway, model, size, quality=None):
        self.gw, self.model, self.size, self.quality = gateway, model, size, quality

    def generate(self, prompt, refs):
        images, price = self.gw.image(self.model, prompt, self.size, self.quality, n=1, images=refs)
        return images[0], price


def make_backend(gateway):
    model = os.environ.get("IMAGE_MODEL", "").strip()
    size = os.environ.get("IMAGE_SIZE", "").strip()
    if not model or not size:
        sys.exit("Задайте IMAGE_MODEL и IMAGE_SIZE в .env (python scripts/list_models.py image)")
    return GatewayBackend(gateway, model, size, os.environ.get("IMAGE_QUALITY", "").strip() or None)


# ------------------------------------------------------------------ генератор

class Generator:
    def __init__(self, backend, video_dir, look, *, judge_gw=None, judge_model=None, variants=1, rounds=1):
        self.backend, self.video_dir, self.look = backend, video_dir, look
        self.look_sig = look.signature()
        self.jgw, self.jmodel = judge_gw, judge_model
        self.variants, self.rounds = max(1, variants), max(1, rounds)
        self.cache_dir = os.path.join(video_dir, "media_plan", "image_cache")
        self.judge_cache = os.path.join(video_dir, "media_plan", "judge_cache")
        self.out_dir = os.path.join(video_dir, "frames")
        for d in (self.cache_dir, self.judge_cache, self.out_dir):
            os.makedirs(d, exist_ok=True)
        self.lock = threading.Lock()
        self.spent = 0

    # --- рисование с кэшем по (модель, размер, качество, облик, промпт, вариант)
    def _variant(self, prompt, v, with_hero):
        b = self.backend
        key = hashlib.sha256(f"{GEN_VERSION}|{b.model}|{b.size}|{b.quality}|{self.look_sig}|{with_hero}|{prompt}|{v}"
                             .encode("utf-8")).hexdigest()[:20]
        path = os.path.join(self.cache_dir, key + ".png")
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path, True
        img, price = b.generate(prompt, self.look.refs(with_hero))
        with self.lock:
            self.spent += price
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(img)
        os.replace(tmp, path)
        return path, False

    # --- чтение текста на картинке (кэш по файлу)
    def _read(self, path):
        cp = path + ".read.txt"
        if os.path.exists(cp):
            return open(cp, encoding="utf-8").read()
        from PIL import Image
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((1280, 1280))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=90)
        content = [{"type": "text", "text": READ_PROMPT}, {"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}}]
        text, _u, price = self.jgw.chat(self.jmodel, content, 300, 1200, reasoning=False)
        with self.lock:
            self.spent += price
        with open(cp, "w", encoding="utf-8") as f:
            f.write(text)
        return text

    def _judge(self, frame, cands):
        """{путь: {"text_ok", "grid", "answers", "vector"}} для кандидатов."""
        import shot_judge
        info = {p: {"text_ok": None, "grid": None, "answers": None, "vector": None} for p in cands}
        if not self.jgw:
            return info
        for p in cands:
            try:
                ok, det = text_score([], self._read(p))     # на сыром кадре букв быть не должно
                info[p].update(text_ok=ok >= TEXT_MATCH_MIN, text=det)
            except Exception as e:  # noqa: BLE001 — не прочли: текст не проверен
                info[p]["text_error"] = f"{type(e).__name__}: {e}"[:200]
        rep = {}
        grid = shot_judge.judge(self.jgw, self.jmodel, phrase=frame["text"], brief=frame["spec"]["focus"],
                                candidates=[(p, p) for p in cands], cache_dir=self.judge_cache,
                                report=rep)
        with self.lock:
            self.spent += rep.get("cost", 0)
        for p in cands:
            info[p]["grid"] = (grid or {}).get(p)
        order = sorted(cands, key=lambda p: (info[p]["text_ok"] is not False, info[p]["grid"] or 0), reverse=True)
        for p in order[:VERIFY_TOP]:
            ans, vinfo = shot_judge.verify_claims(
                self.jgw, self.jmodel, phrase=frame["text"], spec=frame["spec"], setting=None,
                path=p, cache_dir=self.judge_cache)
            with self.lock:
                self.spent += vinfo.get("cost") or 0
            info[p]["answers"] = ans
            if ans is not None and not shot_judge.shows_nothing(frame["spec"], ans, info[p]["grid"]):
                info[p]["vector"] = shot_judge.claims_vector(frame["spec"], ans, cg_veto=False)
        return info

    @staticmethod
    def _acceptable(i):
        return i["text_ok"] is not False and i["vector"] is not None and (i["grid"] is None or i["grid"] > 0)

    def frame(self, frame):
        import shot_judge
        with_hero = bool(frame.get("hero")) and self.look.hero is not None
        prompt = build_prompt(frame, len(self.look.style), with_hero)
        rec = {"index": frame["index"], "kind": frame["kind"], "labels": frame.get("labels"),
               "hero": with_hero, "prompt": prompt, "rounds": [], "path": None}
        pool = {}
        for rnd in range(self.rounds):
            new = []
            for v in range(rnd * self.variants, (rnd + 1) * self.variants):
                try:
                    p, _cached = self._variant(prompt, v, with_hero)
                    new.append(p)
                except Exception as e:  # noqa: BLE001 — сбой одного варианта не сбой кадра
                    rec["rounds"].append({"round": rnd, "variant": v, "error": f"{type(e).__name__}: {e}"[:300]})
                    if type(e).__name__ in ("BudgetExhausted", "PaymentRequired"):
                        break
            if not new and not pool:
                continue
            pool.update(self._judge(frame, new) if new else {})
            if not self.jgw or any(self._acceptable(i) for i in pool.values()):
                break
        if not pool:
            rec["status"] = "failed"
            return rec
        if not self.jgw:
            ranked, status = sorted(pool), "unchecked"
        else:
            good = [p for p, i in pool.items() if self._acceptable(i)]
            ranked = sorted(good, key=lambda p: (pool[p]["vector"], pool[p]["grid"] or 0), reverse=True)
            if len(ranked) > 1:
                top = [p for p in ranked if (pool[p]["vector"], pool[p]["grid"]) == (pool[ranked[0]]["vector"], pool[ranked[0]]["grid"])]
                if len(top) > 1:
                    order, _ = shot_judge.rank_look(self.jgw, self.jmodel, paths=top,
                                                    cache_dir=self.judge_cache)
                    if order:
                        ranked = [top[k] for k in order] + [p for p in ranked if p not in top]
            status = "ok"
        out = os.path.join(self.out_dir, f"{frame['index'] + 1:03d}.png")
        rec["candidates"] = {os.path.basename(p): i for p, i in pool.items()}
        if not ranked:
            rec["status"] = "rejected"
            rec["reason"] = "ни один вариант не прошёл проверку"
            return rec
        # Подписи — кодом, на лучший вариант; не легли — на следующий годный.
        tries = []
        for p in ranked:
            ok, info = labels.compose(p, out, frame, self.jgw, self.jmodel, self.judge_cache)
            tries.append({"variant": os.path.basename(p), "ok": ok, "info": info})
            if ok:
                rec.update(status=status, path=os.path.relpath(out, self.video_dir), chosen=os.path.basename(p),
                           labels_placed=info, label_tries=tries)
                return rec
        rec.update(status="rejected", reason="подписи не легли ни на один годный вариант", label_tries=tries)
        return rec


def write_report(video_dir, recs, extra):
    path = os.path.join(video_dir, "media_plan", "frames_report.json")
    old = {}
    try:
        old = {r["index"]: r for r in json.load(open(path, encoding="utf-8"))["frames"]}
    except (OSError, ValueError, KeyError):
        pass
    for r in recs:
        old[r["index"]] = r
    data = dict(extra, frames=[old[k] for k in sorted(old)])
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1, default=str)
    os.replace(path + ".tmp", path)
    return path


def main():
    env.load_env()
    import llm_gateway
    import look as look_mod
    import shot_judge
    ap = argparse.ArgumentParser()
    ap.add_argument("video_dir")
    ap.add_argument("--confirm-spend", action="store_true", help="разрешить платную генерацию")
    ap.add_argument("--only", default="", help="номера кадров через запятую (1-based)")
    ap.add_argument("--force", action="store_true", help="перевыбрать даже готовые frames/NNN.png")
    a = ap.parse_args()
    try:
        look = look_mod.load()
    except look_mod.LookError as e:
        sys.exit(str(e))
    plan = json.load(open(os.path.join(a.video_dir, "media_plan", "frame_plan.json"), encoding="utf-8"))
    if plan.get("has_hero") and look.hero is None:
        sys.exit("План составлен с героем, а look/hero.* нет: положите героя или перепланируйте (--force).")
    only = {int(x) - 1 for x in a.only.split(",") if x.strip()}
    todo = [f for f in plan["frames"] if (not only or f["index"] in only) and
            (a.force or not os.path.exists(os.path.join(a.video_dir, "frames", f"{f['index'] + 1:03d}.png")))]
    if not todo:
        print("Все кадры уже выбраны.")
        return 0

    cap = os.environ.get("IMAGE_MAX_SPEND", "").strip()
    img_gw = llm_gateway.Gateway(spend_cap=int(cap) if cap else None)
    backend = make_backend(img_gw)
    variants = int(os.environ.get("IMAGE_VARIANTS", "1"))
    rounds = int(os.environ.get("IMAGE_ROUNDS", "1"))
    per = img_gw.image_cost(backend.model, backend.size, backend.quality, 1)
    print(f"Модель {backend.model} ({backend.size}, {backend.quality or 'auto'}): картинка {per}, "
          f"до {per * variants * rounds * len(todo)} токенов баланса на {len(todo)} кадров")
    if per > 0 and not a.confirm_spend:
        sys.exit("Платная генерация: запустите с --confirm-spend, когда цена устраивает.")
    if per > 0 and not cap:
        sys.exit("Для платной модели задайте потолок IMAGE_MAX_SPEND в .env.")

    gw = llm_gateway.Gateway(spend_cap=int(os.environ.get("SHOT_JUDGE_MAX_SPEND", "300000")))
    judge_model = os.environ.get("SHOT_JUDGE_MODEL") or "qwen/qwen3.7-plus"
    jgw = None
    if gw.configured and os.environ.get("SHOT_JUDGE", "1") != "0":
        ok, why = shot_judge.vision_check(gw, judge_model)
        if ok:
            jgw = gw
        else:
            print(f"СУДЬЯ ВЫКЛЮЧЕН: {judge_model} не прошёл проверку зрения ({why}). Кадры не проверяются.")
    else:
        print("Судья не работает (нет LLM_GATEWAY_API_KEY или SHOT_JUDGE=0): кадры берутся без проверки.")
    gen = Generator(backend, a.video_dir, look, judge_gw=jgw, judge_model=judge_model,
                    variants=variants, rounds=rounds)
    print(f"Кадров: {len(todo)}, с героем {sum(1 for f in todo if f.get('hero'))}, образцов стиля "
          f"{len(look.style)}, вариантов {variants} x раундов до {rounds}, судья {judge_model if jgw else '—'}")
    with ThreadPoolExecutor(max(1, int(os.environ.get("IMAGE_WORKERS", "3")))) as ex:
        recs = list(ex.map(gen.frame, todo))
    path = write_report(a.video_dir, recs, {"model": backend.model, "size": backend.size,
                                            "quality": backend.quality, "look": gen.look_sig,
                                            "judge_model": judge_model if jgw else None,
                                            "spent_this_run": gen.spent, "gateway": gw.summary()})
    by = {}
    for r in recs:
        by[r["status"]] = by.get(r["status"], 0) + 1
    print(f"Готово: {by}. Потрачено: {gen.spent}. Отчёт: {path}")
    for r in recs:
        if r["status"] in ("rejected", "failed"):
            print(f"  кадр {r['index'] + 1}: {r['status']} — на экран не пойдёт, время отдаётся соседу")
    return 2 if any(r["status"] in ("rejected", "failed") for r in recs) else 0


if __name__ == "__main__":
    sys.exit(main())
