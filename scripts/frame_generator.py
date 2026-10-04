#!/usr/bin/env python3
"""Кадры по плану (media_plan/frame_plan.json) -> frames/NNN.png.

КАК ВЫБИРАЕТСЯ КАДР (картинка на фразу одна — она платная):
  1. Рисуется IMAGE_VARIANTS вариантов (по умолчанию 1).
  2. Текст: нейросеть букв НЕ пишет никогда. Модель со зрением читает
     вариант — на сыром кадре не должно быть ни буквы, ни цифры (брак).
     Русские подписи кладёт код (labels.py).
  3. Проверка по утверждениям (shot_judge.verify_claims): главное фразы и
     must из плана; «судья ответил: не то» — брак, «проверка не состоялась»
     (сбой сети, нет судьи) — не брак, кадр идёт с пометкой unchecked.
     Сетка судьи (shot_judge.judge) — только при 2+ вариантах: она сравнивает,
     а её оценки на рисунках не откалиброваны. Запрет «мультфильм = брак»
     выключен (cg_veto=False): канал рисованный.
  4. Подписи на годный вариант. Брак или нет места под подписи — ещё раунд
     (IMAGE_ROUNDS, по умолчанию 1 — решение владельца 04.10: одна попытка на кадр).
     Годный рисунок без места под подписи и после всех раундов — подписи на
     полосе цвета фона внизу, а не выброс кадра.
  5. Брак во всех раундах — rejected: на экран не идёт, время отдаётся
     соседнему кадру («ни карточек, ни повторов», решение владельца).

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

GEN_VERSION = 7
TEXT_MATCH_MIN = 1.0        # только точное совпадение букв (см. докстринг, п.2)
VERIFY_TOP = 3              # сколько лучших по сетке проверять по утверждениям
READ_PROMPT = ("Transcribe every piece of text visible in this image exactly as written, "
               "letter by letter, one text fragment per line. Keep the original language and "
               "letters, do not translate or correct spelling. If there is no text, answer exactly: NONE")


# ------------------------------------------------------------------ промпт

def paper_background():
    """Фон схем и подписей из .env (DIAGRAM_BACKGROUND), пусто — как в
    образцах стиля. Одно место: и промпт, и выравнивание пятен под подписями
    (labels.flatten_paper) включаются от одной настройки — ровный лист
    просили, значит пятно на нём брак; на фактурной бумаге выравнивание
    сделало бы гладкие заплаты."""
    return os.environ.get("DIAGRAM_BACKGROUND", "").strip()


_PERSON_RE = re.compile(r"\b(?:a|the) person\b|\bpersons?\b", re.I)


def hero_spec(spec, hero_text):
    """Копия спецификации, где человек в focus/subject/claims — это герой."""
    def sub(t):
        return _PERSON_RE.sub(hero_text, t) if isinstance(t, str) else t
    out = dict(spec)
    for k in ("focus", "subject"):
        if k in out:
            out[k] = sub(out[k])
    if isinstance(out.get("claims"), list):
        out["claims"] = [dict(c, text=sub(c.get("text"))) for c in out["claims"]]
    return out


# Порядок и форма задания — по руководству Google к Nano Banana (разбор 04.10):
# роли референсов первыми, затем СЦЕНА, затем РАСКЛАДКА, стиль последним.
# Запреты не перечисляются словами: список «no text, letters…» сам подсказывал
# модели, что рисовать, — вместо него утверждение о чистых поверхностях.
CLEAN_SURFACES = ("Every surface is clean and unmarked: blank paper, blank screens, plain envelopes, plain walls, "
                  "unprinted floor and unprinted clothes")


def _tidy(text):
    """Описание от планировщика без сдвоенных артиклей («The the envelope»):
    модель читает такую ошибку как два разных предмета."""
    text = re.sub(r"\b(the|a|an)\s+(?:the|a|an)\b", r"\1", text, flags=re.I)
    return " ".join(text.split())


def build_prompt(frame, n_style, with_hero, hero_states=None):
    """Задание модели картинок: кто есть кто среди референсов (n_style образцов
    стиля, герой последним — look.refs), сцена от планировщика, ОДНО требование
    размера и раскладка, места под код, фон, чистые поверхности, стиль."""
    imgs = "Reference image 1 shows" if n_style == 1 else f"Reference images 1-{n_style} show"
    parts = [f"{imgs} only the drawing style: take from them the line work, colors, shading and simplicity, "
             "never their content or characters"]
    if with_hero:
        # «character», а не «person»: герой может быть и не человеком (маскот-кот),
        # и слово «person» подталкивало бы модель нарисовать человека.
        parts.append(f"Reference image {n_style + 1} is the main character: keep exactly its head, face, colors, "
                     "markings, body proportions and clothing if any; change only pose, action and expression")
    parts.append("SCENE: " + _tidy(frame["picture"]))
    if with_hero:
        if re.search(r"foot ?prints?|tracks?\b", frame["picture"], re.I):
            # критик 04.10: следы кота нарисованы подошвами ботинок
            parts.append("Any footprints or tracks are prints of the main character's own feet, exactly as its feet "
                         "look in the reference image")
        st = (hero_states or {}).get(frame.get("hero_state") or "")
        if st:
            parts.append(st["draw"])
    # Требование размера одно: два «большое» спорят, и модель ужимает оба.
    # Наезд важнее героя — камера заполняет предметом весь экран.
    zoom = (frame.get("zoom") or {}).get("object")
    if zoom:
        parts.append(f"LAYOUT: the {zoom} is the biggest single object in the picture, its whole outline visible, "
                     "with plain empty background around it")
        if with_hero:
            # камера режет кадр то на предмет, то на героя: касание срезало бы второго
            parts.append(f"The main character and the {zoom} sit side by side with a clear gap of plain background "
                         "between them; neither touches or covers the other")
    elif with_hero:
        parts.append("LAYOUT: the main character is drawn big, at least a third of the image height")   # критик: кот на 10% кадра
    if frame.get("key_thought"):
        near = f" next to the {frame['key_near']}" if frame.get("key_near") else ""
        parts.append(f"Keep a calm area of plain empty background{near}, about a third of the image, for "
                     "handwriting added later")
    if frame.get("kind") == "diagram":
        # критика 04.10: схемы выходили чистой векторной инфографикой и выпадали из рисунков ролика
        parts.append("The diagram is drawn by hand like the rest of the film: ink lines with soft watercolor "
                     "washes, small drawn objects and figures instead of flat icons, no clean vector graphics")
    # Буквы нейросеть не пишет никогда: русские подписи кладёт код (labels.py).
    labs = frame.get("labels") or []
    if frame.get("kind") == "caption" and labs:
        parts.append("Leave the bottom fifth of the image as plain empty background for a caption added later")
    elif labs:
        # Замер 01.10: без размера и отступа модель оставила под «ДОФАМИН»
        # щель у края (кегль 66% шкалы) и залила места светлыми пятнами.
        parts.append(f"Leave exactly {len(labs)} wide empty patches of plain background, one next to each labelled "
                     "part, each big enough for a word in large letters (about a fifth of the image wide) and well "
                     "away from the image edges, each with a short hand-drawn arrow to its part. The patches are the "
                     "very same background as around them: no lighter fill, glow, shading, box or outline")
        # Цвет стрелок — решение владельца в .env (читается здесь, после
        # load_env); пусто — цвет из образцов стиля. Входит в промпт, а значит
        # и в ключ кэша: смена цвета перерисует схемы.
        arrow = os.environ.get("DIAGRAM_ARROW_COLOR", "").strip()
        if arrow:
            parts.append(f"Draw every arrow in {arrow}")
    # Фон — решение владельца в .env (DIAGRAM_BACKGROUND): образцы стиля задают
    # бумагу, а кадру без места нужен чистый светлый лист. Схема и подпись —
    # всегда лист. Сцена — лист, только если у неё нет конкретного места
    # (решение владельца 03.10: «белый фон, но не всегда»); место действия
    # (комната, улица) рисуется как есть.
    paper = paper_background()
    if paper and frame.get("kind") in ("diagram", "caption"):
        parts.append(f"The whole background is plain {paper} paper, even and untinted, "
                     "even if the reference images use a darker or coloured paper")
    elif paper:
        parts.append(f"If the picture has no specific place, its background is plain {paper} paper, even and "
                     "untinted, even if the reference images use a darker or coloured paper; a specific place "
                     "is drawn as that place")
    parts.append(CLEAN_SURFACES)
    # Камера наезжает на кадр до ~10% и вписывает его в 16:9 — главное у
    # самого края срезалось бы.
    parts.append("Keep the main subject well inside the frame, away from the edges")
    parts.append("Draw everything in exactly the drawing style of the style references")
    return ". ".join(p.rstrip(". ") for p in parts) + "."


def normalize(s):
    s = s.upper().replace("Ё", "Е")
    return re.sub(r"[^0-9A-ZА-Я]+", "", s)


def text_score(expected, transcript):
    """(0..1, детали). Подпись найдена целиком в прочитанном (строки и склейки
    соседних строк: модель может разбить подпись на две) — 1.0, иначе 0.
    Без подписей: 1.0, если не прочитано ни букв, ни цифр."""
    lines = [ln.strip() for ln in (transcript or "").splitlines() if ln.strip()]
    if len(lines) == 1 and lines[0].strip(" .").upper() == "NONE":
        lines = []
    if not expected:
        letters = sum(len(re.sub(r"[^0-9A-Za-zА-Яа-яЁё]", "", ln)) for ln in lines)   # и цифры: «777» — тоже брак
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
        self.jgw, self.jmodel = judge_gw, judge_model
        self.variants, self.rounds = max(1, variants), max(1, rounds)
        self.cache_dir = os.path.join(video_dir, "media_plan", "image_cache")
        self.judge_cache = os.path.join(video_dir, "media_plan", "judge_cache")
        self.out_dir = os.path.join(video_dir, "frames")
        for d in (self.cache_dir, self.judge_cache, self.out_dir):
            os.makedirs(d, exist_ok=True)
        self.lock = threading.Lock()
        self.spent = 0
        self.flat_paper = bool(paper_background())

    def task(self, frame):
        """(промпт, с героем ли, отпечаток задания). Отпечаток — фраза, промпт,
        модель, размер, качество и ТОТ облик, что реально уходит в модель
        (герой — только на кадрах с героем): совпал — готовый кадр годен."""
        with_hero = bool(frame.get("hero")) and self.look.hero is not None
        prompt = build_prompt(frame, len(self.look.style), with_hero, getattr(self.look, "hero_states", None))
        b = self.backend
        sig = hashlib.sha256(f"{GEN_VERSION}|{labels.COMPOSE_VERSION}|{frame.get('key')}|{b.model}|{b.size}|{b.quality}|"
                             f"{self.look.signature(with_hero)}|{prompt}".encode("utf-8")).hexdigest()[:20]
        return prompt, with_hero, sig

    # --- рисование с кэшем по (модель, размер, качество, облик, промпт, вариант)
    def _variant(self, prompt, v, with_hero):
        b = self.backend
        key = hashlib.sha256(f"{GEN_VERSION}|{b.model}|{b.size}|{b.quality}|{self.look.signature(with_hero)}|{prompt}|{v}"
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

    def _judge_spec(self, frame):
        """Спецификация для судьи. На кадре с героем «a person» в пунктах
        заменяется описанием героя: иначе герой-не-человек отклоняется."""
        spec = frame["spec"]
        if frame.get("hero") and self.look.hero is not None:
            return hero_spec(spec, self.look.hero_text or "the main character")
        return spec

    def _judge(self, frame, cands):
        """{путь: {"text_ok", "grid", "answers", "vector"}} для кандидатов."""
        import shot_judge
        info = {p: {"text_ok": None, "grid": None, "answers": None, "vector": None} for p in cands}
        spec = self._judge_spec(frame)
        if not self.jgw:
            return info
        for p in cands:
            try:
                ok, det = text_score([], self._read(p))     # на сыром кадре букв быть не должно
                info[p].update(text_ok=ok >= TEXT_MATCH_MIN, text=det)
            except Exception as e:  # noqa: BLE001 — не прочли: текст не проверен
                info[p]["text_error"] = f"{type(e).__name__}: {e}"[:200]
        # Сетка — СРАВНЕНИЕ вариантов; один вариант сравнивать не с чем, а как
        # вето её оценки на рисунках не откалиброваны (вопросы писались под
        # документальную съёмку) — поэтому сетка только при 2+ вариантах.
        if len(cands) > 1:
            rep = {}
            grid = shot_judge.judge(self.jgw, self.jmodel, phrase=frame["text"], brief=spec["focus"],
                                    candidates=[(p, p) for p in cands], cache_dir=self.judge_cache, report=rep)
            with self.lock:
                self.spent += rep.get("cost", 0)
            for p in cands:
                info[p]["grid"] = (grid or {}).get(p)
        order = sorted(cands, key=lambda p: (info[p]["text_ok"] is not False, info[p]["grid"] or 0), reverse=True)
        for p in order[:VERIFY_TOP]:
            ans, vinfo = shot_judge.verify_claims(
                self.jgw, self.jmodel, phrase=frame["text"], spec=spec, setting=None,
                path=p, cache_dir=self.judge_cache)
            with self.lock:
                self.spent += vinfo.get("cost") or 0
            info[p]["answers"] = ans
            if ans is not None and not shot_judge.shows_nothing(spec, ans, info[p]["grid"]):
                info[p]["vector"] = shot_judge.claims_vector(spec, ans, cg_veto=False)
        return info

    @staticmethod
    def _verdict(i):
        """"ok" — проверен и годен; "unchecked" — проверка не состоялась (нет
        судьи, сбой сети, неразобранный ответ): это не брак, кадр идёт с пометкой;
        None — брак (буквы на кадре, судья ответил «не то»)."""
        if i["text_ok"] is False or (i["grid"] is not None and i["grid"] <= 0):
            return None
        if i["vector"] is not None:
            return "ok"
        return "unchecked" if i["answers"] is None else None

    def frame(self, frame):
        prompt, with_hero, sig = self.task(frame)
        rec = {"index": frame["index"], "key": frame.get("key"), "sig": sig, "kind": frame["kind"],
               "labels": frame.get("labels"), "hero": with_hero, "prompt": prompt, "rounds": [], "path": None}
        out = os.path.join(self.out_dir, f"{frame['index'] + 1:03d}.png")
        pool, tries = {}, []

        def ranked():
            good = [p for p, i in pool.items() if self._verdict(i)]
            return sorted(good, key=lambda p: (self._verdict(pool[p]) == "ok", pool[p]["vector"] or (),
                                               pool[p]["grid"] or 0), reverse=True)

        def done(p, info, fallback=None):
            rec.update(status=self._verdict(pool[p]), path=os.path.relpath(out, self.video_dir),
                       chosen=os.path.basename(p), labels_placed=info, label_tries=tries,
                       candidates={os.path.basename(q): i for q, i in pool.items()})
            # где на рисунке предмет наезда и главный предмет — для камеры сборки
            try:
                import objects
                rec["objects"] = objects.objects_for(self.jgw, self.jmodel, p, frame, self.look.hero_text,
                                                     self.judge_cache)
            except Exception as e:  # noqa: BLE001 — без рамок камера просто не наезжает
                rec["objects"] = []
                rec["objects_error"] = f"{type(e).__name__}: {e}"[:200]
            if fallback:
                rec["labels_fallback"] = fallback
            return rec

        # Раунд: нарисовать, проверить, положить подписи. Второй раунд (платный)
        # — только если в первом не вышло ни проверки, ни подписей.
        for rnd in range(self.rounds):
            new = []
            for v in range(rnd * self.variants, (rnd + 1) * self.variants):
                try:
                    new.append(self._variant(prompt, v, with_hero)[0])
                except Exception as e:  # noqa: BLE001 — сбой одного варианта не сбой кадра
                    rec["rounds"].append({"round": rnd, "variant": v, "error": f"{type(e).__name__}: {e}"[:300]})
                    if type(e).__name__ in ("BudgetExhausted", "PaymentRequired"):
                        break
            if new:
                pool.update(self._judge(frame, new))
            for p in [q for q in ranked() if q in new]:
                ok, info = labels.compose(p, out, frame, self.jgw, self.jmodel, self.judge_cache,
                                           flat_paper=self.flat_paper)
                tries.append({"variant": os.path.basename(p), "ok": ok, "info": info})
                if ok:
                    return done(p, info)
        good = ranked()
        if good:
            # Годный рисунок, но места под подпись не нашлось ни в одном раунде:
            # подпись — на полосе цвета фона, а не выброс кадра (иначе на экране
            # висела бы прошлая картинка под новую фразу).
            ok, info = labels.compose(good[0], out, frame, self.jgw, self.jmodel, self.judge_cache, fallback=True,
                                       flat_paper=self.flat_paper)
            tries.append({"variant": os.path.basename(good[0]), "ok": ok, "info": info, "fallback": True})
            if ok:
                return done(good[0], info, fallback=info[0].get("fallback") if info else "unlabeled")
        rec["candidates"] = {os.path.basename(q): i for q, i in pool.items()}
        rec["label_tries"] = tries
        if not pool:
            rec["status"] = "failed"
        else:
            rec.update(status="rejected", reason="ни один вариант не прошёл проверку (буквы на кадре или «не то»)")
        return rec


def done_sigs(video_dir):
    """{номер кадра: отпечаток} готовых кадров из отчёта (годных к показу)."""
    try:
        frames = json.load(open(os.path.join(video_dir, "media_plan", "frames_report.json"), encoding="utf-8"))["frames"]
    except (OSError, ValueError, KeyError):
        return {}
    return {r["index"]: r.get("sig") for r in frames if r.get("status") not in ("rejected", "failed")}


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
    cap = os.environ.get("IMAGE_MAX_SPEND", "").strip()
    img_gw = llm_gateway.Gateway(spend_cap=int(cap) if cap else None)
    backend = make_backend(img_gw)
    only = {int(x) - 1 for x in a.only.split(",") if x.strip()}
    done = done_sigs(a.video_dir)
    probe = Generator(backend, a.video_dir, look)
    # Готовый кадр берётся, только если его отпечаток совпал: фраза, промпт,
    # модель и облик те же. Правка сценария, плана или образцов — перерисовка.
    todo = [f for f in plan["frames"] if (not only or f["index"] in only) and
            (a.force or done.get(f["index"]) != probe.task(f)[2]
             or not os.path.exists(os.path.join(a.video_dir, "frames", f"{f['index'] + 1:03d}.png")))]
    if not todo:
        print("Все кадры уже выбраны.")
        return 0

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
                                            "quality": backend.quality, "look": look.signature(True),
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
