#!/usr/bin/env python3
"""Кадры по плану (media_plan/frame_plan.json) -> frames/NNN.png.

КАК ВЫБИРАЕТСЯ КАДР — связка старого генератора, не новая:
  1. На кадр рисуется IMAGE_VARIANTS вариантов (локальная модель бесплатна,
     поэтому по умолчанию 3; судья сравнивает их на одной сетке).
  2. Подписи: модель со зрением читает текст на каждом варианте, код сверяет
     буквы ТОЧНО (без пунктуации и пробелов, Ё=Е). Похожесть не
     засчитывается: «ПОЛНОСТЮ» вместо «ПОЛНОСТЬЮ» — брак. Сцена проверяется
     на обратное: букв быть не должно.
  3. Сетка судьи (shot_judge.judge): все варианты одной картинкой, оценка
     0-3 по фразе и описанию кадра, мир эпизода одной строкой.
  4. Проверка по утверждениям (shot_judge.verify_claims) лучших по сетке:
     главное фразы, must/should из плана, мир отдельным вопросом. Кадры
     сравниваются вектором shot_judge.claims_vector в порядке спецификации.
     Запрет «мультфильм/3D = брак» здесь выключен (cg_veto=False): канал
     рисованный по замыслу.
  5. Ничья по смыслу — shot_judge.rank_look («лучший как кадр фильма»)
     с обликом фильма.
  6. Ни один вариант не годен — ещё раунд (IMAGE_ROUNDS). Не вышло и после
     него — кадр помечается rejected и на экран НЕ идёт: сборщик отдаёт его
     время соседнему проверенному кадру (NEVER_SHOW_KNOWN_BAD старого
     генератора — «ни карточек, ни повторов», решение владельца).

КУДА РИСОВАТЬ (IMAGE_BACKEND):
  comfyui — локальный ComfyUI (IMAGE_BASE_URL, по умолчанию http://127.0.0.1:8188),
            граф в API-формате из IMAGE_COMFY_WORKFLOW с метками {{PROMPT}},
            {{NEGATIVE}}, {{SEED}}, {{WIDTH}}, {{HEIGHT}};
  openai  — любой локальный сервер с OpenAI-совместимым /v1/images/generations;
  gateway — модель шлюза (платная): цена печатается до вызова, без
            --confirm-spend генерации нет, потолок IMAGE_MAX_SPEND обязателен.

Судья и чтение текста — через шлюз (SHOT_JUDGE_MODEL, по умолчанию та же
Qwen 3.7 Plus, что в старом генераторе), с проверкой зрения в начале прогона.
Без ключа шлюза кадр берётся первым вариантом и помечается unchecked.

Usage: python scripts/frame_generator.py <video_dir> [--confirm-spend] [--only 3,7] [--force]"""
import argparse
import base64
import hashlib
import io
import json
import os
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import channel  # noqa: E402

GEN_VERSION = 2
TEXT_MATCH_MIN = 1.0        # только точное совпадение букв (см. докстринг, п.2)
VERIFY_TOP = 3              # сколько лучших по сетке проверять по утверждениям
NEGATIVE = "photo, photorealistic, 3d render, blurry, watermark, signature, extra limbs, misspelled text, latin letters"
READ_PROMPT = ("Transcribe every piece of text visible in this image exactly as written, "
               "letter by letter, one text fragment per line. Keep the original language and "
               "letters, do not translate or correct spelling. If there is no text, answer exactly: NONE")


# ------------------------------------------------------------------ промпт

def build_prompt(frame, profile):
    """Главное первым, стиль коротким хвостом В КОНЦЕ — урок старого
    генератора (shot_generator): длинный стиль впереди съедал предмет."""
    st = profile["style"]
    parts = [frame["picture"]]
    if frame.get("mascot"):
        parts.append("The main character: " + profile["mascot"]["description"])
    labels = frame.get("labels") or []
    if labels:
        quoted = "; ".join(f'label {i}: "{lab}"' for i, lab in enumerate(labels, 1))
        parts.append(f"Russian text in the image, {st['lettering']}, spelled exactly as given, "
                     f"Cyrillic letters: {quoted}. No other text, letters or numbers anywhere")
    else:
        parts.append("No text, no letters, no numbers, no signs anywhere in the image")
    # Камера наезжает на кадр до ~10% и вписывает его в 16:9 — подпись у самого
    # края срезалась бы; просьба держать главное в центре дешевле, чем кроп.
    parts.append("Keep all text and the main subject well inside the frame, away from the edges")
    parts.append(st["backdrops"].get(frame["backdrop"], st["backdrops"]["paper"]))
    parts.append(st["base"])
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

class BackendError(Exception):
    pass


def _http_json(url, body=None, timeout=600, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


class ComfyBackend:
    """Локальный ComfyUI: граф в API-формате, метки подставляются в строки."""
    name = "comfyui"

    def __init__(self, base_url, workflow_path, size):
        self.base = base_url.rstrip("/")
        with open(workflow_path, encoding="utf-8") as f:
            self.workflow = f.read()
        self.w, self.h = (int(x) for x in size.split("x"))
        self.model = "comfyui:" + os.path.basename(workflow_path)

    def _fill(self, prompt, seed):
        def js(s):
            return json.dumps(s, ensure_ascii=False)[1:-1]
        wf = (self.workflow.replace("{{PROMPT}}", js(prompt)).replace("{{NEGATIVE}}", js(NEGATIVE))
              .replace('"{{SEED}}"', str(seed)).replace("{{SEED}}", str(seed))
              .replace('"{{WIDTH}}"', str(self.w)).replace('"{{HEIGHT}}"', str(self.h))
              .replace("{{WIDTH}}", str(self.w)).replace("{{HEIGHT}}", str(self.h)))
        return json.loads(wf)

    def generate(self, prompt, seed):
        pid = _http_json(self.base + "/prompt", {"prompt": self._fill(prompt, seed)})["prompt_id"]
        deadline = time.time() + float(os.environ.get("IMAGE_TIMEOUT_SEC", "1800"))
        while time.time() < deadline:
            hist = _http_json(f"{self.base}/history/{pid}")
            if pid in hist:
                for node in hist[pid].get("outputs", {}).values():
                    for img in node.get("images", []):
                        q = urllib.parse.urlencode({"filename": img["filename"], "subfolder": img.get("subfolder", ""),
                                                    "type": img.get("type", "output")})
                        with urllib.request.urlopen(f"{self.base}/view?{q}", timeout=120) as r:
                            return r.read(), 0
                raise BackendError(f"ComfyUI: граф отработал без картинки ({pid})")
            time.sleep(1.5)
        raise BackendError("ComfyUI: таймаут генерации")


class OpenAIBackend:
    """Любой сервер с OpenAI-совместимым /images/generations (локальный)."""
    name = "openai"

    def __init__(self, base_url, model, size, api_key=""):
        self.base, self.model, self.size, self.key = base_url.rstrip("/"), model, size, api_key

    def generate(self, prompt, seed):
        body = {"model": self.model, "prompt": prompt, "n": 1, "size": self.size,
                "response_format": "b64_json", "seed": seed}
        hdr = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        r = _http_json(self.base + "/images/generations", body,
                       timeout=float(os.environ.get("IMAGE_TIMEOUT_SEC", "1800")), headers=hdr)
        data = [d for d in r.get("data") or [] if d.get("b64_json")]
        if not data:
            raise BackendError("сервер вернул ответ без картинки")
        return base64.b64decode(data[0]["b64_json"]), 0


class GatewayBackend:
    """Платная модель шлюза — через llm_gateway (потолок, цена из каталога)."""
    name = "gateway"

    def __init__(self, gateway, model, size, quality=None):
        self.gw, self.model, self.size, self.quality = gateway, model, size, quality

    def generate(self, prompt, seed):
        images, price = self.gw.image(self.model, prompt, self.size, self.quality, n=1)
        return images[0], price


def make_backend(profile, gateway=None):
    kind = (os.environ.get("IMAGE_BACKEND") or "comfyui").strip().lower()
    # IMAGE_SIZE — если модель умеет только свои размеры (у платных часто 1536x1024);
    # кадр не 16:9 сборщик вписывает целиком на подложку, подписи не режутся.
    size = (os.environ.get("IMAGE_SIZE") or "").strip() or profile["image"]["size"]
    if kind == "comfyui":
        wf = os.environ.get("IMAGE_COMFY_WORKFLOW", "").strip()
        if not wf or not os.path.exists(wf):
            sys.exit("IMAGE_BACKEND=comfyui: задайте IMAGE_COMFY_WORKFLOW — граф ComfyUI в API-формате "
                     "с метками {{PROMPT}}, {{SEED}} (см. assets/comfyui/README.md)")
        return ComfyBackend(os.environ.get("IMAGE_BASE_URL") or "http://127.0.0.1:8188", wf, size)
    if kind == "openai":
        base = os.environ.get("IMAGE_BASE_URL", "").strip()
        if not base:
            sys.exit("IMAGE_BACKEND=openai: задайте IMAGE_BASE_URL локального сервера")
        return OpenAIBackend(base, os.environ.get("IMAGE_MODEL", "local"), size, os.environ.get("IMAGE_API_KEY", ""))
    if kind == "gateway":
        model = os.environ.get("IMAGE_MODEL", "").strip()
        if not model:
            sys.exit("IMAGE_BACKEND=gateway: задайте IMAGE_MODEL (python scripts/list_models.py image)")
        return GatewayBackend(gateway, model, size, profile["image"].get("quality"))
    sys.exit(f"Неизвестный IMAGE_BACKEND={kind!r} (comfyui | openai | gateway)")


# ------------------------------------------------------------------ генератор

def seed_for(prompt, variant):
    return int(hashlib.sha256(f"{prompt}|{variant}".encode("utf-8")).hexdigest()[:8], 16)


class Generator:
    def __init__(self, backend, video_dir, profile, *, judge_gw=None, judge_model=None, setting=None,
                 world=None, look_style=None, variants=3, rounds=2):
        self.backend, self.video_dir, self.profile = backend, video_dir, profile
        self.jgw, self.jmodel = judge_gw, judge_model
        self.setting, self.world, self.look_style = setting, world, look_style
        self.variants, self.rounds = max(1, variants), max(1, rounds)
        self.cache_dir = os.path.join(video_dir, "media_plan", "image_cache")
        self.judge_cache = os.path.join(video_dir, "media_plan", "judge_cache")
        self.out_dir = os.path.join(video_dir, "frames")
        for d in (self.cache_dir, self.judge_cache, self.out_dir):
            os.makedirs(d, exist_ok=True)
        self.lock = threading.Lock()
        self.spent = 0

    # --- рисование с кэшем по (бэкенд, модель, промпт, вариант)
    def _variant(self, prompt, v):
        key = hashlib.sha256(f"{GEN_VERSION}|{self.backend.name}|{self.backend.model}|{prompt}|{v}"
                             .encode("utf-8")).hexdigest()[:20]
        path = os.path.join(self.cache_dir, key + ".png")
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path, True
        img, price = self.backend.generate(prompt, seed_for(prompt, v))
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
                ok, det = text_score(frame.get("labels") or [], self._read(p))
                info[p].update(text_ok=ok >= TEXT_MATCH_MIN, text=det)
            except Exception as e:  # noqa: BLE001 — не прочли: текст не проверен
                info[p]["text_error"] = f"{type(e).__name__}: {e}"[:200]
        rep = {}
        grid = shot_judge.judge(self.jgw, self.jmodel, phrase=frame["text"], brief=frame["spec"]["focus"],
                                candidates=[(p, p) for p in cands], cache_dir=self.judge_cache,
                                report=rep, setting=self.setting)
        with self.lock:
            self.spent += rep.get("cost", 0)
        for p in cands:
            info[p]["grid"] = (grid or {}).get(p)
        order = sorted(cands, key=lambda p: (info[p]["text_ok"] is not False, info[p]["grid"] or 0), reverse=True)
        for p in order[:VERIFY_TOP]:
            ans, vinfo = shot_judge.verify_claims(
                self.jgw, self.jmodel, phrase=frame["text"], spec=frame["spec"], setting=self.world,
                path=p, cache_dir=self.judge_cache, world_separate=bool(self.world))
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
        prompt = build_prompt(frame, self.profile)
        rec = {"index": frame["index"], "kind": frame["kind"], "labels": frame.get("labels"),
               "prompt": prompt, "rounds": [], "path": None}
        pool = {}
        for rnd in range(self.rounds):
            new = []
            for v in range(rnd * self.variants, (rnd + 1) * self.variants):
                try:
                    p, _cached = self._variant(prompt, v)
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
            best = sorted(pool)[0]
            status = "unchecked"
        else:
            good = [p for p, i in pool.items() if self._acceptable(i)]
            if good:
                top = max(pool[p]["vector"] for p in good)
                tied = [p for p in good if pool[p]["vector"] == top]
                tied.sort(key=lambda p: pool[p]["grid"] or 0, reverse=True)
                tied = [p for p in tied if (pool[p]["grid"] or 0) == (pool[tied[0]]["grid"] or 0)]
                best = tied[0]
                if len(tied) > 1:
                    order, _ = shot_judge.rank_look(self.jgw, self.jmodel, paths=tied,
                                                    cache_dir=self.judge_cache, style=self.look_style)
                    if order:
                        best = tied[order[0]]
                status = "ok"
            else:
                best = max(pool, key=lambda p: (pool[p]["text_ok"] is not False, pool[p]["grid"] or 0))
                status = "rejected" if any(pool[p]["text_ok"] is not None or pool[p]["grid"] is not None
                                           for p in pool) else "unchecked"
        out = os.path.join(self.out_dir, f"{frame['index'] + 1:03d}.png")
        with open(best, "rb") as src, open(out + ".tmp", "wb") as dst:
            dst.write(src.read())
        os.replace(out + ".tmp", out)
        rec.update(status=status, path=os.path.relpath(out, self.video_dir), chosen=os.path.basename(best),
                   candidates={os.path.basename(p): i for p, i in pool.items()})
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
    channel.load_env()
    import llm_gateway
    import shot_judge
    ap = argparse.ArgumentParser()
    ap.add_argument("video_dir")
    ap.add_argument("--confirm-spend", action="store_true", help="разрешить платную генерацию (IMAGE_BACKEND=gateway)")
    ap.add_argument("--only", default="", help="номера кадров через запятую (1-based)")
    ap.add_argument("--force", action="store_true", help="перевыбрать даже готовые frames/NNN.png")
    a = ap.parse_args()
    profile = channel.load_profile()
    plan = json.load(open(os.path.join(a.video_dir, "media_plan", "frame_plan.json"), encoding="utf-8"))
    only = {int(x) - 1 for x in a.only.split(",") if x.strip()}
    todo = [f for f in plan["frames"] if (not only or f["index"] in only) and
            (a.force or not os.path.exists(os.path.join(a.video_dir, "frames", f"{f['index'] + 1:03d}.png")))]
    if not todo:
        print("Все кадры уже выбраны.")
        return 0

    gw = llm_gateway.Gateway(spend_cap=int(os.environ.get("SHOT_JUDGE_MAX_SPEND", "300000")))
    backend_kind = (os.environ.get("IMAGE_BACKEND") or "comfyui").lower()
    img_gw = None
    if backend_kind == "gateway":
        cap = os.environ.get("IMAGE_MAX_SPEND", "").strip()
        img_gw = llm_gateway.Gateway(spend_cap=int(cap) if cap else None)
    backend = make_backend(profile, img_gw)
    variants = int(os.environ.get("IMAGE_VARIANTS", "1" if backend_kind == "gateway" else "3"))
    rounds = int(os.environ.get("IMAGE_ROUNDS", "2"))
    if backend_kind == "gateway":
        per = img_gw.image_cost(backend.model, backend.size, backend.quality, 1)
        print(f"Платная модель {backend.model}: картинка {per}, верхняя граница "
              f"{per * variants * rounds * len(todo)} токенов баланса на {len(todo)} кадров")
        if per > 0 and not a.confirm_spend:
            sys.exit("Платная генерация: запустите с --confirm-spend, когда цена устраивает.")
        if per > 0 and not os.environ.get("IMAGE_MAX_SPEND"):
            sys.exit("Для платной модели задайте потолок IMAGE_MAX_SPEND в .env.")

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
    # Мир эпизода (эпоха/культура) судье здесь не передаётся: у рисованного
    # объяснялки главные браки — буквы и «не тот предмет», их ловят проверка
    # текста и утверждения спецификации. Облик фильма для «лучший как кадр» —
    # стиль канала.
    gen = Generator(backend, a.video_dir, profile, judge_gw=jgw, judge_model=judge_model,
                    look_style=profile["style"]["base"], variants=variants, rounds=rounds)
    print(f"Кадров: {len(todo)}, бэкенд {backend.name} ({backend.model}), вариантов {variants} x раундов до {rounds}, "
          f"судья {judge_model if jgw else '—'}")
    workers = int(os.environ.get("IMAGE_WORKERS", "1" if backend_kind == "comfyui" else "3"))
    with ThreadPoolExecutor(max(1, workers)) as ex:
        recs = list(ex.map(gen.frame, todo))
    path = write_report(a.video_dir, recs, {"backend": backend.name, "model": backend.model,
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
