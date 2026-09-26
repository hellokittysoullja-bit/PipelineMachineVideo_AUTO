#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Есть ли нужный кадр в пуле, и доходит ли он до проверки.

ЗАЧЕМ. Все прежние числа подбора мерили ПОБЕДИТЕЛЯ слота. Они не отвечают
на главный вопрос: плохой кадр на экране — потому что хорошего не было в
выдаче источников, или потому что он был, но отбор до него не дошёл
(фильтр выбросил, порядок пула поставил его 300-м при 20 местах). Это
разные болезни с разным лечением, и без замера их путают.

ВХОД — pools.jsonl прогона харнесса (selection_freeze.py): пул каждого
вызова отбора в том порядке, в каком его получает ранжирование, плюс
кандидаты, которых выбросил фильтр, с причиной (поле removed).

КОМАНДЫ
  sheet  RUN_DIR OUT_DIR [--k 12]
      Для каждого слота и вида — лист кандидатов на разметку: объединение
      первых K трёх порядков (как сейчас / по описанию кадра в пуле после
      фильтров / по описанию кадра в пуле ДО фильтров). Под плиткой — номер,
      источник, отметка «выброшен фильтром» и в каких порядках кандидат
      стоит в первых K. Номер плитки -> id кандидата лежит в OUT_DIR/index.json.
  recall RUN_DIR LABELS [--handoff 20]
      LABELS — json {"<слот>:<вид>": {"<номер плитки>": 2|1|0}}: 2 — точный
      кадр, 1 — годная замена, 0 — брак. Печатает по слоту лучшую метку в
      пуле до фильтров, после фильтров и в первых N каждого порядка.

  snapshot RUN_DIR... --labels L --episode EP --out FIXTURE
      Снимок для гейта регрессий: пулы всех прогонов (вход каскада), их
      эмбеддинги превью из дисковых кэшей каскада, метки и спецификации
      кадров эпизода. Метки — {"<слот>|<вид>|<id>": 0|1|2}, у каждой
      разметки своё имя оценщика (--rater); метки владельца старше.
  rankcheck FIXTURE [--write-baseline]
      Гейт регрессий: порядок пула считает ПРОД-код (cascade_reorder) на
      всех снятых пулах. Падает, если в каком-то пуле лучший размеченный
      кадр ушёл из первых --handoff (их видит судья) или годных там стало
      меньше в сумме. Формулу каскада нельзя сменить, не пройдя все прошлые
      пулы — так проверяется не один последний прогон, а все.

ЧЕСТНЫЕ ПРЕДЕЛЫ. «Лучшее в пуле» — лучшее среди РАЗМЕЧЕННЫХ: размечается
объединение верхушек порядков, а не весь пул в 700 кандидатов, поэтому
recall@pool — оценка снизу. Порядок «по описанию кадра» — тот же каскад,
что в проде (эмбеддинг превью моделью гейта против текста брифа), и тот же
дисковый кэш эмбеддингов, если он передан.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import urllib.parse
import urllib.request

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import runs_sheet  # noqa: E402

UA = "Mozilla/5.0 (X11; Linux x86_64) pool_recall"
TW, TH = 300, 170


def load_pools(run_dir):
    """{(слот, вид): запись} — первая запись каждого слота и вида (повторные
    попытки того же вида строят тот же пул)."""
    out = {}
    with open(os.path.join(run_dir, "pools.jsonl"), encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            out.setdefault((rec["index"], rec["kind"]), rec)
    return out


def section_last(rec, phrase_queries):
    """Порядок «сначала запросы фразы, запросы секции последними»: внутри
    каждой группы — прежний круг по запросам. Кандидат приходит в пул с
    меткой запроса (via), поэтому порядок восстанавливается без повторного
    поиска."""
    own = set(phrase_queries or []) | {rec.get("query")}
    first = [r for r in rec["pool"] if r.get("via") in own]
    rest = [r for r in rec["pool"] if r.get("via") not in own]
    return first + rest


class Embedder:
    """Эмбеддинг превью и текста моделью гейта pipeline_smart — ровно тот,
    что у каскада. Кэш: ключ как у каскада (модель + адрес превью, у Pixabay
    — номер кадра)."""

    def __init__(self, cache_dirs, write_dir):
        import pipeline_smart as ps
        self.ps = ps
        self.cache_dirs = [d for d in cache_dirs if d and os.path.isdir(d)]
        self.write_dir = write_dir
        self.dead = set()
        os.makedirs(write_dir, exist_ok=True)

    def _keys(self, url, cand=None):
        """Ключи кэша превью: сначала тот же, что у прод-каскада
        (_cascade_ident — у Pixabay по номеру кадра, подписанный адрес
        меняется и протухает), затем прежний по адресу — векторы, снятые
        до правки, остаются в деле."""
        keys = []
        if cand is not None:
            keys.append(self.ps._cascade_key(self.ps._cascade_ident(cand, url)))
        k = self.ps._cascade_key(url)
        if k not in keys:
            keys.append(k)
        return keys

    def image_vec(self, url, headers, cand=None):
        import numpy as np
        if not url:
            return None
        keys = self._keys(url, cand)
        for key in keys:
            for d in self.cache_dirs + [self.write_dir]:
                fp = os.path.join(d, key + ".npy")
                if os.path.exists(fp):
                    try:
                        return np.load(fp)
                    except Exception:
                        pass
        if url in self.dead:
            return None
        path = fetch(url, headers, (cand or {}).get("id") if isinstance(cand, dict) else None)
        if not path:
            # Мёртвый адрес (протухшая подпись Pixabay отвечает 400) не
            # запрашивается повторно на каждом порядке каждого слота.
            self.dead.add(url)
            return None
        key = keys[0]
        try:
            with Image.open(path) as im:
                vec = self.ps._gate_embed(images=[im.convert("RGB")])
        except Exception:
            vec = None
        finally:
            os.remove(path)
        if vec is None:
            return None
        np.save(os.path.join(self.write_dir, key + ".npy"), vec[0])
        return vec[0]

    def text_vec(self, text):
        v = self.ps._gate_embed(text=text)
        return None if v is None else v[0]


# Скачанные превью хранятся по номеру кандидата: ссылки Pixabay подписанные
# и протухают за сутки, и замер, повторённый назавтра, молча терял каждый
# кадр Pixabay (живой случай 25.09: второй прогон оценил две трети кадров, и
# числа вариантов стали несравнимы). Пустое — не хранить.
IMG_STORE = os.environ.get("POOL_RECALL_IMG_STORE", "")


def _store_path(cand_id):
    if not IMG_STORE or cand_id is None:
        return None
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(cand_id))
    return os.path.join(IMG_STORE, safe + ".jpg")


def _pixabay_fresh_url(cand_id):
    """Свежая ссылка на то же превью (640 px) через API Pixabay по номеру."""
    key = (os.environ.get("PIXABAY_API_KEY") or "").strip()
    if not key or not str(cand_id).startswith("pixabay:"):
        return None
    try:
        q = urllib.parse.urlencode({"key": key, "id": str(cand_id).split(":", 1)[1]})
        with urllib.request.urlopen(urllib.request.Request(
                "https://pixabay.com/api/?" + q, headers={"User-Agent": UA}), timeout=20) as r:
            hits = json.loads(r.read().decode("utf-8")).get("hits") or []
        return hits[0].get("webformatURL") if hits else None
    except Exception:
        return None


def _download(url, headers, path):
    h = {"User-Agent": UA}
    h.update(headers or {})
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=20) as r, \
                open(path, "wb") as f:
            f.write(r.read())
        return os.path.getsize(path) > 0
    except Exception:
        return False


def fetch(url, headers, cand_id=None):
    """Превью во временный файл; None при сбое. С IMG_STORE — сначала из
    хранилища по номеру кандидата; протухшая ссылка Pixabay — заново через
    API по номеру."""
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    stored = _store_path(cand_id)
    if stored and os.path.exists(stored):
        shutil.copyfile(stored, path)
        return path
    ok = bool(url) and _download(url, headers, path)
    if not ok:
        fresh = _pixabay_fresh_url(cand_id)
        ok = bool(fresh) and _download(fresh, {}, path)
    if ok:
        if stored:
            os.makedirs(IMG_STORE, exist_ok=True)
            shutil.copyfile(path, stored)
        return path
    os.remove(path)
    return None


def rank_by_text(rows, text, emb, kind="photo"):
    """Кандидаты по убыванию близости превью к тексту; без превью — в хвост
    в прежнем порядке."""
    t = emb.text_vec(text) if text else None
    if t is None:
        return list(rows)
    scored, rest = [], []
    for k, r in enumerate(rows):
        v = emb.image_vec(r.get("probe_url"), r.get("headers"), _cand(r, kind))
        if v is None:
            rest.append(r)
        else:
            scored.append((-float(v @ t), k, r))
    return [r for _s, _k, r in sorted(scored, key=lambda x: (x[0], x[1]))] + rest


VIDEO_MAX_TIME_STRETCH = 1.5   # как в pipeline_smart; держит тест синхронности


def base_slot_durs(pools):
    """{слот: настоящая длительность}. В режиме «только пул» слот остаётся
    без кадра, и его время уходит следующему (поглощение): slot_dur слота k
    = его собственная длительность + slot_dur слота k-1. Восстанавливается
    разностью соседних. Первый слот и слот после слота без записи — как есть."""
    sd = {}
    for (slot, _kind), rec in pools.items():
        if rec.get("slot_dur"):
            sd.setdefault(slot, float(rec["slot_dur"]))
    out = {}
    for slot in sorted(sd):
        prev = sd.get(slot - 1)
        base = sd[slot] - prev if prev is not None and sd[slot] > prev else sd[slot]
        out[slot] = round(base, 3)
    return out


def too_short(row, slot_dur):
    """Та же формула, что _video_candidate_too_short: неизвестная длина —
    годен (fail-open)."""
    try:
        dur = float(row.get("duration") or 0)
    except (TypeError, ValueError):
        return False
    return bool(slot_dur) and dur > 0 and dur * VIDEO_MAX_TIME_STRETCH < slot_dur


def full_rows(rec):
    """Все кандидаты в порядке ДО фильтра; выброшенные помечены _removed и
    причиной. Старая запись без prefilter_order — пул после фильтра, за ним
    выброшенные."""
    by_id = {}
    for r in rec["pool"]:
        by_id[str(r.get("id"))] = dict(r)
    for r in rec.get("removed", []):
        by_id[str(r.get("id"))] = dict(r, _removed=True)
    order = rec.get("prefilter_order")
    if not order:
        return list(by_id.values())
    return [by_id[str(i)] for i in order if str(i) in by_id]


def real_pool(rec, base_dur, blocklist=True):
    """Пул, который получил бы отбор в настоящем прогоне: запрещённые id и
    слишком короткие ролики (по настоящей длительности слота) выброшены;
    жанровый список запретов — если blocklist."""
    out = []
    for r in full_rows(rec):
        reason = r.get("reason") or ""
        if reason == "blocked_id":
            continue
        if blocklist and reason.startswith("blocklist:"):
            continue
        if rec["kind"] == "video" and too_short(r, base_dur):
            continue
        out.append(r)
    return out


def orders_for(rec, emb, phrase_queries, base_dur=None):
    """Именованные порядки кандидатов слота."""
    brief = rec.get("shot_brief") or rec.get("query")
    now = real_pool(rec, base_dur)
    noblock = real_pool(rec, base_dur, blocklist=False)
    as_rec = dict(rec, pool=now)
    return {
        "now": now,
        "section_last": section_last(as_rec, phrase_queries),
        "cascade": rank_by_text(now, brief, emb, rec.get("kind", "photo")),
        "cascade_noblock": rank_by_text(noblock, brief, emb, rec.get("kind", "photo")),
    }


def tile(row, label):
    path = fetch(row.get("probe_url"), row.get("headers"), row.get("id"))
    bg = Image.new("RGB", (TW, TH + 34), (16, 16, 16))
    if path:
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((TW, TH))
                bg.paste(im, ((TW - im.width) // 2, (TH - im.height) // 2))
        except Exception:
            pass
        os.remove(path)
    d = ImageDraw.Draw(bg)
    d.text((4, TH + 2), label, fill=(255, 220, 0) if row.get("_removed") else (230, 230, 230),
           font=runs_sheet._font(13))
    d.text((4, TH + 18), (row.get("text") or "")[:44], fill=(160, 160, 160), font=runs_sheet._font(11))
    return bg


def cmd_sheet(a):
    pools = load_pools(a.run_dir)
    plan = load_phrase_queries(a.phrase_queries)
    emb = Embedder(a.emb_cache or [], os.path.join(a.out_dir, "emb"))
    os.makedirs(a.out_dir, exist_ok=True)
    base = base_slot_durs(pools)
    print("  настоящие длительности слотов:", base)
    index = {}
    for (slot, kind), rec in sorted(pools.items()):
        orders = orders_for(rec, emb, plan.get(rec.get("block_text") or "", []), base.get(slot))
        picked, where = [], {}
        for name, rows in orders.items():
            for r in rows[:a.k]:
                key = str(r.get("id"))
                where.setdefault(key, []).append(name)
                if key not in {str(p.get("id")) for p in picked}:
                    picked.append(r)
        cols = 6
        rows_n = (len(picked) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * TW, 40 + rows_n * (TH + 34)), (0, 0, 0))
        d = ImageDraw.Draw(sheet)
        d.text((6, 4), f"слот {slot} {kind}: {rec.get('block_text')}", fill=(255, 255, 255),
               font=runs_sheet._font(16))
        d.text((6, 22), f"бриф: {rec.get('shot_brief')}"[:150], fill=(180, 180, 180), font=runs_sheet._font(12))
        key = f"{slot}:{kind}"
        index[key] = {}
        for n, r in enumerate(picked, 1):
            lab = f"#{n} {r.get('channel')} [{','.join(sorted(where[str(r.get('id'))]))}]"
            if r.get("_removed"):
                lab += " ВЫБРОШЕН: " + (r.get("reason") or "")
            sheet.paste(tile(r, lab[:52]), ((n - 1) % cols * TW, 40 + (n - 1) // cols * (TH + 34)))
            index[key][str(n)] = {"id": r.get("id"), "removed": bool(r.get("_removed")),
                                  "reason": r.get("reason"), "text": r.get("text")}
        out = os.path.join(a.out_dir, f"slot{slot:02d}_{kind}.jpg")
        sheet.save(out, quality=85)
        print(f"  {out}: {len(picked)} кандидатов")
    with open(os.path.join(a.out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)
    return 0


def load_phrase_queries(path):
    """{текст фразы: [запросы]} из media_plan/stock_queries.json."""
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return {u.get("text", ""): u.get("queries", []) for u in d.get("units", {}).values()}


def recall_table(orders, labels, handoff):
    """{порядок: лучшая метка в первых handoff}; плюс 'pool' и 'prefilter'."""
    def best(rows):
        vals = [labels[str(r.get("id"))] for r in rows if str(r.get("id")) in labels]
        return max(vals) if vals else None
    out = {name: best(rows[:handoff]) for name, rows in orders.items()}
    out["pool"] = best(orders["now"])
    out["pool_noblock"] = best(orders["cascade_noblock"])
    return out


def cmd_recall(a):
    pools = load_pools(a.run_dir)
    plan = load_phrase_queries(a.phrase_queries)
    with open(a.labels, encoding="utf-8") as f:
        raw = json.load(f)
    with open(a.index, encoding="utf-8") as f:
        index = json.load(f)
    emb = Embedder(a.emb_cache or [], os.path.join(os.path.dirname(a.index), "emb"))
    base = base_slot_durs(pools)
    table = {}
    for key, marks in raw.items():
        slot, kind = key.split(":")
        rec = pools.get((int(slot), kind))
        if not rec:
            continue
        labels = {str(index[key][n]["id"]): v for n, v in marks.items() if n in index.get(key, {})}
        orders = orders_for(rec, emb, plan.get(rec.get("block_text") or "", []), base.get(int(slot)))
        table[key] = recall_table(orders, labels, a.handoff)
        print(key, json.dumps(table[key], ensure_ascii=False))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(table, f, ensure_ascii=False, indent=1)
    return 0


def _label_rows(pools, index, labels, base, emb, plan, kind):
    """[(ключ, запись, [(строка, метка)] в порядке каскада)] размеченных слотов."""
    out = []
    for key, marks in labels.items():
        slot, k = key.split(":")
        if kind and k != kind:
            continue
        rec = pools.get((int(slot), k))
        if not rec:
            continue
        by_id = {str(index[key][n]["id"]): v for n, v in marks.items()}
        orders = orders_for(rec, emb, plan.get(rec.get("block_text") or "", []), base.get(int(slot)))
        rows, seen = [], set()
        for r in orders["cascade"] + orders["cascade_noblock"] + orders["now"]:
            rid = str(r.get("id"))
            if rid in by_id and rid not in seen:
                seen.add(rid)
                rows.append((r, by_id[rid]))
        out.append((key, rec, rows))
    return out


def rank_key(rank):
    """claims_vector -> сравнимый ключ: отказ ниже всего."""
    return (-9,) if rank is None else rank


def _plan_unit_texts(episode):
    """Тексты фраз плана — чтобы найти задание кусочка, на который монтаж
    разрезал длинную фразу."""
    try:
        with open(os.path.join(episode, "media_plan", "stock_queries.json"), encoding="utf-8") as f:
            units = json.load(f).get("units") or {}
    except (OSError, ValueError):
        return []
    return [v["text"] for v in units.values() if isinstance(v, dict) and v.get("text")]


def _spec_for_block(specs, unit_texts, block_text):
    """Задание слота так, как его получает рендер. Рендер привязывает задание
    к фразе ДО нарезки длинных фраз, и кусочки наследуют задание целой
    фразы; бенч видит только текст кусочка и искал задание по нему — на
    эп.93 так 9 слотов из 14 молча оценивались по брифу. Кусочек ищется
    внутри фраз плана; нашёлся не ровно в одной — задания нет."""
    import shot_planner_llm
    spec = specs.get(shot_planner_llm.unit_key(block_text))
    if spec or not block_text.strip():
        return spec
    norm = " ".join(block_text.split())
    hits = [t for t in unit_texts if norm in " ".join(t.split())]
    return specs.get(shot_planner_llm.unit_key(hits[0])) if len(hits) == 1 else None


def _screen_count(stats, head, pick):
    """Что окажется на экране слота: победитель-брак опустошает слот (его
    поглощает сосед), иначе на экране метка победителя. Главное мерило бенча:
    промежуточные счётчики («брак принят», «лучший выбран») не видят, что
    брак-победитель опустошает слот, — по ним 24.09 был снят вопрос о
    предмете, который по экрану оказался выигрышем."""
    rank, lab = head[pick][2], head[pick][1]
    best = max(x[1] for x in head)
    stats["screen_slots"] = stats.get("screen_slots", 0) + 1
    if rank is None or rank[0] == -5:
        stats["screen_empty"] = stats.get("screen_empty", 0) + 1
        stats["screen_empty_good"] = stats.get("screen_empty_good", 0) + (1 if best >= 1 else 0)
        return
    stats["screen_brak"] = stats.get("screen_brak", 0) + (1 if lab == 0 else 0)
    stats["screen_best"] = stats.get("screen_best", 0) + (1 if lab == best else 0)
    stats["screen_label_sum"] = stats.get("screen_label_sum", 0) + lab


def cmd_bench(a):
    """Проверка финалистов против разметки. Платно (шлюз): ответы кэшируются
    по байтам картинки и тексту вопроса — повторный прогон бесплатен."""
    import llm_gateway
    import shot_judge
    import world_card
    pools = load_pools(a.run_dir)
    plan = load_phrase_queries(a.phrase_queries)
    index = json.load(open(a.index, encoding="utf-8"))
    labels = json.load(open(a.labels, encoding="utf-8"))
    emb = Embedder(a.emb_cache or [], os.path.join(os.path.dirname(a.index), "emb"))
    base = base_slot_durs(pools)
    card = json.load(open(a.world_card, encoding="utf-8")) if a.world_card else None
    # Как в пайплайне: сетке — judge_setting, проверке по утверждениям —
    # claims_setting (или прежняя строка мира с --claims-world plain).
    grid_setting = world_card.judge_setting(card) if card else None
    setting = (world_card.world_to_check if a.claims_world == "plain"
               else world_card.claims_setting)(card) if card else None
    gw = llm_gateway.Gateway(spend_cap=a.max_spend)
    import shot_planner_llm
    import stock_query_planner
    _unit_key = shot_planner_llm.unit_key
    specs = stock_query_planner.load_specs(a.episode) if a.episode else {}
    unit_texts = _plan_unit_texts(a.episode) if a.episode else []
    if a.episode and not specs:
        # Эпизод передан, а спецификаций нет (план не той версии, битый
        # файл): бенч молча оценивал бы по брифу одним утверждением и
        # выдавал это за сравнение планов. Так 26.09 «честная база» оказалась
        # вовсе без плана — отказ громкий.
        raise SystemExit(f"--episode {a.episode}: спецификации кадров не загружены — "
                         f"сравнение планов без них недействительно")
    model = a.model
    cache = a.cache_dir or os.path.join(os.path.dirname(a.index), "judge_cache")
    stats = {"pairs": 0, "pairs_ok": 0.0, "bad_accepted": 0, "bad": 0, "good_vetoed": 0, "good": 0,
             "world_good_rejected": 0, "world_bad_passed": 0, "slots": [], "cost": 0}
    for key, rec, rows in _label_rows(pools, index, labels, base, emb, plan, a.kind):
        brief = rec.get("shot_brief") or rec.get("query")
        spec = _spec_for_block(specs, unit_texts, rec.get("block_text") or "")
        stats["spec_from_plan" if spec else "spec_from_brief"] = \
            stats.get("spec_from_plan" if spec else "spec_from_brief", 0) + 1
        spec = spec or shot_judge.spec_from_brief(rec.get("block_text"), brief)

        def ask(item):
            r, lab = item
            path = fetch(r.get("probe_url"), r.get("headers"), r.get("id"))
            if not path:
                return None
            try:
                # Проверка по утверждениям спецификации — как в пайплайне:
                # второй голос, если сомнение в must-утверждении.
                ans, info = shot_judge.verify_claims(
                    gw, model, phrase=rec.get("block_text"), spec=spec, setting=setting, path=path,
                    kind=rec["kind"], cache_dir=cache, max_side=a.side,
                    reasoning={"on": True, "off": False}.get(a.reasoning),
                    caption=r.get("text") if a.caption else None, frames=1,
                    world_separate=a.world_separate)
                wok, winfo = None, {}
                if a.world:
                    wok, _why, winfo = shot_judge.world_check(
                        gw, model, phrase=rec.get("block_text"), brief=brief, setting=setting,
                        path=path, kind=rec["kind"], cache_dir=cache)
                grid_path = None
                if a.grid:
                    grid_path = path + ".keep.jpg"
                    os.replace(path, grid_path)
                    path = grid_path
            finally:
                if not a.grid and os.path.exists(path):
                    os.remove(path)
            return r, lab, ans, wok, info.get("cost", 0) + winfo.get("cost", 0), grid_path

        import concurrent.futures
        if a.only_head:
            # Сокращённый замер: только те кадры, из которых выбирается
            # победитель слота (первые --handoff) — «выбран ли лучший»
            # считается по ним же, пары — внутри них.
            rows = rows[:a.handoff]
        if a.finalists:
            # Как в пайплайне: проверяются только лучшие по сетке (оценки
            # сетки — из кэша прошлого прогона со --grid).
            have = []
            for r, lab in rows[:a.handoff]:
                pth = fetch(r.get("probe_url"), r.get("headers"), r.get("id"))
                if pth:
                    have.append((r, lab, pth))
            gr = shot_judge.judge(gw, model, phrase=rec.get("block_text"), brief=spec["focus"],
                                  candidates=[(str(r.get("id")), pth) for r, _l, pth in have],
                                  cache_dir=a.grid_cache or cache, report={},
                                  kind=rec["kind"], setting=grid_setting) or {}
            for _r, _l, pth in have:
                os.remove(pth)
            # Как verify_finalists_of: лучшие по сетке ∪ первые по порядку пула.
            top = sorted(range(len(have)), key=lambda k: (-(gr.get(str(have[k][0].get("id")), -1)), k))
            keep = {id(have[k][0]) for k in top[:a.finalists] + list(range(min(a.finalists, len(have))))}
            rows = [(r, lab) for r, lab, _p in have if id(r) in keep]
        with concurrent.futures.ThreadPoolExecutor(a.workers) as ex:
            got = list(ex.map(ask, rows))
        grid = {}
        if a.grid:
            have = [(str(g[0].get("id")), g[5]) for g in got if g is not None and g[5]]
            rep = {}
            grid = shot_judge.judge(gw, model, phrase=rec.get("block_text"), brief=brief,
                                    candidates=have, cache_dir=a.grid_cache or cache, report=rep,
                                    kind=rec["kind"], setting=grid_setting) or {}
            stats["cost"] += rep.get("cost", 0)
            for _cid, gp in have:
                os.remove(gp)
        scored = []
        for g in got:
            if g is None:
                continue
            r, lab, ans, wok, cost, _gp = g
            stats["cost"] += cost
            if ans is None:
                continue
            rank = shot_judge.claims_vector(spec, ans, cg_veto=not world_card.renders_allowed(card))
            focus = shot_judge.focus_met(spec, ans)
            # «Принят» — не отклонён и хоть что-то обязательное выполнено;
            # считается ДО приписки оценки сетки: (-5,) + сетка уже не равно
            # (-5,), и прежний счётчик записывал такой кадр в принятые.
            gs = grid.get(str(r.get("id"))) if a.grid else None
            gs = gs if isinstance(gs, int) else None
            # То же правило брака, что в рендере (сетка 0 — брак).
            if rank is not None and shot_judge.shows_nothing(spec, ans, gs):
                rank = (-5,)
            accepted = rank is not None and rank != (-5,)
            if a.grid:
                rank = None if rank is None else rank + ((gs if gs is not None else -1),)
            scored.append((r, lab, rank, wok, ans))
            stats.setdefault("tiles", []).append({
                "key": key, "id": r.get("id"), "label": lab, "answers": ans, "world_ok": wok,
                "grid": grid.get(str(r.get("id"))) if a.grid else None, "pos": len(scored) - 1})
            if lab == 0:
                stats["bad"] += 1
                stats["bad_accepted"] += 1 if accepted else 0
                stats["world_bad_passed"] += 1 if wok else 0
            else:
                stats["good"] += 1
                stats["good_vetoed"] += 1 if rank is None else 0
                stats["world_good_rejected"] += 1 if wok is False else 0
        for i in range(len(scored)):
            for j in range(i + 1, len(scored)):
                a1, a2 = scored[i], scored[j]
                if a1[1] == a2[1]:
                    continue
                hi, lo = (a1, a2) if a1[1] > a2[1] else (a2, a1)
                stats["pairs"] += 1
                kh, kl = rank_key(hi[2]), rank_key(lo[2])
                stats["pairs_ok"] += 1.0 if kh > kl else 0.5 if kh == kl else 0.0
        head = scored[:a.handoff]
        if head:
            pick = max(range(len(head)), key=lambda k: (rank_key(head[k][2]), -k))
            _screen_count(stats, head, pick)
            stats["slots"].append({"key": key, "best": max(x[1] for x in head),
                                   "pick": head[pick][1], "pick_id": head[pick][0].get("id"),
                                   "pick_answers": head[pick][4]})
        print(f"  {key}: оценено {len(scored)}", flush=True)
    s = stats
    print(f"пары верно: {s['pairs_ok']:.1f}/{s['pairs']}; брак принят за точный: "
          f"{s['bad_accepted']}/{s['bad']}; годных отклонено: {s['good_vetoed']}/{s['good']}; "
          f"прежняя проверка мира: годных отклонено {s['world_good_rejected']}/{s['good']}, "
          f"брака пропущено {s['world_bad_passed']}/{s['bad']}; цена {s['cost']}")
    print(f"  на экране (слотов {s.get('screen_slots', 0)}): брак {s.get('screen_brak', 0)}, "
          f"лучший кадр {s.get('screen_best', 0)}, сумма меток {s.get('screen_label_sum', 0)}, "
          f"пустых {s.get('screen_empty', 0)} (из них при годном в первых {a.handoff}: "
          f"{s.get('screen_empty_good', 0)})")
    print(f"  спецификации: из плана {s.get('spec_from_plan', 0)}, "
          f"по брифу {s.get('spec_from_brief', 0)}")
    for sl in s["slots"]:
        print(f"  {sl['key']}: лучшее в первых {a.handoff} = {sl['best']}, выбрано = {sl['pick']}")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False, indent=1)
    return 0


# --- гейт регрессий каскада -------------------------------------------------

RATER_PRIORITY = ("owner", "claude")   # первым — чья метка побеждает


def _cand(row, kind):
    """Кандидат в той форме, которую ждёт cascade_reorder: id и признак
    видео (по нему у Pixabay выбирается имя превью в кэше)."""
    p = {"id": row.get("id"), "_probe": row.get("probe_url")}
    if kind == "video":
        p["video_files"] = [None]
    return p


def cmd_snapshot(a):
    import numpy as np
    import pipeline_smart as ps
    os.makedirs(a.out, exist_ok=True)
    n = ps.cascade_preview_n()
    pools, want = [], {}
    for run_dir in a.run_dir:
        name = os.path.basename(os.path.normpath(run_dir))
        for (index, kind), rec in sorted(load_pools(run_dir).items()):
            rows = [{"id": r.get("id"), "probe_url": r.get("probe_url")} for r in rec["pool"][:n]]
            pools.append({"run": name, "index": index, "kind": kind,
                          "block_text": rec.get("block_text"), "shot_brief": rec.get("shot_brief"),
                          "query": rec.get("query"), "rows": rows})
            for r in rows:
                key = ps._cascade_key(ps._cascade_ident(_cand(r, kind), r["probe_url"]))
                want[key] = None
    for d in a.emb_cache or []:
        for key in want:
            if want[key] is None:
                fp = os.path.join(d, key + ".npy")
                if os.path.exists(fp):
                    want[key] = np.load(fp)
    keys = sorted(k for k, v in want.items() if v is not None)
    np.savez_compressed(os.path.join(a.out, "emb.npz"), keys=np.array(keys),
                        vecs=np.stack([want[k] for k in keys]).astype(np.float16))
    import gzip
    with gzip.open(os.path.join(a.out, "pools.json.gz"), "wt", encoding="utf-8") as f:
        json.dump(pools, f, ensure_ascii=False)
    with open(os.path.join(a.episode, "media_plan", "stock_queries.json"), encoding="utf-8") as f:
        plan = json.load(f)
    texts = {rec.get("block_text") for rec in pools}
    plan = {"units": {k: u for k, u in (plan.get("units") or {}).items() if u.get("text") in texts}}
    with open(os.path.join(a.out, "specs.json"), "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=1)
    lab_path = os.path.join(a.out, "labels.json")
    labels = json.load(open(lab_path, encoding="utf-8")) if os.path.exists(lab_path) else {}
    with open(a.labels, encoding="utf-8") as f:
        new = json.load(f)
    labels.setdefault(a.rater, {}).update(
        {k: (v[0] if isinstance(v, list) else v) for k, v in new.items()})
    with open(lab_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=0, sort_keys=True)
    print(f"пулов {len(pools)}, эмбеддингов {len(keys)} из {len(want)}, "
          f"меток {sum(len(v) for v in labels.values())}")


def merged_labels(labels):
    """Одна метка на кадр: у кого выше приоритет, того и метка."""
    out = {}
    for rater in reversed(RATER_PRIORITY):
        out.update(labels.get(rater, {}))
    for rater, d in labels.items():
        if rater not in RATER_PRIORITY:
            for k, v in d.items():
                out.setdefault(k, v)
    return out


def fixture_orders(fix):
    """Порядок каждого снятого пула по ПРОД-коду каскада. Эмбеддинги превью
    — из снимка (в кэш каскада процесса), скачиваний нет."""
    import numpy as np
    import pipeline_smart as ps
    z = np.load(os.path.join(fix, "emb.npz"))
    for k, v in zip(z["keys"], z["vecs"]):
        ps._CASCADE_EMB[str(k)] = v.astype(np.float32)
    import gzip
    with gzip.open(os.path.join(fix, "pools.json.gz"), "rt", encoding="utf-8") as f:
        pools = json.load(f)
    plan = json.load(open(os.path.join(fix, "specs.json"), encoding="utf-8"))
    specs = {u.get("text"): u for u in ((plan or {}).get("units") or {}).values()}

    def no_probe(p, path):
        raise OSError("снимок: превью без эмбеддинга не скачивается")

    import shutil
    out = {}
    tmp = tempfile.mkdtemp(prefix="rankcheck_")
    try:
        for rec in pools:
            kind = rec["kind"]
            cands = [_cand(r, kind) for r in rec["rows"]]
            spec = specs.get(rec.get("block_text"))
            brief = rec.get("shot_brief") or rec.get("query")
            ranked = ps.cascade_reorder(
                cands, ps.cascade_texts(spec, brief, kind), os.path.join(tmp, "x"), no_probe,
                index=rec["index"], url_of=lambda p: p["_probe"],
                claims=ps.cascade_claims(spec, kind))
            out[f"{rec['run']}|{rec['index']}|{kind}"] = [str(p["id"]) for p in ranked]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def rank_metrics(orders, labels, handoff):
    """По каждому пулу: лучшая метка среди первых handoff и сколько там
    годных (метка >= 1). Пулы без меток в первых handoff не считаются."""
    lab = merged_labels(labels)
    out = {}
    for key, ids in orders.items():
        run, index, kind = key.split("|")
        head = [lab.get(f"{index}|{kind}|{i}") for i in ids[:handoff]]
        head = [x for x in head if x is not None]
        pool = [lab.get(f"{index}|{kind}|{i}") for i in ids]
        pool = [x for x in pool if x is not None]
        if not pool:
            continue
        out[key] = {"best": max(head) if head else None, "good": sum(1 for x in head if x >= 1),
                    "best_in_pool": max(pool)}
    return out


def rankcheck_failures(now, base):
    """Что стало хуже против базовой линии: лучший кадр ушёл из головы в
    каком-то пуле, либо годных в головах стало меньше в сумме."""
    bad = []
    for key, b in base.items():
        m = now.get(key)
        if m is None:
            bad.append(f"{key}: пул пропал из замера")
            continue
        if (m["best"] if m["best"] is not None else -1) < (b["best"] if b["best"] is not None else -1):
            bad.append(f"{key}: лучший в первых — {m['best']} вместо {b['best']}")
    g_now = sum(m["good"] for m in now.values())
    g_base = sum(b["good"] for b in base.values())
    if g_now < g_base:
        bad.append(f"годных в первых всего {g_now} вместо {g_base}")
    return bad


def cmd_rankcheck(a):
    labels = json.load(open(os.path.join(a.fixture, "labels.json"), encoding="utf-8"))
    now = rank_metrics(fixture_orders(a.fixture), labels, a.handoff)
    base_path = os.path.join(a.fixture, "baseline.json")
    for key in sorted(now):
        m = now[key]
        print(f"  {key:22} лучший {m['best']}  годных {m['good']:2}  (лучший в пуле {m['best_in_pool']})")
    print(f"годных в первых {a.handoff}: {sum(m['good'] for m in now.values())}")
    if a.write_baseline:
        with open(base_path, "w", encoding="utf-8") as f:
            json.dump({"handoff": a.handoff, "metrics": now}, f, ensure_ascii=False, indent=1, sort_keys=True)
        print(f"базовая линия записана: {base_path}")
        return 0
    base = json.load(open(base_path, encoding="utf-8"))
    bad = rankcheck_failures(now, base["metrics"])
    for line in bad:
        print("  ХУЖЕ:", line)
    print("гейт регрессий:", "ПРОЙДЕН" if not bad else f"НЕ ПРОЙДЕН ({len(bad)})")
    return 1 if bad else 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheet")
    s.add_argument("run_dir")
    s.add_argument("out_dir")
    s.add_argument("--k", type=int, default=12)
    r = sub.add_parser("recall")
    r.add_argument("run_dir")
    r.add_argument("labels")
    r.add_argument("--index", required=True)
    r.add_argument("--handoff", type=int, default=20)
    r.add_argument("--out")
    b = sub.add_parser("bench")
    b.add_argument("run_dir")
    b.add_argument("labels")
    b.add_argument("--index", required=True)
    b.add_argument("--world-card")
    b.add_argument("--model", default="qwen/qwen3.7-plus")
    b.add_argument("--kind", default="photo")
    b.add_argument("--handoff", type=int, default=10)
    b.add_argument("--max-spend", type=int, default=150000)
    b.add_argument("--workers", type=int, default=8)
    b.add_argument("--side", type=int, default=512, help="сторона картинки для проверки")
    b.add_argument("--reasoning", choices=("default", "on", "off"), default="default")
    b.add_argument("--caption", action="store_true", help="подпись источника в вопрос")
    b.add_argument("--episode", help="папка эпизода: спецификации кадров из плана фраз v3")
    b.add_argument("--finalists", type=int, default=0,
                   help="проверять только N лучших по сетке (как в пайплайне)")
    b.add_argument("--grid-cache", help="кэш оценок сетки (повтор без кэша проверки)")
    b.add_argument("--cache-dir", help="кэш ответов (по умолчанию рядом с index.json)")
    b.add_argument("--world", action="store_true", help="плюс прежняя проверка мира")
    b.add_argument("--only-head", action="store_true",
                   help="только первые --handoff кадров слота (вдвое дешевле)")
    b.add_argument("--world-separate", action="store_true",
                   help="мир — отдельным вопросом без фразы, один на картинку")
    b.add_argument("--claims-world", choices=("exclude", "plain"), default="exclude",
                   help="строка мира проверки: с культурами «исключить» (как в пайплайне) или прежняя")
    b.add_argument("--grid", action="store_true", help="плюс сетка судьи: разводит равные уровни")
    b.add_argument("--out")
    b.set_defaults(fn=cmd_bench)
    sn = sub.add_parser("snapshot")
    sn.add_argument("run_dir", nargs="+")
    sn.add_argument("--labels", required=True)
    sn.add_argument("--rater", default="claude")
    sn.add_argument("--episode", required=True)
    sn.add_argument("--out", required=True)
    sn.add_argument("--emb-cache", action="append")
    sn.set_defaults(fn=cmd_snapshot)
    rc = sub.add_parser("rankcheck")
    rc.add_argument("fixture")
    rc.add_argument("--handoff", type=int, default=20)
    rc.add_argument("--write-baseline", action="store_true")
    rc.set_defaults(fn=cmd_rankcheck)
    for sp in (s, r, b):
        sp.add_argument("--phrase-queries", help="media_plan/stock_queries.json эпизода")
        sp.add_argument("--emb-cache", action="append",
                        help="папка кэша эмбеддингов каскада (можно несколько)")
    s.set_defaults(fn=cmd_sheet)
    r.set_defaults(fn=cmd_recall)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
