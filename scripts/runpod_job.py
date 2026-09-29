#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Задача на видеокарте Runpod: под живёт ровно столько, сколько идёт задача.

    python scripts/runpod_job.py --plan                       # цены и наличие, ничего не создаёт
    python scripts/runpod_job.py --upload . --cmd "python scripts/calibrate_vision.py" \\
        --fetch assets/calibration --fetch docs/quality

ЗАЧЕМ. У Runpod посекундная тарификация подов, а удалённый под не стоит ничего
(диск контейнера после удаления не оплачивается). Значит, «видеокарта работает
только пока идёт задача» — это не тариф, а дисциплина: создать под под задачу
и удалить сразу после неё. Этот скрипт так и делает:

  1. создаёт под на самой дешёвой свободной карте community, которая вмещает
     модели (от 24 ГБ; --gpu задаёт свой список), без сетевого диска: сетевой
     диск оплачивается и тогда, когда пода нет;
  2. ждёт исполнителя (scripts/runpod_runner.py, передаётся в переменной
     окружения пода — образ стандартный);
  3. загружает папки (--upload) архивом, кусками — прокси Runpod не любит
     больших тел запроса; докачка по смещению после обрыва;
  4. запускает команду (--cmd) и печатает её лог по мере работы;
  5. забирает результаты (--fetch) и УДАЛЯЕТ под — в finally, при Ctrl-C и
     при любой ошибке; удаление проверяется запросом к API.

Если эта сторона пропала (закрыли сессию, упала сеть), под удаляет себя сам:
исполнитель следит за простоем без задачи и за жёстким потолком времени
(--idle-min, --max-hours). Третья страховка — лимит трат аккаунта Runpod.

Ключ — RUNPOD_API_KEY из .env. В под он НЕ передаётся: там свой ключ,
ограниченный самим подом (им под и удаляет себя). Секреты для задачи в поде
(--env-from-dotenv KEY1,KEY2) передаются явно и поимённо — ни одной лишней
переменной.

Честно о «только эти секунды»: оплачивается жизнь пода целиком — скачивание
образа и весов моделей (~20 ГБ внутри дата-центра, минуты), сама задача и
ожидание сети внутри задачи (судья, источники). Карта не простаивает МЕЖДУ
задачами — её просто нет.
"""
import argparse
import base64
import io
import json
import os
import secrets
import signal
import sys
import tarfile
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
API = "https://api.runpod.io/graphql"
DEFAULT_GPUS = ("NVIDIA GeForce RTX 4090", "NVIDIA L40S", "NVIDIA RTX 6000 Ada Generation",
                "NVIDIA RTX A6000")   # образец порядка для --gpu; по умолчанию — cheapest_gpus
# torch 2.8 + CUDA 12.8: запускается и на картах Blackwell (RTX 5090 и др.).
DEFAULT_IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
PORT = 8000
# Cloudflare перед api.runpod.io отвечает 403 на стандартную подпись клиента
# Python (проверено 29.09: curl проходит, urllib — нет); та же защита, что у
# шлюза моделей.
UA = "pipeline-runpod-job/1.0"
CHUNK = 32 * 1024 * 1024
# Что не едет в под: кэши, записи прогонов, веса, секреты — всё это либо
# скачается там заново, либо не должно покидать машину.
EXCLUDE_DIRS = {".git", "__pycache__", ".venv", "venv", "temp_smart", "models", "node_modules",
                ".pytest_cache", "temp_rerank_cache", "temp_cascade_embed_cache"}
EXCLUDE_FILES = {".env"}


def api_key():
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not key:
        try:
            from dotenv import dotenv_values
            key = (dotenv_values(os.path.join(REPO, ".env")).get("RUNPOD_API_KEY") or "").strip()
        except Exception:  # noqa: BLE001
            pass
    if not key:
        raise SystemExit("нет RUNPOD_API_KEY (в окружении или в .env)")
    return key


def gql(query, key, variables=None):
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request(API, data=body, method="POST", headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {key}", "User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.loads(r.read())
    except urllib.error.HTTPError as e:
        # Тело ответа с причиной (ошибка схемы, нет карты) — не терять.
        raise RuntimeError(f"HTTP {e.code}: {e.read()[:500].decode('utf-8', 'replace')}") from None
    if d.get("errors"):
        raise RuntimeError("; ".join(e.get("message", str(e)) for e in d["errors"]))
    return d["data"]


# Сколько видеопамяти нужно одной карте: две модели зрения — 19.1 ГиБ весов
# плюс запас (vision_model.WEIGHTS_GIB / HEADROOM_GIB). 24 ГБ — впритык, но
# помещаются; меньше — рендер откажет.
MIN_GPU_GB = 24


def plan(key, gpus=None, community=True):
    """Карты и цены. gpus=None — все типы Runpod (для выбора самой дешёвой).
    community — цена и наличие ТОЛЬКО по community-облаку (secureCloud:false):
    без этого Runpod отдаёт самую низкую цену по обоим облакам, и карта,
    свободная только в secure, выглядела бы свободной и дешёвой (проверено
    29.09: RTX A6000 «$0.33 в наличии» без фильтра и нет её с фильтром)."""
    lp = "lowestPrice(input:{gpuCount:1%s})" % (", secureCloud:false" if community else "")
    if gpus:
        q = ('query($ids:[String!]){ gpuTypes(input:{ids:$ids}){ id displayName memoryInGb '
             'securePrice communityPrice ' + lp + '{ uninterruptablePrice '
             'stockStatus } } myself { clientBalance spendLimit currentSpendPerHr } }')
        return gql(q, key, {"ids": list(gpus)})
    q = ('query { gpuTypes { id displayName memoryInGb securePrice communityPrice communityCloud '
         + lp + '{ uninterruptablePrice stockStatus } } '
         'myself { clientBalance spendLimit currentSpendPerHr } }')
    return gql(q, key)


# Карты Blackwell (sm_120) требуют CUDA 12.8 и torch от 2.7: на образе
# старше модель не запустится, и оплаченный под упал бы на первом проходе.
BLACKWELL = ("5090", "5080", "B200", "B300", "RTX PRO")


def image_supports_blackwell(image):
    import re
    m = re.search(r"cuda(\d+)\.(\d+)|cu(\d{2})(\d)", image or "")
    if not m:
        return False
    major, minor = (int(m.group(1)), int(m.group(2))) if m.group(1) else (int(m.group(3)), int(m.group(4)))
    return (major, minor) >= (12, 8)


def cheapest_gpus(gpu_types, min_gb=MIN_GPU_GB, image=DEFAULT_IMAGE):
    """Карты community, которые сейчас есть в наличии и вмещают модели, —
    от дешёвой к дорогой (по текущей цене, а не прейскуранту: у карты без
    свободных машин цены «сейчас» нет)."""
    rows = []
    for g in gpu_types:
        lp = g.get("lowestPrice") or {}
        price = lp.get("uninterruptablePrice")
        name = f"{g.get('id', '')} {g.get('displayName', '')}"
        if any(t in name for t in BLACKWELL) and not image_supports_blackwell(image):
            continue
        if (g.get("communityCloud") and price is not None and lp.get("stockStatus")
                and (g.get("memoryInGb") or 0) >= min_gb):
            rows.append((float(price), -(g.get("memoryInGb") or 0), g["id"]))
    return [gid for _p, _m, gid in sorted(rows)]


def runner_env(token, idle_min, max_hours, extra):
    import gzip
    src = open(os.path.join(HERE, "runpod_runner.py"), "rb").read()
    # Сжатие: значение переменной окружения пода втрое короче (~4 КБ вместо 13).
    env = {"RUNNER_TOKEN": token, "RUNNER_B64": base64.b64encode(gzip.compress(src)).decode(),
           "RUNNER_IDLE_SEC": str(int(idle_min * 60)), "RUNNER_MAX_SEC": str(int(max_hours * 3600)),
           "RUNNER_PORT": str(PORT)}
    env.update(extra)
    return [{"key": k, "value": v} for k, v in env.items()]


# Команда старта контейнера — СПИСКОМ аргументов (REST API Runpod,
# dockerStartCmd), а не строкой: строковое поле старого API разбирается
# самим Runpod, и ошибку кавычек там не видно ни в ответе, ни в статусе пода
# (две первые аренды 29.09 простояли с контейнером, который так и не
# ответил). Список идёт в контейнер как есть.
START_ARGV = ["bash", "-c",
              'echo "$RUNNER_B64" | base64 -d | gunzip > /runner.py && exec python3 /runner.py']
REST = "https://rest.runpod.io/v1"
SMOKE_IMAGE = "python:3.11-slim"


def rest(method, path, key, body=None):
    """REST API Runpod. None — 404 (пода нет)."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(REST + path, data=data, method=method, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {key}", "User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise RuntimeError(f"HTTP {e.code}: {e.read()[:500].decode('utf-8', 'replace')}") from None
    return json.loads(raw) if raw.strip() else {}


def budget_seconds(price_per_hr, max_usd, max_hours):
    """Сколько секунд под может жить: меньшее из потолка времени и потолка
    денег при цене этой карты."""
    cap = max_hours * 3600
    if max_usd and price_per_hr:
        cap = min(cap, max_usd / float(price_per_hr) * 3600)
    return cap


def create_pod(key, gpus, image, disk_gb, env, cloud, cpu=False):
    """env — список переменных или функция gpu -> список (у каждой карты
    своя цена, а значит и свой потолок жизни пода в секундах). cpu — под без
    видеокарты (проверка пути за доли цента, см. --smoke)."""
    last = None
    for gpu in ([None] if cpu else gpus):
        e = env(gpu) if callable(env) else env
        body = {"name": "pipeline-job", "imageName": image, "containerDiskInGb": disk_gb,
                "volumeInGb": 0, "ports": [f"{PORT}/http"], "dockerStartCmd": START_ARGV,
                "env": {x["key"]: x["value"] for x in e}, "cloudType": cloud}
        if cpu:
            body.update(computeType="CPU", cpuFlavorIds=["cpu3c", "cpu5c", "cpu3g"], vcpuCount=2)
        else:
            body.update(computeType="GPU", gpuTypeIds=[gpu], gpuCount=1)
            if cloud == "COMMUNITY":
                body["supportPublicIp"] = False
        try:
            pod = rest("POST", "/pods", key, body)
            if pod and pod.get("id"):
                pod.setdefault("gpuName", "CPU" if cpu else gpu)
                return pod
        except RuntimeError as e:
            last = e
            print(f"  {gpu or 'CPU'}: нет — {e}")
    raise SystemExit(f"ни одной машины из списка нет в наличии ({last})")


def pod_status(key, pod_id):
    """(секунд работы контейнера или None, статус пода)."""
    pod = gql('query($id:String!){ pod(input:{podId:$id}){ desiredStatus lastStatusChange '
              'runtime { uptimeInSeconds } } }', key, {"id": pod_id}).get("pod") or {}
    rt = pod.get("runtime") or {}
    # Пока тянется образ, Runpod отдаёт runtime с uptimeInSeconds = 0 (замер
    # 29.09 на живом поде): «работает» — только положительное время.
    up = rt.get("uptimeInSeconds") or 0
    return (up if up > 0 else None), f"{pod.get('desiredStatus')} ({pod.get('lastStatusChange')})"


def terminate(key, pod_id):
    """Удалить под и убедиться, что его больше нет. Повторяет — удаление
    обязано случиться, иначе карта стоит за деньги."""
    for attempt in range(6):
        try:
            rest("DELETE", f"/pods/{pod_id}", key)
        except Exception as e:  # noqa: BLE001 — запасной путь: старый API
            print(f"  удаление пода (REST): {e}")
            try:
                gql('mutation($id:String!){ podTerminate(input:{podId:$id}) }', key, {"id": pod_id})
            except Exception as e2:  # noqa: BLE001
                print(f"  удаление пода: {e2}")
        try:
            left = rest("GET", f"/pods/{pod_id}", key)
        except Exception:  # noqa: BLE001
            left = {"id": pod_id}
        if not left or left.get("desiredStatus") == "TERMINATED":
            print(f"  под {pod_id} удалён")
            return True
        time.sleep(5 * (attempt + 1))
    print(f"  ВНИМАНИЕ: под {pod_id} не подтвердил удаление — проверьте https://www.runpod.io/console/pods")
    return False


class Runner:
    def __init__(self, base, token):
        self.base, self.token = base.rstrip("/"), token

    def call(self, method, path, body=None, timeout=90, raw=False):
        req = urllib.request.Request(self.base + path, data=body, method=method,
                                     headers={"X-Runner-Token": self.token, "User-Agent": UA})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
        return data if raw else json.loads(data)

    def wait_ready(self, limit_sec, status_fn=None, run_grace_sec=300):
        """Ждать исполнителя. status_fn() -> (секунд работы контейнера или
        None, статус) — чтобы отличать «образ ещё тянется» (ждать) от
        «контейнер работает, а исполнитель не отвечает» (сразу стоп: деньги
        идут, а ждать нечего). Каждая смена состояния печатается."""
        t0, last_err, shown, next_status = time.time(), None, None, 0.0
        while time.time() - t0 < limit_sec:
            try:
                if self.call("GET", "/health", timeout=15).get("ok"):
                    self.call("GET", "/log?offset=0")   # клиент на месте — таймер простоя с нуля
                    print(f"  исполнитель готов через {time.time() - t0:.0f} с")
                    return True
            except Exception as e:  # noqa: BLE001 — под ещё поднимается
                last_err = f"{type(e).__name__}: {getattr(e, 'code', '') or e}"[:120]
            if status_fn is not None and time.time() >= next_status:
                next_status = time.time() + 20
                try:
                    uptime, status = status_fn()
                except Exception as e:  # noqa: BLE001
                    uptime, status = None, f"статус не спросился ({e})"
                state = (status, uptime is not None)
                if state != shown:
                    shown = state
                    print(f"  {time.time() - t0:4.0f} с: под {status}, контейнер "
                          f"{'работает' if uptime is not None else 'ещё не запущен (тянется образ)'}; "
                          f"исполнитель: {last_err}")
                if uptime is not None and uptime > run_grace_sec:
                    print(f"  контейнер работает {uptime} с, а исполнитель не отвечает ({last_err})")
                    return False
            # Под оплачивается посекундно с момента создания: чем раньше
            # замечена готовность, тем меньше секунд простоя до задачи.
            time.sleep(2)
        return False

    def upload_dir(self, path):
        """Папка едет в /work по своему пути относительно репозитория (корень
        репозитория — в сам /work), чтобы команды на поде видели ту же
        раскладку, что и здесь."""
        full = os.path.abspath(path)
        rel = os.path.relpath(full, REPO)
        arc = "." if rel == "." else (rel if not rel.startswith("..") else os.path.basename(full))
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(full, arcname=arc, filter=_exclude)
        data = buf.getvalue()
        fname = f"up_{secrets.token_hex(4)}.tar.gz"
        off = 0
        while off < len(data):
            piece = data[off:off + CHUNK]
            try:
                off = self.call("PUT", f"/upload?name={fname}&offset={off}", piece, timeout=300)["size"]
            except urllib.error.HTTPError as e:
                if e.code != 409:
                    raise
                off = json.loads(e.read())["size"]     # докачка с того места, где под остановился
            print(f"  загрузка {path}: {off / 2**20:.0f} / {len(data) / 2**20:.0f} МБ", end="\r")
        print()
        self.call("POST", f"/extract?name={fname}", b"")

    def run(self, cmd, deadline=None):
        """deadline — момент (time.time()), когда потолок денег исчерпан:
        задача прерывается, под удаляется вызывающим (finally)."""
        self.call("POST", "/run", json.dumps({"cmd": cmd}).encode())
        off = 0
        while True:
            if deadline is not None and time.time() > deadline:
                raise SystemExit("потолок денег на запуск исчерпан — задача прервана, под удаляется")
            st = self.call("GET", f"/log?offset={off}")
            if st["text"]:
                sys.stdout.write(st["text"])
                sys.stdout.flush()
            off = st["offset"]
            if not st["running"] and st["exit"] is not None and not st["text"]:
                return st["exit"]
            time.sleep(3 if st["text"] else 10)

    def fetch(self, path, dest):
        data = self.call("GET", f"/download?path={path}", timeout=600, raw=True)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            tar.extractall(dest)


def _exclude(info):
    parts = set(info.name.replace("\\", "/").split("/"))
    if parts & EXCLUDE_DIRS or os.path.basename(info.name) in EXCLUDE_FILES \
            or os.path.basename(info.name).startswith(".env."):
        return None
    return info


def dotenv_subset(names):
    if not names:
        return {}
    from dotenv import dotenv_values
    vals = dotenv_values(os.path.join(REPO, ".env"))
    out = {}
    for n in names:
        if vals.get(n):
            out[n] = vals[n]
        else:
            print(f"  {n}: нет в .env — в под не передаётся")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--plan", action="store_true", help="цены и наличие карт, ничего не создаёт")
    p.add_argument("--no-smoke", action="store_true",
                   help="не проверять путь на CPU-поде перед арендой видеокарты")
    p.add_argument("--smoke", action="store_true",
                   help="проверить весь путь (под, прокси, исполнитель, загрузка, лог, возврат, "
                        "удаление) на поде БЕЗ видеокарты с маленьким образом — доли цента")
    p.add_argument("--gpu", action="append",
                   help="тип карты (можно несколько, по порядку); без него — самые дешёвые "
                        "свободные карты community от --min-gb")
    p.add_argument("--min-gb", type=int, default=MIN_GPU_GB)
    p.add_argument("--cloud", default="COMMUNITY", choices=("COMMUNITY", "SECURE", "ALL"))
    p.add_argument("--image", default=None, help=f"образ (по умолчанию {DEFAULT_IMAGE}; "
                                                f"для --smoke — {SMOKE_IMAGE})")
    p.add_argument("--disk-gb", type=int, default=80)
    p.add_argument("--upload", action="append", default=[], help="папка, едет в /work")
    p.add_argument("--cmd", help="команда в /work на поде")
    p.add_argument("--fetch", action="append", default=[], help="путь в /work, вернуть сюда")
    p.add_argument("--dest", default=".")
    p.add_argument("--env-from-dotenv", default="", help="KEY1,KEY2 — передать в под из .env")
    p.add_argument("--idle-min", type=float, default=10)
    p.add_argument("--max-hours", type=float, default=4)
    p.add_argument("--max-usd", type=float, default=2.0,
                   help="потолок денег на запуск: под удаляется, когда его цена дошла до лимита")
    a = p.parse_args(argv)
    key = api_key()
    if a.smoke:
        return smoke(key, a)
    a.image = a.image or DEFAULT_IMAGE
    info = plan(key, a.gpu, community=a.cloud == "COMMUNITY")
    me = info["myself"]
    print(f"Баланс Runpod ${me['clientBalance']:.2f}, лимит трат ${me['spendLimit']}/ч, "
          f"сейчас тратится ${me['currentSpendPerHr']}/ч")
    by_id = {g["id"]: g for g in info["gpuTypes"]}
    gpus = a.gpu or cheapest_gpus(info["gpuTypes"], a.min_gb, a.image)[:6]
    if not gpus:
        print(f"  нет свободных карт community от {a.min_gb} ГБ")
        if a.plan:
            return 0
        raise SystemExit(1)
    for gid in gpus:
        g = by_id.get(gid, {"displayName": gid, "memoryInGb": "?"})
        lp = g.get("lowestPrice") or {}
        print(f"  {g['displayName']:<16} {g['memoryInGb']} ГБ  сейчас "
              f"{lp.get('uninterruptablePrice')} $/ч  наличие {lp.get('stockStatus')}")
    if a.plan:
        return 0
    if not a.cmd:
        raise SystemExit("нет --cmd")
    if not a.no_smoke:
        # Перед арендой видеокарты — тот же образ на поде без неё (доли
        # цента): старт контейнера, исполнитель, прокси, удаление. Путь не
        # работает — карта не арендуется вообще.
        print("Проверка пути до аренды видеокарты:")
        if smoke(key, a) != 0:
            raise SystemExit("проверка пути не пройдена — видеокарта не арендована, деньги не потрачены")
    token = secrets.token_urlsafe(32)
    extra = dotenv_subset([n.strip() for n in a.env_from_dotenv.split(",") if n.strip()])

    def price(gid):
        lp = (by_id.get(gid) or {}).get("lowestPrice") or {}
        return lp.get("uninterruptablePrice")

    def env_for(gid):
        # Потолок жизни пода на его стороне — по цене ЭТОЙ карты: если
        # связь пропадёт, под всё равно не проживёт дольше, чем на --max-usd.
        return runner_env(token, a.idle_min, budget_seconds(price(gid), a.max_usd, a.max_hours) / 3600,
                          extra)
    pod = create_pod(key, gpus, a.image, a.disk_gb, env_for, a.cloud)
    cap_sec = budget_seconds(pod["costPerHr"], a.max_usd, a.max_hours)
    print(f"Под {pod['id']}: {pod['gpuName']}, ${pod['costPerHr']}/ч; потолок "
          f"${a.max_usd:.2f} = {cap_sec / 60:.0f} мин, самоудаление через {a.idle_min:.0f} мин простоя")
    return drive(key, pod, token, cap_sec, a.upload, a.cmd, a.fetch, a.dest)


def drive(key, pod, token, cap_sec, uploads, cmd, fetches, dest):
    """Под создан: дождаться исполнителя, загрузить, запустить, забрать,
    и удалить под в ЛЮБОМ исходе."""
    pod_id, t0, code = pod["id"], time.time(), 1

    def _sigint(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _sigint)
    try:
        r = Runner(f"https://{pod_id}-{PORT}.proxy.runpod.net", token)
        if not r.wait_ready(min(1800, cap_sec), lambda: pod_status(key, pod_id)):
            raise SystemExit("исполнитель на поде не поднялся")
        for path in uploads:
            r.upload_dir(path)
        code = r.run(cmd, deadline=t0 + cap_sec)
        print(f"\nКоманда завершилась с кодом {code}")
        for path in fetches:
            r.fetch(path, dest)
            print(f"  забрано: {path}")
    finally:
        terminate(key, pod_id)
        sec = time.time() - t0
        print(f"Под жил {sec / 60:.1f} мин ≈ ${sec / 3600 * float(pod['costPerHr']):.2f}")
    return code


def smoke(key, a):
    """Весь путь на поде без видеокарты: REST-создание, команда старта,
    исполнитель, прокси Runpod, загрузка, запуск, лог, возврат, удаление.
    Стоит доли цента — ошибка пути ловится здесь, а не на аренде видеокарты.
    --image задаёт образ (например, образ видеокарты — проверить, за сколько
    он скачивается и стартует)."""
    import tempfile
    image = a.image or SMOKE_IMAGE
    token = secrets.token_urlsafe(32)
    env = runner_env(token, a.idle_min, min(a.max_hours, 0.5), {})
    print(f"Проверка пути: под без видеокарты, образ {image}")
    pod = create_pod(key, [], image, 20, env, "SECURE", cpu=True)
    print(f"Под {pod['id']}: CPU, ${pod['costPerHr']}/ч")
    src = tempfile.mkdtemp(prefix="smoke_up_")
    with open(os.path.join(src, "probe.txt"), "w", encoding="utf-8") as f:
        f.write("ping")
    out = tempfile.mkdtemp(prefix="smoke_out_")
    cmd = (f"python3 -c \"import os;d=open('{os.path.basename(src)}/probe.txt').read();"
           f"os.makedirs('res',exist_ok=True);open('res/pong.txt','w').write(d+'-pong');"
           f"print('исполнитель жив', d)\"")
    code = drive(key, pod, token, 1800, [src], cmd, ["res"], out)
    got = open(os.path.join(out, "res", "pong.txt"), encoding="utf-8").read() \
        if os.path.exists(os.path.join(out, "res", "pong.txt")) else None
    ok = code == 0 and got == "ping-pong"
    print("ПРОВЕРКА ПУТИ:", "пройдена" if ok else f"НЕ пройдена (код {code}, ответ {got!r})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
