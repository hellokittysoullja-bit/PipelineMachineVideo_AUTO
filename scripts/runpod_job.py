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
# Релизная сборка образа с ОБЫЧНЫМ torch 2.8.0. Прежний образ
# (2.8.0-py3.11-cuda12.8.1-cudnn-devel, март 2025) нёс ночную сборку
# torch 2.8.0.dev — на ней живой прогон 29.09 на L40S получил «CUDA unknown
# error», и калибровка отказала без видеокарты.
DEFAULT_IMAGE = "runpod/pytorch:1.3.3-cu1281-torch280-ubuntu2204"
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
    need = image_cuda(image)
    return need is not None and need >= (12, 8)


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
# Версии CUDA, которые принимает фильтр хостов REST API (allowedCudaVersions).
CUDA_VERSIONS = ("11.8", "12.0", "12.1", "12.2", "12.3", "12.4", "12.5", "12.6", "12.7", "12.8",
                 "12.9", "13.0")
# Проверка видеокарты на поде ДО загрузки данных: драйвер виден, torch видит
# карту и реально на ней считает. Хост, где карта не работает, стоит денег
# и не даёт ничего — его надо заменить сразу, а не после загрузки 500 МБ.
# Интерпретатор на поде — `python`: в образах runpod/pytorch torch стоит
# именно для него (python3.12), а `python3` — системный Python без torch
# (живой прогон 29.09: две аренды ушли на «No module named torch»).
POD_PY = "python"
# Код выхода проверки «в образе у python нет torch» — ошибка ОБРАЗА, а не
# хоста: другой хост её не исправит, перебирать хосты за деньги нельзя.
NO_TORCH_EXIT = 3
TORCH_CHECK = (f"{POD_PY} -c \"import torch, torchvision; print('torch', torch.__version__, "
               f"'torchvision', torchvision.__version__)\" || {{ echo 'в образе у {POD_PY} нет "
               f"torch/torchvision'; exit {NO_TORCH_EXIT}; }}")
GPU_PREFLIGHT = ("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader && "
                 f"{TORCH_CHECK} && "
                 f"{POD_PY} -c \"import torch;assert torch.cuda.is_available(),'torch не видит CUDA';"
                 "x=torch.ones(1024,1024,device='cuda');torch.cuda.synchronize();"
                 "print('видеокарта работает:',torch.cuda.get_device_name(0),float((x@x).sum()))\"")
PREFLIGHT_SEC = 300
HOST_ATTEMPTS = 3


def image_cuda(image):
    """(major, minor) CUDA образа или None."""
    import re
    m = re.search(r"cuda(\d+)\.(\d+)|cu(\d{2})(\d)", image or "")
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2))) if m.group(1) else (int(m.group(3)), int(m.group(4)))


def allowed_cuda(image):
    """Версии CUDA хоста, на которых образ запустится: драйвер хоста обязан
    поддерживать CUDA не ниже образа. Не распознали — фильтра нет."""
    need = image_cuda(image)
    if need is None:
        return None
    return [v for v in CUDA_VERSIONS if tuple(int(x) for x in v.split(".")) >= need]


class NoStock(SystemExit):
    """Ни одной карты из списка сейчас нет: пода нет, денег не тратится."""


class BadHost(Exception):
    """Видеокарта пода не работает — под удалён, нужен другой хост."""
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


POD_PREFIX = "pipeline-job-"


def pod_name(deadline):
    """Имя пода несёт его крайний срок (unix-время): sweep_expired удаляет
    наши поды с истёкшим сроком, даже если исполнитель в них так и не
    стартовал и удалить себя сам не может."""
    return f"{POD_PREFIX}{int(deadline)}"


def sweep_expired(key, now=None):
    """Удалить поды этого скрипта, чей срок истёк. Чужие поды (другое имя) и
    живые запуски (срок впереди) не трогаются. Возвращает удалённые id."""
    now = time.time() if now is None else now
    gone = []
    for pod in rest("GET", "/pods", key) or []:
        name = pod.get("name") or ""
        if not name.startswith(POD_PREFIX):
            continue
        try:
            deadline = int(name[len(POD_PREFIX):])
        except ValueError:
            continue
        if deadline < now and pod.get("desiredStatus") != "TERMINATED":
            print(f"  под {pod['id']} ({name}) пережил свой срок — удаляю")
            terminate(key, pod["id"])
            gone.append(pod["id"])
    return gone


def create_pod(key, gpus, image, disk_gb, env, cloud, cpu=False, life_sec=None):
    """env — список переменных или функция gpu -> список (у каждой карты
    своя цена, а значит и свой потолок жизни пода в секундах). cpu — под без
    видеокарты (проверка пути за доли цента, см. --smoke)."""
    last = None
    for gpu in ([None] if cpu else gpus):
        e = env(gpu) if callable(env) else env
        life = life_sec(gpu) if callable(life_sec) else (life_sec or 4 * 3600)
        # Запас 10 мин на удаление: срок в имени — не раньше, чем исполнитель
        # сам удалит под по своему потолку.
        body = {"name": pod_name(time.time() + life + 600), "imageName": image, "containerDiskInGb": disk_gb,
                "volumeInGb": 0, "ports": [f"{PORT}/http"], "dockerStartCmd": START_ARGV,
                "env": {x["key"]: x["value"] for x in e}, "cloudType": cloud}
        if cpu:
            body.update(computeType="CPU", cpuFlavorIds=["cpu3c", "cpu5c", "cpu3g"], vcpuCount=2)
        else:
            body.update(computeType="GPU", gpuTypeIds=[gpu], gpuCount=1)
            cuda = allowed_cuda(image)
            if cuda:
                # Хост со старым драйвером не арендуется вовсе.
                body["allowedCudaVersions"] = cuda
            if cloud == "COMMUNITY":
                body["supportPublicIp"] = False
        try:
            pod = rest("POST", "/pods", key, body)
            if pod and pod.get("id"):
                pod.setdefault("gpuName", "CPU" if cpu else gpu)
                pod["gpuTypeId"] = gpu
                return pod
        except RuntimeError as e:
            last = e
            print(f"  {gpu or 'CPU'}: нет — {e}")
    raise NoStock(f"ни одной машины из списка нет в наличии ({last})")


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


def pack_dir(path, streams=6):
    """Папка -> (подпись, [архивы tar]). Файлы делятся на streams архивов
    поровну по объёму. БЕЗ сжатия: основной груз — JPEG и веса, они не
    сжимаются (замер 29.09: gzip на 540 МБ превью — 17.5 с ради 3%), а
    сборка идёт, пока под стартует (см. rent_and_drive), то есть вне
    оплачиваемого ожидания. Папка едет в /work по своему пути относительно
    репозитория (корень репозитория — в сам /work)."""
    full = os.path.abspath(path)
    rel = os.path.relpath(full, REPO)
    arc = "." if rel == "." else (rel if not rel.startswith("..") else os.path.basename(full))
    files = []
    for root, dirs, names in os.walk(full):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for n in names:
            fp = os.path.join(root, n)
            name = os.path.normpath(os.path.join(arc, os.path.relpath(fp, full)))
            if os.path.isfile(fp) and _exclude(tarfile.TarInfo(name)) is not None:
                files.append((os.path.getsize(fp), fp, name))
    n = max(1, min(streams, len(files)))
    groups, sizes = [[] for _ in range(n)], [0] * n
    for size, fp, name in sorted(files, reverse=True):     # поровну по объёму
        k = sizes.index(min(sizes))
        groups[k].append((fp, name))
        sizes[k] += size
    blobs = []
    for g in groups:
        if not g:
            continue
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            for fp, name in g:
                tar.add(fp, arcname=name, recursive=False)
        blobs.append(buf.getvalue())
    return path, blobs


def upload_progress(shown):
    """(загружено, всего) байт: shown — {архив: байт на поде, "_total": всего}."""
    return sum(v for k, v in shown.items() if k != "_total"), shown["_total"]


class Runner:
    def __init__(self, base, token):
        self.base, self.token = base.rstrip("/"), token
        self.log_off = 0      # лог задач на поде общий: следующая задача читается с конца прошлой

    # Прокси Runpod между нами и подом иногда отвечает случайной ошибкой при
    # живом исполнителе (живой прогон 29.09: пустой 404 посреди лога задачи
    # уронил запуск). Такие ответы пережидаются повтором; все запросы
    # исполнителя повторяемы безопасно (см. runpod_runner: загрузка по
    # смещению, распаковка и запуск — по идентификатору, дважды не делаются).
    RETRY_SEC = 240
    TRANSIENT_HTTP = {404, 408, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 525, 530}

    def call(self, method, path, body=None, timeout=90, raw=False, retry=True):
        t0, attempt = time.time(), 0
        while True:
            req = urllib.request.Request(self.base + path, data=body, method=method,
                                         headers={"X-Runner-Token": self.token, "User-Agent": UA})
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    data = r.read()
                return data if raw else json.loads(data)
            except urllib.error.HTTPError as e:
                payload = e.read()
                # Ответ самого исполнителя — JSON с причиной: это не сбой
                # связи, повторять нечего. Пустое тело — прокси.
                own = payload.strip().startswith(b"{")
                if own or e.code not in self.TRANSIENT_HTTP or not retry:
                    e.fp = io.BytesIO(payload)
                    e.read = e.fp.read
                    raise
                err = f"HTTP {e.code} от прокси"
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                if not retry:
                    raise
                err = f"{type(e).__name__}: {e}"
            attempt += 1
            if time.time() - t0 > self.RETRY_SEC:
                raise RuntimeError(f"исполнитель недоступен {self.RETRY_SEC} с ({err})")
            if attempt == 1 or attempt % 5 == 0:
                print(f"\n  связь с подом: {err} — повтор", flush=True)
            time.sleep(min(15, 2 * attempt))

    def wait_ready(self, limit_sec, status_fn=None, run_grace_sec=300):
        """Ждать исполнителя. status_fn() -> (секунд работы контейнера или
        None, статус) — чтобы отличать «образ ещё тянется» (ждать) от
        «контейнер работает, а исполнитель не отвечает» (сразу стоп: деньги
        идут, а ждать нечего). Каждая смена состояния печатается."""
        t0, last_err, shown, next_status = time.time(), None, None, 0.0
        while time.time() - t0 < limit_sec:
            try:
                if self.call("GET", "/health", timeout=15, retry=False).get("ok"):
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

    UPLOAD_STREAMS = 6

    def _upload_blob(self, data, label, shown):
        fname = f"up_{secrets.token_hex(4)}.tar"
        off = 0
        while off < len(data):
            piece = data[off:off + CHUNK]
            try:
                off = self.call("PUT", f"/upload?name={fname}&offset={off}", piece, timeout=300)["size"]
            except urllib.error.HTTPError as e:
                if e.code != 409:
                    raise
                off = json.loads(e.read())["size"]     # докачка с того места, где под остановился
            shown[fname] = off
            done, total = upload_progress(shown)
            print(f"  загрузка {label}: {done / 2**20:.0f} / {total / 2**20:.0f} МБ", end="\r", flush=True)
        self.call("POST", f"/extract?name={fname}", b"")        # повтор безопасен: по имени

    def upload_dir(self, path, streams=None):
        self.upload_packed(pack_dir(path, streams or self.UPLOAD_STREAMS))

    def upload_packed(self, packed):
        """Уже собранные архивы папки (pack_dir) едут ОДНОВРЕМЕННО: прокси
        Runpod режет скорость одного соединения, а под оплачивается
        посекундно."""
        from concurrent.futures import ThreadPoolExecutor
        label, blobs = packed
        if not blobs:
            return
        shown = {"_total": sum(len(b) for b in blobs)}
        t0 = time.time()
        with ThreadPoolExecutor(len(blobs)) as ex:
            for f in [ex.submit(self._upload_blob, b, label, shown) for b in blobs]:
                f.result()
        dt = time.time() - t0
        print(f"\n  загружено {shown['_total'] / 2**20:.0f} МБ за {dt:.0f} с "
              f"({shown['_total'] / 2**20 / max(dt, 0.1):.1f} МБ/с, потоков {len(blobs)})")

    def start(self, cmd):
        # id задачи: повтор запроса после сбоя связи не запускает её дважды.
        # set -e: упавший шаг останавливает цепочку и даёт ненулевой код, даже
        # если шаги склеены «;» (живой прогон 29.09: калибровка отказала, а
        # «;» довёл команду до кода 0).
        self.call("POST", "/run", json.dumps({"cmd": f"set -eo pipefail; {cmd}",
                                              "id": secrets.token_hex(8)}).encode())

    def run(self, cmd, deadline=None):
        """deadline — момент (time.time()), когда потолок денег исчерпан:
        задача прерывается, под удаляется вызывающим (finally)."""
        self.start(cmd)
        return self.follow(deadline)

    def follow(self, deadline=None):
        off, last_up = self.log_off, None
        while True:
            if deadline is not None and time.time() > deadline:
                raise SystemExit("потолок денег на запуск исчерпан — задача прервана, под удаляется")
            st = self.call("GET", f"/log?offset={off}")
            # Время жизни исполнителя упало — контейнер пода перезапустился
            # (например, нехватка памяти): новый исполнитель про задачу не
            # знает, и ждать её конца бессмысленно до самого потолка денег.
            up = st.get("uptime")
            if last_up is not None and up is not None and up < last_up:
                raise SystemExit("контейнер пода перезапустился посреди задачи — задача потеряна")
            last_up = up if up is not None else last_up
            if st["text"]:
                sys.stdout.write(st["text"])
                sys.stdout.flush()
            off = self.log_off = st["offset"]
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
    p.add_argument("--cleanup", action="store_true",
                   help="только удалить поды этого скрипта с истёкшим сроком")
    p.add_argument("--selftest-autodelete", action="store_true",
                   help="живая проверка: под без видеокарты удаляет себя сам по простою")
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
    p.add_argument("--prepare", help="подготовка на поде (библиотеки, веса) — идёт сразу, "
                                     "параллельно с загрузкой данных; --cmd ждёт её и данные")
    p.add_argument("--fetch", action="append", default=[], help="путь в /work, вернуть сюда")
    p.add_argument("--dest", default=".")
    p.add_argument("--env-from-dotenv", default="", help="KEY1,KEY2 — передать в под из .env")
    p.add_argument("--idle-min", type=float, default=10)
    p.add_argument("--max-hours", type=float, default=4)
    p.add_argument("--wait-stock-min", type=float, default=15,
                   help="нет свободных карт — ждать столько минут, перепроверяя (пода нет — "
                        "денег не тратится); 0 — не ждать")
    p.add_argument("--max-usd", type=float, default=2.0,
                   help="потолок денег на запуск: под удаляется, когда его цена дошла до лимита")
    a = p.parse_args(argv)
    key = api_key()
    # Каждый запуск сначала убирает наши поды с истёкшим сроком — страховка
    # на случай, когда и эта сторона пропала, и исполнитель не стартовал.
    sweep_expired(key)
    if a.cleanup:
        return 0
    if a.smoke:
        return smoke(key, a)
    if a.selftest_autodelete:
        return selftest_autodelete(key, a.image or SMOKE_IMAGE)
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
        if not a.wait_stock_min:
            raise NoStock(1)
        # иначе — ожидание наличия (create_when_in_stock) после проверки пути
    for gid in gpus:
        g = by_id.get(gid, {"displayName": gid, "memoryInGb": "?"})
        lp = g.get("lowestPrice") or {}
        print(f"  {g['displayName']:<16} {g['memoryInGb']} ГБ  сейчас "
              f"{lp.get('uninterruptablePrice')} $/ч  наличие {lp.get('stockStatus')}")
    if a.plan:
        return 0
    if not a.cmd:
        raise SystemExit("нет --cmd")
    if not a.no_smoke and smoke_passed_recently(a.image):
        print(f"Проверка пути: пройдена для этого образа и кода исполнителя "
              f"за последние {SMOKE_VALID_SEC // 3600} ч — не повторяется")
    elif not a.no_smoke:
        # Перед арендой видеокарты — тот же образ на поде без неё (доли
        # цента): старт контейнера, исполнитель, прокси, удаление. Путь не
        # работает — карта не арендуется вообще.
        print("Проверка пути до аренды видеокарты:")
        if smoke(key, a) != 0:
            raise SystemExit("проверка пути не пройдена — видеокарта не арендована, деньги не потрачены")
        record_smoke(a.image)
    token = secrets.token_urlsafe(32)
    extra = dotenv_subset([n.strip() for n in a.env_from_dotenv.split(",") if n.strip()])

    return rent_and_drive(key, a, gpus, by_id, token, extra)


def rent_and_drive(key, a, gpus, by_id, token, extra, attempts=HOST_ATTEMPTS):
    """Аренда с проверкой видеокарты: хост, где карта не работает, удаляется
    и заменяется (до attempts раз). Потолок --max-usd — на ВСЕ попытки
    вместе, а не на каждую."""
    from concurrent.futures import ThreadPoolExecutor
    order, spent = list(gpus), 0.0
    # Архивы собираются ЗДЕСЬ, до создания пода: пока под стартует (18-90 с)
    # и проверяет карту, сборка уже идёт. Раньше она шла после старта и
    # стоила оплачиваемых ~110 с (живой прогон 29.09). Собранное переживает
    # смену хоста — повтор не собирает заново.
    packer = ThreadPoolExecutor(max(1, len(a.upload)))
    uploads = [packer.submit(pack_dir, path, Runner.UPLOAD_STREAMS) for path in a.upload]
    try:
        return _rent_attempts(key, a, order, by_id, token, extra, attempts, uploads, spent)
    finally:
        packer.shutdown(wait=False, cancel_futures=True)


STOCK_POLL_SEC = 20


def create_when_in_stock(key, a, order, create, by_id):
    """Создать под; нет карт — ждать и перепроверять наличие (пода нет —
    денег не тратится) до --wait-stock-min минут. Живой прогон 29.09:
    все карты от 24 ГБ в community разом «нет в наличии», и запуск
    просто сдавался. Без --gpu список карт перечитывается при каждой
    проверке: освободиться может и та, которой в нём не было; типы, на
    которых уже попалась неисправная карта, остаются в конце очереди."""
    deadline = time.time() + max(0.0, getattr(a, "wait_stock_min", 0) or 0) * 60
    while True:
        try:
            return create(order)
        except NoStock as e:
            if time.time() >= deadline:
                raise
            print(f"  {e} — жду {STOCK_POLL_SEC} с и проверяю снова "
                  f"(ещё {max(0, deadline - time.time()) / 60:.0f} мин)", flush=True)
            time.sleep(STOCK_POLL_SEC)
            if not a.gpu:
                info = plan(key, None, community=a.cloud == "COMMUNITY")
                by_id.update({g["id"]: g for g in info["gpuTypes"]})
                fresh = cheapest_gpus(info["gpuTypes"], a.min_gb, a.image)[:6]
                tail = [g for g in order if g not in fresh]
                order[:] = [g for g in fresh if g not in tail] + tail


def _rent_attempts(key, a, order, by_id, token, extra, attempts, uploads, spent):
    for attempt in range(1, attempts + 1):
        left = a.max_usd - spent
        if left <= 0.05:
            raise SystemExit(f"потолок ${a.max_usd:.2f} исчерпан попытками (${spent:.2f})")

        def price(gid):
            lp = (by_id.get(gid) or {}).get("lowestPrice") or {}
            return lp.get("uninterruptablePrice")

        def env_for(gid, left=left):
            # Потолок жизни пода на его стороне — по цене ЭТОЙ карты и
            # остатку денег: если связь пропадёт, под не проживёт дольше.
            return runner_env(token, a.idle_min, budget_seconds(price(gid), left, a.max_hours) / 3600,
                              extra)
        pod = create_when_in_stock(
            key, a, order, lambda gids, left=left: create_pod(
                key, gids, a.image, a.disk_gb, env_for, a.cloud,
                life_sec=lambda gid: budget_seconds(price(gid), left, a.max_hours)), by_id)
        cap_sec = budget_seconds(pod["costPerHr"], left, a.max_hours)
        print(f"Под {pod['id']} (попытка {attempt}/{attempts}): {pod['gpuName']}, ${pod['costPerHr']}/ч; "
              f"потолок ${left:.2f} = {cap_sec / 60:.0f} мин, самоудаление через {a.idle_min:.0f} мин простоя")
        try:
            return drive(key, pod, token, cap_sec, uploads, a.cmd, a.fetch, a.dest,
                         prepare=a.prepare, preflight=GPU_PREFLIGHT)
        except BadHost as e:
            spent += pod.get("spent_usd", 0.0)
            print(f"  {e} — хост заменяется (потрачено ${spent:.2f})")
            # Та же карта, скорее всего, достанется с того же хоста: сначала
            # остальные типы, эта — в конец очереди.
            gpu = pod.get("gpuTypeId")
            if gpu in order and len(order) > 1:
                order.remove(gpu)
                order.append(gpu)
    raise SystemExit(f"видеокарта не заработала за {attempts} попытки — задача не запускалась "
                     f"(потрачено ${spent:.2f})")


UPLOADS_DONE = ".uploads_done"


def overlapped_cmd(prepare, cmd):
    """Подготовка (установка библиотек, скачивание весов) идёт на поде
    СРАЗУ, параллельно с загрузкой данных отсюда; команда ждёт отметку «всё
    загружено». Посекундная оплата: минуты скачивания весов больше не
    складываются с минутами загрузки."""
    return (f"( {prepare} ) & PREP=$!; while [ ! -f {UPLOADS_DONE} ]; do sleep 2; done; "
            f"wait $PREP || {{ echo 'подготовка упала'; exit 97; }}; {cmd}")


def _send(r, item):
    """Элемент загрузки: путь к папке или уже запущенная сборка её архивов
    (Future от pack_dir — собиралась, пока под стартовал)."""
    if hasattr(item, "result"):
        r.upload_packed(item.result())
    else:
        r.upload_dir(item)


def drive(key, pod, token, cap_sec, uploads, cmd, fetches, dest, prepare=None, preflight=None):
    """Под создан: дождаться исполнителя, загрузить, запустить, забрать,
    и удалить под в ЛЮБОМ исходе. prepare — идёт на поде параллельно с
    загрузкой (см. overlapped_cmd); первая папка (код) едет до старта.
    preflight — проверка видеокарты до загрузки: не прошла — BadHost (под
    удалён, вызывающий берёт другой хост). Потраченное — pod["spent_usd"]."""
    pod_id, t0, code = pod["id"], time.time(), 1

    def stage(name):
        # Время каждого этапа от создания пода: куда уходят оплачиваемые секунды.
        print(f"[+{time.time() - t0:5.0f} с] {name}", flush=True)

    def _sigint(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _sigint)
    try:
        r = Runner(f"https://{pod_id}-{PORT}.proxy.runpod.net", token)
        if not r.wait_ready(min(1800, cap_sec), lambda: pod_status(key, pod_id)):
            raise SystemExit("исполнитель на поде не поднялся")
        stage("исполнитель готов")
        if preflight:
            try:
                pc = r.run(preflight, deadline=min(t0 + cap_sec, time.time() + PREFLIGHT_SEC))
            except SystemExit as e:
                pc = f"не уложилась ({e})"
            if pc == NO_TORCH_EXIT:
                raise SystemExit(f"в образе нет torch для `{POD_PY}` — ошибка образа, другой хост "
                                 f"её не исправит; под удалён")
            if pc != 0:
                raise BadHost(f"видеокарта пода не работает (проверка: {pc})")
            stage("видеокарта проверена")
        if prepare:
            first, rest_up = uploads[:1], uploads[1:]
            for item in first:
                _send(r, item)
            r.start(overlapped_cmd(prepare, cmd))
            for item in rest_up:
                _send(r, item)
            r.call("POST", f"/touch?name={UPLOADS_DONE}", b"")
            stage("данные загружены")
            code = r.follow(deadline=t0 + cap_sec)
        else:
            for item in uploads:
                _send(r, item)
            stage("данные загружены")
            code = r.run(cmd, deadline=t0 + cap_sec)
        print()
        stage(f"команда завершилась с кодом {code}")
        for path in fetches:
            # Несозданный результат — не повод бросить остальные: забираем
            # всё, что есть, а пропуск называем.
            try:
                r.fetch(path, dest)
                print(f"  забрано: {path}")
            except Exception as e:  # noqa: BLE001
                print(f"  НЕ забрано: {path} ({getattr(e, 'code', '') or e})")
                if code == 0:
                    code = 98
    finally:
        terminate(key, pod_id)
        sec = time.time() - t0
        pod["spent_usd"] = sec / 3600 * float(pod['costPerHr'])
        print(f"Под жил {sec / 60:.1f} мин ≈ ${pod['spent_usd']:.2f}")
    return code


def selftest_autodelete(key, image=SMOKE_IMAGE, idle_sec=60, limit_sec=360):
    """Живая проверка страховки: под без видеокарты, к которому никто не
    обращается, обязан удалить себя сам по простою. Не удалился за
    limit_sec — удаляется отсюда, проверка не пройдена."""
    token = secrets.token_urlsafe(32)
    env = runner_env(token, idle_sec / 60, 0.25, {})
    pod = create_pod(key, [], image, 20, env, "SECURE", cpu=True, life_sec=limit_sec)
    pid, t0 = pod["id"], time.time()
    print(f"Под {pid}: CPU, ${pod['costPerHr']}/ч — жду самоудаления после {idle_sec} с простоя")
    try:
        while time.time() - t0 < limit_sec:
            left = rest("GET", f"/pods/{pid}", key)
            if not left or left.get("desiredStatus") == "TERMINATED":
                print(f"  под удалил себя сам через {time.time() - t0:.0f} с после создания")
                return 0
            time.sleep(10)
        print(f"  под НЕ удалил себя за {limit_sec} с")
        return 1
    finally:
        left = rest("GET", f"/pods/{pid}", key)
        if left and left.get("desiredStatus") != "TERMINATED":
            terminate(key, pid)


# Проверка пути тянет тот же образ (11 ГБ) на CPU-под — минуты ожидания на
# каждый запуск. Результат зависит только от образа, команды старта и кода
# исполнителя, поэтому запоминается на сутки по их отпечатку: сменился
# любой — проверка идёт заново.
SMOKE_VALID_SEC = 24 * 3600
SMOKE_MARK = os.path.join(os.path.expanduser("~"), ".cache", "pipeline_runpod", "smoke_ok.json")


def smoke_fingerprint(image):
    import hashlib
    src = open(os.path.join(HERE, "runpod_runner.py"), "rb").read()
    return hashlib.sha256(json.dumps([image, START_ARGV, PORT, smoke_check(image)]).encode()
                          + src).hexdigest()


def smoke_check(image):
    """Проверка содержимого образа на CPU-поде: у образа видеокарты — torch
    для того интерпретатора, которым пойдут задачи. Ошибка образа ловится
    здесь за доли цента, а не на аренде видеокарты."""
    return "" if image == SMOKE_IMAGE else TORCH_CHECK


def smoke_passed_recently(image, now=None):
    try:
        marks = json.load(open(SMOKE_MARK, encoding="utf-8"))
    except (OSError, ValueError):
        return False
    t = marks.get(smoke_fingerprint(image))
    return t is not None and 0 <= (now or time.time()) - t < SMOKE_VALID_SEC


def record_smoke(image):
    try:
        marks = json.load(open(SMOKE_MARK, encoding="utf-8"))
    except (OSError, ValueError):
        marks = {}
    marks[smoke_fingerprint(image)] = time.time()
    os.makedirs(os.path.dirname(SMOKE_MARK), exist_ok=True)
    tmp = SMOKE_MARK + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(marks, f)
    os.replace(tmp, SMOKE_MARK)


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
    pod = create_pod(key, [], image, 20, env, "SECURE", cpu=True, life_sec=1800)
    print(f"Под {pod['id']}: CPU, ${pod['costPerHr']}/ч")
    src = tempfile.mkdtemp(prefix="smoke_up_")
    with open(os.path.join(src, "probe.txt"), "w", encoding="utf-8") as f:
        f.write("ping")
    out = tempfile.mkdtemp(prefix="smoke_out_")
    cmd = (f"python3 -c \"import os;d=open('{os.path.basename(src)}/probe.txt').read();"
           f"os.makedirs('res',exist_ok=True);open('res/pong.txt','w').write(d+'-pong');"
           f"print('исполнитель жив', d)\"")
    if smoke_check(image):
        cmd = f"{smoke_check(image)} && {cmd}"
    code = drive(key, pod, token, 1800, [src], cmd, ["res"], out)
    got = open(os.path.join(out, "res", "pong.txt"), encoding="utf-8").read() \
        if os.path.exists(os.path.join(out, "res", "pong.txt")) else None
    ok = code == 0 and got == "ping-pong"
    print("ПРОВЕРКА ПУТИ:", "пройдена" if ok else f"НЕ пройдена (код {code}, ответ {got!r})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
