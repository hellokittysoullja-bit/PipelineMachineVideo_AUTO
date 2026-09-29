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
    src = open(os.path.join(HERE, "runpod_runner.py"), "rb").read()
    env = {"RUNNER_TOKEN": token, "RUNNER_B64": base64.b64encode(src).decode(),
           "RUNNER_IDLE_SEC": str(int(idle_min * 60)), "RUNNER_MAX_SEC": str(int(max_hours * 3600)),
           "RUNNER_PORT": str(PORT)}
    env.update(extra)
    return [{"key": k, "value": v} for k, v in env.items()]


START_CMD = ("bash -c 'echo \"$RUNNER_B64\" | base64 -d > /runner.py && "
             "exec python3 /runner.py'")


def create_pod(key, gpus, image, disk_gb, env, cloud):
    mut = ('mutation($in: PodFindAndDeployOnDemandInput){ podFindAndDeployOnDemand(input:$in)'
           '{ id costPerHr machine { gpuDisplayName } } }')
    last = None
    for gpu in gpus:
        inp = {"cloudType": cloud, "gpuCount": 1, "gpuTypeId": gpu, "name": "pipeline-job",
               "imageName": image, "containerDiskInGb": disk_gb, "volumeInGb": 0,
               "ports": f"{PORT}/http", "dockerArgs": START_CMD, "env": env,
               "supportPublicIp": False}
        try:
            pod = gql(mut, key, {"in": inp})["podFindAndDeployOnDemand"]
            if pod and pod.get("id"):
                return pod
        except RuntimeError as e:
            last = e
            print(f"  {gpu}: нет — {e}")
    raise SystemExit(f"ни одной карты из списка нет в наличии ({last})")


def terminate(key, pod_id):
    """Удалить под и убедиться, что его больше нет. Повторяет — удаление
    обязано случиться, иначе карта стоит за деньги."""
    for attempt in range(6):
        try:
            gql('mutation($id:String!){ podTerminate(input:{podId:$id}) }', key, {"id": pod_id})
        except Exception as e:  # noqa: BLE001
            print(f"  удаление пода: {e}")
        try:
            left = gql('query($id:String!){ pod(input:{podId:$id}){ id desiredStatus } }', key,
                       {"id": pod_id}).get("pod")
        except Exception:  # noqa: BLE001
            left = {"id": pod_id}
        if not left:
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

    def wait_ready(self, limit_sec):
        t0 = time.time()
        while time.time() - t0 < limit_sec:
            try:
                if self.call("GET", "/health", timeout=15).get("ok"):
                    self.call("GET", "/log?offset=0")   # клиент на месте — таймер простоя с нуля
                    return True
            except Exception:  # noqa: BLE001 — под ещё поднимается
                pass
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

    def run(self, cmd):
        self.call("POST", "/run", json.dumps({"cmd": cmd}).encode())
        off = 0
        while True:
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
    p.add_argument("--gpu", action="append",
                   help="тип карты (можно несколько, по порядку); без него — самые дешёвые "
                        "свободные карты community от --min-gb")
    p.add_argument("--min-gb", type=int, default=MIN_GPU_GB)
    p.add_argument("--cloud", default="COMMUNITY", choices=("COMMUNITY", "SECURE", "ALL"))
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--disk-gb", type=int, default=80)
    p.add_argument("--upload", action="append", default=[], help="папка, едет в /work")
    p.add_argument("--cmd", help="команда в /work на поде")
    p.add_argument("--fetch", action="append", default=[], help="путь в /work, вернуть сюда")
    p.add_argument("--dest", default=".")
    p.add_argument("--env-from-dotenv", default="", help="KEY1,KEY2 — передать в под из .env")
    p.add_argument("--idle-min", type=float, default=10)
    p.add_argument("--max-hours", type=float, default=4)
    a = p.parse_args(argv)
    key = api_key()
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
    token = secrets.token_urlsafe(32)
    env = runner_env(token, a.idle_min, a.max_hours,
                     dotenv_subset([n.strip() for n in a.env_from_dotenv.split(",") if n.strip()]))
    pod = create_pod(key, gpus, a.image, a.disk_gb, env, a.cloud)
    pod_id, t0 = pod["id"], time.time()
    print(f"Под {pod_id}: {pod['machine']['gpuDisplayName']}, ${pod['costPerHr']}/ч, "
          f"самоудаление через {a.idle_min:.0f} мин простоя и {a.max_hours:.0f} ч в любом случае")
    code = 1

    def _sigint(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _sigint)
    try:
        r = Runner(f"https://{pod_id}-{PORT}.proxy.runpod.net", token)
        if not r.wait_ready(900):
            raise SystemExit("исполнитель на поде не поднялся за 15 минут")
        for path in a.upload:
            r.upload_dir(path)
        code = r.run(a.cmd)
        print(f"\nКоманда завершилась с кодом {code}")
        for path in a.fetch:
            r.fetch(path, a.dest)
            print(f"  забрано: {path}")
    finally:
        terminate(key, pod_id)
        sec = time.time() - t0
        print(f"Под жил {sec / 60:.1f} мин ≈ ${sec / 3600 * float(pod['costPerHr']):.2f}")
    return code


if __name__ == "__main__":
    sys.exit(main())
