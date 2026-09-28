#!/usr/bin/env python3
"""Управление RunPod: REST API (rest.runpod.io/v1) + агент внутри пода. Только stdlib.

Ключ — RUNPOD_API_KEY из окружения или .env. Платные/необратимые действия требуют --yes.

  status                          поды, эндпоинты, тома, предупреждение о работающих
  up --gpu "NVIDIA RTX A5000" --yes [--hours 6 --idle 20 --max-price 1.0 --spot
                                     --disk 50 --volume 50 --image ... --cloud SECURE]
                                  поднять под с агентом, самоостановкой и потолком цены
  exec <podId> "<cmd>" [--cwd D]  запустить команду на поде, дождаться и напечатать вывод
  put <podId> <local> <remote>    залить файл;   get <podId> <remote> <local>  забрать
  agent <podId>                   состояние агента: GPU, простой, остаток жизни
  down <podId> [--delete] --yes   остановить (диск /workspace сохраняется) или удалить
  guard [--include-unknown] --yes остановить поды, у которых вышел срок (для cron/рутины)
  stopall --yes                   остановить все RUNNING
  start|restart <podId>           запуск/перезапуск
  billing [hour|day|week|month]   расходы
"""
import argparse
import base64
import json
import os
import secrets
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://rest.runpod.io/v1"
ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / ".runpod_state.json"
AGENT = ROOT / "scripts" / "pod_agent.py"
AGENT_PORT = 8000
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


def load_key():
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if key:
        return key
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("RUNPOD_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("RUNPOD_API_KEY не задан (окружение или .env)")


def http(method, url, headers, body=None, raw=False, timeout=60):
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code} {url}: {e.read().decode(errors='replace')[:400]}")
    if raw:
        return data
    return json.loads(data) if data else {}


def call(method, path, body=None):
    return http(method, BASE + path,
                {"Authorization": f"Bearer {load_key()}", "Content-Type": "application/json",
                 "User-Agent": "pipeline-runpod-ctl/1.1"},
                json.dumps(body).encode() if body is not None else None)


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(s):
    STATE.write_text(json.dumps(s, indent=1))
    STATE.chmod(0o600)


def agent_call(pod_id, method, path, body=None, raw=False, timeout=120):
    st = load_state().get(pod_id)
    if not st:
        sys.exit(f"под {pod_id} создан не через `up` — токена агента нет")
    url = f"https://{pod_id}-{AGENT_PORT}.proxy.runpod.net{path}"
    data = body if isinstance(body, bytes) else (json.dumps(body).encode() if body is not None else None)
    return http(method, url, {"Authorization": f"Bearer {st['token']}"}, data, raw, timeout)


def need_yes(a, what):
    if not a.yes:
        sys.exit(f"{what} — платное/необратимое действие, добавь --yes")


def pod_line(p):
    gpu = (p.get("gpu") or {}).get("displayName") or p.get("gpuTypeId") or "?"
    return (f"{p.get('id')}  {p.get('desiredStatus'):<8} {gpu} x{p.get('gpuCount', '?')}  "
            f"${p.get('costPerHr', '?')}/ч  {p.get('name')}")


def cmd_up(a):
    need_yes(a, "up")
    if not a.gpu:
        sys.exit("нужен --gpu (id из RunPod, напр. 'NVIDIA RTX A5000')")
    token = secrets.token_urlsafe(32)
    boot = f'echo "$POD_AGENT_B64" | base64 -d > /agent.py && exec python3 /agent.py'
    body = {
        "name": a.name, "imageName": a.image, "gpuTypeIds": [a.gpu], "gpuCount": 1,
        "cloudType": a.cloud, "interruptible": a.spot,
        "containerDiskInGb": a.disk, "volumeInGb": a.volume, "volumeMountPath": "/workspace",
        "ports": [f"{AGENT_PORT}/http"],
        "env": {"POD_AGENT_TOKEN": token, "POD_IDLE_MIN": str(a.idle),
                "POD_MAX_HOURS": str(a.hours),
                "POD_AGENT_B64": base64.b64encode(AGENT.read_bytes()).decode()},
        "dockerStartCmd": ["bash", "-c", boot],
    }
    p = call("POST", "/pods", body)
    pid = p["id"]
    price = float(p.get("costPerHr") or 0)
    if a.max_price and price > a.max_price:
        call("DELETE", f"/pods/{pid}")
        sys.exit(f"цена ${price}/ч выше потолка ${a.max_price}/ч — под удалён")
    st = load_state()
    st[pid] = {"token": token, "created": time.time(), "deadline": time.time() + a.hours * 3600,
               "gpu": a.gpu, "cost_per_hr": price}
    save_state(st)
    print(pod_line(p))
    print(f"жду агента (до {a.wait} мин)...")
    t0 = time.time()
    while time.time() - t0 < a.wait * 60:
        try:
            s = agent_call(pid, "GET", "/status", timeout=15)
            print("агент готов:", json.dumps(s))
            return
        except SystemExit:
            time.sleep(15)
    print(f"агент не ответил за {a.wait} мин. Под ЗАПУЩЕН и списывает деньги: "
          f"проверь `status`, при сомнении `down {pid} --delete --yes`")


def cmd_exec(a):
    r = agent_call(a.arg, "POST", "/exec", {"cmd": a.rest, "cwd": a.cwd})
    jid, shown = r["job"], 0
    while True:
        j = agent_call(a.arg, "GET", f"/job/{jid}?tail=1000000")
        if len(j["log"]) > shown:
            sys.stdout.write(j["log"][shown:])
            sys.stdout.flush()
            shown = len(j["log"])
        if not j["running"]:
            sys.exit(j["code"] or 0)
        time.sleep(3)


def cmd_guard(a):
    need_yes(a, "guard")
    st = load_state()
    for p in call("GET", "/pods"):
        if p.get("desiredStatus") != "RUNNING":
            continue
        s = st.get(p["id"])
        if s and time.time() > s["deadline"]:
            call("POST", f"/pods/{p['id']}/stop")
            print("срок вышел, остановлен", p["id"])
        elif not s and a.include_unknown:
            call("POST", f"/pods/{p['id']}/stop")
            print("неизвестный под остановлен", p["id"])
        elif not s:
            print("неизвестный под RUNNING (не трогаю без --include-unknown):", p["id"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("cmd", choices=["status", "up", "exec", "put", "get", "agent", "down",
                                    "guard", "stopall", "start", "restart", "billing"])
    ap.add_argument("arg", nargs="?")
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--gpu")
    ap.add_argument("--name", default="pipeline-render")
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--cloud", default="SECURE", choices=["SECURE", "COMMUNITY"])
    ap.add_argument("--spot", action="store_true", help="прерываемый под, дешевле, но может быть отобран")
    ap.add_argument("--disk", type=int, default=50)
    ap.add_argument("--volume", type=int, default=50)
    ap.add_argument("--hours", type=float, default=6, help="жёсткий предел жизни пода")
    ap.add_argument("--idle", type=float, default=20, help="минут простоя до самоостановки")
    ap.add_argument("--max-price", type=float, default=1.0, help="потолок $/ч, 0 = без потолка")
    ap.add_argument("--wait", type=float, default=10)
    ap.add_argument("--cwd")
    ap.add_argument("--delete", action="store_true")
    ap.add_argument("--include-unknown", action="store_true")
    a = ap.parse_args()
    if a.cmd == "exec":
        a.rest = " ".join(a.rest)

    if a.cmd == "status":
        pods, eps, vols = call("GET", "/pods"), call("GET", "/endpoints"), call("GET", "/networkvolumes")
        print(f"Поды ({len(pods)}):")
        for p in pods:
            print("  " + pod_line(p))
        print(f"Эндпоинты ({len(eps)}):")
        for e in eps:
            print(f"  {e.get('id')}  {e.get('name')}  workers {e.get('workersMin')}-{e.get('workersMax')}")
        print(f"Тома ({len(vols)}):")
        for v in vols:
            print(f"  {v.get('id')}  {v.get('name')}  {v.get('size')} ГБ  {v.get('dataCenterId')}")
        run = [p for p in pods if p.get("desiredStatus") == "RUNNING"]
        if run:
            print(f"\nВНИМАНИЕ: {len(run)} под(ов) RUNNING — идёт списание")
    elif a.cmd == "up":
        cmd_up(a)
    elif a.cmd == "exec":
        cmd_exec(a)
    elif a.cmd == "put":
        Path(a.arg)  # podId
        agent_call(a.arg, "PUT", f"/file?path={a.rest[1]}", Path(a.rest[0]).read_bytes(), timeout=600)
        print("залито", a.rest[1])
    elif a.cmd == "get":
        Path(a.rest[1]).write_bytes(agent_call(a.arg, "GET", f"/file?path={a.rest[0]}", raw=True, timeout=600))
        print("забрано", a.rest[1])
    elif a.cmd == "agent":
        print(json.dumps(agent_call(a.arg, "GET", "/status"), indent=1))
    elif a.cmd == "down":
        need_yes(a, "down")
        if a.delete:
            call("DELETE", f"/pods/{a.arg}")
            st = load_state()
            st.pop(a.arg, None)
            save_state(st)
            print("удалён", a.arg)
        else:
            call("POST", f"/pods/{a.arg}/stop")
            print(pod_line(call("GET", f"/pods/{a.arg}")))
    elif a.cmd == "guard":
        cmd_guard(a)
    elif a.cmd == "stopall":
        need_yes(a, "stopall")
        for p in call("GET", "/pods"):
            if p.get("desiredStatus") == "RUNNING":
                call("POST", f"/pods/{p['id']}/stop")
                print("остановлен", p["id"], p.get("name"))
    elif a.cmd in ("start", "restart"):
        call("POST", f"/pods/{a.arg}/{a.cmd}")
        print(pod_line(call("GET", f"/pods/{a.arg}")))
    elif a.cmd == "billing":
        b = a.arg or "day"
        print(json.dumps({"pods": call("GET", f"/billing/pods?bucketSize={b}"),
                          "endpoints": call("GET", f"/billing/endpoints?bucketSize={b}")},
                         indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
