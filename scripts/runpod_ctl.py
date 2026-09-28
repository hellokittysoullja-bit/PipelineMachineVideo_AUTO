#!/usr/bin/env python3
"""Управление RunPod через официальный REST API (rest.runpod.io/v1). Только stdlib.

Ключ — RUNPOD_API_KEY из окружения или .env. Платные и необратимые действия
(create/delete/stopall) требуют --yes.

  runpod_ctl.py status                       # поды, эндпоинты, тома
  runpod_ctl.py create --json '{...}' --yes  # тело — PodCreateInput
  runpod_ctl.py start|stop|restart <podId>
  runpod_ctl.py delete <podId> --yes
  runpod_ctl.py stopall --yes                # остановить все RUNNING поды
  runpod_ctl.py billing [hour|day|week|month]
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://rest.runpod.io/v1"


def load_key():
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if key:
        return key
    env = Path(__file__).resolve().parent.parent / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("RUNPOD_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("RUNPOD_API_KEY не задан (окружение или .env)")


def call(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Authorization": f"Bearer {load_key()}",
                 "Content-Type": "application/json",
                 "User-Agent": "pipeline-runpod-ctl/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code} {path}: {e.read().decode()[:500]}")
    return json.loads(raw) if raw else {}


def need_yes(args, what):
    if not args.yes:
        sys.exit(f"{what} — платное/необратимое действие, добавь --yes")


def pod_line(p):
    gpu = (p.get("gpu") or {}).get("displayName") or p.get("gpuTypeId") or "?"
    return (f"{p.get('id')}  {p.get('desiredStatus'):<8} {gpu} x{p.get('gpuCount', '?')}  "
            f"${p.get('costPerHr', '?')}/ч  {p.get('name')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("cmd", choices=["status", "create", "start", "stop", "restart",
                                    "delete", "stopall", "billing"])
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--json")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()

    if a.cmd == "status":
        pods = call("GET", "/pods")
        eps = call("GET", "/endpoints")
        vols = call("GET", "/networkvolumes")
        print(f"Поды ({len(pods)}):")
        for p in pods:
            print("  " + pod_line(p))
        print(f"Эндпоинты ({len(eps)}):")
        for e in eps:
            print(f"  {e.get('id')}  {e.get('name')}  workers {e.get('workersMin')}-{e.get('workersMax')}")
        print(f"Тома ({len(vols)}):")
        for v in vols:
            print(f"  {v.get('id')}  {v.get('name')}  {v.get('size')} ГБ  {v.get('dataCenterId')}")
        running = [p for p in pods if p.get("desiredStatus") == "RUNNING"]
        if running:
            print(f"\nВНИМАНИЕ: {len(running)} под(ов) RUNNING — идёт списание")
    elif a.cmd == "create":
        need_yes(a, "create")
        if not a.json:
            sys.exit("нужен --json с телом PodCreateInput")
        p = call("POST", "/pods", json.loads(a.json))
        print(pod_line(p))
    elif a.cmd in ("start", "stop", "restart"):
        if not a.arg:
            sys.exit("нужен podId")
        call("POST", f"/pods/{a.arg}/{a.cmd}")
        print(pod_line(call("GET", f"/pods/{a.arg}")))
    elif a.cmd == "delete":
        need_yes(a, "delete")
        if not a.arg:
            sys.exit("нужен podId")
        call("DELETE", f"/pods/{a.arg}")
        print("удалён", a.arg)
    elif a.cmd == "stopall":
        need_yes(a, "stopall")
        for p in call("GET", "/pods"):
            if p.get("desiredStatus") == "RUNNING":
                call("POST", f"/pods/{p['id']}/stop")
                print("остановлен", p["id"], p.get("name"))
    elif a.cmd == "billing":
        b = a.arg or "day"
        print(json.dumps({"pods": call("GET", f"/billing/pods?bucketSize={b}"),
                          "endpoints": call("GET", f"/billing/endpoints?bucketSize={b}")},
                         indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
