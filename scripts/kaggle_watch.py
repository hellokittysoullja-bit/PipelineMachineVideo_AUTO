#!/usr/bin/env python3
"""Живой пульс ноутбука Kaggle: скачивает датасет <user>/pmv-pulse и печатает состояние.

    python scripts/kaggle_watch.py [--follow]
Пульс пишет ноутбук раз в 2 минуты (нужен секрет KAGGLE_API_TOKEN в Kaggle Secrets).
"""
import json, os, subprocess, sys, tempfile, time
def once(user):
    d = tempfile.mkdtemp()
    r = subprocess.run(["kaggle", "datasets", "download", f"{user}/pmv-pulse", "-p", d, "--unzip", "--force"],
                       capture_output=True, text=True)
    f = os.path.join(d, "pulse.json")
    if r.returncode or not os.path.exists(f):
        print("пульса ещё нет:", (r.stderr or r.stdout).strip()[:200]); return False
    i = json.load(open(f))
    print(f"[{i['t']}] идёт {i['elapsed_sec']//60} мин  load={i['load']}")
    print("GPU (idx, util%, mem used/total, temp):"); print(i["gpu"])
    print("--- хвост лога ---"); print(i["log_tail"][-1500:]); return True
if __name__ == "__main__":
    user = os.environ.get("KAGGLE_USERNAME", "rokxaguy")
    while True:
        once(user)
        if "--follow" not in sys.argv: break
        time.sleep(120)
