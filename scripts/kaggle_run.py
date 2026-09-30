#!/usr/bin/env python3
"""Запуск отбора кадров эпизода на Kaggle (T4 x2) и забор результата.

    python scripts/kaggle_run.py <video_dir> [--render] [--pull-only]

Что делает:
  1. Собирает приватный датасет <user>/pmv-code: репозиторий без .env, secrets/,
     videos/, .git, тяжёлых кэшей + папка эпизода.
  2. Ключи НЕ загружаются: ноутбук читает их из Kaggle Secrets (Add-ons -> Secrets).
  3. Пушит ноутбук с GPU: ставит зависимости, копирует код, запускает
     pipeline_smart.py --select-only (или полный рендер с --render).
  4. Ждёт завершения и кладёт результат в <video_dir>/kaggle_out/.

Токен Kaggle — KAGGLE_API_TOKEN из .env / окружения. В stdout ключи не печатаются.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXCLUDE = {".git", ".venv", "__pycache__", "videos", "secrets", "temp_smart", "models",
           "temp_museum_cache", "temp_openverse_cache", "temp_commons_cache",
           "temp_cascade_embed_cache", "temp_aesthetic_cache", ".pytest_cache"}
EXCLUDE_FILES = {".env"}


def load_env():
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def kg(*args, check=True):
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    if check and r.returncode:
        sys.exit(f"kaggle {' '.join(args[:2])}: {r.stdout}{r.stderr}")
    return r


def whoami():
    r = kg("datasets", "list", "--mine", check=False)
    for line in r.stdout.splitlines()[2:]:
        return line.split("/")[0]
    return os.environ.get("KAGGLE_USERNAME") or sys.exit("нужен KAGGLE_USERNAME")


def push_dataset(folder, slug, title, user):
    (folder / "dataset-metadata.json").write_text(json.dumps(
        {"title": title, "id": f"{user}/{slug}", "licenses": [{"name": "CC0-1.0"}]}))
    r = kg("datasets", "version", "-p", str(folder), "-m", "update", "-r", "zip", check=False)
    if r.returncode:
        kg("datasets", "create", "-p", str(folder), "-r", "zip")


def build_code(dst, video_dir):
    for item in ROOT.iterdir():
        if item.name in EXCLUDE or item.name in EXCLUDE_FILES:
            continue
        if item.is_dir():
            shutil.copytree(item, dst / item.name,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(item, dst / item.name)
    ep = dst / "videos" / video_dir.name
    shutil.copytree(video_dir, ep, ignore=shutil.ignore_patterns("kaggle_out", "final.mp4"))


KERNEL = r'''
import os, shutil, subprocess, sys, time
W = "/kaggle/working"
def sh(c, **k):
    print("$", c, flush=True)
    return subprocess.run(c, shell=True, **k)
code = next(p for p in ("/kaggle/input/pmv-code", "/kaggle/input/datasets/%(user)s/pmv-code") if os.path.isdir(p))
repo = W + "/repo"
shutil.copytree(code, repo)
from kaggle_secrets import UserSecretsClient
_sc = UserSecretsClient()
_names = "KAGGLE_API_TOKEN PEXELS_API_KEY PIXABAY_API_KEY OPENVERSE_CLIENT_ID OPENVERSE_CLIENT_SECRET LUMEAN_API_KEY LLM_GATEWAY_API_KEY UNSPLASH_API_KEY GEMINI_API_KEY".split()
_lines = ["LLM_GATEWAY_BASE_URL=https://anymodel.org/v1"]
for _n in _names:
    try: _lines.append(_n + "=" + _sc.get_secret(_n))
    except Exception: print("secret не задан:", _n)
open(repo + "/.env", "w").write("\n".join(_lines) + "\n")
for _l in _lines:
    if _l.startswith("KAGGLE_API_TOKEN="): os.environ["KAGGLE_API_TOKEN"] = _l.split("=", 1)[1]
os.environ["KAGGLE_USERNAME"] = "%(user)s"
os.chdir(repo)
print("EP_FILES", os.listdir("videos/%(ep)s") if os.path.isdir("videos/%(ep)s") else "NO EP DIR", "CODE_MOUNT", code, flush=True)
sh("nvidia-smi -L")
req = [l for l in open("requirements.txt") if l.strip() and not l.startswith("#")
       and not l.lower().startswith(("torch", "pytest"))]
open("req_k.txt", "w").writelines(req)
sh("pip install -q -r req_k.txt transformers accelerate sentencepiece protobuf onnxruntime-gpu huggingface_hub 2>&1 | tail -3")

import threading, json
def _pulse_loop():
    pdir = W + "/pulse"; os.makedirs(pdir, exist_ok=True)
    tok = os.environ.get("KAGGLE_API_TOKEN"); made = False
    while True:
        try:
            gpu = subprocess.run("nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader",
                                 shell=True, capture_output=True, text=True).stdout.strip()
            try: tail = open(W + "/run.log", errors="ignore").read()[-6000:]
            except Exception: tail = ""
            info = {"t": time.strftime("%%H:%%M:%%S"), "elapsed_sec": round(time.time() - T0), "gpu": gpu,
                    "load": os.getloadavg(), "log_tail": tail}
            json.dump(info, open(pdir + "/pulse.json", "w"), ensure_ascii=False)
            if tok:
                open(pdir + "/dataset-metadata.json", "w").write(json.dumps(
                    {"title": "pmv-pulse", "id": "%(user)s/pmv-pulse", "licenses": [{"name": "CC0-1.0"}]}))
                c = "kaggle datasets version -p %%s -m pulse -r zip" %% pdir if made else "kaggle datasets create -p %%s -r zip" %% pdir
                r = subprocess.run(c, shell=True, capture_output=True, text=True)
                if r.returncode and not made:
                    r = subprocess.run("kaggle datasets version -p %%s -m pulse -r zip" %% pdir, shell=True, capture_output=True, text=True)
                made = made or not r.returncode
        except Exception as e:
            print("pulse error:", e, flush=True)
        time.sleep(120)
T0 = time.time()
threading.Thread(target=_pulse_loop, daemon=True).start()
env = dict(os.environ, ML_DEVICE="cuda", CLIP_ENCODER="auto", SLOT_SPECULATE="1")
t = time.time()
r = sh("python -u scripts/%(entry)s videos/%(ep)s %(flags)s 2>&1 | tee %(W)s/run.log", env=env)
print("ELAPSED_SEC", round(time.time() - t), flush=True)
out = W + "/out"
os.makedirs(out, exist_ok=True)
src = repo + "/videos/%(ep)s"
for name in ("media_plan", "final.mp4"):
    p = src + "/" + name
    if os.path.isdir(p): shutil.copytree(p, out + "/" + name, dirs_exist_ok=True)
    elif os.path.isfile(p): shutil.copy(p, out)
shutil.copy(W + "/run.log", out)
shutil.rmtree(repo, ignore_errors=True)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_dir")
    ap.add_argument("--render", action="store_true", help="полный рендер (нужен audio.mp3)")
    ap.add_argument("--pull-only", action="store_true")
    a = ap.parse_args()
    load_env()
    vd = Path(a.video_dir).resolve()
    user = whoami()
    slug = f"{user}/pmv-run"
    work = Path(os.environ.get("TMPDIR", "/tmp")) / "pmv_kaggle"
    out_dir = vd / "kaggle_out"
    if not a.pull_only:
        shutil.rmtree(work, ignore_errors=True)
        (work / "code").mkdir(parents=True)
        (work / "kernel").mkdir()
        build_code(work / "code", vd)
        push_dataset(work / "code", "pmv-code", "pmv-code", user)
        flags = "" if a.render else "--select-only"
        entry = "render_episode.py" if a.render else "pipeline_smart.py"
        (work / "kernel" / "run.py").write_text(KERNEL % dict(
            user=user, ep=vd.name, flags=flags, entry=entry, W="/kaggle/working"))
        (work / "kernel" / "kernel-metadata.json").write_text(json.dumps({
            "id": slug, "title": "pmv-run", "code_file": "run.py", "language": "python",
            "kernel_type": "script", "is_private": "true", "enable_gpu": "true",
            "enable_internet": "true", "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [f"{user}/pmv-code"]}))
        print(kg("kernels", "push", "-p", str(work / "kernel")).stdout.strip())
        while True:
            s = kg("kernels", "status", slug, check=False).stdout
            print(s.strip(), flush=True)
            if any(w in s for w in ("COMPLETE", "ERROR", "CANCEL")):
                break
            time.sleep(60)
    out_dir.mkdir(exist_ok=True)
    print(kg("kernels", "output", slug, "-p", str(out_dir), check=False).stdout[-300:])
    print("Результат:", out_dir)


if __name__ == "__main__":
    main()
