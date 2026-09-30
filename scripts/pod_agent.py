#!/usr/bin/env python3
"""Агент внутри RunPod-пода: команды, файлы и самоостановка по HTTP (порт 8000).

Ставится командой `runpod_ctl.py up`, руками не запускается. Только stdlib.
Авторизация — Bearer POD_AGENT_TOKEN. Самоостановка защищает от забытой GPU:
под гасится, если нет работы POD_IDLE_MIN минут (нет задач агента и GPU < 5%) или прошло POD_MAX_HOURS часов с запуска.
"""
import hmac
import json
import os
import subprocess
import threading
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

TOKEN = os.environ["POD_AGENT_TOKEN"]
PORT = int(os.environ.get("POD_AGENT_PORT", "8000"))
IDLE_MIN = float(os.environ.get("POD_IDLE_MIN", "20"))
MAX_H = float(os.environ.get("POD_MAX_HOURS", "6"))
GPU_BUSY_PCT = 5.0
START = time.time()
JOBS = {}
LAST_BUSY = [time.time()]
LOG = "/tmp/pod_agent.log"


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def gpu_util():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return max(float(x) for x in out.split())
    except Exception:
        return -1.0


def running_jobs():
    return [j for j, p in JOBS.items() if p["proc"].poll() is None]


def is_busy():
    return bool(running_jobs()) or gpu_util() > GPU_BUSY_PCT


def stop_pod(reason):
    log(f"stop: {reason}")
    pod, key = os.environ.get("RUNPOD_POD_ID"), os.environ.get("RUNPOD_API_KEY")
    if pod and key:
        try:
            req = urllib.request.Request(
                f"https://rest.runpod.io/v1/pods/{pod}/stop", method="POST",
                headers={"Authorization": f"Bearer {key}", "User-Agent": "pod-agent/1.0"})
            urllib.request.urlopen(req, timeout=30).read()
            return True
        except Exception as e:
            log(f"rest stop failed: {e}")
    r = subprocess.run(["runpodctl", "stop", "pod", pod or ""], capture_output=True, text=True)
    log(f"runpodctl rc={r.returncode} {r.stdout[-200:]} {r.stderr[-200:]}")
    return r.returncode == 0


def watchdog():
    while True:
        time.sleep(30)
        now = time.time()
        if is_busy():
            LAST_BUSY[0] = now
        if now - START > MAX_H * 3600:
            stop_pod(f"max lifetime {MAX_H}h")
        elif now - LAST_BUSY[0] > IDLE_MIN * 60:
            stop_pod(f"idle {IDLE_MIN} min")


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _auth(self):
        got = self.headers.get("Authorization", "")
        if not hmac.compare_digest(got, f"Bearer {TOKEN}"):
            self._send(401, {"error": "unauthorized"})
            return False
        return True

    def _send(self, code, obj=None, raw=None):
        body = raw if raw is not None else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/octet-stream" if raw is not None
                         else "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        return self.rfile.read(int(self.headers.get("Content-Length", 0)))

    def do_GET(self):
        if not self._auth():
            return
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/status":
            idle = (time.time() - LAST_BUSY[0]) / 60
            self._send(200, {"uptime_min": round((time.time() - START) / 60, 1),
                             "gpu_util": gpu_util(), "load": os.getloadavg()[0],
                             "running_jobs": running_jobs(), "idle_min": round(idle, 1),
                             "idle_limit_min": IDLE_MIN,
                             "life_left_h": round(MAX_H - (time.time() - START) / 3600, 2)})
        elif u.path.startswith("/job/"):
            j = JOBS.get(u.path[5:])
            if not j:
                return self._send(404, {"error": "no such job"})
            tail = int(q.get("tail", ["4000"])[0])
            with open(j["log"], "rb") as f:
                data = f.read()
            code = j["proc"].poll()
            self._send(200, {"running": code is None, "code": code,
                             "log": data[-tail:].decode("utf-8", "replace")})
        elif u.path == "/file":
            try:
                with open(q["path"][0], "rb") as f:
                    self._send(200, raw=f.read())
            except Exception as e:
                self._send(404, {"error": str(e)})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self._auth():
            return
        if self.path == "/exec":
            req = json.loads(self._body())
            jid = uuid.uuid4().hex[:8]
            lf = f"/tmp/job_{jid}.log"
            proc = subprocess.Popen(["bash", "-lc", req["cmd"]], cwd=req.get("cwd") or None,
                                    stdout=open(lf, "wb"), stderr=subprocess.STDOUT)
            JOBS[jid] = {"proc": proc, "log": lf}
            LAST_BUSY[0] = time.time()
            log(f"job {jid}: {req['cmd'][:200]}")
            self._send(200, {"job": jid})
        elif self.path == "/stop":
            self._send(200, {"stopping": True})
            threading.Thread(target=stop_pod, args=("manual",), daemon=True).start()
        else:
            self._send(404, {"error": "not found"})

    def do_PUT(self):
        if not self._auth():
            return
        path = parse_qs(urlparse(self.path).query)["path"][0]
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = self._body()
        with open(path, "wb") as f:
            f.write(data)
        self._send(200, {"bytes": len(data)})


if __name__ == "__main__":
    threading.Thread(target=watchdog, daemon=True).start()
    log(f"agent up port={PORT} idle={IDLE_MIN}m max={MAX_H}h")
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
