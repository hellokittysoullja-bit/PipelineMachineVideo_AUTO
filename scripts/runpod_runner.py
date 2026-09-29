#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Исполнитель задач НА ПОДЕ Runpod (сторона видеокарты для runpod_job.py).

Запускается командой контейнера пода и живёт ровно столько, сколько нужна
задача. Снаружи к нему ходят только через HTTPS-прокси Runpod
(https://<pod>-<порт>.proxy.runpod.net): SSH из среды, где работает
пайплайн, закрыт.

Всё по токену (заголовок X-Runner-Token, переменная RUNNER_TOKEN): без него
любой, кто знает адрес пода, мог бы запускать команды на чужой карте.

  GET  /health                        — жив ли (без токена: только «ok»)
  PUT  /upload?name=X&offset=N        — кусок архива tar / tar.gz (докачка по смещению)
  POST /extract?name=X                — распаковать загруженный архив в /work
  POST /run                           — {"cmd": "...", "cwd": "..."} запустить
  GET  /log?offset=N                  — лог задачи с байта N, статус и код выхода
  GET  /download?path=P               — tar.gz файла/папки из /work

НИКОГДА НЕ СТОИТ ПРОСТО ТАК. Под удаляет сам себя (runpodctl remove pod
$RUNPOD_POD_ID — ключ пода ограничен самим подом):
  * задача не идёт и обращений нет RUNNER_IDLE_SEC секунд (по умолчанию 600)
    — локальная сторона пропала (обрыв сети, закрытая сессия);
  * прошло RUNNER_MAX_SEC секунд с запуска (по умолчанию 4 часа) — жёсткий
    потолок на любой случай, даже если задача зависла.
Удалённый под не стоит ничего: у Runpod посекундная тарификация, диск
контейнера после удаления не оплачивается.
"""
import io
import json
import os
import subprocess
import tarfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

WORK = os.environ.get("RUNNER_WORK", "/work")
UPLOADS = os.path.join(WORK, ".uploads")
LOG = os.path.join(WORK, ".job.log")
PORT = int(os.environ.get("RUNNER_PORT", "8000"))
TOKEN = os.environ.get("RUNNER_TOKEN", "")
IDLE_SEC = float(os.environ.get("RUNNER_IDLE_SEC", "600"))
MAX_SEC = float(os.environ.get("RUNNER_MAX_SEC", str(4 * 3600)))
POLL_SEC = float(os.environ.get("RUNNER_POLL_SEC", "15"))

STATE = {"started": time.time(), "last_seen": time.time(), "proc": None,
         "exit": None, "cmd": None, "terminating": False, "job_id": None, "extracted": set()}
LOCK = threading.Lock()
EXTRACT_LOCK = threading.Lock()


UA = "pipeline-runpod-runner/1.0"   # Cloudflare перед API Runpod режет подпись Python по умолчанию (403)


def _api_delete(pod, key):
    """Удалить под через API ключом пода: REST, затем GraphQL. True — API
    ответил без ошибки."""
    import urllib.request
    for req in (
        urllib.request.Request(f"https://rest.runpod.io/v1/pods/{pod}", method="DELETE",
                               headers={"Authorization": f"Bearer {key}", "User-Agent": UA}),
        urllib.request.Request(
            "https://api.runpod.io/graphql", method="POST",
            data=json.dumps({"query": f'mutation {{ podTerminate(input: {{podId: "{pod}"}}) }}'}).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}",
                     "User-Agent": UA})):
        try:
            urllib.request.urlopen(req, timeout=60).read()
            return True
        except Exception as e:  # noqa: BLE001 — следующий путь
            print(f"runner: {req.get_method()} {req.full_url}: {e}", flush=True)
    return False


def self_terminate(reason):
    """Удалить этот под. runpodctl (ключ пода), затем REST, затем GraphQL —
    и так по кругу, пока одно не сработает: под, который не смог удалить
    себя, стоит за деньги. Повторно не зовётся."""
    with LOCK:
        if STATE["terminating"]:
            return
        STATE["terminating"] = True
    print(f"runner: удаляю под — {reason}", flush=True)
    pod = os.environ.get("RUNPOD_POD_ID", "")
    if os.environ.get("RUNNER_DRY_TERMINATE"):
        print(f"runner: (сухой режим) remove pod {pod}", flush=True)
        return
    key = os.environ.get("RUNPOD_API_KEY", "")
    for attempt in range(1000):
        try:
            subprocess.run(["runpodctl", "remove", "pod", pod], timeout=60, check=True)
            return
        except Exception as e:  # noqa: BLE001 — пробуем API
            print(f"runner: runpodctl не сработал ({e})", flush=True)
        if _api_delete(pod, key):
            return
        time.sleep(min(60, 5 * (attempt + 1)))


def job_running():
    p = STATE["proc"]
    return p is not None and p.poll() is None


def watchdog():
    while not STATE["terminating"]:
        now = time.time()
        if now - STATE["started"] > MAX_SEC:
            self_terminate(f"потолок времени {MAX_SEC:.0f} с")
        elif not job_running() and now - STATE["last_seen"] > IDLE_SEC:
            self_terminate(f"нет задачи и обращений {IDLE_SEC:.0f} с")
        time.sleep(POLL_SEC)


def _safe(path):
    full = os.path.realpath(os.path.join(WORK, path))
    if full != os.path.realpath(WORK) and not full.startswith(os.path.realpath(WORK) + os.sep):
        raise ValueError("путь вне рабочей папки")
    return full


def _reap(proc):
    code = proc.wait()
    STATE["exit"] = code
    STATE["last_seen"] = time.time()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, payload, ctype="application/json"):
        data = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _auth(self):
        if not TOKEN or self.headers.get("X-Runner-Token") != TOKEN:
            self._send(403, {"error": "нет токена"})
            return False
        STATE["last_seen"] = time.time()
        return True

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == "/health":
            return self._send(200, {"ok": True})
        if not self._auth():
            return
        try:
            if u.path == "/log":
                off = int(q.get("offset", 0))
                chunk = b""
                if os.path.exists(LOG):
                    with open(LOG, "rb") as f:
                        f.seek(off)
                        chunk = f.read(1 << 20)
                return self._send(200, {"offset": off + len(chunk),
                                        "text": chunk.decode("utf-8", "replace"),
                                        "running": job_running(), "exit": STATE["exit"],
                                        "uptime": time.time() - STATE["started"]})
            if u.path == "/download":
                src = _safe(q["path"])
                buf = io.BytesIO()
                with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                    # Путь внутри архива — от /work: у запросившего файл ляжет
                    # по тому же относительному пути (assets/calibration/...).
                    tar.add(src, arcname=os.path.relpath(src, os.path.realpath(WORK)))
                return self._send(200, buf.getvalue(), "application/gzip")
        except (KeyError, ValueError, OSError) as e:
            return self._send(400, {"error": str(e)})
        self._send(404, {"error": "нет такого"})

    def do_PUT(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if not self._auth():
            return
        if u.path != "/upload":
            return self._send(404, {"error": "нет такого"})
        try:
            os.makedirs(UPLOADS, exist_ok=True)
            name = os.path.basename(q["name"])
            dest = os.path.join(UPLOADS, name)
            off = int(q.get("offset", 0))
            have = os.path.getsize(dest) if os.path.exists(dest) else 0
            if off != have:
                # Кусок не на своём месте (повтор после обрыва) — сказать, с
                # какого байта продолжать, а не портить файл.
                return self._send(409, {"size": have})
            with open(dest, "ab") as f:
                f.write(self._body())
            return self._send(200, {"size": os.path.getsize(dest)})
        except (KeyError, ValueError, OSError) as e:
            return self._send(400, {"error": str(e)})

    def do_POST(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if not self._auth():
            return
        try:
            if u.path == "/extract":
                name = os.path.basename(q["name"])
                if name in STATE["extracted"]:     # повтор после сбоя связи
                    return self._send(200, {"ok": True, "again": True})
                src = os.path.join(UPLOADS, name)
                with tarfile.open(src, "r:*") as tar:        # tar или tar.gz
                    members = tar.getmembers()
                    for m in members:
                        _safe(m.name)          # архив не пишет вне /work
                    # Архивы одной папки распаковываются одновременно (загрузка
                    # в несколько потоков): tarfile создаёт общие папки без
                    # «уже есть» и падает на гонке — папки создаются заранее,
                    # сама запись идёт под замком.
                    with EXTRACT_LOCK:
                        for m in members:
                            d = os.path.dirname(os.path.join(WORK, m.name))
                            os.makedirs(d, exist_ok=True)
                        tar.extractall(WORK)
                os.remove(src)
                STATE["extracted"].add(name)
                return self._send(200, {"ok": True})
            if u.path == "/run":
                req = json.loads(self._body() or b"{}")
                if req.get("id") and req["id"] == STATE["job_id"]:
                    # Повтор того же запуска после сбоя связи: задача уже
                    # запущена (или даже закончилась) — второй раз не запускать.
                    return self._send(200, {"ok": True, "again": True})
                if job_running():
                    return self._send(409, {"error": "задача уже идёт"})
                cwd = _safe(req.get("cwd", "."))
                with open(LOG, "ab") as log:
                    proc = subprocess.Popen(["bash", "-lc", req["cmd"]], cwd=cwd, stdout=log,
                                            stderr=subprocess.STDOUT)
                STATE.update(proc=proc, exit=None, cmd=req["cmd"], job_id=req.get("id"))
                threading.Thread(target=_reap, args=(proc,), daemon=True).start()
                return self._send(200, {"ok": True, "pid": proc.pid})
            if u.path == "/touch":
                # Отметка для задачи на поде (например, «все данные загружены»):
                # задача ждёт этот файл, а не начало загрузки.
                open(_safe(os.path.basename(q["name"])), "a").close()
                return self._send(200, {"ok": True})
            if u.path == "/terminate":
                self._send(200, {"ok": True})
                threading.Thread(target=self_terminate, args=("по запросу",), daemon=True).start()
                return
        except (KeyError, ValueError, OSError) as e:
            return self._send(400, {"error": str(e)})
        self._send(404, {"error": "нет такого"})


def serve():
    os.makedirs(WORK, exist_ok=True)
    threading.Thread(target=watchdog, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"runner: слушаю :{PORT}, простой {IDLE_SEC:.0f} с, потолок {MAX_SEC:.0f} с", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    if not TOKEN:
        raise SystemExit("RUNNER_TOKEN не задан — без токена исполнитель не запускается")
    serve()
