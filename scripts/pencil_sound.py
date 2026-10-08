#!/usr/bin/env python3
"""Звук карандаша: дорожка из коротких зёрен настоящих записей, по штрихам письма.

Записи — Freesound #383867 и #370789, обе CC0 (LICENSES.md). Решения из демо
(04.10, на слух владельца и по замерам):
  * зёрна 60-140 мс из активных участков записи, громкость каждого выровнена
    (без «провалов» и «щелчков» громкости), соседние зёрна не повторяются
    (пул ~400 зёрен) — склейки с перекрытием 10 мс, без щелчков;
  * запись чуть ускорена (×1.12 полифазным ресемплингом, без потери верха):
    в исходной скорости штрих звучал вяло;
  * громкость внутри штриха идёт за скоростью руки (медленнее на концах);
    точка — короткий спадающий «тык»;
  * фонового «шороха» между штрихами нет (решение владельца: лишний звук);
  * уровень в миксе — от ЗАМЕРА голоса: дорожка ставится на PENCIL_GAP_LU ниже
    голоса (интегральная громкость; тишина между штрихами в замер не входит —
    её отсекает абсолютный гейт EBU R128)."""
import json
import os
import subprocess

import numpy as np
from scipy import signal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 48000
FILES = [os.path.join(ROOT, "assets", "sfx", f) for f in ("pencil_a.mp3", "pencil_b.mp3")]
PENCIL_GAP_LU = float(os.environ.get("PENCIL_GAP_LU", "14"))
LABEL_GAIN_DB = -7.0


def _load(p):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).astype(np.float64)


class Grains:
    def __init__(self, files=FILES, seed=3, rate=(25, 28)):
        self.rng = np.random.default_rng(seed)
        self.pool, self.recent = [], []
        hp = signal.butter(2, 160, "hp", fs=SR, output="sos")
        for f in files:
            x = signal.resample_poly(signal.sosfilt(hp, _load(f)), *rate)
            n = int(0.01*SR); m = len(x)//n
            rms = np.sqrt((x[:m*n].reshape(m, n)**2).mean(1)) + 1e-9
            act = 20*np.log10(rms/np.percentile(rms, 99)) > -22
            i = 0
            while i < m:
                if not act[i]:
                    i += 1
                    continue
                j = i
                while j < m and act[j]:
                    j += 1
                st = i*n
                while st + int(0.06*SR) <= j*n:
                    L = int(self.rng.uniform(0.06, 0.14)*SR)
                    if st + L > j*n:
                        break
                    g = x[st:st + L].copy()
                    self.pool.append(g/(np.sqrt((g**2).mean()) + 1e-9))
                    st += int(L*0.7)
                i = j
        if not self.pool:
            raise RuntimeError("звук карандаша: в записях нет активных участков")

    def stroke(self, dur, xf=0.010):
        out = np.zeros(int(dur*SR) + 1)
        pos, x = 0, int(xf*SR)
        while pos < len(out):
            for _ in range(20):
                k = int(self.rng.integers(len(self.pool)))
                if k not in self.recent:
                    break
            self.recent = (self.recent + [k])[-24:]
            g = self.pool[k].copy()
            fade = max(1, min(x, len(g)//3))
            g[:fade] *= np.linspace(0, 1, fade); g[-fade:] *= np.linspace(1, 0, fade)
            e = min(len(out), pos + len(g))
            out[pos:e] += g[:e - pos]
            pos += len(g) - fade
        return out[:int(dur*SR)]


def track(cues, total, seed=3):
    """cues — [(t0, t1, kind, gain_db)] в секундах ролика -> моно float32 длиной total.
    Пусто — None (дорожки нет, в микс ничего не идёт)."""
    if not cues:
        return None
    G = Grains(seed=seed)
    n = int(total*SR)
    mono = np.zeros(n + SR)
    for t0, t1, kind, gdb in sorted(cues):
        seg = G.stroke(max(0.04, t1 - t0))
        if not len(seg):
            continue
        u = np.linspace(0, 1, len(seg))
        v = np.exp(-u*4) if kind == "dot" else 0.5 + 0.5*(30*u**2*(1 - u)**2/1.875)
        f8 = max(1, min(len(seg)//3, int(0.006*SR)))
        v[:f8] *= np.linspace(0, 1, f8); v[-f8:] *= np.linspace(1, 0, f8)
        seg *= v*10**((gdb + G.rng.uniform(-1.5, 1.5))/20)*0.08
        i0 = int(t0*SR)
        if i0 >= n:
            continue
        e = min(len(mono), i0 + len(seg))
        mono[i0:e] += seg[:e - i0]
    return mono[:n].astype(np.float32)


def write_wav(mono, path):
    tmp = path + ".f32"
    mono.astype(np.float32).tofile(tmp)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", tmp,
                    "-c:a", "pcm_s24le", path], check=True)
    os.remove(tmp)


def integrated_lufs(path):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-af", "ebur128=peak=true",
                        "-f", "null", "-"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    tail = r.stderr[r.stderr.rfind("Summary:"):]
    for line in tail.splitlines():
        line = line.strip()
        if line.startswith("I:"):
            try:
                return float(line.split()[1])
            except ValueError:
                return None
    return None


def has_voice(voice_lufs):
    """В дорожке голоса есть голос: интегральная громкость измерена и не ниже audio_master.NO_VOICE_LUFS.
    Замер не удался — True: без уверенности мастер работает как обычно."""
    import audio_master
    return voice_lufs is None or voice_lufs >= audio_master.NO_VOICE_LUFS


def gain_for(voice_lufs, pencil_path, gap=PENCIL_GAP_LU):
    """Усиление дорожки карандаша, дБ: на gap LU ниже голоса (voice_lufs — уже измеренная интегральная
    громкость голоса, один замер на сборку). Голоса нет (тишина, заглушка превью) — на gap LU ниже цели
    мастера: иначе единственный звук файла, штрих, уходил в мастер сырым и loudnorm поднимал его до
    −14 LUFS (превью эп.01 08.10: пик штриха −1.4 dBFS). (усиление, пояснение)."""
    import audio_master
    v, p = voice_lufs, integrated_lufs(pencil_path)
    if p is None or p < -69:
        return None, f"замер не удался (голос {v}, карандаш {p})"
    if not has_voice(v):
        g = float(np.clip(audio_master.LOUDNORM_TARGET_I - gap - p, -40, 20))
        return g, json.dumps({"voice_lufs": v, "pencil_lufs": p, "gap_lu": gap, "ref": "master_target"})
    g = float(np.clip(v - gap - p, -40, 20))
    return g, json.dumps({"voice_lufs": v, "pencil_lufs": p, "gap_lu": gap})
