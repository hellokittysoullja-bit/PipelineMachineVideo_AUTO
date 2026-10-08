"""Пилот куклы в сборщике: baseline (до правок кода) | nomascot (после, без куклы — обязан совпасть с baseline по md5)
| mascot | samecam (план с куклой, рендер без неё — эталон камеры для замера ореола)."""
import hashlib
import json
import os
import sys
import time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "scripts")); sys.path.insert(0, HERE)
import pilot_scene as scene
import frame_clip
mode = sys.argv[1]            # baseline | nomascot | mascot | samecam
src, objs = scene.build(); work = os.path.join(scene.P, "work"); os.makedirs(work, exist_ok=True)
fr = frame_clip.prepare(src, [], objs, work, seed=0)
kw = {}
if mode in ("mascot", "samecam"):
    kw["mascot"] = dict(text=scene.TEXT, state="ember", seed=0)
t = time.time()
p = frame_clip.plan_clip(fr, scene.D, scene.words(), key="Пять минут", T0=0.0, zoom_in=True, fps=24, **kw)
if mode == "samecam":
    p.pop("mascot", None)
out = os.path.join(scene.P, f"clip_{mode}.mp4")
cues = frame_clip.render(fr, p, scene.D, out, fps=24, end_fade=False, crf="18")
print(mode, "render", round(time.time()-t, 1), "s;", "md5", hashlib.md5(open(out,"rb").read()).hexdigest(), "mascot" in p, [s["kind"] for s in p["segments"]])
json.dump(frame_clip.plan_record(p), open(os.path.join(scene.P, f"plan_{mode}.json"), "w"), ensure_ascii=False, indent=1)
