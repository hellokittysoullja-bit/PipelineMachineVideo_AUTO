"""Метрики монтажа по готовому файлу (те же оси, что у рецензентов 08.10): склейки по доле изменившихся
пикселей, длины планов (CV, доля соседей <0.2 с, самая длинная цепочка ±0.3 с), ΔY и сдвиг центра внимания на
склейках, доля чернил по планам, герой в последних 3 с (доля тёмных пикселей), планы >3.54 с."""
import sys, json, subprocess
import numpy as np
sys.path.insert(0, '/home/user/DoodleExplainer_AUTO/scripts')
import verify_timing as vt

path = sys.argv[1]
W, H = 192, 108
raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-vf', f'scale={W}:{H}', '-f', 'rawvideo', '-pix_fmt', 'gray', '-'],
                     capture_output=True).stdout
fr = np.frombuffer(raw, np.uint8).reshape(-1, H, W).astype(np.float32)
fps, diffs = vt.frame_change_share_curve(path, 64, 36, vt.SHARE_PIXEL_STEP)
cuts = vt.cuts_from_curve(fps, diffs, ratio=4.0, min_abs=vt.SHARE_MIN_ABS)[0]
n = len(fr); dur = n/fps
bounds = [0.0] + list(cuts) + [dur]
plans = [(a, b) for a, b in zip(bounds, bounds[1:])]
lens = np.array([b - a for a, b in plans])
cv = float(lens.std()/lens.mean())
near = float(np.mean([abs(l2 - l1) < 0.2 for l1, l2 in zip(lens, lens[1:])])) if len(lens) > 1 else 0.0
# самая длинная цепочка планов в полосе ±0.3 с
best = run = 1
for l1, l2 in zip(lens, lens[1:]):
    run = run + 1 if abs(l2 - l1) <= 0.3 else 1; best = max(best, run)
def luma(t): return float(fr[min(n-1, int(round(t*fps)))].mean())
def ink_share(t):
    f = fr[min(n-1, int(round(t*fps)))]; return float((f < 200).mean())
def centre(t):
    f = fr[min(n-1, int(round(t*fps)))]; m = (255 - f); m = m*(f < 200)
    if m.sum() < 1: return 0.5
    xs = np.arange(W); return float((m.sum(0)*xs).sum()/m.sum()/W)
dY = [abs(luma(c + 1.5/fps) - luma(c - 1.5/fps)) for c in cuts]
shift = [abs(centre(c + 1.5/fps) - centre(c - 1.5/fps)) for c in cuts]
ink_plans = [min(ink_share(a + 0.3), ink_share((a + b)/2), ink_share(b - 0.3)) for a, b in plans]
tail_ink = float(np.mean([(fr[i] < 90).mean() for i in range(max(0, n - int(3*fps)), n)]))   # очень тёмное = шерсть кота
out = dict(file=path, duration=round(dur, 2), plans=len(plans), cuts=[round(c, 2) for c in cuts],
           plan_len=dict(min=round(float(lens.min()), 2), median=round(float(np.median(lens)), 2), max=round(float(lens.max()), 2), cv=round(cv, 3)),
           near_equal_share=round(near, 3), longest_similar_run=best, over_3_54=int((lens > 3.54).sum()),
           cut_dY=dict(max=round(max(dY), 1), mean=round(float(np.mean(dY)), 1), over_12=sum(d > 12 for d in dY)),
           cut_shift=dict(max=round(max(shift), 2), over_025=sum(s > 0.25 for s in shift)),
           min_plan_ink=round(min(ink_plans), 3), plans_ink_below_0_10=sum(i < 0.10 for i in ink_plans),
           dark_share_last3s=round(tail_ink, 4))
print(json.dumps(out, ensure_ascii=False, indent=1))
