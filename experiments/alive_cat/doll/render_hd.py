import subprocess, time, numpy as np, cv2
from multiprocessing import Pool
import doll_rig as d
_R=None; _S=None
def work(a):
    global _R,_S
    t,dur,act=a
    if _R is None: _R=d.Rig(); _S=d.plan(dur,act)
    return _R.frame_hd(_S(t),t).tobytes()
def render(out,dur,act,fps=30):
    p=subprocess.Popen(["ffmpeg","-v","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s","1920x1080","-r",str(fps),"-i","-","-c:v","libx264","-crf","17","-preset","medium","-pix_fmt","yuv420p",out],stdin=subprocess.PIPE)
    t0=time.time()
    with Pool(4) as pool:
        for b in pool.imap(work,[(i/fps,dur,act) for i in range(int(fps*dur))],chunksize=8): p.stdin.write(b)
    p.stdin.close(); p.wait(); return time.time()-t0
if __name__=="__main__":
    ACT=[{"t":1.2,"do":"look","x":.9,"y":.3,"dur":1.6},{"t":1.6,"do":"tilt","deg":5,"dur":1.2},{"t":4.4,"do":"look","x":-.8,"y":0,"dur":1.2}]
    print(f"новый CPU (один пересчёт в 1920): {render('bench_hd.mp4',9.0,ACT):.1f} с на 9 с ролика")
