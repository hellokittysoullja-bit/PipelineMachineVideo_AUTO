import os, sys, time
import doll_rig as D
mode = sys.argv[1]; D.FLAME_NOISE = mode == "noise"
ACT = [{"t": 1.2, "do": "look", "x": .9, "y": .3, "dur": 1.6}, {"t": 7.6, "do": "look", "x": 0, "y": 0, "dur": .5}]
sec = D.render(f"flame_{mode}.mp4", 9.0, ACT)
print(mode, f"{sec:.1f}")
