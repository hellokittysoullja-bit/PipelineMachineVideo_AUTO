import json
import sys
import time
from PIL import Image
from rembg import new_session, remove
boxes=json.load(open("boxes.json"))
model=sys.argv[1]
s=new_session(model)
import os; os.makedirs(model,exist_ok=True)
only=sys.argv[2:]
for k,b in sorted(boxes.items()):
    if only and k not in only: continue
    if not b: continue
    im=Image.open(f"src/{k}.png").convert("RGB"); W,H=im.size
    x1,y1,x2,y2=b; px=(x2-x1)*.12; py=(y2-y1)*.12
    c=(max(0,int(x1-px)),max(0,int(y1-py)),min(W,int(x2+px)),min(H,int(y2+py)))
    t=time.time(); m=remove(im.crop(c),session=s,only_mask=True)
    full=Image.new("L",im.size,0); full.paste(m,c[:2])
    full.save(f"{model}/{k}_mask.png"); print(k,model,round(time.time()-t,1),"s",flush=True)
