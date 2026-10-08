import numpy as np
import onnxruntime as ort
import sys
import time
from PIL import Image, ImageFilter
s=ort.InferenceSession("lama_fp32.onnx")
for k in sys.argv[1:]:
    im=Image.open(f"src/{k}.png").convert("RGB"); W,H=im.size
    m=Image.open(f"birefnet-general/{k}_mask.png").point(lambda v:255 if v>40 else 0)
    m=m.filter(ImageFilter.MaxFilter(25))   # +12px: ореол и мех
    bb=m.getbbox(); cx,cy=(bb[0]+bb[2])/2,(bb[1]+bb[3])/2
    side=int(min(max(bb[2]-bb[0],bb[3]-bb[1])*1.5,min(W,H)))
    x0=int(min(max(cx-side/2,0),W-side)); y0=int(min(max(cy-side/2,0),H-side))
    box=(x0,y0,x0+side,y0+side)
    ci=np.asarray(im.crop(box).resize((512,512),Image.LANCZOS)).astype(np.float32)/255
    cm=(np.asarray(m.crop(box).resize((512,512),Image.NEAREST))>127).astype(np.float32)
    t=time.time()
    out=s.run(None,{"image":ci.transpose(2,0,1)[None],"mask":cm[None,None]})[0][0]
    out=out.transpose(1,2,0); print(k,"range",out.min(),out.max(),round(time.time()-t,1),"s")
    if out.max()<=1.5: out=out*255
    patch=Image.fromarray(np.clip(out,0,255).astype(np.uint8)).resize((side,side),Image.LANCZOS)
    res=im.copy(); res.paste(patch,box[:2],m.crop(box).filter(ImageFilter.GaussianBlur(3)))
    res.save(f"lama/{k}_clean.png")
