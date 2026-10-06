import sys, io, base64
sys.path.insert(0, "/home/user/DoodleExplainer_AUTO/scripts")
from llm_gateway import Gateway
from PIL import Image
src = Image.open("../cutout/src/10.png").convert("RGB")
cx, cy, side = 375, 380, 400            # квадрат вокруг головы (глаза ~283..467 x, 341..421 y)
box = (cx - side // 2, cy - side // 2, cx + side // 2, cy + side // 2)
crop = src.crop(box); crop.save("head.png")
b = io.BytesIO(); crop.resize((1024, 1024), Image.LANCZOS).save(b, "PNG")
url = "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()
gw = Gateway(spend_cap=0)
prompt = ("Edit the reference image. Keep EVERYTHING exactly the same: the same black watercolor cartoon cat, "
          "same position, same size, same ears, same fur, same whiskers, same colors, same paper background, same framing. "
          "Change only one thing: both eyes are now fully closed, each eye drawn as a soft curved dark eyelid line "
          "(a gentle downward arc) on dark fur, like a calm blink. No green iris visible. No other changes.")
for i in range(int(sys.argv[1]) if len(sys.argv) > 1 else 2):
    imgs, price = gw.image("am/flux.2-klein-4b", prompt, "1024x1024", images=[url])
    Image.open(io.BytesIO(imgs[0])).save(f"closed_{i}.png"); print(i, "price", price, Image.open(io.BytesIO(imgs[0])).size)
