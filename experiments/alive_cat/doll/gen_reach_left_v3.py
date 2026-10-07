import sys, io, base64
sys.path.insert(0, "/home/user/DoodleExplainer_AUTO/scripts")
from llm_gateway import Gateway
from PIL import Image
src = Image.open("/home/user/DoodleExplainer_AUTO/look/hero.png").convert("RGB")
b = io.BytesIO(); src.save(b, "PNG")
url = "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()
prompt = (
    "Edit this exact drawing of a cartoon cat sitting and facing the viewer. Keep everything identical: same black "
    "watercolor-and-ink cat, same size and position, same head, face, big green eyes, whiskers, orange cheek marks, "
    "same ears (the ear on the viewer's left is folded down, the ear on the viewer's right stands up), same tail with "
    "the small flame, same hind paws with orange pads, the same front foot on the viewer's right, same plain white "
    "paper background, same line work and colors.\n"
    "Change only one small thing: the front leg on the viewer's LEFT (the left one of the two small front feet in "
    "the middle, between the hind paws). Its top part stays exactly where it is now: it starts from the same place "
    "under the chest, with the same thickness, mirroring the other front leg one to one — it must NOT come out from "
    "the side of the body and must NOT be outside the left hind paw. Only the lower part of this leg moves a little: "
    "the front paw slides slightly down-and-left and lifts a tiny bit, so it now lies IN FRONT OF the left hind paw, "
    "partly overlapping and covering the upper-right part of that hind paw (the front paw is closer to the viewer). "
    "The leg keeps exactly the same length as the other front leg. The front paw is seen from above: the top of the "
    "paw with the dark fur and toes is visible, the pads face DOWN toward the floor and are hidden, the soft toes "
    "point to the left. A tiny, careful movement, as if the cat is about to touch something lying right next to it on "
    "the floor. The cat has exactly two front legs and two hind paws. Nothing is added to the background. "
    "No text, no other changes.")
gw = Gateway(spend_cap=160000)
imgs, price = gw.image("ag/gemini-3.1-flash-image", prompt, "1536x1024", quality="medium", images=[url])
Image.open(io.BytesIO(imgs[0])).save(sys.argv[1]); print("price", price)
