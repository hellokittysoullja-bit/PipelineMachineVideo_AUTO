import sys
import io
import base64
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
    "the small flame, same hind paws with orange pads, same plain white paper background, same line work and colors.\n"
    "Change only one thing, with correct real cat anatomy: the front leg on the viewer's LEFT is lifted the way a real "
    "sitting cat lifts a front paw. The leg comes from under the chest, the elbow stays tucked against the chest, the "
    "forearm rises forward in front of the chest, and the paw is bent down at the wrist (a soft curled paw, like a "
    "beckoning lucky cat), held in front of the upper chest just below the chin. The paw does not touch the face and "
    "does not cover the eyes, nose or mouth. The leg does NOT come out from the side of the body and is not raised like "
    "a human arm. Because this leg is lifted, only ONE front foot now stands on the ground (the one on the viewer's "
    "right); where the lifted foot used to stand, the lower chest fur and its ink outline are drawn. The cat has exactly "
    "two front legs and two hind paws. No text, no other changes.")
gw = Gateway(spend_cap=160000)
imgs, price = gw.image("ag/gemini-3.1-flash-image", prompt, "1536x1024", quality="medium", images=[url])
out = sys.argv[1] if len(sys.argv) > 1 else "beckon.png"
Image.open(io.BytesIO(imgs[0])).save(out); print("price", price)
