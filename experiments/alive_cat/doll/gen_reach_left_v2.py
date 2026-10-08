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
    "the small flame, same hind paws with orange pads, the same front foot on the viewer's right, same plain white "
    "paper background, same line work and colors.\n"
    "Change only one small thing: the front leg on the viewer's LEFT. Keep its top part exactly as it is now — it "
    "starts from the same place under the chest, with the same thickness, mirroring the other front leg one to one. "
    "Only the lower part of this leg moves a little: the paw slides slightly down-and-out to the viewer's left, at a "
    "steep angle of about 30 to 45 degrees from vertical (NOT horizontal, NOT stretched out sideways), and lifts just "
    "a tiny bit off the floor. The leg keeps exactly the same length as the other front leg — it is NOT longer. The "
    "paw ends only about half a paw-width to the left of where it stood before, still close to the body outline. "
    "The pads face down toward the floor, the soft toes point toward the left. A tiny, careful, lazy movement, as if "
    "the cat is about to touch something lying right next to it on the floor. The cat has exactly two front legs and "
    "two hind paws. Nothing is added to the background. No text, no other changes.")
gw = Gateway(spend_cap=160000)
imgs, price = gw.image("ag/gemini-3.1-flash-image", prompt, "1536x1024", quality="medium", images=[url])
Image.open(io.BytesIO(imgs[0])).save(sys.argv[1]); print("price", price)
