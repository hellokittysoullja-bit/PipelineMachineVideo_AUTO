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
    "the small flame, same hind paws with orange pads, same plain white paper background, same line work and colors.\n"
    "Change only one thing, with correct real cat anatomy: the front leg on the viewer's LEFT gently reaches out "
    "sideways to the viewer's left and a little forward, LOW, close to the floor, the way a sitting cat cautiously "
    "stretches one front paw toward something lying on the floor beside it. The leg comes from under the chest, the "
    "elbow is slightly bent, the forearm goes out to the left at a low angle, the paw hovers just above the floor "
    "with the soft toes pointing to the left and the pads facing down. It is a small, careful, lazy reach — not raised "
    "up, not to the chest, not like a human arm. Only ONE front foot still stands on the ground (the one on the "
    "viewer's right); where the reaching leg used to stand, the lower chest fur and its ink outline are drawn. The cat "
    "has exactly two front legs and two hind paws. Nothing is added to the background. No text, no other changes.")
gw = Gateway(spend_cap=160000)
imgs, price = gw.image("ag/gemini-3.1-flash-image", prompt, "1536x1024", quality="medium", images=[url])
Image.open(io.BytesIO(imgs[0])).save(sys.argv[1]); print("price", price)
