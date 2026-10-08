import sys
import io
import base64
sys.path.insert(0, "/home/user/DoodleExplainer_AUTO/scripts")
from llm_gateway import Gateway
from PIL import Image
src = Image.open("beckon1.png").convert("RGB")
b = io.BytesIO(); src.save(b, "PNG")
url = "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()
prompt = (
    "Edit this exact drawing of a cartoon cat sitting and facing the viewer with one front paw raised in front of its "
    "chest. Keep everything identical: same black watercolor-and-ink cat, same size and position, same head, face, big "
    "green eyes, whiskers, orange cheek marks, same ears (the ear on the viewer's left is folded down), same tail with "
    "the small flame, same hind paws, the same single front foot standing on the ground, same lower chest, same plain "
    "white paper background, same line work and colors.\n"
    "Change only one thing, small and realistic: the raised paw makes a gentle, barely-there pointing gesture toward the "
    "RIGHT side of the picture. With correct real cat anatomy: the elbow stays tucked against the chest exactly where it "
    "is, only the forearm swings a little to the viewer's right across the front of the chest, the wrist unbends a little "
    "so the toes point to the right, the paw stays at the same height below the chin and in front of the chest. It is a "
    "subtle, lazy movement, not a human pointing finger: no extended single toe, just the soft paw tipped toward the "
    "right. The paw does not touch the face. The cat still has exactly two front legs and two hind paws. "
    "No text, no other changes.")
gw = Gateway(spend_cap=160000)
imgs, price = gw.image("ag/gemini-3.1-flash-image", prompt, "1536x1024", quality="medium", images=[url])
Image.open(io.BytesIO(imgs[0])).save(sys.argv[1]); print("price", price)
