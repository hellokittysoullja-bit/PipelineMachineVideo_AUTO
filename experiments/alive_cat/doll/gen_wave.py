import sys, io, base64, json
sys.path.insert(0, "/home/user/DoodleExplainer_AUTO/scripts")
from llm_gateway import Gateway
from PIL import Image
src = Image.open("/home/user/DoodleExplainer_AUTO/look/hero.png").convert("RGB")
b = io.BytesIO(); src.save(b, "PNG")
url = "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()
prompt = (
    "Edit this exact drawing. It is the reference character sheet of a cartoon cat; keep it identical: same black "
    "watercolor-and-ink cat, same size and same position in the frame, same head, same face, same big green eyes, same "
    "whiskers, same orange cheek marks, same ears (the ear on the viewer's left is folded down, the ear on the viewer's "
    "right stands up), same curled tail with the small flame on its tip, same hind paws with orange pads, same plain "
    "white paper background, same line work and colors. "
    "Change only one thing: the cat's front leg on the viewer's LEFT side is raised in a friendly wave. That leg lifts "
    "away from the body to the viewer's left and up, the paw held open at about the height of the cheek, palm with "
    "orange toe beans facing the viewer, completely outside the outline of the head and body, with a clear gap of white "
    "paper between the raised paw and the face. The cat's chest and belly where that leg used to be are fully drawn: "
    "fur continues and the dark ink outline of the lower edge of the body is drawn there. The other front leg stays "
    "down exactly as before. No text, no other changes.")
gw = Gateway(spend_cap=160000)
imgs, price = gw.image("ag/gemini-3.1-flash-image", prompt, "1536x1024", quality="medium", images=[url])
Image.open(io.BytesIO(imgs[0])).save("raised.png")
print("price", price, Image.open(io.BytesIO(imgs[0])).size, json.dumps(gw.summary(), ensure_ascii=False)[:300])
