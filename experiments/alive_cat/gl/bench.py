import time
import numpy as np
import cv2
import moderngl
from PIL import Image
src = np.asarray(Image.open("/home/user/DoodleExplainer_AUTO/look/hero.png").convert("RGBA"))
H, W = src.shape[:2]
NECK = (618., 575.); FB = (920., 270., 1050., 480.)
# ---------- OpenCV: тот же расчёт, что в риге (поворот с весом + огонёк), 1264x848, потом апскейл до 1920
YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)
w = np.clip((585 - YY) / 55, 0, 1); w = w * w * (3 - 2 * w)
def cv_frame(t):
    ang = np.radians(4 * np.sin(t)) * w
    c, s = np.cos(ang), np.sin(ang); dx, dy = XX - NECK[0], YY - NECK[1]
    X = NECK[0] + c * dx - s * dy; Y = NECK[1] + s * dx + c * dy
    x0, y0, x1, y1 = FB; inside = (XX >= x0) & (XX < x1) & (YY >= y0) & (YY < y1)
    k = np.clip(1 - (YY - y0) / (y1 - y0), 0, 1) ** 1.5 * inside
    X = X - k * (4 * np.sin(2 * np.pi * (YY / 45 - t * 2.3)))
    out = cv2.remap(src, X.astype(np.float32), Y.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return cv2.resize(out, (1920, 1288), interpolation=cv2.INTER_LANCZOS4)
# ---------- ModernGL: тот же расчёт во фрагментном шейдере, рисуем сразу в 1920x1288
ctx = moderngl.create_standalone_context(backend="egl")
prog = ctx.program(vertex_shader="""#version 330
in vec2 p; out vec2 uv; void main(){ uv=(p+1.0)/2.0; uv.y=1.0-uv.y; gl_Position=vec4(p,0,1);}""",
fragment_shader="""#version 330
uniform sampler2D tex; uniform float t; uniform vec2 size; in vec2 uv; out vec4 col;
void main(){
  vec2 P = uv*size; vec2 neck=vec2(618.,575.);
  float w = smoothstep(0.0,1.0,clamp((585.0-P.y)/55.0,0.0,1.0));
  float a = radians(4.0*sin(t))*w; float c=cos(a), s=sin(a); vec2 d=P-neck;
  vec2 Q = neck + vec2(c*d.x - s*d.y, s*d.x + c*d.y);
  if (P.x>=920.0 && P.x<1050.0 && P.y>=270.0 && P.y<480.0){
     float k = pow(clamp(1.0-(P.y-270.0)/210.0,0.0,1.0),1.5);
     Q.x -= k*4.0*sin(6.2831853*(P.y/45.0 - t*2.3)); }
  col = texture(tex, Q/size);
}""")
tex = ctx.texture((W, H), 4, src.tobytes()); tex.filter = (moderngl.LINEAR, moderngl.LINEAR); tex.use(0)
vbo = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], "f4").tobytes())
vao = ctx.vertex_array(prog, [(vbo, "2f", "p")])
fbo = ctx.simple_framebuffer((1920, 1288)); fbo.use()
prog["size"].value = (float(W), float(H))
def gl_frame(t):
    prog["t"].value = t; vao.render(moderngl.TRIANGLE_STRIP)
    return np.frombuffer(fbo.read(components=4), np.uint8).reshape(1288, 1920, 4)
for name, f in (("OpenCV + апскейл", cv_frame), ("ModernGL (llvmpipe)", gl_frame)):
    f(0.0); t0 = time.time()
    for i in range(30): o = f(i / 30)
    print(f"{name}: {(time.time()-t0)/30*1000:.0f} мс/кадр")
a = cv_frame(1.0)[..., :3].astype(float); b = gl_frame(1.0)[..., :3].astype(float)
print("разница картинок, среднее:", np.abs(a - b).mean().round(2), "max:", np.abs(a-b).max())
Image.fromarray(cv_frame(1.0)[..., :3]).crop((800, 300, 1400, 900)).save("cv.png"); Image.fromarray(gl_frame(1.0)[..., :3]).crop((800, 300, 1400, 900)).save("gl.png")
