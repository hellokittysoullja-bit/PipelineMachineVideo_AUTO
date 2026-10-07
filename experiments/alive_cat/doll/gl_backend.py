"""Рендер куклы через ModernGL (EGL, без экрана): изгиб головы/уха/хвоста, дыхание, наложение на бумагу
и увеличение сразу до 1920x1080. Та же математика, что Rig.frame() на CPU; при недоступном OpenGL —
None, вызывающий код рисует старым способом."""
import numpy as np

FRAG = """#version 330
uniform sampler2D cat; uniform sampler2D wts;   // wts: r=голова, g=ухо, b=хвост
uniform vec2 size; uniform vec4 bb; uniform float scale, off, fl, br;
uniform vec2 neck, earb, tailb; uniform float ah, ae, at; uniform vec3 paper;
in vec2 uv; out vec4 col;
vec4 cr(float x){ // веса Catmull-Rom для дробной части x
  float x2=x*x, x3=x2*x;
  return vec4(-0.5*x3+x2-0.5*x, 1.5*x3-2.5*x2+1.0, -1.5*x3+2.0*x2+0.5*x, 0.5*x3-0.5*x2); }
vec4 bicubic(sampler2D t, vec2 Q){  // Q в пикселях, центр пикселя = целое
  vec2 i=floor(Q), f=Q-i; vec4 wx=cr(f.x), wy=cr(f.y); vec4 acc=vec4(0);
  for(int y=0;y<4;y++){ vec4 row=vec4(0);
    for(int x=0;x<4;x++){ ivec2 p=clamp(ivec2(i)+ivec2(x-1,y-1), ivec2(0), ivec2(size)-1); row+=texelFetch(t,p,0)*wx[x]; }
    acc+=row*wy[y]; }
  return clamp(acc,0.0,1.0); }
vec2 rot(vec2 Q, vec2 p, float a){ float c=cos(a), s=sin(a); vec2 d=Q-p; return p+vec2(c*d.x-s*d.y, s*d.x+c*d.y); }
void main(){
  vec2 O = vec2(uv.x*1920.0, uv.y*1080.0);                  // пиксель результата
  vec2 P = vec2((O.x+0.5)/scale-0.5, (O.y+off+0.5)/scale-0.5); // та же точка в координатах рисунка
  if (P.x<bb.x || P.y<bb.y || P.x>=bb.z || P.y>=bb.w){ col=vec4(paper/255.0,1); return; }
  vec3 w = texture(wts, (P+0.5)/size).rgb;
  vec2 Q = P; Q.y = fl-(fl-Q.y)/br;
  Q = rot(Q, tailb, at*w.b); Q = rot(Q, earb, ae*w.g); Q = rot(Q, neck, ah*w.r);
  vec4 c = bicubic(cat, Q);
  if (Q.x<0.0||Q.y<0.0||Q.x>size.x-1.0||Q.y>size.y-1.0) c=vec4(0);
  col = vec4(mix(paper/255.0, c.rgb, c.a), 1.0);
}"""
VERT = """#version 330
in vec2 p; out vec2 uv; void main(){ uv=(p+1.0)/2.0; gl_Position=vec4(p,0,1);}"""

class GLRenderer:
    def __init__(self, rig):
        import moderngl
        self.rig = rig; r = rig.r; H, W = rig.H, rig.W
        self.ctx = moderngl.create_standalone_context(backend="egl")
        self.prog = self.ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        x0, y0, x1, y1 = rig.bb
        wts = np.zeros((H, W, 3), np.float32)
        wts[y0:y1, x0:x1, 0] = rig.w_head; wts[y0:y1, x0:x1, 1] = rig.w_ear; wts[y0:y1, x0:x1, 2] = rig.w_tail
        self.wtex = self.ctx.texture((W, H), 3, wts.tobytes(), dtype="f4"); self.wtex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.ctex = self.ctx.texture((W, H), 4, dtype="f1"); self.ctex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.scale = 1920 / W; self.off = (H * self.scale - 1080) / 2
        vbo = self.ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], "f4").tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [(vbo, "2f", "p")])
        self.fbo = self.ctx.simple_framebuffer((1920, 1080))
        p = self.prog
        p["cat"].value = 0; p["wts"].value = 1
        p["size"].value = (float(W), float(H)); p["bb"].value = tuple(float(v) for v in rig.bb)
        p["scale"].value = self.scale; p["off"].value = self.off; p["fl"].value = float(r["floor_y"])
        p["neck"].value = tuple(map(float, r["neck"])); p["earb"].value = tuple(map(float, r["ear_base"]))
        p["tailb"].value = tuple(map(float, r["tail_base"])); p["paper"].value = tuple(map(float, rig.paper))

    def frame(self, st, t):
        cat = self.rig.cat_layer(st, t)                       # глаза, позы, огонёк — как раньше
        self.ctex.write(np.ascontiguousarray(np.clip(cat, 0, 255).astype(np.uint8)).tobytes())
        return self.frame_from_layer(st)

    def frame_from_layer(self, st):
        self.ctex.use(0); self.wtex.use(1); self.fbo.use()
        p = self.prog
        p["br"].value = float(st["breath"] * (1 - .02 * st.get("squash", 0)))
        p["ah"].value = float(-np.radians(st["head"])); p["ae"].value = float(-np.radians(st["ear"]))
        p["at"].value = float(-np.radians(st["tail"]))
        self.vao.render(5)                                    # TRIANGLE_STRIP
        img = np.frombuffer(self.fbo.read(components=3), np.uint8).reshape(1080, 1920, 3)
        return img.copy()                                     # строка 0 буфера = верх кадра (uv.y считается сверху в шейдере)

def make(rig):
    try:
        return GLRenderer(rig)
    except Exception as e:                                    # нет EGL/драйвера — рисуем по-старому
        print("ModernGL недоступен, рендер на CPU:", str(e)[:120])
        return None

def render(out, dur, actions, fps=30):
    """Ролик через GL: кадры сразу 1920x1080, ffmpeg только сжимает. Слой кота (CPU) считается
    в параллельных процессах, изгиб и наложение — в GL. Нет GL — None (вызывающий рендерит по-старому)."""
    import subprocess, time
    from multiprocessing import Pool
    import doll_rig as d
    rig = d.Rig(); gl = make(rig)
    if gl is None: return None
    st = d.plan(dur, actions); n = int(fps * dur)
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1920x1080", "-r", str(fps), "-i", "-",
                          "-c:v", "libx264", "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    t0 = time.time()
    with Pool(3) as pool:
        for i, cat in enumerate(pool.imap(_cat, [(i / fps, dur, actions) for i in range(n)], chunksize=8)):
            t = i / fps; s = st(t)
            gl.ctex.write(cat); p.stdin.write(gl.frame_from_layer(s))
    p.stdin.close(); p.wait()
    return time.time() - t0

_R = None; _S = None
def _cat(args):
    global _R, _S
    import doll_rig as d
    t, dur, actions = args
    if _R is None: _R = d.Rig(); _S = d.plan(dur, actions)
    return np.ascontiguousarray(np.clip(_R.cat_layer(_S(t), t), 0, 255).astype(np.uint8)).tobytes()
