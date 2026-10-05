"""Клип кадра: подпись появляется в момент слова и остаётся, мысль карандашом
дописывается и впекается в рисунок, конец — в крем, а не в чёрное; звук карандаша."""
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="нужен ffmpeg")


@pytest.fixture(autouse=True)
def _no_neural(monkeypatch):
    monkeypatch.setenv("UPSCALE_NEURAL", "0")


def _frame_at(path, t):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", path, "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(1080, 1920, 3).astype(int)


def test_label_appears_at_its_word_and_key_thought_is_written_and_kept(tmp_path):
    import canvas
    import frame_clip
    src = tmp_path / "raw.png"
    im = Image.new("RGB", (1264, 848), (251, 251, 246))
    ImageDraw.Draw(im).ellipse((500, 300, 760, 560), outline=(20, 20, 20), width=8)
    im.save(src)
    rec = {"text": "ДОФАМИН", "font": "ShantellSans-ExtraBold.ttf", "size": 50, "lines": ["ДОФАМИН"],
           "color": "dark", "box": [820, 380, 1100, 470]}
    tex = canvas.paper_texture(w=1152, h=648)
    fr = frame_clip.prepare(str(src), [rec], [], str(tmp_path), tex=tex)
    words = [{"word": w, "start": 0.3 + 0.5*i, "end": 0.6 + 0.5*i}
             for i, w in enumerate("сначала идёт дофамин а потом тянет ещё и ещё".split())]
    p = frame_clip.plan_clip(fr, 5.0, words, key="ещё")
    assert p["label_times"] == pytest.approx([1.3])
    out = str(tmp_path / "c.mp4")
    cues = frame_clip.render(fr, p, 5.0, out, end_fade=True)
    # подпись: до слова её нет, после — есть (в своей рамке, со сдвигом холста)
    sx = 1920/fr["SW"]
    x0, y0 = int((820 + fr["off"][0])*sx), int(380*sx)
    x1, y1 = int((1100 + fr["off"][0])*sx), int(470*sx)
    before = _frame_at(out, 1.0)[y0:y1, x0:x1].min()
    after = _frame_at(out, 1.8)[y0:y1, x0:x1].min()
    assert before > 200 and after < 90
    # мысль карандашом: штрихи есть и идут в моменты после её слова
    assert p.get("key_time") is not None and p["key_time"] >= 3.3 - 1.0
    assert cues and min(c[0] for c in cues) >= p["key_time"] - 1e-6
    # конец — крем, а не чёрное
    last = _frame_at(out, 4.94)
    assert np.abs(last.mean((0, 1)) - canvas.CREAM).max() < 12


def test_pencil_track_is_silent_between_strokes_and_scaled_to_voice(tmp_path):
    import pencil_sound
    cues = [(0.5, 0.9, "stroke", 0.0), (1.0, 1.05, "dot", 0.0), (2.0, 2.4, "stroke", -7.0)]
    trk = pencil_sound.track(cues, 3.0)
    sr = pencil_sound.SR
    assert len(trk) == 3*sr
    assert np.abs(trk[:int(0.45*sr)]).max() == 0 and np.abs(trk[int(1.2*sr):int(1.95*sr)]).max() == 0
    assert np.abs(trk[int(0.6*sr):int(0.8*sr)]).max() > 0
    assert pencil_sound.track([], 3.0) is None
    p = str(tmp_path / "p.wav")
    pencil_sound.write_wav(trk, p)
    v = str(tmp_path / "v.wav")
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=f=220:duration=3", "-ar", "48000", v],
                   check=True)
    g, why = pencil_sound.gain_for(v, p)
    assert g is not None
    vl, pl = pencil_sound.integrated_lufs(v), pencil_sound.integrated_lufs(p)
    assert pl + g == pytest.approx(vl - pencil_sound.PENCIL_GAP_LU, abs=0.2)


def test_diagram_assembles_part_by_part_with_the_voice(tmp_path):
    import canvas
    import frame_clip
    src = tmp_path / "raw.png"
    im = Image.new("RGB", (1264, 848), (251, 251, 246))
    d = ImageDraw.Draw(im)
    d.ellipse((150, 300, 350, 500), fill=(180, 30, 30))          # часть 1
    d.ellipse((800, 300, 1000, 500), fill=(30, 30, 180))         # часть 2
    im.save(src)
    recs = [{"text": t, "font": "ShantellSans-ExtraBold.ttf", "size": 40, "lines": [t], "color": "dark",
             "box": b} for t, b in (("КРАСНЫЙ", [150, 560, 360, 630]), ("СИНИЙ", [800, 560, 1010, 630]))]
    objs = [{"name": "a", "role": "label:0", "box": [150, 300, 350, 500]},
            {"name": "b", "role": "label:1", "box": [800, 300, 1000, 500]}]
    fr = frame_clip.prepare(str(src), recs, objs, str(tmp_path), tex=canvas.paper_texture(w=1152, h=648))
    assert fr["assemble"]
    words = [{"word": w, "start": 0.3 + 0.5*i, "end": 0.6 + 0.5*i}
             for i, w in enumerate("сначала красный а потом синий вот так".split())]
    p = frame_clip.plan_clip(fr, 4.0, words)
    out = str(tmp_path / "a.mp4")
    frame_clip.render(fr, p, 4.0, out)
    sx = 1920/fr["SW"]

    def red_at(t):
        f = _frame_at(out, t)
        w = shots_win(p, t, fr)
        x = ((250 + fr["off"][0]) - w[0])*1920/(w[2] - w[0]); y = (400 - w[1])*1920/(w[2] - w[0])
        return f[int(y), int(x)]
    import shots

    def shots_win(p, t, fr):
        return shots.window_at(p, t, fr["SW"], fr["SH"])
    dim, lit = red_at(0.5), red_at(1.6)                       # «красный» звучит на 0.8 с
    assert dim[0] - dim[1] < 80 and lit[0] - lit[1] > 100     # до слова тускло, после — во всю силу
    assert sx > 0


def test_key_thought_speeds_up_instead_of_disappearing(tmp_path):
    import canvas
    import frame_clip
    src = tmp_path / "raw.png"
    Image.new("RGB", (1264, 848), (251, 251, 246)).save(src)
    fr = frame_clip.prepare(str(src), [], [], str(tmp_path), tex=canvas.paper_texture(w=1152, h=648))
    words = [{"word": w, "start": 0.3 + 0.45*i, "end": 0.6 + 0.45*i}
             for i, w in enumerate("поэтому договорись с собой только открыть письмо".split())]
    p = frame_clip.plan_clip(fr, 4.6, words, key="только открыть")
    assert p["key_time"] is not None and p["key"]["speed"] > 1.0
    assert any("быстрее" in n for n in p["notes"])


def test_accent_is_written_by_hand_on_its_word_and_stays(tmp_path):
    import canvas
    import frame_clip
    src = tmp_path / "raw.png"
    im = Image.new("RGB", (1264, 848), (251, 251, 246))
    ImageDraw.Draw(im).ellipse((150, 250, 450, 600), fill=(40, 40, 40))
    im.save(src)
    fr = frame_clip.prepare(str(src), [], [], str(tmp_path), tex=canvas.paper_texture(w=1152, h=648))
    words = [{"word": w, "start": 0.3 + 0.45*i, "end": 0.6 + 0.45*i}
             for i, w in enumerate("ответить на письмо это пять минут не больше".split())]
    D = 6.0
    p = frame_clip.plan_clip(fr, D, words, accent="пять минут")
    assert p["accent_time"] == pytest.approx(words[4]["start"]) and p.get("accent")
    out = str(tmp_path / "a.mp4")
    cues = frame_clip.render(fr, p, D, out)
    # брендбук: на экране только рукописные слова, со звуком карандаша — акцент тоже пишется штрихами
    assert cues and min(c[0] for c in cues) >= p["accent_time"] - 1e-6
    end = max(c[1] for c in cues)
    cx, cy = [int(v) for v in p["accent"]["center"]]
    box = (slice(cy - 40, cy + 40), slice(cx - 160, cx + 160))
    assert _frame_at(out, p["accent_time"] - 0.3)[box].min() > 180     # до слова — пусто
    mid = _frame_at(out, (p["accent_time"] + end)/2)[box]
    done = _frame_at(out, D - 0.1)[box]
    assert done.min() < 120                                            # дописано и остаётся
    assert (mid < 150).sum() < (done < 150).sum()                      # на середине — написана только часть
