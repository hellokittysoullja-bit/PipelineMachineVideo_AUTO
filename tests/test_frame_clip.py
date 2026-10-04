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
    p = frame_clip.plan_clip(fr, 4.0, words, key="ещё")
    assert p["label_times"] == pytest.approx([1.3])
    out = str(tmp_path / "c.mp4")
    cues = frame_clip.render(fr, p, 4.0, out, end_fade=True)
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
    last = _frame_at(out, 3.94)
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
