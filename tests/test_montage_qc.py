"""montage_qc проверяется на синтетике с ИЗВЕСТНОЙ истиной до того, как судит ролик.

Три плана с известными склейками (3.0 и 5.5 с), первые два светлые, третий тёмный
(скачок яркости на второй склейке), звук с паузами ровно на склейках. Две версии
дрейфа: ease_io (камера тормозит к склейке) и линейный (идёт сквозь). Прибор обязан
различить их по скорости и не выдумать лишних резов.

Числа истины: ease_io даёт ~17% окон со скоростью <20% средней и пик/среднее 1.875;
линейный — 0 и 1.0 (оценка по пикселям шумит до ~1.1)."""
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import camera  # noqa: E402
import montage_qc as mq  # noqa: E402

cv2 = pytest.importorskip("cv2")
FPS, W, H = 24, 1920, 1080


def _still(seed, bright):
    r = np.random.default_rng(seed)
    img = np.full((H * 2, W * 2, 3), bright, np.uint8)
    for _ in range(60):
        x, y = r.integers(200, W * 2 - 200), r.integers(200, H * 2 - 200)
        cv2.circle(img, (int(x), int(y)), int(r.integers(20, 120)), (int(r.integers(0, 80)),) * 3, -1)
    return img


def _clip(img, out, dur, profile, zoom_in):
    n = int(dur * FPS); h, w = img.shape[:2]
    win = camera.window(w / 2, h / 2, w * 0.8, w, h)
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for k in range(n):
        u = k / n
        e = camera.ease_io(u) if profile == "ease" else u
        z = 1 + camera.DRIFT * (e if zoom_in else 1 - e)
        x0, y0, x1, y1 = camera.window(w / 2, h / 2, (win[2] - win[0]) / z, w, h)
        vw.write(cv2.resize(img[int(y0):int(y1), int(x0):int(x1)], (W, H), interpolation=cv2.INTER_AREA))
    vw.release()


@pytest.fixture(scope="module")
def material(tmp_path_factory):
    d = tmp_path_factory.mktemp("mqc")
    out = {}
    for prof in ("ease", "lin"):
        parts = []
        for i, (seed, br, dur) in enumerate([(1, 235, 3.0), (2, 235, 2.5), (3, 120, 3.2)]):
            p = str(d / f"{prof}_{i}.mp4"); _clip(_still(seed, br), p, dur, prof, zoom_in=(i % 2 == 0)); parts.append(p)
        lst = d / f"{prof}.txt"; lst.write_text("".join(f"file '{p}'\n" for p in parts))
        ep = d / f"ep_{prof}"; (ep / "media_plan").mkdir(parents=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-f", "lavfi", "-i",
                        "aevalsrc=if(between(t\\,0.2\\,2.85)+between(t\\,3.15\\,5.35)+between(t\\,5.65\\,8.7)\\,0.3*sin(2*PI*220*t)\\,0):s=48000:d=8.7",
                        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(ep / "final.mp4")],
                       check=True)
        out[prof] = str(ep)
    return out


@pytest.fixture(scope="module")
def reports(material):
    return {k: mq.build(v)[0] for k, v in material.items()}


class TestCutsAndHook:
    def test_known_cuts_found_and_no_extra(self, reports):
        cuts = reports["lin"]["cuts"]
        assert len(cuts) == 2 and abs(cuts[0] - 3.0) < 0.1 and abs(cuts[1] - 5.5) < 0.1

    def test_luma_jump_caught_on_dark_picture(self, reports):
        c = reports["lin"]["checks"]["cut_luma_jump"]
        assert c["state"] == "violation" and c["value"] > 50

    def test_voice_onset_and_cuts_in_silence(self, reports):
        ch = reports["lin"]["checks"]
        assert abs(ch["hook_voice_onset_sec"]["value"] - 0.2) < 0.08
        assert ch["cuts_in_silence_share"]["value"] == 1.0 and ch["cuts_in_silence_share"]["state"] == "ok"


class TestCameraSpeed:
    def test_ease_is_caught_linear_is_clean(self, reports):
        e, l = reports["ease"]["measured"], reports["lin"]["measured"]
        assert 0.10 <= e["frozen_share"] <= 0.25 and l["frozen_share"] <= 0.02
        assert e["speed_max_over_mean"] > 1.5 and l["speed_max_over_mean"] < 1.25

    def test_direction_without_picture_map_is_no_signal(self, reports):
        assert reports["lin"]["checks"]["direction_flips_excess"]["state"] == "no_signal"


class TestDiscipline:
    def test_no_signal_never_counts_as_ok(self):
        rec = mq._check("frozen_share", None)
        assert rec["state"] == "no_signal"

    def test_hypothesis_threshold_cannot_block(self):
        for name, (_, _, level, status) in mq.THRESHOLDS.items():
            if level == "block":
                assert status == "calibrated", name

    def test_missing_video_blocks(self, tmp_path):
        rep, code = mq.build(str(tmp_path))
        assert rep["verdict"] == "no_video" and code == mq.EXIT_BLOCK

    def test_qc_is_cheaper_than_realtime(self, reports):
        if os.environ.get("PYTEST_XDIST_WORKER"):
            pytest.skip("замер времени под параллельными воркерами делит ядра — не показатель")
        r = reports["lin"]
        assert r["qc_seconds"] < r["video_seconds"]
