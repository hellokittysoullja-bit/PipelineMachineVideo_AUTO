"""Резкость — есть ли в кадре резкая область, а не средняя по кадру.

Фикстуры — НАСТОЯЩИЕ кадры роликов Pexels, которые прогон judge14 (эп.94,
24.09) скачал, судья одобрил, а прежний гейт резкости выбросил:
  * palm_shallow_dof — ладонь крупно на размытом фоне (ролик 13370112);
  * knife_anvil_shallow_dof — клинок на наковальне, фон размыт (5735079);
  * hand_out_of_focus — рука целиком не в фокусе (9831124) — вот это брак.
Прежняя мера (дисперсия Лапласиана по всему кадру, порог 400) отклоняла
все три; на 38 роликах прогонов judge11-14 — 26, а размыты глазами 1-2."""
import os
import subprocess
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]

import pipeline_smart as ps  # noqa: E402

FIX = os.path.join(REPO_ROOT, "tests", "fixtures", "sharpness")
SHALLOW = ["palm_shallow_dof.jpg", "knife_anvil_shallow_dof.jpg"]
BLURRED = "hand_out_of_focus.jpg"


@pytest.mark.parametrize("name", SHALLOW)
def test_shallow_depth_of_field_is_sharp(name):
    path = os.path.join(FIX, name)
    assert ps.image_local_sharpness(path) >= ps.VIDEO_SHARPNESS_REJECT
    assert ps.image_local_sharpness(path) >= ps.PHOTO_SHARPNESS_REJECT
    assert ps.image_sharpness_score(path) < 400, "прежняя мера отклоняла этот кадр — фикстура про это"


def test_out_of_focus_frame_is_not_sharp():
    assert ps.image_local_sharpness(os.path.join(FIX, BLURRED)) < ps.VIDEO_SHARPNESS_REJECT


@pytest.mark.parametrize("name", SHALLOW + [BLURRED])
def test_local_is_never_below_the_average(name):
    """Правка может только перестать отклонять — не отклонить нового."""
    path = os.path.join(FIX, name)
    assert ps.image_local_sharpness(path) >= ps.image_sharpness_score(path)


def _still_video(src, out):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", src, "-t", "2.5",
                    "-r", "24", "-vf", "scale=1280:-2", "-c:v", "libx264", "-crf", "18",
                    "-pix_fmt", "yuv420p", out], check=True)
    return out


def test_video_gate_keeps_the_cinematic_clip_and_drops_the_blurred(tmp_path):
    ok = _still_video(os.path.join(FIX, "palm_shallow_dof.jpg"), str(tmp_path / "palm.mp4"))
    bad = _still_video(os.path.join(FIX, BLURRED), str(tmp_path / "blur.mp4"))
    assert ps.video_sharpness_ok(ok) is True
    assert ps.video_sharpness_ok(bad) is False


def test_photo_winner_check_uses_the_local_measure():
    import inspect
    src = inspect.getsource(ps.PhotoAdapter)
    assert "sharp = image_local_sharpness(cf) if fetched else None" in src
