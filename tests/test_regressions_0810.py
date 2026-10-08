"""Откаты, найденные сравнением с прежним роликом эпизода 01 (08.10, три независимых рецензента), заперты
кодом: голова героя целиком и выше зоны плеера, без обрубков тела, без планов на пустой бумаге, дописанная
мысль стоит в кадре, направление дрейфа от смысла склейки, ритм без метронома, кукла без эмоций страха,
герой в финале, must-утверждение судьи, карандаш без голоса, моргания не пачками."""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, os.path.join(REPO_ROOT, "experiments", "alive_cat", "doll"))

import canvas  # noqa: E402
import placement  # noqa: E402
import shots  # noqa: E402
import camera  # noqa: E402


def _scene():
    """Кот слева (голова сверху), конверт справа внизу — как кадр 1 эпизода 01."""
    im = Image.new("RGB", (1264, 848), (251, 251, 246))
    d = ImageDraw.Draw(im)
    d.ellipse((150, 120, 450, 400), fill=(40, 40, 40))         # голова
    d.rectangle((200, 380, 420, 700), fill=(30, 30, 30))       # тело
    d.rectangle((700, 560, 1100, 760), fill=(60, 60, 60))      # конверт
    cv, off, _ = canvas.prepare(im)
    busy = placement.busy_map(cv.astype(np.float32), margin=8)
    ox = off[0]
    objs = [{"role": "subject", "name": "a letter", "box": (700 + ox, 560, 1100 + ox, 760)},
            {"role": "hero", "box": (150 + ox, 120, 450 + ox, 700)},
            {"role": "hero_head", "box": (150 + ox, 120, 450 + ox, 400)},
            {"role": "detail", "name": "the envelope", "box": (700 + ox, 560, 1100 + ox, 760)},
            {"role": "detail", "name": "the eyes", "box": (220 + ox, 220, 380 + ox, 300)}]
    return busy, objs, ox


def test_view_rules_head_body_and_ink():
    busy, objs, ox = _scene()
    p = shots.plan(8.0, busy, [], objects=objs)
    ok = shots.make_view_ok(busy, objs, p["views"]["wide"])
    SH, SW = busy.shape
    head = objs[2]["box"]
    # голова целиком, но подбородок у нижнего края — лицо в зоне плеера
    low = camera.window((head[0] + head[2])/2, head[3] + 0.06*450 - 225, 800, SW, SH)   # низ головы у низа окна
    assert (head[3] - low[1])/(low[3] - low[1]) > shots.HEAD_MAX_BOTTOM and not ok(low)
    # та же крупность, голова выше зоны плеера — годен
    fine = camera.window((head[0] + head[2])/2, head[1] + 0.55*(head[3] - head[1]) + 60, 800, SW, SH)
    assert ok(fine) or (head[3] - fine[1])/(fine[3] - fine[1]) > shots.HEAD_MAX_BOTTOM
    # обрубок: окно конверта захватывает тело без головы
    torso = camera.window(600 + ox, 650, 700, SW, SH)
    assert not ok(torso, objs[3]["box"])
    # план на пустой бумаге
    blank = camera.window(1000 + ox, 150, 500, SW, SH)
    assert not ok(blank)
    # деталь внутри головы: лицо без подбородка — нет, настоящий макро глаз — да
    eyes = objs[4]["box"]
    half = camera.window((eyes[0] + eyes[2])/2, 200, 600, SW, SH)                 # окно 337 px высотой: подбородок срезан
    assert not ok(half, eyes)
    macro = camera.window((eyes[0] + eyes[2])/2, (eyes[1] + eyes[3])/2, (eyes[2] - eyes[0])/0.6, SW, SH)
    assert ok(macro, eyes, ink=False)
    # и всё, что план реально выбрал, этим правилам подчиняется
    for sg in p["segments"]:
        if sg["win"] != p["views"]["wide"]:
            assert ok(sg["win"], ink=False)


def test_written_thought_stays_in_frame_until_held():
    busy, objs, ox = _scene()
    key_box = (900 + ox, 100, 1250 + ox, 200)        # надпись справа вверху — в средний план кота не влезает
    p = shots.plan(8.0, busy, [], objects=objs, key="только открыть", key_dur=3.0, key_box=key_box)   # письмо до 3.9, удержание до 4.8: склейка после письма попадает в удержание
    assert p["key_time"] is not None
    hold_end = p["key_time"] + 3.0 + shots.KEY_HOLD_SEC
    for sg in p["segments"]:
        if sg["t0"] < hold_end - 1e-6 and sg["t1"] > p["key_time"]:
            w = sg["win"]
            assert w[0] <= key_box[0] and w[1] <= key_box[1] and w[2] >= key_box[2] and w[3] >= key_box[3], \
                f"план {sg['t0']:.2f}-{sg['t1']:.2f} режет надпись"
    # без рамки надписи склейка после письма шла бы раньше конца удержания — рамка её сдвигает
    p0 = shots.plan(8.0, busy, [], objects=objs, key="только открыть", key_dur=3.0)
    cuts0 = [sg["t0"] for sg in p0["segments"][1:]]
    cuts = [sg["t0"] for sg in p["segments"][1:]]
    assert not cuts0 or not cuts or min(cuts) >= min(cuts0) - 1e-6


def test_drift_direction_follows_the_cut():
    busy, objs, ox = _scene()
    p = shots.plan(9.0, busy, [], objects=objs)
    SH, SW = busy.shape
    prev = None
    for sg in p["segments"]:
        z = camera.zoom_of(sg["win"], SW, SH)
        if prev is not None and sg["kind"] == "drift":
            if z > prev*1.02:
                assert sg["zoom_in"] is True
            elif z < prev/1.02:
                assert sg["zoom_in"] is False
        prev = z


def test_splits_without_word_timings_are_not_a_metronome_but_deterministic():
    busy, objs, ox = _scene()
    a = shots.plan(9.0, busy, [], objects=objs, T0=0.0)
    b = shots.plan(9.0, busy, [], objects=objs, T0=0.0)
    c = shots.plan(9.0, busy, [], objects=objs, T0=17.3)
    la = [round(s["t1"] - s["t0"], 3) for s in a["segments"]]
    assert la == [round(s["t1"] - s["t0"], 3) for s in b["segments"]]          # детерминизм
    assert len(la) >= 2
    assert la != [round(s["t1"] - s["t0"], 3) for s in c["segments"]]          # другой кадр — другие длины
    assert max(la) - min(la) > 0.05                                               # не ровные куски


def test_live_doll_does_not_take_frames_that_need_fear_or_shame():
    import frame_planner as fp
    f = {"hero": True, "hero_action": "look", "picture": "the main character sits beside the envelope",
         "spec": {"focus": "shame sticks to the task", "claims": [{"id": "c2", "text": "the character looks at it with dread", "tier": "must"}]}}
    g = {"hero": True, "hero_action": "look", "picture": "the main character sits beside the open envelope, calm",
         "spec": {"focus": "only open it", "claims": [{"id": "c1", "text": "the envelope is open", "tier": "must"}]}}
    assert fp.live_emotion_blocked(f) and not fp.live_emotion_blocked(g)
    assert fp.live_hero_pass([f, g]) == 1
    assert f["hero"] is True and not f.get("hero_live") and f.get("hero_live_blocked") == "emotion"
    assert g["hero"] is False and g["hero_live"] is True


def test_hero_limit_keeps_the_finale():
    import frame_planner as fp
    frames = [{"hero": True, "picture": "the main character here"} for _ in range(5)]
    fp.limit_hero(frames, max_run=2, max_share=0.35)
    assert frames[-1]["hero"] is True


def test_pencil_without_voice_is_set_from_master_target(monkeypatch):
    import pencil_sound as ps
    monkeypatch.setattr(ps, "integrated_lufs", lambda path: (-70.0 if "voice" in path else -20.0))
    g, why = ps.gain_for("voice.wav", "pencil.wav")
    assert abs(g - (ps.MASTER_TARGET_I - ps.PENCIL_GAP_LU + 20.0)) < 1e-6 and "master_target" in why
    monkeypatch.setattr(ps, "integrated_lufs", lambda path: (-16.0 if "voice" in path else -20.0))
    g2, why2 = ps.gain_for("voice.wav", "pencil.wav")
    assert abs(g2 - (-16.0 - ps.PENCIL_GAP_LU + 20.0)) < 1e-6 and "master_target" not in why2


def test_master_does_not_normalize_a_voiceless_mix():
    import audio_master as am
    silent = {"input_i": "-55.0", "input_tp": "-20", "input_lra": "1", "input_thresh": "-65", "target_offset": "0"}
    assert am.build_master_af(silent, 30.0, 0.05).startswith("anull,")
    voiced = dict(silent, input_i="-18.0")
    assert am.build_master_af(voiced, 30.0, 0.05).startswith("loudnorm=")
    # одиночный штрих карандаша в тишине меряется с гейтом как −28 LUFS — сборщик говорит явно, что голоса нет
    assert am.build_master_af(dict(silent, input_i="-28.0"), 30.0, 0.05, voiceless=True).startswith("anull,")


def test_text_in_a_view_is_whole_or_absent():
    busy, objs, ox = _scene()
    text = (900 + ox, 100, 1250 + ox, 200)
    ok = shots.make_view_ok(busy, objs, camera.window(754, 424, 1508, *busy.shape[::-1]), [text])
    SH, SW = busy.shape
    cut = camera.window(1000 + ox, 300, 700, SW, SH)          # край окна по надписи
    assert not ok(cut, ink=False)
    whole = camera.window(1075 + ox, 300, 900, SW, SH)
    assert ok(whole, ink=False) or not (whole[0] <= text[0] and whole[2] >= text[2])
    away = camera.window(900 + ox, 660, 600, SW, SH)          # план конверта: надписи в нём нет вовсе
    assert ok(away, ink=False)


def test_doll_blinks_are_spaced_and_quiet_before_the_cut():
    import doll_rig
    out = doll_rig.thin_blinks([1.0, 1.08, 1.17, 1.5, 3.0, 3.32, 6.8, 6.9], 7.0)
    assert out == [1.0, 1.5, 3.0, 3.32]
    assert all(b2 - b1 >= doll_rig.BLINK_MIN_GAP for b1, b2 in zip(out, out[1:]))
    assert all(b <= 7.0 - doll_rig.END_QUIET_SEC for b in out)


def test_qc_does_not_count_the_handwriting_plan_as_too_long(tmp_path):
    import montage_qc as mq
    plans = [(0.0, 2.0), (2.0, 6.6), (6.6, 9.0)]
    import pytest
    assert mq.length_metrics(plans)["plan_max_sec"] == pytest.approx(4.6)
    m = mq.length_metrics(plans, writing=[(2.3, 6.3)])       # мысль пишется с 2.3, стоит до 6.3, склейка 6.6
    assert m["plan_max_sec"] == pytest.approx(2.4) and m["writing_plans"] == 1
    m2 = mq.length_metrics(plans, writing=[(2.3, 4.0)])      # план тянется 2.6 с после удержания — считается
    assert m2["plan_max_sec"] == pytest.approx(4.6)
    mp = tmp_path / "media_plan"; mp.mkdir()
    (mp / "shots_report.json").write_text(json.dumps({"clips": [
        {"index": 0, "note": "x"}, {"index": 0, "duration": 2.0, "key_at": None, "key_hold_until": None},
        {"index": 1, "duration": 7.0, "key_at": 0.3, "key_hold_until": 4.3}]}), encoding="utf-8")
    (w0, w1), = mq.writing_windows(str(tmp_path))
    assert (w0, w1) == (pytest.approx(2.3), pytest.approx(6.3))
