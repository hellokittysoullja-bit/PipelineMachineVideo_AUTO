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
    fine = camera.window((head[0] + head[2])/2, (head[1] + head[3])/2, 800, SW, SH)   # голова по центру окна
    assert (head[3] - fine[1])/(fine[3] - fine[1]) <= shots.HEAD_MAX_BOTTOM and ok(fine)
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
    # письмо до 3.9, удержание до 4.8: склейка после письма попадает в удержание
    p = shots.plan(8.0, busy, [], objects=objs, key="только открыть", key_dur=3.0, key_box=key_box)
    assert p["key_time"] is not None
    hold_end = p["key_time"] + 3.0 + shots.KEY_HOLD_SEC
    for sg in p["segments"]:
        if sg["t0"] < hold_end - 1e-6 and sg["t1"] > p["key_time"]:
            w = sg["win"]
            assert camera.inside(key_box, w, tol=0.0), f"план {sg['t0']:.2f}-{sg['t1']:.2f} режет надпись"
    # без рамки надписи склейка после письма идёт сразу после хвоста письма (до конца удержания) —
    # рамка сдвигает её к концу удержания
    p0 = shots.plan(8.0, busy, [], objects=objs, key="только открыть", key_dur=3.0)
    cuts0 = [sg["t0"] for sg in p0["segments"][1:]]
    cuts = [sg["t0"] for sg in p["segments"][1:]]
    assert cuts0 and cuts and min(cuts0) < hold_end - 1e-6 <= min(cuts) + 1e-6
    # и пока мысль пишется и стоит — камера идёт наездом к ней (намерение правки, не пересказ формулы)
    p_out = shots.plan(8.0, busy, [], objects=objs, key="только открыть", key_dur=3.0, key_box=key_box, zoom_in=False)
    assert all(sg["zoom_in"] is True for sg in p_out["segments"] if sg["t0"] <= p_out["key_time"] < sg["t1"])


def test_drift_direction_is_one_per_picture_and_pushes_in_on_the_thought():
    busy, objs, ox = _scene()
    for zoom_in in (True, False):
        p = shots.plan(9.0, busy, [], objects=objs, zoom_in=zoom_in)
        assert {sg["zoom_in"] for sg in p["segments"] if sg["kind"] == "drift"} == {zoom_in}   # §1.3: одно на картинку
    key_box = (900 + ox, 100, 1250 + ox, 200)
    p = shots.plan(8.0, busy, [], objects=objs, key="только открыть", key_dur=2.0, key_box=key_box, zoom_in=False)
    assert p["key_time"] is not None
    assert {sg["zoom_in"] for sg in p["segments"] if sg["kind"] == "drift"} == {True}          # к мысли — наездом


def test_hero_body_rule_counts_ink_not_the_empty_bbox():
    """Рамка героя включает пустую бумагу между телом и хвостом: окно по ней — не обрубок (аудит 08.10:
    по площади рамки терялся крупный план конверта, и камера повторяла планы)."""
    busy, objs, ox = _scene()
    SH, SW = busy.shape
    hero = (150 + ox, 120, 900 + ox, 700)          # рамка героя растянута вправо пустой бумагой (как «хвост»)
    objs2 = [dict(o) for o in objs]
    objs2[1]["box"] = hero
    ok = shots.make_view_ok(busy, objs2, camera.window(754, 424, 1508, SW, SH))
    empty_part = camera.window(750 + ox, 400, 400, SW, SH)       # внутри рамки героя, но чернил героя там нет
    assert ok(empty_part, ink=False)
    torso = camera.window(600 + ox, 650, 700, SW, SH)              # а настоящее тело без головы — по-прежнему нет
    assert not ok(torso, objs2[3]["box"])


def test_splits_without_word_timings_are_not_a_metronome_but_deterministic():
    busy, objs, ox = _scene()
    a = shots.plan(9.0, busy, [], objects=objs, T0=0.0)
    b = shots.plan(9.0, busy, [], objects=objs, T0=0.0)
    c = shots.plan(9.3, busy, [], objects=objs, T0=0.0)
    la = [round(s["t1"] - s["t0"], 3) for s in a["segments"]]
    assert la == [round(s["t1"] - s["t0"], 3) for s in b["segments"]]          # детерминизм
    assert len(la) >= 2
    assert la != [round(s["t1"] - s["t0"], 3) for s in c["segments"]]          # другая длина кадра — другие куски
    d = shots.plan(9.0, busy, [], objects=objs, T0=17.3)
    assert la == [round(s["t1"] - s["t0"], 3) for s in d["segments"]]          # T0 на куски не влияет (кэш клипов)
    assert max(la) - min(la) > 0.05                                               # не ровные куски


def test_live_doll_does_not_take_frames_that_need_fear_or_shame():
    import frame_planner as fp
    f = {"hero": True, "hero_action": "look", "picture": "the main character sits beside the envelope",
         "spec": {"focus": "shame sticks to the task",
                  "claims": [{"id": "c2", "text": "the character looks at it with dread", "tier": "must"}]}}
    g = {"hero": True, "hero_action": "look", "picture": "the main character sits beside the open envelope, calm",
         "spec": {"focus": "only open it", "claims": [{"id": "c1", "text": "the envelope is open", "tier": "must"}]}}
    assert fp.live_emotion_blocked(f) and not fp.live_emotion_blocked(g)
    assert fp.live_hero_pass([f, g]) == 1
    assert f["hero"] is True and not f.get("hero_live") and f.get("hero_live_blocked") == "emotion"
    assert g["hero"] is False and g["hero_live"] is True


def test_hero_limit_keeps_the_finale_of_the_episode_not_of_the_free_list():
    import frame_planner as fp
    frames = [{"hero": True, "picture": "the main character here"} for _ in range(5)]
    fp.limit_hero(frames, max_run=2, max_share=0.35)
    assert frames[-1]["hero"] is True
    # боевой путь: нарисованные кадры заперты, но считаются — финал эпизода (запертый) остаётся,
    # серия через нарисованный кадр видна, снимается только свободный
    frames = [{"hero": True, "picture": "the main character here", "kept_drawn": i in (2, 5)} for i in range(6)]
    n = fp.limit_hero(frames, max_run=2, max_share=0.35, locked=lambda f: f.get("kept_drawn"))
    assert frames[5]["hero"] is True and frames[2]["hero"] is True
    assert all(f["hero"] for f in frames if f.get("kept_drawn")) and n >= 1
    assert sum(f["hero"] for f in frames) <= max(1, int(0.35*6)) + 2    # заперто два, они не снимаются


def test_emotion_gate_ignores_tearing_the_envelope_and_sober_looks():
    import frame_planner as fp
    for text in ("the character tears the envelope open", "a sober look at the letter", "an exhaustive list on the desk"):
        assert not fp.live_emotion_blocked({"picture": text, "spec": {"claims": []}}), text
    for text in ("the character in tears beside the letter", "sobbing", "exhausted, head on the desk", "a nervous glance"):
        assert fp.live_emotion_blocked({"picture": text, "spec": {"claims": []}}), text


def test_pencil_without_voice_is_set_from_master_target(monkeypatch):
    import pencil_sound as ps
    import audio_master as am
    monkeypatch.setattr(ps, "integrated_lufs", lambda path: -20.0)
    g, why = ps.gain_for(-70.0, "pencil.wav")
    assert abs(g - (am.LOUDNORM_TARGET_I - ps.PENCIL_GAP_LU + 20.0)) < 1e-6 and "master_target" in why
    assert not ps.has_voice(-70.0) and ps.has_voice(-45.0) and ps.has_voice(None)
    g2, why2 = ps.gain_for(-16.0, "pencil.wav")
    assert abs(g2 - (-16.0 - ps.PENCIL_GAP_LU + 20.0)) < 1e-6 and "master_target" not in why2


def test_master_normalizes_unless_the_assembler_says_there_is_no_voice():
    import audio_master as am
    stats = {"input_i": "-45.0", "input_tp": "-20", "input_lra": "1", "input_thresh": "-65", "target_offset": "0"}
    assert am.build_master_af(stats, 30.0, 0.05).startswith("loudnorm=")          # тихий настоящий голос — нормализуется
    assert am.build_master_af(stats, 30.0, 0.05, voiceless=True).startswith("anull,")


def test_doll_blinks_are_spaced_and_quiet_before_the_cut():
    import doll_rig
    out = doll_rig.thin_blinks([1.0, 1.08, 1.17, 1.5, 3.0, 3.32, 6.8, 6.9], 7.0)
    assert out == [1.0, 1.5, 3.0, 3.32]
    assert all(b2 - b1 >= doll_rig.BLINK_MIN_GAP for b1, b2 in zip(out, out[1:]))
    assert all(b <= 7.0 - doll_rig.END_QUIET_SEC for b in out)
    # функциональное моргание (взгляд под веком) побеждает случайное фоновое рядом
    assert doll_rig.thin_blinks([1.0, 3.0], 7.0, keep=[1.1]) == [1.1, 3.0]


def test_qc_thresholds_follow_the_planner_constants():
    import montage_qc as mq
    assert mq.HOOK_ZONE_SEC == shots.HOOK_ZONE_SEC
    assert mq.THRESHOLDS["plan_max_sec"][0] == shots.MAX_VIEW_SEC + 1/24


def test_qc_does_not_count_the_handwriting_plan_as_too_long(tmp_path):
    import montage_qc as mq
    plans = [(0.0, 2.0), (2.0, 6.6), (6.6, 9.0)]
    import pytest
    assert mq.length_metrics(plans)["plan_max_sec"] == pytest.approx(4.6)
    m = mq.length_metrics(plans, writing=[(2.3, 6.3)])       # мысль пишется с 2.3, стоит до 6.3, склейка 6.6
    assert m["plan_max_sec"] == pytest.approx(2.4) and m["writing_plans"] == 1
    m2 = mq.length_metrics(plans, writing=[(2.3, 4.0)])      # план тянется 2.6 с после удержания: остаток 4.6-2.4=2.2 < 2.4
    assert m2["plan_max_sec"] == pytest.approx(2.4)
    # письмо — вычитание, не индульгенция: план шёл 5 с ДО письма — это длинный план
    m3 = mq.length_metrics([(0.0, 6.6), (6.6, 9.0)], writing=[(5.0, 6.3)])
    assert m3["plan_max_sec"] == pytest.approx(6.6 - 1.6)
    mp = tmp_path / "media_plan"; mp.mkdir()
    (mp / "shots_report.json").write_text(json.dumps({"clips": [
        {"index": 0, "note": "x"}, {"index": 0, "duration": 2.0, "key_at": None, "key_hold_until": None},
        {"index": 1, "duration": 7.0, "key_at": 0.3, "key_hold_until": 4.3}]}), encoding="utf-8")
    (w0, w1), = mq.writing_windows(str(tmp_path))
    assert (w0, w1) == (pytest.approx(2.3), pytest.approx(6.3))


def test_second_opinion_on_cuts_and_paper_tone(tmp_path):
    """Приёмка: второе мнение о склейках (PySceneDetect) и тон бумаги между планами (ΔE2000)."""
    import subprocess
    import montage_qc as mq
    assert mq.cuts_crosscheck([1.0, 2.0, 3.0], [1.05, 2.0, 4.0]) == dict(cuts_unconfirmed=1, cuts_extra_by_scenedetect=1)
    assert mq.cuts_crosscheck([1.0], None) == dict(cuts_unconfirmed=None, cuts_extra_by_scenedetect=None)
    # две бумаги: кремовая и заметно желтее — склейка между ними должна дать ΔE выше порога, одинаковые — нет
    vid = str(tmp_path / "tone.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y",
                    "-f", "lavfi", "-i", "color=c=0xfaf7ee:s=192x108:d=1.0:r=24",
                    "-f", "lavfi", "-i", "color=c=0xf5e6b0:s=192x108:d=1.0:r=24",
                    "-f", "lavfi", "-i", "color=c=0xf5e6b0:s=192x108:d=1.0:r=24",
                    "-filter_complex", "[0][1][2]concat=n=3:v=1:a=0[v]", "-map", "[v]", "-pix_fmt", "yuv420p", vid], check=True)
    m = mq.paper_tone_jumps(vid, [1.0, 2.0], 24.0)
    assert m["cut_paper_de"] > mq.CUT_PAPER_DE_MAX                      # кремовая -> жёлтая: видно
    same = mq.paper_tone_jumps(vid, [2.0], 24.0)
    assert same["cut_paper_de"] < 0.5                                  # жёлтая -> та же жёлтая: шум сжатия
