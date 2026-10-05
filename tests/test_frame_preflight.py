import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import frame_preflight as fp  # noqa: E402

HERO = "a black cartoon cat"


def frame(**kw):
    f = {"index": 0, "section": "HOOK", "text": "Ты открываешь письмо.", "kind": "scene", "labels": [],
         "hero": True,
         "picture": "medium shot: the main character stands on the floor beside a desk, both paws on one closed "
                    "envelope that lies on the desk, eyes wide and tail raised; plain light background around them"}
    f.update(kw)
    return f


def test_clean_frame_has_no_issues():
    assert fp.issues(frame(), HERO) == []


def test_issues_catch_the_real_defects_of_04_10():
    # кот без референса: герой снят правилом кода, а зверь остался словами
    f = frame(hero=False, picture=frame()["picture"].replace("the main character", "a black cartoon cat"))
    assert "character_without_reference" in fp.issues(f, HERO)
    assert "duplicate_article" in fp.issues(frame(picture="The the envelope " + frame()["picture"]), HERO)
    z = frame(zoom={"object": "the timer", "word": "таймер"})
    assert "zoom_object_not_in_picture" in fp.issues(z, HERO)
    two_big = frame(picture=frame()["picture"] + ", a huge clock and a giant lamp")
    assert "several_size_demands" in fp.issues(two_big, HERO)
    lab = frame(picture=frame()["picture"] + ", blobs labeled as shame")
    assert any(i.startswith("writing_in_picture") for i in fp.issues(lab, HERO))


def test_accept_rejects_a_rewrite_that_loses_kept_objects_or_the_hero():
    f = frame(zoom={"object": "envelope", "word": "письмо"})
    good = ("close-up: the main character sits on the floor beside one big envelope, the flap already folded open "
            "and the letter half out, a small smile; a clear gap of plain light background between them")
    assert fp.accept(f, good, HERO)[0] == good
    no_zoom = good.replace("envelope", "box")
    assert fp.accept(f, no_zoom, HERO)[1].startswith("still:")
    no_hero = good.replace("the main character", "a small dog")
    assert fp.accept(f, no_hero, HERO)[1] == "hero_lost"
    assert fp.accept(f, "too short", HERO)[1] is not None
    # живой случай 04.10: «the boulder is the only large object» — не второе требование размера
    two = frame(picture=frame()["picture"] + ", a large brain and a massive boulder")
    fix = ("medium shot: the main character stands on the floor holding its head; a massive dark boulder sits beside "
           "it, chained to its ankle; the boulder is the only large object, plain light background around")
    assert fp.accept(two, fix, HERO)[0] == fix
    assert fp.accept(frame(), fix + ", a huge lamp", HERO)[1] == "still:more_size_demands"


class FakeGateway:
    def __init__(self, answer):
        self.answer, self.calls = answer, 0

    def chat(self, model, content, max_tokens, est, **kw):
        self.calls += 1
        self.prompt = content[0]["text"]
        return self.answer, {}, 0


def test_check_chapter_rewrites_only_bad_frames_and_keeps_the_original(tmp_path):
    bad = frame(index=1, picture="medium shot: the main character walks around one envelope lying on a desk, "
                                 "lifting its flap, plain light background around them and the desk in the middle")
    ok = frame(index=0)
    new = ("medium shot: the main character stands on the floor beside a desk with one envelope on it, the flap "
           "folded open and the letter half out, tail raised; plain light background around them")
    gw = FakeGateway(json.dumps({"n": 1, "ok": True}) + "\n" +
                     json.dumps({"n": 2, "ok": False, "faults": [1, 2], "picture": new}))
    s = fp.check_chapter([ok, bad], gw, "m", str(tmp_path), HERO)
    assert s["rewritten"] == 1 and ok["preflight"]["rewritten"] is False
    assert bad["picture"] == new and bad["preflight"]["original"].startswith("medium shot: the main character walks")
    assert "Main character: NO CHARACTER" not in gw.prompt and "the cat" in gw.prompt
    # кэш: второй прогон не платит
    gw2 = FakeGateway("")
    assert fp.check_chapter([frame()], gw2, "m", str(tmp_path), HERO)["cached"] is False   # другой вопрос — не из кэша
    s3 = fp.check_chapter([frame(index=0), frame(index=1, picture=bad["preflight"]["original"])],
                          FakeGateway("должно не понадобиться"), "m", str(tmp_path), HERO)
    assert s3["cached"] is True


def test_a_bad_rewrite_is_refused_and_a_dead_gateway_leaves_the_plan_alone(tmp_path):
    f = frame()
    before = f["picture"]
    gw = FakeGateway(json.dumps({"n": 1, "ok": False, "faults": [4], "picture": "a cat"}))
    s = fp.check_chapter([f], gw, "m", str(tmp_path), HERO)
    assert s["rejected"] == 1 and f["picture"] == before and f["preflight"]["rejected"]

    class Dead:
        def chat(self, *a, **k):
            raise RuntimeError("503")
    g = frame(picture="The the envelope " + before)
    s = fp.check_chapter([g], Dead(), "m", str(tmp_path / "x"), HERO)
    assert s["error"] and g["picture"].startswith("The envelope")       # бесплатная правка — всё равно


def test_fallback_frames_are_not_sent():
    f = frame(fallback=True)
    gw = FakeGateway("")
    assert fp.check_chapter([f], gw, "m", "/nonexistent", HERO)["checked"] == 0 and gw.calls == 0


def test_hero_ears_are_not_described():
    f = frame(picture=frame()["picture"].replace("eyes wide", "ears back, eyes wide"))
    assert "hero_ears_described" in fp.issues(f, HERO)
    assert "hero_ears_described" not in fp.issues(dict(f, hero=False, picture="a rabbit with long ears sits on "
                                                       "the grass beside a fence, plain light background around, "
                                                       "one carrot lying at its feet on the grass"), HERO)
