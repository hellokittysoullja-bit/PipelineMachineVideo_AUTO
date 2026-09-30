import channel
import frame_generator as g


def test_exact_letters_required():
    # одна пропущенная буква на крупной подписи — брак, а не «почти совпало»
    assert g.text_score(["ЖИВ. ПОЛНОСТЬЮ."], "ЖИВ. ПОЛНОСТЮ.")[0] < g.TEXT_MATCH_MIN
    assert g.text_score(["ЖИВ. ПОЛНОСТЬЮ."], "жив полностью")[0] >= g.TEXT_MATCH_MIN


def test_label_split_across_lines_is_found():
    assert g.text_score(["ЕДА, БЕЗОПАСНОСТЬ"], "ЕДА,\nБЕЗОПАСНОСТЬ")[0] == 1.0


def test_yo_equals_e():
    assert g.text_score(["ЕЩЁ"], "ЕЩЕ")[0] == 1.0


def test_scene_must_have_no_text():
    assert g.text_score([], "NONE")[0] == 1.0
    assert g.text_score([], "СЛОВО")[0] == 0.0


def test_prompt_contains_labels_verbatim_and_mascot():
    prof = channel.load_profile()
    f = {"kind": "diagram", "backdrop": "white", "mascot": True, "labels": ["РЕДКОЕ."], "picture": "a pyramid"}
    p = g.build_prompt(f, prof)
    assert '"РЕДКОЕ."' in p and prof["mascot"]["description"] in p
    s = g.build_prompt(dict(f, kind="scene", labels=[], mascot=False), prof)
    assert "No text" in s and prof["mascot"]["description"] not in s


class FakeGW:
    def __init__(self, answers):
        self.answers, self.images = list(answers), 0

    def image(self, model, prompt, size, quality=None, n=1):
        self.images += 1
        from io import BytesIO
        from PIL import Image
        b = BytesIO()
        Image.new("RGB", (64, 40), (self.images * 40, 0, 0)).save(b, "PNG")
        return [b.getvalue()], 5

    def chat(self, model, content, max_tokens, est, reasoning=None):
        return self.answers.pop(0), {}, 1


def test_retry_until_text_is_right_and_cache(tmp_path):
    gw = FakeGW(["ЖИВ ПОЛНОСТЮ", "ЖИВ. ПОЛНОСТЬЮ."])
    gen = g.Generator(gw, str(tmp_path), channel.load_profile(), "img", "chk", attempts=3)
    fr = {"index": 0, "kind": "caption", "backdrop": "paper", "mascot": False,
          "labels": ["ЖИВ. ПОЛНОСТЬЮ."], "picture": "a stick figure in the sun"}
    rec = gen.frame(fr)
    assert rec["status"] == "ok" and gw.images == 2 and len(rec["tries"]) == 2
    gw2 = FakeGW([])     # повтор: картинки и чтение из кэша, ни одного вызова
    rec2 = g.Generator(gw2, str(tmp_path), channel.load_profile(), "img", "chk", attempts=3).frame(fr)
    assert rec2["status"] == "ok" and gw2.images == 0


def test_all_attempts_wrong_marks_mismatch(tmp_path):
    gw = FakeGW(["ЖИФ", "ЖИФ"])
    gen = g.Generator(gw, str(tmp_path), channel.load_profile(), "img", "chk", attempts=2)
    rec = gen.frame({"index": 4, "kind": "caption", "backdrop": "paper", "mascot": False,
                     "labels": ["ЖИВ"], "picture": "a stick figure in the sun"})
    assert rec["status"] == "text_mismatch" and (tmp_path / "frames" / "005.png").exists()
