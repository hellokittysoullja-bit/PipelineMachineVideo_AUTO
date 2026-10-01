import io
import json

import frame_generator as g
import look as look_mod

FRAME = {"index": 0, "kind": "caption", "hero": False, "text": "Жив. Полностью.",
         "labels": ["ЖИВ. ПОЛНОСТЬЮ."], "picture": "a stick figure glowing in the desert sun",
         "spec": {"focus": "a happy stick figure", "claims": [
             {"id": "core", "text": "a stick figure is visible", "tier": "must"}], "queries": []}}


class FakeBackend:
    model, size, quality = "fake-model", "1536x1024", "low"

    def __init__(self):
        self.calls, self.refs = 0, []

    def generate(self, prompt, refs):
        from PIL import Image
        self.calls += 1
        self.refs.append(refs)
        b = io.BytesIO()
        Image.new("RGB", (640, 360), (250, 250, 245 - self.calls)).save(b, "PNG")
        return b.getvalue(), 0


class FakeJudge:
    """Отвечает на вопросы shot_judge и чтение текста. reads — что «прочитано»
    на вариантах по порядку появления; grid — оценка сетки всем."""

    def __init__(self, reads, grid=3, claim="yes"):
        self.reads, self.grid, self.claim = list(reads), grid, claim

    def chat(self, model, content, max_tokens, est, reasoning=None, **kw):
        text = content[0]["text"]
        if text.startswith("Transcribe"):
            return self.reads.pop(0), {}, 1
        if "grid of" in text:
            n = int(text.split("grid of ")[1].split()[0])
            return json.dumps({"scores": {str(i): self.grid for i in range(1, n + 1)}}), {}, 1
        if "For each statement" in text:
            return json.dumps({"claims": {"core": self.claim}, "medium": "artwork", "why": "x"}), {}, 1
        if "Rank them" in text:
            return json.dumps({"order": [1, 2, 3]}), {}, 1
        raise AssertionError(text[:80])


def make_look(tmp_path, hero=True, color=(200, 200, 200)):
    from PIL import Image
    d = tmp_path / "look"
    (d / "style").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 36), color).save(d / "style" / "a.png")
    Image.new("RGB", (64, 36), (10, 10, 10)).save(d / "style" / "b.jpg")
    if hero:
        Image.new("RGB", (40, 40), (255, 150, 200)).save(d / "hero.png")
    return look_mod.load(str(d))


def gen(tmp_path, judge, variants=2, rounds=2, look=None):
    return g.Generator(FakeBackend(), str(tmp_path), look or make_look(tmp_path), judge_gw=judge,
                       judge_model="q", variants=variants, rounds=rounds)


def test_exact_letters_required():
    assert g.text_score(["ЖИВ. ПОЛНОСТЬЮ."], "ЖИВ. ПОЛНОСТЮ.")[0] < g.TEXT_MATCH_MIN
    assert g.text_score(["ЖИВ. ПОЛНОСТЬЮ."], "жив полностью")[0] >= g.TEXT_MATCH_MIN
    assert g.text_score(["ЕДА, БЕЗОПАСНОСТЬ"], "ЕДА,\nБЕЗОПАСНОСТЬ")[0] == 1.0
    assert g.text_score(["ЕЩЁ"], "ЕЩЕ")[0] == 1.0
    assert g.text_score([], "NONE")[0] == 1.0 and g.text_score([], "СЛОВО")[0] == 0.0


def test_prompt_never_asks_the_model_for_letters_and_points_at_references():
    p = g.build_prompt(FRAME, 2, False)
    assert p.startswith(FRAME["picture"])
    assert "ЖИВ" not in p and "No text" in p and "bottom fifth" in p
    assert "style of reference images 1-2" in p and "main character" not in p
    h = g.build_prompt(dict(FRAME, hero=True), 2, True)
    assert "person in reference image 3" in h          # герой — последним, после двух образцов
    d = g.build_prompt(dict(FRAME, kind="diagram", labels=["А", "Б"]), 1, False)
    assert "2 empty patches" in d and "arrow" in d and "reference image 1:" in d and "А" not in d


def test_style_refs_always_hero_only_where_planned(tmp_path):
    look = make_look(tmp_path)
    g1 = gen(tmp_path, None, variants=1, rounds=1, look=look)
    g1.frame(FRAME)
    g1.frame(dict(FRAME, index=1, hero=True, picture="the main character waves"))
    plain, with_hero = g1.backend.refs
    assert len(plain) == 2 and len(with_hero) == 3
    assert all(r.startswith("data:image/jpeg;base64,") for r in with_hero)


def test_hero_frame_without_hero_file_is_drawn_without_hero(tmp_path):
    g1 = gen(tmp_path, None, variants=1, rounds=1, look=make_look(tmp_path, hero=False))
    rec = g1.frame(dict(FRAME, hero=True))
    assert len(g1.backend.refs[0]) == 2 and rec["hero"] is False


def test_changing_the_look_redraws(tmp_path):
    gen(tmp_path, None, variants=1, rounds=1).frame(FRAME)
    g2 = gen(tmp_path, None, variants=1, rounds=1, look=make_look(tmp_path, color=(50, 90, 200)))
    g2.frame(FRAME)
    assert g2.backend.calls == 1                       # новый образец стиля — кэш не подходит


def test_look_rules(tmp_path):
    import pytest
    from PIL import Image
    d = tmp_path / "lk"
    (d / "style").mkdir(parents=True)
    with pytest.raises(look_mod.LookError):
        look_mod.load(str(d))                          # без образцов стиля не рисуем
    for i in range(4):
        Image.new("RGB", (8, 8)).save(d / "style" / f"{i}.png")
    with pytest.raises(look_mod.LookError):
        look_mod.load(str(d))                          # больше трёх образцов
    (d / "style" / "3.png").unlink()
    Image.new("RGB", (8, 8)).save(d / "hero.png")
    Image.new("RGB", (8, 8)).save(d / "hero.jpg")
    with pytest.raises(look_mod.LookError):
        look_mod.load(str(d))                          # два героя
    (d / "hero.jpg").unlink()
    lk = look_mod.load(str(d))
    assert len(lk.style) == 3 and lk.hero.endswith("hero.png")


def test_gateway_sends_reference_images():
    import llm_gateway
    gw = llm_gateway.Gateway(api_key="k")
    gw._prices = {"m": {"unit": "image", "base_tokens": 100, "coefficient": {"input": 1, "output": 1}}}
    sent = {}

    def fake_request(method, path, body=None, **kw):
        sent.update(body)
        import base64
        return {"data": [{"b64_json": base64.b64encode(b"img").decode()}]}
    gw._request = fake_request
    imgs, price = gw.image("m", "p", "1024x1024", images=["data:image/jpeg;base64,AA"])
    assert sent["images"] == ["data:image/jpeg;base64,AA"] and imgs == [b"img"] and price == 100
    sent.clear()
    gw.image("m", "p", "1024x1024")
    assert "images" not in sent                        # без референсов поле не шлётся


def test_variant_with_model_letters_loses_and_caption_is_drawn_by_code(tmp_path):
    # первый вариант с псевдонадписью модели — брак; второй чистый, подпись кладёт код
    rec = gen(tmp_path, FakeJudge(["ЖИВ ПОЛНОСТЮ", "NONE"])).frame(FRAME)
    assert rec["status"] == "ok"
    clean = [k for k, v in rec["candidates"].items() if v["text_ok"]]
    assert rec["chosen"] in clean and (tmp_path / "frames" / "001.png").exists()
    assert rec["labels_placed"][0]["text"] == "ЖИВ. ПОЛНОСТЬЮ." and rec["labels_placed"][0]["font"].startswith("Shantell")


def test_all_rounds_with_model_letters_is_rejected_not_shown(tmp_path):
    rec = gen(tmp_path, FakeJudge(["ЖИФ"] * 4)).frame(FRAME)
    assert rec["status"] == "rejected" and len(rec["candidates"]) == 4


def test_grid_zero_is_rejected(tmp_path):
    rec = gen(tmp_path, FakeJudge(["NONE"] * 4, grid=0)).frame(FRAME)
    assert rec["status"] == "rejected"


def test_second_round_only_when_first_failed(tmp_path):
    g1 = gen(tmp_path, FakeJudge(["NONE"] * 2))
    g1.frame(FRAME)
    assert g1.backend.calls == 2


def test_no_judge_takes_variant_unchecked(tmp_path):
    rec = gen(tmp_path, None).frame(FRAME)
    assert rec["status"] == "unchecked" and rec["path"]


def test_rerun_reuses_cache_without_drawing(tmp_path):
    gen(tmp_path, FakeJudge(["NONE"] * 2)).frame(FRAME)
    g2 = gen(tmp_path, FakeJudge([]))      # чтение и судья из кэша
    rec = g2.frame(FRAME)
    assert rec["status"] == "ok" and g2.backend.calls == 0
