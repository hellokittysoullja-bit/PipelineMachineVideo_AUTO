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
        if text.startswith("You inspect ONE hand-drawn"):
            return json.dumps(self.defects_answer(content)), {}, 1
        raise AssertionError(text[:80])

    def defects_answer(self, content):
        return {"defects": [], "severe": False, "hero_present": True, "off_model": [],
                "wrong_character": False, "text": False, "why": "clean"}


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
    # роли референсов первыми, сцена после них; запрета словами нет — утверждение о чистых поверхностях
    assert p.startswith("Reference images 1-2 show only the drawing style") and "SCENE: " + FRAME["picture"] in p
    assert "ЖИВ" not in p and "No text" not in p and "clean and unmarked" in p and "bottom fifth" in p
    assert "main character" not in p
    h = g.build_prompt(dict(FRAME, hero=True), 2, True)
    assert "Reference image 3 is the main character" in h and "person" not in h   # герой — последним, после двух образцов
    assert h.index("Reference image 3") < h.index("SCENE:")
    d = g.build_prompt(dict(FRAME, kind="diagram", labels=["А", "Б"]), 1, False)
    assert "2 wide empty patches" in d and "arrow" in d and d.startswith("Reference image 1 shows") and "А" not in d


def test_prompt_carries_hero_state_zoom_key_and_hand_drawn_diagram():
    states = {"ember": {"when": "stuck", "draw": "the tail flame is a dim ember with smoke"}}
    h = g.build_prompt(dict(FRAME, hero=True, hero_state="ember"), 2, True, states)
    assert "dim ember with smoke" in h
    assert "dim ember" not in g.build_prompt(dict(FRAME, hero_state="ember"), 2, False, states)   # без героя — нет
    z = g.build_prompt(dict(FRAME, zoom={"object": "envelope", "word": "письмо"}, key_thought="только открыть"), 2, False)
    assert "envelope is the biggest single object" in z and "handwriting" in z and "только" not in z
    d = g.build_prompt(dict(FRAME, kind="diagram", labels=["А"]), 1, False)
    assert "watercolor" in d and "no clean vector graphics" in d


def test_one_size_demand_and_hero_beside_the_zoom_object():
    zoom = {"object": "envelope", "word": "письмо"}
    both = g.build_prompt(dict(FRAME, hero=True, zoom=zoom), 2, True)
    assert "biggest single object" in both and "third of the image height" not in both   # одно требование размера
    assert "side by side with a clear gap" in both
    hero = g.build_prompt(dict(FRAME, hero=True), 2, True)
    assert "third of the image height" in hero and "side by side" not in hero
    assert "side by side" not in g.build_prompt(dict(FRAME, zoom=zoom), 2, False)


def test_duplicate_articles_are_removed():
    p = g.build_prompt(dict(FRAME, picture="The the envelope lies on a a desk"), 2, False)
    assert "SCENE: The envelope lies on a desk" in p


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


def test_digits_on_the_raw_frame_are_a_defect():
    assert g.text_score([], "777")[0] == 0.0


def test_judge_failure_is_unchecked_not_rejected(tmp_path):
    class Silent(FakeJudge):
        def chat(self, model, content, max_tokens, est, reasoning=None, **kw):
            if "For each statement" in content[0]["text"]:
                return "не JSON", {}, 1                     # проверка не состоялась
            return super().chat(model, content, max_tokens, est, reasoning=reasoning, **kw)
    rec = gen(tmp_path, Silent(["NONE"]), variants=1, rounds=2).frame(FRAME)
    assert rec["status"] == "unchecked" and rec["path"]


class BusyBackend(FakeBackend):
    """Рисунок, занятый до самого низа: под подпись места нет."""

    def generate(self, prompt, refs):
        from PIL import Image, ImageDraw
        self.calls += 1
        im = Image.new("RGB", (640, 360), (245, 235, 215))
        d = ImageDraw.Draw(im)
        for x in range(0, 640, 12):
            d.line((x, 0, x, 360), fill=(20, 20, 20), width=3)
        b = io.BytesIO()
        im.save(b, "PNG")
        return b.getvalue(), 0


def test_no_room_for_caption_redraws_then_uses_a_band_not_rejects(tmp_path):
    class NoBox(FakeJudge):
        def chat(self, model, content, max_tokens, est, reasoning=None, **kw):
            if "empty spaces left" in content[0]["text"]:
                return "{}", {}, 1                          # модель места не нашла
            return super().chat(model, content, max_tokens, est, reasoning=reasoning, **kw)
    g1 = g.Generator(BusyBackend(), str(tmp_path), make_look(tmp_path), judge_gw=NoBox(["NONE"] * 2),
                     judge_model="q", variants=1, rounds=2)
    rec = g1.frame(FRAME)
    assert g1.backend.calls == 2                            # второй раунд нарисован
    assert rec["status"] == "ok" and rec["labels_fallback"] == "band"
    assert rec["labels_placed"][0]["text"] == "ЖИВ. ПОЛНОСТЬЮ."


def test_diagram_prompt_asks_wide_clean_patches_and_the_owner_arrow_colour(monkeypatch):
    import frame_generator as fg
    frame = {"kind": "diagram", "labels": ["А", "Б"], "picture": "a loop"}
    monkeypatch.setenv("DIAGRAM_ARROW_COLOR", "red")
    p = fg.build_prompt(frame, 3, False)
    assert "exactly 2 wide empty patches" in p and "away from the image edges" in p
    assert "no lighter fill" in p and "Draw every arrow in red" in p
    monkeypatch.setenv("DIAGRAM_ARROW_COLOR", "")
    assert "Draw every arrow" not in fg.build_prompt(frame, 3, False)   # пусто — цвет из образцов
    assert "Draw every arrow" not in fg.build_prompt({"kind": "scene", "labels": [], "picture": "x"}, 3, False)


def test_diagram_and_caption_get_the_owner_background_scenes_keep_their_place(monkeypatch):
    import frame_generator as fg
    monkeypatch.setenv("DIAGRAM_BACKGROUND", "near-white")
    for kind in ("diagram", "caption"):
        assert "plain near-white paper" in fg.build_prompt({"kind": kind, "labels": ["А"], "picture": "x"}, 3, False)
    scene = fg.build_prompt({"kind": "scene", "labels": [], "picture": "x"}, 3, False)
    assert "If the picture has no specific place" in scene and "a specific place is drawn as that place" in scene
    monkeypatch.setenv("DIAGRAM_BACKGROUND", "")
    assert "plain near-white" not in fg.build_prompt({"kind": "diagram", "labels": ["А"], "picture": "x"}, 3, False)
    assert "background is plain" not in fg.build_prompt({"kind": "diagram", "labels": ["А"], "picture": "x"}, 3, False)


def test_flat_paper_follows_the_owner_background(monkeypatch, tmp_path):
    import frame_generator as fg

    class L:
        def signature(self, h):
            return "s"
    monkeypatch.setenv("DIAGRAM_BACKGROUND", "near-white")
    assert fg.Generator(None, str(tmp_path), L()).flat_paper is True
    monkeypatch.setenv("DIAGRAM_BACKGROUND", "")
    assert fg.Generator(None, str(tmp_path), L()).flat_paper is False


def test_hero_marks_go_into_every_hero_prompt_only(tmp_path):
    marks = "Its ear on the viewer's left is folded down"
    h = g.build_prompt(dict(FRAME, hero=True), 2, True, None, marks)
    assert marks in h and h.index(marks) < h.index("SCENE:")
    assert marks not in g.build_prompt(FRAME, 2, False, None, marks)
    import look
    from PIL import Image
    d = tmp_path / "look"
    (d / "style").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(d / "style" / "a.png")
    Image.new("RGB", (8, 8)).save(d / "hero.png")
    (d / "hero_marks.txt").write_text(marks + "\n", encoding="utf-8")
    assert look.load(str(d)).hero_marks == marks


def test_brand_palette_goes_into_scene_prompts_not_diagrams():
    pal = "Colours: black ink and soft grey washes on cream paper"
    assert pal in g.build_prompt(dict(FRAME, kind="scene", labels=[]), 2, False, palette=pal)
    assert pal not in g.build_prompt(dict(FRAME, kind="diagram", labels=["А"]), 2, False, palette=pal)


def test_palette_without_hero_drops_the_character_sentence():
    import frame_generator as fg
    pal = ("Colours: black ink and soft grey washes on cream paper. The only warm colours in the picture are the "
           "main character's orange tail flame and cheek marks; every other object is ink and grey.")
    out = fg.palette_without_hero(pal)
    assert "main character" not in out and "flame" not in out and "ink" in out
    p = fg.build_prompt({"kind": "scene", "picture": "one envelope on the floor", "hero_live": True}, 2, False,
                        palette=pal)
    assert "main character's orange tail flame" not in p and "no character, creature" in p
    p2 = fg.build_prompt({"kind": "scene", "picture": "the main character sits", "hero": True}, 2, True, palette=pal)
    assert "main character's orange tail flame" in p2          # с героем палитра прежняя


def test_drawn_frame_survives_prompt_wording_change_but_not_plan_change(tmp_path):
    """Отпечаток плана: правка обёртки задания в коде не перерисовывает оплаченный кадр, правка плана
    (описание, герой) — перерисовывает; кадр старого образца без отпечатка годен по ключу фразы."""
    import frame_generator as fg

    class B:
        model, size, quality = "m", "1536x1024", "low"

    class L:
        hero = None; style = ["a", "b"]
        def signature(self, h): return "look"
    g = fg.Generator.__new__(fg.Generator); g.backend = B(); g.look = L()
    f = {"index": 0, "key": "k1", "kind": "scene", "labels": [], "hero": False, "picture": "one envelope on the floor"}
    s1 = g.plan_sig(f)
    assert s1 == g.plan_sig(dict(f, details=["x"], accent="пять минут"))        # детали и акцент — не рисунок
    assert s1 != g.plan_sig(dict(f, picture="one open envelope"))
    assert s1 != g.plan_sig(dict(f, key="k2"))
    done = {0: ("k1", s1), 1: ("k2", None)}
    assert fg.frame_is_done(done, f, s1) and not fg.frame_is_done(done, f, "other")
    assert fg.frame_is_done(done, {"index": 1, "key": "k2"}, "whatever")          # старый образец — по ключу
    assert not fg.frame_is_done(done, {"index": 1, "key": "k3"}, "whatever")
    mp = tmp_path / "media_plan"; mp.mkdir()
    json.dump({"frames": [{"index": 1, "key": "k2", "status": "ok", "path": "frames/002.png"}]},
              open(mp / "frames_report.json", "w"))
    assert fg.stamp_plan_sigs(str(tmp_path), {1: "ps"}) == 1
    assert json.load(open(mp / "frames_report.json"))["frames"][0]["plan_sig"] == "ps"


class ClaimsJudge(FakeJudge):
    """Судья с заданными ответами по утверждениям."""

    def __init__(self, reads, claims, grid=3):
        super().__init__(reads, grid=grid)
        self.claims_answer = claims

    def chat(self, model, content, max_tokens, est, reasoning=None, **kw):
        if "For each statement" in content[0]["text"]:
            return json.dumps({"claims": dict(self.claims_answer), "medium": "artwork", "why": "x"}), {}, 1
        return super().chat(model, content, max_tokens, est, reasoning=reasoning, **kw)


def test_must_answered_no_is_weak_not_ok(tmp_path):
    frame = dict(FRAME, spec={"focus": "an open envelope", "claims": [
        {"id": "core", "text": "an envelope is visible", "tier": "must"},
        {"id": "c1", "text": "the envelope is open", "tier": "must"}], "queries": []})
    rec = gen(tmp_path, ClaimsJudge(["NONE"] * 4, {"core": "yes", "c1": "no"}), variants=1, rounds=1).frame(frame)
    assert rec["status"] == "weak" and rec["must_failed"] == ["c1"]


def test_weak_never_passes_a_figure_on_a_doll_frame(tmp_path):
    frame = dict(FRAME, hero_live=True, spec={"focus": "an envelope", "claims": [
        {"id": "core", "text": "an envelope is visible", "tier": "must"},
        {"id": "c1", "text": "the envelope is open", "tier": "must"},
        {"id": "nofig", "text": "no figure anywhere", "tier": "must"}], "queries": []})
    rec = gen(tmp_path, ClaimsJudge(["NONE"] * 4, {"core": "yes", "c1": "no", "nofig": "no"}), variants=1, rounds=1).frame(frame)
    assert rec["status"] == "rejected"


class DefectJudge(FakeJudge):
    """Судья дефектов: ответ по номеру варианта (порядок вопросов о дефектах)."""

    def __init__(self, reads, answers, grid=3):
        super().__init__(reads, grid=grid)
        self.answers = list(answers)

    def defects_answer(self, content):
        base = {"defects": [], "severe": False, "hero_present": True, "off_model": [],
                "wrong_character": False, "text": False, "why": "x"}
        return dict(base, **self.answers.pop(0))


def test_defect_verdict_policy():
    import shot_judge as sj
    clean = {"defects": [], "severe": False, "hero_present": True, "off_model": [], "wrong_character": False,
             "text": False, "why": ""}
    assert sj.defect_verdict(None) is None
    assert sj.defect_verdict(clean) == "clean"
    assert sj.defect_verdict(dict(clean, defects=["stray stick under the arm"])) == "minor"
    assert sj.defect_verdict(dict(clean, defects=["three front paws"], severe=True)) == "minor"   # severe — не отказ
    assert sj.defect_verdict(dict(clean, wrong_character=True)) == "reject"
    # герой чуть не на модели — замечание только на кадре, где герой по плану есть
    assert sj.defect_verdict(dict(clean, off_model=["both ears up"]), expect_hero=True) == "minor"
    assert sj.defect_verdict(dict(clean, off_model=["both ears up"]), expect_hero=False) == "clean"
    assert sj.parse_defects_answer("no json here") is None
    assert sj.parse_defects_answer('{"severe": "yes"}') is None


def test_wrong_character_rejects_variant_and_minor_ranks_below_clean(tmp_path):
    # вариант 0 — чужой персонаж (reject), 1 — с палочкой (minor), 2 — чистый: на экран идёт чистый
    judge = DefectJudge(["NONE"] * 3, [{"wrong_character": True, "off_model": ["a grey ghost"]},
                                        {"defects": ["a stray stick near the paw"]}, {}])
    rec = gen(tmp_path, judge, variants=3, rounds=1).frame(dict(FRAME, hero=True))
    assert rec["status"] == "ok"
    c = rec["candidates"]
    verdicts = {k: v["defect_verdict"] for k, v in c.items()}
    assert sorted(verdicts.values()) == ["clean", "minor", "reject"]
    assert verdicts[rec["chosen"]] == "clean"


def test_wrong_character_alone_rejects_the_only_variant(tmp_path):
    rec = gen(tmp_path, DefectJudge(["NONE"], [{"wrong_character": True, "off_model": ["a grey ghost instead of the cat"]}]),
              variants=1, rounds=1).frame(dict(FRAME, hero=True))
    assert rec["status"] == "rejected"


def test_defect_check_failure_is_not_a_rejection(tmp_path):
    class Broken(FakeJudge):
        def defects_answer(self, content):
            raise RuntimeError("gateway down")
    rec = gen(tmp_path, Broken(["NONE"]), variants=1, rounds=1).frame(FRAME)
    assert rec["status"] == "ok" and rec["candidates"][rec["chosen"]]["defect_verdict"] is None


def test_defect_question_sends_hero_reference_only_for_hero_frames(tmp_path):
    seen = []

    class Spy(FakeJudge):
        def defects_answer(self, content):
            seen.append(sum(1 for c in content if c.get("type") == "image_url"))
            return super().defects_answer(content)
    g_ = gen(tmp_path, Spy(["NONE", "NONE"]), variants=1, rounds=1)
    g_.frame(dict(FRAME, hero=True))
    g_.frame(dict(FRAME, index=FRAME["index"] + 1))
    assert seen == [2, 1]
