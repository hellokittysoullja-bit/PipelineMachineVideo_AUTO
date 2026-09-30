import io
import json

import channel
import frame_generator as g

FRAME = {"index": 0, "kind": "caption", "backdrop": "paper", "mascot": False, "text": "Жив. Полностью.",
         "labels": ["ЖИВ. ПОЛНОСТЬЮ."], "picture": "a stick figure glowing in the desert sun",
         "spec": {"focus": "a happy stick figure", "claims": [
             {"id": "core", "text": "a stick figure is visible", "tier": "must"}], "queries": []}}


class FakeBackend:
    name, model = "fake", "fake-model"

    def __init__(self):
        self.calls = 0

    def generate(self, prompt, seed):
        from PIL import Image
        self.calls += 1
        b = io.BytesIO()
        Image.new("RGB", (64, 40), ((self.calls * 37) % 255, 0, 0)).save(b, "PNG")
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


def gen(tmp_path, judge, variants=2, rounds=2):
    return g.Generator(FakeBackend(), str(tmp_path), channel.load_profile(), judge_gw=judge,
                       judge_model="q", variants=variants, rounds=rounds)


def test_exact_letters_required():
    assert g.text_score(["ЖИВ. ПОЛНОСТЬЮ."], "ЖИВ. ПОЛНОСТЮ.")[0] < g.TEXT_MATCH_MIN
    assert g.text_score(["ЖИВ. ПОЛНОСТЬЮ."], "жив полностью")[0] >= g.TEXT_MATCH_MIN
    assert g.text_score(["ЕДА, БЕЗОПАСНОСТЬ"], "ЕДА,\nБЕЗОПАСНОСТЬ")[0] == 1.0
    assert g.text_score(["ЕЩЁ"], "ЕЩЕ")[0] == 1.0
    assert g.text_score([], "NONE")[0] == 1.0 and g.text_score([], "СЛОВО")[0] == 0.0


def test_prompt_puts_subject_first_and_style_last():
    prof = channel.load_profile()
    p = g.build_prompt(dict(FRAME, mascot=True), prof)
    assert p.startswith(FRAME["picture"]) and p.rstrip(".").endswith(prof["style"]["base"].rstrip("."))
    assert '"ЖИВ. ПОЛНОСТЬЮ."' in p and prof["mascot"]["description"] in p
    assert "No text" in g.build_prompt(dict(FRAME, kind="scene", labels=[]), prof)


def test_variant_with_right_letters_wins(tmp_path):
    rec = gen(tmp_path, FakeJudge(["ЖИВ ПОЛНОСТЮ", "ЖИВ. ПОЛНОСТЬЮ."])).frame(FRAME)
    assert rec["status"] == "ok"
    good = [k for k, v in rec["candidates"].items() if v["text_ok"]]
    assert rec["chosen"] in good and (tmp_path / "frames" / "001.png").exists()


def test_all_rounds_wrong_letters_is_rejected_not_shown(tmp_path):
    rec = gen(tmp_path, FakeJudge(["ЖИФ"] * 4)).frame(FRAME)
    assert rec["status"] == "rejected" and len(rec["candidates"]) == 4


def test_grid_zero_is_rejected(tmp_path):
    rec = gen(tmp_path, FakeJudge(["ЖИВ. ПОЛНОСТЬЮ."] * 4, grid=0)).frame(FRAME)
    assert rec["status"] == "rejected"


def test_second_round_only_when_first_failed(tmp_path):
    g1 = gen(tmp_path, FakeJudge(["ЖИВ. ПОЛНОСТЬЮ."] * 2))
    g1.frame(FRAME)
    assert g1.backend.calls == 2


def test_no_judge_takes_variant_unchecked(tmp_path):
    rec = gen(tmp_path, None).frame(FRAME)
    assert rec["status"] == "unchecked" and rec["path"]


def test_rerun_reuses_cache_without_drawing(tmp_path):
    gen(tmp_path, FakeJudge(["ЖИВ. ПОЛНОСТЬЮ."] * 2)).frame(FRAME)
    g2 = gen(tmp_path, FakeJudge([]))      # чтение и судья из кэша
    rec = g2.frame(FRAME)
    assert rec["status"] == "ok" and g2.backend.calls == 0


def test_comfy_workflow_placeholders(tmp_path):
    wf = tmp_path / "wf.json"
    wf.write_text(json.dumps({"3": {"inputs": {"seed": "{{SEED}}", "text": "{{PROMPT}}",
                                               "width": "{{WIDTH}}", "height": "{{HEIGHT}}"}}}), encoding="utf-8")
    b = g.ComfyBackend("http://x", str(wf), "1536x1024")
    filled = b._fill('say "ПРИВЕТ"', 42)
    assert filled["3"]["inputs"] == {"seed": 42, "text": 'say "ПРИВЕТ"', "width": 1536, "height": 1024}
