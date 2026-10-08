"""Кадр вне хука не длиннее BODY_SLOT_MAX_SEC (решение владельца 08.10).

Живой случай эп.05: после хука шли кадры по 9-11 с. Длинный блок резался
ОДИН раз, и половинки длиннее 8 с больше не проверялись; кусок «3.2 с по
словам», звучавший 2.9 с, потом склеивался с соседом уже без потолка (10.6 с).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import pipeline_smart as ps  # noqa: E402


def _block(section, n_words, sentence_every=6):
    words = []
    for i in range(n_words):
        w = f"слово{i}"
        if (i + 1) % sentence_every == 0:
            w += "."
        words.append(w)
    text = " ".join(words)
    return {"section": section, "text": text, "words": n_words, "pause_after": 0.0,
            "stat": None, "stat_word_pos": None}


def _durs(pieces, total_words, est):
    return [est * (c - a) / total_words for a, c in pieces]


def test_long_body_piece_is_cut_until_it_fits():
    words = _block("BLOCK 1: T", 40)["text"].split()
    out = ps._enforce_body_slot_max(words, [(0, 40)], 20.0, 3.0, 8.0)
    d = _durs(out, 40, 20.0)
    assert max(d) <= 8.0, d
    assert min(d) >= 3.0, d
    assert out[0][0] == 0 and out[-1][1] == 40
    assert all(a < c for a, c in out)


def test_cut_prefers_sentence_end():
    words = _block("BLOCK 1: T", 20, sentence_every=10)["text"].split()
    out = ps._enforce_body_slot_max(words, [(0, 20)], 10.0, 3.0, 8.0)
    assert out == [(0, 10), (10, 20)]


def test_real_time_short_piece_is_glued_then_recut():
    # Кусок [0, 6) по словам «3 с», а по времени 2.5 с — склеить и порезать
    # заново, а не оставить на склейку без потолка.
    words = _block("BLOCK 1: T", 20, sentence_every=6)["text"].split()
    times = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.5] + [2.5 + 0.55 * k for k in range(1, 15)]
    out = ps._enforce_body_slot_max(words, [(0, 6), (6, 20)], 10.0, 3.0, 8.0, times=times)
    d = [times[c] - times[a] for a, c in out]
    assert min(d) >= 3.0 and max(d) <= 8.0, (out, d)


def test_zero_disables():
    words = _block("BLOCK 1: T", 40)["text"].split()
    assert ps._enforce_body_slot_max(words, [(0, 40)], 20.0, 3.0, 0.0) == [(0, 40)]


def test_split_long_blocks_caps_body_and_leaves_hook_alone(monkeypatch):
    monkeypatch.setattr(ps, "BODY_SLOT_MAX_SEC", 8.0)
    hook = _block("HOOK", 8)
    body = _block("BLOCK 1: T", 48, sentence_every=24)
    blocks, weights = ps.split_long_blocks([hook, body], [3.5, 24.0])
    body_w = [w for b, w in zip(blocks, weights) if b["section"].startswith("BLOCK")]
    assert len(body_w) >= 3
    assert max(body_w) <= 8.0 + 1e-6, body_w
    assert [b for b in blocks if b["section"] == "HOOK"] == [hook]


def test_cap_off_reproduces_old_behaviour(monkeypatch):
    body = _block("BLOCK 1: T", 48, sentence_every=24)
    monkeypatch.setattr(ps, "BODY_SLOT_MAX_SEC", 0.0)
    _b, w_off = ps.split_long_blocks([dict(body)], [24.0])
    monkeypatch.setattr(ps, "BODY_SLOT_MAX_SEC", 8.0)
    _b, w_on = ps.split_long_blocks([dict(body)], [24.0])
    assert max(w_off) > 8.0          # одна нарезка пополам: 12 + 12
    assert max(w_on) <= 8.0 + 1e-6
