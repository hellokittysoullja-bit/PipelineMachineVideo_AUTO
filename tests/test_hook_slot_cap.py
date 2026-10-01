"""Слоты хука: потолок 5 с (HOOK_SLOT_MAX_SEC), без потери слов и тегов."""
import sys, tempfile
sys.argv = ["x", tempfile.mkdtemp()]
sys.path.insert(0, "scripts")
import pipeline_smart as ps


def _block(text, section="HOOK", pause=0.0):
    return {"section": section, "text": text, "words": len(text.split()),
            "pause_after": pause, "stat": None, "stat_word_pos": None, "sfx": []}


TEXT = ("Трое на девятьсот человек в железе и с оружием в руках, и таких битв "
        "тогда было много, настолько много, что никто уже не удивлялся этому, "
        "а просто шёл дальше и считал деньги.")


def test_long_hook_block_is_cut_to_at_most_five_seconds():
    nb, nw = ps.split_long_blocks([_block(TEXT)], [18.0])
    assert len(nb) >= 4
    assert all(x <= ps.HOOK_SLOT_MAX_SEC + 1e-6 for x in nw)
    assert abs(sum(nw) - 18.0) < 1e-6
    assert " ".join(b["text"] for b in nb) == TEXT


def test_short_hook_block_is_left_alone():
    nb, nw = ps.split_long_blocks([_block("Пять секунд хватит.")], [4.0])
    assert len(nb) == 1 and nw == [4.0]


def test_block_slightly_over_limit_is_still_cut():
    nb, nw = ps.split_long_blocks([_block(TEXT)], [5.6])
    assert len(nb) == 2 and max(nw) <= 5.0 + 1e-6


def test_pause_after_counts_toward_last_slot():
    nb, nw = ps.split_long_blocks([_block(TEXT, pause=1.0)], [8.0])
    assert nb[-1]["pause_after"] == 1.0 and nb[0]["pause_after"] == 0.0
    assert max(w + (b["pause_after"]) for b, w in zip(nb, nw)) <= 5.0 + 1e-6


def test_body_blocks_are_not_affected_by_hook_cap():
    nb, nw = ps.split_long_blocks([_block(TEXT, section="BLOCK 1: X")], [6.0])
    assert len(nb) == 1


def test_first_slot_is_at_most_three_seconds():
    nb, nw = ps.split_long_blocks([_block(TEXT)], [18.0])
    assert nw[0] <= ps.HOOK_FIRST_SLOT_MAX_SEC + 1e-6
    assert all(x <= ps.HOOK_SLOT_MAX_SEC + 1e-6 for x in nw)
    assert " ".join(b["text"] for b in nb) == TEXT


def test_first_slot_rule_only_for_the_first_block():
    blocks = [_block("Короткий первый блок хука, он идёт раньше."), _block(TEXT)]
    nb, nw = ps.split_long_blocks(blocks, [2.0, 18.0])
    second = [w for b, w in zip(nb, nw) if b["text"] in TEXT]
    assert second[0] > ps.HOOK_FIRST_SLOT_MAX_SEC


def test_first_slot_rule_skipped_when_block_too_short_for_it():
    # 5.0 с: 3 + остаток 2 < HOOK_MIN_CLIP — остаток склеился бы обратно
    nb, nw = ps.split_long_blocks([_block(TEXT)], [5.0])
    assert len(nb) == 1


def test_real_word_times_beat_the_word_share_estimate():
    """Речь быстрее оценки по словам: по доле слов первый кусок «2.8 с», а по
    настоящим временам он звучит 2.0 с — и склеился бы обратно. С реальными
    временами первый слот укладывается в потолок."""
    words = TEXT.split()
    n = len(words)
    # равномерная речь 0.5 с на слово, пауза в 1 с в конце блока
    times = [0.5 * k for k in range(n)] + [0.5 * n + 1.0]
    cuts = ps._hook_split_points(words, 0.5 * n, 1.0, first_slot=True, times=times)
    bounds = [0] + cuts + [n]
    dur = [times[c] - times[a] for a, c in zip(bounds, bounds[1:])]
    assert dur[0] <= ps.HOOK_FIRST_SLOT_MAX_SEC
    assert max(dur) <= ps.HOOK_SLOT_MAX_SEC
    assert min(dur) >= ps.HOOK_MIN_CLIP - 1e-6
