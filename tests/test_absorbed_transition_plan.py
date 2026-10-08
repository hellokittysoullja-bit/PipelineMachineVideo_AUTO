"""Склейка после поглощённого слота идёт по плану ПОЛНОГО списка блоков.

Живой случай эп.05 (08.10): после двух поглощений (слоты 78 и 82) склейка
пересчитывала план по укороченному списку клипов. Тип перехода выбирается
хэшем номера склейки, поэтому все следующие переходы менялись, а длительности
клипов были посчитаны по старому плану: рез отставал от фразы на 0.2-0.35 с до
конца ролика (verify_timing, медиана 3 мс до поглощения и 212 мс после).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import pipeline_smart as ps  # noqa: E402


def _blocks(n, per_section=7):
    return [{"section": f"BLOCK {i // per_section + 1}: T", "text": f"фраза {i}"}
            for i in range(n)]


def test_without_absorption_layout_equals_full_plan():
    blocks = _blocks(130)
    sections = [b["section"] for b in blocks]
    plan, bounds = ps.present_transition_layout(blocks, list(range(len(blocks))))
    assert plan == ps.plan_transitions(sections, blocks)
    assert bounds == ps._chunk_bounds(len(blocks), sections, ps.XFADE_CHUNK_SIZE,
                                      ps.chapter_card_no_split(blocks))


def test_after_absorption_transitions_keep_their_planned_type():
    blocks = _blocks(130)
    sections = [b["section"] for b in blocks]
    full = ps.plan_transitions(sections, blocks)
    absorbed = {78, 82}
    present = [i for i in range(len(blocks)) if i not in absorbed]
    plan, _bounds = ps.present_transition_layout(blocks, present)
    for pos in range(1, len(present)):
        j = present[pos]
        run_start = present[pos - 1] + 1
        assert plan[pos - 1] == full[run_start - 1], (j, run_start)
    # Переходы ПОСЛЕ поглощения — те же, что в полном плане (раньше они
    # пересчитывались по сдвинутым номерам и менялись).
    after = [plan[p - 1] for p in range(1, len(present)) if present[p] > 83]
    assert after == [full[j - 1] for j in present if j > 83]


def test_absorbed_slot_carries_duration_minus_its_outgoing_overlap():
    src = open(ps.__file__, encoding="utf-8").read()
    assert "_carry_sec = max(0.0, d - (_pt_plan[i][1] if i < len(_pt_plan) else 0.0))" in src
    assert "plan=_x_plan, bounds=_x_bounds" in src
