"""SHOW_BEST_REJECTED: лучший из отклонённых судьёй кадр показывается."""
import sys, tempfile
sys.argv = ["x", tempfile.mkdtemp()]
sys.path.insert(0, "scripts")
import pipeline_smart as ps


def test_only_judge_verdict_with_score_allows_showing_best_rejected():
    assert ps.best_rejected_score([("judge", {"score": 2})]) == 2
    assert ps.best_rejected_score([("judge", {"score": 0})]) == 0


def test_other_rejection_reasons_still_absorb():
    assert ps.best_rejected_score([("judge", {"score": 2}), ("arbiter", {})]) is None
    assert ps.best_rejected_score([("smart_veto", {})]) is None
    assert ps.best_rejected_score(None) is None


def test_zero_grid_score_is_below_the_show_threshold():
    assert ps.best_rejected_score([("judge", {"score": 0})]) < ps.SHOW_BEST_REJECTED_MIN
