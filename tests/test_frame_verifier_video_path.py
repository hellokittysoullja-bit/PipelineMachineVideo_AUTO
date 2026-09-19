"""Зрячий гейт кадра (frame_verifier) не вызывался на видео-пути вообще.

Реальный, живьём найденный случай (19.09, тестовый рендер videos/
_test_anchor_medieval после добавления [shot:] брифов). Слот «Конница
мчится через поле прямо на пехоту» выиграло Pexels-видео современного
реконструкторского фестиваля с толпой зрителей в кадре — ровно то, что мир
канала (content_world.json) прямо запрещает, и ровно тот класс кадра,
который frame_verifier.verify() умеет ловить (см. её докстринг: живой
замер поймал ВСЕ ШЕСТЬ похожих промахов на фото-пути).

Причина: `frame_verifier.verify()` вызывалась РОВНО в одном месте всего
файла — внутри `pexels_photo()`. `pexels_video()` вообще не принимала
`block_text` и не звала зрячий гейт ни разу. Тот же класс асимметрии
«работает на фото и забыто на видео», что уже четырежды стоил этому
репозиторию половины эпизода (filter_alt_blocklist 07.09, director_score_fn
08.09, бриф стокам 15.09, video_negative_anchor_violation 08.09).

Здесь проверяется, что: (1) зрячий гейт реально вызывается из pexels_video()
и способен отклонить и переподобрать кандидата (аналог test_arbiter_
refusal.py для VLM-арбитра); (2) отказ доходит до FRAME_VERIFIER_MISSES
с kind="video"; (3) при отказе всех кандидатов слот не пустеет — остаётся
на лучшем.
"""
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]
import pipeline_smart as ps   # noqa: E402
import frame_verifier          # noqa: E402


@pytest.fixture(autouse=True)
def _clean_state():
    ps.FRAME_VERIFIER_MISSES.clear()
    yield
    ps.FRAME_VERIFIER_MISSES.clear()


def _stub_common(monkeypatch, tmp_path, ids=(1, 2, 3)):
    monkeypatch.setattr(ps, "PEXELS_API_KEY", "k")
    monkeypatch.setattr(ps, "TEMP_FOLDER", str(tmp_path))
    monkeypatch.setattr(ps, "atomic_url_download",
                        lambda req, dest, timeout=None: open(dest, "wb").write(b"x"))
    monkeypatch.setattr(ps, "_pexels_search_videos", lambda q: [
        {"id": i, "url": f"https://www.pexels.com/video/knight-armor-{i}/",
         "video_files": [{"file_type": "video/mp4", "width": 1920,
                          "link": f"http://x/{i}.mp4"}]}
        for i in ids])
    monkeypatch.setattr(ps, "_pixabay_search_videos", lambda q: [])
    monkeypatch.setattr(ps, "is_relevant_candidate", lambda *a, **k: True)
    monkeypatch.setattr(ps, "video_domain_guard_violation", lambda *a, **k: (False, None))
    monkeypatch.setattr(ps, "video_negative_anchor_violation", lambda *a, **k: (False, None))
    monkeypatch.setattr(ps, "video_sharpness_ok", lambda *a, **k: True)
    monkeypatch.setattr(ps, "extract_video_probe_frame", lambda p, **kw: (p + ".probe.jpg", False))
    monkeypatch.setattr(ps, "video_probe_in_window", lambda p, dur, offset=0.5: (p + ".fv.jpg", False))
    monkeypatch.setattr(ps, "measure_luma", lambda p: 0.4)
    # Разные хэши на разные файлы — тот же приём, что test_arbiter_refusal.py:
    # без этого дедуп схлопнул бы пул до одного кандидата.
    monkeypatch.setattr(ps, "ahash",
                        lambda p, size=8: format(abs(hash(os.path.basename(p))) % (1 << 64), "064b"))


class TestWiring:
    """Source-level: frame_verifier.verify() реально вызывается из
    pexels_video(), block_text доходит до неё из обеих точек main()."""

    def test_called_from_pexels_video(self):
        src = open(os.path.join(SCRIPTS_DIR, "pipeline_smart.py"), encoding="utf-8").read()
        assert src.count("def pexels_video(") == 1
        block = src[src.index("def pexels_video("):src.index("def quantize_dur_to_frame")]
        assert "frame_verifier.verify(probe_fv, block_text, VIDEO_FOLDER)" in block

    def test_block_text_param_exists(self):
        src = open(os.path.join(SCRIPTS_DIR, "pipeline_smart.py"), encoding="utf-8").read()
        sig_start = src.index("def pexels_video(")
        sig = src[sig_start:src.index("):", sig_start) + 2]
        assert "block_text=None" in sig

    def test_both_call_sites_pass_block_text(self):
        src = open(os.path.join(SCRIPTS_DIR, "pipeline_smart.py"), encoding="utf-8").read()
        calls = [i for i in range(len(src)) if src.startswith("video = pexels_video(", i)]
        assert len(calls) == 2
        for start in calls:
            end = src.index(")\n", start)
            call_block = src[start:end]
            assert 'block_text=b["text"]' in call_block


def test_rejected_candidate_is_recorded_and_slot_stays_filled(tmp_path, monkeypatch):
    """Все кандидаты отклонены зрячим гейтом -> слот не пустеет, остаёмся
    на лучшем, промах записан с kind='video'."""
    _stub_common(monkeypatch, tmp_path)
    monkeypatch.setattr(frame_verifier, "enabled", lambda: True)
    monkeypatch.setattr(frame_verifier, "verify", lambda *a, **k: {
        "verdict": "no", "seen": "толпа современных зрителей",
        "missing": "конница, атака, поле боя"})
    out = ps.pexels_video(
        "medieval cavalry charge field", 5, used_ids=set(), used_hashes=[],
        sentence_score_fn=lambda probe: 0.5, block_text="Конница мчится через поле.")
    assert out is not None
    assert os.path.exists(out)
    misses = [m for m in ps.FRAME_VERIFIER_MISSES if m["index"] == 5]
    assert misses and misses[0]["kind"] == "video"
    assert misses[0]["missing"] == "конница, атака, поле боя"


def test_accepted_candidate_is_not_flagged(tmp_path, monkeypatch):
    _stub_common(monkeypatch, tmp_path)
    monkeypatch.setattr(frame_verifier, "enabled", lambda: True)
    monkeypatch.setattr(frame_verifier, "verify", lambda *a, **k: {
        "verdict": "yes", "seen": "конница атакует пехоту", "missing": ""})
    out = ps.pexels_video(
        "medieval cavalry charge field", 6, used_ids=set(), used_hashes=[],
        sentence_score_fn=lambda probe: 0.5, block_text="Конница мчится через поле.")
    assert out is not None
    assert [m for m in ps.FRAME_VERIFIER_MISSES if m["index"] == 6] == []


def test_disabled_gate_is_byte_for_byte_unchanged(tmp_path, monkeypatch):
    """Выключенный гейт -> verify() не зовётся вообще, ноль регрессии."""
    _stub_common(monkeypatch, tmp_path)
    monkeypatch.setattr(frame_verifier, "enabled", lambda: False)

    def _boom(*a, **k):
        raise AssertionError("verify() не должен вызываться при enabled()=False")

    monkeypatch.setattr(frame_verifier, "verify", _boom)
    out = ps.pexels_video(
        "medieval cavalry charge field", 7, used_ids=set(), used_hashes=[],
        sentence_score_fn=lambda probe: 0.5, block_text="Конница мчится через поле.")
    assert out is not None
    assert ps.FRAME_VERIFIER_MISSES == []


def test_no_block_text_is_byte_for_byte_unchanged(tmp_path, monkeypatch):
    """Старые вызовы без block_text (block_text=None) -> гейт не зовётся,
    прежнее поведение."""
    _stub_common(monkeypatch, tmp_path)
    monkeypatch.setattr(frame_verifier, "enabled", lambda: True)

    def _boom(*a, **k):
        raise AssertionError("verify() не должен вызываться без block_text")

    monkeypatch.setattr(frame_verifier, "verify", _boom)
    out = ps.pexels_video(
        "medieval cavalry charge field", 8, used_ids=set(), used_hashes=[],
        sentence_score_fn=lambda probe: 0.5)
    assert out is not None
