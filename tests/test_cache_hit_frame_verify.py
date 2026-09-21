"""Кэш-хит `pexels_photo()`/`pexels_video()` был слеп к собственному же
вердикту зрячего гейта (найдено живым прогоном 21.09).

Победитель слота, подобранный ОДНИМ прогоном, кэшируется файлом по имени
(index, хэш запроса, подпись гейтов). Прошлый прогон, оборванный ДО того,
как он дописал финальные отчёты (`timeout`, SIGKILL, OOM — реальный случай:
`timeout 280` убил процесс ПОСЛЕ того, как файл уже лежал на диске), мог
успеть спросить зрячий гейт и получить «нет» — вердикт лежит в `media_plan/
frame_verdicts/`, ключуемый по содержимому кадра+фразы+мира+брифа. Старый
код на кэш-хите СРАЗУ отдавал файл, ни разу туда не заглядывая: живая
проверка на реальном эпизоде показала слот «Вот кинжал.», отдавший рисунок
двух мечей с Wikimedia, хотя `frame_verdicts/` по этому же файлу+фразе уже
хранил `{"verdict": "no", ...}`.

Починка: кэш-хит спрашивает `frame_verifier.verify()` (сам кэшируется по
содержимому — платного вызова не будет, если вопрос уже задавался) ПЕРЕД
тем, как отдать файл. «Нет» -> кэш-хит не считается валидным, функция падает
в обычный путь свежего подбора."""
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
    ps.FRAME_VERIFIER_GAVE_UP.clear()
    yield
    ps.FRAME_VERIFIER_MISSES.clear()
    ps.FRAME_VERIFIER_GAVE_UP.clear()


def _stub_photo(monkeypatch, tmp_path, search_calls):
    monkeypatch.setattr(ps, "PEXELS_API_KEY", "k")
    monkeypatch.setattr(ps, "TEMP_FOLDER", str(tmp_path))

    def _search(q):
        search_calls.append(q)
        return [{"id": i, "alt": "", "url": f"https://www.pexels.com/photo/knife-{i}/",
                 "src": {"large2x": f"http://x/{i}.jpg"}} for i in range(1, 6)]

    monkeypatch.setattr(ps, "_pexels_search_photos", _search)
    monkeypatch.setattr(ps, "disambiguate_search_query", lambda q: q)
    monkeypatch.setattr(ps, "atomic_url_download",
                        lambda req, dest, timeout=None: open(dest, "wb").write(b"x"))
    monkeypatch.setattr(ps, "is_relevant_candidate", lambda *a, **k: True)
    monkeypatch.setattr(ps, "image_sharpness_score", lambda p: 999.0)
    monkeypatch.setattr(ps, "aesthetic_score", lambda p: 5.0)
    monkeypatch.setattr(ps, "estimate_shot_size", lambda p: "medium")
    monkeypatch.setattr(ps, "measure_luma", lambda p: 0.4)
    monkeypatch.setattr(ps, "clip_relevance", lambda p, t: 0.5)
    monkeypatch.setattr(ps, "ahash",
                        lambda p, size=8: format(abs(hash(os.path.basename(p))) % (1 << 64), "064b"))


def _stub_video(monkeypatch, tmp_path, search_calls):
    monkeypatch.setattr(ps, "PEXELS_API_KEY", "k")
    monkeypatch.setattr(ps, "TEMP_FOLDER", str(tmp_path))

    def _search(q):
        search_calls.append(q)
        return [{"id": i, "url": f"https://www.pexels.com/video/knife-{i}/",
                 "video_files": [{"file_type": "video/mp4", "width": 1920,
                                  "link": f"http://x/{i}.mp4"}]} for i in range(1, 6)]

    monkeypatch.setattr(ps, "atomic_url_download",
                        lambda req, dest, timeout=None: open(dest, "wb").write(b"x"))
    monkeypatch.setattr(ps, "_pexels_search_videos", _search)
    monkeypatch.setattr(ps, "_pixabay_search_videos", lambda q: [])
    monkeypatch.setattr(ps, "is_relevant_candidate", lambda *a, **k: True)
    monkeypatch.setattr(ps, "video_domain_guard_violation", lambda *a, **k: (False, None))
    monkeypatch.setattr(ps, "video_negative_anchor_violation", lambda *a, **k: (False, None))
    monkeypatch.setattr(ps, "video_sharpness_ok", lambda *a, **k: True)
    monkeypatch.setattr(ps, "extract_video_probe_frame", lambda p, **kw: (p + ".probe.jpg", False))
    monkeypatch.setattr(ps, "video_probe_in_window", lambda p, dur, offset=0.5: (p + ".fv.jpg", False))
    monkeypatch.setattr(ps, "measure_luma", lambda p: 0.4)
    monkeypatch.setattr(ps, "ahash",
                        lambda p, size=8: format(abs(hash(os.path.basename(p))) % (1 << 64), "064b"))


class TestPhotoCacheHitAsksFrameVerifier:
    def test_cached_no_verdict_forces_fresh_search(self, tmp_path, monkeypatch):
        search_calls = []
        _stub_photo(monkeypatch, tmp_path, search_calls)
        # Прогон 1: гейт выключен, файл ложится на диск по обычному пути.
        monkeypatch.setattr(frame_verifier, "enabled", lambda: False)
        first = ps.pexels_photo(
            "arrow deflecting breastplate", 3, used_ids=set(), used_hashes=[],
            block_text="Стрела скользит по нагруднику и уходит в сторону.")
        assert first is not None and os.path.exists(first)
        assert len(search_calls) == 1

        # Прогон 2: тот же слот, тот же запрос -> тот же cf уже на диске.
        # Гейт включён и говорит «нет» на кэш-хит -> обязан переподобрать,
        # а не отдать файл молча.
        monkeypatch.setattr(frame_verifier, "enabled", lambda: True)
        monkeypatch.setattr(frame_verifier, "verify", lambda *a, **k: {
            "verdict": "no", "seen": "бронзовый церемониальный меч-скипетр",
            "missing": "стрела, нагрудник, рыцарь"})
        second = ps.pexels_photo(
            "arrow deflecting breastplate", 3, used_ids=set(), used_hashes=[],
            block_text="Стрела скользит по нагруднику и уходит в сторону.")
        assert second is not None
        assert len(search_calls) == 2, (
            "кэш-хит с вердиктом «нет» обязан пойти в обычный путь подбора "
            "(новый поиск), а не отдать файл молча")
        cache_hit_misses = [m for m in ps.FRAME_VERIFIER_MISSES
                            if m["index"] == 3 and m.get("candidate") == "cache_hit"]
        assert cache_hit_misses, "отказ по кэш-хиту обязан оставить след в MISSES"

    def test_cached_yes_verdict_keeps_fast_path(self, tmp_path, monkeypatch):
        search_calls = []
        _stub_photo(monkeypatch, tmp_path, search_calls)
        monkeypatch.setattr(frame_verifier, "enabled", lambda: False)
        first = ps.pexels_photo(
            "arrow deflecting breastplate", 4, used_ids=set(), used_hashes=[],
            block_text="Стрела скользит по нагруднику и уходит в сторону.")
        assert len(search_calls) == 1

        monkeypatch.setattr(frame_verifier, "enabled", lambda: True)
        monkeypatch.setattr(frame_verifier, "verify", lambda *a, **k: {
            "verdict": "yes", "seen": "стрела бьёт по нагруднику", "missing": ""})
        second = ps.pexels_photo(
            "arrow deflecting breastplate", 4, used_ids=set(), used_hashes=[],
            block_text="Стрела скользит по нагруднику и уходит в сторону.")
        assert second == first
        assert len(search_calls) == 1, "вердикт «да» обязан сохранить быстрый путь без нового поиска"

    def test_disabled_gate_never_touches_cache_hit(self, tmp_path, monkeypatch):
        search_calls = []
        _stub_photo(monkeypatch, tmp_path, search_calls)
        monkeypatch.setattr(frame_verifier, "enabled", lambda: False)
        first = ps.pexels_photo(
            "arrow deflecting breastplate", 5, used_ids=set(), used_hashes=[],
            block_text="Стрела скользит по нагруднику и уходит в сторону.")
        second = ps.pexels_photo(
            "arrow deflecting breastplate", 5, used_ids=set(), used_hashes=[],
            block_text="Стрела скользит по нагруднику и уходит в сторону.")
        assert second == first
        assert len(search_calls) == 1, "выключенный гейт — байт-в-байт прежнее поведение кэш-хита"


class TestVideoCacheHitAsksFrameVerifier:
    def test_cached_no_verdict_forces_fresh_search(self, tmp_path, monkeypatch):
        search_calls = []
        _stub_video(monkeypatch, tmp_path, search_calls)
        monkeypatch.setattr(frame_verifier, "enabled", lambda: False)
        first = ps.pexels_video(
            "medieval cavalry charge field", 6, used_ids=set(), used_hashes=[],
            sentence_score_fn=lambda probe: 0.5, block_text="Конница мчится через поле.")
        assert first is not None
        assert len(search_calls) == 1

        monkeypatch.setattr(frame_verifier, "enabled", lambda: True)
        monkeypatch.setattr(frame_verifier, "verify", lambda *a, **k: {
            "verdict": "no", "seen": "рыцари дерутся, современные зрители",
            "missing": "скачущая конница, пехота"})
        second = ps.pexels_video(
            "medieval cavalry charge field", 6, used_ids=set(), used_hashes=[],
            sentence_score_fn=lambda probe: 0.5, block_text="Конница мчится через поле.")
        assert second is not None
        assert len(search_calls) == 2, (
            "видео-кэш-хит с вердиктом «нет» обязан пойти в обычный путь "
            "подбора, а не отдать файл молча")
        cache_hit_misses = [m for m in ps.FRAME_VERIFIER_MISSES
                            if m["index"] == 6 and m.get("candidate") == "cache_hit"]
        assert cache_hit_misses

    def test_cached_yes_verdict_keeps_fast_path(self, tmp_path, monkeypatch):
        search_calls = []
        _stub_video(monkeypatch, tmp_path, search_calls)
        monkeypatch.setattr(frame_verifier, "enabled", lambda: False)
        first = ps.pexels_video(
            "medieval cavalry charge field", 7, used_ids=set(), used_hashes=[],
            sentence_score_fn=lambda probe: 0.5, block_text="Конница мчится через поле.")
        assert len(search_calls) == 1

        monkeypatch.setattr(frame_verifier, "enabled", lambda: True)
        monkeypatch.setattr(frame_verifier, "verify", lambda *a, **k: {
            "verdict": "yes", "seen": "конница атакует пехоту", "missing": ""})
        second = ps.pexels_video(
            "medieval cavalry charge field", 7, used_ids=set(), used_hashes=[],
            sentence_score_fn=lambda probe: 0.5, block_text="Конница мчится через поле.")
        assert second == first
        assert len(search_calls) == 1
