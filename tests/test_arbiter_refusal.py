"""Отказ VLM-арбитра доходит до вызывающего кода и не притворяется кадром.

Разбор реального случая (07.09). Промпт арбитра прямо разрешает ответить 0 —
«если ни одна картинка реально не подходит». В опубликованном эпизоде модель
этим воспользовалась: в `media_plan/shot_director_cache/` лежит вердикт для
фразы «Не вставай никуда. Просто вспомни, сколько весит пакет молока...» —
`candidate_ids [33508363, 4867362, 8285555], choice: 0`.

Но `_resolve_choice()` возвращала на этот ответ ровно тот же `None`, что и на
«режим off», «нет ключа», «лимит исчерпан», «сеть упала». А вызывающий код на
`None` делает `if arbiter_pick is not None` — то есть молча остаётся на выборе
эмбеддинга. Итог: система спросила самого сильного судью, получила «ни один не
годится» и показала зрителю кадр с младенцем и детской бутылочкой в хуке.

Здесь проверяется НЕ сам `shot_director` (это делает test_shot_director.py), а
именно поведение вызывающих его точек в `pipeline_smart.py`: обе обязаны
опознать отказ, записать его и НЕ пытаться найти файл по сентинелу.
"""
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]
import pipeline_smart  # noqa: E402
import shot_director  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_miss_lists():
    pipeline_smart.ARBITER_REJECTED_ALL.clear()
    yield
    pipeline_smart.ARBITER_REJECTED_ALL.clear()


def _stub_common(monkeypatch, tmp_path):
    monkeypatch.setenv("VLM_ARBITER_MODE", "on")
    monkeypatch.setattr(pipeline_smart, "PEXELS_API_KEY", "k")
    monkeypatch.setattr(pipeline_smart, "TEMP_FOLDER", str(tmp_path))
    monkeypatch.setattr(pipeline_smart, "atomic_url_download",
                        lambda req, dest, timeout=None: open(dest, "wb").write(b"x"))
    monkeypatch.setattr(pipeline_smart, "is_relevant_candidate", lambda *a, **k: True)
    # Разные хэши на разные файлы: одинаковый ahash сделал бы всех кандидатов
    # взаимными near-duplicate, дедуп схлопнул бы пул до одного, и шорт-лист
    # арбитра (нужно >= 2) вообще не собрался бы — тест проверял бы не то.
    monkeypatch.setattr(pipeline_smart, "ahash",
                        lambda p, size=8: format(abs(hash(os.path.basename(p))) % (1 << 64), "064b"))


class TestVideoPath:
    """Видео-путь: здесь же был риск уронить рендер NameError.

    `import shot_director` в этой ветке стоит ВНУТРИ `if len(probes) >= 2`,
    поэтому сравнение с сентинелом обязано идти после проверки на None —
    иначе при коротком шорт-листе имени в скоупе нет и падает весь рендер.
    """

    def _stub_video(self, monkeypatch, tmp_path, ids=(1, 2)):
        _stub_common(monkeypatch, tmp_path)
        monkeypatch.setattr(pipeline_smart, "_pexels_search_videos", lambda q: [
            {"id": i, "url": f"https://www.pexels.com/video/knight-armor-{i}/",
             "video_files": [{"file_type": "video/mp4", "width": 1920,
                              "link": f"http://x/{i}.mp4"}]}
            for i in ids])
        monkeypatch.setattr(pipeline_smart, "extract_video_probe_frame",
                            lambda p, **kw: (p + ".jpg", False))
        monkeypatch.setattr(pipeline_smart, "video_domain_guard_violation",
                            lambda *a, **k: (False, None))
        monkeypatch.setattr(pipeline_smart, "measure_luma", lambda p: 0.4)

    def test_refusal_is_recorded_and_does_not_crash(self, tmp_path, monkeypatch):
        self._stub_video(monkeypatch, tmp_path)
        monkeypatch.setattr(shot_director, "arbitrate_hook_candidates",
                            lambda *a, **k: shot_director.NO_CANDIDATE_FITS)
        out = pipeline_smart.pexels_video(
            "medieval sword close up", 7, used_ids=set(), used_hashes=[],
            sentence_score_fn=lambda probe: 0.5,
            arbiter_text="Вот это. Это вес настоящего боевого меча.")
        # Слот по-прежнему заполнен: лестница фолбэков — отдельный шаг, здесь
        # проверяется только то, что отказ ЗАМЕЧЕН, а не что он уже обработан.
        assert out is not None
        assert os.path.exists(out)
        assert [m["index"] for m in pipeline_smart.ARBITER_REJECTED_ALL] == [7]
        assert pipeline_smart.ARBITER_REJECTED_ALL[0]["kind"] == "video"

    def test_sentinel_is_never_treated_as_a_file_path(self, tmp_path, monkeypatch):
        """Главная защита: сентинел не должен доехать до файловой системы."""
        self._stub_video(monkeypatch, tmp_path)
        monkeypatch.setattr(shot_director, "arbitrate_hook_candidates",
                            lambda *a, **k: shot_director.NO_CANDIDATE_FITS)
        out = pipeline_smart.pexels_video(
            "medieval sword close up", 3, used_ids=set(), used_hashes=[],
            sentence_score_fn=lambda probe: 0.5, arbiter_text="Текст.")
        assert isinstance(out, str) and out.endswith(".mp4")

    def test_short_shortlist_does_not_raise_name_error(self, tmp_path, monkeypatch):
        """Один кандидат -> арбитр не зовётся, shot_director не импортирован.

        Именно здесь наивная проверка `arbiter_pick is shot_director.NO_...`
        уронила бы весь рендер NameError после часов работы.
        """
        self._stub_video(monkeypatch, tmp_path, ids=(1,))
        out = pipeline_smart.pexels_video(
            "medieval sword close up", 4, used_ids=set(), used_hashes=[],
            sentence_score_fn=lambda probe: 0.5, arbiter_text="Текст.")
        assert out is not None
        assert pipeline_smart.ARBITER_REJECTED_ALL == []

    def test_normal_pick_still_works(self, tmp_path, monkeypatch):
        """Ноль регрессии: обычный выбор арбитра по-прежнему применяется."""
        self._stub_video(monkeypatch, tmp_path)
        seen = {}

        def fake(text, paths, ids, video_dir, is_opening=False):
            seen["paths"] = list(paths)
            return paths[-1]

        monkeypatch.setattr(shot_director, "arbitrate_hook_candidates", fake)
        out = pipeline_smart.pexels_video(
            "medieval sword close up", 5, used_ids=set(), used_hashes=[],
            sentence_score_fn=lambda probe: 0.5, arbiter_text="Текст.")
        assert out is not None
        assert pipeline_smart.ARBITER_REJECTED_ALL == []
        assert len(seen.get("paths", [])) >= 2


class TestPhotoPath:
    def _stub_photo(self, monkeypatch, tmp_path, n=3):
        _stub_common(monkeypatch, tmp_path)
        monkeypatch.setattr(pipeline_smart, "_pexels_search_photos", lambda q: [
            {"id": i, "alt": "", "url": f"https://www.pexels.com/photo/knight-{i}/",
             "src": {"large2x": f"http://x/{i}.jpg"}}
            for i in range(1, n + 1)])
        monkeypatch.setattr(pipeline_smart, "disambiguate_search_query", lambda q: q)
        monkeypatch.setattr(pipeline_smart, "image_sharpness_score", lambda p: 999.0)
        monkeypatch.setattr(pipeline_smart, "aesthetic_score", lambda p: 5.0)
        monkeypatch.setattr(pipeline_smart, "estimate_shot_size", lambda p: "medium")
        monkeypatch.setattr(pipeline_smart, "measure_luma", lambda p: 0.4)
        monkeypatch.setattr(pipeline_smart, "clip_relevance", lambda p, t: 0.5)

    def test_refusal_is_recorded(self, tmp_path, monkeypatch):
        self._stub_photo(monkeypatch, tmp_path)
        monkeypatch.setattr(shot_director, "arbitrate_hook_candidates",
                            lambda *a, **k: shot_director.NO_CANDIDATE_FITS)
        # is_opening_shot: шорт-лист собирается по эстетике из ВСЕГО пула
        # (_build_opening_shortlist), а не только из кандидатов своего
        # запроса — иначе на этом стабе он схлопывается до одного элемента,
        # арбитр не зовётся, и тест проверял бы не то. Заодно это ровно тот
        # слот, где в опубликованном эпизоде и случился разбираемый провал.
        # director_score_fn обязателен, а не декоративен: без него цикл
        # подбора фото останавливается на ПЕРВОМ прошедшем гейты кандидате,
        # пул равен одному, а шорт-лист арбитра требует >= 2 — то есть при
        # выключенном Visual Director арбитр на фото-пути не вызывается
        # вообще. В .env этого канала стоит assist, поэтому в проде он
        # работает; тест воспроизводит именно продовую конфигурацию.
        out = pipeline_smart.pexels_photo(
            "milk bottle hand", 5, used_ids=set(), used_hashes=[],
            is_opening_shot=True, director_score_fn=lambda *a, **k: 0.5,
            arbiter_text="Не вставай никуда. Просто вспомни, сколько весит пакет молока.")
        assert out is not None and os.path.exists(out)
        rec = pipeline_smart.ARBITER_REJECTED_ALL
        assert [m["index"] for m in rec] == [5], rec
        assert rec[0]["kind"] == "photo"
        assert "пакет молока" in rec[0]["text"]

    def test_arbiter_can_now_run_without_the_director(self, tmp_path, monkeypatch):
        """Раньше зафиксированный дефект, теперь исправлен и закреплён иначе.

        До BASE_MIN_POOL (07.09) цикл подбора фото останавливался на ПЕРВОМ
        кандидате, прошедшем гейты, если Visual Director не просил больший
        пул: пул из одного кандидата -> шорт-лист из одного -> `len(shortlist)
        >= 2` не выполняется -> VLM-арбитр не вызывался НИ РАЗУ, молча. Самый
        сильный судья в системе выключался побочным эффектом другого флага.

        BASE_MIN_POOL=4 (независимый от VISUAL_DIRECTOR_MODE пол пула) это
        чинит: даже при выключенном Director пул реально собирается, и у
        арбитра снова есть из чего строить шорт-лист. Этот тест теперь
        закрепляет ЖЕЛАЕМОЕ поведение, а не задокументированный баг.
        """
        self._stub_photo(monkeypatch, tmp_path)
        called = []
        monkeypatch.setattr(shot_director, "arbitrate_hook_candidates",
                            lambda *a, **k: called.append(1) or shot_director.NO_CANDIDATE_FITS)
        out = pipeline_smart.pexels_photo(
            "milk bottle hand", 6, used_ids=set(), used_hashes=[],
            is_opening_shot=True, arbiter_text="Текст.")
        assert out is not None
        assert called, (
            "арбитр снова не вызвался без Director — пул опять схлопнулся "
            "до одного кандидата, проверить BASE_MIN_POOL/_base_min_pool_for")
