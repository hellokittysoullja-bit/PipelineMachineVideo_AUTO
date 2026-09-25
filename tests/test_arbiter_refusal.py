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
from _media_calls import pick_photo  # noqa: E402
from _video_world import QUERY, infra, video  # noqa: E402,F401
import selection_engine  # noqa: E402
import dataclasses  # noqa: E402


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
    """Видео-путь (VideoAdapter в общем ядре). Раньше здесь был риск уронить
    рендер NameError: `import shot_director` стоял внутри ветки длинного
    шорт-листа. Теперь арбитр зовётся тем же построением шорт-листа, что у
    фото, — эти тесты держат, что отказ записывается, сентинел не становится
    путём, короткий шорт-лист не зовёт арбитра, а обычный выбор применяется."""

    def _world(self, infra, monkeypatch):
        # Шорт-лист арбитра — победитель и лучший по СВОЕМУ запросу слота:
        # кандидаты приходят из разных запросов пула, иначе он из одного.
        infra["videos"] = [video(1), video(2)]
        infra["relevant"] = {1, 2}
        infra["rel"] = {1: 0.1, 2: 0.4}
        monkeypatch.setattr(pipeline_smart, "_pexels_search_videos",
                            lambda q: [video(1)] if "sword" in q else [video(2)])
        monkeypatch.setattr(pipeline_smart.feature_flags, "mode",
                            lambda n: "on" if n == "VLM_ARBITER_MODE" else "off")

    def _run(self, index, **kw):
        att = pipeline_smart.new_attempt(index, "video")
        fields = {f.name: None for f in dataclasses.fields(selection_engine.SlotRequest)}
        fields.update(index=index, query=QUERY, extra_queries=("mounted knight field",),
                      is_opening=False, director_assist=False, arbiter_text="Текст.")
        fields.update(kw)
        with pipeline_smart.selection_attempt.activate(att):
            out = pipeline_smart.select_media(pipeline_smart.build_slot_request(**fields), "video")
        return out, [v for k, v in att.verdicts if k == "arbiter"]

    def test_refusal_is_recorded_and_does_not_crash(self, infra, monkeypatch):
        self._world(infra, monkeypatch)
        monkeypatch.setattr(shot_director, "arbitrate_hook_candidates",
                            lambda *a, **k: shot_director.NO_CANDIDATE_FITS)
        out, refusals = self._run(7, arbiter_text="Вот это. Это вес настоящего боевого меча.")
        # Слот заполнен: что делать с отказом, решает лестница после
        # отбора; здесь проверяется, что отказ ЗАМЕЧЕН.
        assert out is not None and os.path.exists(out)
        assert [r["index"] for r in refusals] == [7] and refusals[0]["kind"] == "video"

    def test_sentinel_is_never_treated_as_a_file_path(self, infra, monkeypatch):
        self._world(infra, monkeypatch)
        monkeypatch.setattr(shot_director, "arbitrate_hook_candidates",
                            lambda *a, **k: shot_director.NO_CANDIDATE_FITS)
        out, _ = self._run(3)
        assert isinstance(out, str) and out.endswith(".mp4")

    def test_short_shortlist_does_not_call_the_arbiter(self, infra, monkeypatch):
        self._world(infra, monkeypatch)
        infra["videos"] = [video(1)]
        monkeypatch.setattr(pipeline_smart, "_pexels_search_videos", lambda q: [video(1)])
        called = []
        monkeypatch.setattr(shot_director, "arbitrate_hook_candidates",
                            lambda *a, **k: called.append(1))
        out, refusals = self._run(4)
        assert out is not None and refusals == [] and called == []

    def test_normal_pick_still_works(self, infra, monkeypatch):
        self._world(infra, monkeypatch)
        seen = {}

        def fake(text, paths, ids, video_dir, is_opening=False):
            seen["ids"] = list(ids)
            return paths[-1]
        monkeypatch.setattr(shot_director, "arbitrate_hook_candidates", fake)
        out, refusals = self._run(5)
        assert out is not None and refusals == []
        assert len(seen.get("ids", [])) >= 2
        # Скачан именно выбранный арбитром кандидат.
        assert infra["downloads"][0] == seen["ids"][-1]


class TestPhotoPath:
    def _stub_photo(self, monkeypatch, tmp_path, n=3):
        _stub_common(monkeypatch, tmp_path)
        monkeypatch.setattr(pipeline_smart, "_pexels_search_photos", lambda q: [
            {"id": i, "alt": "", "url": f"https://www.pexels.com/photo/knight-{i}/",
             "src": {"large2x": f"http://x/{i}.jpg"}}
            for i in range(1, n + 1)])
        monkeypatch.setattr(pipeline_smart, "disambiguate_search_query", lambda q: q)
        monkeypatch.setattr(pipeline_smart, "image_local_sharpness", lambda p: 999.0)
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
        out = pick_photo(pipeline_smart, 
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
        out = pick_photo(pipeline_smart, 
            "milk bottle hand", 6, used_ids=set(), used_hashes=[],
            is_opening_shot=True, arbiter_text="Текст.")
        assert out is not None
        assert called, (
            "арбитр снова не вызвался без Director — пул опять схлопнулся "
            "до одного кандидата, проверить BASE_MIN_POOL/_base_min_pool_for")
