"""FRAME_VERIFIER_GAVE_UP — сигнал, которого не хватало лестнице фолбэков.

Реальный, найденный живым прогоном случай (19.09), прямое продолжение
разбора tests/test_frame_verifier_video_path.py. Прямая жалоба владельца:
«Стрела скользит по нагруднику и уходит в сторону» получило современные
кухонные ножи; «Он не поднимается — доспех держит человека, как капкан»
получило турнирную сшибку на конях. Оба случая система УЖЕ диагностировала
правильно — `chosen_by` содержал `frame_verify_repick`, консоль печатала
«слот N: зрячий гейт отклонил всех кандидатов», отчёт `frame_verifier_
report.json` был на диске — но решение «показать честную карточку вместо
этого» никто не принимал, потому что `_slot_known_bad_reason()` (её и
читает лестница фолбэков) никогда не проверяла `FRAME_VERIFIER_MISSES`
или что-либо производное от неё.

Комментарий у объявления `FRAME_VERIFIER_MISSES` годами обещал: «решение
"лучше карточка, чем чужой кадр" принимает лестница фолбэков, у которой
для этого есть причина» — обещание не выполнялось.

`FRAME_VERIFIER_GAVE_UP` — ОТДЕЛЬНЫЙ от `FRAME_VERIFIER_MISSES` список:
MISSES копит КАЖДУЮ отклонённую попытку, включая те, после которых
переподбор нашёл нормальный кадр (цикл кончился вердиктом «да»). По
присутствию индекса в MISSES нельзя понять, чем кончился слот — нужен
именно ФИНАЛЬНЫЙ исход: цикл кончился, а последний известный вердикт
всё ещё «нет»."""
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


def _stub_photo(monkeypatch, tmp_path, n=5):
    monkeypatch.setattr(ps, "PEXELS_API_KEY", "k")
    monkeypatch.setattr(ps, "TEMP_FOLDER", str(tmp_path))
    monkeypatch.setattr(ps, "_pexels_search_photos", lambda q: [
        {"id": i, "alt": "", "url": f"https://www.pexels.com/photo/knife-{i}/",
         "src": {"large2x": f"http://x/{i}.jpg"}}
        for i in range(1, n + 1)])
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


def _stub_video(monkeypatch, tmp_path, ids=(1, 2, 3, 4, 5)):
    monkeypatch.setattr(ps, "PEXELS_API_KEY", "k")
    monkeypatch.setattr(ps, "TEMP_FOLDER", str(tmp_path))
    monkeypatch.setattr(ps, "atomic_url_download",
                        lambda req, dest, timeout=None: open(dest, "wb").write(b"x"))
    monkeypatch.setattr(ps, "_pexels_search_videos", lambda q: [
        {"id": i, "url": f"https://www.pexels.com/video/knife-{i}/",
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
    monkeypatch.setattr(ps, "ahash",
                        lambda p, size=8: format(abs(hash(os.path.basename(p))) % (1 << 64), "064b"))


class TestPhotoPathGaveUp:
    def test_exhausted_repicks_are_recorded_as_gave_up(self, tmp_path, monkeypatch):
        """Все кандидаты отклонены -> цикл исчерпывает бюджет переподбора,
        FRAME_VERIFIER_GAVE_UP получает индекс, и лестница фолбэков теперь
        видит это как причину для карточки."""
        _stub_photo(monkeypatch, tmp_path)
        monkeypatch.setattr(frame_verifier, "enabled", lambda: True)
        monkeypatch.setattr(frame_verifier, "verify", lambda *a, **k: {
            "verdict": "no", "seen": "современные кухонные ножи на доске",
            "missing": "стрела, нагрудник, рыцарь"})
        out = ps.pexels_photo(
            "arrow deflecting breastplate", 7, used_ids=set(), used_hashes=[],
            block_text="Стрела скользит по нагруднику и уходит в сторону.")
        assert out is not None
        gave_up = [m for m in ps.FRAME_VERIFIER_GAVE_UP if m["index"] == 7]
        assert gave_up, "исчерпанный гейт обязан оставить след для лестницы фолбэков"
        assert gave_up[0]["kind"] == "photo"
        assert gave_up[0]["missing"] == "стрела, нагрудник, рыцарь"
        # И главное — тот самый недостающий провод: лестница фолбэков
        # теперь ВИДИТ это как причину.
        assert ps._slot_known_bad_reason(7) == "frame_verifier_gave_up"

    def test_accepted_candidate_leaves_no_trace(self, tmp_path, monkeypatch):
        """Гейт сразу принял кандидата -> ни MISSES, ни GAVE_UP, ни причины
        для карточки. Ноль ложных срабатываний на нормальном отборе."""
        _stub_photo(monkeypatch, tmp_path)
        monkeypatch.setattr(frame_verifier, "enabled", lambda: True)
        monkeypatch.setattr(frame_verifier, "verify", lambda *a, **k: {
            "verdict": "yes", "seen": "стрела бьёт по нагруднику", "missing": ""})
        out = ps.pexels_photo(
            "arrow deflecting breastplate", 8, used_ids=set(), used_hashes=[],
            block_text="Стрела скользит по нагруднику и уходит в сторону.")
        assert out is not None
        assert [m for m in ps.FRAME_VERIFIER_GAVE_UP if m["index"] == 8] == []
        assert ps._slot_known_bad_reason(8) is None

    def test_disabled_gate_never_populates_gave_up(self, tmp_path, monkeypatch):
        _stub_photo(monkeypatch, tmp_path)
        monkeypatch.setattr(frame_verifier, "enabled", lambda: False)
        out = ps.pexels_photo(
            "arrow deflecting breastplate", 9, used_ids=set(), used_hashes=[],
            block_text="Стрела скользит по нагруднику и уходит в сторону.")
        assert out is not None
        assert ps.FRAME_VERIFIER_GAVE_UP == []


class TestVideoPathGaveUp:
    def test_exhausted_repicks_are_recorded_as_gave_up(self, tmp_path, monkeypatch):
        """Тот же случай, что и жалоба владельца про толпу зрителей вместо
        конницы — видео-путь обязан вести себя так же, как фото-путь."""
        _stub_video(monkeypatch, tmp_path)
        monkeypatch.setattr(frame_verifier, "enabled", lambda: True)
        monkeypatch.setattr(frame_verifier, "verify", lambda *a, **k: {
            "verdict": "no", "seen": "рыцари дерутся, современные зрители",
            "missing": "скачущая конница, пехота"})
        out = ps.pexels_video(
            "medieval cavalry charge field", 5, used_ids=set(), used_hashes=[],
            sentence_score_fn=lambda probe: 0.5, block_text="Конница мчится через поле.")
        assert out is not None
        gave_up = [m for m in ps.FRAME_VERIFIER_GAVE_UP if m["index"] == 5]
        assert gave_up and gave_up[0]["kind"] == "video"
        assert ps._slot_known_bad_reason(5) == "frame_verifier_gave_up"

    def test_accepted_candidate_leaves_no_trace(self, tmp_path, monkeypatch):
        _stub_video(monkeypatch, tmp_path)
        monkeypatch.setattr(frame_verifier, "enabled", lambda: True)
        monkeypatch.setattr(frame_verifier, "verify", lambda *a, **k: {
            "verdict": "yes", "seen": "конница атакует пехоту", "missing": ""})
        out = ps.pexels_video(
            "medieval cavalry charge field", 6, used_ids=set(), used_hashes=[],
            sentence_score_fn=lambda probe: 0.5, block_text="Конница мчится через поле.")
        assert out is not None
        assert [m for m in ps.FRAME_VERIFIER_GAVE_UP if m["index"] == 6] == []
        assert ps._slot_known_bad_reason(6) is None


class TestSnapshotRestoreIncludesFrameVerifier:
    """Тот же класс, что уже закрыт для relevance/stock/arbiter (см.
    _slot_miss_snapshot): вердикт зрячего гейта на ОТВЕРГНУТОМ видео не
    имеет права остаться висеть на слоте после того, как видео-фото-
    спасение заменит его совсем другим кадром."""

    def test_snapshot_removes_and_restore_returns_it(self):
        ps.FRAME_VERIFIER_GAVE_UP[:] = [{"index": 5, "kind": "video"}]
        snap = ps._slot_miss_snapshot(5)
        assert ps.FRAME_VERIFIER_GAVE_UP == []
        assert snap["frame_verifier"] == [{"index": 5, "kind": "video"}]
        ps._slot_miss_restore(snap)
        assert ps.FRAME_VERIFIER_GAVE_UP == [{"index": 5, "kind": "video"}]
        ps.FRAME_VERIFIER_GAVE_UP.clear()

    def test_rescue_success_does_not_bring_the_old_verdict_back(self):
        """Спасение удалось -> откат не делается -> старый вердикт про
        видео не всплывает на слоте, где теперь стоит фотография."""
        ps.FRAME_VERIFIER_GAVE_UP[:] = [{"index": 5, "kind": "video"}]
        ps._slot_miss_snapshot(5)   # спасение удалось, restore НЕ вызываем
        assert ps.FRAME_VERIFIER_GAVE_UP == []
        assert ps._slot_known_bad_reason(5) is None


class TestPhotoPickStaysInSyncAcrossFrameVerifyRepicks:
    """РЕАЛЬНЫЙ найденный баг (19.09, поймано прямым сравнением байтов
    файла с его же sidecar на живом прогоне): `pick` в pexels_photo()
    синхронизировалась с `winner` на КАЖДОЙ итерации цикла резкости
    (`pick = winner["p"]` внутри `while True`), но цикл зрячего гейта
    переставляет `winner` и перекачивает `cf` под НОВОГО победителя, ни
    разу не трогая `pick`. Если гейт хоть раз отклонил кандидата, `pick`
    замирает на ПЕРВОМ (уже отвергнутом) кандидате, а сам файл на диске
    (`cf`) и `winner` в памяти — на последнем принятом.

    Всё, что ниже читает `pick` — dedup `used_ids`, `pexels_id`/
    `candidate_text` в sidecar, кредит источника (`_source_bump`),
    провенанс/лицензия — после этого называет ДРУГОГО кандидата, чем тот,
    что реально попал в кадр. Живой пример: meta.json называл кандидата
    "gauntlets holding a sword" (id 349455, первая попытка), а сам jpg на
    диске оказался рукописной миниатюрой битвы (openverse, кандидат после
    третьего переподбора зрячим гейтом)."""

    def test_sidecar_and_dedup_reference_the_final_accepted_candidate(
            self, tmp_path, monkeypatch):
        _stub_photo(monkeypatch, tmp_path)
        # Первые два кандидата (id 1, 2) отклоняются зрячим гейтом, третий
        # (id 3) принимается — ровно тот сценарий, где `pick` и `winner`
        # расходятся, если синхронизации нет.
        calls = {"n": 0}

        def fake_verify(path, text, video_folder, shot_brief=None):
            calls["n"] += 1
            if calls["n"] <= 2:
                return {"verdict": "no", "seen": "не то", "missing": "кинжал"}
            return {"verdict": "yes", "seen": "кинжал", "missing": ""}

        monkeypatch.setattr(frame_verifier, "enabled", lambda: True)
        monkeypatch.setattr(frame_verifier, "verify", fake_verify)
        used_ids = set()
        out = ps.pexels_photo(
            "dagger", 3, used_ids=used_ids, used_hashes=[],
            block_text="Вот кинжал.")
        assert out is not None
        assert calls["n"] == 3, "гейт обязан был отклонить дважды и принять на третьей попытке"

        import json
        meta = json.loads(open(out + ".meta.json").read())
        # Реальный найденный дефект: без синхронизации здесь стоял бы id
        # ПЕРВОГО (уже отвергнутого) кандидата, а used_ids не содержал бы
        # id реально показанного кадра вовсе.
        assert meta["pexels_id"] == 3, (
            f"sidecar называет кандидата {meta['pexels_id']!r}, а принят был id=3 — "
            "pick разошёлся с winner после переподбора зрячим гейтом"
        )
        assert 3 in used_ids, "дедуп обязан видеть РЕАЛЬНО показанный кадр"
        assert 1 not in used_ids, "дедуп не должен помнить отвергнутого кандидата как показанного"
