# -*- coding: utf-8 -*-
"""Откуда пришёл кадр и с каким числом он выиграл — в отчёте, а не только в кэше.

Измерено 15.09 на videos/_test60s (первый полный прогон со всеми источниками):

    source_contribution.json   pexels 5 · met 2 · pixabay 2 · cleveland 1
    shotlist.json              source: "pexels" × 10

То есть ярлык источника был неверен на ПОЛОВИНЕ слотов. Данные при этом не
терялись: настоящий источник лежал в sidecar соседним файлом
(pexels_id="pixabay:2565957"), его просто никто не спрашивал —
shotlist_source_for() возвращала литерал "pexels" для всего, что не в media/.

Вторая половина той же дыры: у ВИДЕО в sidecar стояло relevance: null даже
при chosen_by="video_relevance_best" — число, которым гейт принял решение,
не сохранялось нигде, и видео-слоты (половина эпизода) были неаудируемы.

Почему это не косметика: разметка эпизода владельцем — единственный сегодня
способ откалибровать отбор (см. золотой набор), и без ответа «откуда кадр»
она не отвечает на главный вопрос — какой источник даёт годное, а какой брак.
"""
import json
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)
sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]

import pipeline_smart as ps  # noqa: E402


def _with_sidecar(tmp_path, payload, name="a.jpg"):
    media = tmp_path / name
    media.write_bytes(b"x")
    (tmp_path / (name + ".meta.json")).write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return str(media)


class TestProvenanceComesFromTheSidecar:
    def test_real_payload_of_the_measured_run(self, tmp_path):
        """Ровно тот sidecar, что лежал у слота 0 эпизода _test60s."""
        p = _with_sidecar(tmp_path, {
            "pexels_id": "pixabay:2565957",
            "query": "medieval knight plate armour closeup",
            "kind": "photo", "ahash": "1111", "relevance": 0.2509435713291168,
            "chosen_by": "director", "written_at": 1789471825.58,
        })
        got = ps.shotlist_provenance(p)
        assert got["provider"] == "pixabay"
        assert got["candidate_id"] == "pixabay:2565957"
        assert got["relevance"] == pytest.approx(0.2509, abs=1e-4)
        assert got["chosen_by"] == "director"

    @pytest.mark.parametrize("cid,provider", [
        ("met:32684", "met"), ("cleveland:1234", "cleveland"),
        ("chicago:77", "chicago"), ("openverse:e8b7", "openverse"),
        ("euro:2048128/x", "euro"),          # Europeana — префикс именно "euro"
        ("unsplash:abc", "unsplash"), (33508363, "pexels"), ("33508363", "pexels"),
    ])
    def test_provider_matches_the_contribution_counter(self, tmp_path, cid, provider):
        """Та же функция, что считает вклад источников — не вторая копия
        разбора префикса: разойдясь, они дали бы два разных ответа на вопрос
        «кто принёс этот кадр» в двух отчётах одного прогона."""
        p = _with_sidecar(tmp_path, {"pexels_id": cid})
        assert ps.shotlist_provenance(p)["provider"] == provider


class TestFailOpenNeverCostsASlot:
    def test_no_sidecar_is_empty_not_a_guess(self, tmp_path):
        media = tmp_path / "b.jpg"
        media.write_bytes(b"x")
        assert ps.shotlist_provenance(str(media)) == {}

    def test_broken_json_is_empty(self, tmp_path):
        media = tmp_path / "c.jpg"
        media.write_bytes(b"x")
        (tmp_path / "c.jpg.meta.json").write_text("{ не json", encoding="utf-8")
        assert ps.shotlist_provenance(str(media)) == {}

    def test_non_dict_json_is_empty(self, tmp_path):
        media = tmp_path / "d.jpg"
        media.write_bytes(b"x")
        (tmp_path / "d.jpg.meta.json").write_text("[1, 2]", encoding="utf-8")
        assert ps.shotlist_provenance(str(media)) == {}

    def test_missing_path_is_empty(self):
        assert ps.shotlist_provenance(None) == {}

    def test_null_fields_do_not_become_keys(self, tmp_path):
        """relevance: null у старого кадра не должен печататься на плитке как
        «rel None» — поля просто нет."""
        p = _with_sidecar(tmp_path, {"pexels_id": 42, "relevance": None,
                                     "chosen_by": None})
        got = ps.shotlist_provenance(p)
        assert "relevance" not in got and "chosen_by" not in got
        assert got["provider"] == "pexels"


class TestTheTwoAxesAreNotConfused:
    """`source` — КАК слот разрешён, `provider` — КТО принёс кадр. Смешение
    этих осей и было исходным дефектом."""

    def test_picked_no_longer_claims_to_be_pexels(self, tmp_path):
        vd = str(tmp_path)
        p = os.path.join(vd, "temp_smart", "pexels_cache", "a.jpg")
        assert ps.shotlist_source_for(p, vd) == "picked"

    def test_resolution_path_values_are_unchanged(self, tmp_path):
        vd = str(tmp_path)
        assert ps.shotlist_source_for(os.path.join(vd, "media", "1.jpg"), vd) == "local"
        assert ps.shotlist_source_for(os.path.join(vd, "media", "1.jpg"), vd,
                                      locked=True) == "shotlist_lock"
        assert ps.shotlist_source_for(None, vd) == "missing"


class TestBothRecordSitesCarryIt:
    """Негативный контроль самой правки: убрать любую из двух подстановок —
    и соответствующий тест падает."""

    def _src(self):
        return open(os.path.join(SCRIPTS_DIR, "pipeline_smart.py"),
                    encoding="utf-8").read()

    def test_normal_pick_site(self):
        src = self._src()
        block = src[src.index('"source": shotlist_source_for(video or photo'):]
        block = block[:block.index("recent_media_types.append")]
        assert "**shotlist_provenance(video or photo)" in block

    def test_cache_hit_site_reads_the_file_not_the_old_entry(self):
        """Прошлая запись шотлиста могла быть сделана до этой правки и
        провенанса не содержать вовсе; sidecar живёт рядом с файлом и
        кэш-хит переживает — спрашивать надо его."""
        src = self._src()
        block = src[src.index('"source": "shotlist_lock" if (lock_photo or lock_video)'):]
        block = block[:block.index("prev_file = shotlist_resolve_file")]
        assert "shotlist_provenance(" in block
        assert "prev_shot.get(\"file\")" in block


class TestVideoRelevanceIsRecorded:
    """До правки relevance у видео было null на ВСЕХ выходах, включая
    chosen_by="video_relevance_best"."""

    def _src(self):
        return open(os.path.join(SCRIPTS_DIR, "pipeline_smart.py"),
                    encoding="utf-8").read()

    def test_winner_writes_its_relevance(self):
        # Якорь — сам ВЫЗОВ, а не строка "video_relevance_best": она же
        # встречается в комментарии выше по файлу, и первая версия этого
        # теста проверяла комментарий вместо кода.
        src = self._src()
        block = src[src.index("write_media_sidecar(cf, pexels_id=best[3]"):]
        block = block[:block.index(")\n")]
        assert "relevance=best_rel" in block
        assert 'chosen_by="video_relevance_best"' in block

    def test_fallbacks_write_theirs(self):
        src = self._src()
        block = src[src.index("write_media_sidecar(\n                cf, pexels_id=vid"):]
        block = block[:block.index("))\n")]
        assert "relevance=chosen_rel" in block

    def test_gate_receives_the_precomputed_value_no_second_forward(self):
        """is_relevant_candidate() принимает готовое число именно для этого
        случая — считать его вторым вызовом значило бы платить лишним
        прогоном модели на каждого кандидата."""
        src = self._src()
        assert "is_relevant_candidate(probe, query, relevance=cand_rel)" in src

    def test_relevance_is_keyed_by_path_not_by_tuple_position(self):
        """Кортеж `good` разбирается по позиции в трёх местах — восьмой
        элемент молча перепутал бы путь/id/hash местами."""
        src = self._src()
        assert "cand_relevance[trial] = cand_rel" in src
        assert "cand_relevance.get(best[2])" in src


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(REPO_ROOT, "videos", "_test60s", "media_plan")),
    reason="эпизод не в git (videos/ в .gitignore) — тест для машины владельца")
class TestAgainstTheRealEpisode:
    def test_sidecars_of_the_chosen_files_match_the_contribution_counter(self):
        """Две НЕЗАВИСИМЫЕ записи одного прогона обязаны сойтись.

        Счётчик вкладов пишется в момент победы кандидата, sidecar — в
        момент записи файла в кэш. Сверяются именно они, а не шотлист с
        самим собой: равенство `shotlist.by_provider == won` верно только
        на прогоне, который реально ВЁЛ отбор — на повторном прогоне с
        прогретым кэшем клипов ни один кандидат не побеждает, счётчик
        пуст, а провенанс у файлов остаётся. Сравнивать их там значило бы
        написать заведомо ложное ожидание.
        """
        mp = os.path.join(REPO_ROOT, "videos", "_test60s", "media_plan")
        vd = os.path.join(REPO_ROOT, "videos", "_test60s")
        shots = json.load(open(os.path.join(mp, "shotlist.json"),
                               encoding="utf-8"))["shots"]
        contrib = json.load(open(os.path.join(mp, "source_contribution.json"),
                                 encoding="utf-8"))["sources"]
        won = {k: v["won"] for k, v in contrib.items() if v["won"]}
        by_provider = {}
        for s in shots:
            f = ps.shotlist_resolve_file(s.get("file"), vd)
            prov = ps.shotlist_provenance(f).get("provider")
            if prov:
                by_provider[prov] = by_provider.get(prov, 0) + 1
        assert by_provider == won, (by_provider, won)
        # Тот самый факт, из-за которого правка и написана: победителей
        # больше одного, а прежний ярлык называл их всех "pexels".
        assert len(by_provider) > 1
