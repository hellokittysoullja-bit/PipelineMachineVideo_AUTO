"""Происхождение музейного/архивного кадра доживает до готового ролика.

РЕАЛЬНАЯ дыра, найденная разбором 14.09: museum_sources кладёт в кандидата
`_museum_meta`, а _openverse_search_photos — `_openverse_meta`, и прямой
grep по scripts/ показывал, что оба поля СТАВЯТСЯ и не читаются НИ ОДНОЙ
строкой. Комментарий у второго прямо обещал «пригодится, если кандидат
победит и понадобится атрибуция/аудит источника» — забирать было некому.

Юридический след существовал только в stock_fetch_multisource.py
(openverse_license_manifest.jsonl, Шаг 4, ручной мультисток), то есть НЕ на
том пути, которым эти кандидаты реально попадают в эпизод. Нарушения
лицензии тут нет — берётся только CC0 и public domain, атрибуция не
требуется, — но по готовому ролику нельзя было ответить, из какого архива
пришёл конкретный кадр.

Тот же класс, что уже ловили у Openverse, Pixabay, Unsplash и
reveal-акцентов, только про ДАННЫЕ, а не про функцию: test_no_dead_layers
строит граф по функциям и такое увидеть не может.
"""
import json
import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]
import pipeline_smart as ps  # noqa: E402


MUSEUM = {"id": "met:32684", "alt": "Rondel Dagger", "url": "https://met/32684",
          "_museum_meta": {"museum": "met", "culture": "French",
                           "era_from": 1450, "era_to": 1500,
                           "license": "public_domain"}}
OPENVERSE = {"id": "openverse:abc", "alt": "Medieval map", "url": "https://ov/abc",
             "_openverse_meta": {"creator": "Lloyd", "license": "cc0",
                                 "source": "wikimedia"}}
PEXELS = {"id": 12345, "alt": "knight", "url": "https://pexels/12345"}


class TestProvenanceExtraction:
    def test_museum_candidate_keeps_its_passport(self):
        p = ps.candidate_provenance(MUSEUM)
        assert p["id"] == "met:32684"
        assert p["culture"] == "French" and p["era_from"] == 1450
        assert p["license"] == "public_domain"

    def test_openverse_candidate_keeps_licence_and_source(self):
        p = ps.candidate_provenance(OPENVERSE)
        assert p["license"] == "cc0" and p["source"] == "wikimedia"

    def test_plain_stock_has_no_provenance(self):
        """У Pexels/Pixabay/Unsplash паспорта предмета нет — журнал не должен
        засоряться пустыми строками про обычный сток."""
        assert ps.candidate_provenance(PEXELS) is None
        assert ps.candidate_provenance(None) is None


class TestEpisodeLicenceJournal:
    def test_winner_is_written_as_one_json_line(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ps, "VIDEO_FOLDER", str(tmp_path))
        monkeypatch.setattr(ps, "_LICENSE_MANIFEST_SEEN", set())
        ps.log_candidate_license(ps.candidate_provenance(MUSEUM), "medieval dagger")
        path = tmp_path / "media_plan" / "source_license_manifest.jsonl"
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 1
        assert rows[0]["id"] == "met:32684"
        assert rows[0]["query"] == "medieval dagger"

    def test_journal_appends_and_never_rewrites(self, tmp_path, monkeypatch):
        """Перезапись целиком повторила бы дыру merge_slot_report: на
        частичном ре-рендере журнал остался бы почти пустым, хотя кадры в
        ролике те же."""
        monkeypatch.setattr(ps, "VIDEO_FOLDER", str(tmp_path))
        monkeypatch.setattr(ps, "_LICENSE_MANIFEST_SEEN", set())
        ps.log_candidate_license(ps.candidate_provenance(MUSEUM), "q1")
        ps.log_candidate_license(ps.candidate_provenance(OPENVERSE), "q2")
        path = tmp_path / "media_plan" / "source_license_manifest.jsonl"
        assert len(path.read_text(encoding="utf-8").splitlines()) == 2

    def test_same_candidate_is_not_logged_twice_in_one_run(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ps, "VIDEO_FOLDER", str(tmp_path))
        monkeypatch.setattr(ps, "_LICENSE_MANIFEST_SEEN", set())
        for _ in range(3):
            ps.log_candidate_license(ps.candidate_provenance(MUSEUM), "q")
        path = tmp_path / "media_plan" / "source_license_manifest.jsonl"
        assert len(path.read_text(encoding="utf-8").splitlines()) == 1

    def test_unwritable_journal_never_breaks_selection(self, tmp_path, monkeypatch):
        """Потерять кадр из-за строчки в журнале хуже, чем потерять строчку."""
        monkeypatch.setattr(ps, "VIDEO_FOLDER", "/proc/nonexistent/nowhere")
        monkeypatch.setattr(ps, "_LICENSE_MANIFEST_SEEN", set())
        ps.log_candidate_license(ps.candidate_provenance(MUSEUM), "q")


class TestSidecarSurvivesCacheHit:
    def test_provenance_is_stored_next_to_the_file(self, tmp_path):
        media = tmp_path / "x.jpg"
        media.write_bytes(b"x")
        ps.write_media_sidecar(str(media), pexels_id="met:32684", query="q",
                               kind="photo",
                               provenance=ps.candidate_provenance(MUSEUM))
        got = ps.read_media_sidecar(str(media))
        assert got["provenance"]["culture"] == "French"

    def test_plain_stock_sidecar_has_no_provenance_key(self, tmp_path):
        media = tmp_path / "y.jpg"
        media.write_bytes(b"y")
        ps.write_media_sidecar(str(media), pexels_id=1, query="q", kind="photo")
        assert "provenance" not in ps.read_media_sidecar(str(media))


def test_photo_winner_actually_calls_the_journal():
    """Source-level: без этого вызова обе функции выше снова стали бы
    написанными и никем не вызванными — ровно то, что здесь чинится."""
    src = open(os.path.join(REPO_ROOT, "scripts", "pipeline_smart.py"),
               encoding="utf-8").read()
    start = src.index("class PhotoAdapter(")
    block = src[start:src.index("\ndef ", start + 10)]
    # Строка журнала — эффект попытки: пишется коммитом, только если кадр
    # встал на экран (выброшенный кадр в журнал лицензий не попадает).
    assert 'record_effect("license", _prov, query)' in block
    assert "provenance=_prov" in block


def test_every_museum_records_the_field_it_was_admitted_by():
    """Журнал лицензий обязан называть и лицензию, и поле источника, по
    которому предмет в него попал — иначе постфактум нельзя отличить
    «музей пометил как public domain» от «мы так решили»."""
    import museum_sources as ms
    got = {}
    for fn, src in ((ms.search_met, "met"), (ms.search_cleveland, "cleveland"),
                    (ms.search_chicago, "chicago")):
        import inspect
        body = inspect.getsource(fn)
        assert '"license"' in body and '"license_field"' in body, src
        got[src] = True
    assert len(got) == 3
