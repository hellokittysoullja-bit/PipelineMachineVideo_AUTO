"""Вклад каждого источника кандидатов — измеряется, а не предполагается.

Три реальных случая, когда источник давал НОЛЬ и об этом никто не знал:
Openverse отвечал 401 и fail-open глотал это молча (поймано вживую 13.09);
`if not PEXELS_API_KEY: return None` стоял ДО сборки пула, и без ключа Pexels
умирали и музеи, и Openverse, и Pixabay, и Unsplash (найдено 13.09 при
попытке измерить отбор без ключа: все слоты вернули None за 0 секунд);
до 07.09 Openverse/Pixabay/Unsplash не вызывались вовсе. Во всех трёх
случаях ролик выглядел штатно собранным.

Здесь запирается: (1) ключ Pexels гейтит только Pexels; (2) счёт
предложено/рассмотрено/прошло/выиграло/ошибок ведётся по источникам;
(3) ошибка поиска источника видна, а не проглочена; (4) сводка пишется.
"""
import io
import json
import os
import sys
import tempfile
import urllib.error

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

_TMP_VIDEO_DIR = tempfile.mkdtemp(prefix="srcstats_")
sys.argv = ["pipeline_smart.py", _TMP_VIDEO_DIR]
import pipeline_smart as ps  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_stats(monkeypatch, tmp_path):
    # Рабочая папка задаётся ФИКСТУРОЙ, а не подменой sys.argv на импорте:
    # VIDEO_FOLDER/TEMP_FOLDER считаются один раз на импорте модуля, и
    # подмена действует ровно до тех пор, пока ЭТОТ файл импортирует
    # pipeline_smart первым. Стоит запустить его после соседнего файла —
    # и TEMP_FOLDER оказывается путём к чужому .py (измерено 15.09).
    monkeypatch.setattr(ps, "VIDEO_FOLDER", str(tmp_path))
    monkeypatch.setattr(ps, "TEMP_FOLDER", str(tmp_path / "temp_smart"))
    ps.reset_source_stats()
    yield
    ps.reset_source_stats()


def _jpeg(path, colour=(120, 80, 60)):
    from PIL import Image
    Image.new("RGB", (640, 360), colour).save(path, "JPEG")
    return path


class TestCandidateSource:
    @pytest.mark.parametrize("cid,expected", [
        ("met:32684", "met"), ("chicago:7", "chicago"), ("cleveland:1", "cleveland"),
        ("openverse:e8b7-aaaa", "openverse"), ("pixabay:55", "pixabay"),
        ("unsplash:abc", "unsplash"), (33508363, "pexels"), ("33508363", "pexels"),
    ])
    def test_prefix_decides_source(self, cid, expected):
        assert ps.candidate_source({"id": cid}) == expected
        assert ps.candidate_source(cid) == expected


class TestPexelsKeyGatesOnlyPexels:
    def test_search_without_key_makes_no_request_and_returns_empty(self, monkeypatch):
        monkeypatch.setattr(ps, "PEXELS_API_KEY", "")
        ps._PEXELS_SEARCH_CACHE.clear()
        ps._PEXELS_VIDEO_SEARCH_CACHE.clear()
        monkeypatch.setattr(ps.urllib.request, "urlopen",
                            lambda *a, **k: (_ for _ in ()).throw(AssertionError("сеть тронута")))
        assert ps._pexels_search_photos("medieval sword") == []
        assert ps._pexels_search_videos("medieval sword") == []

    def test_museum_candidate_wins_without_pexels_key(self, monkeypatch, tmp_path):
        """Главный инвариант: без ключа Pexels отбор ДОХОДИТ до музеев и
        возвращает их кандидата. Раньше — None на входе в функцию."""
        img = _jpeg(str(tmp_path / "met.jpg"))
        cand = {"id": "met:1", "alt": "Rondel Dagger", "url": "http://met/1",
                "src": {"large2x": "file://" + img}}
        monkeypatch.setattr(ps, "PEXELS_API_KEY", "")
        monkeypatch.setattr(ps, "_museum_search_photos", lambda q, department=None: [dict(cand)])
        monkeypatch.setattr(ps, "_openverse_search_photos", lambda q: [])
        monkeypatch.setattr(ps, "_pixabay_search_photos", lambda q: [])
        monkeypatch.setattr(ps, "_unsplash_search_photos", lambda q: [])
        monkeypatch.setattr(ps, "filter_alt_blocklist", lambda items: items)
        ps._PEXELS_SEARCH_CACHE.clear()
        out = ps.pexels_photo("medieval rondel dagger", 0, used_ids=set(), used_hashes=None,
                              text_key="t-no-key")
        assert out is not None and os.path.exists(out) and os.path.getsize(out) > 0
        st = ps.SOURCE_STATS["met"]
        assert st["offered"] == 1 and st["won"] == 1
        assert "pexels" not in ps.SOURCE_STATS or ps.SOURCE_STATS["pexels"]["offered"] == 0


class TestSearchErrorsAreVisible:
    def test_openverse_failure_is_counted_not_swallowed(self, monkeypatch, capsys):
        monkeypatch.setattr(ps.feature_flags, "enabled",
                            lambda name, *a, **k: True if name == "OPENVERSE_ENABLED" else False)
        ps._OPENVERSE_SEARCH_CACHE.clear()
        monkeypatch.setattr(ps, "_openverse_fetch_one",
                            lambda q, ov: (_ for _ in ()).throw(
                                urllib.error.HTTPError("u", 401, "Unauthorized", {}, io.BytesIO(b""))))
        assert ps._openverse_search_photos("medieval sword") == []
        assert ps.SOURCE_STATS["openverse"]["search_errors"] == 1
        out = capsys.readouterr().out
        assert "openverse" in out and "401" in out

    def test_error_line_is_printed_once_per_source(self, monkeypatch, capsys):
        for i in range(3):
            ps._note_source_search_error("pixabay", RuntimeError("x"), f"q{i}")
        assert ps.SOURCE_STATS["pixabay"]["search_errors"] == 3
        assert capsys.readouterr().out.count("pixabay: поиск не отвечает") == 1


class TestReport:
    def test_report_written_even_when_empty(self, tmp_path):
        path = ps.write_source_contribution(str(tmp_path))
        data = json.load(open(path, encoding="utf-8"))
        assert data["sources"] == {} and data["total_won"] == 0

    def test_silent_source_is_named(self, tmp_path):
        ps._source_bump("met", "offered", 5)
        ps._source_bump("met", "won", 2)
        ps._source_bump("openverse", "search_errors", 4)
        path = ps.write_source_contribution(str(tmp_path))
        data = json.load(open(path, encoding="utf-8"))
        assert data["silent_sources"] == ["openverse"]
        assert data["sources"]["met"]["won"] == 2 and data["total_won"] == 2
        assert data["museum_fetch"] is not None

    def test_reset_clears_everything(self):
        ps._source_bump("met", "offered")
        ps._note_source_search_error("met", RuntimeError("x"))
        ps.reset_source_stats()
        assert ps.SOURCE_STATS == {} and not ps._SOURCE_ERROR_PRINTED
