"""Прямые API музеев: эпоха кандидата ИЗВЕСТНА, а не угадана по пикселям.

Почему источник появился (разбор 10.09, эпизод 02_ne-mechom). В ролик про
Азенкур попали танк на кульминации «Рыцарей убивала земля», терракотовая
армия, египетский саркофаг, космонавт и кавалерия XIX века. Замер показал,
что запросы слотов были правильные («medieval rondel dagger»), а пустым был
пул: у Pexels средневековья нет, и на запрос без совпадений он отдаёт
ближайшее по вектору.

Три попытки отличить эпоху ПО КАРТИНКЕ провалились — последняя (разбиение по
эпохе) не прошла золотой набор: настоящее макро доспеха выглядит «не
средневековым» сильнее танка, а брак с современной улицей выглядит
«средневековым» из-за костюма рыцаря в кадре. Здесь угадывать не нужно:
музей отдаёт паспорт предмета (дата + культура) вместе со снимком.

Живой замер на восьми сломанных запросах эпизода: 128 кандидатов, среди них
«Two-Handed Sword, 1375-1475, Western European», «Sallet in the Shape of a
Lion's Head, 1450-1505, Italian», «Tomb of Ermengol X, 1297-1353, Catalan».
Сеть в тестах не трогается — ответы API подставляются.
"""
import json
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

import museum_sources as ms  # noqa: E402

sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]
import pipeline_smart as ps  # noqa: E402


class TestEraWindow:
    """Дата предмета — главный фильтр: и танк (XX век), и терракотовая армия
    (III век до н.э.), и саркофаг отсеиваются ею, не доходя до пула."""

    def test_medieval_object_passes(self):
        assert ms.era_overlaps(1300, 1375)

    def test_japanese_18th_century_armor_is_rejected(self):
        """Реальный кандидат из выдачи Мет по запросу 'plate armor'."""
        assert not ms.era_overlaps(1701, 1800)

    def test_19th_century_indian_shield_is_rejected(self):
        assert not ms.era_overlaps(1800, 1850)

    def test_partial_overlap_is_enough(self):
        """'Armor, 1375-1950' — итальянский доспех XIV века, часть которого
        реставрирована в XX. Требовать полного вхождения значило бы
        выбрасывать подлинники."""
        assert ms.era_overlaps(1375, 1950)

    def test_missing_or_zero_dates_are_rejected(self):
        assert not ms.era_overlaps(None, None)
        assert not ms.era_overlaps(0, 0)
        assert not ms.era_overlaps("", "")


class TestCultureFilter:
    """Дата спасает не всегда: 'Iranian or Turkish, 1401-1600' проходит по
    эпохе и обязано отсекаться по культуре — иначе восточный доспех XV века
    попадёт в ролик про Азенкур как «эпоха совпала»."""

    def test_european_cultures_pass(self):
        for c in ("Italian", "German, Nuremberg", "British or Western European",
                  "Catalan", "French", "probably Flemish"):
            assert not ms.culture_is_foreign(c), c

    def test_foreign_cultures_are_rejected(self):
        for c in ("Japanese, Toyohara", "Tibetan, and possibly Bhutanese",
                  "North Indian", "Iranian or Turkish", "Chinese", "Egyptian"):
            assert ms.culture_is_foreign(c), c

    def test_checks_every_passport_field(self):
        assert ms.culture_is_foreign(None, "Japan", None)


def _fake_urls(mapping):
    """Подменяет _get_json: {подстрока_url: ответ}."""
    def fake(url):
        for key, payload in mapping.items():
            if key in url:
                return payload
        raise AssertionError(f"неожиданный URL в тесте: {url}")
    return fake


MET_OBJECT_OK = {
    "isPublicDomain": True, "primaryImage": "https://img/full.jpg",
    "primaryImageSmall": "https://img/small.jpg", "title": "Bascinet",
    "objectBeginDate": 1300, "objectEndDate": 1375, "culture": "possibly Italian",
    "country": "possibly Italy", "department": "Arms and Armor",
    "objectURL": "https://met/23239",
}


class TestMet:
    def test_returns_pexels_shaped_candidate(self, monkeypatch):
        monkeypatch.setattr(ms, "_get_json", _fake_urls({
            "/search": {"objectIDs": [23239]}, "/objects/": MET_OBJECT_OK}))
        (c,) = ms.search_met("bascinet")
        assert c["id"] == "met:23239"
        assert c["alt"] == "Bascinet"
        assert c["src"]["large2x"] == "https://img/full.jpg"
        assert c["_museum_meta"]["culture"] == "possibly Italian"

    def test_prefers_full_resolution_over_web_version(self, monkeypatch):
        """Замер: 1.4-2 МБ (3171x4000) против 55-84 КБ (495x624). Кадр идёт
        в 1920x1080 и ещё проходит зум Ken Burns — на web-версии это мыло."""
        monkeypatch.setattr(ms, "_get_json", _fake_urls({
            "/search": {"objectIDs": [1]}, "/objects/": MET_OBJECT_OK}))
        (c,) = ms.search_met("bascinet")
        assert c["src"]["large2x"] == "https://img/full.jpg"

    def test_non_public_domain_object_is_skipped(self, monkeypatch):
        obj = dict(MET_OBJECT_OK, isPublicDomain=False)
        monkeypatch.setattr(ms, "_get_json", _fake_urls({
            "/search": {"objectIDs": [1]}, "/objects/": obj}))
        assert ms.search_met("bascinet") == []

    def test_out_of_era_object_is_skipped(self, monkeypatch):
        obj = dict(MET_OBJECT_OK, objectBeginDate=1701, objectEndDate=1800)
        monkeypatch.setattr(ms, "_get_json", _fake_urls({
            "/search": {"objectIDs": [1]}, "/objects/": obj}))
        assert ms.search_met("plate armor") == []

    def test_foreign_culture_object_is_skipped(self, monkeypatch):
        obj = dict(MET_OBJECT_OK, culture="Japanese", country="Japan")
        monkeypatch.setattr(ms, "_get_json", _fake_urls({
            "/search": {"objectIDs": [1]}, "/objects/": obj}))
        assert ms.search_met("armor") == []


class TestCleveland:
    def test_cc0_only(self, monkeypatch):
        base = {"id": 1, "title": "Armor", "creation_date_earliest": 1500,
                "creation_date_latest": 1550, "culture": ["North Italy"],
                "images": {"print": {"url": "https://cle/print.jpg"}}}
        monkeypatch.setattr(ms, "_get_json", _fake_urls({
            "artworks": {"data": [dict(base, share_license_status="CC BY")]}}))
        assert ms.search_cleveland("armor") == []

    def test_prefers_print_over_web(self, monkeypatch):
        """'full' сознательно не берётся — это TIFF на 60+ МБ."""
        monkeypatch.setattr(ms, "_get_json", _fake_urls({"artworks": {"data": [{
            "id": 1, "title": "Armor", "share_license_status": "CC0",
            "creation_date_earliest": 1500, "creation_date_latest": 1550,
            "culture": ["North Italy"],
            "images": {"web": {"url": "https://cle/web.jpg"},
                       "print": {"url": "https://cle/print.jpg"},
                       "full": {"url": "https://cle/full.tif"}}}]}}))
        (c,) = ms.search_cleveland("armor")
        assert c["src"]["large2x"] == "https://cle/print.jpg"


class TestChicago:
    def _payload(self, **over):
        art = {"id": 7, "title": "Initial G", "date_start": 1301,
               "date_end": 1400, "place_of_origin": "Europe",
               "is_public_domain": True, "image_id": "abc"}
        art.update(over)
        return {"config": {"iiif_url": "https://iiif"}, "data": [art]}

    def test_image_headers_travel_with_the_candidate(self, monkeypatch):
        """У Института искусств IIIF отвечает 403 на ЛЮБУЮ ширину без их
        AIC-User-Agent (проверено на 843, 1920 и full/full)."""
        monkeypatch.setattr(ms, "_get_json", _fake_urls({"artworks": self._payload()}))
        (c,) = ms.search_chicago("manuscript")
        assert "AIC-User-Agent" in c["_download_headers"]

    def test_asks_for_full_width(self, monkeypatch):
        monkeypatch.setattr(ms, "_get_json", _fake_urls({"artworks": self._payload()}))
        (c,) = ms.search_chicago("manuscript")
        assert "/full/1920," in c["src"]["large2x"]

    def test_out_of_era_is_skipped(self, monkeypatch):
        monkeypatch.setattr(ms, "_get_json", _fake_urls({
            "artworks": self._payload(date_start=1850, date_end=1900)}))
        assert ms.search_chicago("armor") == []


class TestFailOpen:
    def test_one_broken_museum_does_not_take_down_the_others(self, monkeypatch):
        monkeypatch.setattr(ms, "search_met",
                            lambda q, **k: (_ for _ in ()).throw(OSError("сеть")))
        monkeypatch.setattr(ms, "search_cleveland", lambda q, **k: [{"id": "cleveland:1"}])
        monkeypatch.setattr(ms, "search_chicago", lambda q, **k: [])
        monkeypatch.setattr(ms, "_SOURCES", (("met", ms.search_met),
                                             ("cleveland", ms.search_cleveland),
                                             ("chicago", ms.search_chicago)))
        ms._SEARCH_CACHE.clear()
        monkeypatch.setattr(ms.feature_flags, "enabled", lambda *a, **k: True)
        assert ms.search_museums("q") == [{"id": "cleveland:1"}]

    def test_flag_off_makes_no_requests(self, monkeypatch):
        touched = []
        monkeypatch.setattr(ms, "_get_json", lambda u: touched.append(u))
        monkeypatch.setattr(ms.feature_flags, "enabled", lambda *a, **k: False)
        ms._SEARCH_CACHE.clear()
        assert ms.search_museums("medieval helmet") == []
        assert touched == []


class TestWiredIntoPipeline:
    def test_pipeline_wrapper_uses_the_query_cascade(self, monkeypatch):
        """Длинный запрос не находит ничего и в музейном поиске тоже — тот же
        каскад, что у архивов."""
        seen = []

        def fake(q):
            seen.append(q)
            return [{"id": "met:1"}] if q == "medieval helmet" else []

        monkeypatch.setattr(ms, "search_museums", fake)
        ps._MUSEUM_SEARCH_CACHE.clear()
        out = ps._museum_search_photos("medieval helmet lying dirt")
        assert out == [{"id": "met:1"}]
        assert seen == ["medieval helmet lying dirt", "medieval helmet"]

    def test_download_merges_candidate_headers(self):
        """Скачивающий код не знает про конкретные музеи — заголовки едут в
        самом кандидате."""
        src = open(os.path.join(SCRIPTS_DIR, "pipeline_smart.py"),
                   encoding="utf-8").read()
        assert '_download_headers' in src

    def test_source_is_part_of_the_selection_signature(self):
        src = open(os.path.join(SCRIPTS_DIR, "pipeline_smart.py"),
                   encoding="utf-8").read()
        start = src.index("def _selection_stack_signature")
        block = src[start:src.index("\ndef candidate_gate_signature", start)]
        assert "MUSEUM_SOURCES_ENABLED" in block
        assert "MUSEUM_SOURCES_VERSION" in block

    def test_conftest_forces_the_source_off_in_tests(self):
        """Дефолт реестра у него '1' — без гашения в conftest каждый тест,
        дошедший до сборки пула, ходил бы живьём в три музейных API."""
        assert os.environ.get("MUSEUM_SOURCES_ENABLED") == "0"


class TestVideoPhotoRescue:
    """Разбивка брака эпизода 02 по типу слота: среди ВИДЕО брак 63% (41 из
    65), среди фото — 23% (35 из 154). Причина структурная: видео-корпус
    Pexels на исторические темы тоньше фото-корпуса, а музейные API видео не
    отдают вообще. Поэтому заведомо негодное видео уступает место фотографии
    ДО карточки-фолбэка."""

    def test_snapshot_removes_verdicts_from_every_report(self):
        ps.RELEVANCE_GATE_MISSES[:] = [{"index": 5}, {"index": 7}]
        ps.STOCK_EXHAUSTED_MISSES[:] = [{"index": 5}]
        ps.ARBITER_REJECTED_ALL[:] = [{"index": 9}]
        snap = ps._slot_miss_snapshot(5)
        assert [m["index"] for m in ps.RELEVANCE_GATE_MISSES] == [7]
        assert ps.STOCK_EXHAUSTED_MISSES == []
        assert ps.ARBITER_REJECTED_ALL == [{"index": 9}]
        assert snap["relevance"] == [{"index": 5}]
        assert snap["stock"] == [{"index": 5}]

    def test_restore_puts_them_back_when_rescue_fails(self):
        """Не нашлось фото — кадр прежний, значит и вердикт о нём прежний."""
        ps.RELEVANCE_GATE_MISSES[:] = [{"index": 5}]
        ps.STOCK_EXHAUSTED_MISSES[:] = []
        ps.ARBITER_REJECTED_ALL[:] = []
        snap = ps._slot_miss_snapshot(5)
        assert ps.RELEVANCE_GATE_MISSES == []
        ps._slot_miss_restore(snap)
        assert ps.RELEVANCE_GATE_MISSES == [{"index": 5}]

    def test_rescued_slot_is_not_counted_as_shipped_bad(self):
        """Итоговая строка считает брак по этим спискам — вердикт про
        отвергнутое видео иначе висел бы на слоте, где стоит другой кадр."""
        ps.RELEVANCE_GATE_MISSES[:] = [{"index": 5}]
        ps.STOCK_EXHAUSTED_MISSES[:] = []
        ps.ARBITER_REJECTED_ALL[:] = []
        ps._slot_miss_snapshot(5)          # спасение удалось, откат не делаем
        known = ({m["index"] for m in ps.RELEVANCE_GATE_MISSES} |
                 {m["index"] for m in ps.STOCK_EXHAUSTED_MISSES} |
                 {m["index"] for m in ps.ARBITER_REJECTED_ALL})
        assert known == set()

    def test_rescue_flag_is_part_of_the_selection_signature(self):
        src = open(os.path.join(SCRIPTS_DIR, "pipeline_smart.py"),
                   encoding="utf-8").read()
        start = src.index("def _selection_stack_signature")
        block = src[start:src.index("\ndef candidate_gate_signature", start)]
        assert "VIDEO_PHOTO_RESCUE" in block

    def test_rescue_is_attempted_before_the_fallback_card(self):
        """Карточка — последний уровень лестницы, а не первый: подлинный
        кинжал 1450 года сильнее текстовой плашки."""
        src = open(os.path.join(SCRIPTS_DIR, "pipeline_smart.py"),
                   encoding="utf-8").read()
        assert src.index("VIDEO_PHOTO_RESCUE") < src.index("build_slot_fallback_card(i, b[\"text\"], bad_reason)")


class TestPexelsOutageDoesNotKillOtherSources:
    """Реальный катастрофический баг, найденный вживую 11.09 (эпизод 02):
    Pexels вернул 500 шесть раз подряд (внешний сбой), _note_pexels_failure()
    честно взвела PEXELS_BROKEN — и внешний `if ... and use_pexels:` в main()
    погасил вызов pexels_photo()/pexels_video() ЦЕЛИКОМ для ВСЕХ оставшихся
    215 из 219 слотов. Музеи и архивы живут ВНУТРИ этих же функций и от
    Pexels не зависят, но вызвать их было уже некому — весь поиск просто
    переставал запускаться. 215 слотов получили карточку "no_media_at_all",
    хотя архивы в тот момент отвечали (лог: десятки "Архивы: ... (20 канд.)"
    ДО обрыва, ноль после). Комментарий самой _note_pexels_failure()
    прямо обещает "одиночная ошибка стоит одному слоту, не эпизоду" —
    этот тест защищает именно это обещание."""

    def test_search_call_is_not_gated_by_use_pexels(self):
        src = open(os.path.join(SCRIPTS_DIR, "pipeline_smart.py"),
                   encoding="utf-8").read()
        marker = "is_opening_shot = (i == 0)"
        start = src.index(marker)
        # Первая строка `if` после этой точки — тот самый гейт на вызов
        # pexels_photo()/pexels_video(). Она не должна содержать use_pexels:
        # музеи/архивы внутри этих функций работают независимо от Pexels.
        block = src[start:start + 2000]
        gate_line = next(line for line in block.splitlines()
                         if line.strip().startswith("if not photo and not video"))
        assert "use_pexels" not in gate_line, (
            "Гейт вызова pexels_photo()/pexels_video() снова завязан на "
            "use_pexels — обрыв Pexels опять погасит музеи/архивы/Openverse "
            "для всего оставшегося эпизода, не только для Pexels")
