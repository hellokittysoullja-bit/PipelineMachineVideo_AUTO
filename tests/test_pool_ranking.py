"""Два измеренных регресса глубокого пула и их исправления (A/B, 13.09).

A/B на 9 реальных слотах эпизода 02 (тот же pexels_photo, те же гейты; без
ключа Pexels — пул из музеев и Openverse, то есть меряется ровно дельта):
старая глубина музеев дала 6 хороших кадров из 9, новая — 4 из 9. Ни одного
слота лучше. Две причины, обе структурные:

1. Список кандидатов запроса шёл «все музеи, потом Openverse, потом
   Pexels...». С глубиной 60-111 первые PHOTO_DEDUP_MAX_TRIES кандидатов
   пробной выборки оказывались все музейными: на «medieval castle moat
   water» фотография замка из Openverse (relevance 0.28) не рассматривалась
   вообще, победила рукопись (0.21) — единственный музейный кандидат,
   прошедший гейт.
2. Среди прошедших гейт relevance победителя выбирала ЭСТЕТИКА (relevance
   участвовала как 0/1). Глубокий пул = больше красивых, но не тех.
"""
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)
sys.argv = ["pipeline_smart.py", tempfile.mkdtemp(prefix="poolrank_")]
import pipeline_smart as ps  # noqa: E402


def _cand(cid, relevance, aesthetic, dup_free=1, size_ok=1, relevant=1, sharp=1):
    return {"path": f"/x/{cid}.jpg", "p": {"id": cid}, "is_dup_free": dup_free,
            "size_ok": size_ok, "is_relevant": relevant, "sharp_ok": sharp,
            "aesthetic_val": aesthetic, "luma_score": 0.0, "min_d": 99,
            "relevance": relevance}


class TestRelevanceBeforeAesthetics:
    def test_castle_photo_beats_prettier_manuscript(self):
        """Реальные числа слота 6: замок 0.282/эстетика 7.16 против рукописи
        0.204/6.12 — но и против ДРУГОГО замка 0.221/7.53: раньше побеждал
        самый красивый из прошедших, теперь самый по теме."""
        pool = [_cand("openverse:pretty-castle", 0.221, 7.53),
                _cand("openverse:castle", 0.282, 7.16),
                _cand("met:manuscript", 0.204, 6.12)]
        base, _ = ps._score_and_pick(pool)
        assert base["p"]["id"] == "openverse:castle"

    def test_within_one_bucket_aesthetics_still_decides(self):
        """Кинжалы 0.321/0.317: разница меньше корзины — красота не отменена,
        она подчинена смыслу."""
        pool = [_cand("met:dagger-plain", 0.321, 4.81), _cand("met:dagger-nice", 0.317, 5.09)]
        base, _ = ps._score_and_pick(pool)
        assert base["p"]["id"] == "met:dagger-nice"

    def test_gates_still_outrank_relevance(self):
        pool = [_cand("a", 0.40, 9.0, dup_free=0), _cand("b", 0.20, 3.0)]
        base, _ = ps._score_and_pick(pool)
        assert base["p"]["id"] == "b"

    def test_no_clip_means_previous_behaviour(self):
        """CLIP недоступен -> relevance None у всех -> корзина 0 -> решает
        эстетика, как и раньше, байт-в-байт."""
        pool = [_cand("a", None, 5.0), _cand("b", None, 6.0)]
        base, _ = ps._score_and_pick(pool)
        assert base["p"]["id"] == "b"

    def test_bucket_is_in_selection_signature(self):
        sig = ps._selection_stack_signature()
        assert str(ps.RELEVANCE_RANK_BUCKET) in sig
        assert str(ps.POOL_SOURCE_INTERLEAVE_VERSION) in sig


class TestSourceInterleave:
    def test_deep_museum_list_cannot_crowd_out_other_sources(self, monkeypatch, tmp_path):
        """60 музейных + 3 из Openverse: в первых 20 кандидатах обязаны быть
        ВСЕ три архивных, иначе они не попадают в пробную выборку вообще."""
        seen_order = []

        def fake_pick(candidates_info, director_score_fn=None):
            for c in candidates_info:
                seen_order.append(c["p"]["id"])
            return candidates_info[0], None

        # Запрос ПРЕДМЕТНЫЙ («armour»): маршрутизация по типу кадра
        # (shot_types) отправляет в музеи только предметные и
        # иллюстративные слоты, у сценического музейного пула не будет
        # вовсе — и тест про вытеснение источников проверял бы пустоту.
        met = [{"id": f"met:{i}", "alt": "x", "url": "u", "src": {"large2x": "file:///nonexistent"}}
               for i in range(60)]
        ov = [{"id": f"openverse:{i}", "alt": "x", "url": "u", "src": {"large2x": "file:///nonexistent"}}
              for i in range(3)]
        monkeypatch.setattr(ps, "PEXELS_API_KEY", "")
        monkeypatch.setattr(ps, "_museum_search_photos", lambda q, department=None: list(met))
        monkeypatch.setattr(ps, "_openverse_search_photos", lambda q: list(ov))
        monkeypatch.setattr(ps, "_pixabay_search_photos", lambda q: [])
        monkeypatch.setattr(ps, "_unsplash_search_photos", lambda q: [])
        monkeypatch.setattr(ps, "filter_alt_blocklist", lambda items: items)
        captured = {}

        def fake_download(*a, **k):
            raise OSError("нет сети")

        monkeypatch.setattr(ps, "atomic_url_download", fake_download)
        # Порядок пула виден по offered-счёту до скачивания: смотрим первые 20.
        ps.reset_source_stats()
        ps._PEXELS_SEARCH_CACHE.clear()
        order = []
        real_bump = ps._source_bump

        def spy_bump(source, field, n=1):
            if field == "offered":
                order.append(source)
            return real_bump(source, field, n)

        monkeypatch.setattr(ps, "_source_bump", spy_bump)
        ps.pexels_photo("medieval armour", 0, used_ids=set(), used_hashes=[], text_key="interleave")
        first20 = order[:20]
        assert first20.count("openverse") == 3, first20
        assert first20[:2] == ["met", "openverse"], first20

    def test_stock_sources_share_the_pool_evenly_when_museums_are_empty(self, monkeypatch, tmp_path):
        """ОБРАТНЫЙ случай к тесту выше — и он не был проверен НИ РАЗУ до
        этого коммита (внешний вопрос владельца: "он должен смотреть во всех
        источниках, а не где-то больше, где-то меньше", проверено, а не
        принято на слово). Для ниши без физических музейных предметов
        (content_world.py гейтит use_museum_sources, шелф/openverse
        естественно пустые) round-robin обязан честно делить пул между
        Pexels/Pixabay/Unsplash — тот же общий механизм (zip_longest по
        источникам), НИКАКОГО отдельного правила под "новую нишу" не
        заводится и не нужно: он симметричен по построению."""
        def mk(prefix, n):
            return [{"id": f"{prefix}:{i}", "alt": "x", "url": "u",
                      "src": {"large2x": "file:///nonexistent"}} for i in range(n)]

        monkeypatch.setattr(ps, "PEXELS_API_KEY", "fake")
        monkeypatch.setattr(ps, "_museum_search_photos", lambda q, department=None: [])
        monkeypatch.setattr(ps, "_openverse_search_photos", lambda q: [])
        monkeypatch.setattr(ps, "_shelf_search_photos", lambda q, brief=None: [])
        monkeypatch.setattr(ps, "_pexels_search_photos", lambda q: mk("pexels", 80))
        monkeypatch.setattr(ps, "_pixabay_search_photos", lambda q: mk("pixabay", 80))
        monkeypatch.setattr(ps, "_unsplash_search_photos", lambda q: mk("unsplash", 80))
        monkeypatch.setattr(ps, "filter_alt_blocklist", lambda items: items)

        def fake_download(*a, **k):
            raise OSError("нет сети")

        monkeypatch.setattr(ps, "atomic_url_download", fake_download)
        ps.reset_source_stats()
        ps._PEXELS_SEARCH_CACHE.clear()
        order = []
        real_bump = ps._source_bump

        def spy_bump(source, field, n=1):
            if field == "offered":
                order.append(source)
            return real_bump(source, field, n)

        monkeypatch.setattr(ps, "_source_bump", spy_bump)
        ps.pexels_photo("modern office laptop", 0, used_ids=set(), used_hashes=[],
                        text_key="reverse-interleave")
        first20 = order[:20]
        from collections import Counter
        counts = Counter(first20)
        # Ни один сток не должен получить меньше 6 из 20 (равная доля — 6.67)
        # и ни один — больше 8: позиция "pexels идёт раньше в кортеже
        # источников" не имеет права дать ему систематическое преимущество.
        assert counts["pexels"] >= 6 and counts["pixabay"] >= 6 and counts["unsplash"] >= 6, first20
        assert max(counts.values()) - min(counts.values()) <= 1, first20


class TestVideoSourceInterleave:
    """pexels_video() имел ТОТ ЖЕ дефект, что фото чинили 13.09 — "весь
    Pexels, потом весь Pixabay" внутри ОДНОГО запроса, — и его никогда не
    портировали на видео (найдено 17.09 прямой проверкой, а не чтением).
    Цена не абстрактная: VIDEO_RELEVANCE_MAX_TRIES=3 реально скачивает
    первых по списку кандидатов, а для однозапросного слота (нет
    extra_queries — самый частый случай) список был "все Pexels", потом
    "все Pixabay" — второй видео-корпус, заведённый именно против брака
    63% у видео, для таких слотов не участвовал в отборе НИКОГДА."""

    def test_pixabay_video_is_actually_tried_within_the_budget(self, monkeypatch, tmp_path):
        def mk(prefix, n):
            return [{"id": f"{prefix}:{i}",
                     "video_files": [{"file_type": "video/mp4", "width": 1920, "link": "http://x"}]}
                    for i in range(n)]

        monkeypatch.setattr(ps, "PEXELS_API_KEY", "fake")
        # id без префикса -> candidate_source() читает его как "pexels".
        monkeypatch.setattr(ps, "_pexels_search_videos", lambda q: mk("9", 40))
        monkeypatch.setattr(ps, "_pixabay_search_videos", lambda q: mk("pixabay", 40))
        monkeypatch.setattr(ps, "filter_alt_blocklist", lambda items: items)

        def fake_download(*a, **k):
            raise OSError("нет сети")

        monkeypatch.setattr(ps, "atomic_url_download", fake_download)
        ps.reset_source_stats()
        order = []
        real_bump = ps._source_bump

        def spy_bump(source, field, n=1):
            if field == "offered":
                order.append(source)
            return real_bump(source, field, n)

        monkeypatch.setattr(ps, "_source_bump", spy_bump)
        ps.pexels_video("modern office worker typing", 0, used_ids=set(), used_hashes=[])
        # order — ВЕСЬ собранный пул (обе выдачи по 40 реально попали в
        # него); настоящая проверка — что источники внутри ordered НЕ идут
        # блоками. Первые VIDEO_RELEVANCE_MAX_TRIES=3 позиции — это ровно
        # то, что try_budget реально скачивает и оценивает.
        assert set(order[:ps.VIDEO_RELEVANCE_MAX_TRIES]) == {"pexels", "pixabay"}, order[:10]
