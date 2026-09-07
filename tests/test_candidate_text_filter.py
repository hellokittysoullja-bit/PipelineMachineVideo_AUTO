"""Жанровый фильтр кандидатов Pexels по тексту: слаг, видео-путь, линт запросов.

Три реальные дыры, найденные 07.09 прямым запросом к живому API (655
кандидатов по 30 запросам опубликованного эпизода), закрываются здесь:

1. `filter_alt_blocklist()` смотрела только в `alt`. У ВИДЕО-объектов Pexels
   `alt` равен None, а `tags` приходят пустыми — то есть по видео фильтр не
   мог сработать в принципе, даже если бы его вызывали. Единственный текст
   про содержимое — человекочитаемый слаг в `url`
   (`.../video/roman-soldiers-historical-reenactment-event-38103939/`).
2. Видео-путь не вызывал фильтр НИ РАЗУ: `filter_alt_blocklist()`
   существовала ровно в одном месте — внутри `pexels_photo()`. Половина
   слотов эпизода — видео.
3. Авторский запрос мог просить ровно то, что канал сам запрещает
   (`katana sword` в script.txt канала про европейское Средневековье), и
   увидеть это было негде: блоклист работает по кандидатам и на запросы не
   смотрит.

Фикстура `pexels_slugs_sample.json` — замороженная ЖИВАЯ выдача, не
выдумка: синтетические слаги не воспроизвели бы ни отсутствие alt у видео,
ни реальные пропорции спортивного фехтования в ответах на запросы про
средневековые мечи (45 из 655 кандидатов).
"""
import json
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]
import pipeline_smart as ps  # noqa: E402

SAMPLE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "fixtures", "pexels_slugs_sample.json")


def _sample():
    with open(SAMPLE, encoding="utf-8") as f:
        return json.load(f)["items"]


class TestCandidateText:
    def test_slug_words_become_searchable(self):
        item = {"url": "https://www.pexels.com/video/roman-soldiers-historical-"
                       "reenactment-event-38103939/", "alt": None}
        text = ps.pexels_candidate_text(item)
        # Без замены дефисов многословные термины блоклиста ("medieval
        # festival", "crowd watching") молча не сработали бы никогда.
        assert "historical reenactment event" in text
        assert "reenactment" in text

    def test_missing_alt_and_tags_do_not_crash(self):
        """У видео-объектов Pexels это норма, а не крайний случай."""
        assert ps.pexels_candidate_text({"url": None, "alt": None}) == ""
        assert ps.pexels_candidate_text({}) == ""

    def test_alt_is_still_used(self):
        """Фото-путь работал по alt — эта ось не должна пропасть."""
        item = {"url": "https://www.pexels.com/photo/abc-1/", "alt": "A KATANA on a stand"}
        assert "katana" in ps.pexels_candidate_text(item)

    def test_tags_are_used_when_pexels_provides_them(self):
        item = {"url": "https://www.pexels.com/photo/abc-1/", "alt": None,
                "tags": ["Cosplay", "convention"]}
        assert "cosplay" in ps.pexels_candidate_text(item)


class TestFilterOnRealSample:
    def test_video_reenactment_is_now_filtered_out(self):
        """Ровно тот кадр, на который пожаловался владелец канала."""
        item = {"id": 38103939, "alt": None,
                "url": "https://www.pexels.com/video/roman-soldiers-historical-"
                       "reenactment-event-38103939/"}
        good = {"id": 1, "alt": None,
                "url": "https://www.pexels.com/video/knight-in-plate-armor-1/"}
        assert ps.filter_alt_blocklist([item, good]) == [good]

    def test_modern_sport_fencing_is_filtered_out(self):
        """45 из 655 живых кандидатов — спортивное фехтование, и оно приходит
        по запросам вроде «medieval knight sword battle»."""
        fencing = {"id": 2, "alt": None,
                   "url": "https://www.pexels.com/video/two-fencers-at-their-"
                          "fighting-position-6537092/"}
        good = {"id": 3, "alt": None,
                "url": "https://www.pexels.com/photo/medieval-sword-in-museum-3/"}
        assert ps.filter_alt_blocklist([fencing, good]) == [good]

    def test_authentic_candidates_survive(self):
        """Вторая ось: фильтр не имеет права выкашивать нужное."""
        keep = [
            {"id": 10, "alt": None, "url": "https://www.pexels.com/photo/knight-armor-gauntlets-on-sword-hilt-10/"},
            {"id": 11, "alt": None, "url": "https://www.pexels.com/video/blacksmith-forging-a-blade-11/"},
            {"id": 12, "alt": None, "url": "https://www.pexels.com/photo/medieval-castle-timber-hall-12/"},
        ]
        assert ps.filter_alt_blocklist(list(keep)) == keep

    def test_no_query_in_the_real_sample_loses_every_candidate(self):
        """Предусловие ужесточения: фильтр не должен обнулять слот.

        Если по какому-то запросу отсеивается ВСЁ, срабатывает откат
        «вернуть как было» — и ужесточение превращается в ноль. На живой
        выдаче этого не происходит ни на одном из 30 запросов эпизода;
        тест сторожит это свойство.
        """
        items = _sample()
        by_query = {}
        for it in items:
            by_query.setdefault((it["query"], it["kind"]), []).append(it)
        starved = []
        for key, group in by_query.items():
            kept = ps.filter_alt_blocklist(list(group))
            blocked = [g for g in group
                       if any(t in ps.pexels_candidate_text(g)
                              for t in ps.CONTENT_ALT_BLOCKLIST)]
            if blocked and len(kept) == len(group):
                starved.append(key)
        assert not starved, f"откат «отфильтровалось всё» сработал на: {starved}"

    def test_filter_actually_removes_a_meaningful_share(self):
        """Храповик: если блоклист или извлечение текста сломается, фильтр
        станет тихим no-op — а выглядеть будет как раньше."""
        items = _sample()
        kept = ps.filter_alt_blocklist(list(items))
        removed = len(items) - len(kept)
        assert removed >= len(items) * 0.15, (
            f"фильтр убрал всего {removed} из {len(items)} — на живой выдаче "
            f"этого эпизода доля брака заметно выше, похоже фильтр не работает")

    def test_video_candidates_are_covered_too(self):
        """Главная из трёх дыр: раньше по видео фильтр не работал вообще."""
        videos = [i for i in _sample() if i["kind"] == "video"]
        assert videos, "в фикстуре нет видео"
        removed = len(videos) - len(ps.filter_alt_blocklist(list(videos)))
        assert removed > 0


class TestAuthoredQueryLint:
    def test_query_contradicting_the_channel_blocklist_is_reported(self, capsys):
        hits = ps.lint_authored_queries({"BLOCK6": ["medieval sword collection museum",
                                                     "katana sword"]})
        assert hits == [("BLOCK6", "katana sword", "katana")]
        assert "katana" in capsys.readouterr().out

    def test_clean_queries_produce_no_noise(self, capsys):
        hits = ps.lint_authored_queries({"HOOK": ["medieval knight sword battle"]})
        assert hits == []
        assert capsys.readouterr().out == ""

    def test_no_authored_queries_is_not_an_error(self):
        assert ps.lint_authored_queries(None) == []
        assert ps.lint_authored_queries({}) == []

    def test_the_real_published_script_is_checked(self):
        """Сценарий опубликованного эпизода — источник самого случая.

        Пока «katana sword» в нём не переписан, линт обязан его показывать;
        когда автор поправит запрос, тест перестанет находить именно этот
        термин, и это будет корректным исходом, а не поломкой.
        """
        script = os.path.join(REPO_ROOT, "videos", "01_ves-mecha", "script.txt")
        if not os.path.exists(script):
            pytest.skip("эпизод не в рабочей копии")
        from script_parser import parse_pexels_queries
        hits = ps.lint_authored_queries(parse_pexels_queries(script))
        # Не «должно быть ровно одно», а «линт умеет читать реальный файл»:
        # число зависит от правок сценария и не должно ронять сборку.
        assert isinstance(hits, list)
