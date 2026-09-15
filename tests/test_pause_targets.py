# -*- coding: utf-8 -*-
"""Тишину задаёт ПЛАН, речь — движок.

Замер по alignment реального оплаченного заказа (14.09) показал, что длину
тега движок держит как придётся, и расхождение доходит до шести раз:

    [pause]        в середине текста   1.027с   (документировано 0.8)
    [pause]        в конце заказа      0.243с
    [short pause]  в середине текста   0.169с   (документировано 0.4)

Весь звуковой монтаж при этом считает тишину по факту: точечный кюй живёт
ТОЛЬКО в реальной паузе, и в 0.169с не влезает ни один ассет — кюй честно
отбрасывался с причиной no_silence_for_object.

Две правки, обе односторонние по построению:
1. fix_pauses доводит короткую тег-паузу до её же документированной длины.
   Подрезка живёт выше THRESH_SEC=1.0с, оба целевых значения ниже — логики
   физически не пересекаются.
2. Объектный кюй выбирает запись ПОД доступную тишину (у armour_clank в
   библиотеке 0.60/1.93/2.00/2.27с, ротация выдавала 2.27).
"""
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)
sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]

import fix_pauses as fp  # noqa: E402
import sfx_plan  # noqa: E402


class TestExtensionIsOneSided:
    def test_targets_are_the_documented_tag_durations(self):
        """Цель не выдумана: это ровно ЧАСТЬ 10 CLAUDE.md."""
        import script_parser as sp
        for tag, target in fp.TAG_PAUSE_TARGETS.items():
            assert sp.PAUSE_DURATIONS[tag] == target

    def test_extension_and_trimming_can_never_touch_the_same_pause(self):
        """Гарантия односторонности — арифметическая, а не по договорённости:
        подрезка начинается выше THRESH_SEC, все цели ниже."""
        assert max(fp.TAG_PAUSE_TARGETS.values()) < fp.THRESH_SEC

    def test_a_long_enough_pause_is_left_alone(self):
        segments = [(0.0, 10.0)]
        planned = [(4.0, 5.2, 0.8, "[pause]")]          # 1.2с, уже больше цели
        segs, ins = fp.apply_tag_pause_targets(segments, planned, sil=[], fine=[(4.0, 5.2)])
        assert ins == []
        assert segs == [("copy", 0.0, 10.0)]

    def test_a_short_pause_is_brought_up_to_target(self):
        segments = [(0.0, 10.0)]
        planned = [(4.0, 4.169, 0.4, "[short pause]")]   # ровно измеренные 0.169с
        segs, ins = fp.apply_tag_pause_targets(segments, planned, sil=[],
                                               fine=[(4.0, 4.169)])
        assert len(ins) == 1
        pos, sec = ins[0]
        assert sec == pytest.approx(0.4 - 0.169, abs=0.001)
        assert any(k == "silence" for k, *_ in segs)

    def test_insertion_never_shortens_anything(self):
        """Вставка может только удлинить: сумма copy-кусков не меняется."""
        segments = [(0.0, 10.0)]
        planned = [(4.0, 4.169, 0.4, "[short pause]")]
        segs, _ = fp.apply_tag_pause_targets(segments, planned, sil=[],
                                             fine=[(4.0, 4.169)])
        copied = sum(b - a for k, a, b in [x for x in segs if x[0] == "copy"])
        assert copied == pytest.approx(10.0, abs=1e-9)


class TestDoubleCountingWasCaughtByMeasurement:
    """Первая версия сравнивала цель с длиной ТЕГА и добавила 0.557с поверх
    уже вставленных склейкой 0.705с. detect_silences() по устройству не
    показывает ничего короче THRESH_SEC=1.0, а на границе секций лежало
    0.95с — чуть ниже порога."""

    def test_adjacent_silence_is_counted_not_missed(self):
        """Тег и вставленная склейкой пауза стоят ВСТЫК, а не внахлёст."""
        tag = (21.28, 21.523)
        fine = [(21.523, 22.23)]            # пауза склейки сразу за тегом
        have = fp.real_silence_at(tag[0], tag[1], sil=[], fine=fine)
        assert have == pytest.approx(0.95, abs=0.02)

    def test_a_boundary_that_already_has_room_gets_nothing(self):
        segments = [(0.0, 30.0)]
        planned = [(21.28, 21.523, 0.8, "[pause]")]
        _segs, ins = fp.apply_tag_pause_targets(segments, planned, sil=[],
                                                fine=[(21.523, 22.23)])
        assert ins == []

    def test_measurement_uses_a_finer_threshold_than_trimming(self):
        assert fp.FINE_SILENCE_MIN_SEC < fp.THRESH_SEC


class TestObjectAssetFitsTheRoom:
    """У концепта несколько записей — брать надо ту, что помещается."""

    def test_resolver_filters_by_room_and_keeps_rotation(self, monkeypatch):
        import pipeline_smart as ps
        files = ["/x/a.flac", "/x/b.flac", "/x/c.flac", "/x/d.flac"]
        durs = {"/x/a.flac": 0.60, "/x/b.flac": 1.93,
                "/x/c.flac": 2.00, "/x/d.flac": 2.27}
        monkeypatch.setattr(ps, "library_sounds", lambda k, n: files)
        monkeypatch.setattr(ps, "media_duration_or_none", lambda p: durs[p])
        wide = ps.object_asset_for("armour_clank")
        tight = ps.object_asset_for("armour_clank", max_sec=0.8)
        assert wide is not None and tight is not None
        assert tight[1] <= 0.8, tight
        # Без ограничения поведение прежнее — фильтр не применяется.
        assert wide[0] in files

    def test_no_variant_fits_is_an_honest_none(self, monkeypatch):
        import pipeline_smart as ps
        monkeypatch.setattr(ps, "library_sounds", lambda k, n: ["/x/big.flac"])
        monkeypatch.setattr(ps, "media_duration_or_none", lambda p: 3.0)
        assert ps.object_asset_for("armour_clank", max_sec=0.8) is None
        assert ps.object_asset_for("armour_clank") is not None

    def test_planner_asks_again_only_when_the_natural_pick_does_not_fit(self):
        """Правка строго добавляющая: повторный запрос идёт ТОЛЬКО когда
        природный выбор длиннее доступной тишины."""
        src = open(os.path.join(SCRIPTS_DIR, "sfx_plan.py"), encoding="utf-8").read()
        block = src[src.index("if cls == OBJECT_CLASS_POINT and asset_for:"):]
        block = block[:block.index("gain_db = got[3]")]
        assert "asset_dur > room" in block
        assert "max_sec=room" in block

    def test_max_sec_reaches_only_resolvers_that_declare_it(self):
        """Старый резолвер (без max_sec) не должен получить лишний аргумент."""
        def old_style(name, at=None):
            return ("/x/a.flac", 2.5, sfx_plan.OBJECT_CLASS_POINT)
        assert sfx_plan._call_asset_for(old_style, "n", 1.0) is not None
        assert sfx_plan._call_asset_for(old_style, "n", 1.0, max_sec=0.5) is None


class TestTimeMapKnowsAboutInserts:
    def test_inserted_silence_shifts_later_times(self, monkeypatch):
        import pipeline_smart as ps
        monkeypatch.setattr(ps, "load_pause_inserts", lambda: [(10.0, 0.5)])
        assert ps.raw_to_real_time(5.0, []) == pytest.approx(5.0)
        assert ps.raw_to_real_time(20.0, []) == pytest.approx(20.5)

    def test_old_episode_without_the_key_behaves_exactly_as_before(self, monkeypatch):
        import pipeline_smart as ps
        monkeypatch.setattr(ps, "load_pause_inserts", lambda: [])
        assert ps.raw_to_real_time(20.0, [[1.0, 2.0]]) == pytest.approx(19.0)

    def test_the_map_is_read_in_one_place_not_passed_by_hand(self):
        """Шесть вызывающих не обязаны знать про вставки: карту, которую
        надо передавать руками, рано или поздно кто-нибудь не передаст."""
        src = open(os.path.join(SCRIPTS_DIR, "pipeline_smart.py"), encoding="utf-8").read()
        body = src[src.index("def raw_to_real_time(t, cuts):"):]
        body = body[:body.index("\n\n\n")]
        assert "load_pause_inserts()" in body
