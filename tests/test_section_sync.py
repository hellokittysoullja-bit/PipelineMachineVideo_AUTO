"""Регрессия на реальный найденный баг (videos/01_ves-mecha, 30 августа):
offsets.get(section_order[i-1], 0.0) в detect_section_offsets() молча
подставлял 0.0 вместо реального/оценённого смещения предыдущей секции,
если та была ОТКЛОНЕНА хоть раз где-то раньше по цепочке — "разрыв между
секциями" превращался в "почти абсолютную позицию в аудио", sanity-ratio
улетал за SANITY_RATIO_MAX, и ВСЕ секции после первого отказа отклонялись
каскадно, даже когда их собственный offset (найденный кросс-корреляцией)
был точным. Фикс — section_start_estimate, который всегда хранит валидную
оценку (принятый offset ИЛИ структурный cursor), никогда 0.0 по дефолту
словаря. Живая проверка на реальном эпизоде 01_ves-mecha (BLOCK6-8/FINAL,
5 из 9 секций) — до фикса rejected/sanity_failed с raw_gap за 500-870с
разницы с chapters.txt, после — все 10/10 приняты, raw_gap в пределах ~5с
от real_gap. Здесь — минимальный синтетический повтор той же цепочки:
HOOK -> BLOCK1 (принят, offset=500) -> BLOCK2 (намеренно отклонён,
offset=None) -> BLOCK3 (должен приняться, используя оценку старта BLOCK2,
а не 0.0)."""
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import section_sync as ss  # noqa: E402


def _write_alignment_csv(path, chars):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["char", "start", "end"])
        for c, s, e in chars:
            w.writerow([c, s, e])


class _FakePS:
    """Лёгкая замена pipeline_smart — только то, что реально использует
    detect_section_offsets/local_pause_events, без тяжёлого импорта
    (torch/CLIP и т.д.) и без побочных эффектов на диск."""
    import re as _re
    ALIGNMENT_TAG_RE = _re.compile(r'\[short pause\]|\[pause\]')
    ALIGNMENT_STRIP_TAGS = ("[energetic]", "[slowly]", "[emphasis]")

    def __init__(self, media_duration):
        self._media_duration = media_duration
        from script_parser import parse_blocks
        self.parse_blocks = parse_blocks

    def get_media_duration(self, path):
        return self._media_duration


def _build_episode(tmp_path, words_by_section):
    script_path = tmp_path / "script.txt"
    body = ""
    for header, n_words in words_by_section:
        text = " ".join(["слово"] * n_words)
        body += f"=== {header} ===\n{text}\n\n"
    script_path.write_text(body, encoding="utf-8")

    (tmp_path / "audio.mp3").write_bytes(b"fake-audio-not-real")

    align_dir = tmp_path / "media_plan" / "alignment"
    align_dir.mkdir(parents=True)
    # HOOK/BLOCK1/BLOCK2/BLOCK3 -> 00.csv.. по порядку появления в script.txt.
    _write_alignment_csv(align_dir / "00.csv", [("H", 0.0, 0.1), ("!", 1.9, 2.0)])
    _write_alignment_csv(align_dir / "01.csv", [("Б", 0.0, 0.1), ("!", 2.9, 3.0)])
    _write_alignment_csv(align_dir / "02.csv", [("Б", 0.0, 0.1), ("!", 1.9, 2.0)])
    _write_alignment_csv(align_dir / "03.csv", [("Б", 0.0, 0.1), ("!", 2.9, 3.0)])
    return script_path


def test_section_after_rejected_predecessor_not_compared_against_zero(tmp_path, monkeypatch):
    tw = 2 + 400 + 50 + 400  # HOOK, BLOCK1, BLOCK2, BLOCK3 — см. докстринг файла
    words_by_section = [("HOOK", 2), ("BLOCK 1: A", 400),
                         ("BLOCK 2: B", 50), ("BLOCK 3: C", 400)]
    _build_episode(tmp_path, words_by_section)

    fake_ps = _FakePS(media_duration=float(tw))  # -> wps == 1.0 слово/сек ровно
    monkeypatch.setattr(ss, "_load_pipeline_smart", lambda video_dir: fake_ps)
    monkeypatch.setattr(ss, "detect_fine_silences", lambda audio_path: [])

    # estimate_section_offset вызывается по разу на HOOK->BLOCK1, BLOCK1->BLOCK2,
    # BLOCK2->BLOCK3 (в этом порядке). Канонические возвраты:
    #   BLOCK1: offset=500.0, conf=1.0   (ratio=(500-0)/400=1.25 -> принят)
    #   BLOCK2: offset=None              (гарантированный отказ, "no_cluster")
    #   BLOCK3: offset=853.0, conf=1.0   (см. расчёт ниже)
    canned = [(500.0, 1.0), (None, 0.0), (853.0, 1.0)]
    calls = {"n": 0}

    def fake_estimate(local_events, global_silences, search_lo, search_hi):
        offset, conf = canned[calls["n"]]
        calls["n"] += 1
        return offset, conf

    monkeypatch.setattr(ss, "estimate_section_offset", fake_estimate)

    offsets, report = ss.detect_section_offsets(str(tmp_path))

    assert report["BLOCK 2: B"]["method"] == "rejected"
    assert report["BLOCK 2: B"]["reason"] == "no_cluster"
    assert "BLOCK 2: B" not in offsets

    # BLOCK1 закончился на cursor = 500.0 (offset) + 3.0 (local_span) = 503.0 —
    # это и есть структурная оценка старта BLOCK2 (section_start_estimate),
    # даже притом что BLOCK2 сам был отклонён. BLOCK3 сравнивается ИМЕННО
    # против неё (503.0), а не против 0.0:
    #   ratio_fixed  = (853.0 - 503.0) / 400 = 0.875  -> в [0.5, 2.0] -> ПРИНЯТ
    #   ratio_buggy  = (853.0 -   0.0) / 400 = 2.1325  -> ВНЕ [0.5, 2.0] -> был бы отклонён
    # (ratio_buggy — не вызывается напрямую, а прямое следствие бага:
    # offsets.get("BLOCK 2: B", 0.0) == 0.0, потому что BLOCK2 отклонён и
    # никогда не попадает в offsets — см. докстринг файла).
    assert report["BLOCK 3: C"]["method"] == "pause_cross_correlation", (
        "BLOCK3 должен быть принят: без фикса он бы отклонился 'sanity_failed', "
        "потому что старый код сравнивал его offset против 0.0 вместо реальной "
        "структурной оценки старта отклонённого BLOCK2"
    )
    assert offsets["BLOCK 3: C"] == 853.0
    assert offsets["BLOCK 1: A"] == 500.0
    assert offsets["HOOK"] == 0.0
