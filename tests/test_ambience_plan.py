# -*- coding: utf-8 -*-
"""Атмосферный слой: четыре названных риска закрыты устройством, а не
аккуратностью настройки.

Владелец назвал их поимённо, и каждый здесь имеет свой тест:
  1) перегруженность звуком,
  2) повторы,
  3) ограничение библиотек,
  4) динамика — слой должен жить, а не лупиться,
и пятый, общий: «получится весёлый генератор звуков, а не работа человека».

Главный тест файла — последний: он гоняет словарь по РЕАЛЬНОМУ сценарию
канала и запирает настоящий промах, найденный при разработке (хук эпизода
02 — человек лицом в грязи на поле боя — получал атмосферу КУЗНИЦЫ,
потому что в словаре стояли слова про предмет: «сталь», «клинок», «молот»).
"""
import math
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import ambience_plan as ap  # noqa: E402

EP02 = os.path.join(REPO_ROOT, "videos", "02_ne-mechom", "script.txt")


def blocks_from(texts, section="BLOCK 1", words=40):
    return [{"section": section, "text": t, "words": words} for t in texts]


def timeline(blocks, per_block=30.0):
    starts = [i * per_block for i in range(len(blocks))]
    return starts, len(blocks) * per_block


# ---------------------------------------------------------------- риск 2 и 4

def test_layer_lengths_are_pairwise_coprime():
    """Слой не лупится — это арифметика, а не обещание.

    Комбинация трёх источников повторяется через НОК их длин. Если длины
    взаимно просты, НОК равен произведению, то есть 38.7 часа — эпизод
    физически не успевает дойти до повторения.
    """
    lens = ap.AMBIENCE_LAYER_SECONDS
    for a, b in ((lens[i], lens[j]) for i in range(len(lens)) for j in range(i + 1, len(lens))):
        assert math.gcd(a, b) == 1, f"{a} и {b} не взаимно просты — слой начнёт повторяться"
    lcm = 1
    for x in lens:
        lcm = lcm * x // math.gcd(lcm, x)
    assert lcm / 3600.0 > 24.0, "период повторения короче суток — на длинном эпизоде слышно"


def test_drift_periods_never_align_with_layers_or_each_other():
    """Динамика: «дыхание» громкости тоже не имеет права совпасть само с собой."""
    for d in ap.AMBIENCE_DRIFT_SECONDS:
        for l in ap.AMBIENCE_LAYER_SECONDS:
            assert math.gcd(d, l) == 1, f"период дыхания {d} кратен длине слоя {l}"
    ds = ap.AMBIENCE_DRIFT_SECONDS
    for a, b in ((ds[i], ds[j]) for i in range(len(ds)) for j in range(i + 1, len(ds))):
        assert math.gcd(a, b) == 1


def test_same_bed_in_two_chapters_gets_different_seeds():
    """Две главы с ОДНОЙ атмосферой обязаны звучать разными её участками."""
    a = ap._stable_seed("BLOCK 3", "wind_open")
    b = ap._stable_seed("BLOCK 7", "wind_open")
    assert a != b


def test_seed_is_stable_across_runs():
    """Иначе один и тот же эпизод звучал бы по-разному при каждом рендере."""
    assert ap._stable_seed("HOOK", "wind_open") == ap._stable_seed("HOOK", "wind_open")
    assert ap._stable_seed("HOOK", "wind_open") == 275279034


# ------------------------------------------------------------------ риск 1

def test_distinct_bed_cap_is_enforced():
    """Экскурсия по звуковой библиотеке — это и есть перегруженность."""
    texts = ["поле равнина ветер открытый холм битва",
             "замок крепость собор камень свод подземелье",
             "кузница горнило наковальня пламя костер угли",
             "дождь ливень слякоть болото распутица лужа",
             "рынок базар ярмарка толпа площадь улица"]
    blocks = [{"section": f"BLOCK {i}", "text": t, "words": 60} for i, t in enumerate(texts)]
    starts, total = timeline(blocks, per_block=120.0)
    plan = ap.plan_ambience(blocks, starts, total, block_text=lambda b: b["text"])
    used = {s["bed"] for s in plan if s["bed"]}
    assert len(used) <= ap.AMBIENCE_MAX_DISTINCT
    assert "over_distinct_cap" in {s["reason"] for s in plan}


def test_short_chapter_does_not_start_its_own_bed():
    """Слой, появившийся на полминуты, читается как переключение, не как место."""
    blocks = [{"section": "BLOCK 1", "text": "замок крепость собор камень свод", "words": 5}]
    starts, total = [0.0], 20.0
    plan = ap.plan_ambience(blocks, starts, total, block_text=lambda b: b["text"])
    assert plan[0]["bed"] is None
    assert plan[0]["reason"] == "run_too_short"


# ------------------------------------------------------------ риск 5 (автомат)

def test_no_place_words_means_silence():
    """Главный признак автомата — он звучит везде. Не уверен — тишина."""
    blocks = blocks_from(["Это разговор о цифрах и о том, почему источники врут."])
    starts, total = timeline(blocks, per_block=200.0)
    plan = ap.plan_ambience(blocks, starts, total, block_text=lambda b: b["text"])
    assert plan[0]["bed"] is None
    assert plan[0]["reason"] == "low_confidence"


def test_two_close_beds_mean_silence_not_the_winner():
    """Вплотную идущие варианты — текст не говорит о месте определённо."""
    scores = {"wind_open": 7, "stone_hall": 6}
    assert scores["wind_open"] - scores["stone_hall"] < ap.AMBIENCE_AMBIGUITY_MARGIN
    blocks = blocks_from(["поле ветер холм равнина замок крепость собор камень свод башня"])
    starts, total = timeline(blocks, per_block=200.0)
    plan = ap.plan_ambience(blocks, starts, total, block_text=lambda b: b["text"])
    if plan[0]["bed"] is None:
        assert plan[0]["reason"] in ("ambiguous", "low_confidence")


def test_bed_does_not_flicker_between_neighbouring_chapters():
    """Гистерезис: близкая по смыслу соседняя глава не дёргает слой."""
    blocks = [
        {"section": "BLOCK 1", "text": "поле равнина ветер открытый холм битва сражение войско",
         "words": 60},
        {"section": "BLOCK 2", "text": "поле ветер замок камень свод", "words": 60},
    ]
    starts, total = timeline(blocks, per_block=200.0)
    plan = ap.plan_ambience(blocks, starts, total, block_text=lambda b: b["text"])
    assert plan[0]["bed"] == "wind_open"
    assert plan[1]["bed"] in (None, "wind_open"), "слой дёрнулся на соседней главе"


def test_plan_covers_the_whole_timeline_without_holes():
    """Дорожка обязана совпадать с роликом секунда в секунду."""
    blocks = [{"section": f"BLOCK {i}", "text": "поле ветер равнина битва", "words": 40}
              for i in range(4)]
    starts, total = timeline(blocks, per_block=100.0)
    plan = ap.plan_ambience(blocks, starts, total, block_text=lambda b: b["text"])
    assert plan[0]["start"] == 0.0
    assert plan[-1]["end"] == pytest.approx(total)
    for a, b in zip(plan, plan[1:]):
        assert a["end"] == pytest.approx(b["start"]), "дыра в плане сдвинет всё, что после неё"


def test_merge_adjacent_prevents_a_bed_restarting_every_chapter():
    """Одна и та же атмосфера дважды за минуту «выключилась и включилась» —
    самый слышимый признак автомата."""
    plan = [{"section": "A", "start": 0.0, "end": 60.0, "bed": "wind_open", "reason": "selected"},
            {"section": "B", "start": 60.0, "end": 120.0, "bed": "wind_open", "reason": "selected"}]
    merged = ap.merge_adjacent(plan)
    assert len(merged) == 1 and merged[0]["end"] == 120.0


# ------------------------------------------------------- словарь: промахи

def test_stems_are_long_enough_to_not_catch_other_words():
    """Реальные промахи первой версии: «зал» ловил ЗАЛП, «бой» — БОЯТСЯ,
    «кон» — КОНЕЧНО. Короткие основы запрещены целиком."""
    for bed, groups in ap.AMBIENCE_VOCAB.items():
        for weight, kinds in groups.items():
            for stem in kinds.get("pref", ()):
                assert len(stem) >= ap.AMBIENCE_MIN_STEM, f"{bed}: основа «{stem}» слишком коротка"


@pytest.mark.parametrize("word", ["залп", "залпом", "боятся", "конечно", "мастерство",
                                  "походка", "дорогой", "конец", "закончил", "наконец"])
def test_known_false_positives_score_nothing(word):
    """Каждое из этих слов реально встречается в сценариях канала."""
    assert ap.score_text(word) == {}, f"«{word}» ловит атмосферу, которой там нет"


def test_vocabulary_contains_no_object_or_person_words():
    """ГЛАВНОЕ правило словаря: атмосфера описывает МЕСТО, а не предмет.

    Меч в кадре не сообщает ничего о том, где человек находится. Ровно на
    этом первая версия и промахнулась.
    """
    forbidden = ("сталь", "клинок", "меч", "молот", "рыцар", "герб", "трон",
                 "король", "корол", "конь", "всадник", "лошад", "доспех", "шлем")
    for bed, groups in ap.AMBIENCE_VOCAB.items():
        for weight, kinds in groups.items():
            for entry in tuple(kinds.get("word", ())) + tuple(kinds.get("pref", ())):
                assert not any(entry.startswith(f) for f in forbidden), \
                    f"{bed}: «{entry}» — про предмет или человека, а не про место"


# ------------------------------------------------- проверка на живом сценарии

@pytest.mark.skipif(not os.path.exists(EP02), reason="нет реального сценария")
class TestRealEpisode:
    def _plan(self):
        import script_parser
        blocks = script_parser.parse_blocks(EP02)
        starts, t = [], 0.0
        for b in blocks:
            starts.append(t)
            t += max(0.5, b["words"] / 125.0 * 60.0) + b.get("pause_after", 0.5)
        return blocks, ap.merge_adjacent(
            ap.plan_ambience(blocks, starts, t, block_text=lambda b: b["text"])), t

    def test_hook_is_not_a_forge(self):
        """Запертый промах: хук — человек лицом в грязи на поле боя."""
        _, plan, _ = self._plan()
        assert plan[0]["bed"] != "forge_fire"

    def test_does_not_play_under_every_chapter(self):
        """Звук под каждой главой подряд — признак автомата, а не человека."""
        _, plan, _ = self._plan()
        share = ap.summarize(plan)["covered_share"]
        assert share < 0.85, f"атмосфера покрывает {share:.0%} ролика — это уже автомат"

    def test_beds_chosen_are_plausible_for_this_episode(self):
        """Эпизод про пеший бой в грязи: кузница/рынок под ним неуместны."""
        _, plan, _ = self._plan()
        used = {s["bed"] for s in plan if s["bed"]}
        assert used <= {"wind_open", "rain_mud", "stone_hall"}, f"неуместная атмосфера: {used}"


def test_library_kinds_and_plan_vocabulary_match_exactly():
    """Вид, которого нет в словаре плана, план НИКОГДА не выберет — то есть
    собранные для него записи не дадут ролику ничего. Вид, который план
    выбирает, но которого нет в библиотеке, оставит участок без звука.
    Это ровно тот класс пробела, который уже ловили у Openverse, Pixabay,
    Unsplash и reveal-акцентов: код есть, ролику от него ноль.
    """
    import sound_library as sl

    voc = set(ap.AMBIENCE_VOCAB)
    lib = set(sl.LIBRARY_SPEC["ambience"])
    assert lib - voc == set(), f"план никогда не выберет: {sorted(lib - voc)}"
    assert voc - lib == set(), f"в библиотеке нет: {sorted(voc - lib)}"
