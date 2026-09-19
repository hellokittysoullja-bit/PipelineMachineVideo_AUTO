# -*- coding: utf-8 -*-
"""Загрузка профессиональных пакетов (Sonniss GDC) в библиотеку звуков.

Главное, что здесь заперто, — НЕ скачивание (оно ручное и многогигабайтное),
а два решения, каждое из которых найдено живым прогоном 14.09:

1. Лицензия пакета не спрашивается и не угадывается, а `allow_ai_embeddings`
   жёстко False: Sonniss прямо запрещает обучение ИИ на своих звуках.
2. У предметного слоя появился блоклист по названию — до этого он был
   единственным из восемнадцати видов вообще без него, и на живом пакете это
   стоило записи «INSTRU STRING Double Bass, Bowed, Harmonic», принятой как
   обнажение меча.
"""
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

import sound_library as sl  # noqa: E402
import sonniss_ingest as si  # noqa: E402


class TestObjectTitleBlocklist:
    """Дыра, найденная живым пакетом: у object не было блоклиста вообще."""

    def test_every_object_concept_has_a_blocklist(self):
        missing = [n for n in sl.LIBRARY_SPEC["object"] if n not in sl.TITLE_BLOCK]
        assert not missing, (
            "виды предметного слоя без блоклиста по названию: " + ", ".join(missing) +
            " — ровно так контрабас попал в библиотеку как обнажение меча")

    def test_the_double_bass_that_got_in_is_blocked_now(self):
        title = "INSTRU STRING Double Bass, Bowed, Harmonic, Low, Slide, Whale.wav"
        assert sl.title_blocked("sword_draw", title)

    @pytest.mark.parametrize("name,title", [
        ("sword_draw", "metal-blade-friction-6.mp3"),
        ("sword_draw", "metal_hit_004.wav"),
        ("armour_clank", "METAL CHAIN Old Barn, Manipulation, Clinking 04.wav"),
        ("armour_clank", "Metallic Clank 4.wav"),
        ("arrow_shot", "Longbow Release 1.wav"),
        ("arrow_shot", "ARCHERY flyby 2.wav"),
        ("hammer_anvil", "Anvil - Lokomo A 100 kg - Forging hot steel 1 time short.wav"),
        ("hammer_anvil", "Metal,Hits,Pipe,Mixed,Fast,Irregular.wav"),
        ("footsteps_mud", "A Step On Dirt.wav"),
        ("footsteps_mud", "Bluezone_BC0254_wood_handling_003.wav"),
    ])
    def test_real_accepted_records_are_not_blocked(self, name, title):
        """Негативный контроль: блоклист обязан не задеть НИ ОДНУ из записей,
        которые реально лежат в библиотеке (проверено на всех 24)."""
        assert not sl.title_blocked(name, title)

    def test_musical_words_are_shared_not_copy_pasted(self):
        for name in sl.LIBRARY_SPEC["object"]:
            assert set(sl.TITLE_BLOCK_INSTRUMENT) <= set(sl.TITLE_BLOCK[name])


class TestLicenceIsNotGuessed:
    def test_ai_training_is_refused_by_construction(self):
        """Не флаг и не умолчание: пакет запрещает обучение, значит в вызове
        ingest_dir стоит литеральный False, который нельзя переключить."""
        src = open(os.path.join(SCRIPTS_DIR, "sonniss_ingest.py"), encoding="utf-8").read()
        assert "allow_ai_embeddings=False" in src
        assert "allow_ai_embeddings=True" not in src

    def test_licence_string_names_the_package_and_its_terms(self):
        assert "Sonniss" in si.SONNISS_LICENSE
        assert "no attribution" in si.SONNISS_LICENSE.lower()
        assert si.SONNISS_LICENSE_URL.startswith("https://")


class TestShortlist:
    def test_shortlist_only_narrows_never_invents(self):
        spec = dict(sl.LIBRARY_SPEC["object"]["arrow_shot"], _name="arrow_shot")
        files = ["/x/Longbow Release 1.wav", "/x/Honda,Civic,2001,By,Wet.wav",
                 "/x/FX_0421.wav", "/x/archery bow shot.wav"]
        got = si.shortlist_for(spec, "arrow_shot", files)
        assert set(got) <= set(files)
        assert "/x/Longbow Release 1.wav" in got
        assert "/x/Honda,Civic,2001,By,Wet.wav" not in got   # блоклист вида
        # Честный предел, названный в докстринге: безымянный файл до гейтов
        # не доедет. Это цена скорости, и она заперта тестом, а не забыта.
        assert "/x/FX_0421.wav" not in got

    def test_no_name_filter_hands_everything_to_the_gates(self):
        spec = dict(sl.LIBRARY_SPEC["object"]["arrow_shot"], _name="arrow_shot")
        files = ["/x/FX_0421.wav", "/x/whatever.wav"]
        assert si.shortlist_for(spec, "arrow_shot", files, use_names=False) == files


class TestBundleUrls:
    def test_known_years_resolve_to_reachable_mirrors(self):
        urls = si.bundle_urls(2019, 1)
        assert urls and all(u.startswith("https://") for u in urls)
        # sonniss.com отдаёт скрипту 403 (Cloudflare, замерено 14.09) —
        # в зеркалах его быть не должно, иначе прогон падает на первом же.
        assert not any("//sonniss.com" in u or "//downloads.sonniss.com" in u for u in urls)

    def test_part_out_of_range_is_refused(self):
        with pytest.raises(SystemExit):
            si.bundle_urls(2019, 99)
        with pytest.raises(SystemExit):
            si.bundle_urls(1999, 1)


def test_object_library_is_tracked_by_git():
    """Смысл всей загрузки: у локального пакета НЕТ url, значит restore его
    не восстановит — папка обязана лежать в git, иначе звуки исчезают вместе
    с контейнером. 2.8 МБ (замер) против 639 МБ у ambience."""
    gi = open(os.path.join(REPO_ROOT, ".gitignore"), encoding="utf-8").read()
    lines = [ln.strip() for ln in gi.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    assert "assets/library/object/" not in lines
    assert "assets/library/ambience/" in lines
