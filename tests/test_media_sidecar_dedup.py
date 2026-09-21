"""Кэш-хит обязан возвращать кадр в анти-дубль.

Фон — РЕАЛЬНЫЙ дубль в опубликованном эпизоде 01_ves-mecha (найден 04.09
покадровым просмотром, не гипотеза): одна и та же фотография (Pexels 31474665)
стоит на 58-й секунде (слот 9, хук) и на 11-й минуте (слот 109, БЛОК 6).

Механика, которая это допустила, — двухуровневая:

1. N8 (docs/AUDIT_2026-09_DEEP.md:299): на кэш-хите КАНДИДАТА Pexels-ID не
   попадал в used_ids — дедуп по ID был мёртв для всех закэшированных слотов.
2. Глубже: при кэш-хите самого КЛИПА (temp_smart/clip_*.mp4) функция подбора не
   вызывается вообще, поэтому слот не попадал ни в used_ids, ни в used_hashes.
   На частичном ре-рендере (161 кэш-хит из 165) структуры дедупа оставались
   почти пустыми — и любой заново подбираемый слот спокойно брал кадр, уже
   стоящий в другом месте ролика.

Тесты проверяют механику возврата кадра в дедуп (register_cached_media) и
sidecar, на котором она держится.
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

FIXTURES = os.path.join(REPO_ROOT, "tests", "fixtures", "golden_media")
SWORD = os.path.join(FIXTURES, "euro_sword_2.jpg")
KATANA = os.path.join(FIXTURES, "katana.jpg")

pytestmark = pytest.mark.skipif(
    not os.path.exists(SWORD), reason="нет golden-фикстур")


def test_sidecar_roundtrip(tmp_path):
    f = tmp_path / "0001_q_g.jpg"
    f.write_bytes(b"x")
    ps.write_media_sidecar(str(f), pexels_id=31474665, query="knight armor",
                           kind="photo", ahash_hex="1010", chosen_by="arbiter")
    meta = ps.read_media_sidecar(str(f))
    assert meta["pexels_id"] == 31474665
    assert meta["query"] == "knight armor"
    assert meta["chosen_by"] == "arbiter"
    assert meta["ahash"] == "1010"


def test_sidecar_missing_and_corrupt_are_empty_dict(tmp_path):
    """Отсутствие/битый sidecar — норма (старый кэш), не исключение."""
    f = tmp_path / "a.jpg"
    f.write_bytes(b"x")
    assert ps.read_media_sidecar(str(f)) == {}
    with open(ps.media_sidecar_path(str(f)), "w", encoding="utf-8") as fh:
        fh.write("{не json")
    assert ps.read_media_sidecar(str(f)) == {}


def test_sidecar_write_never_raises_on_unwritable_path():
    """Метаданные вспомогательные: сбой записи не должен ронять подбор кадра."""
    ps.write_media_sidecar("/proc/definitely/not/writable/x.jpg", pexels_id=1)


def test_register_returns_pexels_id_into_used_ids(tmp_path):
    """N8: ID из sidecar обязан вернуться в used_ids на кэш-хите."""
    f = tmp_path / "0109.jpg"
    f.write_bytes(open(SWORD, "rb").read())
    ps.write_media_sidecar(str(f), pexels_id=31474665, kind="photo")
    used_ids, used_hashes = set(), []
    assert ps.register_cached_media(str(f), used_ids, used_hashes, kind="photo")
    assert 31474665 in used_ids


def test_register_computes_ahash_for_legacy_cache_without_sidecar(tmp_path):
    """Файлы, скачанные ДО появления sidecar, всё равно должны дедуплицироваться.

    ID у них взять неоткуда (в имени файла его нет), но визуальный хэш
    считается прямо от файла — и этого достаточно, чтобы поймать повтор.
    """
    f = tmp_path / "legacy.jpg"
    f.write_bytes(open(SWORD, "rb").read())
    assert not os.path.exists(ps.media_sidecar_path(str(f)))
    used_ids, used_hashes = set(), []
    assert ps.register_cached_media(str(f), used_ids, used_hashes, kind="photo")
    assert used_hashes and len(used_hashes[0]) == 64
    assert not used_ids


def test_cache_hit_slot_makes_identical_photo_a_detected_duplicate(tmp_path):
    """Главный регрессионный тест: воспроизводит опубликованный дубль #9 ≡ #109.

    Сценарий как в проде: слот A собран в прошлом прогоне и в этом идёт
    кэш-хитом клипа (подбор не вызывается). Слот B подбирается заново и
    натыкается на ТУ ЖЕ картинку. До правки used_hashes на момент подбора B был
    пуст -> дистанция считалась от пустого списка (default=99) -> кадр
    принимался как уникальный. После правки кадр слота A возвращён в дедуп, и
    расстояние Хэмминга до идентичного файла = 0, то есть дубль ловится.
    """
    a = tmp_path / "slot_a.jpg"
    b = tmp_path / "slot_b.jpg"
    data = open(SWORD, "rb").read()
    a.write_bytes(data)
    b.write_bytes(data)

    used_ids, used_hashes = set(), []
    # Слот A: клип взят из кэша, подбор не вызывался — раньше здесь не
    # происходило РОВНО НИЧЕГО, и в этом был баг.
    ps.register_cached_media(str(a), used_ids, used_hashes, kind="photo")

    # Слот B: подбор идёт заново, кандидат — тот же кадр.
    h_b = ps.ahash(str(b))
    distance = min(ps.hamming(h_b, uh) for uh in used_hashes)
    assert distance == 0
    assert distance <= ps.PHOTO_DEDUP_HAMMING, (
        "идентичный кадр не распознан как дубль — анти-дубль снова слеп "
        "к слотам, собранным кэш-хитом"
    )


def test_distinct_photos_are_not_falsely_flagged(tmp_path):
    """Обратная сторона: возврат в дедуп не должен блокировать РАЗНЫЕ кадры."""
    a = tmp_path / "sword.jpg"
    b = tmp_path / "katana.jpg"
    a.write_bytes(open(SWORD, "rb").read())
    b.write_bytes(open(KATANA, "rb").read())
    used_ids, used_hashes = set(), []
    ps.register_cached_media(str(a), used_ids, used_hashes, kind="photo")
    distance = min(ps.hamming(ps.ahash(str(b)), uh) for uh in used_hashes)
    assert distance > ps.PHOTO_DEDUP_HAMMING, (
        f"два разных кадра приняты за дубль (расстояние {distance})"
    )


def test_video_registration_uses_id_only(tmp_path):
    """Для видео aHash по файлу не считается (нужен ffmpeg) — только ID."""
    f = tmp_path / "0133.mp4"
    f.write_bytes(b"not really a video")
    ps.write_media_sidecar(str(f), pexels_id=5846389, kind="video")
    used_ids, used_hashes = set(), []
    assert ps.register_cached_media(str(f), used_ids, used_hashes, kind="video")
    assert 5846389 in used_ids
    assert used_hashes == []
