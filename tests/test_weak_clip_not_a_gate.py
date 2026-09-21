"""Слабый CLIP-порог сходства с запросом больше не решает членство в пуле
для ранжирования (CANDIDATE_GATE_RULES_VERSION=3, 19.09).

Прямое требование владельца после разбора реальных кадров эпизода
("там, где решает, кого пускать в пул — нахуй убирай слабую [модель]"),
проверенное собственным замером (docs/quality/so400m_as_judge.json,
коммит 65b4031): и слабая (clip-vit-base-patch32), и сильная (SigLIP2-
so400m) embedding-модель дают ПОЛНОЕ перекрытие распределений годных и
брака на пороге сходства картинки с текстом запроса (AUC 0.566 у слабой —
почти монетка). Заменить судью на другую модель того же класса не решает
задачу — держать ЛЮБОЙ такой порог хард-гейтом, отбраковывающим кандидата
до того, как его увидит любой другой механизм (ранжирование, зрячий гейт),
значит выбрасывать часть настоящих годных кадров монеткой.

Три ДРУГИЕ, калиброванные проверки (risky-margin/домен-гвард/контрастивное
вето — candidate_passes_guards()) реально разделяют годное и брак (0
ложных отказов на золотом наборе) и остаются гейтом. Порог сходства с
запросом становится ТОЛЬКО ранжирующим сигналом (relevance_rank_bucket в
_score_and_pick()), а не решением "рассматривать кандидата вообще или нет".

is_relevant_candidate() (используется golden_set_eval.py/базовой линией
golden_set_baseline.json) НЕ ТРОНУТА — байт-в-байт то же самое решение
(порог AND защиты), только путь отбора в pexels_photo()/pexels_video()
теперь зовёт candidate_passes_guards() напрямую, без порога.
"""
import os
import sys
import tempfile
from unittest.mock import patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

sys.argv = ["pipeline_smart.py", tempfile.gettempdir()]
import pipeline_smart as ps  # noqa: E402


def _low_relevance():
    """Заведомо ниже CLIP_RELEVANCE_THRESHOLD (0.19) — та самая зона, где
    слабая модель, по собственному замеру репозитория, не разделяет
    годное и брак лучше монетки."""
    return ps.CLIP_RELEVANCE_THRESHOLD - 0.1


class TestCandidatePassesGuardsIgnoresRawThreshold:
    """candidate_passes_guards() — то, что теперь реально решает пул."""

    def test_low_relevance_alone_does_not_reject(self):
        """Кандидат с низкой сырой релевантностью, но без единого сработавшего
        калиброванного гварда — ПРОХОДИТ. Раньше (is_relevant_candidate())
        такой кандидат отсекался ещё до того, как гварды успевали хоть
        что-то проверить (короткое замыкание на пороге)."""
        with patch.object(ps, "is_risky_query", return_value=False), \
             patch.object(ps, "visual_domain_guard_violation", return_value=(False, None)), \
             patch.object(ps, "negative_anchor_violation", return_value=(False, None)):
            assert ps.candidate_passes_guards("dummy.jpg", "any query",
                                               relevance=_low_relevance()) is True

    def test_is_relevant_candidate_still_rejects_the_same_input(self):
        """Контрольная сверка: is_relevant_candidate() (golden-тест/базовая
        линия золотого набора) НЕ ТРОНУТА — на ТОМ ЖЕ входе она по-прежнему
        отклоняет по сырому порогу. Обе функции обязаны РАЗОЙТИСЬ на этом
        входе — иначе правка ничего не изменила."""
        with patch.object(ps, "is_risky_query", return_value=False), \
             patch.object(ps, "visual_domain_guard_violation", return_value=(False, None)), \
             patch.object(ps, "negative_anchor_violation", return_value=(False, None)):
            assert ps.is_relevant_candidate("dummy.jpg", "any query",
                                             relevance=_low_relevance()) is False

    def test_domain_guard_violation_still_rejects_regardless_of_relevance(self):
        """Калиброванные защиты остаются ЖЁСТКИМ гейтом — высокая сырая
        релевантность не спасает кандидата с анахронизмом формы клинка."""
        with patch.object(ps, "is_risky_query", return_value=False), \
             patch.object(ps, "visual_domain_guard_violation", return_value=(True, "east_asian_sword")), \
             patch.object(ps, "negative_anchor_violation", return_value=(False, None)):
            assert ps.candidate_passes_guards("dummy.jpg", "sword", relevance=0.9) is False

    def test_negative_anchor_violation_still_rejects_regardless_of_relevance(self):
        """То же для контрастивного вето (современное вторжение)."""
        with patch.object(ps, "is_risky_query", return_value=False), \
             patch.object(ps, "visual_domain_guard_violation", return_value=(False, None)), \
             patch.object(ps, "negative_anchor_violation", return_value=(True, "crowd of modern spectators")):
            assert ps.candidate_passes_guards("dummy.jpg", "sword", relevance=0.9) is False

    def test_risky_margin_still_rejects_regardless_of_raw_threshold(self):
        """risky-margin (museum/exhibition/... запросы) — тоже сохранён как
        гейт, независимо от того, взят ли слабый порог."""
        with patch.object(ps, "is_risky_query", return_value=True), \
             patch.object(ps, "clip_relevance", return_value=0.05), \
             patch.object(ps, "visual_domain_guard_violation", return_value=(False, None)), \
             patch.object(ps, "negative_anchor_violation", return_value=(False, None)):
            # relevance=0.3 (высокая), но anchor_relevance подмокан на 0.05 —
            # margin 0.25 >= RISKY_QUERY_MARGIN, значит НЕ отклоняется этим
            # правилом; проверяем обратный случай — маленький margin отклоняет.
            assert ps.candidate_passes_guards("dummy.jpg", "museum display case",
                                               relevance=0.06) is False


class TestScoreAndPickAdmitsLowRelevanceGuardPassingCandidate:
    """Допустимое множество _score_and_pick() строится по is_relevant в
    candidates_info — код функции НЕ менялся, меняется то, ЧТО вызывающий
    код (pexels_photo/pexels_video) кладёт в это поле. Здесь проверяем
    напрямую: если is_relevant=1 (guard_ok) при низкой relevance, кандидат
    участвует в допустимом множестве и может победить по эстетике/яркости —
    раньше (is_relevant = порог) он туда бы не попал вовсе."""

    def test_low_relevance_guard_passing_candidate_can_win_via_admissible_set(self):
        candidates_info = [
            {"path": "a.jpg", "p": {"id": "a"}, "is_dup_free": 1, "size_ok": 1,
             "is_relevant": 1, "sharp_ok": 1, "aesthetic_val": 5.0,
             "luma_score": 0.0, "min_d": 20, "relevance": _low_relevance()},
            {"path": "b.jpg", "p": {"id": "b"}, "is_dup_free": 1, "size_ok": 1,
             "is_relevant": 0, "sharp_ok": 1, "aesthetic_val": 9.0,
             "luma_score": 0.0, "min_d": 20, "relevance": 0.9},
        ]
        base_winner, _ = ps._score_and_pick(candidates_info, None)
        # "a" прошёл гварды (is_relevant=1) хотя его сырая релевантность
        # низкая; "b" её провалил (is_relevant=0), несмотря на высокую сырую
        # релевантность и лучшую эстетику. Допустимое множество — только
        # "a", он и побеждает.
        assert base_winner["p"]["id"] == "a"
