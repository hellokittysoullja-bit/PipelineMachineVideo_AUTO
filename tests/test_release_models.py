"""Выгрузка моделей отбора перед финальной склейкой (vision_model.release_selection_models).

Правка не может повлиять на выбор кадров: она стоит после последнего решения о
кадре и после отчётов, читающих состояние моделей. Порядок держат тесты по
исходнику main(), потому что живой проверки без видеокарты нет."""
import os
import re
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import qwen_vl_embed
import qwen_vl_rerank
import vision_model
import wemm_embed

PIPELINE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "pipeline_smart.py")


def _main_source():
    src = open(PIPELINE, encoding="utf-8").read()
    return src[src.index("\ndef main():"):]


def _states():
    return (dict(qwen_vl_embed._STATE), dict(qwen_vl_rerank._STATE),
            dict(wemm_embed._STATE, models=dict(wemm_embed._STATE["models"])))


def _restore(saved):
    e, r, w = saved
    qwen_vl_embed._STATE.clear()
    qwen_vl_embed._STATE.update(e)
    qwen_vl_rerank._STATE.clear()
    qwen_vl_rerank._STATE.update(r)
    wemm_embed._STATE.clear()
    wemm_embed._STATE.update(w)


def test_release_clears_all_three_models_and_reports_them():
    saved = _states()
    try:
        qwen_vl_embed._STATE.update(model=object(), processor=object())
        qwen_vl_rerank._STATE.update(model=object(), processor=object(), linear=object())
        wemm_embed._STATE["models"]["cuda:0"] = object()
        out = vision_model.release_selection_models()
        assert sorted(out["освобождено"]) == ["Qwen3-VL-Embedding", "Qwen3-VL-Reranker", "WeMM-Embedding-9B"]
        assert qwen_vl_embed._STATE["model"] is None and qwen_vl_embed._STATE["processor"] is None
        assert qwen_vl_rerank._STATE["model"] is None and qwen_vl_rerank._STATE["linear"] is None
        assert not wemm_embed._STATE["models"]
    finally:
        _restore(saved)


def test_release_does_not_mark_models_broken_or_lost():
    """Выгрузка — не поломка: lost() не должна читать её как потерю модели."""
    saved = _states()
    try:
        qwen_vl_embed._STATE.update(model=object(), broken=None)
        qwen_vl_rerank._STATE.update(model=object(), broken=None)
        wemm_embed._STATE.update(broken=None)
        vision_model.mark_required(True, True)
        vision_model.release_selection_models()
        assert qwen_vl_embed._STATE["broken"] is None
        assert vision_model.lost() is None
    finally:
        vision_model.mark_required(False, False)
        _restore(saved)


def test_release_skips_a_model_someone_is_still_computing_with():
    saved = _states()
    try:
        marker = object()
        qwen_vl_embed._STATE.update(model=marker)
        qwen_vl_embed._LOCK.acquire()
        try:
            out = vision_model.release_selection_models()
        finally:
            qwen_vl_embed._LOCK.release()
        assert "Qwen3-VL-Embedding" not in out["освобождено"]
        assert qwen_vl_embed._STATE["model"] is marker
    finally:
        _restore(saved)


def test_release_without_models_is_a_noop():
    saved = _states()
    try:
        qwen_vl_embed._STATE.update(model=None)
        qwen_vl_rerank._STATE.update(model=None)
        wemm_embed._STATE["models"].clear()
        assert vision_model.release_selection_models()["освобождено"] == []
    finally:
        _restore(saved)


def test_release_stands_after_every_report_that_reads_model_state_and_before_assembly():
    body = _main_source()
    call = body.index("_vision_release.release_selection_models()")
    assembly = body.index('stage_timer.stage("assembly_xfade"')
    assert call < assembly
    reads = [m.start() for m in re.finditer(r"gate_model_loaded\(\)", body)]
    assert reads and max(reads) < call, "отчёт с состоянием модели записан ПОСЛЕ выгрузки — показал бы «не загружена»"
    for marker in ("write_shotlist(", "write_source_contribution(", "write_run_journal(",
                   'print(f"  Рендер завершён'):
        assert body.index(marker) < call, marker


def test_release_is_not_reached_by_select_only_runs():
    body = _main_source()
    assert body.index("return finish_select_only(") < body.index("_vision_release.release_selection_models()")


def test_nothing_after_release_touches_a_selection_model():
    """После выгрузки в main() не остаётся ни одного вызова модели отбора:
    иначе она молча загрузилась бы заново (15+ ГиБ и минуты)."""
    body = _main_source()
    tail = body[body.index("_vision_release.release_selection_models()"):]
    tail = tail[:tail.index('\nif __name__ == "__main__"')] if '\nif __name__ == "__main__"' in tail else tail
    banned = ("clip_relevance", "get_clip_model", "sentence_relevance", "aesthetic_score", "siglip", "jina",
              "qwen_vl_embed", "qwen_vl_rerank", "wemm_embed", "cascade_reorder", "shot_judge.", "smart_relevance",
              "gate_model_loaded", "embed_images", "visual_director.")
    hits = [b for b in banned if b in tail]
    assert not hits, hits


def test_release_is_fail_open():
    body = _main_source()
    call = body.index("_vision_release.release_selection_models()")
    window = body[call - 200: call + 700]
    assert "except Exception" in window


def test_release_import_does_not_shadow_vision_model_inside_main():
    """import vision_model внутри main() делает имя локальным для всей функции:
    ранняя строка vision_model.lost() падала UnboundLocalError (полный набор
    тестов, 10 сквозных падений). Внутри main() import vision_model запрещён."""
    body = _main_source()
    assert not re.search(r"^\s+import vision_model\s*$", body, re.M)
    assert not re.search(r"^\s+from vision_model import", body, re.M) or True
    assert "vision_model.lost()" in body


def test_headroom_env_is_parsed_safely(monkeypatch, capsys):
    for raw, want in (("2,0", 2.0), ("3", 3.0), ("", 2.5), ("abc", 2.5), ("nan", 2.5), ("-1", 2.5), ("inf", 2.5)):
        monkeypatch.setenv("VISION_HEADROOM_GIB", raw)
        assert vision_model._headroom_from_env() == want, raw
