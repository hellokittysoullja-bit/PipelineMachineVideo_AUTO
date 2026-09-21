"""Замер времени по стадиям пайплайна — честный 100%-ный разрез прогона.

ЗАЧЕМ: половина времени рендера живёт в subprocess ffmpeg, где Python-
профайлер (py-spy/cProfile) видит только `wait()`. Без этого модуля любая
оптимизация — спор о том, чья гипотеза красивее: реальный разбор бюджета
(01.09) дал два несовместимых ответа на вопрос «где время» именно потому,
что ни у кого не было замера, только реконструкция.

ПРИНЦИП БЕЗОПАСНОСТИ (важно, не косметика): таймеры НЕЛЬЗЯ ставить внутрь
функций, чей ИСХОДНЫЙ КОД хэшируется в pipeline_smart.render_recipe_
signature() (film_look/kenburns/video_render/parallax_kenburns/
choose_motion_mode/add_overlays/...) и candidate_gate_signature()
(is_relevant_candidate/image_sharpness_score/video_sharpness_ok/...) —
любая правка их тела, даже комментарий, меняет params_hash и молча
инвалидирует ВЕСЬ кэш temp_smart/ (реальный случай 01.09: monkeypatch
choose_motion_mode запустил незапланированный перерендер всех 165 клипов).
Такие стадии меряются СНАРУЖИ, с места вызова. visual_director.py
безопасен: его cache_signature() хэширует только константы, не исходники.

Выключен по умолчанию (STAGE_TIMER=0) — при выключенном флаге это пустой
контекст-менеджер без записи на диск и без обращения к времени.

Использование:
    from stage_timer import stage, set_output_path
    set_output_path("videos/01_ves-mecha/media_plan/stage_timings.jsonl")
    with stage("download", clip_idx=7, n=20):
        ...
Сводка:
    python scripts/stage_timer.py <stage_timings.jsonl>
"""
import json
import os
import sys
import time
from contextlib import contextmanager

STAGE_TIMER_ENABLED = os.environ.get("STAGE_TIMER", "0") != "0"

_output_path = os.environ.get("STAGE_TIMER_PATH", "") or None


def set_output_path(path):
    """Куда писать JSONL. Вызывается один раз из main() после того, как
    известен video_dir. Директория создаётся, если её нет.

    Путь дублируется в os.environ: пул рендера (ProcessPoolExecutor) на
    старте "spawn" НЕ наследует глобалы родителя — воркер импортирует
    модуль заново и прочитает путь из окружения. При "fork" оба механизма
    дают то же значение, конфликта нет."""
    global _output_path
    _output_path = path
    if path:
        os.environ["STAGE_TIMER_PATH"] = path
    if STAGE_TIMER_ENABLED and path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


@contextmanager
def stage(name, clip_idx=None, **extra):
    """Замер одной стадии. При STAGE_TIMER=0 — почти нулевой оверхед
    (один if, без time.perf_counter и без записи).

    Запись — одна строка JSON на событие, режим "a" (O_APPEND): пул
    рендера (ProcessPoolExecutor) пишет в тот же файл из НЕСКОЛЬКИХ
    процессов, и короткая строка < PIPE_BUF под O_APPEND не рвётся на
    Linux. Отдельный лок ради этого не заводим — он бы означал
    синхронизацию процессов ради телеметрии.

    Исключение внутри блока НЕ проглатывается: время всё равно
    записывается (с ok=False), исключение летит дальше — замер не должен
    менять поведение пайплайна ни в успехе, ни в сбое."""
    if not STAGE_TIMER_ENABLED or not _output_path:
        yield
        return
    t0 = time.perf_counter()
    ok = True
    try:
        yield
    except BaseException:
        ok = False
        raise
    finally:
        rec = {"stage": name, "t_wall": round(time.perf_counter() - t0, 4),
               "ok": ok, "pid": os.getpid(), "ts": round(time.time(), 3)}
        if clip_idx is not None:
            rec["clip_idx"] = clip_idx
        rec.update(extra)
        try:
            with open(_output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            # Телеметрия никогда не роняет прогон (диск полон и т.п.) —
            # тот же принцип fail-open, что у остальных опциональных слоёв.
            pass


def record(name, t_wall, **extra):
    """Записать уже измеренную длительность (когда обернуть блок
    контекст-менеджером мешает форма кода — например, длинный
    subprocess.run внутри try/except, переиндентация которого была бы
    правкой ради телеметрии). Семантика записи та же, что у stage()."""
    if not STAGE_TIMER_ENABLED or not _output_path:
        return
    rec = {"stage": name, "t_wall": round(t_wall, 4), "ok": True,
           "pid": os.getpid(), "ts": round(time.time(), 3)}
    rec.update(extra)
    try:
        with open(_output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass


def summarize(path):
    """Агрегат по JSONL: сумма/среднее/доля по стадиям — та самая таблица
    «до/после», которой требует CLAUDE.md перед любым разговором об
    ускорении."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if not rows:
        return "Пусто: ни одной записи."
    agg = {}
    for r in rows:
        a = agg.setdefault(r["stage"], {"n": 0, "total": 0.0, "fail": 0})
        a["n"] += 1
        a["total"] += r.get("t_wall", 0.0)
        if not r.get("ok", True):
            a["fail"] += 1
    grand = sum(a["total"] for a in agg.values())
    out = [f"{'стадия':<22} {'вызовов':>8} {'сумма, с':>11} {'среднее, с':>11} {'доля':>7}",
           "-" * 64]
    for name, a in sorted(agg.items(), key=lambda kv: -kv[1]["total"]):
        share = (a["total"] / grand * 100) if grand else 0.0
        fail = f"  (сбоев: {a['fail']})" if a["fail"] else ""
        out.append(f"{name:<22} {a['n']:>8} {a['total']:>11.1f} "
                   f"{a['total']/a['n']:>11.2f} {share:>6.1f}%{fail}")
    out.append("-" * 64)
    out.append(f"{'ИТОГО (сумма стадий)':<22} {len(rows):>8} {grand:>11.1f}")
    out.append("")
    out.append("ВНИМАНИЕ: стадии вложены и идут в НЕСКОЛЬКИХ процессах (пул рендера) —")
    out.append("сумма НЕ равна wall-clock прогона и может его превышать. Для доли")
    out.append("сравнивай стадии между собой, а не с временем на часах.")
    return "\n".join(out)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    print(summarize(sys.argv[1]))
