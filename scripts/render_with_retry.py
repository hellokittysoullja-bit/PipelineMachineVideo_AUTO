"""Обёртка над pipeline_smart.py с авто-перезапуском при ОБРЫВЕ процесса
(SIGKILL/OOM/убитый контейнер — реальный случай, пойманный вживую: много-
часовой фоновый рендер один раз пропал без единой строчки трейсбека).

Безопасно ретраить можно ТОЛЬКО благодаря атомарной записи клипов
(render_tmp_path()/finalize_render() в pipeline_smart.py) — до неё обрыв
посреди рендера клипа оставлял ОБРЕЗАННЫЙ mp4 ровно под кэш-именем, и
повторный запуск тихо принимал бы битый файл за готовый. Теперь клип либо
полностью готов под своим именем, либо файла там нет вообще — повторный
запуск просто доделывает то, что не успело доехать.

Отличаем "процесс убит сигналом" (retcode < 0 на POSIX — SIGKILL/SIGTERM/
OOM) от "процесс завершился сам, просто с ненулевым кодом" (retcode 1 —
это pipeline_smart.py честно сигналит про пропущенные кадры/QC-дубли,
см. конец main() — ролик УЖЕ собран, ретраить нечего, результат не
изменится). Ретраим только первый случай.

Использование: python scripts/render_with_retry.py <video_dir> [--max N]
"""
import subprocess
import sys
import time

MAX_ATTEMPTS_DEFAULT = 5
RETRY_DELAY_SEC = 15


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--max")]
    max_attempts = MAX_ATTEMPTS_DEFAULT
    for a in sys.argv[1:]:
        if a.startswith("--max="):
            max_attempts = int(a.split("=", 1)[1])
    if not args:
        print("Использование: python scripts/render_with_retry.py <video_dir> [--max=N]")
        return 1
    video_dir = args[0]

    for attempt in range(1, max_attempts + 1):
        print(f"=== Попытка {attempt}/{max_attempts} ===", flush=True)
        r = subprocess.run([sys.executable, "scripts/pipeline_smart.py", video_dir])
        if r.returncode == 0:
            print("Готово, без замечаний.")
            return 0
        if r.returncode > 0:
            print(f"Процесс завершился сам (код {r.returncode}) — ролик собран, "
                  f"есть отражённые в отчёте замечания (пропуски/QC), это не обрыв, ретраить нечего.")
            return r.returncode
        # retcode < 0 -> убит сигналом (POSIX: -9 SIGKILL, -15 SIGTERM и т.п.)
        print(f"Процесс убит сигналом {-r.returncode} — не штатное завершение. "
              f"Кэш клипов на диске цел (атомарная запись), пробую снова через {RETRY_DELAY_SEC}с...")
        if attempt < max_attempts:
            time.sleep(RETRY_DELAY_SEC)

    print(f"Не удалось завершить за {max_attempts} попыток — обрыв повторяется систематически, "
          f"это уже не транзиентный сбой, нужно разбираться руками (ресурсы окружения?).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
