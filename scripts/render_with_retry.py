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
OOM) от "процесс завершился сам, просто с ненулевым кодом". Ретраим только
первый случай: собственный ненулевой код — детерминированный результат,
повтор даст ровно то же самое.

Коды pipeline_smart.py (EXIT_* в его начале): 0 — чисто; 2 — ролик СОБРАН,
но есть пропуски/QC-дубли; 1 — final.mp4 НЕ собран. Раньше 1 означал оба
последних случая сразу, и эта обёртка на честном "не собрано" печатала
"ролик собран, есть замечания" — прямо противоположное правде.

Использование: python scripts/render_with_retry.py <video_dir> [--max N]
"""
import os
import subprocess
import sys
import time

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
# Путь к пайплайну — от МЕСТА ЭТОГО ФАЙЛА, а не относительный "scripts/...":
# относительный работал только при запуске строго из корня репозитория, а из
# любой другой рабочей папки (или из планировщика/cron с другим cwd) обёртка
# запускала python на несуществующий файл и "падала" 5 раз подряд с
# непонятной причиной вместо рендера.
PIPELINE = os.path.join(SCRIPTS_DIR, "pipeline_smart.py")

# Должны совпадать с EXIT_* в pipeline_smart.py (тест
# test_exit_codes_stay_in_sync_across_consumers следит за этим — импортировать
# оттуда напрямую значило бы тянуть в обёртку PIL/numpy/CLIP ради трёх чисел).
PIPELINE_EXIT_OK = 0
PIPELINE_EXIT_NOT_BUILT = 1
PIPELINE_EXIT_BUILT_WITH_WARNINGS = 2

MAX_ATTEMPTS_DEFAULT = 5
RETRY_DELAY_SEC = 15


def parse_args(argv):
    """Разбирает <video_dir> [--max N]. Раньше здесь понимался ТОЛЬКО
    `--max=N` (через знак равенства), при том что докстринг модуля и usage-
    сообщение документируют `--max N` (через пробел, как в примере
    использования выше) — вызов ровно по документации тихо игнорировал флаг
    (значение "5" просто оседало отдельным позиционным аргументом, не
    влияющим ни на что, а max_attempts молча оставался дефолтным — по
    совпадению тем же числом 5, что маскировало баг). Теперь поддержаны обе
    формы. Возвращает (video_dir|None, max_attempts)."""
    max_attempts = MAX_ATTEMPTS_DEFAULT
    positional = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--max":
            if i + 1 >= len(argv):
                raise ValueError("--max требует значение: --max N")
            max_attempts = int(argv[i + 1])
            i += 2
            continue
        if a.startswith("--max="):
            max_attempts = int(a.split("=", 1)[1])
            i += 1
            continue
        positional.append(a)
        i += 1
    video_dir = positional[0] if positional else None
    return video_dir, max_attempts


def main():
    try:
        video_dir, max_attempts = parse_args(sys.argv[1:])
    except ValueError as e:
        print(f"Использование: python scripts/render_with_retry.py <video_dir> [--max N] ({e})")
        return 1
    if not video_dir:
        print("Использование: python scripts/render_with_retry.py <video_dir> [--max N]")
        return 1

    for attempt in range(1, max_attempts + 1):
        print(f"=== Попытка {attempt}/{max_attempts} ===", flush=True)
        r = subprocess.run([sys.executable, PIPELINE, video_dir])
        if r.returncode == PIPELINE_EXIT_OK:
            print("Готово, без замечаний.")
            return PIPELINE_EXIT_OK
        if r.returncode == PIPELINE_EXIT_BUILT_WITH_WARNINGS:
            print("Ролик СОБРАН, но есть отражённые в отчётах замечания "
                  "(пропущенные блоки/QC-дубли) — это не обрыв, ретраить нечего. "
                  "Смотри media_plan/render_manifest.json и Шаг 7.5 (визуальный QC).")
            return r.returncode
        if r.returncode > 0:
            print(f"final.mp4 НЕ собран (код {r.returncode}) — процесс завершился сам, "
                  f"значит причина детерминированная и повтор даст то же самое. "
                  f"Причина — в логе выше и в media_plan/render_manifest.json.")
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
