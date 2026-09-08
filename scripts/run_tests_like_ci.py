# -*- coding: utf-8 -*-
"""Прогнать суиту так, как её гоняет CI-job `linux`: БЕЗ torch/transformers.

Зачем нужен отдельный скрипт, если есть `pytest tests/`. Потому что локально
ML-зависимости стоят, а job `linux` их намеренно не ставит (веса моделей
качает отдельный job `ml`, см. ЧАСТЬ 19 CLAUDE.md). Значит зелёная локальная
суита НЕ доказывает зелёный CI — и это не теория: тест с голым `import torch`
вместо `pytest.importorskip("torch")` ронял ветку три пуша подряд, оставаясь
зелёным на этой машине каждый раз.

Метод — не удаление пакетов, а блокировка импорта в дочернем процессе:
временный `sitecustomize.py` ставит в `sys.meta_path` хук, который на
заблокированные имена поднимает **ModuleNotFoundError**. Тип исключения здесь
принципиален: `pytest.importorskip()` пропускает тест на ModuleNotFoundError
(«модуля нет») и намеренно ПРОБРАСЫВАЕТ обычный ImportError («модуль есть, но
сломан») — чтобы не прятать настоящую поломку. Первая версия этого скрипта
поднимала ImportError и потому показывала падения там, где реальный CI
спокойно пропускает; поймано сравнением с логом настоящего прогона.

Запуск:
    .venv/bin/python scripts/run_tests_like_ci.py            # вся суита
    .venv/bin/python scripts/run_tests_like_ci.py tests/test_x.py -q
Всё, что после имени скрипта, уходит в pytest как есть.
"""
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ровно то, чего нет в job `linux`. Список держать в соответствии с
# .github/workflows/tests.yml: пакеты, которые ставит только job `ml`.
BLOCKED = ("torch", "transformers", "onnxruntime")

_SITECUSTOMIZE = '''
import sys
_BLOCKED = {blocked!r}


class _Blocker:
    def find_module(self, name, path=None):
        if name.split(".")[0] in _BLOCKED:
            raise ModuleNotFoundError("No module named '" + name + "'", name=name)
        return None


sys.meta_path.insert(0, _Blocker())
'''


def main(argv):
    with tempfile.TemporaryDirectory(prefix="ci_no_ml_") as tmp:
        with open(os.path.join(tmp, "sitecustomize.py"), "w", encoding="utf-8") as f:
            f.write(_SITECUSTOMIZE.format(blocked=set(BLOCKED)))

        env = dict(os.environ)
        # Свой путь ПЕРВЫМ: иначе чужой sitecustomize окружения победит.
        env["PYTHONPATH"] = os.pathsep.join(
            [tmp] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))

        args = argv or ["tests/"]
        print(f"Блокирую импорт: {', '.join(BLOCKED)}")
        print(f"pytest {' '.join(args)}\n")
        return subprocess.call(
            [sys.executable, "-m", "pytest", *args], cwd=REPO, env=env)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
