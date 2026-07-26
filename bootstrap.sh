#!/usr/bin/env bash
# ============================================================
#  FACELESS PIPELINE — Bootstrap (Linux / macOS)
#  Разворачивает окружение: ffmpeg, Python venv, зависимости,
#  структуру папок, конфиг. Идемпотентен.
#
#  Claude Code запускает это сам при первом старте (см. PART 0
#  в CLAUDE.md). Вручную:  bash bootstrap.sh
# ============================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "==> Bootstrap в $ROOT"

# 1. FFmpeg ----------------------------------------------------
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "==> FFmpeg не найден, ставлю..."
  if command -v apt-get >/dev/null 2>&1; then sudo apt-get update && sudo apt-get install -y ffmpeg
  elif command -v brew >/dev/null 2>&1; then brew install ffmpeg
  elif command -v dnf >/dev/null 2>&1; then sudo dnf install -y ffmpeg
  else echo "!! Поставь FFmpeg вручную: https://ffmpeg.org/download.html"; fi
else
  echo "==> FFmpeg на месте"
fi

# 2. Python venv ----------------------------------------------
PY=$(command -v python3 || command -v python)
[ -z "$PY" ] && { echo "Python не найден. Поставь Python 3.11+"; exit 1; }
if [ ! -d "$ROOT/.venv" ]; then echo "==> Создаю venv..."; "$PY" -m venv "$ROOT/.venv"; fi
echo "==> Ставлю Python-зависимости..."
"$ROOT/.venv/bin/pip" install --upgrade pip >/dev/null
"$ROOT/.venv/bin/pip" install -r "$ROOT/requirements.txt"

# 3. Структура папок ------------------------------------------
mkdir -p "$ROOT/videos" "$ROOT/scripts" "$ROOT/secrets"
echo "==> Папки готовы (videos, scripts, secrets)"

# 4. Конфиг ключей --------------------------------------------
if [ ! -f "$ROOT/.env" ]; then
  cp "$ROOT/config.example.env" "$ROOT/.env"
  echo "==> Создан .env — ВПИШИ туда свои ключи"
else
  echo "==> .env уже есть"
fi

# 5. Маркер завершения ----------------------------------------
echo "setup ok $(date -Iseconds)" > "$ROOT/.setup_complete"
echo ""
echo "==> Готово. Дальше: 1) заполни .env  2) Claude сгенерит скрипты в scripts/ по SCRIPTS SPEC из CLAUDE.md"
