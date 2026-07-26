# ============================================================
#  FACELESS PIPELINE — контейнер headless-ядра сборки видео
#
#  Что контейнеризуется: FFmpeg + Python-скрипты (Ken Burns,
#  сборка слотов, сток-фетч, фикс пауз) — вся оффлайн-обработка
#  медиа. Воспроизводимое окружение "как у автора", один в один.
#
#  Что НЕ контейнеризуется (нужен хост/браузер/аккаунт):
#   - Claude Code сам (он оркестратор),
#   - Grok Imagine / Google Flow (сессия браузера),
#   - озвучка ElevenLabs (ручной шаг),
#   - DaVinci Resolve (GUI).
#  Их выполняет оператор/Claude на хосте; контейнер — только
#  детерминированная сборка mp4 из готовых media/ + audio.mp3.
#
#  Сборка:  docker build -t faceless-pipeline .
#  Запуск:  docker run --rm -v "$PWD/videos:/app/videos" \
#             --env-file .env faceless-pipeline \
#             python scripts/pipeline_smart.py videos/01_topic
# ============================================================
FROM python:3.12-slim

# Системные зависимости: ffmpeg для рендера, шрифты на всякий
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Скрипты пайплайна (идут в комплекте, генерировать ничего не нужно)
COPY scripts/ ./scripts/

# По умолчанию — показать доступные скрипты
CMD ["python", "-c", "import os; print('Pipeline ready. Scripts:', os.listdir('scripts'))"]
