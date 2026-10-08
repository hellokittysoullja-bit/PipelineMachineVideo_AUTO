#!/usr/bin/env bash
# Надзиратель рендера 05_dospeh (08.10). Держит pipeline_smart.py живым:
# умер не с кодом 0/2 — пауза и перезапуск с кэша. Повторный запуск надзирателя
# безопасен: если жив прежний — выходит. Будильник сессии перезапускает его
# после рестарта контейнера.
cd /home/user/PipelineMachineVideo_AUTO || exit 1
EP=videos/05_dospeh
LOG=$EP/overnight.log
PIDF=$EP/overnight.pid
MAX_RESTARTS=${MAX_RESTARTS:-60}
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
  echo "$(date -u +%FT%TZ) надзиратель уже жив (pid $(cat "$PIDF"))" >> "$LOG"; exit 0
fi
echo $$ > "$PIDF"
set -a; . ./.env; set +a
export STAGE_TIMER=1 PYTHONUNBUFFERED=1

export SHOT_JUDGE_PAID_SLOTS=30 SHOT_JUDGE_MAX_SPEND=400000 IMAGE_GEN_MAX_SPEND=400000
n=0
while [ "$n" -lt "$MAX_RESTARTS" ]; do
  n=$((n+1))
  echo "$(date -u +%FT%TZ) === запуск $n (код $(git rev-parse --short HEAD)) ===" >> "$LOG"
  echo "running $n $(date -u +%FT%TZ)" > $EP/overnight.state
  python3 scripts/pipeline_smart.py $EP >> "$LOG" 2>&1
  rc=$?
  echo "$(date -u +%FT%TZ) выход rc=$rc" >> "$LOG"
  if [ "$rc" -eq 0 ] || [ "$rc" -eq 2 ]; then
    echo "done rc=$rc $(date -u +%FT%TZ)" > $EP/overnight.state
    rm -f "$PIDF"; exit 0
  fi
  if [ -f $EP/overnight.stop ]; then echo "stopped $(date -u +%FT%TZ)" > $EP/overnight.state; rm -f "$PIDF"; exit 0; fi
  sleep 60
done
echo "gave_up after $n $(date -u +%FT%TZ)" > $EP/overnight.state
rm -f "$PIDF"
