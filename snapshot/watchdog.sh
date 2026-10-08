#!/usr/bin/env bash
# Сторож рендера 05_dospeh (08.10). Каждые 2 минуты:
#  - надзиратель (overnight.sh) мёртв, а ролик не готов -> поднять;
#  - лог рендера молчит > STALL_SEC при живом pipeline_smart -> убить (надзиратель перезапустит с кэша);
#  - свободно < 1.5 ГБ -> громкая строка в watchdog.log;
#  - раз в 15 минут -> копия платных результатов в ветку backup/05_dospeh (backup.sh).
# Повторный запуск безопасен: живой сторож уже есть -> выход.
cd /home/user/PipelineMachineVideo_AUTO || exit 1
EP=videos/05_dospeh
LOG=$EP/watchdog.log
PIDF=$EP/watchdog.pid
STALL_SEC=${STALL_SEC:-1500}
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then exit 0; fi
echo $$ > "$PIDF"
say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
say "сторож запущен"
last_bk=0
while true; do
  state=$(cat $EP/overnight.state 2>/dev/null)
  case "$state" in
    done*) say "рендер завершён: $state"; bash $EP/backup.sh >> "$LOG" 2>&1; rm -f "$PIDF"; exit 0;;
  esac
  [ -f $EP/overnight.stop ] && { say "стоп-файл, выхожу"; rm -f "$PIDF"; exit 0; }
  sup=$(cat $EP/overnight.pid 2>/dev/null)
  if [ -f $EP/audio_fixed.flac ] && ! { [ -n "$sup" ] && kill -0 "$sup" 2>/dev/null; }; then
    say "надзиратель мёртв (state='$state') -> поднимаю"
    rm -f $EP/overnight.pid
    case "$state" in gave_up*) mv $EP/overnight.state $EP/overnight.state.gaveup.$(date +%s);; esac
    setsid nohup bash $EP/overnight.sh >/dev/null 2>&1 < /dev/null &
  fi
  if [ -f $EP/overnight.log ] && pgrep -f "scripts/pipeline_smart.py $EP" >/dev/null; then
    age=$(( $(date +%s) - $(stat -c %Y $EP/overnight.log) ))
    if [ $age -gt $STALL_SEC ]; then
      say "лог молчит ${age}с -> убиваю зависший рендер"
      echo "$(date -u +%FT%TZ) СТОРОЖ: лог молчал ${age}с, рендер убит" >> $EP/overnight.log
      pkill -f "scripts/pipeline_smart.py $EP"
    fi
  fi
  free_kb=$(df --output=avail / | tail -1)
  [ "$free_kb" -lt 1572864 ] && say "МАЛО МЕСТА: $((free_kb/1024)) МБ"
  now=$(date +%s)
  if [ $((now - last_bk)) -ge 900 ]; then bash $EP/backup.sh >> "$LOG" 2>&1; last_bk=$now; fi
  sleep 120
done
