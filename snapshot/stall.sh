#!/bin/bash
# Следит за зависанием: лог не менялся >20 мин при живом рендере -> убить python, надзиратель поднимет сам.
cd /home/user/PipelineMachineVideo_AUTO/videos/05_dospeh
end=$(( $(date +%s) + 43200 ))
while [ $(date +%s) -lt $end ]; do
  case "$(cat overnight.state)" in done*|gave_up*) echo FINISHED; exit 0;; esac
  age=$(( $(date +%s) - $(stat -c %Y overnight.log) ))
  if [ $age -gt 1200 ] && pgrep -f "scripts/pipeline_smart.py" >/dev/null; then
    echo "$(date -u +%T) лог молчит ${age}с — убиваю рендер для перезапуска"
    echo "$(date -u +%FT%TZ) СТОРОЖ: лог молчал ${age}с, убит рендер" >> overnight.log
    pkill -f "scripts/pipeline_smart.py"; sleep 200
  fi
  sleep 120
done
echo "окно сторожа закончилось"
