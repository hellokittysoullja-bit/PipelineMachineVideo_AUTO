#!/usr/bin/env bash
# Копия платных и невосстановимых результатов 05_dospeh в ветку backup/05_dospeh.
# Отдельный worktree -> рабочая ветка и рендер не трогаются. Видео не копируется.
# Восстановление: git fetch origin backup/05_dospeh && скопировать snapshot/ обратно в videos/05_dospeh/.
R=/home/user/PipelineMachineVideo_AUTO
W=/home/user/bk_05_dospeh
B=backup/05_dospeh
cd $R || exit 1
if [ ! -d $W/.git ] && [ ! -f $W/.git ]; then
  if git ls-remote --exit-code origin "refs/heads/$B" >/dev/null 2>&1; then
    git fetch -q origin $B && git worktree add -q $W origin/$B -B $B
  else
    git worktree add -q --detach $W && (cd $W && git checkout -q --orphan $B && git rm -rq --cached . && find . -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +)
  fi
fi
S=$W/snapshot; mkdir -p $S
E=$R/videos/05_dospeh
for f in script.txt audio.mp3 audio_fixed.flac overnight.sh watchdog.sh backup.sh stall.sh; do [ -f $E/$f ] && cp -u $E/$f $S/; done
mkdir -p $S/media_plan $S/temp_smart
rsync -a --exclude '*.mp4' --exclude '*.jpg' --exclude '*.png' $E/media_plan/ $S/media_plan/ 2>/dev/null
for d in shot_judge_cache caption_screen_cache; do [ -d $E/temp_smart/$d ] && rsync -a $E/temp_smart/$d $S/temp_smart/; done
# крупнее 50 МБ в обычный git не кладём
find $S -size +50M -print -delete
cd $W
git add -A snapshot >/dev/null 2>&1
if ! git diff --cached --quiet; then
  git -c user.email=noreply@anthropic.com -c user.name=Claude commit -qm "backup 05_dospeh $(date -u +%FT%TZ)"
  for i in 1 2 3 4; do git push -q -u origin $B 2>/dev/null && break; sleep $((2**i)); done
  echo "$(date -u +%FT%TZ) backup: отправлено"
fi
