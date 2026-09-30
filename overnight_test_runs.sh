#!/usr/bin/env bash
# Overnight visual-only test renders across 4 different niches, to review
# which real stock candidates the SigLIP2-base256 gate picks per phrase.
set -u
cd /home/user/PipelineMachineVideo_AUTO

export MUSIC_BED=0 AMBIENCE_BED=0 TYPEWRITER_CLICKS=0 REVEAL_SFX=0 \
       SFX_DIRECTOR=0 VOICE_PROCESS=0 RENDER_STRICT_GATE=0

EPISODES="90_egypt_mummies 91_apollo_space 92_procrastination 93_deep_sea"

for ep in $EPISODES; do
    echo "=================================================="
    echo "START $ep  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=================================================="
    timeout 3600 python3 scripts/pipeline_smart.py "videos/$ep" \
        > "videos/$ep/render_log.txt" 2>&1
    rc=$?
    echo "  exit code: $rc"
    if [ -f "videos/$ep/final.mp4" ]; then
        echo "  final.mp4 OK: $(ffprobe -v error -show_entries format=duration,size -of default=noprint_wrappers=1 "videos/$ep/final.mp4")"
    else
        echo "  final.mp4 MISSING"
        tail -60 "videos/$ep/render_log.txt"
    fi
    python3 scripts/shotlist_contact.py "videos/$ep" > "videos/$ep/contact_log.txt" 2>&1
    echo "  contact sheets: $(ls videos/$ep/media_plan/shotlist_contact_*.jpg 2>/dev/null | wc -l)"
    echo "DONE $ep  $(date '+%Y-%m-%d %H:%M:%S')"
    echo
done

echo "=================================================="
echo "ALL EPISODES FINISHED  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================="
