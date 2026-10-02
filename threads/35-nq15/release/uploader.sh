#!/bin/bash
# Stage done-marked res files (hardlinks) and upload them; README/spec go up too.
# Never stages rank*.bin / rank*.json / artifact_stamp.json (those are held until gates pass).
R=/tmp/nestquant/35-nq15/release; S=$R/stage; REPO=jarrelscy/GLM-5.3-NestQuant-1.75-4bit
LOG=$R/logs/uploader.log
export HF_XET_HIGH_PERFORMANCE=1
while true; do
  n=0
  for m in $R/done/L*.json; do
    [ -e "$m" ] || continue; L=$(basename $m .json); L=${L#L}
    for r in 0 1 2 3; do
      mkdir -p $S/res/rank$r
      if [ ! -e $S/res/rank$r/L$L.pt ]; then ln $R/repo/res/rank$r/L$L.pt $S/res/rank$r/L$L.pt && n=$((n+1)); fi
    done
  done
  if [ $n -gt 0 ] || [ ! -e $R/logs/upload.first ]; then
    b=$(du -sbL --exclude=.cache $S | cut -f1); t0=$(date +%s)
    echo "$(date -u +%FT%TZ) upload start staged_bytes=$b new_files=$n" >> $LOG
    nice -n 10 hf upload-large-folder $REPO $S --repo-type model --num-workers 8 \
      --include 'res/*' --include 'README.md' --include 'NQ_RES_V2.md' \
      --exclude 'rank*' --exclude 'artifact_stamp.json' --no-bars >> $R/logs/upload_hf.log 2>&1
    rc=$?; t1=$(date +%s)
    echo "$(date -u +%FT%TZ) upload end rc=$rc secs=$((t1-t0))" >> $LOG
    touch $R/logs/upload.first
  fi
  if [ -e $R/done/ALL_DONE ] && [ $n -eq 0 ] && [ -e $R/logs/upload.first ]; then
    nres=$(ls $S/res/rank*/L*.pt | wc -l); [ $nres -eq 300 ] && { echo "$(date -u +%FT%TZ) res all staged+uploaded" >> $LOG; break; }
  fi
  sleep 120
done
