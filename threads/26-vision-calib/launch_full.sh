#!/bin/bash
# T26 full vision capture into /tmp/nestquant/19-capture-mm (T19 stats format).
#   launch_full.sh <stage1_gpu> <stage2_gpu> [<stage2_gpu> ...]
# Stage 1 (capture_mm_fwd.py = capture_fwd_fast.py + image splice) runs shard s00 = all 198 fit windows + the 33
# held-out windows (eval/val); stage-2 workers (capture_mm_stats.py, auto mode, per-layer locks) merge each layer as
# soon as stage 1 has written it, and exit after 15 idle minutes.  Then: state files removed, small-only backup.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
R=${NQ26_ROOT:-/tmp/nestquant/19-capture-mm}
C=$R/corpus/c2048_mm
mkdir -p $R/shards $R/logs
[ -f $C/feats.json ] || { echo "no $C/feats.json (run mm_corpus.py + vis_feats.py first)"; exit 1; }
NF=$(python3 -c "import json;print(json.load(open('$C/split.json'))['fit'][1])")
[ -f $R/plan.json ] || cat > $R/plan.json <<J
{"chunk0": [0], "note": "T26 vision capture: s00 = all $NF fit windows of $C + whole val split (held-out 200 samples)",
 "shards": {"0": {"corpus": "$C", "fit_start": 0, "fit_windows": $NF, "val_windows": -1, "matched": false}}}
J
g1=$1; shift
echo "$(date -u +%FT%TZ) stage1 on GPU $g1, stage2 on GPUs $*" >> $R/logs/launch.log
( CUDA_VISIBLE_DEVICES=$g1 NQ19_GPU_GB=12 nice -n 5 $HERE/run.sh capture_mm_fwd.py --out $R/shards/s00 --corpus $C \
    --shard-id 0 --fit-start 0 --fit-windows $NF --val-windows -1 --no-matched --ckpt-every 4 --acts-budget-gb 400 \
    >> $R/logs/stage1_s00.log 2>&1; echo "$(date -u +%FT%TZ) stage1 rc=$?" >> $R/logs/launch.log ) &
pids=""
i=0
for g in "$@"; do
  ( CUDA_VISIBLE_DEVICES=$g NQ19_GPU_GB=12 nice -n 5 $HERE/run.sh capture_mm_stats.py --root $R --keep-x-shards "" \
      --max-shards 1 --exit-when-idle 900 >> $R/logs/stage2_w$i.log 2>&1 ) &
  pids="$pids $!"; i=$((i+1))
done
wait
n=$(ls -d $R/stats/L*.v1 2>/dev/null | wc -l)
echo "$(date -u +%FT%TZ) stage2 workers done; $n layer stats" >> $R/logs/launch.log
if [ "$n" = 75 ]; then
  rm -f $R/shards/s00/state/state_0.bf16 $R/shards/s00/state/state_1.bf16
  ln -sfn shards/s00/eval $R/eval
  touch $R/VISION_STATS_READY
  $HERE/fb_backup26.sh backup 19-capture-mm --no-follow-symlinks --exclude "*.f32" --exclude "shards/*/acts/*/x.bf16" \
      --exclude "shards/*/state/*" --exclude "shards/*/eval/*" --exclude "bnd_rows/*/x.bf16" >> $R/logs/launch.log 2>&1
fi
