#!/bin/bash
# Thread 19 progressive capture driver.
#   ./driver.sh stage1 <gpu> <shard> [<shard> ...]   run stage 1 for the listed shards sequentially on <gpu>
#   ./driver.sh stage2 <gpu>                          stage-2 merge worker (auto mode) on <gpu>
#   ./driver.sh plan <gpu> <id> [<id> ...]            stage 1 for shards of ROOT/plan.json (group corpora):
#       plan.json {"chunk0": [ids], "shards": {"<id>": {"corpus": dir, "fit_start": n, "fit_windows": n,
#                  "val_windows": n (-1 = whole val split, 0 = none), "matched": bool}}}
# Shard k = fit windows [2048 k, min(2048 (k+1), 28656)); shard 0 also writes the val + matched eval sets.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${NQ19_OUT:-/tmp/nestquant/19-capture}
mkdir -p $ROOT/shards $ROOT/logs
mode=$1; gpu=$2; shift 2
free_gb() { df -BG --output=avail $ROOT | tail -1 | tr -dc 0-9; }
if [ "$mode" = stage1 ]; then
  for k in "$@"; do
    kk=$(printf %02d $k); out=$ROOT/shards/s$kk
    if [ -f $out/state/progress.json ] && python3 -c "import json,sys;sys.exit(0 if json.load(open('$out/state/progress.json'))['next_layer']>77 else 1)"; then
      echo "shard $k already done"; continue; fi
    while [ $(free_gb) -lt 250 ]; do echo "$(date) disk low ($(free_gb) GB), waiting"; sleep 300; done
    start=$((2048*k)); nw=$((28656-start)); [ $nw -gt 2048 ] && nw=2048
    extra="--val-windows 0 --no-matched"; [ $k -eq 0 ] && extra="--val-windows 128"
    echo "$(date) shard $k windows $start+$nw on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu nice -n 5 $HERE/run.sh capture_fwd.py --out $out --fit-start $start --fit-windows $nw \
        $extra --ckpt-every ${NQ19_CKPT_EVERY:-4} --acts-budget-gb ${NQ19_ACTS_BUDGET_GB:-1500} >> $ROOT/logs/stage1_s$kk.log 2>&1 || { echo "shard $k FAILED"; exit 1; }
    rm -f $out/state/state_0.bf16 $out/state/state_1.bf16      # 13.7 GB each; not needed once all layers are done
    echo "$(date) shard $k done"
  done
elif [ "$mode" = plan ]; then
  for k in "$@"; do
    kk=$(printf %02d $k); out=$ROOT/shards/s$kk
    if [ -f $out/state/progress.json ] && python3 -c "import json,sys;sys.exit(0 if json.load(open('$out/state/progress.json'))['next_layer']>77 else 1)"; then
      echo "shard $k already done"; continue; fi
    while [ $(free_gb) -lt 250 ]; do echo "$(date) disk low ($(free_gb) GB), waiting"; sleep 300; done
    args=$(python3 -c "
import json; s=json.load(open('$ROOT/plan.json'))['shards']['$k']
print('--corpus', s['corpus'], '--shard-id', $k, '--fit-start', s['fit_start'], '--fit-windows', s['fit_windows'],
      '--val-windows', s.get('val_windows', 0), '' if s.get('matched') else '--no-matched')") || { echo "bad plan for $k"; exit 1; }
    echo "$(date) shard $k: $args on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu nice -n 5 $HERE/run.sh capture_fwd.py --out $out $args \
        --ckpt-every ${NQ19_CKPT_EVERY:-4} --acts-budget-gb ${NQ19_ACTS_BUDGET_GB:-1500} >> $ROOT/logs/stage1_s$kk.log 2>&1 || { echo "shard $k FAILED"; exit 1; }
    rm -f $out/state/state_0.bf16 $out/state/state_1.bf16
    echo "$(date) shard $k done"
  done
elif [ "$mode" = stage2 ]; then
  CUDA_VISIBLE_DEVICES=$gpu exec nice -n 5 $HERE/run.sh capture_stats.py --root $ROOT --keep-x-shards ${NQ19_KEEP_X:-0} "$@"
fi
