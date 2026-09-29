#!/bin/bash
# repack every layer (P4 records + resident serving planes) to the root NVMe as it lands: fit output (R/L{L}/tp*.pt)
# or the HF download (R/layers/L{L}/tp*.safetensors). Env: R, OUT, LOG, L0, L1, STEP (run several watchers on interleaved layers), GPU.
cd /data/Jarrel/nestquant
PY=/data/Jarrel/nqenv/bin/python;OUT=${OUT:-/home/jarrelscy/nq-p4rec/prod};R=${R:-/rawdata/Jarrel/nq-glm53-prod};LOG=${LOG:-/tmp/nq_repack_wait.log}
ready(){ local d=$R/L$1; [ -f $d/manifest.json ] || d=$R/layers/L$1; [ -f $d/manifest.json ] || return 1
  for s in 0 1 2 3 4 5 6 7; do [ -f $d/tp$s.pt ] || [ -f $d/tp$s.safetensors ] || return 1; done; }
# layers land in any order; repack whichever is ready until all are done
while :; do
  left=0
  for L in $(seq ${L0:-3} ${STEP:-1} ${L1:-77}); do
    [ -f $OUT/res/rank3/L$L.pt ] && continue; left=1
    ready $L || continue
    CUDA_VISIBLE_DEVICES=${GPU:-2} $PY streaming/repack.py $R $OUT 4 $L-$L >> $LOG 2>&1
    echo "L$L repacked $(date -u +%H:%M)" >> $LOG
  done
  [ $left = 0 ] && break; sleep 60
done
