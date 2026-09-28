#!/bin/bash
# repack every fitted layer (P4 records + resident serving planes) to the root NVMe as the fit lands it
cd /data/Jarrel/nestquant
PY=/data/Jarrel/nqenv/bin/python;OUT=/home/jarrelscy/nq-p4rec/prod;R=/rawdata/Jarrel/nq-glm53-prod
for L in $(seq ${L0:-3} ${L1:-77}); do
  while [ ! -f $R/L$L/manifest.json ]; do sleep 60; done
  [ -f $OUT/res/rank3/L$L.pt ] || sleep 30
  CUDA_VISIBLE_DEVICES=${GPU:-2} $PY streaming/repack.py $R $OUT 4 $L-$L >> /tmp/nq_repack_wait.log 2>&1
  echo "L$L repacked $(date -u +%H:%M)" >> /tmp/nq_repack_wait.log
done
