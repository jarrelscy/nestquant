#!/bin/bash
# after each fitted layer lands: repack it to the root NVMe, then run the stream smoke on every rank
cd /data/Jarrel/nestquant
PY=/data/Jarrel/nqenv/bin/python;OUT=/home/jarrelscy/nq-p4rec/prod
for L in 3 4 5 6 7 8 9 10; do
  while [ ! -f /rawdata/Jarrel/nq-glm53-prod/L$L/manifest.json ]; do sleep 60; done
  sleep 120
  CUDA_VISIBLE_DEVICES=2 $PY streaming/repack.py /rawdata/Jarrel/nq-glm53-prod $OUT 4 $L-$L > streaming/results/repack_L$L.log 2>&1
  for r in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=2 $PY streaming/smoke_stream.py /rawdata/Jarrel/nq-glm53-prod $OUT $L $r 4 64 300 1 2>&1 | grep -v "^    step" >> streaming/results/smoke_stream_prod_L$L.log
  done
  echo "L$L $(grep -c 'STREAM SMOKE PASS' streaming/results/smoke_stream_prod_L$L.log)/4 pass"
done
