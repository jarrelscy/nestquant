#!/bin/bash
# once L3-L10 are repacked on every rank: scheduler+executor+engine smoke on real routing
cd /data/Jarrel/nestquant
PY=/data/Jarrel/nqenv/bin/python;OUT=/home/jarrelscy/nq-p4rec/prod
until $PY -c "import json,sys;sys.exit(0 if all(all(str(L) in json.load(open('$OUT/rank%d.json'%r))['layers'] for L in range(3,11)) for r in range(4)) else 1)" 2>/dev/null; do sleep 60; done
sleep 30
for r in 0 2; do
  CUDA_VISIBLE_DEVICES=1 $PY streaming/smoke_sched.py /rawdata/Jarrel/nq-glm53-prod $OUT 3-10 $r 512 > streaming/results/smoke_sched_L3-10_r$r.log 2>&1
  tail -2 streaming/results/smoke_sched_L3-10_r$r.log
done
