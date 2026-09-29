#!/bin/bash
D=/data/Jarrel/nq-serve
until /data/Jarrel/coord/boxlease.sh try nestquant 60; do sleep 60; done
echo "--- kernel A/B rerun (stale build locks from the 06:40 abort) $(date -u +%H:%M)" >> $D/val.jsonl
cd /data/Jarrel/nq-dev/sm120; export NQ_BUILD=/data/Jarrel/nq-dev/build
for v in "0 NQ_PREFETCH" "1 NQ_PDL"; do set -- $v
  CUDA_VISIBLE_DEVICES=$1 timeout 1800 /data/Jarrel/nqenv/bin/python bench_variant.py /rawdata/Jarrel/nq-glm53-hf 10 $2 26,128 1,4,8 0 /data/Jarrel/nq-dev/var_${2}_L10.json > /data/Jarrel/nq-dev/var_${2}_L10.log 2>&1 &
done; wait
BV_SAME=1 CUDA_VISIBLE_DEVICES=0 timeout 1800 /data/Jarrel/nqenv/bin/python bench_variant.py /rawdata/Jarrel/nq-glm53-hf 10 NQ_PREFETCH 26,128 1,4,8 0 /data/Jarrel/nq-dev/var_l2same_L10.json > /data/Jarrel/nq-dev/var_l2same_L10.log 2>&1
echo "kernel A/B rerun done $(date -u +%H:%M)" >> $D/val.jsonl
/data/Jarrel/coord/boxlease.sh release nestquant
