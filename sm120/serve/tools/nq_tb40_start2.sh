#!/bin/bash
# Prod NQ serve at 1M (phase-1 prefill, GBDT, uncapped) then tb4 c1 8h.
L=/data/Jarrel/nq-serve/nq_tb40_start2.log; exec >>"$L" 2>&1
echo "start $(date -u +%H:%M)"
/data/Jarrel/coord/boxlease.sh try nestquant 10080 || { echo "lease busy"; exit 1; }
S=/data/Jarrel/nestquant/sm120/serve/serve_nq.sh
export NQ_REPACK_DIR=/home/jarrelscy/nq-p4rec/hf NQ_LAYERS=3-77 NUM_SPEC=3 'ARVQ_CAPTURE_SIZES=[1,2,3,4,5,6,8,16,32]' \
  NQ_MAXLEN=1048576 NQ_UTIL=0.92 NQ_MAX_NUM_SEQS=1 NQ_PREDICTOR=gbdt NQ_CAP_GBPS=0
$S up || { echo "up failed"; exit 1; }
echo "ready $(date -u +%H:%M)"
$S smoke || { echo "smoke failed"; exit 1; }
docker logs glm53-nestquant 2>&1 | grep -E "predictor|KV cache size|max_model_len|max_num_batched" | tail -6
cd /home/jarrelscy/homeassistant/benchmarks && nohup /home/jarrelscy/.local/share/uv/tools/harbor/bin/python3 watch_nq_tb40_8h_c1.py > watch_nq_tb40_8h_c1.log 2>&1 &
echo "watcher pid $! $(date -u +%H:%M)"
