#!/bin/bash
# T32 decisive KLD: T18 decode harness, nq-tail + wikitext 32 windows each, contig shard, chain=1, one rank per GPU.
# Arms: gbdt (gbdt_p64_s5) | gbdt_v2sal (9-feature salience model) | gbdt_x_mps (old model x EMA128 sal/hit) |
#       orc_sal (T18 anchor).  Waits until passD3 is gone.
set -euo pipefail
TAG=${TAG:-passT32}
LOGS=/tmp/nestquant/18-e2e/logs
while pgrep -f "tag passD3" > /dev/null; do sleep 20; done
while IFS=, read -r i free; do
  [ "${free// /}" -ge 14000 ] || { echo "GPU $i only ${free}MiB free"; exit 1; }
done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
A=adapt:lo=/tmp/nestquant/18-e2e/farm_H/nq2,hi=/tmp/nestquant/18-e2e/farm_H_all4,manifest=/tmp/nestquant/28-serve-release/out/serving/tp4/manifest.json,chain=1
S=/home/coder/git/nestquant/streaming
export NQ_SHARD=contig OMP_NUM_THREADS=2 NQ_VRAM_GB=${NQ_VRAM_GB:-11}
echo "launch $TAG $(date -u)"
pids=()
for r in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$r RANK=$r WORLD=8 nohup /home/coder/git/nestquant/threads/18-e2e-eval/run.sh run \
    --corpora nq-tail,wikitext --max-windows 32 --moe-chunk 16384 --tag $TAG \
    --cand gbdt=$A,predictor=gbdt \
    --cand gbdt_v2sal=$A,predictor=gbdt,gbdt_model=$S/gbdt_v2sal_p64.txt \
    --cand gbdt_x_mps=$A,predictor=gbdt,gbdt_scale=mps \
    --cand orc_sal=$A,oracle=sal \
    > $LOGS/$TAG.r$r.log 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done
echo "done $(date -u)"
