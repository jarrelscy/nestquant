#!/bin/bash
# T32 KLD pass (approved 2026-09-29): T18 decode harness, nq-tail + wikitext 32 windows each, contig shard, chain=1,
# one rank per GPU, all arms in one pass (paired in-pass).  Adapt arms carry salstat=1 (replay hot-route % and
# salience-weighted hot %).  static26 / static77 are Adapt equivalents of nqdef / nqfloat0 (n_float=0: fixed 26 only;
# refresh=chain/2: fixed + floating_default, never updated; CPU-verified identical level sets, parity_static.py).
# Waits for T18's passP1 to exit.
set -euo pipefail
TAG=${TAG:-passT32K}
LOGS=/tmp/nestquant/18-e2e/logs
while pgrep -f "[n]q_e2e.py run.*tag passP1" > /dev/null; do sleep 20; done
sleep 30
while IFS=, read -r i free; do
  [ "${free// /}" -ge 14000 ] || { echo "GPU $i only ${free}MiB free $(date -u)"; exit 1; }
done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
A=adapt:lo=/tmp/nestquant/18-e2e/farm_H/nq2,hi=/tmp/nestquant/18-e2e/farm_H_all4,manifest=/tmp/nestquant/28-serve-release/out/serving/tp4/manifest.json,chain=1,salstat=1
S=/home/coder/git/nestquant/streaming
export NQ_SHARD=contig OMP_NUM_THREADS=2 NQ_VRAM_GB=${NQ_VRAM_GB:-11}
echo "take GPUs $TAG $(date -u)"
pids=()
for r in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$r RANK=$r WORLD=8 nohup /home/coder/git/nestquant/threads/18-e2e-eval/run.sh run \
    --corpora nq-tail,wikitext --max-windows 32 --moe-chunk 16384 --tag $TAG \
    --cand ema=$A \
    --cand gbdt=$A,predictor=gbdt \
    --cand gbdt_sync=$A,predictor=gbdt,gmode=sync \
    --cand gbdt_x_mps=$A,predictor=gbdt,gbdt_scale=mps \
    --cand gbdt_x_mps_sync=$A,predictor=gbdt,gbdt_scale=mps,gmode=sync \
    --cand v2_sync_ba=$A,predictor=gbdt,gbdt_model=$S/gbdt_v2sal_p64.txt,gmode=sync,grlo=0,grhi=256 \
    --cand static26=$A,n_float=0 \
    --cand static77=$A,refresh=4096 \
    --cand orc_count=$A,oracle=count \
    --cand orc_sal=$A,oracle=sal \
    > $LOGS/$TAG.r$r.log 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait $p || rc=1; done
echo "release GPUs $TAG rc=$rc $(date -u)"
