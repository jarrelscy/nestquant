#!/bin/bash
# T27 pilot worker: ./run_pilot.sh GPU W NW  -- runs every arm on pairs[W::NW] (receipts make it idempotent).
cd "$(dirname "$0")" && source env.sh
export CUDA_VISIBLE_DEVICES=$1 NQ27_DECODE_DEV=cuda
W=$2; NW=$3; O=/tmp/nestquant/27-pv-tune
ALL=($(cat $O/pairs_all.txt)); FLAG=(${ALL[@]:0:8})
mine() { local a=("$@"); local out=(); for ((i=W; i<${#a[@]}; i+=NW)); do out+=("${a[i]}"); done; echo "${out[@]}"; }
R="--steps 1500 --batch 4096 --eval-every 100 --patience 4 --gpu-gb 10.5 --warmup 100 --lr 3e-3 --norm"
run() { local arm=$1; shift; nice -n 10 $PY nq27_run.py --arm $arm "$@" >> $O/logs/${arm}_w$W.log 2>&1; }
run act_c99 --objective act --cap-q 0.99 $R $(mine "${ALL[@]}")
run H_n     --objective H $R $(mine "${ALL[@]}")
run act_raw --objective act --steps 1500 --batch 4096 --eval-every 100 --patience 4 --gpu-gb 10.5 --lr 1e-2 $(mine "${ALL[@]}")
run act_c99_a25 --objective act --cap-q 0.99 --a 0.25 $R $(mine "${FLAG[@]}")
run act_c99_a75 --objective act --cap-q 0.99 --a 0.75 $R $(mine "${FLAG[@]}")
run act_c99_a100 --objective act --cap-q 0.99 --a 1.0 $R $(mine "${FLAG[@]}")
echo "worker $W done $(date -u)" >> $O/logs/pilot_done.txt
