#!/bin/bash
# One-command SM120 bench. Results (JSON) land in sm120/results/bench_sm120/ and streaming/results/bench_sm120/.
#   sm120/bench_sm120.sh [ROOT=/rawdata/Jarrel/nq-glm53-prod] [REPACK=/home/jarrelscy/nq-p4rec/prod] [LAYERS=3-10]
# Steps: kernel us vs EXL3 (TP4 shard), repack of any fitted layer not yet repacked, SSD sweep, per-layer stream smoke,
# scheduler sim on tb4 routing, scheduler+executor smoke (normal, SSD unavailable, pool full), graph-safety timing
# with all 4 ranks streaming at once. Run on an idle box: timings taken next to other GPU jobs are not meaningful.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/data/Jarrel/nqenv/bin/python}
ROOT=${1:-/rawdata/Jarrel/nq-glm53-prod}; RP=${2:-/home/jarrelscy/nq-p4rec/prod}; LY=${3:-3-10}
A=${LY%-*}; Z=${LY#*-}
S=sm120/results/bench_sm120; T=streaming/results/bench_sm120; mkdir -p $S $T
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$busy" -gt 0 ] && echo "WARNING: $busy other GPU processes running; timings will be noisy"
echo "== kernel: NQ vs EXL3, TP4 shard"
(cd sm120 && CUDA_VISIBLE_DEVICES=0 $PY bench_tp4.py results/bench_sm120/bench_tp4.json > results/bench_sm120/bench_tp4.log 2>&1); tail -n 3 $S/bench_tp4.log
echo "== repack $LY"
CUDA_VISIBLE_DEVICES=0 $PY streaming/repack.py $ROOT $RP 4 $LY > $T/repack.log 2>&1; tail -n 2 $T/repack.log
echo "== SSD sweep"
(cd streaming && OUT=results/bench_sm120/ssd_sweep.jsonl bash ssd_sweep.sh > /dev/null 2>&1); wc -l < $T/ssd_sweep.jsonl
echo "== stream smoke (io_uring engine), every layer x rank"
for L in $(seq $A $Z); do for r in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$r $PY streaming/smoke_stream.py $ROOT $RP $L $r 4 64 300 1 2>&1 | grep -v "^    step" >> $T/smoke_stream.log &
done; wait; done
echo "  $(grep -c 'STREAM SMOKE PASS' $T/smoke_stream.log) pass / $(grep -c 'STREAM SMOKE' $T/smoke_stream.log)"
echo "== scheduler sim (tb4 routing)"
: > $T/sched_sim.jsonl
for cfg in "1 6 512" "8 6 512" "1 1000 512"; do $PY streaming/sched_sim.py $cfg 1000 | tee -a $T/sched_sim.jsonl; done
echo "== scheduler + executor smoke"
CUDA_VISIBLE_DEVICES=0 $PY streaming/smoke_sched.py $ROOT $RP $LY 0 512 | tail -n 1 > $T/smoke_sched.json
dd if=$RP/rank1.bin of=$T/fault_rank1.bin bs=2560000 count=256 status=none
NQ_FAULT_FILE=$T/fault_rank1.bin CUDA_VISIBLE_DEVICES=1 $PY streaming/smoke_sched.py $ROOT $RP $LY 1 256 | tail -n 1 > $T/smoke_sched_fault_ssd.json &
CUDA_VISIBLE_DEVICES=2 $PY streaming/smoke_sched.py $ROOT $RP $LY 2 256 24 | tail -n 1 > $T/smoke_sched_fault_pool.json &
wait; rm -f $T/fault_rank1.bin; cat $T/smoke_sched*.json
echo "== graph safety: 4 ranks streaming at once"
for ops in 1 4; do
  for r in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$r $PY streaming/graph_timing.py $ROOT $RP $LY $r 4 10 200 $ops | tail -n 1 > $T/graph_timing_ops${ops}_r$r.json & done; wait
  cat $T/graph_timing_ops${ops}_r*.json
done
