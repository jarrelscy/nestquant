#!/bin/bash
# Aggregate throughput on one GPU: tput.sh GPU NPROC GROUP NEXP  -> experts/s over all procs (wall incl. startup)
G=$1; N=$2; GR=$3; NE=${4:-12}
cd /home/coder/git/nestquant/threads/23-encode-throughput
source ../12-reference-encoder/env.sh >/dev/null 2>&1
export NQ23_T12=${NQ23_T12:-/tmp/nestquant/23-encode-throughput/t12pin_f128} CUDA_VISIBLE_DEVICES=$G
PY=/home/coder/git/glm52/.venv/bin/python
S=/tmp/nestquant/23-encode-throughput; LG=$S/logs/tput; mkdir -p $LG
t0=$(date +%s.%N); pids=()
for i in $(seq 0 $((N-1))); do
  L=$((10 + 7*i)); ex=$(for e in $(seq 0 $((NE-1))); do printf "%d:%d," $L $((e*5+i)); done); ex=${ex%,}
  tag=tput/n${N}g${GR}_$i; rm -rf $S/$tag
  $PY check_bitid.py batch --tag $tag --group $GR --experts $ex > $LG/n${N}g${GR}_$i.log 2>&1 &
  pids+=($!)
done
for p in ${pids[@]}; do wait $p; done
t1=$(date +%s.%N)
tot=$(grep -h "batch total" $LG/n${N}g${GR}_*.log | wc -l)
python3 -c "
import sys; w=$t1-$t0; n=$N*$NE
print(f'GPU $G procs $N group $GR: {n} experts in {w:.0f}s wall = {w/n:.2f} s/expert aggregate ({n/w*3600:.0f} experts/h); finished procs $tot/$N')"
grep -h "batch total" $LG/n${N}g${GR}_*.log
