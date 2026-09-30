#!/bin/bash
# T33i jF-noresid learning curve on sm120tf (coordinator 2026-09-30): GBDT-free (no v2 input / residual), task-level
# folds F1-F3 (test = the 2 held-out tasks), train = 25% of 1 task | 1 | 2 | 4 tasks; fixed recipe, last checkpoint.
# CPU only (no GPU until run 1 ends): 2 workers x 8 threads, nice 10.  Logs + results: models/lc_*.{log,json}
cd /home/coder/git/nestquant/threads/33-search/joint
export LAYOUT=k0 OMP_NUM_THREADS=8
PY=/home/coder/git/prime-radiant/.venv/bin/python; LG=/tmp/nestquant/33-search/joint/logs/lc; mkdir -p $LG
E=embedding-drift-monitor; F=fin-saccr-rwa; FC=formal-crypto; S=sound-change-cascade; FR=freight-dispatch-shift; P=pretrain-shard-corruption
job() {  # NAME TEST TRAIN FRAC
  [ -f /tmp/nestquant/33-search/joint/models/$1.json ] && return
  nice -n 10 $PY -u trainh.py $1 --dev cpu --threads 8 --bs 128 --steps 6000 --max-epochs 4 --eval-blocks 2048 \
    --test-tasks $2 --train-tasks $3 --frac $4 > $LG/$1.log 2>&1
}
fold() {  # F TEST ONE TWO FOUR
  job lc_$1_q1 $2 $3 0.25; job lc_$1_t1 $2 $3 1; job lc_$1_t2 $2 $4 1; job lc_$1_t4 $2 $5 1
}
fold F1 $E,$F $FC $FC,$FR $FC,$S,$FR,$P &
fold F2 $FC,$S $F $F,$FR $E,$F,$FR,$P &
wait
fold F3 $FR,$P $F $F,$FC $E,$F,$FC,$S
echo "lcurve done $(date -u)"
