#!/bin/bash
cd /home/coder/git/nestquant/threads/33-search/scale
export PYTHONPATH=/tmp/nestquant/18-e2e/pylib; PY="nice -n 10 /home/coder/git/glm52/.venv/bin/python"
L=/tmp/nestquant/33-search/scale/logs
POOL=$($PY -c "
import numpy as np
val={3,7,11,15,19,23,27,31}; pool=[c for c in range(32) if c not in val]
for s in (0,1):
  p=list(np.random.default_rng(s).permutation(pool))
  for n in (3,6,12,24):
    if s==1 and n==24: continue
    print(f's{s}n{n}', ','.join(map(str,sorted(p[:n]))))
")
echo "$POOL" > $L/pool.txt
$PY train.py full32 --cf $(seq -s, 0 31) >> $L/train.log 2>&1
while read tag ch; do
  for cap in 1 4; do
    [ -f /tmp/nestquant/33-search/scale/models/${tag}_c$cap.txt ] || $PY train.py ${tag}_c$cap --cf $ch --cap $cap >> $L/train.log 2>&1
  done
done <<< "$POOL"
# stuck-flag retrained (serve today passes no ids)
$PY train.py s0n24_c1_stucktrain --cf $(grep s0n24 $L/pool.txt | cut -d' ' -f2) --stuck >> $L/train.log 2>&1
$PY train.py s0n24_c16 --cf $(grep s0n24 $L/pool.txt | cut -d' ' -f2) --cap 16 >> $L/train.log 2>&1
echo DONE >> $L/train.log
