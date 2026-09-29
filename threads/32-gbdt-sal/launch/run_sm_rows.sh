#!/bin/bash
# training rows then count-only GBDT arms (poisson, next-64 hits, 60 trees x 15 leaves)
cd /home/coder/git/nestquant/threads/32-gbdt-sal
export PYTHONPATH=/tmp/nestquant/18-e2e/pylib; PY="nice -n 10 /home/coder/git/glm52/.venv/bin/python"
[ -f /tmp/nestquant/32-gbdt-sal/private/sm120/rows/calib-fit/meta.json ] || NPROC=16 $PY sm120.py rows calib-fit 4
[ -f /tmp/nestquant/32-gbdt-sal/private/sm120/rows/sm120dec/meta.json ] || NPROC=12 $PY sm120.py rows sm120dec 64
F5=ema32,ema128,mem_cur_state,tok_since_hit,hits16; LO=ema512,ema2048
LH=ema8192,cum_rate,pers8,pers32,pers128,gap_mean,gap_cv,dom_max,dom_busy; DOM=dom_max,dom_busy
for src in c:calib-fit: sA:sm120dec:A sB:sm120dec:B s:sm120dec:; do
  IFS=: read p st fo <<< "$src"
  for arm in base:$F5 long:$F5,$LO lh:$F5,$LO,$LH dom:$F5,$DOM; do
    n=${p}_${arm%%:*}; f=${arm#*:}
    [ -f /tmp/nestquant/32-gbdt-sal/models/sm120/$n.txt ] || THR=48 $PY sm120.py train $n $st $f $fo
  done
done
echo "trained $(date -u)"
