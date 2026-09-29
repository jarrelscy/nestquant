#!/bin/bash
# T32 long-horizon / domain-memory arms (CPU): band-all v2 base + long EMAs (rows_x) + build_lh.py features;
# horizons next-64 / 128 / 256; combined + big (200 x 31).
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=/tmp/nestquant/18-e2e/pylib; PY=/home/coder/git/glm52/.venv/bin/python
M=/tmp/nestquant/32-gbdt-sal/models/ideas; L=/tmp/nestquant/32-gbdt-sal/logs/ideas; mkdir -p $M $L
B=ema32,ema128,mem_cur_state,tok_since_hit,hits16,sema32,sema128,sal16,mps128
LO=ema512,ema2048,sema512,sema2048
LH=ema8192,sema8192,cum_rate,cum_srate,pers8,pers32,pers128,gap_mean,gap_cv,dom_max,dom_busy,dom_smax
DOM=dom_max,dom_busy,dom_smax
run() { local a=$1; shift; [ -f $M/$a.txt ] || nice -n 10 $PY train.py --band all --threads ${THR:-12} "$@" --out $M/$a.txt > $L/train_$a.log 2>&1 & }
run lh      --target sal    --obj tweedie:1.5 --feats "$B,$LH"
run dom     --target sal    --obj tweedie:1.5 --feats "$B,$DOM"
run lhlong  --target sal    --obj tweedie:1.5 --feats "$B,$LO,$LH"
run lhl_t128 --target sal128 --obj tweedie:1.5 --feats "$B,$LO,$LH"
run lhl_t256 --target sal256 --obj tweedie:1.5 --feats "$B,$LO,$LH"
run base_t128 --target sal128 --obj tweedie:1.5 --feats "$B"
run base_t256 --target sal256 --obj tweedie:1.5 --feats "$B"
run big_lhlong --target sal --obj tweedie:1.5 --feats "$B,$LO,$LH" --iters 200 --leaves 31 --lr 0.1
wait
echo "trained $(date -u)"
