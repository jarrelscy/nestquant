#!/bin/bash
# T32 idea arms (CPU): band-all rows, v2 salience base + one idea at a time; then sim next_refresh (lag 1) + sync (lag 0).
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=/tmp/nestquant/18-e2e/pylib; PY=/home/coder/git/glm52/.venv/bin/python
M=/tmp/nestquant/32-gbdt-sal/models/ideas; L=/tmp/nestquant/32-gbdt-sal/logs/ideas; mkdir -p $M $L
B=ema32,ema128,mem_cur_state,tok_since_hit,hits16,sema32,sema128,sal16,mps128
CX=cx_e32,cx_e128,cx_s32,cx_s128; CO=co_e32,co_s32; LO=ema512,ema2048,sema512,sema2048
TK=tc_word,tc_num,tc_code,tc_punct,tc_ws,pos
declare -A F=( [base]="$B" [cx]="$B,$CX" [co]="$B,$CO" [long]="$B,$LO" [h32]="$B" [tok]="$B,$TK" [rank]="$B"
  [mtp1]="$B,mtp1_cnt,mtp1_sal" [mtp2]="$B,mtp2_cnt,mtp2_sal" [mtp4]="$B,mtp4_cnt,mtp4_sal" )
ARMS=${ARMS:-"base cx co long h32 tok rank mtp1 mtp2 mtp4"}
for a in $ARMS; do
  [ -f $M/$a.txt ] && continue
  tgt=sal; obj=tweedie:1.5
  [ $a = h32 ] && tgt=sal32
  [ $a = rank ] && obj=lambdarank
  nice -n 10 $PY train.py --band all --target $tgt --obj $obj --feats "${F[$a]}" --threads ${THR:-12} \
    ${EXTRA:-} --out $M/$a.txt > $L/train_$a.log 2>&1 &
done
wait
echo "trained $(date -u)"
