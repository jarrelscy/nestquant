#!/bin/bash
until grep -q PRE_DONE /tmp/nq_inkling/logs/pre_mm.log; do sleep 60; done
cd /tmp/nq_inkling/code
ev() {  # name wdir wsel ref [mm]
  O=/tmp/nq_inkling/kld/$1; [ -f $O/final/kld.json ] && return; rm -rf $O
  env ${5:+INK_MM=/tmp/nq_inkling/corpus_mm} flock /tmp/nq_inkling/kld.lock /tmp/nq_inkling/venv/bin/torchrun --nproc-per-node 8 --master-port 29718 cap_ink.py \
    --out $O --layers 0-47 --nfit 0,0 --no-stats --wdir /tmp/nq_inkling/$2 --wsel $3 --wmax 48 --ref /tmp/nq_inkling/$4 > $O.log 2>&1
  echo "$1 $(grep 'KLD vs ref' $O.log)"
}
ev mm48_2 pre_mm 2 ref; ev mm48_hot64 pre_mm hot:64 ref; ev mm48_4 pre_mm 4 ref
ev v_mm48_2 pre_mm 2 ref_mm mm; ev v_mm48_hot64 pre_mm hot:64 ref_mm mm; ev v_mm48_4 pre_mm 4 ref_mm mm
ev v_p48_2 pre 2 ref_mm mm; ev v_p48_hot64 pre hot:64 ref_mm mm
/tmp/nq_inkling/venv/bin/python /tmp/nq_inkling/kld/tools/perkld.py p48_2 mm48_2 p48_hot64 mm48_hot64 p48_4 mm48_4
echo EVAL_DONE
