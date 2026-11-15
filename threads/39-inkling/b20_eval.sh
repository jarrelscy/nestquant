#!/bin/bash
unset CUDA_VISIBLE_DEVICES
cd /tmp/nq_inkling/code
for spec in "l5b175_2 pre_l5 2" "l5b175_hot64 pre_l5 hot:64" "l5b175_4 pre_l5 4" "l5b20_2 pre_b20 2"; do
  set -- $spec; O=/tmp/nq_inkling/kld/$1; [ -f $O/final/kld.json ] && continue; rm -rf $O
  flock /tmp/nq_inkling/kld.lock /tmp/nq_inkling/venv/bin/torchrun --nproc-per-node 8 --master-port 29714 cap_ink.py --out $O --layers 0-47 --nfit 0,0 --no-stats \
    --wdir /tmp/nq_inkling/$2 --wsel $3 --wmax 6 --ref /tmp/nq_inkling/ref > $O.log 2>&1
  echo "$1 $(grep 'KLD vs ref' $O.log)"
done
/tmp/nq_inkling/venv/bin/python /tmp/nq_inkling/kld/tools/perkld.py l5b175_2 l5b175_hot64 l5b175_4 l5b20_2
