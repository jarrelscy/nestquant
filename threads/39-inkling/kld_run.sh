#!/bin/bash
# usage: kld_run.sh TAG WSEL...   (val-only forward, native weights for layers not predecoded)
cd /tmp/nq_inkling/code; TAG=$1; shift
for W in "$@"; do
  O=/tmp/nq_inkling/kld/${TAG}_${W/:/}
  [ -f $O/final/kld.json ] && continue
  rm -rf $O
  flock /tmp/nq_inkling/kld.lock /tmp/nq_inkling/venv/bin/torchrun --nproc-per-node 8 --master-port 29713 cap_ink.py --out $O --layers 0-47 \
    --nfit 0,0 --no-stats --wdir /tmp/nq_inkling/pre --wsel $W --wmax ${WMAX:-48} --ref /tmp/nq_inkling/ref > $O.log 2>&1
  echo "$TAG $W $(grep 'KLD vs ref' $O.log)"
done
