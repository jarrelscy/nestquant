#!/bin/bash
cd /home/coder/git/nestquant/threads/11-matquant-baseline
E=${E:-36}; OUT=${OUT:-/tmp/nestquant/11-matquant-baseline/e${E}.json}
first=1
for f in ${CFGS:-/tmp/nestquant/11-matquant-baseline/cfg/b*.json}; do
  if [ $first = 1 ]; then R=--refs; first=0; else R=; fi
  ./run.sh sweep.py --expert $E $R --out $OUT --configs $f 2>&1 | grep -v -i warn
done
