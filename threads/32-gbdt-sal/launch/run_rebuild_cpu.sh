#!/bin/bash
# T32 rebuild (CPU, <=~20 threads) after the /tmp wipe: rows (default band + band all) + v2 rows from trace ->
# retrain v2_sal_tweedie1.5 -> hot_eval heldout (expect v2 74.77 / 2.78) -> other rows_* (x, lh, dx, t1, v3; m1 after trace3).
set -euo pipefail
O=/tmp/nestquant/32-gbdt-sal; C=/home/coder/git/nestquant/threads/32-gbdt-sal; cd $C
export PYTHONPATH=/tmp/nestquant/18-e2e/pylib OMP_NUM_THREADS=1; PY="nice -n 10 /home/coder/git/glm52/.venv/bin/python -W ignore"
NP=${NP:-18}
until grep -q "trace/trace2 ready" $O/logs/rebuild_cap.log; do sleep 30; done
echo "rows start $(date -u)"
for c in glm52-heldout calib-fit; do $PY build.py $c $NP > /dev/null; $PY build_v2.py $c $NP; done
for c in glm52-heldout calib-fit; do T32_BAND=all $PY build.py $c $NP > /dev/null; T32_BAND=all $PY build_v2.py $c $NP; done
echo "rows done $(date -u)"
mkdir -p $O/models
$PY train.py --target sal --obj tweedie:1.5 --v2 --threads $NP --out $O/models/v2_sal_tweedie1.5.txt 2>&1 | tail -3
echo "trained $(date -u)"
for c in glm52-heldout calib-fit; do NPROC=$NP $PY hot_eval.py $c v2=/home/coder/git/nestquant/streaming/gbdt_v2sal_p64.txt \
  v2_retrain=$O/models/v2_sal_tweedie1.5.txt; done
echo "V2CHECK done $(date -u)"
mkdir -p $O/stats; $PY tokclass.py; $PY stats_x.py; echo "stats done $(date -u)"
for b in x lh dx t1; do for c in glm52-heldout calib-fit; do $PY build_$b.py $c $NP > /dev/null; done; echo "rows_$b done $(date -u)"; done
for c in glm52-heldout calib-fit; do $PY build_v3.py $c $NP > /dev/null; done; echo "rows_v3 done $(date -u)"
until grep -q "all captures done" $O/logs/rebuild_cap.log; do sleep 30; done
for c in glm52-heldout calib-fit; do $PY build_m1.py $c $NP > /dev/null; done; echo "rows_m1 done $(date -u)"
echo "ALL CPU done $(date -u)"
