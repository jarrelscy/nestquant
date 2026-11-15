#!/bin/bash
# cumulative KLD at layer milestones (layers < N quantized, rest native)
for N in 12 24 36 48; do
  until [ -f /tmp/nq_inkling/pre/L$((N-1)).relerr.json ]; do sleep 120; done
  WMAX=$N /tmp/nq_inkling/code/kld_run.sh p$N 2 4 hot:64
  /tmp/nq_inkling/venv/bin/python /tmp/nq_inkling/kld/tools/perkld.py p${N}_2 p${N}_4 p${N}_hot64
done
