#!/bin/bash
# T33i: coordinator 2026-09-30 (user priority): hold all KLD passes until T33l's run 1 is complete
# (run1_driver.log "run1 driver end") AND T32's fp8dec corpora exist; then decode (3 corpora, one tag), human.
# Order (coordinator): T32's primary fp8dec pass first (T32_DONE = a file it writes at the end, e.g. its merged
# results json), then dec; human last and optional (SKIP_HUM=1 if the queue is long).
# Not launched until T33l / T32 confirm.  HMD / HMH / REF pass through to run_kld_J.sh.
D=/tmp/nestquant/33-search/ceiling/run1_driver.log; J=/home/coder/git/nestquant/threads/33-search/joint
F=/tmp/nestquant/32-gbdt-sal/private/fp8dec_t32/corpora
until grep -q "run1 driver end" $D 2>/dev/null && [ -f $F/fp8dec-heldout.map.npz ] && [ -f $F/fp8dec-tb21.map.npz ] && [ -f $F/sm120tfk8.map.npz ] \
      && [ -f "${T32_DONE:-/tmp/nestquant/32-gbdt-sal/private/kld_passT32KF.json}" ]; do
  sleep 60
done
echo "run 1 complete + fp8dec corpora present $(date -u)"
PH="dec"; [ "${SKIP_HUM:-0}" = 1 ] || PH="$PH hum"
for ph in $PH; do $J/run_kld_J.sh $ph; echo "$ph rc=$? $(date -u)"; done
