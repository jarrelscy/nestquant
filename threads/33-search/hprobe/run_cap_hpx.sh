#!/bin/bash
# coordinator rule (rev): enqueue after T33l gen.py is gone and once T32 kld_k has STARTED (or finished)
cd /home/coder/git/nestquant/threads/33-search/hprobe
kld_started() { for p in $(pgrep -f "[f]lock /tmp/nestquant/33-search/gpu.lock /tmp/nestquant/32-gbdt-sal/kld_k"); do pgrep -P $p > /dev/null && return 0; done; return 1; }
kld_queued() { pgrep -f "[f]lock /tmp/nestquant/33-search/gpu.lock /tmp/nestquant/32-gbdt-sal/kld_k" > /dev/null; }
while pgrep -f "[r]un_gen.sh" > /dev/null || pgrep -f "[p]redec_k.tmp.sh" > /dev/null || { kld_queued && ! kld_started; }; do sleep 20; done
echo "deps ok $(date -u)"
P=/tmp/nestquant/33-search/hprobe/private
MODE=pool CORPORA=hpx NQ_CORPUS_DIR=$P/corpora HPDIR=$P/pool_hpx TRACE=$P/trace_hpx TAG=hp_hpx \
  flock /tmp/nestquant/33-search/gpu.lock ./capture.sh
