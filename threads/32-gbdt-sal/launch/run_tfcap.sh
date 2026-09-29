#!/bin/bash
# T32 SM120 (b): teacher-forced FP8 ref recapture of the SM120 decode stretches (sm120_tf_prep.py corpus). PRIVATE.
set -euo pipefail
O=/tmp/nestquant/32-gbdt-sal; P=$O/private
echo "take GPUs tfcap $(date -u)"
NQ_CORPUS_DIR=$P/corpora NQ_OUT=$P/e2e_sm120 NQ_TRACE_DIR=$P/trace_sm120 NQ_VRAM_GB=60 CORPORA=sm120tf TAG=t32tf \
  /home/coder/git/nestquant/threads/32-gbdt-sal/capture.sh
echo "release GPUs tfcap $(date -u)"
