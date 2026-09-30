#!/bin/bash
# T32 rebuild after the 2026-09-30 08:58 AEST /tmp wipe: re-run the FP8 ref captures (PRIVATE, stay under
# /tmp/nestquant/32-gbdt-sal).  A: trace2 (ids,w,xn + router probs; calib-fit+glm52-heldout) -> trace = hardlinks
# (same ref routing; the old trace/ was a probs-less capture of the same corpora).  B: hidcov (calib-fit) -> PCA-32.
# C: trace3 (PCA projections + lm_head entropy).  Each capture holds /tmp/nestquant/33-search/gpu.lock.
set -euo pipefail
O=/tmp/nestquant/32-gbdt-sal; P=$O/private; C=/home/coder/git/nestquant/threads/32-gbdt-sal
export LD_LIBRARY_PATH=/tmp/compat13/usr/local/cuda-13.2/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
L=/tmp/nestquant/33-search/gpu.lock
while pgrep -f "hf download zai-org/GLM-5.3" > /dev/null || [ ! -f /tmp/nestquant/src/glm53-fp8/model.safetensors.index.json ]; do sleep 20; done
echo "fp8 ready $(date -u)"
export NQ_VRAM_GB=${NQ_VRAM_GB:-60}
cap() { echo "take GPUs $TAG $(date -u)"; flock $L env "$@" $C/capture.sh; echo "release GPUs $TAG $(date -u)"; }
TAG=t32cap2; cap NQ_TRACE_PROBS=1 NQ_TRACE_DIR=$O/trace TAG=$TAG
mkdir -p $O/trace2; for f in $O/trace/*; do ln -f "$f" $O/trace2/; done; echo "trace/trace2 ready $(date -u)"
TAG=t32m1cov; cap NQ_TRACE_HID=10,40,70 NQ_TRACE_DIR=$P/hidcov CORPORA=calib-fit TAG=$TAG
PYTHONPATH=/tmp/nestquant/18-e2e/pylib /home/coder/git/glm52/.venv/bin/python $C/fit_pca.py $P/hidcov $P/pca 32
TAG=t32m1; cap NQ_TRACE_HID=10,40,70 NQ_TRACE_HID_PCA=$P/pca NQ_TRACE_HEAD=1 NQ_TRACE_DIR=$P/trace3 CORPORA=calib-fit,glm52-heldout TAG=$TAG
echo "all captures done $(date -u)"
