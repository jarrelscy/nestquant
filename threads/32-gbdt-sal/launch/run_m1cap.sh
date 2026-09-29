#!/bin/bash
# T32 M1/M2 GPU recapture (approved after passT32K): pass 1 hidden cov on calib-fit (L10/L40/L70) -> PCA-32 fit (CPU)
# -> pass 2 PCA projections + lm_head entropy/top-1 on calib-fit + glm52-heldout.  PRIVATE: everything under private/.
set -euo pipefail
O=/tmp/nestquant/32-gbdt-sal; P=$O/private; C=/home/coder/git/nestquant/threads/32-gbdt-sal
while IFS=, read -r i free; do
  [ "${free// /}" -ge 14000 ] || { echo "GPU $i only ${free}MiB free $(date -u)"; exit 1; }
done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
echo "take GPUs m1cap $(date -u)"
NQ_TRACE_HID=10,40,70 NQ_TRACE_DIR=$P/hidcov CORPORA=calib-fit TAG=t32m1cov $C/capture.sh
echo "pass1 done $(date -u)"
PYTHONPATH=/tmp/nestquant/18-e2e/pylib /home/coder/git/glm52/.venv/bin/python $C/fit_pca.py $P/hidcov $P/pca 32
NQ_TRACE_HID=10,40,70 NQ_TRACE_HID_PCA=$P/pca NQ_TRACE_HEAD=1 NQ_TRACE_DIR=$P/trace3 CORPORA=calib-fit,glm52-heldout \
  TAG=t32m1 $C/capture.sh
echo "release GPUs m1cap $(date -u)"
