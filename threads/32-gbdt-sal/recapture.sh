#!/bin/bash
# T32 recapture with full router scores (NQ_TRACE_PROBS): waits until passD3 is gone, then capture.sh -> trace2.
set -euo pipefail
while pgrep -f "tag passD3" > /dev/null; do sleep 15; done
echo "take GPUs $(date -u)"
NQ_TRACE_PROBS=1 NQ_TRACE_DIR=/tmp/nestquant/32-gbdt-sal/trace2 TAG=t32cap2 /home/coder/git/nestquant/threads/32-gbdt-sal/capture.sh
echo "release GPUs $(date -u)"
