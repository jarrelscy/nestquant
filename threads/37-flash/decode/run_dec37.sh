#!/bin/bash
# run_dec37.sh [dec37.py args...]   GLM-5.3-Flash on-policy decoder (HF-based, layers pipelined over 8 A100s).
# Waits on the shared /tmp/nestquant/33-search/gpu.lock, so it never overlaps capture/encode jobs.
# Output defaults to the PRIVATE /tmp/nestquant/37-flash/private/dec_trace (never upload).
#   ./run_dec37.sh --smoke --slots 8 --max-new 512 --out /tmp/nestquant/37-flash/private/dec_smoke
#   ./run_dec37.sh --input prompts.jsonl --n-samples 4
#   ./run_dec37.sh --finalize /tmp/nestquant/37-flash/private/dec_trace     (CPU only; no lock needed but harmless)
#   STOP: touch OUT/STOP (graceful: finishes the step, flushes the shard) or send SIGTERM.
set -euo pipefail
cd /home/coder/git/nestquant/threads/37-flash/decode
used=$(awk '$1=="anon"||$1=="shmem"{s+=$2} END{printf "%.0f", s/1e9}' /sys/fs/cgroup/memory.stat)
if [ "$used" -gt 1200 ]; then echo "refusing: host anon+shmem ${used} GB > 1200 GB" >&2; exit 1; fi
mkdir -p /tmp/nestquant/37-flash/logs
log=/tmp/nestquant/37-flash/logs/dec37.$(TZ=Australia/Melbourne date +%Y%m%d-%H%M%S).log
echo "host anon+shmem ${used} GB; log $log" >&2
exec > >(tee -a "$log") 2>&1
exec flock /tmp/nestquant/33-search/gpu.lock env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} TZ=Australia/Melbourne OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1 \
  nice -n 10 /tmp/venv-t37g/bin/python dec37.py "$@"
