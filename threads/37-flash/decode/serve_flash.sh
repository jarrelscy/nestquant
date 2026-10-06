#!/bin/bash
# serve_flash.sh — vLLM 0.31.0 OpenAI server for GLM-5.3-Flash (FP8 block-128 checkpoint), TP8, port 8137.
# Used to generate on-policy decode traces (routing via --enable-return-routed-experts, see README.md).
#
#   ./serve_flash.sh                 # default: CUDA graphs on, BF16 KV on sm80/sm90, FP8 KV on sm100+
#   EAGER=1 ./serve_flash.sh         # --enforce-eager (first bring-up / debugging)
#   ROUTING=0 ./serve_flash.sh       # do not capture routed experts
#   MTP=1 ./serve_flash.sh           # MTP spec decode (do NOT use for routing traces: draft rows get mixed in)
#   KV=bfloat16|fp8|auto ./serve_flash.sh   # override the KV cache dtype choice
#
# The A100 (sm_80) guard: stock vLLM 0.31.0 cannot run this model on sm_80. The DSA indexer needs DeepGEMM
# (sm90+), the sparse-MLA + kpool-tail backends are FlashInfer sm90/100/120 only, and Triton rejects float8_e4m3fn
# on sm_80 (kpool K-cache and indexer-Q quant kernels). See README.md "Blockers". The script refuses to start on
# sm<90 unless T37_SM80_PATCHED=1, which means the sm80 patch set has been applied to /tmp/venv-t37v.
set -euo pipefail

VENV=${VENV:-/tmp/venv-t37v}
MODEL=${MODEL:-/tmp/nestquant/37-flash/fp8}
PORT=${PORT:-8137}
MAXLEN=${MAXLEN:-32768}
MAXSEQS=${MAXSEQS:-64}
UTIL=${UTIL:-0.90}
LOCK=${LOCK:-/tmp/nestquant/33-search/gpu.lock}
LOG=${LOG:-/tmp/nestquant/37-flash/logs/serve_flash.$(date +%Y%m%d_%H%M%S).log}

cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
major=${cc%%.*}
if [ "$major" -lt 9 ] && [ "${T37_SM80_PATCHED:-0}" != 1 ]; then
  echo "serve_flash.sh: GPU compute capability $cc < 9.0 and T37_SM80_PATCHED!=1." >&2
  echo "  vLLM 0.31.0 has no sm_80 path for GLM-5.3-Flash's DSA layers (DeepGEMM indexer, FlashInfer sparse MLA," >&2
  echo "  Triton fp8e4nv). Expected failure: no valid sparse-MLA attention backend at engine init. See README.md." >&2
  exit 2
fi

if [ -z "${KV:-}" ]; then
  # Recipe: FP8 KV is Blackwell-only for this model ("Hopper ... must run BF16 KV"). sm_80 can't do e4m3 in Triton.
  if [ "$major" -ge 10 ]; then KV=fp8; else KV=auto; fi
fi

args=(
  --served-model-name glm53-flash
  --tensor-parallel-size 8
  --max-model-len "$MAXLEN" --max-num-seqs "$MAXSEQS"
  --gpu-memory-utilization "$UTIL"
  --kv-cache-dtype "$KV"
  --port "$PORT" --host 0.0.0.0
  --reasoning-parser glm47 --tool-call-parser glm47 --enable-auto-tool-choice
  --limit-mm-per-prompt '{"image":4,"video":0}'
)
[ "$major" -lt 9 ] && args+=(--moe-backend marlin)          # W8A16 Marlin FP8 MoE (no FP8 tensor cores on sm_80)
[ "${EAGER:-0}" = 1 ] && args+=(--enforce-eager)
[ "${ROUTING:-1}" = 1 ] && args+=(--enable-return-routed-experts)
[ "${MTP:-0}" = 1 ] && args+=(--speculative-config '{"method":"mtp","num_speculative_tokens":3}')

mkdir -p "$(dirname "$LOG")"
unset VLLM_API_KEY                                            # stray key => 401 on every request
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_CACHE_ROOT=/tmp/nestquant/37-flash/tmp/vllm-cache TRITON_CACHE_DIR=/tmp/nestquant/37-flash/tmp/triton-cache
export TILELANG_CACHE_DIR=/tmp/nestquant/37-flash/tmp/tilelang-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
echo "serve_flash: cc=$cc kv=$KV eager=${EAGER:-0} routing=${ROUTING:-1} mtp=${MTP:-0} log=$LOG" >&2
# gpu.lock: same lock as run_cap.sh / the T33 jobs, so this never overlaps the capture.
exec flock "$LOCK" nice -n 10 "$VENV/bin/vllm" serve "$MODEL" "${args[@]}" >>"$LOG" 2>&1
