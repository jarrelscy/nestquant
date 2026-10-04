#!/bin/bash
# NestQuant GLM-5.3 b1.75/4 on 2x DGX Spark: TP2 across the two nodes, one GB10 per node, ConnectX-7 between them.
#
#   HEAD_IP=<head CX-7 IP> spark/start_spark.sh head   [up|fetch|down|logs|smoke]     # node 0: API on :8001
#   HEAD_IP=<head CX-7 IP> spark/start_spark.sh worker [up|fetch|down|logs]           # node 1: headless
#
# Start the worker and the head within a few minutes of each other (either order); the head serves once both are up.
# First run per node downloads ~255 GB (backbone 58 GB + this node's two TP4 rank files, 196 GB), converts the MTP
# layer's experts to fp8 (spark/mtp_fp8.py, +12 GB) and merges the two TP4 ranks into one TP2 rank (streaming/tp4to2.py,
# ~10 min); peak disk ~460 GB in NQ_DATA, ~270 GB with NQ_DROP_TP4=1.
#
# Env (all optional except HEAD_IP):
#   NQ_PRESET       speed (default): 128K context, MTP ns=1, 18 floating 4-bit experts/layer (jF), 21 slots/layer
#                   quality:         128K context, no MTP, 29 floating experts/layer, 32 slots/layer
#                   (see spark/README.md "Memory budget"; NUM_SPEC / NQ_MAXLEN / NQ_JF_NFLOAT / NQ_SLOTS_PER_LAYER override)
#   NQ_IMAGE        image built from spark/Dockerfile (default nestquant-spark:b175)
#   NQ_DATA         per-node data dir on the internal NVMe (default $HOME/nq-spark)
#   NQ_UTIL         --gpu-memory-utilization (default 0.95)
#   NCCL_SOCKET_IFNAME / NCCL_IB_HCA   CX-7 netdev and RoCE devices (defaults below; check with `ibdev2netdev`)
#   NQ_TAP_RATE_GBPS                   sustained SSD read rate per node for the tap scheduler (default 6.6)
#   NQ_DEC_BLOCK_MS / NQ_DEC_BLOCK_FIRSTN / NQ_DEC_BLOCK_AFTER_MS / NQ_DEC_ASYNC   decode-step wait for landed planes
#                   (SM120 step 2 defaults -1 / 64 / 0.04 / 3; NQ_DEC_BLOCK_MS=0 turns it off, see spark/README.md)
#   VLLM_API_KEY    API key (default none)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd);REPO=$(dirname "$HERE")
ROLE=${1:-};CMD=${2:-up}
case "$ROLE" in head) R=0 ;; worker) R=1 ;; *) echo "usage: HEAD_IP=<ip> $0 head|worker [up|fetch|down|logs|smoke]"; exit 2 ;; esac
NAME=nq-spark-$ROLE
NQ_IMAGE=${NQ_IMAGE:-nestquant-spark:b175}
NQ_DATA=${NQ_DATA:-$HOME/nq-spark}
BACKBONE_REPO=${BACKBONE_REPO:-jarrelscy/GLM-5.3-Vision-NVFP4-ARVQ-v2-hybrid}
NQ_REPACK_REPO=${NQ_REPACK_REPO:-jarrelscy/GLM-5.3-NestQuant-1.75-4bit}
PRED_REPO=${PRED_REPO:-jarrelscy/GLM-5.3-NestQuant-2-4bit}
SERVED=${NQ_SERVED_NAME:-glm-5.3-nq}
# b1.75/4 kernel build: residual codes 0 and 9 (down 2.5625) + 3 (gate/up 2.25), base codes 0 (K2) and 1 (K1.75)
NQ_DEFS=${NQ_DEFS:-NQ_RK_CODES=0x209,NQ_RK_GU=0x9,NQ_RK_DN=0x201,NQ_BK_CODES=0x3}
case "${NQ_PRESET:-speed}" in
  speed)   D_SPEC=1; D_LEN=131072; D_NF=18; D_SLOTS=21 ;;
  quality) D_SPEC=0; D_LEN=131072; D_NF=29; D_SLOTS=32 ;;
  *) echo "NQ_PRESET must be quality or speed"; exit 2 ;;
esac
NUM_SPEC=${NUM_SPEC:-$D_SPEC};MAXLEN=${NQ_MAXLEN:-$D_LEN}
NF=${NQ_JF_NFLOAT:-$D_NF};SLOTS=${NQ_SLOTS_PER_LAYER:-$D_SLOTS}
[ "$SLOTS" -ge $((NF+3)) ] || { echo "NQ_SLOTS_PER_LAYER ($SLOTS) must be >= NQ_JF_NFLOAT+3 ($((NF+3)))"; exit 2; }
T0=$((2*R));T1=$((2*R+1))
# BF16 linears converted to e4m3 (per output channel, W8A16) at load: MLA q_a/kv_a and q_b, shared experts, dense MLP
# (layers 0-2). o_proj stays NVFP4 (P4 path) on layers 0-77; only the MTP layer's o_proj takes fp8. kv_b (absorbed into
# MLA), indexer, router, embed and lm_head stay BF16. NQ_FP8_TARGETS overrides the list; NQ_FP8_TARGETS=none keeps all BF16.
FP8_TARGETS=self_attn.fused_qkv_a_proj,self_attn.q_b_proj,self_attn.o_proj,shared_experts.gate_up_proj,shared_experts.down_proj,mlp.gate_up_proj,mlp.down_proj

drun(){ docker run --rm --gpus all --ipc host --network host --entrypoint "$@"; }
hfdl(){   # hf download inside the image; one --include / --exclude flag per pattern (hf 1.x applies only the first of a list)
  local repo=$1 dst=$2;shift 2
  docker run --rm --network host -e HF_TOKEN="${HF_TOKEN:-}" -e HF_HUB_DISABLE_PROGRESS_BARS=1 -v "$NQ_DATA":/nqdata \
    --entrypoint /opt/vllm/.venv/bin/hf "$NQ_IMAGE" download "$repo" --repo-type model --local-dir "/nqdata/$dst" "$@"
}

fetch(){
  mkdir -p "$NQ_DATA"/{backbone,tp4,tp2,pred,build,cache/vllm,cache/triton,cache/flashinfer,cache/torch_extensions,dbg}
  docker image inspect "$NQ_IMAGE" >/dev/null 2>&1 || { echo "image $NQ_IMAGE not found: build it on this node with
  docker build -f $REPO/spark/Dockerfile -t $NQ_IMAGE $REPO/spark"; exit 1; }
  if [ ! -f "$NQ_DATA/backbone/.done" ]; then
    echo "== backbone (non-expert weights, ~58 GB): $BACKBONE_REPO"
    hfdl "$BACKBONE_REPO" backbone --exclude 'arvq-layer-*' --exclude 'hot-layer-*' --exclude 'reproduce/*' \
      --exclude 'cold_manifests/*' --exclude 'pv_layers/*'
    touch "$NQ_DATA/backbone/.done"
  fi
  # MTP layer 78 experts BF16 -> e4m3 128x128 blocks (9 -> 4.5 GiB/node). Runs on every fetch: the index patch must be
  # redone whenever hf download has restored the original index; the conversion itself is skipped once done.
  echo "== MTP experts to fp8 (spark/mtp_fp8.py)"
  docker run --rm -v "$REPO":/nq:ro -v "$NQ_DATA":/nqdata --entrypoint /opt/vllm/.venv/bin/python "$NQ_IMAGE" \
    /nq/spark/mtp_fp8.py /nqdata/backbone /nqdata/backbone
  if [ ! -f "$NQ_DATA/pred/.done" ]; then
    echo "== jF predictor: $PRED_REPO serving/predictor"
    hfdl "$PRED_REPO" pred --include 'serving/predictor/joint/*' --include 'serving/predictor/delta_table.json'
    touch "$NQ_DATA/pred/.done"
  fi
  if [ ! -f "$NQ_DATA/tp2/rank$R.json" ]; then
    if [ ! -f "$NQ_DATA/tp4/.done$R" ]; then
      echo "== NestQuant TP4 ranks $T0,$T1 (~196 GB): $NQ_REPACK_REPO"
      hfdl "$NQ_REPACK_REPO" tp4 --include "rank$T0.json" --include "rank$T0.bin" --include "res/rank$T0/*" \
        --include "rank$T1.json" --include "rank$T1.bin" --include "res/rank$T1/*" --include 'artifact_stamp.json'
      touch "$NQ_DATA/tp4/.done$R"
    fi
    echo "== merge TP4 ranks $T0,$T1 -> TP2 rank $R (streaming/tp4to2.py)"
    docker run --rm -e NT="${NQ_MERGE_THREADS:-16}" -v "$REPO":/nq:ro -v "$NQ_DATA":/nqdata \
      --entrypoint /opt/vllm/.venv/bin/python "$NQ_IMAGE" /nq/streaming/tp4to2.py /nqdata/tp4 /nqdata/tp2 "$R"
    cp "$NQ_DATA/tp4/artifact_stamp.json" "$NQ_DATA/tp2/"
    [ -f "$NQ_DATA/tp2/rank$R.json" ] || { echo "tp4to2 did not write rank$R.json"; exit 1; }
    if [ "${NQ_DROP_TP4:-0}" = 1 ]; then rm -rf "$NQ_DATA/tp4/rank$T0".* "$NQ_DATA/tp4/rank$T1".* "$NQ_DATA/tp4/res"; fi
  fi
  echo "== NestQuant kernels (nqmoe b1.75, nqsal, nqstream) for sm_121a"
  drun bash -e NQ_BUILD=/nqbuild -e NQ_DEFS="$NQ_DEFS" -e TORCH_CUDA_ARCH_LIST=12.1a -e LIBURING=/opt/liburing \
    -v "$REPO":/nq:ro -v "$NQ_DATA/build":/nqbuild "$NQ_IMAGE" \
    -c 'cd /nq/sm120 && python -c "import build;build.get();build.get_sal()" && cd ../streaming && python -c "import stream_engine as S;S.mod()"'
}

up(){
  : "${HEAD_IP:?set HEAD_IP to the head node CX-7 address}"
  fetch
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  # in-boot knob files (--ipc host: /dev/shm is the host's, a previous boot's knobs would still apply)
  rm -f /dev/shm/nq_la_ctl /dev/shm/nq_pf_off /dev/shm/nq_sr_ctl /dev/shm/nq_tier_drop /dev/shm/nq_dec_block \
        /dev/shm/nq_dec_block_firstn /dev/shm/nq_dec_async /dev/shm/nq_dec_async_switch /dev/shm/nq_hit_carry \
        /dev/shm/nq_pred_inputs /dev/shm/nq_tap_ctl /dev/shm/nq_pf_block /dev/shm/nq_force_sets 2>/dev/null || true
  IB=();[ -d /dev/infiniband ] && IB=(--device /dev/infiniband)
  SC=();[ "$NUM_SPEC" -gt 0 ] && SC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$NUM_SPEC,\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"standard\"}")
  CAP=${ARVQ_CAPTURE_SIZES:-[1,2,4,8]}
  ROLEARGS=(--port 8001 --served-model-name "$SERVED" local)
  [ "$R" = 1 ] && ROLEARGS=(--headless)
  echo "== $ROLE (node $R): preset ${NQ_PRESET:-speed}, MTP ns $NUM_SPEC, ${MAXLEN} ctx, $NF floating / $SLOTS slots per layer"
  docker run -d --name "$NAME" --gpus all --ipc host --network host --shm-size 16g "${IB[@]}" \
    --ulimit memlock=-1 --ulimit stack=67108864 --security-opt seccomp=unconfined \
    -v "$REPO":/nq:ro \
    -v "$REPO/spark/overlay/nvfp4_arvq_hybrid.py":/opt/vllm/vllm/model_executor/layers/quantization/nvfp4_arvq_hybrid.py:ro \
    -v "$NQ_DATA/backbone":/model:ro -v "$NQ_DATA/tp2":/nqrepack:ro -v "$NQ_DATA/pred/serving/predictor":/nqpred:ro \
    -v "$NQ_DATA/build":/nqbuild -v "$NQ_DATA/dbg":/dbg \
    -v "$NQ_DATA/cache/vllm":/root/.cache/vllm -v "$NQ_DATA/cache/triton":/root/.triton \
    -v "$NQ_DATA/cache/flashinfer":/root/.cache/flashinfer -v "$NQ_DATA/cache/torch_extensions":/root/.cache/torch_extensions \
    -e VLLM_API_KEY="${VLLM_API_KEY:-}" -e VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}" -e PYTHONHASHSEED=0 \
    -e NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp1s0f1np1}" -e GLOO_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp1s0f1np1}" \
    -e NCCL_IB_HCA="${NCCL_IB_HCA:-rocep1s0f1,roceP2p1s0f1}" -e NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}" \
    -e VLLM_HOST_IP="${NODE_IP:-}" -e MASTER_ADDR="$HEAD_IP" \
    -e PYTHONPATH=/nq/sm120/serve -e NQ_HOME=/nq -e NQ_BUILD=/nqbuild -e LIBURING=/opt/liburing \
    -e NQ_REPACK=/nqrepack -e NQ_REPACK_ALT=none -e NQ_OPLOG=dist -e NQ_OPLOG_ADDR="$HEAD_IP:${NQ_OPLOG_PORT:-29611}" \
    -e NQ_DEFS="$NQ_DEFS" -e TORCH_CUDA_ARCH_LIST=12.1a -e NQ_UNIFIED="${NQ_UNIFIED:-}" \
    -e NQ_JF_NFLOAT="$NF" -e NQ_SLOTS_PER_LAYER="$SLOTS" -e NQ_RAMTIER_GB=0 \
    -e NQ_PREFILL_BORROW=0 -e NQ_PREFILL_KV_OFFLOAD=0 -e NQ_TAP_RATE_GBPS="${NQ_TAP_RATE_GBPS:-6.6}" \
    -e NQ_CFG_GU="${NQ_CFG_GU:-}" -e NQ_CFG_DN="${NQ_CFG_DN:-}" \
    -e NQ_HITS=1 -e NQ_POLL_MS="${NQ_POLL_MS:-4}" -e NQ_ISSUE=1 -e NQ_SESSION_RESTORE="${NQ_SESSION_RESTORE:-0}" \
    -e NQ_PRED_INPUTS="${NQ_PRED_INPUTS:-3}" -e NQ_HIT_CARRY="${NQ_HIT_CARRY:-1}" -e NQ_TAP_QREAL=1 -e NQ_TAP_TODO_FIX=2 \
    -e NQ_PF_BLOCK=1 -e NQ_PF_BLOCK_MS="${NQ_PF_BLOCK_MS:-0}" -e NQ_DEC_BLOCK_MS="${NQ_DEC_BLOCK_MS:--1}" \
    -e NQ_DEC_BLOCK_FIRSTN="${NQ_DEC_BLOCK_FIRSTN:-64}" -e NQ_DEC_BLOCK_AFTER_MS="${NQ_DEC_BLOCK_AFTER_MS:-0.04}" \
    -e NQ_DEC_ASYNC="${NQ_DEC_ASYNC:-3}" -e NQ_DBG_NO_BF16_RED="${NQ_DBG_NO_BF16_RED:-1}" \
    -e NQ_JOINT_HM="${NQ_JOINT_HM:-0.7}" -e NQ_JOINT_NET=/nqpred/joint/jF.pt \
    -e NQ_JOINT_V2=/nqpred/joint/v2_sal_tweedie1.5.txt -e NQ_DELTA_TABLE=/nqpred/delta_table.json \
    -e NQ_PF=1 -e NQ_PF_MIN=384 -e NQ_PF_ROWS=8192 -e NQ_PF_G=16 -e NQ_PF_GEMM=triton \
    -e NQ_PREFILL_ADAPT=lookahead -e NQ_LA_D=1 -e NQ_LA_BUDGET="${NQ_LA_BUDGET:-$(( (NF*45+38)/77 ))}" -e NQ_PF_RANK=gate -e NQ_LA_OUT=/dbg/la_stats.json \
    -e VLLM_ARVQ_CHUNK_TOKENS=128 -e VLLM_ARVQ_GROUPED_PREFILL=1 -e VLLM_ARVQ_COMPACT_PREFILL=1 \
    -e VLLM_SM120_COMPACT_WORKSPACE=1 -e VLLM_MTP_INDEX_SHARE=1 -e VLLM_DSA_CANONICAL_TOPK=inkernel \
    -e GLM_MOE_LANE_ROWS=1 -e GLM_NVFP4_LUT256=1 \
    -e VLLM_ENABLE_NVFP4_P4_O_PROJ=1 -e VLLM_NVFP4_P4_MAX_TOKENS=16 -e VLLM_NVFP4_P4_PAIRED=1 \
    -e VLLM_DISABLE_FP8_W8A16=0 -e NQ_DBG_FP8_TARGETS="${NQ_FP8_TARGETS:-$FP8_TARGETS}" -e NQ_MTP_FP8=1 \
    -e VLLM_ARVQ_FUSED_GATE_PACK=1 -e VLLM_ARVQ_FUSED_COLD_SCATTER=1 -e VLLM_ARVQ_DIRECT_COLD_OUTPUT=1 \
    -e VLLM_ARVQ_FUSED_ACTIVATION_PACK=1 -e VLLM_ARVQ_FUSED_COLD_ACTIVATION=1 -e VLLM_ARVQ_FUSED_COLD_GATHER=1 \
    -e VLLM_ARVQ_FUSED_ROUTE_SUM=1 -e VLLM_ARVQ_PAIRED_HOT_PREFILL=1 -e VLLM_ARVQ_SHARED_HOT_ACTIVATION=1 \
    -e VLLM_ARVQ_WIDE_HOT_PREFILL=1 \
    -e VLLM_GLM_COMM_COALESCE=0 -e VLLM_GLM_COMM_OVERLAP="${VLLM_GLM_COMM_OVERLAP:-1}" -e VLLM_GLM_EMBED_GRAPH=1 \
    -e VLLM_GLM_IDX_FUSED_LOCALIZE=1 -e VLLM_GLM_MM_MASK_REUSE=1 -e VLLM_GLM_Q_BEFORE_ABSORB=1 \
    -e VLLM_GLM_RAW_KV_GATHER=1 -e VLLM_GLM_SKIP_EMPTY_FILL=1 -e VLLM_EXPERIMENT_C1_INDEXER_BUDGET=0 \
    -e VLLM_FORCE_CUSTOM_ALLREDUCE=0 -e VLLM_SM120_ROUTER_GEMM=1 \
    "$NQ_IMAGE" serve /model \
    --tensor-parallel-size 2 --nnodes 2 --node-rank "$R" --master-addr "$HEAD_IP" --master-port "${MASTER_PORT:-29501}" \
    --trust-remote-code --reasoning-parser glm47 --enable-auto-tool-choice --tool-call-parser glm47 \
    --enable-prefix-caching --kv-cache-dtype fp8_ds_mla --max-model-len "$MAXLEN" \
    --max-num-seqs "${NQ_MAX_NUM_SEQS:-4}" --max-num-batched-tokens "${NQ_MNBT:-4096}" \
    --gpu-memory-utilization "${NQ_UTIL:-0.95}" --no-enable-flashinfer-autotune \
    --override-generation-config '{"temperature":1.0,"top_p":0.95}' \
    --compilation-config "{\"mode\":3,\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"compile_sizes\":[1],\"cudagraph_capture_sizes\":$CAP}" \
    "${SC[@]}" "${ROLEARGS[@]}" >/dev/null
  echo "started $NAME; logs: $0 $ROLE logs"
  [ "$R" = 1 ] && return
  auth=();[ -n "${VLLM_API_KEY:-}" ] && auth=(-H "Authorization: Bearer $VLLM_API_KEY")
  echo "waiting for /v1/models (needs the worker up too; loading takes a while) ..."
  for i in $(seq 1 540); do
    curl -sf "${auth[@]}" localhost:8001/v1/models >/dev/null 2>&1 && { echo ready; return; }
    docker ps --format '{{.Names}}' | grep -q "^$NAME\$" || { echo "container exited"; docker logs --tail 80 "$NAME"; exit 1; }
    sleep 10
  done
  echo "timeout"; exit 1
}

case "$CMD" in
  up) up ;;
  fetch) fetch ;;
  down) docker rm -f "$NAME" ;;
  logs) docker logs -f "$NAME" ;;
  smoke)
    auth=();[ -n "${VLLM_API_KEY:-}" ] && auth=(-H "Authorization: Bearer $VLLM_API_KEY")
    curl -s "${auth[@]}" -H 'Content-Type: application/json' localhost:8001/v1/completions \
      -d "{\"model\":\"$SERVED\",\"prompt\":\"The capital of France is\",\"max_tokens\":24,\"temperature\":0}" \
      | python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['text'])"
    docker logs "$NAME" 2>&1 | grep NestQuant | tail -8 ;;
  *) echo "unknown command $CMD"; exit 2 ;;
esac
