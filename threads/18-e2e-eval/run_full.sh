#!/bin/bash
# Thread 18 full-model NestQuant eval (nq2 / nqdef / nq4 + L3-6 attribution arms), two passes to keep /tmp use
# at ~1.7 TB peak (one full fp16 level on disk at a time).  Order used: decB (CPU head start) -> runB -> delete
# predecoded_B -> snap -> decA -> runA1, runA2 (reference is inline in both; the second compares bitwise to the first).
#   ./run_full.sh snap      snapshot the default 4-bit set (manifests) -> $OUT/defset.json, $OUT/l4sub.json
#   ./run_full.sh decA      predecode nq2 (all) + nq4 (default set + all of L3-6)   -> $PD_A/{nq2,nq4}
#   ./run_full.sh runA1     ref + nq2 + nqdef
#   ./run_full.sh runA2     ref + nq2_early (nq2 on L3-6 only) + nqdef_e4 (nqdef with L3-6 all 4-bit)
#   (3 streams per run: hidden + MoE output buffer per stream ~2 GB at 39 windows/rank, under the 11 GB cap)
#   ./run_full.sh decB      predecode nq4 (all)                                      -> $PD_B/nq4
#   ./run_full.sh runB      ref + nq4
#   ./run_full.sh decS      fast predecode nq4 for top-128 U float0 (l4_sweep_union.json) -> $PD_S/nq4 (~4 min)
#   ./run_full.sh runS1     4-bit count sweep: ref + nqdef64 (nested top-64) + nqfloat0 (serving start: 26 U 51)
#   ./run_full.sh runS2     ref + nqdef128 (nested top-128)
#   ./run_full.sh decS2     fast predecode nq4 for the complement (l4_complement.json) -> $OUT/predecoded_S2/nq4 (nqadapt)
#   ./run_full.sh runS3     NQ_SHARD=contig: ref + nqadapt (causal replay of streaming/scheduler.py: fixed 26 +
#                           floating 51, EMA 512, refresh 64, one-refresh lag, reset per window) + nqadapt_chain
#                           (state carried across the rank's consecutive windows of a corpus, document order)
#   WORLD=16 GPUS="0 0 0 3 3 4 4 4 5 5 6 6 6 7 7 7" ./run_full.sh runS4
#                           ref + nqdef + nqfloat0 + nqdef128: eval-token level-4 share of routed slots per static arm
#   ./run_full.sh hdump     T27 PV-pilot dump: ref(fp8) + nqdef on calib-fit, L29-32 (EXTRA args e.g. --expert-override)
#   ./run_full.sh merge TAG
# Every stage: 8 ranks, one per GPU, NQ_VRAM_GB cap, launch refused if a GPU has < MIN_FREE_MB free.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export NQ_OUT=${NQ_OUT:-/tmp/nestquant/18-e2e}
OUT=$NQ_OUT
PD_A=$OUT/predecoded_A PD_B=$OUT/predecoded_B PD_S=$OUT/predecoded_S
export NQ_VRAM_GB=${NQ_VRAM_GB:-11}
MIN_FREE_MB=${MIN_FREE_MB:-14000}
WORLD=${WORLD:-8}
# rank -> GPU map (default skips GPUs 1-2, in use by thread 27; two ranks share GPUs 0 and 7, each <= NQ_VRAM_GB)
read -r -a GMAP <<< "${GPUS:-0 0 3 4 5 6 7 7}"
CORPORA=nq-tail,vllm-docs,wikitext,github
MOE_CHUNK=${MOE_CHUNK:-16384}
SERVE_MAN=${SERVE_MAN:-/tmp/nestquant/28-serve-release/out/serving/tp4/manifest.json}
LOGS=$OUT/logs; mkdir -p "$LOGS"

gpu_check() {
  local bad=0
  while IFS=, read -r i free; do
    [[ " ${GMAP[*]} " == *" $i "* ]] || continue
    if [ "${free// /}" -lt "$MIN_FREE_MB" ]; then echo "GPU $i only ${free}MiB free (< $MIN_FREE_MB)"; bad=1; fi
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
  local avail; avail=$(df --output=avail -B1G /tmp | tail -1)
  echo "/tmp free ${avail} GB"
  [ "$bad" = 0 ] || { echo "refusing to launch"; exit 1; }
}

launch() {   # launch NAME args... : one rank per GPU, wait for all, fail if any failed
  local name=$1; shift
  gpu_check
  local pids=()
  for r in $(seq 0 $((WORLD - 1))); do
    CUDA_VISIBLE_DEVICES=${GMAP[$r]} RANK=$r WORLD=$WORLD nohup "$HERE/run.sh" "$@" > "$LOGS/$name.r$r.log" 2>&1 &
    pids+=($!)
  done
  echo "$name: pids ${pids[*]} (logs $LOGS/$name.r*.log)"
  local fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [ "$fail" = 0 ] || { echo "$name: a rank failed"; exit 1; }
  echo "$name: done $(date -u)"
}

case "${1:-}" in
  snap)
    /home/coder/git/glm52/.venv/bin/python "$HERE/nq_defset.py" --out "$OUT/defset.json"
    /home/coder/git/glm52/.venv/bin/python "$HERE/nq_defset.py" --from-json "$OUT/defset.json" --extra-layers 3-6 \
        --out "$OUT/l4sub.json" ;;
  decA) launch decA predecode-nq --levels 2,4 --l4-set "$OUT/l4sub.json" --out "$PD_A" ;;
  runA1) launch runA1 run --corpora $CORPORA --local-err --save-ref --moe-chunk $MOE_CHUNK --tag passA1 \
          --cand "nq2=dir:$PD_A/nq2" \
          --cand "nqdef=mix:lo=$PD_A/nq2,hi=$PD_A/nq4,set=$OUT/defset.json" ;;
  runA2) launch runA2 run --corpora $CORPORA --local-err --save-ref --moe-chunk $MOE_CHUNK --tag passA2 \
          --cand "nq2_early=dir:$PD_A/nq2,layers=3-6" \
          --cand "nqdef_e4=mix:lo=$PD_A/nq2,hi=$PD_A/nq4,set=$OUT/defset.json,hi_layers=3-6" ;;
  decB) launch decB predecode-nq --levels 4 --l4-set all --out "$PD_B" ;;
  runB) launch runB run --corpora $CORPORA --local-err --save-ref --moe-chunk $MOE_CHUNK --tag passB \
          --cand "nq4=dir:$PD_B/nq4" ;;
  decS) launch decS predecode-nq --fast --levels 4 --l4-set "$OUT/l4_sweep_union.json" --out "$PD_S" ;;
  runS1) launch runS1 run --corpora $CORPORA --local-err --moe-chunk $MOE_CHUNK --tag passS1 \
          --cand "nqdef64=mix:lo=$PD_A/nq2,hi=$PD_S/nq4,set=$OUT/defset_top64.json" \
          --cand "nqfloat0=mix:lo=$PD_A/nq2,hi=$PD_S/nq4,set=$OUT/defset_float0.json" ;;
  runS2) launch runS2 run --corpora $CORPORA --local-err --moe-chunk $MOE_CHUNK --tag passS2 \
          --cand "nqdef128=mix:lo=$PD_A/nq2,hi=$PD_S/nq4,set=$OUT/defset_top128.json" ;;
  decS2) launch decS2 predecode-nq --fast --levels 4 --l4-set "$OUT/l4_complement.json" --out "$OUT/predecoded_S2" ;;
  runS3) export NQ_SHARD=contig; launch runS3 run --corpora $CORPORA --local-err --moe-chunk $MOE_CHUNK --tag passS3 \
          --cand "nqadapt=adapt:lo=$PD_A/nq2,hi=$PD_S/nq4,hi2=$OUT/predecoded_S2/nq4,manifest=$SERVE_MAN" \
          --cand "nqadapt_chain=adapt:lo=$PD_A/nq2,hi=$PD_S/nq4,hi2=$OUT/predecoded_S2/nq4,manifest=$SERVE_MAN,chain=1" ;;
  runS4) launch runS4 run --corpora $CORPORA --local-err --moe-chunk $MOE_CHUNK --tag passS4 \
          --cand "nqdef=mix:lo=$PD_A/nq2,hi=$PD_A/nq4,set=$OUT/defset.json" \
          --cand "nqfloat0=mix:lo=$PD_A/nq2,hi=$PD_S/nq4,set=$OUT/defset_float0.json" \
          --cand "nqdef128=mix:lo=$PD_A/nq2,hi=$PD_S/nq4,set=$OUT/defset_top128.json" ;;
  hdump) shift; launch hdump run --corpora calib-fit --dump-layers ${DUMP_LAYERS:-29-32} --dump-stop \
          --dump-what x,ids,p,shared,moe_out,d_ref,h_in,h_mid,h_out --dump-dir "${DUMP_DIR:-$OUT/hdump}" --tag hdump \
          --cand "nqdef=mix:lo=$PD_A/nq2,hi=$PD_A/nq4,set=$OUT/defset.json" "$@" ;;
  merge) "$HERE/run.sh" merge --tag "$2" ;;
  *) sed -n '2,10p' "$0"; exit 1 ;;
esac
