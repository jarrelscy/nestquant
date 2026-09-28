#!/bin/bash
# Thread 18 full-model NestQuant eval (nq2 / nqdef / nq4 + L3-6 attribution arms), two passes to keep /tmp use
# at ~1.7 TB peak (one full fp16 level on disk at a time).
#   ./run_full.sh snap      snapshot the default 4-bit set (manifests) -> $OUT/defset.json, $OUT/l4sub.json
#   ./run_full.sh decA      predecode nq2 (all) + nq4 (default set + all of L3-6)   -> $PD_A/{nq2,nq4}
#   ./run_full.sh runA      ref + nq2 + nqdef + nq2_early (nq2 on L3-6 only) + nqdef_e4 (nqdef, L3-6 all 4-bit)
#   ./run_full.sh decB      predecode nq4 (all)                                      -> $PD_B/nq4
#   ./run_full.sh runB      ref + nq4
#   ./run_full.sh merge TAG
# Every stage: 8 ranks, one per GPU, NQ_VRAM_GB cap, launch refused if a GPU has < MIN_FREE_MB free.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export NQ_OUT=${NQ_OUT:-/tmp/nestquant/18-e2e}
OUT=$NQ_OUT
PD_A=$OUT/predecoded_A PD_B=$OUT/predecoded_B
export NQ_VRAM_GB=${NQ_VRAM_GB:-11}
MIN_FREE_MB=${MIN_FREE_MB:-14000}
WORLD=8
CORPORA=nq-tail,vllm-docs,wikitext,github
MOE_CHUNK=${MOE_CHUNK:-40960}
LOGS=$OUT/logs; mkdir -p "$LOGS"

gpu_check() {
  local bad=0
  while IFS=, read -r i free; do
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
    CUDA_VISIBLE_DEVICES=$r RANK=$r WORLD=$WORLD nohup "$HERE/run.sh" "$@" > "$LOGS/$name.r$r.log" 2>&1 &
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
  runA) launch runA run --corpora $CORPORA --local-err --save-ref --moe-chunk $MOE_CHUNK --tag passA \
          --cand "nq2=dir:$PD_A/nq2" \
          --cand "nqdef=mix:lo=$PD_A/nq2,hi=$PD_A/nq4,set=$OUT/defset.json" \
          --cand "nq2_early=dir:$PD_A/nq2,layers=3-6" \
          --cand "nqdef_e4=mix:lo=$PD_A/nq2,hi=$PD_A/nq4,set=$OUT/defset.json,hi_layers=3-6" ;;
  decB) launch decB predecode-nq --levels 4 --l4-set all --out "$PD_B" ;;
  runB) launch runB run --corpora $CORPORA --local-err --moe-chunk $MOE_CHUNK --tag passB \
          --cand "nq4=dir:$PD_B/nq4" ;;
  merge) "$HERE/run.sh" merge --tag "$2" ;;
  *) sed -n '2,10p' "$0"; exit 1 ;;
esac
