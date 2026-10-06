#!/bin/bash
# T37 jF pipeline (GLM-5.3-Flash): fp8 / NVFP4 DECODE trace (dec37) -> chains/blocks -> v2 GBDT rows -> v2 GBDT -> jF features -> jF
# train -> offline eval (pooled sal-hot at v2-matched churn) -> streaming parity -> serve export.
#
#   bash run_jf37.sh [stage ...]       stages: chains blk fixed gbdtfeat gbdt jffeat train eval parity export
#                                      (default: all, in order; every stage is resumable / idempotent-ish)
# Env (defaults = full Flash run):
#   J37_TRACE   /tmp/nestquant/37-flash/private/dec_trace   dec37 decode trace (L{L}.r{r}of{W}.npz ids/w/xn as
#               capture37g + tok.r{r}of{W}.npy + seqs.r{r}of{W}.json; schema in blocks37.py).  jF is NEVER fit on the
#               teacher-forced capture trace (blocks37 refuses unless J37_ALLOW_TF=1, smoke only)
#               ':'-separated list allowed (e.g. dec37 + the online Flash-generated corpus teacher-forced, both with
#               seqs/tok metadata); groups shared across traces land in one split.  REQUIRED: a run refuses without
#               decode chains in train and val.
#   J37_PREFILL_TRACE ""   optional SECOND source, kind "prefill": the teacher-forced capture37g trace (windows.r*.json,
#               e.g. /tmp/nestquant/37-flash/private/trace); splits by document (test = corpus held-out windows)
#   J37_DEC_PREFILL 0      1 = the decode traces' prompt rows (before prompt_len) become kind-"prefill" chains
#   J37_PREFILL_FRAC 0.20  prefill share of the strided TRAIN rows (80/20 decode/prefill; prefill blocks subsampled
#               deterministically, decode never); J37_PREFILL_MAXFRAC 1.0 optional guard (refuse above it)
#   J37_PF_CHUNK 1024      prefill_c<N> eval (set frozen per N-token chunk) = upper bound for short (<1K) follow-up
#               turns only; the Mac never refreshes inside a >=1K layer-major prefill (mac/SPEC.md sec. 4)
#   J37_HANDOFF 1  J37_HO_PROMPT 4096  J37_HO_DEC 16   prefill->decode HANDOFF eval (primary prefill-related metric):
#               val/test decode sequences with a prompt; seeded (step_chunk over the last 4096 prompt tokens + one
#               refresh at hm 0) vs cold (floating_default), decode sal-hot over the first 1 / 4 / 16 blocks
#   J37_VALBLK 4096 / J37_VALBLK_PF 2048   decode (model selection) / prefill (secondary) val blocks per layer
#   J37_DEC_PROMPT 0 (prompt-tail tokens leading each chain)  J37_DEC_MAXCHAIN 0 (cut long decodes; 0 = 1 chain/seq)
#   J37_TESTFRAC 0.10
#   J37_OUT     /tmp/nestquant/37-flash/jf              PRIVATE intermediates (chains, blocks, features, models)
#   J37_FIXED   /tmp/nestquant/37-flash/cap-txt/fixed_set.json   REAP fixed set ("fixed_set"[str(L)], 19 / layer)
#   J37_LAYERS  3-44      J37_NF 48     J37_NFIX 19     J37_VALFRAC 0.08    J37_CHAIN 8192
#   J37_REL     /tmp/nestquant/37-flash/release/serving/predictor     serve export target
#   NAME jF  DEV cpu|cuda  BUDGET (train wall minutes; GLM-5.3 jF = 10 on a B200)  TRAIN_ARGS (extra train37 args)
#   NPROC 12 (pool workers, 1 thread each)  THREADS 20 (lightgbm / torch threads)
#   ALLOW_PROVISIONAL_FIXED=1  if J37_FIXED is missing, use a routed-count top-19 set from the trace (SMOKE ONLY)
#   VMEM_POOL_GB 8 (per pool worker, RLIMIT_AS)  VMEM_GB 100 (single-process stages)  MEM_ABORT_GB 1600
#     -> the node-wide unreclaimable memory (anon + shmem in the cgroup) is checked before every stage; abort above
#        MEM_ABORT_GB (cgroup limit 2066 GB; the running capture uses ~290 GB).  An OOM wipes /tmp.
# CPU only by default; nice 10.  Never touches GPUs unless DEV=cuda.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE"
PY=${PY:-/tmp/venv-t37/bin/python}
export J37_TRACE=${J37_TRACE:-/tmp/nestquant/37-flash/private/dec_trace}
export J37_OUT=${J37_OUT:-/tmp/nestquant/37-flash/jf}
export J37_FIXED=${J37_FIXED:-/tmp/nestquant/37-flash/cap-txt/fixed_set.json}
export J37_LAYERS=${J37_LAYERS:-3-44}
export J37_REL=${J37_REL:-/tmp/nestquant/37-flash/release/serving/predictor}
NAME=${NAME:-jF}; DEV=${DEV:-cpu}; BUDGET=${BUDGET:-120}; TRAIN_ARGS=${TRAIN_ARGS:-}
NPROC=${NPROC:-12}; THREADS=${THREADS:-20}
VMEM_POOL_GB=${VMEM_POOL_GB:-8}; VMEM_GB=${VMEM_GB:-100}; MEM_ABORT_GB=${MEM_ABORT_GB:-1600}
PAR_NBK=${PAR_NBK:-64}; PAR_NCH=${PAR_NCH:-2}
[ "$DEV" = cpu ] && export CUDA_VISIBLE_DEVICES=""
(( NPROC <= 16 )) || { echo "NPROC <= 16 (20-thread budget)"; exit 1; }
mkdir -p "$J37_OUT/logs"
LOG="$J37_OUT/logs/run_jf37.$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
ts() { TZ=Australia/Melbourne date '+%F %T %Z'; }

memcheck() {
  local u; u=$(awk '$1=="anon"||$1=="shmem"{s+=$2} END{printf "%d", s/1e9}' /sys/fs/cgroup/memory.stat)
  echo "[$(ts)] cgroup unreclaimable ${u} GB (abort > ${MEM_ABORT_GB})"
  (( u < MEM_ABORT_GB )) || { echo "ABORT: node memory too high"; exit 1; }
}
# run GB cmd...  : RLIMIT_AS cap (per process; pool workers inherit it individually), nice 10
run() {
  local gb=$1; shift; memcheck
  echo "[$(ts)] >>> $* (RLIMIT_AS ${gb} GB / process)"
  ( ulimit -v $((gb * 1024 * 1024)); exec nice -n 10 "$@" )
  echo "[$(ts)] <<< done"
}
pool() { OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 run "$VMEM_POOL_GB" "$@"; }
single() { OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS run "$VMEM_GB" "$@"; }

PROV="$J37_OUT/fixed_provisional.json"
resolve_fixed() {
  if [ ! -f "$J37_FIXED" ] && [ -f "$PROV" ] && [ "${ALLOW_PROVISIONAL_FIXED:-0}" = 1 ]; then export J37_FIXED=$PROV; fi
}
check_fixed() {    # features / models depend on the fixed set: refuse to mix sets within one OUT
  resolve_fixed
  [ -f "$J37_FIXED" ] || { echo "missing fixed set $J37_FIXED (fixed_set37.py; or ALLOW_PROVISIONAL_FIXED=1 for smoke)"; exit 1; }
  local s; s=$(sha256sum "$J37_FIXED" | cut -c1-64)
  if [ -f "$J37_OUT/fixed.sha256" ]; then
    [ "$(cat "$J37_OUT/fixed.sha256")" = "$s" ] || { echo "fixed set changed since features were built: rm -r $J37_OUT/{gbdt_rows,models,feat,scores,eval} $J37_OUT/fixed.sha256"; exit 1; }
  else echo "$s" > "$J37_OUT/fixed.sha256"; fi
  echo "fixed set $J37_FIXED sha256 ${s:0:12}"
}

STAGES=("$@"); [ ${#STAGES[@]} -eq 0 ] && STAGES=(chains blk fixed gbdtfeat gbdt jffeat train eval parity export)
echo "[$(ts)] run_jf37 stages ${STAGES[*]} | TRACE $J37_TRACE PREFILL_TRACE ${J37_PREFILL_TRACE:-none} DEC_PREFILL ${J37_DEC_PREFILL:-0} PREFILL_FRAC ${J37_PREFILL_FRAC:-0.20} | OUT $J37_OUT LAYERS $J37_LAYERS NF ${J37_NF:-48} DEV $DEV"
for st in "${STAGES[@]}"; do
  case $st in
    chains)   pool "$PY" blocks37.py chains ;;
    blk)      pool "$PY" blocks37.py blk "$NPROC" ;;
    fixed)    if [ -f "$J37_FIXED" ]; then echo "fixed set present: $J37_FIXED"
              elif [ "${ALLOW_PROVISIONAL_FIXED:-0}" = 1 ]; then pool "$PY" fixed_from_trace.py "$PROV"
              else echo "missing $J37_FIXED (set ALLOW_PROVISIONAL_FIXED=1 for a smoke run)"; exit 1; fi ;;
    gbdtfeat) check_fixed; pool "$PY" feat37.py gbdt "$NPROC" ;;
    gbdt)     check_fixed; single "$PY" gbdt37.py --threads "$THREADS" ;;
    jffeat)   check_fixed; pool "$PY" feat37.py jf "$NPROC" ;;
    train)    check_fixed; single "$PY" train37.py "$NAME" --arch tf --d 96 --nl 2 --tw 1 --noemb --budget "$BUDGET" \
                --dev "$DEV" --threads "$THREADS" $TRAIN_ARGS ;;
    eval)     check_fixed; pool "$PY" eval37.py "$NAME" --nproc "$NPROC" ;;
    parity)   check_fixed; OMP_NUM_THREADS=8 run "$VMEM_GB" "$PY" parity37.py "$NAME" "$PAR_NBK" "$PAR_NCH" ;;
    export)   check_fixed; single "$PY" export37.py "$NAME" --rel "$J37_REL"
              OMP_NUM_THREADS=8 run "$VMEM_GB" "$PY" parity37.py "$NAME" 16 1 --model-dir "$J37_REL/joint" ;;
    *) echo "unknown stage $st"; exit 1 ;;
  esac
done
echo "[$(ts)] run_jf37 finished; log $LOG"
