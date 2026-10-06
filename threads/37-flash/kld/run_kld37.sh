#!/bin/bash
# T37 end-to-end KLD (NestQuant 1.5/4 GLM-5.3-Flash vs fp8 teacher, 146 held-out val segments, 106K rows) on 8 A100s.
#   nohup bash run_kld37.sh > /tmp/nestquant/37-flash/kld/logs/run.out 2>&1 &
# Waits (WAIT=1, default) until the encode is complete (L3-L44: 288 experts + fin/L{L}.json each), then takes
# /tmp/nestquant/33-search/gpu.lock (flock, blocks while the campaign holds it), checks memory, runs all arms in ONE
# layer-sequential pass. Aggregates -> /tmp/nestquant/37-flash/kld/results_$TAG.json; PRIVATE per-segment detail ->
# /tmp/nestquant/37-flash/private/kld37/$TAG/.  Expected wall ~40-60 min (see README in the final report).
set -u
KD=/home/coder/git/nestquant/threads/37-flash/kld
OUT=/tmp/nestquant/37-flash/kld; LOGD=$OUT/logs; mkdir -p $LOGD
ENC=${NQ37_ENC:-/tmp/nestquant/37-flash/enc_b15}
ARMS=${ARMS:-fp8,all2,mac63,mac59,stat63,all4}
TAG=${TAG:-full}
WAIT=${WAIT:-1}
NPROC=${NPROC:-8}
EXTRA=${EXTRA:-}                  # e.g. "--plan .../plan_bmconf.pt" (brandonmusic Flash BF16 teacher)
NET=${NET:-}                      # default inside kld37.py: jf/models/jF63.pt if present, else release nf48 jF.pt
LOCK=/tmp/nestquant/33-search/gpu.lock
PY=/tmp/venv-t37g/bin
export TZ=Australia/Melbourne
ts() { date '+%F %T %Z'; }

enc_done() {
  for L in $(seq 3 44); do
    [ -f $ENC/fin/L$L.json ] || return 1
    [ "$(ls $ENC/L$L/experts 2>/dev/null | grep -c '^E[0-9]*\.pt$')" = 288 ] || return 1
  done
  return 0
}

while ! enc_done; do
  if [ "$WAIT" != 1 ]; then echo "[$(ts)] encode incomplete ($ENC) -- abort (WAIT=1 to wait)"; exit 1; fi
  echo "[$(ts)] waiting for encode: $(ls $ENC/fin 2>/dev/null | grep -c '^L[0-9]*\.json$')/42 layers finalised"
  sleep 300
done
echo "[$(ts)] encode complete"
[ -f /tmp/nestquant/37-flash/private/kld37/plan.pt ] || { echo "plan missing: run kld37.py prep"; exit 1; }

echo "[$(ts)] waiting for $LOCK"
exec 9>>$LOCK
flock 9
echo "[$(ts)] gpu.lock acquired"

# memory: harness host use is small (~25 GB total: 8 ranks x ~3 GB + page cache); refuse if the box is already loaded
u=$(awk '$1=="anon"||$1=="shmem"{s+=$2} END{printf "%d", s/1e9}' /sys/fs/cgroup/memory.stat)
echo "[$(ts)] cgroup anon+shmem ${u} GB"
if [ "$u" -gt 1200 ]; then echo "anon+shmem ${u} GB > 1200 -- refusing to start"; exit 1; fi
busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>20000' | wc -l)
if [ "$busy" -gt 0 ]; then echo "[$(ts)] WARNING: $busy GPU(s) have >20 GB in use"; nvidia-smi --query-gpu=index,memory.used --format=csv; fi

cd $KD
OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NQ37_ENC=$ENC NQ37_KVQ=${NQ37_KVQ:-} \
  nice -n 10 $PY/torchrun --nproc-per-node $NPROC --master-port 29631 kld37.py run --arms $ARMS --tag $TAG $EXTRA \
  ${NET:+--net $NET} > $LOGD/kld37_$TAG.log 2>&1 &
TR=$!
echo $TR >> /tmp/nestquant/37-flash/guard.pids
echo "[$(ts)] torchrun pid $TR (log $LOGD/kld37_$TAG.log)"
wait $TR; rc=$?
echo "[$(ts)] done rc=$rc"
grep '\[kld37' $LOGD/kld37_$TAG.log | tail -25
exit $rc
