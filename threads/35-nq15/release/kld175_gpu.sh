#!/bin/bash
# T35 b175 full-real KLD (BF16 teacher conf 0000-0003, NQ_KVQ=kvq = fp8_ds_mla KV + fp8 q-latent; 2x Spark fp8 KV 128K
# -> H45 hot/layer, jF k0 hm0.7).  Waits for 75/75 + driver exit; each GPU phase is its own gpu.lock hold (<=85 min).
set -uo pipefail
T=/tmp/nestquant/35-nq15; ENC=$T/enc_b175; PD=$T/pd175; LOCK=/tmp/nestquant/33-search/gpu.lock
PY=/home/coder/git/glm52/.venv/bin/python; C=/home/coder/git/nestquant/threads/35-nq15
log(){ echo "$(date -u +%FT%TZ) $*"; }
fin_ok(){ $PY - <<'P'
import json,glob,sys
ok=0
for f in glob.glob('/tmp/nestquant/35-nq15/enc_b175/fin/L*.json'):
    try: ok+=json.load(open(f)).get('rc',1)==0
    except Exception: pass
sys.exit(0 if ok>=75 else 1)
P
}
until fin_ok && ! kill -0 35009 2>/dev/null; do sleep 60; done
log "75/75 fin and driver 35009 exited"
# ---- hold 1: predecode (8 GPUs, resumable) ----
for attempt in 1 2; do
  n=$(ls $PD/nq2/*.safetensors $PD/nq4/*.safetensors 2>/dev/null | wc -l); [ "$n" -ge 1200 ] && break
  flock $LOCK bash -c "
    echo \"\$(date -u +%FT%TZ) take GPUs T35 predec175 attempt $attempt\"; pids=()
    for g in 0 1 2 3 4 5 6 7; do
      CUDA_VISIBLE_DEVICES=\$g RANK=\$g WORLD=8 NQ_DEV=cuda:0 OMP_NUM_THREADS=2 timeout 5100 nice -n 10 $PY $C/predec175.py $ENC $PD 3-77 2,4 > $T/logs/predec175.r\$g.log 2>&1 & pids+=(\$!)
    done
    for p in \${pids[@]}; do wait \$p; done
    echo \"\$(date -u +%FT%TZ) release GPUs T35 predec175\""
done
n=$(ls $PD/nq2/*.safetensors $PD/nq4/*.safetensors 2>/dev/null | wc -l); log "predecode files $n/1200"; [ "$n" -ge 1200 ] || exit 2
# ---- hold 2: eval, 4 windows, 1 per GPU ----
export NQ_SHARD=contig NQ_VRAM_GB=20 NQ_GBDT_THREADS=2 OMP_NUM_THREADS=2 NQ_OUT=$T/e2e NQ_CORPUS_DIR=/tmp/nestquant/34-tr3/corpora NQ_KVQ=kvq
export NQ_TEACHER=bf16conf=/tmp/nestquant/34-tr3/teacher/reference-full-panel/logits/confirmation
AD=adapt:lo=$PD/nq2,hi=$PD/nq4,chain=map,salstat=1,predictor=gbdt,gmode=sync,joint=/tmp/nestquant/33-search/joint/models/jF_all.pt,manifest=/tmp/nestquant/32-gbdt-sal/k0_manifest.json,hm=0.7
flock $LOCK bash -c "
  echo \"\$(date -u +%FT%TZ) take GPUs T35 passT35F\"; pids=()
  for r in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=\$r RANK=\$r WORLD=4 timeout 5100 nice -n 10 $T/run175.sh run --corpora bf16conf --max-windows 4 --moe-chunk 16384 --tag passT35F --cand real175_45=$AD,n_float=45 > $T/e2e/logs/passT35F.r\$r.log 2>&1 & pids+=(\$!)
  done
  rc=0; for p in \${pids[@]}; do wait \$p || rc=1; done
  echo \"\$(date -u +%FT%TZ) release GPUs T35 passT35F rc=\$rc\""
$PY $C/kld175_report.py $T/e2e/results/passT35F $T/kld175_passT35F.json
log "passT35F done"
