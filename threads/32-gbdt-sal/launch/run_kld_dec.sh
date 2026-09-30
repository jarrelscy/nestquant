#!/bin/bash
# T32 decode-text KLD pass (coordinator 2026-09-30): the same k-arms as passT32Kk on SM120's own generations
# (private/corpora/sm120tfk: 6 tasks x 3 consecutive all-decode 2048-token windows of the sm120tf TF corpus), WORLD=6,
# one task per rank = one chain (state carried across the task's windows), contig, chain=1.  PRIVATE: NQ_OUT under
# private/ (per-token KL never leaves the box, not under 18-e2e/results).  Waits for passT32Kk, one gpu.lock hold.
set -euo pipefail
O=/tmp/nestquant/32-gbdt-sal; P=$O/private; PD=/tmp/nestquant/src/predec
R=/home/coder/git/nestquant/threads/18-e2e-eval/run.sh; TAG=${TAG:-passT32Kd}
until grep -q "release GPUs passT32Kk" $O/logs/kld_k.log; do sleep 30; done
A=adapt:lo=$PD/farm/nq2,hi=$PD/farm/nq4,chain=1,salstat=1,predictor=gbdt,gbdt_model=/home/coder/git/nestquant/streaming/gbdt_v2sal_p64.txt,gmode=sync,grlo=0,grhi=256
M26=/tmp/nestquant/28-serve-release/out/serving/tp4/manifest.json
CANDS="--cand k26_v2=$A,manifest=$M26 --cand k0_hm06=$A,manifest=$O/k0_manifest.json,n_float=77,hm=0.6"
CANDS="$CANDS --cand k6_hm055=$A,manifest=$O/k6_manifest.json,n_float=71,hm=0.55"
CANDS="$CANDS --cand k0_hm05=$A,manifest=$O/k0_manifest.json,n_float=77,hm=0.5"
CANDS="$CANDS --cand k0_hm10=$A,manifest=$O/k0_manifest.json,n_float=77,hm=1.0"
mkdir -p $P/e2e_dec/logs
{ echo '#!/bin/bash'; echo 'set -euo pipefail'; echo "echo \"take GPUs $TAG \$(date -u)\""
  echo "export NQ_SHARD=contig OMP_NUM_THREADS=2 NQ_VRAM_GB=20 NQ_OUT=$P/e2e_dec NQ_CORPUS_DIR=$P/corpora"
  echo 'pids=()'
  echo 'for r in 0 1 2 3 4 5; do'
  echo "  CUDA_VISIBLE_DEVICES=\$r RANK=\$r WORLD=6 nohup $R run --corpora sm120tfk --moe-chunk 16384 --tag $TAG $CANDS > $P/e2e_dec/logs/$TAG.r\$r.log 2>&1 &"
  echo '  pids+=($!)'
  echo 'done'
  echo 'rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done'
  echo "echo \"release GPUs $TAG rc=\$rc \$(date -u)\"; exit \$rc"; } > $O/kld_dec.tmp.sh
chmod +x $O/kld_dec.tmp.sh
flock /tmp/nestquant/33-search/gpu.lock $O/kld_dec.tmp.sh
NQ_OUT=$P/e2e_dec $R merge --tag $TAG
echo "kld_dec done $(date -u)"
