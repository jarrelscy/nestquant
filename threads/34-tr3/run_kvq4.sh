#!/bin/bash
# T34 fp8_ds_mla KV-cost pass, same 4 teacher windows, ONE gpu.lock hold. No-KV control = passT34H1/H2.
# Two WORLD=4 groups, same arms (fp8 ref scored inline, also under KVQ):
#   G1 GPUs 0-3 passT34K1 NQ_KVQ=kvq (full serve: fp8 latent cache + fp8 Q-latent)   G2 GPUs 4-7 passT34K2 NQ_KVQ=kv (KV cache only)
set -euo pipefail
O=/tmp/nestquant/34-tr3; PD=/tmp/nestquant/src/predec
G=/tmp/nestquant/32-gbdt-sal; J=/tmp/nestquant/33-search/joint; M26=/tmp/nestquant/28-serve-release/out/serving/tp4/manifest.json
R=/home/coder/git/nestquant/threads/18-e2e-eval/run.sh
LO=$PD/farm/nq2; HI=$PD/farm/nq4
AD=adapt:lo=$LO,hi=$HI,chain=map,salstat=1
JF="$AD,predictor=gbdt,gmode=sync,joint=$J/models/jF_all.pt"
C1="--cand serve=$AD,manifest=$M26,predictor=gbdt,gbdt_scale=mps,gmode=sync"
C1="$C1 --cand jF77_hm07=$JF,manifest=$G/k0_manifest.json,n_float=77,hm=0.7"
C1="$C1 --cand jF128=$JF,manifest=$G/k0_manifest.json,n_float=128,hm=0.7"
C1="$C1 --cand jF173=$JF,manifest=$G/kB_manifest.json,n_float=173,hm=0.7"
C1="$C1 --cand nq4=dir:$HI"; C2="$C1"
mkdir -p $O/e2e/logs
f=$O/kld_passT34K.tmp.sh
{ echo '#!/bin/bash'; echo 'set -uo pipefail'; echo 'rc=0'; echo "echo \"take GPUs passT34K \$(date -u)\""
  echo "export NQ_SHARD=contig OMP_NUM_THREADS=2 NQ_VRAM_GB=20 NQ_GBDT_THREADS=2 NQ_OUT=$O/e2e NQ_CORPUS_DIR=$O/corpora"
  echo "export NQ_TEACHER=bf16conf=$O/teacher/reference-full-panel/logits/confirmation"
  echo 'pids=()'
  echo 'for r in 0 1 2 3; do'
  echo "  NQ_KVQ=kvq CUDA_VISIBLE_DEVICES=\$r RANK=\$r WORLD=4 nohup timeout 5300 $R run --corpora bf16conf --max-windows 4 --moe-chunk 16384 --tag passT34K1 $C1 > $O/e2e/logs/passT34K1.r\$r.log 2>&1 &"
  echo '  pids+=($!)'
  echo "  NQ_KVQ=kv CUDA_VISIBLE_DEVICES=\$((r+4)) RANK=\$r WORLD=4 nohup timeout 5300 $R run --corpora bf16conf --max-windows 4 --moe-chunk 16384 --tag passT34K2 $C2 > $O/e2e/logs/passT34K2.r\$r.log 2>&1 &"
  echo '  pids+=($!)'
  echo 'done'
  echo 'for p in "${pids[@]}"; do wait $p || rc=1; done'
  echo "echo \"release GPUs passT34K rc=\$rc \$(date -u)\"; exit \$rc"; } > $f
chmod +x $f
flock /tmp/nestquant/33-search/gpu.lock $f
echo "run_kvq4 done $(date -u)"
