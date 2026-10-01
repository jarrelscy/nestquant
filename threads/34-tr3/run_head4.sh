#!/bin/bash
# T34 headline 4-window teacher pass (confirmation-0000..0003 = their README table), ONE gpu.lock hold.
# Two independent WORLD=4 groups (1 window per rank), arms split across groups; fp8 (inline ref) scored in both:
#   G1 GPUs 0-3 passT34H1: serve jF77_hm07 jF173      G2 GPUs 4-7 passT34H2: nq4 nqS173
set -euo pipefail
O=/tmp/nestquant/34-tr3; PD=/tmp/nestquant/src/predec
G=/tmp/nestquant/32-gbdt-sal; J=/tmp/nestquant/33-search/joint; M26=/tmp/nestquant/28-serve-release/out/serving/tp4/manifest.json
R=/home/coder/git/nestquant/threads/18-e2e-eval/run.sh
LO=$PD/farm/nq2; HI=$PD/farm/nq4
AD=adapt:lo=$LO,hi=$HI,chain=map,salstat=1
JF="$AD,predictor=gbdt,gmode=sync,joint=$J/models/jF_all.pt"
C1="--cand serve=$AD,manifest=$M26,predictor=gbdt,gbdt_scale=mps,gmode=sync"
C1="$C1 --cand jF77_hm07=$JF,manifest=$G/k0_manifest.json,n_float=77,hm=0.7"
C1="$C1 --cand jF173=$JF,manifest=$G/kB_manifest.json,n_float=173,hm=0.7"
C2="--cand nq4=dir:$HI --cand nqS173=mix:lo=$LO,hi=$HI,set=$O/nqS173.json"
mkdir -p $O/e2e/logs
f=$O/kld_passT34H.tmp.sh
{ echo '#!/bin/bash'; echo 'set -uo pipefail'; echo 'rc=0'; echo "echo \"take GPUs passT34H \$(date -u)\""
  echo "export NQ_SHARD=contig OMP_NUM_THREADS=2 NQ_VRAM_GB=20 NQ_GBDT_THREADS=2 NQ_OUT=$O/e2e NQ_CORPUS_DIR=$O/corpora"
  echo "export NQ_TEACHER=bf16conf=$O/teacher/reference-full-panel/logits/confirmation"
  echo 'pids=()'
  echo 'for r in 0 1 2 3; do'
  echo "  CUDA_VISIBLE_DEVICES=\$r RANK=\$r WORLD=4 nohup timeout 5300 $R run --corpora bf16conf --max-windows 4 --moe-chunk 16384 --tag passT34H1 $C1 > $O/e2e/logs/passT34H1.r\$r.log 2>&1 &"
  echo '  pids+=($!)'
  echo "  CUDA_VISIBLE_DEVICES=\$((r+4)) RANK=\$r WORLD=4 nohup timeout 5300 $R run --corpora bf16conf --max-windows 4 --moe-chunk 16384 --tag passT34H2 $C2 > $O/e2e/logs/passT34H2.r\$r.log 2>&1 &"
  echo '  pids+=($!)'
  echo 'done'
  echo 'for p in "${pids[@]}"; do wait $p || rc=1; done'
  echo "echo \"release GPUs passT34H rc=\$rc \$(date -u)\"; exit \$rc"; } > $f
chmod +x $f
flock /tmp/nestquant/33-search/gpu.lock $f
echo "run_head4 done $(date -u)"
