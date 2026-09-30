#!/bin/bash
# T35 Part B: full-model KLD vs the BF16 teacher on the 4 confirmation windows, fp8_ds_mla KV emulated (NQ_KVQ=kvq),
# same harness/settings as T34 passT34K1 (threads/34-tr3/run_kvq4.sh).  Cold experts = rate-b emulation
# W_ref + s(b) (W_nq2 - W_ref) (emu_adapt.AdaptEmu), hot = real L4, jF_all k0 hm 0.7 refresh 16 sync.
# ONE gpu.lock hold, two WORLD=4 groups on GPUs 0-3 (passT35B1) and 4-7 (passT35B2).
# s(b) = sqrt(r(b)/r(2)) from /tmp/nestquant/35-nq15/rate_mse.json (iid-Gaussian pattern-rate trellis).
set -euo pipefail
O=/tmp/nestquant/35-nq15; T34=/tmp/nestquant/34-tr3; PD=/tmp/nestquant/src/predec
G=/tmp/nestquant/32-gbdt-sal; J=/tmp/nestquant/33-search/joint
R=/home/coder/git/nestquant/threads/18-e2e-eval/run.sh
EMU=/home/coder/git/nestquant/threads/35-nq15/emu_adapt.py
S() { /home/coder/git/glm52/.venv/bin/python -c "
import json
for f in ('$O/rate_mse.json', '$O/rate_mse_125.json'):
    try:
        print('%.6f' % json.load(open(f))['res']['$1']['s']); break
    except (KeyError, FileNotFoundError): pass
else: raise SystemExit('no s for $1')"; }
# env S175/S15/S10/S125 override (values from the rate_mse logs when the json is not written yet)
S175=${S175:-$(S 1.75)}; S15=${S15:-$(S 1.5)}; S10=${S10:-$(S 1.0)}; S125=${S125:-$(S 1.25 2>/dev/null || echo "")}
[ -n "$S175" ] && [ -n "$S15" ] && [ -n "$S10" ]
echo "s(1.75)=$S175 s(1.5)=$S15 s(1.0)=$S10 s(1.25)=$S125"
A="lo=$PD/farm/nq2,hi=$PD/farm/nq4,chain=map,salstat=1,predictor=gbdt,gmode=sync,joint=$J/models/jF_all.pt,manifest=$G/k0_manifest.json,hm=0.7"
arm() { echo "--cand $1=py:$EMU:AdaptEmu:s=$2,$A,n_float=$3"; }
C1="$(arm e20_77 1.0 77) $(arm b20_6 1.0 6) $(arm b175_32 $S175 32)"
[ -n "$S125" ] && C1="$C1 $(arm b125_71 $S125 71)"
C2="$(arm b15_54 $S15 54) $(arm b10_86 $S10 86) $(arm b15_77 $S15 77)"
mkdir -p $O/e2e/logs
f=$O/kld_passT35B.tmp.sh
{ echo '#!/bin/bash'; echo 'set -uo pipefail'; echo 'rc=0'; echo "echo \"take GPUs passT35B \$(date -u)\""
  echo "export NQ_SHARD=contig OMP_NUM_THREADS=2 NQ_VRAM_GB=20 NQ_GBDT_THREADS=2 NQ_OUT=$O/e2e NQ_CORPUS_DIR=$T34/corpora"
  echo "export NQ_TEACHER=bf16conf=$T34/teacher/reference-full-panel/logits/confirmation"
  echo 'pids=()'
  echo 'for r in 0 1 2 3; do'
  echo "  NQ_KVQ=kvq CUDA_VISIBLE_DEVICES=\$r RANK=\$r WORLD=4 nohup timeout 5100 nice -n 10 $R run --corpora bf16conf --max-windows 4 --moe-chunk 16384 --tag passT35B1 $C1 > $O/e2e/logs/passT35B1.r\$r.log 2>&1 &"
  echo '  pids+=($!)'
  echo "  NQ_KVQ=kvq CUDA_VISIBLE_DEVICES=\$((r+4)) RANK=\$r WORLD=4 nohup timeout 5100 nice -n 10 $R run --corpora bf16conf --max-windows 4 --moe-chunk 16384 --tag passT35B2 $C2 > $O/e2e/logs/passT35B2.r\$r.log 2>&1 &"
  echo '  pids+=($!)'
  echo 'done'
  echo 'for p in "${pids[@]}"; do wait $p || rc=1; done'
  echo "echo \"release GPUs passT35B rc=\$rc \$(date -u)\"; exit \$rc"; } > $f
chmod +x $f
echo "waiting for gpu.lock $(date -u)"
flock /tmp/nestquant/33-search/gpu.lock $f
echo "run_partb done $(date -u)"
