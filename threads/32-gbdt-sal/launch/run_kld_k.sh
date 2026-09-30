#!/bin/bash
# T32 held k=0 / k=6 fixed-set KLD pass (approved 2026-09-30).  After the /tmp wipe: (1) fast-predecode nq2 + nq4 (all
# experts) of L7-77 from the HF TP8 containers into /tmp/nestquant/src/predec/fast (outside 18-e2e, not backed up);
# L3-6 (h512 refit) come from predecoded_H via a z_h512/ subdir (SafeIndex sorted order, as T18 farm_H).
# (2) passT32Kk: nq-tail + wikitext, 32 windows each, contig, chain=1, 8 ranks, arms in the approved order.
# Each GPU phase is its own gpu.lock hold.
set -euo pipefail
O=/tmp/nestquant/32-gbdt-sal; PD=/tmp/nestquant/src/predec; PH=/tmp/nestquant/18-e2e/predecoded_H
R=/home/coder/git/nestquant/threads/18-e2e-eval/run.sh; LK=/tmp/nestquant/33-search/gpu.lock
LOGS=/tmp/nestquant/18-e2e/logs; mkdir -p $LOGS
TAG=${TAG:-passT32Kk}
until grep -q "fetch done" $O/logs/fetch_nq.log; do sleep 30; done
[ "$(ls -d /tmp/nestquant/nq-encode-v1/L* | wc -l)" -ge 75 ] || { echo "missing layer dirs"; exit 1; }

cat > $O/predec_k.tmp.sh <<EOS
#!/bin/bash
set -euo pipefail
echo "take GPUs predec \$(date -u)"
pids=()
for r in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=\$r RANK=\$r WORLD=8 NQ_VRAM_GB=40 OMP_NUM_THREADS=2 nohup $R predecode-nq --fast --levels 2,4 \
    --l4-set all --layers 7-77 --root /tmp/nestquant/nq-encode-v1 --out $PD/fast > $LOGS/t32predec.r\$r.log 2>&1 &
  pids+=(\$!)
done
rc=0; for p in "\${pids[@]}"; do wait \$p || rc=1; done
echo "release GPUs predec rc=\$rc \$(date -u)"; exit \$rc
EOS
chmod +x $O/predec_k.tmp.sh
flock $LK $O/predec_k.tmp.sh
for lv in 2 4; do
  mkdir -p $PD/farm/nq$lv/z_h512
  ln -sf $PD/fast/nq$lv/*.safetensors $PD/farm/nq$lv/
  ln -sf $PH/nq$lv/*.safetensors $PD/farm/nq$lv/z_h512/
  echo "farm nq$lv: $(find -L $PD/farm/nq$lv -name '*.safetensors' | wc -l) files"
done

A=adapt:lo=$PD/farm/nq2,hi=$PD/farm/nq4,chain=1,salstat=1,predictor=gbdt,gbdt_model=/home/coder/git/nestquant/streaming/gbdt_v2sal_p64.txt,gmode=sync,grlo=0,grhi=256
M26=/tmp/nestquant/28-serve-release/out/serving/tp4/manifest.json
cat > $O/kld_k.tmp.sh <<EOS
#!/bin/bash
set -euo pipefail
echo "take GPUs $TAG \$(date -u)"
export NQ_SHARD=contig OMP_NUM_THREADS=2 NQ_VRAM_GB=20
pids=()
for r in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=\$r RANK=\$r WORLD=8 nohup $R run --corpora nq-tail,wikitext --max-windows 32 --moe-chunk 16384 \
    --tag $TAG \
    --cand k26_v2=$A,manifest=$M26 \
    --cand k0_hm06=$A,manifest=$O/k0_manifest.json,n_float=77,hm=0.6 \
    --cand k6_hm055=$A,manifest=$O/k6_manifest.json,n_float=71,hm=0.55 \
    --cand k0_hm05=$A,manifest=$O/k0_manifest.json,n_float=77,hm=0.5 \
    > $LOGS/$TAG.r\$r.log 2>&1 &
  pids+=(\$!)
done
rc=0; for p in "\${pids[@]}"; do wait \$p || rc=1; done
echo "release GPUs $TAG rc=\$rc \$(date -u)"; exit \$rc
EOS
chmod +x $O/kld_k.tmp.sh
flock $LK $O/kld_k.tmp.sh
$R merge --tag $TAG
for b in k0_hm06 k6_hm055 k0_hm05; do
  /home/coder/git/glm52/.venv/bin/python /home/coder/git/nestquant/threads/18-e2e-eval/nq_paired.py --tag $TAG --a k26_v2 --b $b
done
echo "kld_k done $(date -u)"
