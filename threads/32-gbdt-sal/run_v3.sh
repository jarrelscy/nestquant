#!/bin/bash
# T32 (a): after recapture.sh releases the GPUs: v3 router-prob rows -> train (sal/cnt, v2+p and p-only) -> heldout sim.
set -euo pipefail
O32=/tmp/nestquant/32-gbdt-sal; M=$O32/models; OLD=/home/coder/git/nestquant/streaming/gbdt_p64_s5.txt
export PYTHONPATH=/tmp/nestquant/18-e2e/pylib; PY=/home/coder/git/glm52/.venv/bin/python
cd "$(dirname "$0")"
until grep -q "release GPUs" $O32/logs/recapture.out 2>/dev/null; do sleep 30; done
for c in glm52-heldout calib-fit; do $PY build_v3.py $c 12; done
$PY train.py --v2 --v3 --target sal --obj tweedie:1.5 --threads 32 --out $M/v3_sal_tweedie1.5.txt
$PY train.py --v2 --v3 --target cnt --obj poisson --threads 32 --out $M/v3_cnt_poisson.txt
$PY train.py --v3 --target sal --obj tweedie:1.5 --threads 32 --out $M/p_sal_tweedie1.5.txt
$PY train.py --v3 --target cnt --obj poisson --threads 32 --out $M/p_cnt_poisson.txt
for lag in 1 0; do
  T32_LAG=$lag T32_TAG=_v3_lag$lag NPROC=12 $PY sim.py glm52-heldout gbdt_old=$OLD gbdt_x_mps=$OLD@mps \
    v2_sal=$M/v2_sal_tweedie1.5.txt v3_sal=$M/v3_sal_tweedie1.5.txt v3_cnt=$M/v3_cnt_poisson.txt \
    v3_cnt_x_mps=$M/v3_cnt_poisson.txt@mps p_sal=$M/p_sal_tweedie1.5.txt p_cnt=$M/p_cnt_poisson.txt \
    p_cnt_x_mps=$M/p_cnt_poisson.txt@mps
done
$PY - <<'P'
import lightgbm as lgb
for n in ("v3_sal_tweedie1.5", "v3_cnt_poisson", "p_sal_tweedie1.5", "p_cnt_poisson"):
    b = lgb.Booster(model_file=f"/tmp/nestquant/32-gbdt-sal/models/{n}.txt")
    g = b.feature_importance("gain"); s = b.feature_importance("split"); tot = g.sum()
    print(n, " ".join(f"{f}:{si}/{gi / tot:.3f}" for f, si, gi in sorted(zip(b.feature_name(), s, g), key=lambda x: -x[2])))
P
echo V3DONE
