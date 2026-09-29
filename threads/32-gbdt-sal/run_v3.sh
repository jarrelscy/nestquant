#!/bin/bash
# T32 (a): after recapture.sh releases the GPUs: v3 = 5 count + 4 salience (v2) + 8 router-prob features, salience
# target (Tweedie 1.5).  Held-out sim vs gbdt_old in the default mode (band EMA256 20-120, next_refresh) and in
# sync + band all (every expert scored, nothing forced; model trained on band-all rows).
set -euo pipefail
O32=/tmp/nestquant/32-gbdt-sal; M=$O32/models; OLD=/home/coder/git/nestquant/streaming/gbdt_p64_s5.txt
export PYTHONPATH=/tmp/nestquant/18-e2e/pylib; PY=/home/coder/git/glm52/.venv/bin/python
cd "$(dirname "$0")"
until grep -q "release GPUs" $O32/logs/recapture.out 2>/dev/null; do sleep 20; done
for c in glm52-heldout calib-fit; do $PY build_v3.py $c 12; done
$PY train.py --threads 32 --v2 --v3 --target sal --obj tweedie:1.5 --out $M/v3_sal_tweedie1.5.txt
T32_LAG=1 T32_TAG=_v3_default NPROC=12 $PY sim.py glm52-heldout gbdt_old=$OLD v3_sal=$M/v3_sal_tweedie1.5.txt
echo DEFAULT_DONE
until grep -q "^77$" $O32/logs/band_all_calib.log 2>/dev/null; do sleep 20; done
for c in glm52-heldout calib-fit; do T32_BAND=all $PY build_v3.py $c 12; done
$PY train.py --threads 32 --band all --v2 --v3 --target sal --obj tweedie:1.5 --out $M/ba_v3_sal_tweedie1.5.txt
T32_BAND=all T32_LAG=0 T32_TAG=_v3_bandall_sync NPROC=12 $PY sim.py glm52-heldout gbdt_old=$OLD \
  v3_sal=$M/v3_sal_tweedie1.5.txt ba_v3_sal=$M/ba_v3_sal_tweedie1.5.txt
$PY - <<'P'
import lightgbm as lgb
for n in ("v3_sal_tweedie1.5", "ba_v3_sal_tweedie1.5"):
    b = lgb.Booster(model_file=f"/tmp/nestquant/32-gbdt-sal/models/{n}.txt")
    g = b.feature_importance("gain"); s = b.feature_importance("split"); tot = g.sum()
    print(n, " ".join(f"{f}:{si}/{gi / tot:.3f}" for f, si, gi in sorted(zip(b.feature_name(), s, g), key=lambda x: -x[2])))
P
echo V3DONE
