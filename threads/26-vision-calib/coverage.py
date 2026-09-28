"""T26: per-layer vision coverage (routed rows per expert, scalars[:,0]) + vision/text salience agreement."""
import json, os, sys
import numpy as np
from scipy.stats import spearmanr
R = sys.argv[1] if len(sys.argv) > 1 else "/tmp/nestquant/19-capture-mm"
TXT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/nestquant/19-capture-glmfmt"
out = []
for L in range(3, 78):
    d = f"{R}/stats/L{L}"
    if not os.path.exists(f"{d}/scalars.npy"):
        continue
    n = np.load(f"{d}/scalars.npy")[:, 0]
    sv = np.load(f"{d}/sal.npy")[:, 0, 4]
    r = dict(L=L, lt128=int((n < 128).sum()), lt1000=int((n < 1000).sum()), zero=int((n == 0).sum()),
             median=float(np.median(n)), top10_share=float(np.sort(n)[::-1][:10].sum() / n.sum()))
    t = f"{TXT}/stats/L{L}/sal.npy"
    if os.path.exists(t):
        st = np.load(t)[:, 0, 4]
        r["reap_spearman_vs_text"] = float(spearmanr(sv, st).correlation)
    out.append(r)
    print(json.dumps(r), flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "coverage.json"), "w"), indent=0)
