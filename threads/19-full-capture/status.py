"""Progress of the progressive capture: stage-1 layer per shard, cumulative shard count per layer, ESS, disk."""
import glob, json, os, sys
import numpy as np
ROOT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/nestquant/19-capture"
s1 = {}
for d in sorted(glob.glob(f"{ROOT}/shards/s*")):
    try:
        p = json.load(open(f"{d}/state/progress.json")); s1[os.path.basename(d)] = max(int(k) for k in p["layers"])
    except Exception:
        s1[os.path.basename(d)] = None
cnt = {}
for L in range(3, 78):
    try:
        m = json.load(open(f"{ROOT}/stats/L{L}/meta.json")); cnt[L] = len(m["shards"])
    except Exception:
        cnt[L] = 0
s0 = sum(os.path.exists(f"{ROOT}/stats0/L{L}/meta.json") for L in range(3, 78))
st = os.statvfs(ROOT)
out = dict(stage1_last_layer=s1, stats_layers_by_shards={k: sum(v == k for v in cnt.values()) for k in sorted(set(cnt.values()))},
           min_shards=min(cnt.values()), stats0_layers=s0, free_tb=round(st.f_bavail * st.f_frsize / 2**40, 2))
if "--ess" in sys.argv:
    for L in (16, 49, 66):
        try:
            m = json.load(open(f"{ROOT}/stats/L{L}/meta.json")); e = np.array(m["ess"])
            out[f"L{L}"] = dict(shards=len(m["shards"]), min=round(e.min()), p5=round(np.percentile(e, 5)), median=round(np.median(e)))
        except Exception:
            pass
print(json.dumps(out))
