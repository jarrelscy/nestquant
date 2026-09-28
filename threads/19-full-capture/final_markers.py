"""Thread 19: per-layer FINAL markers for the encode (T25).  ROOT/final/L{L}.json is written atomically (tmp + rename)
once stats/L{L} links a version whose merged shard set == every plan shard; stage 2 publishes a version by atomic
symlink swap only after the version dir is complete, and a final layer gets no further merges, so the marker's
version dir is immutable.  ROOT/final/ALL_DONE when all 75 exist.

    final_markers.py --root /tmp/nestquant/19-capture-glmfmt [--loop]
"""
import argparse
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--loop", action="store_true")
    a = ap.parse_args()
    plan = json.load(open(f"{a.root}/plan.json"))
    need = {int(k) for k in plan["shards"]}
    tok = sum(s["fit_windows"] * (512 if s["corpus"].endswith("c512") else 2048) for s in plan["shards"].values())
    os.makedirs(f"{a.root}/final", exist_ok=True)
    while True:
        n = 0
        for L in range(3, 78):
            out = f"{a.root}/final/L{L}.json"
            if os.path.exists(out):
                n += 1
                continue
            vd = os.path.realpath(f"{a.root}/stats/L{L}")
            m = json.load(open(f"{vd}/meta.json"))
            got = {s["shard"] for s in m["shards"]}
            if got != need:
                continue
            d = dict(layer=L, version=os.path.basename(vd), version_dir=vd, link=f"{a.root}/stats/L{L}",
                     shards=sorted(got), fit_tokens=tok, n_routed_total=int(sum(m["n_routed"])),
                     final_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            with open(out + ".tmp", "w") as f:
                json.dump(d, f, indent=1)
            os.replace(out + ".tmp", out)
            print(json.dumps(dict(layer=L, version=d["version"], utc=d["final_utc"])), flush=True)
            n += 1
        if n == 75:
            with open(f"{a.root}/final/ALL_DONE.tmp", "w") as f:
                json.dump(dict(layers=75, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), f)
            os.replace(f"{a.root}/final/ALL_DONE.tmp", f"{a.root}/final/ALL_DONE")
            print("ALL_DONE", flush=True)
            return
        if not a.loop:
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
