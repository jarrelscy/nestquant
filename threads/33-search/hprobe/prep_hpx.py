#!/usr/bin/env python3
"""probe-training corpus hpx (PRIVATE; coordinator-approved: no nq-tail / wikitext / vllm-docs):
  [github 24 windows] + [c2048 fit windows not in calib-fit, 256 seeded] + [c2048_traces fit windows with no
  trace:tb21-* segment (TB2.1 = SM120 task family, excluded to keep sm120tf clean), up to 160]
Each part is a multiple of 4 windows so 4-window chains never straddle parts."""
import json
import numpy as np

CD = "/tmp/nestquant/corpus/glm53_calib_glmfmt_v1"
OUT = "/tmp/nestquant/33-search/hprobe/private/corpora"
man = json.load(open("/tmp/nestquant/18-e2e/corpora/manifest.json"))
calib_rows = set(man["calib-fit"]["rows"])
c2k = np.load(f"{CD}/c2048/tokens.npy", mmap_mode="r")
cf = np.load("/tmp/nestquant/18-e2e/corpora/calib-fit.npy")
assert np.array_equal(np.asarray(c2k[sorted(calib_rows)]).ravel(), cf), "calib-fit rows mismatch"
gh = np.load("/tmp/nestquant/18-e2e/corpora/github.npy")[: 24 * 2048]
rng = np.random.default_rng(33)
pool = [r for r in range(2513) if r not in calib_rows]
rows2k = np.sort(rng.choice(pool, 256, replace=False))
tr = np.load(f"{CD}/c2048_traces/tokens.npy", mmap_mode="r")
ok = []
for i, line in enumerate(open(f"{CD}/c2048_traces/windows.jsonl")):
    if i >= 761:
        break
    segs = json.loads(line)["segments"]
    if not any(s["category"].startswith("trace:tb21") for s in segs):
        ok.append(i)
ok = ok[: min(160, len(ok) // 4 * 4)]
parts = {"github": gh, "c2048x": np.asarray(c2k[rows2k]).ravel(), "traces_notb": np.asarray(tr[ok]).ravel()}
ids = np.concatenate(list(parts.values())).astype(np.int32)
np.save(f"{OUT}/hpx.npy", ids)
meta = {k: len(v) // 2048 for k, v in parts.items()}
cats = {}
for i, line in enumerate(open(f"{CD}/c2048_traces/windows.jsonl")):
    if i in set(ok):
        for s in json.loads(line)["segments"]:
            cats[s["category"]] = cats.get(s["category"], 0) + s["tokens"]
json.dump(dict(parts_windows=meta, c2048_rows=rows2k.tolist(), traces_windows=ok, traces_categories=cats),
          open(f"{OUT}/hpx.meta.json", "w"), indent=1)
print(meta, len(ids) // 2048, cats)
