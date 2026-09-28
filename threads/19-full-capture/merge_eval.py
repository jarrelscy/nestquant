"""Thread 19: combine the held-out val captures written by several stage-1 shards (one per corpus group) into
ROOT/eval/val/layer_L.pt (harness capture format; rows concatenated, document_ids renumbered, domains appended,
bnd_think / bnd_end concatenated), and link ROOT/eval/matched to the shard that wrote the matched set.

    merge_eval.py --root R --shards 0,4 [--layers 3-77]
"""
import argparse
import json
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--layers", default="3-77")
    a = ap.parse_args()
    lo, hi = [int(v) for v in a.layers.split("-")]
    shards = [int(v) for v in a.shards.split(",")]
    os.makedirs(f"{a.root}/eval/val", exist_ok=True)
    for k in shards:
        md = f"{a.root}/shards/s{k:02d}/eval/matched"
        if os.path.exists(f"{md}/layer_{lo}.pt") and not os.path.lexists(f"{a.root}/eval/matched"):
            os.symlink(os.path.relpath(md, f"{a.root}/eval"), f"{a.root}/eval/matched")
    for L in range(lo, hi + 1):
        out = f"{a.root}/eval/val/layer_{L}.pt"
        if os.path.exists(out):
            continue
        parts = [f"{a.root}/shards/s{k:02d}/eval/val/layer_{L}.pt" for k in shards]
        parts = [p for p in parts if os.path.exists(p)]
        if not parts:
            continue
        ds = [torch.load(p, weights_only=True, mmap=True) for p in parts]
        m = dict(layer=L, domains=[], protocol=dict(role="evaluation only", parts=[d["protocol"] for d in ds],
                                                   shards=shards))
        doc = []
        base = 0
        for d in ds:
            doc.append(d["document_ids"] + base); base += len(d["domains"]); m["domains"] += list(d["domains"])
        m["document_ids"] = torch.cat(doc)
        for k in ("x", "ids", "p", "token_positions", "bnd_think", "bnd_end"):
            if all(k in d for d in ds):
                m[k] = torch.cat([d[k] for d in ds])
        torch.save(m, out + ".tmp"); os.replace(out + ".tmp", out)
        print(json.dumps(dict(layer=L, rows=int(m["x"].shape[0]), parts=len(parts))), flush=True)


if __name__ == "__main__":
    main()
