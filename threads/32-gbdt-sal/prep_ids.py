#!/usr/bin/env python3
"""T32 prep: arbitrary token-id input -> packed private corpus for capture.sh (T18 nq_e2e ref forward, teacher-forced).
  prep_ids.py NAME SRC [SRC ...] [--key token_ids]   -> /tmp/nestquant/32-gbdt-sal/corpora/NAME.npy (+ NAME.docs.npy)
SRC: .npy (1-D stream, or 2-D rows), .npz (every array, or --key), .jsonl/.json (per line/record a list of ids, or a
dict holding --key / token_ids / ids / input_ids [+ output_ids appended]), .parquet (column --key), or a directory of
these (sorted).  Documents (requests) are concatenated in order; NAME.docs.npy holds each doc's start offset so the
offline sim can reset state at request boundaries.  capture then takes len // 2048 windows (NQ_SHARD=contig).
PRIVATE: outputs stay in /tmp/nestquant/32-gbdt-sal (never committed / uploaded); the manifest only has counts+sha."""
import argparse
import glob
import hashlib
import json
import os

import numpy as np

OUT = "/tmp/nestquant/32-gbdt-sal/corpora"
KEYS = ("token_ids", "ids", "input_ids", "prompt_token_ids", "tokens")


def from_obj(o, key):
    if isinstance(o, dict):
        for k in ([key] if key else []) + list(KEYS):
            if k in o:
                v = list(o[k])
                for extra in ("output_ids", "output_token_ids", "completion_ids"):
                    if extra in o and k != extra:
                        v += list(o[extra])
                return [v]
        raise KeyError(f"no token key in record keys {list(o)[:10]}")
    if isinstance(o, list) and o and isinstance(o[0], list):
        return o
    return [o]


def docs_of(path, key):
    if os.path.isdir(path):
        for p in sorted(glob.glob(f"{path}/**/*", recursive=True)):
            if os.path.isfile(p) and p.rsplit(".", 1)[-1] in ("npy", "npz", "jsonl", "json", "parquet"):
                yield from docs_of(p, key)
        return
    ext = path.rsplit(".", 1)[-1]
    if ext == "npy":
        a = np.load(path, mmap_mode="r")
        yield from ([np.asarray(a)] if a.ndim == 1 else [np.asarray(r) for r in a])
    elif ext == "npz":
        z = np.load(path)
        for k in ([key] if key else z.files):
            a = z[k]
            yield from ([a] if a.ndim == 1 else list(a))
    elif ext == "jsonl":
        for line in open(path):
            if line.strip():
                yield from from_obj(json.loads(line), key)
    elif ext == "json":
        o = json.load(open(path))
        for r in (o if isinstance(o, list) and o and not isinstance(o[0], int) else [o]):
            yield from from_obj(r, key)
    elif ext == "parquet":
        import pyarrow.parquet as pq
        col = pq.read_table(path).column(key or "token_ids").to_pylist()
        yield from col
    else:
        raise ValueError(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("src", nargs="+")
    ap.add_argument("--key", default=None)
    a = ap.parse_args()
    parts, starts, n = [], [], 0
    for s in a.src:
        for d in docs_of(s, a.key):
            d = np.asarray(d, dtype=np.int64).ravel()
            if not len(d):
                continue
            assert d.min() >= 0 and d.max() < 2 ** 31, (d.min(), d.max())
            starts.append(n); parts.append(d.astype(np.int32)); n += len(d)
    ids = np.concatenate(parts)
    os.makedirs(OUT, exist_ok=True)
    np.save(f"{OUT}/{a.name}.npy", ids)
    np.save(f"{OUT}/{a.name}.docs.npy", np.asarray(starts, np.int64))
    mp = f"{OUT}/manifest.json"
    man = json.load(open(mp)) if os.path.exists(mp) else {}
    man[a.name] = dict(tokens=int(n), docs=len(starts), windows=int(n // 2048), max_id=int(ids.max()),
                       sha256=hashlib.sha256(ids.tobytes()).hexdigest())
    json.dump(man, open(mp, "w"), indent=1)
    print(a.name, man[a.name])


if __name__ == "__main__":
    main()
