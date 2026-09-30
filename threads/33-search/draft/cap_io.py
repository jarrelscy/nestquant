"""Load the draft capture (capture_draft.py, PRIVATE) in corpus window order (same as t32lib.load_layer)."""
import glob
import json

import numpy as np

CAP = "/tmp/nestquant/33-search/draft/private/cap"
SEQ = 2048


def _order(corpus):
    """-> list of (rank file suffix, slice in rank arrays) in corpus window order."""
    parts = {}
    for f in sorted(glob.glob(f"{CAP}/windows.r*of*.json")):
        j = json.load(open(f))
        r, W = j["rank"], j["world"]
        off = 0
        for name, wins in j["windows"]:
            if name == corpus:
                for k, wi in enumerate(wins):
                    parts[wi] = (f"r{r}of{W}", slice(off + k * SEQ, off + (k + 1) * SEQ))
            off += len(wins) * SEQ
    ks = sorted(parts)
    assert ks == list(range(len(ks)))
    return [parts[k] for k in ks]


def load(corpus, name, keys, mmap=False):
    """name = 'L40' | 'head' | 'mtp1' ... -> dict key -> concatenated array in window order"""
    order = _order(corpus)
    cache = {}
    out = {k: [] for k in keys}
    for sfx, s in order:
        if sfx not in cache:
            cache[sfx] = np.load(f"{CAP}/{name}.{sfx}.npz", mmap_mode="r" if mmap else None)
        for k in keys:
            out[k].append(np.asarray(cache[sfx][k][s]))
    return {k: np.concatenate(v) for k, v in out.items()}
