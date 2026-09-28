"""T26 stage 2 for the vision capture: T19's capture_stats.py run UNCHANGED except that each stage-1 shard is
restricted to the real rows of the mm corpus before any statistic is computed (PAD tails dropped; optionally
image rows only).  Output = the identical nestquant-19-stats-v2 layout (stats/L{L} -> L{L}.vN, sal.npy, meta.json)
under a separate root, so nq19_load.Capture(root=...) reads it as is.

Row selection (env NQ26_ROWS): "valid" (default: image + text rows of the mm samples, like glm52 capture_mm53.py,
i.e. v3's mm Hessians) or "image" (the 256 image-token rows + begin/end-of-image only).  After the selection the
shard is re-indexed exactly as load_shard would have (argsort by expert, ctx = randperm(T', seed 20260925 + k)
[:T' // 4]), so ctx rows, n_dc, dc_scale follow T19's rules on the selected rows.

    run.sh-style env;  capture_mm_stats.py --root /tmp/nestquant/19-capture-mm [capture_stats.py args]
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "19-full-capture"))
import nq19            # noqa: E402
import capture_stats as cs  # noqa: E402

ROWS = os.environ.get("NQ26_ROWS", "valid")
_orig_load = cs.load_shard
_kinds = {}


def rowkind(corpus):
    if corpus not in _kinds:
        _kinds[corpus] = np.load(f"{corpus}/rowkind.npy", mmap_mode="r")
    return _kinds[corpus]


def load_shard(acts, max_rows=None):
    S_ = _orig_load(acts, max_rows)
    m = S_["meta"]
    K = np.asarray(rowkind(m["corpus"])[m["fit_start"]:m["fit_start"] + m["fit_windows"]]).reshape(-1)[:m["T"]]
    keep_mask = (K > 0) if ROWS == "valid" else (K == 1)
    keep = torch.from_numpy(np.nonzero(keep_mask)[0])
    X, ids, p = S_["X"][keep].contiguous(), S_["ids"][keep].contiguous(), S_["p"][keep].contiguous()
    del S_
    T = len(keep)
    k = m["shard"]
    flat = ids.reshape(-1)
    order = torch.argsort(flat, stable=True)
    cnt = torch.bincount(flat, minlength=nq19.NEXP)
    n_ctx = T // nq19.CTX_FRACTION
    ctx = torch.randperm(T, generator=torch.Generator().manual_seed(nq19.CTX_SEED + k))[:n_ctx].clone()
    n_dc = min(n_ctx, cs.NDC)
    m.update(T=T, n_ctx=n_ctx, n_dc=n_dc, dc_scale=n_ctx / n_dc, rows_selected=ROWS,
             T_stage1=int(len(K)), rows_image=int((K[keep_mask] == 1).sum()), rows_text=int((K[keep_mask] == 2).sum()))
    return dict(meta=m, X=X, rows_all=order // 8, p_all=p.reshape(-1)[order], offs=[0] + cnt.cumsum(0).tolist(), ctx=ctx,
                ids=ids, p=p)


cs.load_shard = load_shard

if __name__ == "__main__":
    cs.main()
