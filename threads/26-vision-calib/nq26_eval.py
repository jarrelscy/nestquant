"""T26: harness-compatible held-out VISION eval data (the 33 val windows of the mm corpus, 200 held-out samples).

    import nq26_eval as V
    data = V.expert_data(L, E)                    # harness.ExpertData; harness.evaluate(data, {name: [Wg, Wu, Wd]})
    data = V.expert_data(L, E, rows="image")      # image-token rows only ("text" = caption/prompt rows only)

The frozen stage-1 capture (/tmp/nestquant/19-capture-mm/eval/val/layer_L.pt; FP8 GLM-5.3 reference forward, 5.2V
tower + projector features spliced at <|image|>) contains the window pad tails; those rows are DROPPED here
(rowkind 0).  Each document is split into an image part and a text part with domains "img:<domain>" and
"txt:<domain>", so harness.evaluate reports "all" (= the selected rows), the groups "img" / "txt", and per domain
(img:mm_medical, txt:mm_ocr, ...).  Teacher = FP8 GLM-5.3 source (nq19.SRC) via orbit_duet.source.weights, the
same as nq19_load.Capture.expert_data.  data.stats = the vision capture's pilot_stats (unused by evaluate).
"""
import json

import numpy as np
import torch

ROOT = "/tmp/nestquant/19-capture-mm"
_cache = {}


def _rowkind(cap):
    rk = np.load(f"{cap['protocol']['corpus']}/rowkind.npy")
    return torch.from_numpy(rk[cap["protocol"]["windows"]].reshape(-1).astype(np.int64))


def capture(L, rows="valid", root=ROOT):
    key = (L, rows, root)
    if key in _cache:
        return _cache[key]
    cap = torch.load(f"{root}/eval/val/layer_{L}.pt", weights_only=True, mmap=True)
    kind = _rowkind(cap)
    assert len(kind) == len(cap["x"])
    pad_docs = [i for i, d in enumerate(cap["domains"]) if d.endswith(":pad")]
    assert bool(torch.isin(cap["document_ids"], torch.tensor(pad_docs)).eq(kind == 0).all()), "rowkind / pad mismatch"
    keep = {"valid": kind > 0, "image": kind == 1, "text": kind == 2}[rows].nonzero().flatten()
    raw = cap["document_ids"][keep] * 2 + (kind[keep] == 1).long()     # 2d = text part, 2d+1 = image part
    used, doc = torch.unique(raw, return_inverse=True)                  # compact ids: no empty (pad) domains
    doms = [f"{'img' if u % 2 else 'txt'}:{cap['domains'][u // 2].split(':', 1)[1]}" for u in used.tolist()]
    out = dict(x=cap["x"][keep].contiguous(), ids=cap["ids"][keep], p=cap["p"][keep], document_ids=doc,
               token_positions=cap["token_positions"][keep], domains=doms, layer=cap["layer"],
               bnd_think=cap["bnd_think"][keep], bnd_end=cap["bnd_end"][keep],
               protocol=dict(cap["protocol"], t26_rows=rows, t26_note="pad rows dropped; docs split img/txt"))
    _cache.clear()
    _cache[key] = out
    return out


def expert_data(L, E, rows="valid", root=ROOT, source=None, stats_capture=None):
    import harness as h
    import nq19
    import nq19_load as C
    from orbit_duet.source import weights
    teacher = [w.to("cuda", torch.float32) for w in weights(source or nq19.SRC, L, E)]
    sc = stats_capture or C.Capture(root=root)
    return h.ExpertData(L, E, teacher, sc.pilot_stats(L, E), capture(L, rows, root),
                        f"{root}/eval/val/layer_{L}.pt#t26:{rows}", sc._open(L)[0])


if __name__ == "__main__":        # CPU self-check: row counts per kind / domain
    import sys
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    for r in ("valid", "image", "text"):
        c = capture(L, r)
        u, n = np.unique(c["document_ids"].numpy() % 2, return_counts=True)
        print(json.dumps(dict(L=L, rows=r, n=len(c["x"]), img=int(n[u == 1].sum()), txt=int(n[u == 0].sum()),
                              routed_E0=int((c["ids"] == 0).sum()))))
