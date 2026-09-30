"""numpy-only helpers (no t32lib/lightgbm): serve sets, m_L, chains, replay (= t32lib.sim_layer lag 0), metrics."""
import json
import numpy as np

WD = "/tmp/nestquant/33-search/seq"
NE, G = 256, 16
LAYERS = list(range(3, 78))
import os
LAYOUT = os.environ.get("NQ_LAYOUT", "k0")
_m = json.load(open("/tmp/nestquant/32-gbdt-sal/k0_manifest.json" if LAYOUT == "k0" else
                    "/tmp/nestquant/28-serve-release/out/serving/tp4/manifest.json"))
NF = 77 if LAYOUT == "k0" else 51
FIXED = {int(L): sorted(map(int, v)) for L, v in _m["default_allocation"].items()}
FDEF = {int(L): [int(e) for e in v] for L, v in _m["floating_default"].items()}
ML = {int(k): v for k, v in json.load(open("/tmp/nestquant/32-gbdt-sal/models/v2_sal_tweedie1.5.txt.meta.json"))["sal_norm_mL"].items()}
SMMETA = "/tmp/nestquant/32-gbdt-sal/private/sm120/blk/sm120tf/meta.json"


def segs_of(corpus, nb):
    if corpus == "sm120tf":
        bs = json.load(open(SMMETA))["bstart"]
        return [(s, e) for s, e in zip(bs[:-1], bs[1:]) if e > s]
    return [(s, min(s + 512, nb)) for s in range(0, nb, 512)]


def fxmask(L):
    fx = np.zeros(NE, bool); fx[FIXED[L]] = True
    return fx


def replay(S, L, sg, hm=0.5, nf=None):
    nf = nf or NF
    fx = fxmask(L)
    fd = np.zeros(NE, bool); fd[[e for e in FDEF[L] if e not in set(FIXED[L])][:nf]] = True
    nb = S.shape[0]
    serve = np.zeros((nb, NE), bool)
    S = S.astype(np.float32)
    Sm = np.where(fx[None], -np.inf, S)
    pos = np.where(fx[None], 0, np.maximum(S, 0)).sum(1) > 0
    f1 = np.float32(1 + hm)
    for s, e in sg:
        want = fd.copy()
        for k in range(s, e):
            serve[k] = want
            if not pos[k]:
                continue
            v = np.where(want, Sm[k] * f1, Sm[k])
            nw = np.zeros(NE, bool); nw[np.argsort(-v, kind="stable")[:nf]] = True
            want = nw & ~fx
    return serve


def metrics(S, L, bcnt, bsal, sg, hm=0.5):
    sv = replay(S, L, sg, hm)
    ch = (sv[1:] & ~sv[:-1]).sum(1).astype(np.float64)
    sv[:, FIXED[L]] = True
    return dict(sal=float((bsal * sv).sum() / bsal.sum()), cnt=float((bcnt * sv).sum() / bcnt.sum()),
                churn=float(ch.mean()))
