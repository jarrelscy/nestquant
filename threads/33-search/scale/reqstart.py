"""request-start prior test.  Each full 8192-token chain is split: prompt = blocks [0,P) (prefill, routing known after
prefill), decode = blocks [P, 512).  Arms (all restart the predictor at the decode start):
  cold        state reset, floating_default (what the serve does today: prefill chunks are ignored)
  warm        prefill routing fed to the predictor as 16-token blocks (= continuous chain)
  warmmean    16 synthetic blocks of the prompt's MEAN per-block counts / salience (needs only per-expert prompt sums)
  priorset    cold features, initial floating set = top-51 non-fixed by prompt salience sum (last 256 prompt tokens)
  priorcnt    same, by prompt routed counts
Metric: all-slot sal-hot on decode tokens [0,256), [0,1024) and all decode, churn.
  reqstart.py STREAM MODEL [P] [chains]"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scalelib as S
stream, MODEL = sys.argv[1], sys.argv[2]
P = int(sys.argv[3]) if len(sys.argv) > 3 else 32
CH = [int(c) for c in sys.argv[4].split(",")] if len(sys.argv) > 4 else None
KEYS = ("bc", "bs", "bca", "nans", "segl")

def sub(D, parts):
    """parts: list of chains, each a list of (dict-of-arrays) to concatenate -> new D"""
    out = {k: [] for k in KEYS}; sg = []; n = 0
    for chain in parts:
        s = n
        for blk in chain:
            for k in KEYS:
                out[k].append(blk[k])
            n += len(blk["nans"])
        sg.append((s, n))
    D2 = {k: np.concatenate(v) for k, v in out.items()}; D2["sg"] = sg
    return D2

def job(L):
    import lightgbm as lgb
    b = lgb.Booster(model_file=MODEL)
    D = S.load(stream, L)
    fx = np.zeros(S.NE, bool); fx[S.FIXED[L]] = True
    chains = [(s, e) for i, (s, e) in enumerate(D["sg"]) if e - s == S.NBC and (CH is None or i in CH)]
    sl = lambda s, e: {k: D[k][s:e] for k in KEYS}
    dec = [sl(s + P, e) for s, e in chains]
    nd = S.NBC - P
    res = {}
    def run(D2, init=None, skip=0):
        F, _ = S.feats(D2)
        sv = S.replay(S.predict_S(b, F, L), L, D2["sg"], init=init)
        M = S.block_metrics(sv, D2, L)
        keep = np.concatenate([np.arange(s + skip, e) for s, e in D2["sg"]])
        M = M[keep]
        M[np.arange(0, len(M), nd), 4] = np.nan               # churn at decode start not a refresh-to-refresh count
        return M
    res["cold"] = run(sub(D, [[d] for d in dec]))
    res["warm"] = run(sub(D, [[sl(s, s + P), d] for (s, e), d in zip(chains, dec)]), skip=P)
    syn = []
    for s, e in chains:
        m = {k: np.repeat(D[k][s:s + P].mean(0, keepdims=True), 16, 0) for k in KEYS}
        m["segl"] = np.repeat(D["segl"][s + P - 1:s + P], 16)
        syn.append(m)
    res["warmmean"] = run(sub(D, [[m, d] for m, d in zip(syn, dec)]), skip=16)
    for nm, key in (("priorset", "bs"), ("priorcnt", "bc")):
        init = []
        for s, e in chains:
            v = np.where(fx, -np.inf, D[key][s + P - 16:s + P].sum(0))
            m = np.zeros(S.NE, bool); m[np.argsort(-v, kind="stable")[:51]] = True
            init.append(m)
        res[nm] = run(sub(D, [[d] for d in dec]), init=np.array(init))
    return L, res

if __name__ == "__main__":
    with Pool(int(os.environ.get("NPROC", "8"))) as p:
        R = dict(p.map(job, S.LAYERS))
    nd = S.NBC - P
    print(f"[{stream}] P={P * 16} prompt tokens, model {os.path.basename(MODEL)}")
    for arm in R[S.LAYERS[0]]:
        out = []
        for lo, hi in ((0, 16), (0, 64), (0, nd)):
            M = {L: R[L][arm] for L in S.LAYERS}
            mask = (np.arange(len(M[S.LAYERS[0]])) % nd >= lo) & (np.arange(len(M[S.LAYERS[0]])) % nd < hi)
            s = S.summarize(M, mask)
            out.append(f"dec[{lo * 16},{hi * 16}) {s['sal']:6.2f}/{s['churn']:4.2f}")
        print(f"  {arm:10s} " + "   ".join(out), flush=True)
