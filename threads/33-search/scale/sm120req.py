"""SM120 request-structure test of prefill warm-start (raw logged SM120 ids, count-only: salience := routed counts,
metric = all-slot hits-hot on decode rows).  Per task: request runs = maximal row runs of one req id; prefill rows =
non-decode rows before the run's first decode row; decode rows grouped in 16-row blocks (tail < 16 dropped).
Arms:
  cold          state reset at every request, floating_default, prefill ignored
  cold+pmean    reset, then 16 synthetic blocks of the request's mean per-block prefill counts (prefill(sums) hook)
  cold+pexact   reset, then the prefill rows' last <=1024 rows fed as real 16-row blocks
  carry         state persists across the task's requests, prefill ignored (session carry-over / serve today w/o reset)
  carry+pmean   carry, plus min(16, ceil(npre/16)) synthetic mean prefill blocks at each request start
  carry+pexact  carry, plus the prefill rows (last <=1024) as real blocks
  sm120req.py MODEL [layer_step]  -> private/sm120req_raw.npz + table"""
import glob, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scalelib as S
MODEL = sys.argv[1]
LSTEP = int(sys.argv[2]) if len(sys.argv) > 2 else 3
SRC = f"{S.SRC}/sm120_traces/serving/predictor/traces"
THINK, ETHINK = 154841, 154842
TASKS = [f for f in sorted(glob.glob(f"{SRC}/*.npz")) if not os.path.basename(f).startswith("probe-")]
ARMS = ["cold", "cold+pmean", "cold+pexact", "carry", "carry+pmean", "carry+pexact"]
PMAX = 1024

def requests(z):
    req, dec, tok = z["req"], z["dec"], z["tok"]
    b = np.flatnonzero(np.r_[True, req[1:] != req[:-1], True])
    out = []
    for s, e in zip(b[:-1], b[1:]):
        d = np.flatnonzero(dec[s:e])
        if len(d) < 16:
            continue
        d0 = s + d[0]
        pre = np.arange(s, d0)
        drow = s + d
        drow = drow[: len(drow) // 16 * 16]
        out.append((pre, drow))
        if sum(len(o[1]) for o in out) >= int(os.environ.get("DCAP", "200000")):
            break
    return out

def seg_of_rows(tok_next_known):
    return None

def job(args):
    L, fi = args
    import lightgbm as lgb
    b = lgb.Booster(model_file=MODEL)
    z = np.load(TASKS[fi])
    reqs = requests(z)
    if not reqs:
        return L, fi, None
    li = S.LAYERS.index(L)
    ex = z["ex"][:, li, :].astype(np.int64)
    tok = z["tok"]
    def blk_counts(rows):
        n = len(rows) // 16 * 16
        rows = rows[len(rows) - n:]
        nb = n // 16
        bi = np.repeat(np.arange(nb), 8 * 16)
        c = np.bincount(bi * S.NE + ex[rows].ravel(), minlength=nb * S.NE).reshape(nb, S.NE).astype(np.float64)
        return c
    def seg_rows(rows, start_think=True):
        # answer flag per row: state after emitted token tok[t+1]; think at request start
        s = np.zeros(len(rows), np.int8); cur = 0
        nt = np.r_[tok[rows[1:]], -1]
        for i in range(len(rows)):
            if nt[i] == THINK: cur = 0
            elif nt[i] == ETHINK: cur = 1
            s[i] = cur
        return s
    res = {}
    for arm in ARMS:
        carry = arm.startswith("carry")
        parts = []                                   # list of (bc, bca, nans, segl, is_eval, is_req_start)
        for pre, drow in reqs:
            if arm.endswith("pexact") and len(pre) >= 16:
                pr = pre[-PMAX:]
                c = blk_counts(pr)
                parts.append((c, np.zeros_like(c), np.zeros(len(c)), np.zeros(len(c), np.int8), np.zeros(len(c), bool), np.zeros(len(c), bool)))
            elif arm.endswith("pmean") and len(pre) > 0:
                nb = 16 if not carry else int(min(16, np.ceil(len(pre) / 16)))
                m = np.bincount(ex[pre].ravel(), minlength=S.NE) / len(pre) * 16
                c = np.repeat(m[None], nb, 0)
                parts.append((c, np.zeros_like(c), np.zeros(nb), np.zeros(nb, np.int8), np.zeros(nb, bool), np.zeros(nb, bool)))
            c = blk_counts(drow)
            sg = seg_rows(drow).reshape(-1, 16)
            ca = np.zeros_like(c)
            nb = len(c)
            bi = np.repeat(np.arange(nb), 8 * 16)
            a = np.repeat(sg.ravel().astype(bool), 8)
            ca = np.bincount((bi * S.NE + ex[drow].ravel())[a], minlength=nb * S.NE).reshape(nb, S.NE).astype(np.float64)
            rs = np.zeros(nb, bool); rs[0] = True
            parts.append((c, ca, sg.sum(1).astype(np.float64), sg[:, -1], np.ones(nb, bool), rs))
            if not carry:
                parts.append(None)                   # chain break
        # assemble chains
        chains, cur = [], []
        for p in parts:
            if p is None:
                chains.append(cur); cur = []
            else:
                cur.append(p)
        if cur:
            chains.append(cur)
        cat = lambda i: np.concatenate([p[i] for ch in chains for p in ch])
        bc = cat(0)
        D = dict(bc=bc, bs=bc, bca=cat(1), nans=cat(2), segl=cat(3).astype(np.int8))
        lens = np.cumsum([0] + [sum(len(p[0]) for p in ch) for ch in chains])
        D["sg"] = list(zip(lens[:-1], lens[1:]))
        ev, rs = cat(4), cat(5)
        F, _ = S.feats(D)
        F[..., 8] = 1.0                                  # count-only: mps128 = 1 (salience := counts)
        sv = S.replay(S.predict_S(b, F, L), L, D["sg"])
        M = S.block_metrics(sv, D, L)
        M[rs, 4] = np.nan
        M = M[ev]
        rp = np.concatenate([np.arange(len(p[0])) for ch in chains for p in ch if p[4][0]])
        res[arm] = np.c_[M, rp]
    return L, fi, res

if __name__ == "__main__":
    layers = S.LAYERS[::LSTEP]
    jobs = [(L, fi) for L in layers for fi in range(len(TASKS))]
    R = {}
    with Pool(int(os.environ.get("NPROC", "8"))) as p:
        for L, fi, r in p.imap_unordered(job, jobs):
            if r is not None:
                R[(L, fi)] = r
    fis = sorted({fi for _, fi in R})
    np.savez(f"{S.OUT}/private/sm120req_raw.npz", **{f"{L}|{fi}|{a}": R[(L, fi)][a] for (L, fi) in R for a in ARMS})
    print("tasks", [os.path.basename(TASKS[f])[:-4] for f in fis], "layers", layers)
    buckets = [(0, 16), (16, 64), (64, 10 ** 9), (0, 10 ** 9)]
    for arm in ARMS:
        line = []
        for lo, hi in buckets:
            hh, cc = [], []
            for L in layers:
                A = np.concatenate([R[(L, fi)][arm] for fi in fis if (L, fi) in R])
                m = (A[:, 5] >= lo) & (A[:, 5] < hi)
                hh.append(A[m, 2].sum() / A[m, 3].sum()); cc.append(np.nanmean(A[m, 4]))
            line.append(f"dec[{lo * 16},{min(hi * 16, 99999)}) {np.mean(hh) * 100:6.2f}/{np.nanmean(cc):4.2f}")
        print(f"  {arm:13s} " + "  ".join(line), flush=True)
    A = np.concatenate([R[(layers[0], fi)]["cold"] for fi in fis])
    print("decode-token share by bucket:", [round(float(((A[:, 5] >= lo) & (A[:, 5] < hi)).mean()), 3) for lo, hi in buckets])
