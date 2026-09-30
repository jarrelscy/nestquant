"""SM120 serve model: per-session carry (NQ_SESSION_RESTORE) vs prefill lookahead set at decode start.
Raw logged SM120 ids (count-only: salience := routed counts, mps128 = 1), v2 predictor, hits-hot on decode rows.
Session restore restores predictor state + set per session, so each task (= one session key: the terminus-2 first user
message is constant within a task) is one continuous decode chain whatever the interleaving (2 agents alternating never
evict a 4-slot LRU).  Differences between arms only arise at request starts:
  switch request (key != previous request's): restore start downs every resident floating expert not in T and pins T;
    handover sets want := T and the next scheduler step downs the rest -> the serve already does (b) there.
  same-session request (no switch): no restore; lookahead (NQ_PREFILL_ADAPT=lookahead, prefill chunks >= NQ_PF_MIN 384
    tokens, 4096-token chunks) sets want := top-51 non-fixed by the chunk's routing (d=1 router of L+1 on x_L; here the
    ACTUAL routing of the layer = optimistic lookahead), the host scheduler converges to it, and it stays live until
    the predictor's first refresh (first 16-token decode block).
Arms (applied at the first decode block of each same-session request whose prefill had a >= 384-token chunk):
  b    carried set (restore to the pre-prefill session set)
  a    lookahead set (last >= 384-token chunk's top-51 by counts), hysteresis for block 1 on that resident set
  c    carried set, but LA experts inside the carried top-77 go first (then carried order) -> 51
  cB   carried scores with LA members x(1+hm) (hysteresis-style bonus), top-51
  sm120la.py MODEL [layer_step]"""
import glob, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scalelib as S
import sm120req as R0
MODEL = sys.argv[1]
LSTEP = int(sys.argv[2]) if len(sys.argv) > 2 else 6
ARMS = ["b", "a", "c", "cB"]
CH, PFMIN, NF, HM = 4096, 384, 51, 0.5

def replay_ovr(Sm, L, ovr, arm):
    fx = np.zeros(S.NE, bool); fx[S.FIXED[L]] = True
    want = np.zeros(S.NE, bool); want[[e for e in S.FDEF[L] if e not in set(S.FIXED[L])][:NF]] = True
    nb = Sm.shape[0]; serve = np.zeros((nb, S.NE), bool); lastv = None
    for k in range(nb):
        if k in ovr and arm != "b":
            la = ovr[k] & ~fx
            if arm == "a":
                want = la
            elif lastv is not None:
                v = lastv.copy()
                if arm == "c":
                    o = np.argsort(-v, kind="stable"); top77 = np.zeros(S.NE, bool); top77[o[:77]] = True
                    pri = np.where(la & top77, v + 1e30, v)
                else:
                    pri = np.where(la, v * np.float32(1 + HM), v)
                o = np.argsort(-pri, kind="stable"); want = np.zeros(S.NE, bool); want[o[:NF]] = True; want &= ~fx
        serve[k] = want
        Sk = Sm[k]
        v = np.where(fx, -np.inf, Sk).astype(np.float32)
        r = want & ~fx
        v = np.where(r, v * np.float32(1 + HM), v)
        lastv = np.where(fx, -np.inf, Sk).astype(np.float32)     # carried scores (no hysteresis) for c / cB
        if np.where(fx, 0, np.maximum(Sk, 0)).sum() <= 0:
            nw = r
        else:
            o = np.argsort(-v, kind="stable"); nw = np.zeros(S.NE, bool); nw[o[:NF]] = True
        want = nw & ~fx
    return serve

def job(args):
    L, fi = args
    import lightgbm as lgb
    b = lgb.Booster(model_file=MODEL)
    z = np.load(R0.TASKS[fi])
    reqs = R0.requests(z)
    if not reqs:
        return L, fi, None
    li = S.LAYERS.index(L)
    ex = z["ex"][:, li, :].astype(np.int64); tok = z["tok"]
    bcs, bcas, nans, segl, rs, ovr, npre = [], [], [], [], [], {}, []
    k0 = 0
    for ri, (pre, drow) in enumerate(reqs):
        nb = len(drow) // 16
        bi = np.repeat(np.arange(nb), 8 * 16)
        c = np.bincount(bi * S.NE + ex[drow].ravel(), minlength=nb * S.NE).reshape(nb, S.NE).astype(np.float64)
        s = np.zeros(len(drow), np.int8); cur = 0; nt = np.r_[tok[drow[1:]], -1]
        for i in range(len(drow)):
            if nt[i] == R0.THINK: cur = 0
            elif nt[i] == R0.ETHINK: cur = 1
            s[i] = cur
        sg = s.reshape(-1, 16); a = np.repeat(s.astype(bool), 8)
        ca = np.bincount((bi * S.NE + ex[drow].ravel())[a], minlength=nb * S.NE).reshape(nb, S.NE).astype(np.float64)
        bcs.append(c); bcas.append(ca); nans.append(sg.sum(1).astype(np.float64)); segl.append(sg[:, -1])
        r = np.zeros(nb, bool); r[0] = True; rs.append(r)
        chunks = [pre[i:i + CH] for i in range(0, len(pre), CH)]
        big = [ch for ch in chunks if len(ch) >= PFMIN]
        if ri > 0 and big:                            # same-session request with a lookahead-eligible prefill chunk
            cnt = np.bincount(ex[big[-1]].ravel(), minlength=S.NE).astype(np.float64)
            fx = np.zeros(S.NE, bool); fx[S.FIXED[L]] = True
            v = np.where(fx | (cnt <= 0), -np.inf, cnt); o = np.argsort(-v, kind="stable")
            la = np.zeros(S.NE, bool); la[[e for e in o[:NF] if v[e] > -np.inf]] = True
            ovr[k0] = la
        npre.append(np.full(nb, len(pre)))
        k0 += nb
    bc = np.concatenate(bcs)
    D = dict(bc=bc, bs=bc, bca=np.concatenate(bcas), nans=np.concatenate(nans), segl=np.concatenate(segl).astype(np.int8), sg=[(0, len(bc))])
    rs = np.concatenate(rs)
    F, _ = S.feats(D); F[..., 8] = 1.0
    Sm = S.predict_S(b, F, L)
    rp = np.concatenate([np.arange(len(x)) for x in bcs])
    has = np.zeros(len(bc), bool)
    for k in ovr:
        n = len(bcs[np.searchsorted(np.cumsum([len(x) for x in bcs]), k, side="right")])
        has[k:k + n] = True                           # all decode blocks of requests with a lookahead override
    res = {}
    for arm in ARMS:
        sv = replay_ovr(Sm, L, ovr, arm)
        M = S.block_metrics(sv, D, L)
        res[arm] = np.c_[M, rp, has, rs]
    return L, fi, res

if __name__ == "__main__":
    layers = S.LAYERS[::LSTEP]
    jobs = [(L, fi) for L in layers for fi in range(len(R0.TASKS))]
    R = {}
    with Pool(int(os.environ.get("NPROC", "6"))) as p:
        for L, fi, r in p.imap_unordered(job, jobs):
            if r is not None:
                R[(L, fi)] = r
    fis = sorted({fi for _, fi in R})
    np.savez(f"{S.OUT}/private/sm120la_raw.npz", **{f"{L}|{fi}|{a}": R[(L, fi)][a] for (L, fi) in R for a in ARMS})
    print("tasks", [os.path.basename(R0.TASKS[f])[:-4] for f in fis], "layers", layers)
    A0 = np.concatenate([R[(layers[0], fi)]["b"] for fi in fis])
    print(f"decode blocks {len(A0)}  requests {int(A0[:, 7].sum())}  in LA-override requests {A0[:, 6].mean():.3f}  "
          f"LA-override requests {int((A0[:, 7] * A0[:, 6]).sum())}")
    for scope, sel in (("all decode", lambda A: np.ones(len(A), bool)), ("LA requests", lambda A: A[:, 6] > 0)):
        print(scope)
        for arm in ARMS:
            line = []
            for lo, hi in [(0, 1), (0, 16), (16, 64), (64, 10 ** 9), (0, 10 ** 9)]:
                hh, cc = [], []
                for L in layers:
                    A = np.concatenate([R[(L, fi)][arm] for fi in fis if (L, fi) in R])
                    m = sel(A) & (A[:, 5] >= lo) & (A[:, 5] < hi)
                    hh.append(A[m, 2].sum() / A[m, 3].sum()); cc.append(np.nanmean(A[m, 4]))
                line.append(f"[{lo * 16},{min(hi * 16, 99999)}) {np.mean(hh) * 100:6.2f}/{np.nanmean(cc):5.2f}")
            print(f"  {arm:3s} " + "  ".join(line), flush=True)
