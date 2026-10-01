"""T35 ft pilot: jF floating-set hit rate on the held-out decode rows, per arm (CPU replay of saved routing).

For each routing file of nq35_ftp.py (OUT/routing/{arm}.pt: downstream layers' top-8 ids, weights, |x|^2, all rows
of every held-out window, plus the decode mask), replay the serving predictor exactly as quantisers.Adapt._core_gbdt
(joint arm, gmode=sync, hm, n_float, 16-token blocks, fresh predictor per window from floating_default) on THAT arm's
routing; hit = routed slot whose expert is served at level 4 (fixed | floating set) when the token runs.
Reported on decode rows: hit rate (all slots), routing-weight-weighted hit rate, churn; and, for the FP8 reference
routing replayed through the arm's predictor state, nothing else (each arm drives its own predictor, as in serving).
  python nq35_ftp_jf.py --dir /tmp/nestquant/35-nq15/ftp [--procs 8]
"""
import os, sys, json, glob, argparse
import numpy as np
import torch
from multiprocessing import Pool

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--joint", default="/tmp/nestquant/33-search/joint/models/jF_all.pt")
ap.add_argument("--manifest", default="/tmp/nestquant/32-gbdt-sal/k0_manifest.json")
ap.add_argument("--n-float", type=int, default=32); ap.add_argument("--hm", type=float, default=0.7)
ap.add_argument("--procs", type=int, default=8); ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--arms", default="")
ap.add_argument("--extra", default="", help="routing file of extra (upstream) layers prepended to every arm, e.g. routing_L40.pt")
ap.add_argument("--track", default="", help="L:e1,e2,.. report the served-at-L4 fraction of these experts' decode slots")
ap.add_argument("--tag", default="")
a = ap.parse_args()
G = 16
M = json.load(open(a.manifest))
FIX = {int(L): sorted(map(int, v)) for L, v in M["default_allocation"].items()}
FDEF = {int(L): [int(e) for e in v] for L, v in M["floating_default"].items()}


def one(args):
    f, n = args
    torch.set_num_threads(a.threads)
    sys.path.insert(0, "/home/coder/git/nestquant/threads/33-search/joint")
    from joint_predictor import JointPredictor
    Z = torch.load(f, weights_only=False)
    if a.extra:
        X = torch.load(a.extra, weights_only=False)
        assert torch.equal(X["dec"], Z["dec"])
        Z = dict(layers=list(X["layers"]) + list(Z["layers"]), dec=Z["dec"],
                 **{k: torch.cat([X[k].to(Z[k].dtype), Z[k]]) for k in ("ids", "w", "xn2")})
    layers = list(Z["layers"]); NLy = len(layers); NE = 256
    seq = Z["dec"].shape[1]
    ids = Z["ids"].long().numpy().reshape(NLy, -1, seq, 8)[:, n]           # [NL, seq, 8]
    w = Z["w"].float().numpy().reshape(NLy, -1, seq, 8)[:, n]
    xn = Z["xn2"].double().numpy().reshape(NLy, -1, seq)[:, n]
    dec = Z["dec"][n].numpy()
    fixed = np.zeros((NLy, NE), bool); fdef = np.zeros((NLy, NE), bool)
    for k, L in enumerate(layers):
        fixed[k, FIX[L]] = True
        fdef[k, [e for e in FDEF[L] if e not in set(FIX[L])][:a.n_float]] = True
    P = JointPredictor(layers, {L: FIX[L] for L in layers}, a.joint, n_float=a.n_float, hm=a.hm, device="cpu",
                       mode="sync", num_threads=a.threads)
    want = fdef.copy(); serve = np.zeros((seq // G, NLy, NE), bool); churn = []
    sal = (w.astype(np.float64) ** 2) * xn[:, :, None]
    try:
        for t in range(seq):
            if t % G == 0:
                serve[t // G] = want
            c = np.zeros((NLy, NE)); s = np.zeros((NLy, NE))
            for k in range(NLy):
                np.add.at(c[k], ids[k, t], 1.0); np.add.at(s[k], ids[k, t], sal[k, t])
            if P.step(c, 1, None, t == 0, sal=s):
                tg = P.target(want)
                if tg is not None:
                    nw = tg & ~fixed
                    churn.append(int((nw & ~want).sum())); want = nw
                assert t % G == G - 1
    finally:
        P.close()
    hi = serve | fixed[None]                                                # [nblk, NL, NE]
    blk = np.arange(seq) // G
    hit = np.take_along_axis(hi[blk].transpose(1, 0, 2), ids.reshape(NLy, seq, 8), 2)  # [NL, seq, 8]
    d = dec.astype(bool)
    tr = {}
    if a.track:
        tl, te = a.track.split(":"); k = layers.index(int(tl))
        for e in map(int, te.split(",")):
            m = (ids[k] == e) & d[:, None]
            tr[e] = (int(m.sum()), int(hit[k][m].sum()))
    return dict(track=tr, n=n, slots=int(d.sum()) * 8 * NLy, hits=int(hit[:, d].sum()),
                wsum=float(w[:, d].sum()), whit=float((w * hit)[:, d].sum()),
                churn=float(np.sum(churn)), nref=len(churn))


if __name__ == "__main__":
    files = sorted(glob.glob(f"{a.dir}/routing/*.pt"))
    if a.arms:
        files = [f for f in files if os.path.basename(f)[:-3] in a.arms.split(",")]
    out_f = f"{a.dir}/jf_hits{a.tag}.json"
    R = json.load(open(out_f)) if os.path.exists(out_f) else {}
    with Pool(a.procs) as pool:
        for f in files:
            arm = os.path.basename(f)[:-3]
            if arm in R:
                continue
            N = torch.load(f, weights_only=False)["dec"].shape[0]
            rs = pool.map(one, [(f, n) for n in range(N)])
            S = {k: sum(r[k] for r in rs) for k in ("slots", "hits", "wsum", "whit", "churn", "nref")}
            TRK = {}
            for r in rs:
                for e, (c_, h_) in r["track"].items():
                    TRK.setdefault(e, [0, 0]); TRK[e][0] += c_; TRK[e][1] += h_
            R[arm] = dict(track={e: dict(slots=c_, l4_frac=h_ / max(c_, 1)) for e, (c_, h_) in TRK.items()},hit=S["hits"] / S["slots"], whit=S["whit"] / S["wsum"], churn_per_refresh=S["churn"] / max(S["nref"], 1),
                          win_hit=[r["hits"] / max(r["slots"], 1) for r in rs],
                          **{f"n_{k}": v for k, v in S.items()})
            print(arm, json.dumps({k: v for k, v in R[arm].items() if k != "win_hit"}), flush=True)
            json.dump(R, open(out_f, "w"), indent=1)
