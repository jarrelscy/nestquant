"""T35 ft pilot helper (CPU): serving hot set of layer L for every window of the nq35_ftp.py cache (train + held-out).
The jF predictor is per layer (L-only replay == full-depth replay, checked on fp8dec-heldout), so the b1.75 serving
state of layer L is replayed from layer L's own routing: k0 manifest (no fixed set), n_float floating slots, hm,
gmode=sync, 16-token blocks, fresh predictor per window from floating_default (= nq35_ftp_jf.py).
-> {cache}/L{L}_s{seq}_tr{n}_ho{n}_hot_nf{n_float}_hm{hm}.pt  hot [Nwin, seq//16, 256] bool (level-4 served set per
block; token t uses block t//16), plus the CPU routing (ids, w, xn2) and per-split stats.
  PYTHONPATH=/tmp/nestquant/18-e2e/pylib python nq35_ftp_hot.py [--n-win 64 --n-held 0]
"""
import os, sys, json, argparse
from multiprocessing import Pool
T = "/home/coder/git/nestquant/threads"
ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int, default=40)
ap.add_argument("--seq", type=int, default=512); ap.add_argument("--n-win", type=int, default=64)
ap.add_argument("--n-held", type=int, default=0)
ap.add_argument("--cache", default="/tmp/nestquant/35-nq15/private/ftp_cache")
ap.add_argument("--joint", default="/tmp/nestquant/33-search/joint/models/jF_all.pt")
ap.add_argument("--manifest", default="/tmp/nestquant/32-gbdt-sal/k0_manifest.json")
ap.add_argument("--n-float", type=int, default=32); ap.add_argument("--hm", type=float, default=0.7)
ap.add_argument("--procs", type=int, default=8); ap.add_argument("--threads", type=int, default=2)
a = ap.parse_args()
os.environ["NQ_SEQ"] = str(a.seq)
for p in (f"{T}/05-exl3-harness", "/home/coder/git/orbit-duet", f"{T}/25-campaign", f"{T}/27-pv-tune",
          f"{T}/18-e2e-eval", f"{T}/12-reference-encoder"):
    sys.path.insert(0, p)
import numpy as np                  # noqa: E402
import torch                        # noqa: E402
G, NE, L = 16, 256, a.layer
M = json.load(open(a.manifest))
FIX = sorted(map(int, M["default_allocation"][str(L)]))
FDEF = [int(e) for e in M["floating_default"][str(L)] if int(e) not in set(FIX)][:a.n_float]
key = f"L{L}_s{a.seq}_tr{a.n_win}_ho{a.n_held}"
RT = f"{a.cache}/{key}_route_cpu.pt"


def one(n):
    torch.set_num_threads(a.threads)
    sys.path.insert(0, f"{T}/33-search/joint")
    from joint_predictor import JointPredictor
    Z = torch.load(RT, weights_only=False)
    ids = Z["ids"][n].long().numpy(); w = Z["w"][n].float().numpy(); xn = Z["xn2"][n].double().numpy()
    seq = ids.shape[0]
    fixed = np.zeros((1, NE), bool); fixed[0, FIX] = True
    want = np.zeros((1, NE), bool); want[0, FDEF] = True
    P = JointPredictor([L], {L: FIX}, a.joint, n_float=a.n_float, hm=a.hm, device="cpu", mode="sync",
                       num_threads=a.threads)
    serve = np.zeros((seq // G, NE), bool)
    sal = (w.astype(np.float64) ** 2) * xn[:, None]
    try:
        for t in range(seq):
            if t % G == 0:
                serve[t // G] = (want | fixed)[0]
            c = np.zeros((1, NE)); s = np.zeros((1, NE))
            np.add.at(c[0], ids[t], 1.0); np.add.at(s[0], ids[t], sal[t])
            if P.step(c, 1, None, t == 0, sal=s):
                tg = P.target(want)
                if tg is not None:
                    want = tg & ~fixed
    finally:
        P.close()
    return serve


if __name__ == "__main__":
    out = f"{a.cache}/{key}_hot_nf{a.n_float}_hm{a.hm:g}.pt"
    if not os.path.exists(RT):
        import nq_e2e as E2E, nq_io
        torch.set_num_threads(16)
        C = torch.load(f"{a.cache}/{key}.pt", weights_only=False)
        cfg = E2E.load_config()
        layer, sparse = E2E.Backbone(cfg, nq_io.FP8Model(E2E.FP8_DIR), "cpu").build(L)
        assert sparse
        with torch.no_grad():
            ids, w, xn2 = [], [], []
            for c0 in range(0, C["hmid"].shape[0], 8192):
                x = layer.post_attention_layernorm(C["hmid"][c0:c0 + 8192])
                _, w_, i_ = layer.mlp.gate(x)
                ids.append(i_.to(torch.int16)); w.append(w_.half()); xn2.append(x.float().pow(2).sum(-1))
        N = C["tok"].shape[0]
        torch.save(dict(ids=torch.cat(ids).view(N, a.seq, 8), w=torch.cat(w).view(N, a.seq, 8),
                        xn2=torch.cat(xn2).view(N, a.seq)), RT)
        del layer, C
        print(f"routing (CPU) of {N} windows -> {RT}", flush=True)
    N = torch.load(RT, weights_only=False)["ids"].shape[0]
    import multiprocessing as mp                    # spawn: the parent ran multi-threaded torch (fork would hang)
    with mp.get_context("spawn").Pool(a.procs) as pool:
        hot = np.stack(pool.map(one, range(N)))
    torch.save(dict(hot=torch.from_numpy(hot), n_float=a.n_float, hm=a.hm, fixed=FIX, layer=L, G=G), out)
    print(f"hot set -> {out}: {hot.shape}, mean hot/block {hot.sum(-1).mean():.1f}", flush=True)
