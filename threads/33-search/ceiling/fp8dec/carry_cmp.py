"""T33l KV-carry measurement, step 2: sm120tf windows captured WITHOUT context (T32 trace) vs WITH the span's preceding
stream rows as KV context (dec.py --mode tf on carry_tasks.py spans).  Same rows, same tokens, same model.
  (1) routing top-8 overlap carry vs no-carry, by window rank within the span (rank 0 = identical context -> parity)
  (2) k0 serve sim (kzero.py: 77 floating, lag 0, dm grid) on the decode rows of those windows, one chain per task:
      v2 sal-hot and oracles orc64/orc128 at churn 3.2 (within-chain), no-carry vs carry (features AND target from the
      same trace), + cross (v2 on no-carry features scored on carry salience).
  carry_cmp.py CARRY_DIR   (CARRY_DIR/meta.json + CARRY_DIR/run/ dec.py output)  PRIVATE: prints aggregates only."""
import json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool
import numpy as np
CD = sys.argv[1]
sys.argv = [sys.argv[0], "sm120tf", "0"]
sys.path.insert(0, "/home/coder/git/nestquant/threads/33-search/ceiling")
sys.path.insert(0, "/home/coder/git/nestquant/threads/32-gbdt-sal")
import kzero as KZ  # noqa: E402
import sm120 as S1  # noqa: E402
import scalelib as SL  # noqa: E402
C, A, T = KZ.C, KZ.A, KZ.T
G, NE = 16, 256
META = json.load(open(f"{CD}/meta.json"))
IX = json.load(open(f"{CD}/run/index.json"))
SP = IX["sparse_layers"]
TI = {t["id"]: t for t in IX["tasks"]}
Z = {m["task"]: np.load(f"{S1.SRC}/{m['task']}.npz") for m in META}


def rows_meta():
    out = []
    for m in META:
        z = Z[m["task"]]
        r = np.concatenate([r0 + np.arange(2048) for _, r0 in m["windows"]])
        rank = np.repeat(np.arange(len(m["windows"])), 2048)
        pw = np.tile(np.arange(2048), len(m["windows"]))
        dm = z["dec"][r].astype(bool)
        rd = r[dm]
        req = z["req"][rd]
        seg = S1.seg_tokens(z["tok"][rd], np.r_[True, req[1:] != req[:-1]])
        out.append(dict(r=r, rank=rank, pw=pw, dm=dm, seg=seg))
    return out


RM = rows_meta()


def blk(ids, w, xn, seg, nb):
    b = np.repeat(np.arange(nb * G) // G, 8)
    e = ids[:nb * G].astype(np.int64).ravel()
    v = (w[:nb * G].astype(np.float64) ** 2 * xn[:nb * G].astype(np.float64)[:, None]).ravel()
    bc = np.bincount(b * NE + e, minlength=nb * NE).reshape(nb, NE).astype(np.float64)
    bs = np.bincount(b * NE + e, weights=v, minlength=nb * NE).reshape(nb, NE)
    a = np.repeat(seg[:nb * G].astype(bool), 8)
    bca = np.bincount(b[a] * NE + e[a], minlength=nb * NE).reshape(nb, NE).astype(np.float64)
    sk = seg[:nb * G].reshape(nb, G)
    return bc, bs, bca, sk.sum(1).astype(np.float64), sk[:, -1].astype(np.int8)


def job(L):
    import lightgbm as lgb
    j = SP.index(L)
    tr = KZ.load_trace_light(L, "sm120tf", KZ.TR_SM)
    rid = {}
    D = {k: {x: [] for x in ("bc", "bs", "bca", "nans", "segl")} for k in ("nc", "cy")}
    sg, nb0 = [], 0
    ovs = {}
    for m, rm in zip(META, RM):
        t = TI[f"carry/{m['task']}"]
        g = t["g"]
        if g not in rid:
            rid[g] = {k: np.load(f"{CD}/run/{k}.g{g}.npy", mmap_mode="r") for k in ("rid", "rw", "rxn")}
        tri = np.concatenate([np.arange(gi * 2048, (gi + 1) * 2048) for gi, _ in m["windows"]])
        nc = tuple(x[tri] for x in tr)
        ci = t["off"] + rm["r"] - m["a"]
        cy = (np.asarray(rid[g]["rid"][j])[ci], np.asarray(rid[g]["rw"][j])[ci], np.asarray(rid[g]["rxn"][j])[ci])
        Ab = np.zeros((len(ci), NE), bool); np.put_along_axis(Ab, nc[0].astype(np.int64), True, 1)
        ov = np.take_along_axis(Ab, cy[0].astype(np.int64), 1).sum(1) / 8.0
        for key, msk in (("rank0", rm["rank"] == 0), ("rank1-3", (rm["rank"] >= 1) & (rm["rank"] <= 3)),
                         ("rank4+", rm["rank"] >= 4), ("rank4+_dec", (rm["rank"] >= 4) & rm["dm"]),
                         ("rank4+_pw<256", (rm["rank"] >= 4) & (rm["pw"] < 256)),
                         ("rank4+_pw>=1024", (rm["rank"] >= 4) & (rm["pw"] >= 1024))):
            s = ovs.setdefault(key, [0.0, 0]); s[0] += float(ov[msk].sum()); s[1] += int(msk.sum())
        r0m = rm["rank"] == 0
        for key, x, y in (("_w_sum", cy[1][r0m], nc[1][r0m]), ("_xn_sum", cy[2][r0m], nc[2][r0m])):
            s = ovs.setdefault(key, [0.0, 0]); s[0] += float(np.abs(x).astype(np.float64).sum()); s[1] += float(np.abs(y).astype(np.float64).sum())
        nb = int(rm["dm"].sum()) // G
        for k, arr in (("nc", nc), ("cy", cy)):
            out = blk(*(x[rm["dm"]] for x in arr), rm["seg"], nb)
            for x, y in zip(("bc", "bs", "bca", "nans", "segl"), out):
                D[k][x].append(y)
        sg.append((nb0, nb0 + nb)); nb0 += nb
    sgs = np.array([s for s, e in sg], np.int64); sge = np.array([e for s, e in sg], np.int64)
    S, Y = {}, {}
    for k in ("nc", "cy"):
        d = {x: np.concatenate(v) for x, v in D[k].items()}
        d["sg"] = sg
        F, _ = SL.feats(d)
        S[k] = lgb.Booster(model_file=C.V2).predict(F.reshape(-1, 9), num_threads=1).reshape(nb0, NE)
        Y[k] = d["bs"]
    f26 = A.f26_ranked(L)
    fxm = np.zeros(NE, bool)
    fd = np.zeros(NE, bool); fd[[e for e in f26 + list(A.fdef[L])][:77]] = True
    res = {"_ov": ovs, "_nb": nb0}

    def run(name, Sc, Yt):
        tot = Yt.sum(); pts = []
        for dm in KZ.DMS:
            num, ci_, ni, cw, nw = KZ.sim(Sc, Yt, fxm, fd, sgs, sge, 77, 0.0, dm, -1)
            pts.append((num / tot, ci_ / max(ni, 1), cw / max(nw, 1)))
        res[name] = pts
    for k in ("nc", "cy"):
        run(f"{k}/v2", S[k], Y[k])
        for n in (4, 8):
            run(f"{k}/orc{16 * n}", KZ.fut(Y[k], n, sgs, sge), Y[k])
    run("x/v2(nc feats, cy target)", S["nc"], Y["cy"])
    run("x/orc64(nc future, cy target)", KZ.fut(Y["nc"], 4, sgs, sge), Y["cy"])
    return L, res


if __name__ == "__main__":
    with Pool(int(os.environ.get("NPROC", "16"))) as p:
        R = dict(p.map(job, T.LAYERS))
    L0 = T.LAYERS[0]
    print(f"tasks {len(META)}  decode blocks {R[L0]['_nb']}  windows {sum(len(m['windows']) for m in META)}")
    for key in [k for k in R[L0]["_ov"] if k.startswith("_")]:
        print(f"rank-0 scale check {key}: carry/no-carry = {sum(R[L]['_ov'][key][0] for L in R) / sum(R[L]['_ov'][key][1] for L in R):.4f}")
    for key in [k for k in R[L0]["_ov"] if not k.startswith("_")]:
        s = sum(R[L]["_ov"][key][0] for L in R); n = sum(R[L]["_ov"][key][1] for L in R)
        print(f"routing top-8 overlap carry vs no-carry {key:16s} {100 * s / max(n, 1):6.2f}%  (rows x layers {n})")
    out = {}
    for name in [k for k in R[L0] if not k.startswith("_")]:
        a = np.array([R[L][name] for L in T.LAYERS]).mean(0)
        s32, flag = C.at_churn([(x[0], x[2]) for x in a], 3.2)
        out[name] = dict(pts=a.tolist(), at3p2=s32, flag=flag)
        print(f"{name:34s} first {100 * a[0][0]:6.2f}/{a[0][2]:5.2f}  @churn3.2 {100 * s32:6.2f} {flag}", flush=True)
    json.dump(out, open(f"{CD}/carry_cmp.json", "w"), indent=1)
