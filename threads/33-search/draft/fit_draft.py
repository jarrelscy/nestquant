#!/usr/bin/env python3
"""T33 draft: learned routing predictor for the next j = 1..4 positions from the MTP drafts (PRIVATE data).
Stage 1 (global ridge, all 75 layers at once): z_t = [rmsnorm(m_j,t), rmsnorm(hn_t)] (12288) -> raw router logits of
  position t+j at every sparse layer (75x256).  m_j = MTP step-j output residual at anchor t (state of position t+j).
Stage 2 (per layer, per expert, 5 coefs): logit_{t+j} ~ a*C_j + b*lg_t + c*sw_t + d*C_1(j>1) + e  (lg_t = current token
  router logits, sw_t = token-swap probe; both known at the refresh).
Train: calib-fit chains %4 != 3 (all positions; stage 2 on every 4th); lambda by val-chain logit MSE.
Outputs: private/pred/CORPUS/L{L}.npz  P [nb, 4, 256] fp16 predicted logits at anchors t = 16(b+1)-1 for t+1..t+4
(NaN where t+j leaves the window: the capture's windows are independent), recall table -> private/recall.json."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import cap_io as C  # noqa: E402

LAYERS = list(range(3, 78))
PRIV = "/tmp/nestquant/33-search/draft/private"
K = 4
SEQ = 2048
NBC = 512
t00 = time.time()


def log(m):
    print(f"[{time.time() - t00:6.0f}s] {m}", flush=True)


def rn(x):
    x = x.astype(np.float32)
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6)


def bias_of(L):
    return np.load(f"/tmp/nestquant/32-gbdt-sal/trace2/L{L}.r0of8.npz")["bias"].astype(np.float32)


def top8(lg, b):
    s = 1 / (1 + np.exp(-lg.astype(np.float32))) + b
    return np.argpartition(-s, 8, axis=-1)[..., :8]


def recall(pi, ti):
    """mean |pred top8 & true top8| / 8"""
    return float((pi[..., :, None] == ti[..., None, :]).any(-1).mean())


def main():
    corpora = ["calib-fit", "glm52-heldout"]
    Z, LG, SW, IDS = {}, {}, {}, {}
    for c in corpora:
        hn = rn(C.load(c, "head", ["hn"])["hn"])
        Z[c] = [np.concatenate([rn(C.load(c, f"mtp{j}", ["m"])["m"]), hn], 1).astype(np.float16) for j in range(1, K + 1)]
        del hn
        LG[c], SW[c], IDS[c] = {}, {}, {}
        for L in LAYERS:
            d = C.load(c, f"L{L}", ["lg", "sw", "ids"])
            LG[c][L], SW[c][L], IDS[c][L] = d["lg"], d["sw"], d["ids"]
        log(f"loaded {c} T={Z[c][0].shape[0]}")
    T = Z["calib-fit"][0].shape[0]
    pos = np.arange(T) % SEQ
    chain = np.arange(T) // (4 * SEQ)
    is_val = chain % 4 == 3
    # parity of capture routing vs T32 trace (ids)
    sys.path.insert(0, "/home/coder/git/nestquant/threads/32-gbdt-sal")
    import t32lib as T32
    for L in (3, 40, 77):
        tid = T32.load_layer(L, "glm52-heldout")[0]
        log(f"parity L{L} heldout ids == trace: {np.mean(np.sort(tid, 1) == np.sort(IDS['glm52-heldout'][L], 1)):.5f}")
    bias = {L: bias_of(L) for L in LAYERS}
    res = {}
    Wj = {}
    for j in range(1, K + 1):
        ok = pos < SEQ - j
        tr = ok & ~is_val
        va = ok & is_val
        X = Z["calib-fit"][j - 1]
        mu = X[tr].astype(np.float32).mean(0)
        Ymu = np.stack([LG["calib-fit"][L][np.nonzero(tr)[0] + j].astype(np.float32).mean(0) for L in LAYERS], 0)
        D = X.shape[1]
        XtX = np.zeros((D, D), np.float64)
        XtY = np.zeros((D, len(LAYERS) * 256), np.float64)
        idx = np.nonzero(tr)[0]
        for c0 in range(0, len(idx), 16384):
            ii = idx[c0:c0 + 16384]
            x = X[ii].astype(np.float32) - mu
            y = np.concatenate([LG["calib-fit"][L][ii + j].astype(np.float32) for L in LAYERS], 1) - Ymu.reshape(-1)
            XtX += x.T.astype(np.float64) @ x
            XtY += x.T @ y
        log(f"j={j} gram done")
        ev, V = np.linalg.eigh(XtX)
        VtXY = V.T @ XtY
        best = None
        vi = np.nonzero(va)[0][::2]
        xv = (X[vi].astype(np.float32) - mu) @ V
        yv = np.concatenate([LG["calib-fit"][L][vi + j].astype(np.float32) for L in LAYERS], 1) - Ymu.reshape(-1)
        for lam in (1e1, 1e2, 1e3, 1e4, 3e4, 1e5):
            Wv = VtXY / (ev + lam * len(idx) / 1e5)[:, None]
            mse = float(((xv @ Wv - yv) ** 2).mean())
            log(f"j={j} lam {lam:g} val mse {mse:.4f} (var {float((yv ** 2).mean()):.4f})")
            if best is None or mse < best[0]:
                best = (mse, lam, Wv)
        W = (V @ best[2]).astype(np.float32)
        Wj[j] = (W, mu, Ymu)
        res[f"ridge_j{j}"] = dict(lam=best[1], val_mse=best[0])
        del XtX, XtY, V, VtXY, xv, yv
    # stage-1 predictions at every position (fp16), per corpus
    C1 = {}
    for c in corpora:
        C1[c] = []
        for j in range(1, K + 1):
            W, mu, Ymu = Wj[j]
            Zs = Z[c][j - 1][3::4]
            P = np.empty((Zs.shape[0], len(LAYERS) * 256), np.float16)
            for c0 in range(0, P.shape[0], 16384):
                P[c0:c0 + 16384] = ((Zs[c0:c0 + 16384].astype(np.float32) - mu) @ W + Ymu.reshape(-1)).astype(np.float16)
            C1[c].append(P)
        log(f"stage1 preds {c}")
    del Z
    # stage 2 per layer and recall table
    sub = np.zeros(T, bool); sub[3::4] = True
    rec = {}
    for li, L in enumerate(LAYERS):
        cols = slice(li * 256, (li + 1) * 256)
        coefs = {}
        for j in range(1, K + 1):
            ok = pos < SEQ - j
            ii = np.nonzero(ok & ~is_val & sub)[0]
            feats = [C1["calib-fit"][j - 1][ii // 4, cols], LG["calib-fit"][L][ii], SW["calib-fit"][L][ii]]
            if j > 1:
                feats.append(C1["calib-fit"][0][ii // 4, cols])
            A = np.stack([f.astype(np.float32) for f in feats] + [np.ones((len(ii), 256), np.float32)], -1)
            y = LG["calib-fit"][L][ii + j].astype(np.float32)
            coef = np.stack([np.linalg.lstsq(A[:, e], y[:, e], rcond=None)[0] for e in range(256)])   # [256, nf]
            coefs[j] = coef
        for c in corpora:
            Tc = LG[c][L].shape[0]
            pc = np.arange(Tc) % SEQ
            nb = Tc // 16
            anc = (np.arange(nb) + 1) * 16 - 1
            P = np.full((nb, K, 256), np.nan, np.float16)
            for j in range(1, K + 1):
                okj = pc[anc] < SEQ - j
                a = anc[okj]
                feats = [C1[c][j - 1][a // 4, cols], LG[c][L][a], SW[c][L][a]] + ([C1[c][0][a // 4, cols]] if j > 1 else [])
                A = np.stack([f.astype(np.float32) for f in feats] + [np.ones((len(a), 256), np.float32)], -1)
                pr = (A * coefs[j][None]).sum(-1)
                P[okj, j - 1] = pr.astype(np.float16)
                # recall at anchors (heldout: all; calib-fit: val chains only)
                sel = np.ones(len(a), bool) if c != "calib-fit" else is_val[a]
                ti = IDS[c][L][a[sel] + j].astype(np.int64)
                r = rec.setdefault(c, {}).setdefault(j, {k: [] for k in ("pred", "copy", "swap", "stage1")})
                r["pred"].append(recall(top8(pr[sel], bias[L]), ti))
                r["copy"].append(recall(IDS[c][L][a[sel]].astype(np.int64), ti))
                r["swap"].append(recall(top8(SW[c][L][a[sel]], bias[L]), ti))
                r["stage1"].append(recall(top8(C1[c][j - 1][a[sel] // 4, cols], bias[L]), ti))
            os.makedirs(f"{PRIV}/pred/{c}", exist_ok=True)
            np.savez(f"{PRIV}/pred/{c}/L{L}.npz", P=P)
        if L % 10 == 0:
            log(f"layer {L}: " + " ".join(f"j{j} pred {rec['glm52-heldout'][j]['pred'][-1]:.3f} copy "
                                         f"{rec['glm52-heldout'][j]['copy'][-1]:.3f}" for j in range(1, K + 1)))
    summ = {c: {j: {k: float(np.mean(v)) for k, v in r.items()} for j, r in rj.items()} for c, rj in rec.items()}
    res["recall"] = summ
    res["recall_per_layer_heldout"] = {j: {k: v for k, v in r.items()} for j, r in rec["glm52-heldout"].items()}
    json.dump(res, open(f"{PRIV}/recall.json", "w"), indent=1)
    for c in summ:
        for j in summ[c]:
            log(f"{c} j={j} " + " ".join(f"{k} {v:.4f}" for k, v in summ[c][j].items()))


if __name__ == "__main__":
    main()
