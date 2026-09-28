"""Thread 19 loader: rebuild encoder Hessians / diagonal output weights from the full-model capture.

    import nq19_load as C
    cap = C.Capture()                       # root /tmp/nestquant/19-capture
    HG  = cap.glm_H(L, E)                   # thread-12 glm_H format: {"H": [Hx, Hx, Ha], "G": [Gg, Gu, None]}
    st  = cap.pilot_stats(L, E)             # orbit-style {"grams": [W, Wd], "metadata": {"training_rows", "mass"}}
    c   = cap.components(L, E)              # raw sums (A2, A0, D2, D0, Dc, C_ctx, C_all, g, scalars)
    ev  = cap.eval_capture(L, "val"|"matched")   # harness capture dict for harness.evaluate
    cb  = cap.components_bnd(L, E)          # {(kind, bucket): same sums restricted to boundary rows} (bnd19)
    HGb = cap.glm_H(L, E, bnd_w=50)         # boundary-upweighted recipe: EXPERIMENTS ONLY. Encode = bnd_w None (weight 1;
                                            # lead 2026-09-28: 50x costs +3% on all tokens)
    sal = cap.salience(L)                   # REAP-style salience / usage per expert, all rows and per bucket
See FORMAT.md for definitions.
"""
import json
import os

import numpy as np
import torch

import bnd19
import nq19
from nq19 import D, F, OUT, unpack

GU_SIGMA = 0.5; DOWN_SIGMA = 1.0          # thread-08 selected damping (added by the encoder, not stored in H)


class Capture:
    """stats="stats" = cumulative (all shards merged so far); "stats0" = frozen shard-0 snapshot.
    Each layer is resolved to one immutable version directory and all its files are opened on first access,
    so a concurrent merge (atomic symlink swap + deletion of the superseded version) never affects a reader."""

    def __init__(self, root=OUT, stats="stats"):
        self.root = root
        self.stats = stats
        self._layer = {}
        self._bndc = {}
        self._teacher = {}

    def _open(self, L):
        if L not in self._layer:
            for _ in range(5):
                try:
                    vd = os.path.realpath(f"{self.root}/{self.stats}/L{L}")
                    m = json.load(open(f"{vd}/meta.json"))
                    arr = {}
                    if m.get("schema") == "nestquant-19-stats-v2":
                        for k, r in m["files"]["raw"].items():
                            mm = np.memmap(f"{vd}/{r['file']}", dtype=np.float32, mode="r",
                                           shape=(r["rows"], r["stride_bytes"] // 4))
                            arr[k] = mm[:, :r["packed"]]
                        for k in ("C_ctx", "C_all", "gdiag", "scalars"):
                            arr[k] = np.load(f"{vd}/{k}.npy", mmap_mode="r")
                    else:                                    # v1 single-shot layout (pilot)
                        for k in ("A2", "A0", "D2", "D0", "Dc", "C_ctx", "C_all", "gdiag", "scalars"):
                            arr[k] = np.load(f"{vd}/{k}.npy", mmap_mode="r")
                    self._layer[L] = (vd, m, arr)
                    break
                except FileNotFoundError:
                    import time; time.sleep(1)               # raced a version swap; re-resolve
            else:
                raise FileNotFoundError(f"{self.root}/{self.stats}/L{L}")
        return self._layer[L]

    def meta(self, L):
        return self._open(L)[1]

    def _arr(self, L, k):
        return self._open(L)[2][k]

    def packed(self, L, k, E=None, device="cuda"):
        a = self._arr(L, k)
        v = a if E is None else a[E]
        return torch.from_numpy(np.ascontiguousarray(v)).to(device)

    def components(self, L, E, device="cuda", keys=("A2", "A0", "D2", "D0", "Dc", "C_ctx", "C_all")):
        out = {}
        for k in keys:
            n = D if k[0] in "AC" else F
            out[k] = unpack(self.packed(L, k, None if k.startswith("C_") else E, device), n)
        out["g"] = torch.from_numpy(np.array(self._arr(L, "gdiag")[E])).to(device)
        n, sp, sp2, sp4 = [float(v) for v in self._arr(L, "scalars")[E]]
        out.update(n_routed=int(n), sum_p=sp, sum_p2=sp2, sum_p4=sp4, ess=sp2 ** 2 / sp4 if sp4 else 0.,
                   n_ctx=self.meta(L)["n_ctx"])
        return out

    def glm_H(self, L, E, alpha=0.25, ctx_mass=0.25, device="cuda", c=None, bnd_w=None, bnd_ctx=True):
        """Thread-08 recipe (unif0.75): H = alpha nt(W) + (1-alpha) nt(U), count = 1, with
        W = routed p-weighted + context at weight cp,  U = routed + context unweighted,  cp^2 = ctx_mass * sum p^2 / n_ctx.
        G (thread 06/12): diag(dmix(sum w cg^2)^0.5) for gate, same with cu for up."""
        c = c or self.components(L, E, device, keys=("A2", "A0", "D2", "D0", "Dc", "C_ctx"))
        cp2 = ctx_mass * c["sum_p2"] / c["n_ctx"]
        if bnd_w is not None:                    # W' = W + sum_g (w_g - 1) W_g, U' likewise (then nt(), as usual)
            c = dict(c, g=c["g"].clone())
            for k in ("A2", "A0", "D2", "D0", "Dc", "C_ctx"):
                c[k] = c[k].clone()
            for (kind, b), w in bnd_weights(bnd_w).items():
                if w == 1:
                    continue
                cb = self.components_bnd(L, E, device, groups=[(kind, b)], ctx=bnd_ctx)[(kind, b)]
                for k in ("A2", "A0", "D2", "D0", "Dc", "C_ctx"):
                    if k in cb:
                        c[k] += (w - 1) * cb[k]
                c["g"] += (w - 1) * cb["g"]
                del cb
        Hx = nq19.recipe_H(c["A2"], c["A0"], c["C_ctx"], cp2, alpha)
        Ha = nq19.recipe_H(c["D2"], c["D0"], c["Dc"], cp2, alpha)
        g = c["g"]
        dmix = lambda A, B: (alpha * A / A.mean() + (1 - alpha) * B / B.mean()).float()
        Gg = torch.diag(dmix(g[0] + cp2 * g[4], g[1] + g[4]).clamp_min(1e-30).pow(0.5))
        Gu = torch.diag(dmix(g[2] + cp2 * g[5], g[3] + g[5]).clamp_min(1e-30).pow(0.5))
        return dict(H=[Hx, Hx, Ha], G=[Gg, Gu, None],
                    meta=dict(layer=L, expert=E, n_routed=c["n_routed"], n_ctx=c["n_ctx"], ess=c["ess"], cp2=cp2))

    # ------------------------------------------------------------------ boundary rows (bnd19, FORMAT.md)
    def _bnd(self, L):
        """All boundary rows of the shards merged in this layer's stats version (host; one layer cached)."""
        if L not in self._bndc:
            self._bndc.clear()
            m = self.meta(L)
            xs, parts = [], []
            for sh in m["shards"]:
                d = sh.get("bnd_rows")
                if not d or not os.path.exists(f"{d}/rows.npz"):        # restored elsewhere: root-relative
                    d = f"{self.root}/bnd_rows/s{sh['shard']:02d}/L{L}"
                if not os.path.exists(f"{d}/rows.npz"):
                    raise FileNotFoundError(f"{d}: no boundary rows (backfill with capture_bnd.py)")
                r = dict(np.load(f"{d}/rows.npz"))
                xs.append(torch.from_numpy(np.fromfile(f"{d}/x.bf16", np.uint16).reshape(-1, D).view(np.int16)).view(torch.bfloat16))
                parts.append(r)
            cat = lambda k: np.concatenate([p[k] for p in parts])
            self._bndc[L] = dict(x=torch.cat(xs), kind=cat("kind"), bucket=cat("bucket"), is_ctx=cat("is_ctx"),
                                 ids=cat("ids").astype(np.int64), p=cat("p"), d=cat("d"))
        return self._bndc[L]

    def teacher(self, L, E, source=None):
        from orbit_duet.source import weights
        import capture_stats as cs
        if (L, E) not in self._teacher:
            self._teacher.clear()
            self._teacher[(L, E)] = cs.Teacher([w.to("cuda", torch.float32) for w in weights(source or nq19.SRC, L, E)])
        return self._teacher[(L, E)]

    @torch.no_grad()
    def components_bnd(self, L, E, device="cuda", groups=None, ctx=True):
        """{(kind, bucket): {A2, A0, D2, D0[, C_ctx, Dc], g, n_routed, sum_p2, sum_p4, ess, n_ctx_rows}}: the stats
        sums restricted to boundary rows of that kind / distance bucket (unpacked [n, n] fp32, g = [6, 2048] f64
        in the gdiag row layout).  Same arithmetic as capture_stats (bf16 TC Grams, hi/lo p^2 split, fp32 routed gx/ux;
        ctx rows: bf16 gx/ux).  The ctx sums use every boundary context row (no n_dc subsampling).
        kind in bnd19.KINDS ("think", "end"), bucket in bnd19.BUCKET_NAMES ("d1", "d2_4", "d5_16", "d17_32").
        GPU memory ~0.5 GB per group with ctx=True; pass groups=[...] to build a subset."""
        import capture_stats as cs
        B = self._bnd(L)
        te = self.teacher(L, E)
        groups = groups or [(k, b) for k in bnd19.KINDS for b in bnd19.BUCKET_NAMES]
        hit = (B["ids"] == E)
        routed = hit.any(1)
        pE = (B["p"] * hit).sum(1)
        out = {}
        for kind, bname in groups:
            ki, bi = bnd19.KINDS.index(kind) + 1, bnd19.BUCKET_NAMES.index(bname)
            grp = (B["kind"] == ki) & (B["bucket"] == bi)
            r = np.nonzero(grp & routed)[0]
            A2 = torch.zeros(D, D, device="cuda"); A0 = torch.zeros(D, D, device="cuda")
            D2 = torch.zeros(F, F, device="cuda"); D0 = torch.zeros(F, F, device="cuda")
            g = torch.zeros(6, F, device="cuda", dtype=torch.float64)
            pr = torch.from_numpy(pE[r].astype(np.float32))
            for i in range(0, len(r), cs.ROWS):
                xb = B["x"][torch.from_numpy(r[i:i + cs.ROWS])].cuda()
                pp = pr[i:i + cs.ROWS].cuda(); p2 = pp.square()
                cs.gram_(A0, xb); cs.wgram_(A2, xb, p2[:, None])
                h, cg2, cu2 = te.fwd(xb, accurate=True)
                cs.gram_(D0, h); cs.wgram_(D2, h, p2[:, None])
                g[0] += (p2 @ cg2).double(); g[1] += cg2.sum(0).double()
                g[2] += (p2 @ cu2).double(); g[3] += cu2.sum(0).double()
            pd = pr.double()
            o = dict(A2=A2, A0=A0, D2=D2, D0=D0, n_routed=len(r), sum_p2=float(pd.square().sum()),
                     sum_p4=float(pd.pow(4).sum()))
            o["ess"] = o["sum_p2"] ** 2 / o["sum_p4"] if o["sum_p4"] else 0.
            if ctx:
                rc = np.nonzero(grp & B["is_ctx"])[0]
                C = torch.zeros(D, D, device="cuda"); Dc = torch.zeros(F, F, device="cuda")
                for i in range(0, len(rc), cs.ROWS):
                    xb = B["x"][torch.from_numpy(rc[i:i + cs.ROWS])].cuda()
                    cs.gram_(C, xb)
                    h, cg2, cu2 = te.fwd(xb, accurate=False)
                    cs.gram_(Dc, h)
                    g[4] += cg2.sum(0).double(); g[5] += cu2.sum(0).double()
                o.update(C_ctx=C, Dc=Dc, n_ctx_rows=len(rc))
            o["g"] = g
            for k in ("A2", "A0", "D2", "D0", "C_ctx", "Dc", "g"):
                if k in o:
                    o[k] = o[k].to(device)
            out[(kind, bname)] = o
        return out

    def salience(self, L, weights=None):
        """Per-expert usage / REAP salience from sal.npy [256, 9, 6] (bnd19.SAL_CATS x SAL_COLS).
        weights: None -> per category; else bnd_w spec -> boundary-weighted totals (all rows weight 1 plus
        (w_g - 1) x the boundary group), returned as n, sum_p, reap = sum p ||y|| / n, freq = sum_p."""
        vd = self._open(L)[0]
        sal = np.load(f"{vd}/sal.npy")
        if weights is None:
            return dict(categories=bnd19.SAL_CATS, columns=bnd19.SAL_COLS, sums=sal,
                        reap=sal[:, :, 4] / np.maximum(sal[:, :, 0], 1))
        tot = sal[:, 0].copy()
        for (kind, b), w in bnd_weights(weights).items():
            tot += (w - 1) * sal[:, 1 + 4 * bnd19.KINDS.index(kind) + bnd19.BUCKET_NAMES.index(b)]
        return dict(n=tot[:, 0], sum_p=tot[:, 1], reap=tot[:, 4] / np.maximum(tot[:, 0], 1e-30), sums=tot)

    def pilot_stats(self, L, E, ctx_mass=0.25, device="cuda"):
        """orbit-duet accumulate_statistics equivalent: grams = [sum (w x)(w x)^T, sum (w h)(w h)^T] over
        routed (w = p) + context (w = cp) rows; count = n_routed + n_ctx.  (harness ExpertData.H = grams / count.)"""
        c = self.components(L, E, device, keys=("A2", "D2", "Dc", "C_ctx"))
        cp2 = ctx_mass * c["sum_p2"] / c["n_ctx"]
        return dict(grams=[c["A2"] + cp2 * c["C_ctx"], c["D2"] + cp2 * c["Dc"]],
                    metadata=dict(training_rows=c["n_routed"] + c["n_ctx"], mass=c["sum_p2"] * (1 + ctx_mass), layer=L, expert=E))

    def uniform_H(self, L, device="cuda", which="C_all"):
        """Per-layer all-token (or context-row) gate/up input Gram, normalised by its row count."""
        m = self.meta(L)
        return unpack(self.packed(L, which, None, device), D) / (m["T_fit"] if which == "C_all" else m["n_ctx"])

    def ess_table(self, L):
        sc = np.asarray(self._arr(L, "scalars"), np.float64)
        return sc[:, 0], sc[:, 2] ** 2 / np.maximum(sc[:, 3], 1e-300)

    def eval_capture(self, L, kind="val"):
        return torch.load(self.eval_path(L, kind), weights_only=True, mmap=True)

    def eval_path(self, L, kind="val"):
        return f"{self.root}/eval/{kind}/layer_{L}.pt"

    def expert_data(self, L, E, kind="val", source=None):
        """harness.ExpertData with this capture's statistics and eval rows (teacher from the FP8 source)."""
        import harness as h
        from orbit_duet.source import weights
        teacher = [w.to("cuda", torch.float32) for w in weights(source or nq19.SRC, L, E)]
        cap = self.eval_capture(L, kind)
        return h.ExpertData(L, E, teacher, self.pilot_stats(L, E), cap, self.eval_path(L, kind),
                            self._open(L)[0])


def bnd_weights(w):
    """50 -> flat 50x on every (kind, bucket); {"think": 50, "end": {"d1": 50, ...}} -> per kind / bucket."""
    out = {}
    for k in bnd19.KINDS:
        wk = w.get(k, 1) if isinstance(w, dict) else w
        for b in bnd19.BUCKET_NAMES:
            out[(k, b)] = float(wk.get(b, 1) if isinstance(wk, dict) else wk)
    return out
