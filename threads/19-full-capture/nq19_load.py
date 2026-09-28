"""Thread 19 loader: rebuild encoder Hessians / diagonal output weights from the full-model capture.

    import nq19_load as C
    cap = C.Capture()                       # root /tmp/nestquant/19-capture
    HG  = cap.glm_H(L, E)                   # thread-12 glm_H format: {"H": [Hx, Hx, Ha], "G": [Gg, Gu, None]}
    st  = cap.pilot_stats(L, E)             # orbit-style {"grams": [W, Wd], "metadata": {"training_rows", "mass"}}
    c   = cap.components(L, E)              # raw sums (A2, A0, D2, D0, Dc, C_ctx, C_all, g, scalars)
    ev  = cap.eval_capture(L, "val"|"matched")   # harness capture dict for harness.evaluate
See FORMAT.md for definitions.
"""
import json
import os

import numpy as np
import torch

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

    def glm_H(self, L, E, alpha=0.25, ctx_mass=0.25, device="cuda", c=None):
        """Thread-08 recipe (unif0.75): H = alpha nt(W) + (1-alpha) nt(U), count = 1, with
        W = routed p-weighted + context at weight cp,  U = routed + context unweighted,  cp^2 = ctx_mass * sum p^2 / n_ctx.
        G (thread 06/12): diag(dmix(sum w cg^2)^0.5) for gate, same with cu for up."""
        c = c or self.components(L, E, device, keys=("A2", "A0", "D2", "D0", "Dc", "C_ctx"))
        cp2 = ctx_mass * c["sum_p2"] / c["n_ctx"]
        Hx = nq19.recipe_H(c["A2"], c["A0"], c["C_ctx"], cp2, alpha)
        Ha = nq19.recipe_H(c["D2"], c["D0"], c["Dc"], cp2, alpha)
        g = c["g"]
        dmix = lambda A, B: (alpha * A / A.mean() + (1 - alpha) * B / B.mean()).float()
        Gg = torch.diag(dmix(g[0] + cp2 * g[4], g[1] + g[4]).clamp_min(1e-30).pow(0.5))
        Gu = torch.diag(dmix(g[2] + cp2 * g[5], g[3] + g[5]).clamp_min(1e-30).pow(0.5))
        return dict(H=[Hx, Hx, Ha], G=[Gg, Gu, None],
                    meta=dict(layer=L, expert=E, n_routed=c["n_routed"], n_ctx=c["n_ctx"], ess=c["ess"], cp2=cp2))

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
        teacher = weights(source or nq19.SRC, L, E)
        cap = self.eval_capture(L, kind)
        return h.ExpertData(L, E, teacher, self.pilot_stats(L, E), cap, self.eval_path(L, kind),
                            self._open(L)[0])
