"""T35: trellis rel-MSE r(b) of the NestQuant base code (mul1 codebook, 16-bit state, 256-step tail-biting ring) at
pattern rates b (step shift KA + mask bit, threads/13 ref15_spec PATTERNS), on iid Gaussian targets, CPU torch.
Viterbi = threads/12-reference-encoder/nq_patvit._run (vendored: LUT on CPU).  Plain MSE (no LDLQ feedback);
per rate the input gain g is grid-searched and the reconstruction LS-rescaled (the encoder's suh/svh scales do this).
  [KS=1.25,...: only these rates, merged into the existing json] python rate_mse.py [NRING] -> /tmp/nestquant/35-nq15/rate_mse.json"""
import json, sys, time
import torch
sys.path.insert(0, "/home/coder/git/nestquant/threads/05-exl3-harness")
import harness as h
torch.set_num_threads(int(__import__("os").environ.get("NT", "16")))
PAT = {1.0: (1, 0), 1.25: (1, 0x8888), 1.5: (1, 0xAAAA), 1.75: (1, 0xEEEE), 2.0: (2, 0), 2.25: (2, 0x8888), 2.5: (2, 0xAAAA)}
V = h.codebook_lut("mul1", device="cpu")


def vsteps(K):
    KA, MASK = PAT[K]
    return [KA + ((MASK >> ((-i) % 16)) & 1) for i in range(256)]


@torch.no_grad()
def run(w, Dst):
    T, L = w.shape
    inf = float("inf"); backs = [None] * L

    def forward(roll, start):
        if start is None:
            cost = torch.zeros(T, 65536)
        else:
            cost = torch.full((T, 65536), inf); cost.scatter_(1, start[:, None], 0.)
        for i in range(L):
            ri = (i + roll) % L; k = Dst[ri]; E = 1 << (16 - k)
            mn, top = cost.view(T, 1 << k, E).min(1)
            backs[ri] = top.to(torch.uint8)
            d = (V[None] - w[:, ri, None]).square()
            cost = (d.view(T, E, 1 << k) + mn[:, :, None]).view(T, 65536)
        return cost

    def trace(roll, s, stop):
        out = torch.empty((T, L), dtype=torch.int64); tt = torch.arange(T)
        for i in range(L - 1, -1, -1):
            ri = (i + roll) % L; k = Dst[ri]
            out[:, ri] = s; e = s >> k
            s = (backs[ri][tt, e].long() << (16 - k)) | e
            if stop and ri == 0:
                break
        return out, s

    c = forward(L // 2, None)
    _, start = trace(L // 2, c.argmin(1), True)
    forward(0, start)
    idx, _ = trace(0, start, False)
    return V[idx]


if __name__ == "__main__":
    NR = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    g0 = torch.Generator().manual_seed(35)
    w = torch.randn(NR, 256, generator=g0)
    sd = float(V.std())
    import os
    OJ = os.environ.get("OJ", "/tmp/nestquant/35-nq15/rate_mse.json")
    KS = [float(k) for k in os.environ["KS"].split(",")] if os.environ.get("KS") else list(PAT)
    res = {float(k): v for k, v in json.load(open(OJ))["res"].items()} if os.environ.get("KS") and os.path.exists(OJ) else {}
    for K in KS:
        best = None
        for g in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4):
            t0 = time.time()
            q = run(w * g * sd, vsteps(K))
            a = float((w * q).sum() / (q * q).sum())
            r = float((w - a * q).square().sum() / w.square().sum())
            r_rows = ((w - a * q).square().sum(1) / w.square().sum(1))
            if best is None or r < best[0]:
                best = (r, g, float(r_rows.std() / NR ** 0.5))
            print(f"K={K} g={g} rel-MSE {r:.5f} ({time.time() - t0:.1f}s)", flush=True)
        res[K] = dict(rel_mse=best[0], gain=best[1], se=best[2])
        print(f"== K={K} best rel-MSE {best[0]:.5f} at g={best[1]}", flush=True)
    r2 = res[2.0]["rel_mse"]
    for K in res:
        res[K]["ratio_vs_2"] = res[K]["rel_mse"] / r2
        res[K]["s"] = (res[K]["rel_mse"] / r2) ** 0.5
        res[K]["ratio_4pow"] = 4 ** (2 - K)
    json.dump(dict(nring=NR, target="iid N(0,1)", codebook="mul1", res=res),
              open(OJ, "w"), indent=1)
    for K, v in res.items():
        print(f"b={K}: r={v['rel_mse']:.5f} r/r2={v['ratio_vs_2']:.3f} (4^(2-b)={v['ratio_4pow']:.3f}) s={v['s']:.4f}")
