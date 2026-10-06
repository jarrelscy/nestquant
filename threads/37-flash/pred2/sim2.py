"""T37 predictor study 2: EMA residency (NF - SC slots, refresh every G tokens, hysteresis hm, half-life hl) + a scratch
pool of SC slots filled per token with just-in-time loads of the most salient misses (oracle = upper bound for a
cross-layer pre-gate), up to `tb` loads per layer per token on average (credit bucket, cap 4), evicting least-recently
used scratch entries.  Swaps = SSD loads (EMA promotions of experts already in scratch are free moves)."""
import argparse, json, sys
import numpy as np
from sim import TR, FX, NE, NF, load, dense_sal


def run(S, p0, fx, dflt, G, hm, hl, SC, tb):
    n = len(S); a = 0.5 ** (1.0 / hl); nres = NF - SC
    cur = dflt.copy(); cur[np.where(dflt)[0][nres:]] = False
    scr = np.zeros(NE, bool); last = np.zeros(NE)
    state = np.zeros(NE); credit = 0.0; served = tot = 0.0; swaps = 0
    for t in range(n):
        if t and t % G == 0:
            v = np.where(fx, -np.inf, state * np.where(cur, 1 + hm, 1.0))
            new = np.zeros(NE, bool); new[np.lexsort((~cur, -v))[:nres]] = True
            ld = new & ~cur & ~scr
            swaps += int(ld.sum()) if t >= p0 else 0
            moved = new & scr; scr &= ~moved              # promoted scratch entries; demoted residents become scratch
            dem = cur & ~new
            free = SC - int(scr.sum())
            if free > 0 and dem.any():
                d = np.where(dem)[0]; d = d[np.argsort(-state[d])][:free]; scr[d] = True; last[d] = t
            cur = new
        st = S[t]
        if SC and tb:
            credit = min(credit + tb, 4.0)
            miss = np.where(~(fx | cur | scr) & (st > 0))[0]
            k = min(int(credit), len(miss))
            for e in miss[np.argsort(-st[miss])[:k]]:
                if scr.sum() >= SC:
                    s_ = np.where(scr)[0]; scr[s_[np.argmin(last[s_])]] = False
                scr[e] = True; last[e] = t
            credit -= k; swaps += k if t >= p0 else 0
        hit = fx | cur | scr
        last[scr & (st > 0)] = t
        if t >= p0:
            served += float(st[hit].sum()); tot += float(st.sum())
        state *= a; state += st
    return served, tot, swaps, max(0, n - p0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="3,10,20,30,40"); ap.add_argument("--maxseq", type=int, default=12)
    ap.add_argument("--cfgs", default="16:0.5:64:0:0,16:0.5:64:4:0.5,16:0.5:64:8:0.5,16:0.5:64:8:1,16:0.5:64:16:1,16:0.5:64:16:2,16:1:128:8:0.5,16:1:128:16:1")
    a = ap.parse_args()
    cfgs = [tuple(float(v) for v in c.split(":")) for c in a.cfgs.split(",")]
    acc = {c: np.zeros(4) for c in cfgs}
    seqs = json.load(open(f"{TR}/seqs.r0of8.json"))["seqs"]
    for L in map(int, a.layers.split(",")):
        fx = np.zeros(NE, bool); fx[FX["fixed_set"][str(L)]] = True
        nr = np.array(FX["n_routed"][str(L)]); dflt = np.zeros(NE, bool)
        dflt[[e for e in np.argsort(-nr, kind="stable") if not fx[e]][:NF]] = True
        ids, w, xn = load(L, 0); o = 0
        for k, s in enumerate(seqs):
            n = s["rows"]
            if k < a.maxseq:
                S = dense_sal(ids[o:o + n], w[o:o + n], xn[o:o + n])
                for c in cfgs:
                    acc[c] += run(S, s["prompt_len"], fx, dflt, int(c[0]), c[1], c[2], int(c[3]), c[4])
            o += n
        print(f"L{L}", file=sys.stderr, flush=True)
    for c, v in acc.items():
        print(f"G{int(c[0])} hm{c[1]} hl{c[2]:g} scratch{int(c[3])} tb{c[4]:g}  sal_cov {v[0] / v[1]:.4f}  swaps/layer/tok {v[2] / v[3]:.3f}")
