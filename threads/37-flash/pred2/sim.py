"""T37 predictor study (offline, CPU): salience coverage vs swap rate of block-refresh floating-set policies on the
PRIVATE decode traces.  Coverage = sum over rows of salience (w^2 |x|^2) served at level 4 (fixed + floating) / total.
Policies refresh every G tokens, choosing the top-NF non-fixed experts by a score with hysteresis hm:
  orc   salience of the NEXT G tokens (perfect foresight)
  lag   salience of the last W tokens (trailing window)
  ema   per-token exponential decay of salience, half-life h tokens
State starts per sequence at floating_default; scored rows = decode rows (>= prompt_len), the prompt warms the state.
  python sim.py [--ranks 0] [--layers 3,10,20,30,40] [--maxseq 40]"""
import argparse, json, sys, time
import numpy as np

TR = "/tmp/nestquant/37-flash/private/dec_trace_online"
FX = json.load(open("/tmp/nestquant/37-flash/release/fixed_set.json"))
NE, NF = 288, 63


def load(L, r):
    z = np.load(f"{TR}/L{L}.r{r}of8.npz")
    return z["ids"].astype(np.int64), z["w"].astype(np.float32), z["xn"].astype(np.float32)


def dense_sal(ids, w, xn):
    n = len(ids); S = np.zeros((n, NE), np.float32)
    np.add.at(S, (np.repeat(np.arange(n), ids.shape[1]), ids.ravel()), (w.astype(np.float64) ** 2 * xn[:, None]).ravel())
    return S


def run(S, p0, fx, dflt, pol, G, hm, par, tb=0.0):
    """S [n, NE] per-token salience of one sequence; returns (served, total, swaps, decode tokens)."""
    n = len(S); cur = dflt.copy(); served = tot = 0.0; swaps = 0
    state = np.zeros(NE, np.float64); credit = 0.0
    if pol == "ema":
        a = 0.5 ** (1.0 / par)
    for b0 in range(0, n, G):
        b1 = min(n, b0 + G)
        if b0 > 0:
            if pol == "orc":
                sc = S[b0:b1].sum(0).astype(np.float64)
            elif pol == "lag":
                sc = S[max(0, b0 - par):b0].sum(0).astype(np.float64)
            else:
                sc = state
            v = np.where(fx, -np.inf, sc * np.where(cur, 1 + hm, 1.0))
            if sc[~fx].max() > 0:      # ties (mostly zero scores) keep residents: no churn among unused experts
                new = np.zeros(NE, bool); new[np.lexsort((~cur, -v))[:NF]] = True
                swaps += int((new & ~cur).sum()) if b0 >= p0 else 0
                cur = new
        if tb > 0:     # per-token oracle top-up: up to `credit` just-in-time loads of the most salient misses,
            for t in range(b0, b1):    # each evicting the resident floating expert with the lowest score
                credit = min(credit + tb, 4.0)
                st = S[t]; miss = np.where(~(fx | cur) & (st > 0))[0]
                k = min(int(credit), len(miss))
                if k:
                    for e in miss[np.argsort(-st[miss])[:k]]:
                        res = np.where(cur)[0]; sc2 = state if pol == "ema" else S[max(0, t - 64):t].sum(0)
                        cur[res[np.argmin(sc2[res])]] = False; cur[e] = True
                    credit -= k; swaps += k if t >= p0 else 0
                if t >= p0:
                    served += float(st[fx | cur].sum()); tot += float(st.sum())
                if pol == "ema":
                    state *= a; state += st
            continue
        blk = S[b0:b1]
        if b1 > p0:
            m = slice(max(b0, p0) - b0, b1 - b0)
            served += float(blk[m][:, fx | cur].sum()); tot += float(blk[m].sum())
        if pol == "ema":
            for t in range(b0, b1):
                state *= a; state += S[t]
    return served, tot, swaps, max(0, n - p0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", default="0"); ap.add_argument("--layers", default="3,10,20,30,40")
    ap.add_argument("--maxseq", type=int, default=40)
    ap.add_argument("--pols", default="orc:16:0:0,lag:16:0:16,lag:16:0.7:16,ema:16:0:16,ema:16:0:32,ema:16:0:64,ema:16:0.7:32")
    a = ap.parse_args()
    pols = [(p, int(g), float(h), float(x), float(t[0]) if t else 0.0) for p, g, h, x, *t in (s.split(":") for s in a.pols.split(","))]
    acc = {q: np.zeros(4) for q in pols}
    for L in map(int, a.layers.split(",")):
        fx = np.zeros(NE, bool); fx[FX["fixed_set"][str(L)]] = True
        nr = np.array(FX["n_routed"][str(L)]); dflt = np.zeros(NE, bool)
        dflt[[e for e in np.argsort(-nr, kind="stable") if not fx[e]][:NF]] = True
        for r in map(int, a.ranks.split(",")):
            seqs = json.load(open(f"{TR}/seqs.r{r}of8.json"))["seqs"]
            ids, w, xn = load(L, r); o = 0
            for k, s in enumerate(seqs):
                n = s["rows"]
                if k < a.maxseq:
                    S = dense_sal(ids[o:o + n], w[o:o + n], xn[o:o + n])
                    for q in pols:
                        acc[q] += run(S, s["prompt_len"], fx, dflt, q[0], q[1], q[2], int(q[3]) if q[0] == "lag" else q[3], q[4])
                o += n
        print(f"L{L} done", file=sys.stderr, flush=True)
    for q, v in acc.items():
        print(f"{q[0]:4s} G{q[1]:<3d} hm{q[2]:<4} par{q[3]:<6g} tb{q[4]:<5g} sal_cov {v[0] / v[1]:.4f}  swaps/layer/tok {v[2] / v[3]:.3f}")


if __name__ == "__main__":
    main()
