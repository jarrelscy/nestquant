"""9-expert confirmation of best stacks vs same-H EXL3-2/4."""
import sys
from core17 import *
out = {}
for L in (16, 49, 66):
    for E in (36, 92, 165):
        X = Exp(L, E); M = {}
        def run(name, var=None, G=None, lam=0.0, seqw=None, l4=True):
            W2, W4 = [], []
            for p in range(3):
                bq = VariantQ(VARIANTS[var]) if var else None
                if p == 2 and seqw is not None:
                    Wt, H, _ = seq_down_target(X, W2[0], W2[1], seqw)
                else:
                    Wt, H = X.W[p], X.H(p)
                Gv = X.Gdiag(p).pow(G) if (G and p < 2) else None
                o = quantize(Wt, H, sigma=SIG[p], base_q=bq, lam=lam, level4=l4, G=Gv)
                W2.append(o["W2"]); W4.append(o.get("W4"))
            M[name + "@2"] = W2
            if l4: M[name + "@4"] = W4
        run("nat"); run("sign", "sign", l4=False); run("sg4", "sign_gain4", l4=False)
        run("sign+G+b0.3", "sign", G=0.5, lam=0.3); run("sg4+G+b0.3", "sign_gain4", G=0.5, lam=0.3)
        run("sign+G+seqW", "sign", G=0.5, seqw=1.0, l4=False)
        M["exl3_2"] = exl3_anchor(X, 2); M["exl3_4"] = exl3_anchor(X, 4)
        res = X.capture(M); out[f"L{L}E{E}"] = res
        for k, v in res.items(): print(f"L{L}E{E} {k:18s}", v, flush=True)
        jdump(out, "results/confirm9.json")
        del X, M; torch.cuda.empty_cache()
