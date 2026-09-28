"""E36 screen of stacks: per-tile variants, G, seqW down, with/without blend 0.3 (L4)."""
from core17 import *
X = Exp(16, 36)
M = {}; info = {}
def run(name, var=None, G=None, lam=0.0, seqw=None, l4=True):
    W2, W4 = [], []
    for p in range(3):
        bq = VariantQ(VARIANTS[var]) if var else None
        if p == 2 and seqw is not None:
            Wt, Hq, _ = seq_down_target(X, W2[0], W2[1], seqw); H = Hq
        else:
            Wt, H = X.W[p], X.H(p)
        Gv = X.Gdiag(p).pow(G) if (G and p < 2) else None
        o = quantize(Wt, H, sigma=SIG[p], base_q=bq, lam=lam, level4=l4, G=Gv)
        W2.append(o["W2"]); W4.append(o.get("W4"))
    M[name + "@2"] = W2
    if l4: M[name + "@4"] = W4
    print(name, "done", flush=True)
run("nat", l4=True)
run("sign", "sign"); run("sg2", "sign_gain2"); run("sg4", "sign_gain4")
run("G", G=0.5); run("sign+G", "sign", G=0.5)
run("sign+G+seqW", "sign", G=0.5, seqw=1.0)
run("b0.3", lam=0.3); run("sign+G+b0.3", "sign", G=0.5, lam=0.3); run("sign+G+seqW+b0.3", "sign", G=0.5, lam=0.3, seqw=1.0)
M["exl3_2"] = exl3_anchor(X, 2); M["exl3_4"] = exl3_anchor(X, 4)
res = X.capture(M)
for k, v in res.items(): print(f"{k:22s}", v, flush=True)
jdump(res, "results/stack_e36.json")
