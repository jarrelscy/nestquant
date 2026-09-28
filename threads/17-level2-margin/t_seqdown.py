"""Sequential down against quantized hidden, under the thread-08 mixed H (75% uniform tokens)."""
import sys
from core17 import *
split = sys.argv[1]
X = Exp(16, 36, split=split)
gq, uq, dq = fit_expert(X)["W2"]
M = {"nat2": [gq, uq, dq]}
for ridge in [0.1, 0.3, 1.0, 3.0]:
    Ws, Hq, A = seq_down_target(X, gq, uq, ridge)
    M[f"seqW_r{ridge}"] = [gq, uq, quantize(Ws, Hq, sigma=1.0)["W2"]]
    if ridge == 1.0:
        M["seqH"] = [gq, uq, quantize(X.W[2], Hq, sigma=1.0)["W2"]]
        M["seqW_r1_unq"] = [gq, uq, Ws]      # unquantized compensated down (diagnostic)
        M["gu_only"] = [gq, uq, X.W[2]]
res = X.holdout(M) if split == "holdout" else X.capture(M)
for k, v in res.items(): print(f"{split} {k:14s}", v, flush=True)
jdump(res, f"results/seqdown_{split}_e36.json")
