from core17 import *
X = Exp(16, 36)
M = {}
M["nat2"] = fit_expert(X)["W2"]
for sw in [1, 3]:
    r = fit_expert(X, cd=make_cd(sw, log=print)); M[f"cd{sw}"] = r["W2"]
    print(sw, [i["cd"] for i in r["info"]], flush=True)
res = X.capture(M)
for k, v in res.items(): print(f"{k:14s}", v, flush=True)
jdump(res, "results/cd_e36.json")
