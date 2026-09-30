import time, os
os.environ["LAYOUT"] = "k0"
import numpy as np, plib as P, pfeat as Q, feats as FE
t = time.time(); D = P.load("sm120tf", 40); print("load", round(time.time() - t), D["bcnt"].shape, [e - s for s, e in D["segs"]], flush=True)
n = 20000
D2 = dict(F={k: v[:n] for k, v in D["F"].items()}, bcnt=D["bcnt"][:n], bsal=D["bsal"][:n], segs=[(0, n)], mL=D["mL"])
for fam in ("hks", "hmm", "bou", "kf"):
    t = time.time(); FE.feats(D2, 40, [fam]); print(fam, round(time.time() - t, 1), flush=True)
S = D2["F"]["sema128"]
t = time.time(); P.replay(S, 40, [(0, n)], 0.6); print("replay", round(time.time() - t, 1))
