"""T33l fp8dec: dec.py validation task sets + references (PRIVATE, stays on box).
  vA  16 ceiling prefixes (<= 2048: indexer inactive): prompt = corpus window prefix, tf = the real next 64 tokens.
      refs: gen.py decode-path routing of the real continuation (gen.npz), T32 trace routing (prompt tail + cont).
  vB  4 long prompts (3000 tokens spanning windows w, w+1 of glm52-heldout) + tf 64: prefill (tf) vs decode (force)
      path consistency with the indexer ACTIVE; trace routing at the same positions (no KV carry in the trace:
      window w+1 was captured without window w as context).
  mk_valid.py OUT"""
import json
import sys

import numpy as np

sys.path.insert(0, "/home/coder/git/nestquant/threads/33-search/ceiling")
import clib as C  # noqa: E402

T = C.T
OUT = sys.argv[1]
PX = json.load(open("/tmp/nestquant/33-search/ceiling/prefixes.json"))
g = np.load("/tmp/nestquant/33-search/ceiling/gen/gen.npz")
order, K1 = g["order"], int(g["k"]) + 1
sl = g["sparse_layers"].tolist()
corp = {}


def tok(c):
    if c not in corp:
        corp[c] = np.load(f"{T.CORP}/{c}.npy") if hasattr(T, "CORP") else np.load(f"/tmp/nestquant/18-e2e/corpora/{c}.npy")
    return corp[c]


A, refA_gen, metaA = [], [], []
for q in range(16):
    x = PX[order[q]]
    b0 = x["win"] * T.SEQ
    t = tok(x["corpus"])
    pr = t[b0:b0 + x["p"]].astype(int).tolist()
    tf = t[b0 + x["p"]:b0 + x["p"] + 64].astype(int).tolist()
    assert tf == g["tok"][q * K1 + K1 - 1].astype(int).tolist(), "gen real chain != corpus continuation"
    A.append(dict(id=f"vA/{q}", prompt=pr, tf=tf))
    refA_gen.append(g["ids"][:, q * K1 + K1 - 1])            # [75, 64, 8]
    metaA.append((x["corpus"], b0, x["p"]))
json.dump(A, open(f"{OUT}/vA.json", "w"))
B, metaB = [], []
tb = tok("glm52-heldout")
for k, w in enumerate((3, 11, 19, 27)):
    b0 = w * T.SEQ
    B.append(dict(id=f"vB/{k}", prompt=tb[b0:b0 + 3000].astype(int).tolist(), tf=tb[b0 + 3000:b0 + 3064].astype(int).tolist()))
    metaB.append(("glm52-heldout", b0, 3000))
json.dump(B, open(f"{OUT}/vB.json", "w"))
# trace refs: per task [75, P+64, 8] would be large; keep prompt tail 64 + cont 64 (A) and cont 64 (B)
trA = np.zeros((16, len(sl), 128, 8), np.uint8)
trB = np.zeros((4, len(sl), 64, 8), np.uint8)
for j, L in enumerate(sl):
    cache = {}
    for i, (c, b0, p) in enumerate(metaA):
        if c not in cache:
            cache = {c: T.load_layer(L, c)[0]}
        trA[i, j] = cache[c][b0 + p - 64:b0 + p + 64]
    ids = T.load_layer(L, "glm52-heldout")[0]
    for i, (c, b0, p) in enumerate(metaB):
        trB[i, j] = ids[b0 + p:b0 + p + 64]
np.savez(f"{OUT}/vref.npz", refA_gen=np.stack(refA_gen), trA=trA, trB=trB, sl=np.array(sl))
print("wrote", OUT, len(A), len(B))
