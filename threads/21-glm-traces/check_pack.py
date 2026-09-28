"""Check a packed group: windows.jsonl <-> tokens/segments consistency, boundary-array semantics, split disjointness,
manifest sha256s, and no ChatML leftovers in a decoded sample."""
import hashlib, json, sys, random
import numpy as np
sys.path.insert(0, "/home/coder/git/nestquant/threads/21-glm-traces")
import glmfmt as G

g = sys.argv[1]
T = np.load(f"{g}/tokens.npy"); S = np.load(f"{g}/segments.npy")
BT = np.load(f"{g}/bnd_think_d.npy"); BE = np.load(f"{g}/bnd_end_d.npy")
man = json.load(open(f"{g}/manifest.json")); split = json.load(open(f"{g}/split.json"))
rows = [json.loads(l) for l in open(f"{g}/windows.jsonl")]
W, C = T.shape
assert len(rows) == W == man["windows"] and C == man["context"]
for k in ("tokens", "segments", "bnd_think_d", "bnd_end_d"):
    h = hashlib.sha256(open(f"{g}/{k}.npy", "rb").read()).hexdigest()
    assert h == man[f"{k}_sha256"], k
assert not (BT > 0).__and__(BE > 0).any(), "double-counted rows"
n_th = n_en = 0
docs = {"fit": set(), "val": set()}
for w in range(W):
    sp = "fit" if w < split["fit"][1] else "val"
    for s in rows[w]["segments"]:
        o, n = s["window_offset"], s["tokens"]
        assert (S[w, o:o + n] == s["doc_index"]).all()
        docs[sp].add(s["doc_index"])
        t = T[w, o:o + n]; bt = BT[w, o:o + n]; be = BE[w, o:o + n]
        # d=1 rows must be followed (in-segment) by the boundary token of that kind
        for arr, kind in ((bt, "th"), (be, "en")):
            p = np.nonzero(arr == 1)[0]
            assert (p + 1 < n).all()
            nxt = t[p + 1]
            if kind == "th":
                assert (nxt == G.ETHINK).all(); n_th += len(p)
            else:
                assert np.isin(nxt, G.END_TOKS).all(); n_en += len(p)
        # distances decrease by one towards the boundary
        for arr in (bt, be):
            q = np.nonzero(arr > 1)[0]
            assert ((arr[q + 1] == arr[q] - 1) | (arr[q + 1] == 0)).all()
assert not (docs["fit"] & docs["val"])
tok = G.tokenizer()
for w in random.Random(0).sample(range(W), min(200, W)):
    assert "<|im_" not in tok.decode(T[w].tolist(), skip_special_tokens=False)
print(g, "OK", dict(W=W, C=C, fit=split["fit"], val=split["val"], d1_think_rows=n_th, d1_end_rows=n_en,
                    think_rows=int((BT > 0).sum()), end_rows=int((BE > 0).sum())))
