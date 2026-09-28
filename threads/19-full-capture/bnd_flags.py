"""Boundary flags for the orbit glm53_training_15m_v2 corpus (lead's final spec, 2026-09-28):
  think boundary b: token 154842 (</think>) not preceded within 3 tokens by 154841 (<think>)  (skips empty stubs)
  end boundary b:   first token of the "<|im_end|>" text sequence (token ending in '<' followed by | im _end | >),
                    or <|endoftext|> (154820)
  positions t with 1 <= b - t <= 32 inside the same window-local segment get (kind, d = b - t);
  a row near both kinds goes to the nearer boundary (ties -> end).
Output: bnd_think_d, bnd_end_d  int8 [windows, 512] (0 = none), saved to OUT/bnd/flags_15m_v2.npz."""
import numpy as np, json, sys
from transformers import AutoTokenizer
import nq19
DMAX = 32
BUCKETS = ((1, 1), (2, 4), (5, 16), (17, 32))
BUCKET_NAMES = ("d1", "d2_4", "d5_16", "d17_32")
KINDS = ("think", "end")


def bucket_of(d):
    b = np.zeros_like(d)
    for i, (lo, hi) in enumerate(BUCKETS):
        b[(d >= lo) & (d <= hi)] = i + 1
    return b


def flags(T, S, tok):
    W, C = T.shape
    lt = np.array([tok.decode([i]).endswith("<") for i in range(len(tok))] + [False] * 4096)
    think = (T == 154842)
    prev = np.zeros_like(think)
    for k in (1, 2, 3):
        prev[:, k:] |= (T[:, :-k] == 154841) & (S[:, :-k] == S[:, k:])
    think &= ~prev
    tail = np.array([91, 318, 6213, 91, 29])
    end = lt[T]
    for j, v in enumerate(tail):
        m = np.zeros_like(end); m[:, :C - j - 1] = T[:, j + 1:] == v; end &= m
    end |= (T == 154820)
    out = {}
    for name, B in (("think", think), ("end", end)):
        d = np.zeros((W, C), np.int16)                  # distance to nearest boundary at or after t (same segment)
        big = np.full((W, C), 10 ** 4, np.int32)
        nxt = np.full(W, -1)
        for t in range(C - 1, -1, -1):                  # sweep backwards: nearest following boundary position
            if t < C - 1:
                pass
            big[:, t] = np.where(nxt >= 0, nxt - t, 10 ** 4)
            nxt = np.where(B[:, t], t, nxt)
            nxt = np.where((t > 0) & (nxt >= 0), nxt, nxt)
        # same-segment check
        ww, tt = np.nonzero(big <= DMAX)
        bpos = tt + big[ww, tt]
        ok = S[ww, tt] == S[ww, bpos]
        d[ww[ok], tt[ok]] = big[ww[ok], tt[ok]]
        out[name] = d
        out[name + "_boundaries"] = B
    th, en = out["think"], out["end"]
    both = (th > 0) & (en > 0)
    th[both & (en <= th)] = 0; en[both & (th < en) & (th > 0)] = 0
    return th.astype(np.int8), en.astype(np.int8), out["think_boundaries"], out["end_boundaries"]


if __name__ == "__main__":
    T = np.load(f"{nq19.CORPUS}/tokens.npy"); S = np.load(f"{nq19.CORPUS}/segments.npy")
    tok = AutoTokenizer.from_pretrained(nq19.SRC)
    th, en, tb, eb = flags(T, S, tok)
    np.savez(f"{nq19.OUT}/bnd/flags_15m_v2.npz", bnd_think_d=th, bnd_end_d=en, think_boundary=tb, end_boundary=eb)
    rep = {}
    for nm, (a, b) in dict(chunk0=(0, 2048), fit=(0, 28656), val=(28656, 28784)).items():
        r = dict(think_boundaries=int(tb[a:b].sum()), end_boundaries=int(eb[a:b].sum()))
        for k, d in (("think", th), ("end", en)):
            bk = bucket_of(d[a:b].astype(np.int16))
            r[k] = {BUCKET_NAMES[i]: int((bk == i + 1).sum()) for i in range(4)}
            r[k]["total"] = int((bk > 0).sum())
        rep[nm] = r
    print(json.dumps(rep, indent=1))
