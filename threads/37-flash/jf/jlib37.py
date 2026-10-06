"""T37 (GLM-5.3-Flash) port of threads/33-search/joint/jlib.py: shared data / feature / eval library for the jF
floating-set predictor.  Minimal diffs vs the GLM-5.3 version:
  NE 256 -> 288, LAYERS 3-77 -> 3-44, NF 77 (k0) -> 48 floating + 19 fixed (REAP, fixed_set.json), chains are
  built from the T37 capture trace (blocks37.py) instead of calib-fit / sm120 block files, and nothing is read at
  import time (serve code imports this module for net_inputs only).
Feature definitions (features, net_inputs, v2 inputs) are byte-for-byte the T33 ones: they are scale-free per layer
(salience is divided by the causal layer salience per hit, targets by m_L), so D 6144 -> 4096 (xn = sum x^2 of the
normalised MoE input) needs no change.  w in the trace includes routed_scaling_factor 2.5 (top-8), as on GLM-5.3.
PRIVATE outputs (traces, token ids, block matrices, features) live under J37_OUT = /tmp/nestquant/37-flash/jf."""
import json
import os

import numpy as np
from scipy.signal import lfilter


def _layers(s):
    a, _, b = s.partition("-")
    return list(range(int(a), int(b or a) + 1))


OUT = os.environ.get("J37_OUT", "/tmp/nestquant/37-flash/jf")
TRACE = os.environ.get("J37_TRACE", "/tmp/nestquant/37-flash/private/dec_trace")
TRACES = [t for t in TRACE.split(":") if t]           # one or more DECODE traces (dec37, online-corpus TF), ':'-separated
PF_TRACE = os.environ.get("J37_PREFILL_TRACE", "")   # optional teacher-forced capture trace, kind "prefill"
PF_FRAC = float(os.environ.get("J37_PREFILL_FRAC", "0.20"))         # prefill share of TRAIN rows (default 80/20)
PF_MAXFRAC = float(os.environ.get("J37_PREFILL_MAXFRAC", "1.0"))    # optional guard (no cap by default)
DEC_PREFILL = os.environ.get("J37_DEC_PREFILL", "0") == "1"        # decode traces' prompt rows -> kind "prefill"
PF_CHUNK = int(os.environ.get("J37_PF_CHUNK", "1024"))   # short (<1K) follow-up prefill chunk: prefill_c<N> upper bound
HANDOFF = os.environ.get("J37_HANDOFF", "1") == "1"       # build prefill->decode handoff eval pairs (val / test)
HO_PROMPT = int(os.environ.get("J37_HO_PROMPT", "4096"))  # prompt-tail tokens folded in by step_chunk (EMA horizon 2048)
HO_DEC = int(os.environ.get("J37_HO_DEC", "16"))          # decode blocks kept after the handoff (metric windows <= this)
HO_WINS = (1, 4, 16)
# chain kind codes: 0 decode, 1 prefill (train / val), 2 handoff-seeded (prompt tail + first decode blocks), 3
# handoff-cold (the same decode blocks alone = fresh predictor at floating_default); 2 / 3 only in val / test, eval only
KINDS = ("decode", "prefill", "handoff", "handoff_cold")
FIXED_JSON = os.environ.get("J37_FIXED", "/tmp/nestquant/37-flash/cap-txt/fixed_set.json")
CORP = os.environ.get("J37_CORP", "/tmp/nestquant/corpus/glm53_calib_glmfmt_v1")
GROUP_ROOT = {"traces": f"{CORP}/c2048_traces", "c2048": f"{CORP}/c2048",
              "mm": "/tmp/nestquant/19-capture-mm/corpus/c2048_mm"}           # = capture37g.windows()
SEQ = 2048
G, NE, TOPK = 16, 288, 8
NF = int(os.environ.get("J37_NF", "48"))            # floating 4-bit slots per layer (96 GB Mac plan: 19 fixed + 48)
NFIX = int(os.environ.get("J37_NFIX", "19"))
LAYERS = _layers(os.environ.get("J37_LAYERS", "3-44"))
THINK_ID, ETHINK_ID = 154841, 154842                # same tokenizer as GLM-5.3
V2 = os.environ.get("J37_V2", f"{OUT}/models/v2_sal_tweedie1.5.txt")
SPLITS = ("train", "val", "test")

_SETS = None
_ML = None


def serve_sets(path=None):
    """-> fixed {L: [ids]}, floating_default {L: [ids]} from a T19-schema fixed_set.json (fixed_set37.py)."""
    global _SETS
    if _SETS is None or path is not None:
        m = json.load(open(path or FIXED_JSON))
        fixed = {int(L): sorted(map(int, v)) for L, v in m["fixed_set"].items()}
        fdef = {int(L): [int(e) for e in v] for L, v in m.get("floating_default", {}).items()}
        _SETS = fixed, fdef
    return _SETS


def masks(L):
    fixed, fdef = serve_sets()
    fx = np.zeros(NE, bool); fx[fixed[L]] = True
    fd = np.zeros(NE, bool); fd[[e for e in fdef.get(L, []) if e not in set(fixed[L])][:NF]] = True
    return fx, fd


def mL(L):
    """per-layer target scale m_L = train mean w^2 xn per routed slot (stored in the v2 meta, like T32)."""
    global _ML
    if _ML is None:
        _ML = {int(k): v for k, v in json.load(open(V2 + ".meta.json"))["sal_norm_mL"].items()}
    return _ML[L]


def mL_blk(L):
    z = np.load(f"{OUT}/blk/train/L{L}.npz")
    return float(z["slot_sal_sum"] / z["slots"])


def load(split, L):
    """-> dict bcnt, bcnta, nans, segl, bsal, sg list of (s, e) chain block ranges (blocks37.py output)."""
    d = np.load(f"{OUT}/blk/{split}/L{L}.npz")
    meta = json.load(open(f"{OUT}/blk/{split}/meta.json"))
    bs = meta["bstart"]
    r = {k: d[k] for k in ("bcnt", "bcnta", "nans", "segl", "bsal")}
    r["sg"] = [(a, b) for a, b in zip(bs[:-1], bs[1:]) if b > a]
    kind = meta.get("kind", [0] * (len(bs) - 1))                     # per chain: KINDS code
    ok = [b > a for a, b in zip(bs[:-1], bs[1:])]
    r["ckind"] = np.array([k for k, o in zip(kind, ok) if o], np.int8)
    r["choff"] = np.array([k for k, o in zip(meta.get("hoff", [0] * len(ok)), ok) if o], np.int32)
    r["chid"] = np.array([k for k, o in zip(meta.get("hid", [-1] * len(ok)), ok) if o], np.int64)
    bk = np.zeros(len(r["bcnt"]), np.int8)
    for k, (a, b) in zip(r["ckind"], r["sg"]):
        bk[a:b] = k
    r["bkind"] = bk
    return r


def kind_sel(d, kind):
    """chain indices (into d['sg']) of one kind ('decode' / 'prefill')."""
    return [i for i, k in enumerate(d["ckind"]) if KINDS[k] == kind]


def pf_keep(k, q):
    """deterministic per-block uniform (hash of the split's global block index) < q: prefill train subsample."""
    k = np.asarray(k, np.int64)
    return ((k * 2654435761) % (1 << 32)) / float(1 << 32) < q


def chunks(d, maxblk=32768):
    """split a load() dict into chain-aligned sub-dicts of <= maxblk blocks (features are per chain: exact)."""
    sg = d["sg"]
    i = 0
    while i < len(sg):
        j = i; s0 = sg[i][0]
        while j < len(sg) and (sg[j][1] - s0 <= maxblk or j == i):
            j += 1
        e0 = sg[j - 1][1]
        sub = {k: d[k][s0:e0] for k in ("bcnt", "bcnta", "nans", "segl", "bsal", "bkind") if k in d}
        sub["sg"] = [(a - s0, b - s0) for a, b in sg[i:j]]
        yield s0, e0, sub
        i = j


def ema(M, h, sg):
    a = 0.5 ** (G / h)
    E = np.empty(M.shape, np.float64)
    for s, e in sg:
        E[s:e] = lfilter([1.0], [1.0, -a], M[s:e], axis=0) * ((1 - a) / G)
    return E


HC = (8, 32, 64, 128, 256, 512, 2048)      # hit EMA half-lives (tokens)
HS = (8, 32, 64, 128, 256, 512, 2048)      # salience EMA half-lives
V2F = ("ema32", "ema128", "mem_cur_state", "tok_since_hit", "hits16", "sema32", "sema128", "sal16", "mps128")


def features(d):
    """-> dict name -> [nb, NE] float32 raw (v2 features exact serve definitions) + extras.  (= T33 jlib.features)"""
    sg = d["sg"]
    bc = d["bcnt"].astype(np.float64); bs = d["bsal"].astype(np.float32).astype(np.float64)
    nb, ne = bc.shape
    F = {}
    Ec = {h: ema(bc, h, sg) for h in HC + ((256,) if 256 not in HC else ())}
    Es = {h: ema(bs, h, sg) for h in HS}
    nrm = Es[256].sum(1) / np.maximum(Ec[256].sum(1), 1e-30)
    nrm = np.where(nrm > 0, nrm, 1.0)[:, None]
    for h in HC:
        F[f"ema{h}"] = Ec[h]
    for h in (32, 128):                                 # serve arithmetic (float32 recursion) for the v2 inputs
        a = np.float32(0.5 ** (G / h))
        E = np.empty(bc.shape, np.float32)
        for s, e in sg:
            E[s:e] = lfilter(np.array([1.0], np.float32), np.array([1.0, -a], np.float32), bc[s:e].astype(np.float32), axis=0)
        F[f"ema{h}"] = E * ((1 - a) / G)
    for h in HS:
        F[f"sema{h}"] = Es[h] / nrm
    F["hits16"] = bc
    F["sal16"] = bs / nrm
    a128 = (1 - 0.5 ** (G / 128)) / G
    F["mps128"] = np.where(Ec[128] / a128 > 1e-3, Es[128] / np.maximum(Ec[128], 1e-30) / nrm, 1.0)
    a512 = (1 - 0.5 ** (G / 512)) / G
    F["mps512"] = np.where(Ec[512] / a512 > 1e-3, Es[512] / np.maximum(Ec[512], 1e-30) / nrm, 1.0)
    k = np.arange(nb)
    tsh = np.empty((nb, ne))
    for s, e in sg:
        last = np.maximum.accumulate(np.where(bc[s:e] > 0, k[s:e, None] - s, -10 ** 6), 0)
        tsh[s:e] = np.minimum(G * (k[s:e, None] - s + 1 - last), 1e5)
    F["tok_since_hit"] = tsh
    ca = d["bcnta"].astype(np.float32); c32 = bc.astype(np.float32)
    na, sl = d["nans"].astype(np.float64), d["segl"]
    sa = 0.5 ** (1 / 2048)
    out = np.empty((nb, ne), np.float32)
    for s, e in sg:
        Et = np.zeros(ne, np.float32); Ea = np.zeros(ne, np.float32); wt = wa = 0.0
        for j in range(s, e):
            nt = G - na[j]; dt = np.float32(sa ** nt); da = np.float32(sa ** na[j])
            Et = Et * dt + (c32[j] - ca[j]); wt = wt * dt + nt; Ea = Ea * da + ca[j]; wa = wa * da + na[j]
            out[j] = Et / max(wt, 1e-6) if sl[j] == 0 else Ea / max(wa, 1e-6)
    F["mem_cur_state"] = out
    pos = np.empty(nb)
    for s, e in sg:
        pos[s:e] = np.arange(e - s)
    F["_pos"] = np.broadcast_to(np.log1p(pos)[:, None], (nb, ne))
    return {n: np.asarray(v, np.float32) for n, v in F.items()}


_BST = {}


def v2_booster(path=None):
    import lightgbm as lgb
    p = path or V2
    if p not in _BST:
        _BST[p] = lgb.Booster(model_file=p)
    return _BST[p]


def v2_pred(F, path=None):
    """v2 GBDT on ALL experts (fixed included, k0-style use as the net's base) -> P [nb, NE] float32."""
    b = v2_booster(path)
    nb, ne = F["ema32"].shape
    X = np.stack([F[n] for n in b.feature_name()], -1).reshape(-1, 9)
    return b.predict(X, num_threads=int(os.environ.get("PT", "1"))).reshape(nb, ne).astype(np.float32)


def future(M, sg, k):
    """sum of blocks b+1..b+k within chain (nan beyond)."""
    Y = np.full(M.shape, np.nan, np.float32)
    for s, e in sg:
        cs = np.vstack([np.zeros((1, M.shape[1])), np.cumsum(M[s:e], 0)])
        kb = np.arange(max(e - s - k, 0))
        Y[s + kb] = cs[kb + 1 + k] - cs[kb + 1]
    return Y


# network input: log-compressed rates.  order fixed = INPUTS
RATE = [f"ema{h}" for h in HC] + [f"sema{h}" for h in HS] + ["hits16", "sal16"]
INPUTS = RATE + ["mem_cur_state", "tok_since_hit", "mps128", "mps512", "_pos", "v2"]


def net_inputs(F, P):
    cols = [np.log1p(np.maximum(F[n], 0) * (16.0 if n not in ("hits16", "sal16") else 1.0)) for n in RATE]
    cols += [np.log1p(np.maximum(F["mem_cur_state"], 0) * 16.0), np.log1p(F["tok_since_hit"]) / 5.0,
             np.log(np.clip(F["mps128"], 1e-3, None)), np.log(np.clip(F["mps512"], 1e-3, None)), F["_pos"] / 5.0,
             np.log(np.clip(P, 1e-4, None))]
    return np.stack(cols, -1).astype(np.float16)


# ------------------------------------------------------------------------------------------------ eval
def replay(S, fx, fd, sg, hm=0.5, ha=0.0, period=1):
    """sync lag-0 serve replay (= t32lib.sim_layer lag 0 / sm120.replay): S [nb, NE] >=0 scores -> serve [nb, NE].
    period > 1: the set may only change every `period` blocks (prefill chunk of period*16 tokens: the predictor
    sees every 16-token sub-block, the set computed after a chunk's last sub-block serves the whole next chunk)."""
    nb = S.shape[0]
    serve = np.zeros((nb, NE), bool)
    for s, e in sg:
        want = fd.copy()
        for k in range(s, e):
            serve[k] = want
            if (k - s + 1) % period:
                continue
            v = np.where(fx, -np.inf, S[k]).astype(np.float32)
            r = want & ~fx
            v = np.where(r, v * np.float32(1 + hm) + np.float32(ha), v)
            if np.where(fx, 0, np.maximum(S[k], 0)).sum() <= 0:
                nw = r
            else:
                nw = np.zeros(NE, bool); nw[np.argsort(-v, kind="stable")[:NF]] = True
            want = nw & ~fx
    return serve


def metric(S, d, L, hm=0.5, ha=0.0, sel=None, period=1):
    """-> (sal-hot numerator, denominator, churn sum, churn count)  pooled over chains (sel: chain indices).
    sal-hot = share of the salience sum_t w^2 xn landing in fixed | served floating (T32/T33 definition)."""
    fx, fd = masks(L)
    sg = d["sg"] if sel is None else [d["sg"][i] for i in sel]
    sv = replay(S, fx, fd, sg, hm, ha, period)
    bs = d["bsal"].astype(np.float64)
    num = den = cs = cn = 0.0
    for s, e in sg:
        h = sv[s:e] | fx
        num += float((bs[s:e] * h).sum()); den += float(bs[s:e].sum())
        cs += float((sv[s + 1:e] & ~sv[s:e - 1]).sum()); cn += e - s - 1
        if s > 0 and sel is None:                       # hot_eval.py convention: churn over ALL block transitions
            cs += float((sv[s] & ~sv[s - 1]).sum()); cn += 1   # (incl. chain-start resets to floating_default)
    return num, den, cs, cn


# ------------------------------------------------------------------------------------------------ handoff
def serve_from(S, fx, want, s, e, hm=0.7, ha=0.0):
    """sync replay of blocks s..e-1 starting from resident set `want` (same update as replay) -> serve [e-s, NE]."""
    serve = np.zeros((e - s, NE), bool)
    want = want.copy()
    for k in range(s, e):
        serve[k - s] = want
        v = np.where(fx, -np.inf, S[k]).astype(np.float32)
        r = want & ~fx
        v = np.where(r, v * np.float32(1 + hm) + np.float32(ha), v)
        if np.where(fx, 0, np.maximum(S[k], 0)).sum() <= 0:
            nw = r
        else:
            nw = np.zeros(NE, bool); nw[np.argsort(-v, kind="stable")[:NF]] = True
        want = nw & ~fx
    return serve


def seed_set(Srow, fx, fd):
    """handoff refresh at hm = 0: top-NF non-fixed of the seeded score (floating_default if the layer has no score)."""
    if np.where(fx, 0, np.maximum(Srow, 0)).sum() <= 0:
        return fd.copy()
    v = np.where(fx, -np.inf, Srow).astype(np.float32)
    w = np.zeros(NE, bool); w[np.argsort(-v, kind="stable")[:NF]] = True
    return w & ~fx


def handoff_pairs(d):
    """-> [(seeded chain idx, cold chain idx, hoff blocks)] (blocks37 handoff chains, matched by pair id)."""
    cold = {int(h): i for i, (k, h) in enumerate(zip(d["ckind"], d["chid"])) if KINDS[k] == "handoff_cold"}
    return [(i, cold[int(h)], int(d["choff"][i])) for i, (k, h) in enumerate(zip(d["ckind"], d["chid"]))
            if KINDS[k] == "handoff" and int(h) in cold]


def handoff(S, d, L, hm=0.7, ha=0.0, wins=HO_WINS):
    """prefill->decode handoff (SPEC sec. 4: layer-major prefill, no refresh inside a prefill): decode sal-hot over the
    first n decode blocks after the handoff.  seeded = step_chunk over the prompt tail (features carry its history),
    one refresh at hm 0 -> set of decode block 0, then the normal sync replay at hm; cold = fresh predictor on the
    decode blocks alone from floating_default.  -> {(variant, n): [num, den, chains]}."""
    fx, fd = masks(L)
    bs = d["bsal"].astype(np.float64)
    out = {(v, n): [0.0, 0.0, 0] for v in ("seeded", "cold") for n in wins}
    for i, j, h in handoff_pairs(d):
        s, e = d["sg"][i]; s2, e2 = d["sg"][j]
        assert e - (s + h) == e2 - s2 and h >= 1, (i, j, h)
        sv = {"seeded": serve_from(S, fx, seed_set(S[s + h - 1], fx, fd), s + h, e, hm, ha),
              "cold": serve_from(S, fx, fd, s2, e2, hm, ha)}
        b0 = {"seeded": s + h, "cold": s2}
        for v, x in sv.items():
            for n in wins:
                if len(x) >= n:
                    B = bs[b0[v]:b0[v] + n]
                    o = out[(v, n)]; o[0] += float((B * (x[:n] | fx)).sum()); o[1] += float(B.sum()); o[2] += 1
    return out
