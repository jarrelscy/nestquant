# Nested (successively refinable) trellis code inside the full EXL3 pipeline (thread-05 harness).
# Level 2 = native EXL3 K=2 (mul1).  Higher levels replay the frozen base tiles block by block and quantize
# only the residual (with the level's own LDLQ feedback), so level 2 is bit-identical to EXL3-2.
import sys, json, torch
sys.path.insert(0, "/home/coder/git/nestquant/threads/05-exl3-harness")
import harness as h
h.gpu_cap()
ext = h.ExtTileQuantizer("mul1")
L, E = int(sys.argv[1]), int(sys.argv[2])
data = h.load_expert(layer=L, expert=E)

class Recorder:
    def __init__(s, inner, K): s.inner, s.K, s.rec, s.on = inner, K, [], False
    def __call__(s, tiles, K):
        q, i = s.inner(tiles, s.K)
        if s.on: s.rec.append(q.clone())
        return q, i
class Nested:
    """q = base (replayed) + sum_k delta_k * Q_{K_k}(residual / delta_k)  (greedy residual stages)."""
    def __init__(s, base_rec, stages, record=False):
        s.base, s.stages, s.i, s.on, s.record, s.rec, s.ratio = base_rec, stages, 0, False, record, [], []
    def __call__(s, tiles, K):
        if not s.on: raise RuntimeError("scale search must be patched")
        q = s.base[s.i].clone(); s.i += 1
        for Kr, d in s.stages:
            t = tiles - q; sd = t.std().item() + 1e-12; best = None
            for c in (0.8, 0.9, 1.0, 1.1, 1.25):          # per-block residual scale (1 fp16 / 16 rows / level)
                dd = sd / ({2: 1.0, 1: 1.1}[Kr] * c)
                r, _ = ext(t / dd, Kr); m = (t - dd * r).square().mean().item()
                if best is None or m < best[0]: best = (m, dd * r)
            q = q + best[1]
            s.ratio.append(sd)
        if s.record: s.rec.append(q.clone())
        return q, torch.zeros(q.shape, dtype=torch.int16, device=q.device)

orig_gss = h._g_scale_search
out = {}
methods = {}
for proj in range(3):
    pass
def fit_all(make_quant, K, gs_fixed=None):
    Wq, infos = [], []
    for p, name in enumerate(["gate", "up", "down"]):
        H = data.H(p, normalized=False)
        qz = make_quant(p)
        if gs_fixed is not None:
            h._g_scale_search = lambda samples, K_, q_: (gs_fixed[p], 0.0)
        else:
            h._g_scale_search = orig_gss
        # turn recording on only for the LDLQ calls (after scale search)
        _gss = h._g_scale_search
        def gss(samples, K_, q_, _g=_gss, _q=qz):
            r = _g(samples, K_, q_); _q.on = True; return r
        h._g_scale_search = gss
        W, info = h.quantize_exl3_like(data.teacher[p], H, K, count=data.count, quantizer=qz)
        del H; torch.cuda.empty_cache()
        Wq.append(W); infos.append(dict(g_scale=info["g_scale"], proxy=float(info["proxy"])))
    h._g_scale_search = orig_gss
    return Wq, infos

# native controls
recs = {}
def mk_rec(p):
    recs[p] = Recorder(ext, 2); return recs[p]
W2, i2 = fit_all(mk_rec, 2); methods["exl3_2 (=nested L2)"] = W2
gs2 = [x["g_scale"] for x in i2]; print("base g_scale", gs2, "proxy", [x["proxy"] for x in i2], flush=True)
for K in [3, 4]:
    W, inf = fit_all(lambda p: Recorder(ext, K), K); methods[f"exl3_{K}"] = W
    print(f"exl3_{K} proxy", [x["proxy"] for x in inf], flush=True)
# nested 2+2 (level 4, K2 residual stage)
n22 = {}
def mk22(p):
    n22[p] = Nested(recs[p].rec, [(2, 0.2616)]); return n22[p]
W, inf = fit_all(mk22, 4, gs2); methods["nested 2+2 L4"] = W
print("nested 2+2 proxy", [x["proxy"] for x in inf], "resid sd", [sum(n22[p].ratio)/len(n22[p].ratio) for p in range(3)], flush=True)
# nested 2+1 (level 3) and 2+1+1 (level 4)
rec3 = {}
def mk3(p):
    rec3[p] = Nested(recs[p].rec, [(1, 0.238)], record=True); return rec3[p]
W, inf = fit_all(mk3, 3, gs2); methods["nested 2+1 L3"] = W
print("nested 2+1 proxy", [x["proxy"] for x in inf], "resid sd", [sum(rec3[p].ratio)/len(rec3[p].ratio) for p in range(3)], flush=True)
W, inf = fit_all(lambda p: Nested(rec3[p].rec, [(1, 0.124)]), 4, gs2); methods["nested 2+1+1 L4"] = W
print("nested 2+1+1 proxy", [x["proxy"] for x in inf], flush=True)
# control: 2+2 residual code fitted from scratch under 4-bit feedback (base NOT frozen -> not nested)
W, inf = fit_all(lambda p: Nested(None, [(2, 0.2616), (2, 0.0)]), 4, None) if False else (None, None)
ev = h.evaluate(data, methods)
tab = h.table(ev)
for m, v in tab.items(): print(f"{m:32s}", v, flush=True)
json.dump(dict(table=tab), open(f"nested_harness_l{L}_e{E}.json", "w"), indent=1)
