"""Stage profile of thread 12's production single-pass encoder (p4126 inner0) on the T19 stats0 H.

  CUDA_VISIBLE_DEVICES=4 python prof_ref.py 16:36,49:92 [--out JSON]

Every stage is timed with a device sync around it (exclusive times; nested stages are subtracted from their parent).
"""
import os, sys, time, json, argparse, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import torch

T = collections.defaultdict(float); N = collections.defaultdict(int)
_stack = []


def timed(tag, f):
    def g(*a, **k):
        torch.cuda.synchronize(); t = time.time(); _stack.append(0.0)
        r = f(*a, **k)
        torch.cuda.synchronize(); dt = time.time() - t
        child = _stack.pop()
        T[tag] += dt - child; N[tag] += 1
        if _stack:
            _stack[-1] += dt
        return r
    return g


def install():
    import nq_encode as NE, nq_decode as D, harness as h, nq_patvit as PV
    Qm = h._ex()
    _v = NE.viterbi

    def vit(rings, K):
        N[f"rings K{K}"] += rings.shape[0]
        return timed(f"viterbi K{K}", _v)(rings, K)
    NE.viterbi = vit
    for n in ("prep", "base_quant", "encode_rotated", "pack", "ldl_blocks", "refit_dense", "mbn_candidates"):
        setattr(NE, n, timed(n, getattr(NE, n)))
    h._g_scale_search = timed("g_scale_search", h._g_scale_search)
    for n in ("prepare_H_out", "refit_scales", "unrotate_H", "sample_scale_tiles"):
        setattr(Qm, n, timed(n, getattr(Qm, n)))
    D.rotated_levels = timed("check: rotated_levels", D.rotated_levels)
    D.decode_matrix = timed("check: decode_matrix", D.decode_matrix)
    D.dense_from_rotated = timed("dense_from_rotated", D.dense_from_rotated)
    D.fold = timed("fold (LDLQ candidates)", D.fold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experts")
    ap.add_argument("--out")
    a = ap.parse_args()
    C.setup()
    import nq_encode as NE
    install()
    res = []
    for spec in a.experts.split(","):
        L, E = map(int, spec.split(":"))
        T.clear(); N.clear()
        torch.cuda.synchronize(); t0 = time.time()
        W = C.teacher(L, E); torch.cuda.synchronize(); t_w = time.time() - t0
        t1 = time.time(); HG, flags = C.load_HG(L, E); torch.cuda.synchronize(); t_h = time.time() - t1
        t2 = time.time()
        art, _ = NE.encode_expert(W, HG, **C.PROD_KW)
        torch.cuda.synchronize(); t_enc = time.time() - t2
        tot = time.time() - t0
        st = dict(load_teacher=t_w, load_H=t_h, **{k: v for k, v in T.items()})
        st["encode_expert other (python/glue)"] = t_enc - sum(T.values())
        r = dict(layer=L, expert=E, total=tot, encode=t_enc, stages={k: round(v, 2) for k, v in sorted(st.items(), key=lambda x: -x[1])},
                 calls=dict(N), maxmem_gb=torch.cuda.max_memory_allocated() / 2**30,
                 per_proj={p: round(art["meta"]["info"][p]["time"], 1) for p in NE.PROJ})
        print(json.dumps(r, indent=1), flush=True)
        res.append(r)
        del art, W, HG; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    if a.out:
        json.dump(dict(results=res, shas=C.ref_shas()), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
