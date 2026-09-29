"""Thread 31 stage 1: per-(layer, expert) energy gain of the FP8 teacher expert on its routed calibration tokens.
CPU only.  Writes /tmp/nestquant/31-delta/gain/L{L}.npz (small).

    nq31_gain.py --layers 3-77 [--workers 8 --threads 12]

From the T19 text capture (stats/L{L} -> L{L}.v25, schema nestquant-19-stats-v2), per expert e:
  n       = scalars[e, 0] (routed rows),   sp2 = scalars[e, 2] = sum_routed p^2
  Ex2     = tr(A0_e) / n                   E ||x||^2 over routed rows    (A0 = sum_routed x x^T, diagonal only)
  Ex2_p2  = tr(A2_e) / sp2                 p^2-weighted                  (A2 = sum_routed p^2 x x^T)
  Ey2     = tr(M_e D0_e) / n               E ||y||^2,  y = W_d h,  M_e = W_d^T W_d  (W_d = FP8 teacher down, bf16-cast
  Ey2_p2  = tr(M_e D2_e) / sp2                                          like the capture's y), D0/D2 = sum (p^2) h h^T
  kap / kap_p2 = sum_i M_ii D_ii / tr(M D)  (down's gain on channel-uncorrelated noise with the signal's per-channel
                                              energy, relative to its gain on the signal; used by the e_comp column)
  ymean   = sal[e, all, sum_ynorm] / n     E ||y|| (capture, bf16 y)  -> check Ey2 >= ymean^2
Plus the layer-level Ex2_all = tr(C_all) / T_fit (all fit tokens).
"""
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = "/tmp/nestquant/19-capture-glmfmt"
SRC = "/tmp/nestquant/src/glm53-fp8"
OUT = "/tmp/nestquant/31-delta/gain"


def diag_offsets(n):
    i = np.arange(n, dtype=np.int64)
    return i * n - i * (i - 1) // 2          # row-major upper triangle: (i, i) is the first entry of row i


def one_layer(L, threads):
    import torch
    from safetensors import safe_open
    torch.set_num_threads(threads)
    t0 = time.time()
    vd = os.path.realpath(f"{ROOT}/stats/L{L}")
    m = json.load(open(f"{vd}/meta.json"))
    assert m["schema"] == "nestquant-19-stats-v2" and m.get("complete"), (L, vd)
    raw = m["files"]["raw"]
    mm = {k: np.memmap(f"{vd}/{r['file']}", dtype=np.float32, mode="r", shape=(r["rows"], r["stride_bytes"] // 4))
          for k, r in raw.items() if k in ("A0", "A2", "D0", "D2")}
    sc = np.load(f"{vd}/scalars.npy"); sal = np.load(f"{vd}/sal.npy")
    n, sp2 = sc[:, 0].astype(np.float64), sc[:, 2].astype(np.float64)
    assert np.array_equal(n, sal[:, 0, 0]), "scalars n != sal n"
    o6, o2 = diag_offsets(6144), diag_offsets(2048)
    iu, ju = np.triu_indices(2048)
    wpk = torch.from_numpy(np.where(iu == ju, 1.0, 2.0))                     # tr(M D) from packed D
    iu_t, ju_t = torch.from_numpy(iu), torch.from_numpy(ju)
    C = np.load(f"{vd}/C_all.npy", mmap_mode="r")
    Ex2_all = float(C[o6].astype(np.float64).sum() / m["T_fit"])
    idx = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
    res = {k: np.zeros(256) for k in ("trA0", "trA2", "trMD0", "trMD2", "dMD0", "dMD2", "trM")}
    handles = {}
    for e in range(256):
        res["trA0"][e] = mm["A0"][e, o6].astype(np.float64).sum()
        res["trA2"][e] = mm["A2"][e, o6].astype(np.float64).sum()
        key = f"model.layers.{L}.mlp.experts.{e}.down_proj"
        f = idx[key + ".weight"]
        if f not in handles:
            handles[f] = safe_open(f"{SRC}/{f}", framework="pt", device="cpu")
        h = handles[f]
        w = h.get_tensor(key + ".weight").float()
        s = h.get_tensor(key + ".weight_scale_inv").float()
        W = (w * s.repeat_interleave(128, 0)[:w.shape[0]].repeat_interleave(128, 1)[:, :w.shape[1]])
        W = W.bfloat16().float()                                                # [6144, 2048] = capture teacher y
        assert W.shape == (6144, 2048) and torch.isfinite(W).all()
        M = (W.T @ W).double()
        Mp = M[iu_t, ju_t] * wpk
        Md = torch.diagonal(M)
        res["trM"][e] = float(Md.sum())
        for k in ("D0", "D2"):
            d = torch.from_numpy(np.ascontiguousarray(mm[k][e])).double()
            res[f"trM{k}"][e] = float((Mp * d).sum())
            res[f"dM{k}"][e] = float((Md * d[torch.from_numpy(o2)]).sum())
    out = dict(layer=L, stats_dir=vd, n=n, sp2=sp2,
               Ex2=res["trA0"] / n, Ex2_p2=res["trA2"] / sp2, Ey2=res["trMD0"] / n, Ey2_p2=res["trMD2"] / sp2,
               kap=res["dMD0"] / res["trMD0"], kap_p2=res["dMD2"] / res["trMD2"], trM=res["trM"],
               ymean=sal[:, 0, 5] / n, sum_ynorm=sal[:, 0, 5], sum_p_ynorm=sal[:, 0, 4], Ex2_all=Ex2_all,
               seconds=time.time() - t0)
    tmp = f"{OUT}/L{L}.tmp.npz"
    np.savez(tmp, **{k: np.asarray(v) for k, v in out.items()})
    os.replace(tmp, f"{OUT}/L{L}.npz")
    r = out["Ey2"] / out["ymean"] ** 2
    return f"L{L} {out['seconds']:.0f}s  G med {np.median(out['Ey2'] / out['Ex2']):.4g}  Ey2/ymean^2 [{r.min():.4f},{r.max():.4f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="3-77")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    lo, hi = map(int, a.layers.split("-")) if "-" in a.layers else (int(a.layers),) * 2
    os.makedirs(OUT, exist_ok=True)
    todo = [L for L in range(lo, hi + 1) if a.force or not os.path.exists(f"{OUT}/L{L}.npz")]
    with ProcessPoolExecutor(a.workers) as ex:
        for msg in ex.map(one_layer, todo, [a.threads] * len(todo)):
            print(time.strftime("%H:%M:%S"), msg, flush=True)


if __name__ == "__main__":
    main()
