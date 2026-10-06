"""tf_check37.py: teacher-forced parity of dec37.py against capture37g (GPU run; build/compare are CPU-only).

capture37g `final()` saves, for each val segment si of rank R, top64.r{R}.pt[si] = dict(rows=ix, v, i): the top-64
log-probs of the full-sequence (dense-prefill) HF forward, row j predicting token j+1. dec37 `--mode tf` feeds
tokens[:split] through its batched prefill and then tokens[split:] one at a time through its incremental decode
(KDA recurrence + absorbed-MLA/DSA cache). It writes OUT/tf/{id}.npz (v, i [n,64], P = split) with the same row
alignment. Rows < split check the prefill path, rows >= split check the decode path. Routing (ids/w) of the same
rows is optionally compared with the PRIVATE capture trace (trace/L{L}.r{R}of8.npz, rows = rank-R token order).

  # 1. tasks (CPU): val segments of capture rank R, last TAIL tokens of each forced through decode
  /tmp/venv-t37g/bin/python tf_check37.py build --rank 0 --max-segs 64 --tail 256
  # 2. GPU (takes gpu.lock):
  ./run_dec37.sh --mode tf --tasks /tmp/nestquant/37-flash/private/dec_tf/tasks.json --out /tmp/nestquant/37-flash/private/dec_tf --slots 64
  # 3. compare (CPU)
  /tmp/venv-t37g/bin/python tf_check37.py compare --rank 0

Pass bars (expected, bf16 + TF32 vs the capture's bf16 dense path): top-1 agreement >= 0.97 on both paths,
mean KL64 <= 0.01 nats, decode-path KL <= max(1.5x prefill-path KL, 1e-3) (the decode cache adds no extra error),
routing id-set agreement >= 0.97 per MoE layer.
"""
import argparse, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = "/tmp/nestquant/37-flash/cap-txt"
TRACE = "/tmp/nestquant/37-flash/private/trace"
OUT = "/tmp/nestquant/37-flash/private/dec_tf"
W = 8


def load_top(rank, cap):
    import torch
    f = f"{cap}/final/top64.r{rank}.pt"
    if not os.path.exists(f):
        sys.exit(f"missing {f}: capture37g has not run final() yet (cap-txt needs final/)")
    return torch.load(f, weights_only=False)


def build(a):
    sys.path.insert(0, os.path.dirname(HERE))
    import capture37g as C
    data = C.Data(C.windows("txt", 0), a.rank, W)
    top = load_top(a.rank, a.cap)
    sis = sorted(top)[: a.max_segs] if a.max_segs else sorted(top)
    tasks = []
    for si in sis:
        ix = top[si]["rows"].numpy()
        assert np.array_equal(ix, data.segs[si]["ix"]), f"seg {si}: row index mismatch (rank/world differ from capture?)"
        n = len(ix)
        split = max(1, n - a.tail) if n > a.tail + 1 else max(1, n // 2)
        tasks.append(dict(id=f"r{a.rank}s{si}", tokens=data.tok[ix].tolist(), split=int(split), seg=int(si), rows0=int(ix[0])))
    os.makedirs(a.out, exist_ok=True)
    json.dump(tasks, open(f"{a.out}/tasks.json", "w"))
    nt = sum(len(t["tokens"]) for t in tasks)
    print(f"{len(tasks)} tasks, {nt} tokens ({sum(len(t['tokens']) - t['split'] for t in tasks)} decode-path) "
          f"-> {a.out}/tasks.json")


def kl64(vc, ic, vd, id_):
    """KL(cap || dec) on the capture's top-64 support. dec log-probs missing from dec's top-64 are replaced with
    dec's 64th value (an upper bound on the true value), so this is a lower bound on the truncated KL."""
    vc, vd = vc.astype(np.float64), vd.astype(np.float64)
    n = len(vc)
    ld = np.repeat(vd[:, -1:], 64, 1)
    m = ic[:, :, None] == id_[:, None, :]                       # [n, 64c, 64d]
    hit = m.any(2)
    ld[hit] = np.broadcast_to(vd[:, None, :], m.shape)[m]
    pc = np.exp(vc)
    return (pc * (vc - ld)).sum(1), hit.mean(1)


def compare(a):
    top = load_top(a.rank, a.cap)
    tasks = {t["id"]: t for t in json.load(open(f"{a.out}/tasks.json"))}
    res = {"pre": [], "dec": []}
    nmiss = 0
    for tid, t in tasks.items():
        f = f"{a.out}/tf/{tid}.npz"
        if not os.path.exists(f):
            nmiss += 1
            continue
        z = np.load(f)
        c = top[t["seg"]]
        vc, ic = c["v"].float().numpy(), c["i"].numpy()
        vd, id_ = z["v"].astype(np.float32), z["i"]
        P = int(z["P"])
        assert len(vd) == len(vc), f"{tid}: {len(vd)} rows vs capture {len(vc)}"
        kl, cov = kl64(vc, ic, vd, id_)
        t1 = ic[:, 0] == id_[:, 0]
        d1 = np.abs(vc[:, 0] - vd[np.arange(len(vd)), np.argmax(id_ == ic[:, :1], 1)]) * (id_ == ic[:, :1]).any(1)
        for k, sl in (("pre", slice(0, P)), ("dec", slice(P, None))):
            res[k].append(np.stack([kl[sl], t1[sl], cov[sl], d1[sl]], 1))
    print(f"rank {a.rank}: {len(tasks) - nmiss}/{len(tasks)} tasks have tf output")
    ok = True
    summ = {}
    for k in ("pre", "dec"):
        if not res[k]:
            continue
        r = np.concatenate(res[k])
        summ[k] = dict(rows=len(r), kl64=float(r[:, 0].mean()), kl64_p99=float(np.quantile(r[:, 0], 0.99)),
                       top1=float(r[:, 1].mean()), cover=float(r[:, 2].mean()), dlp_top1=float(r[:, 3].mean()))
        print(f"  {k}-path: {summ[k]}")
        ok &= summ[k]["top1"] >= 0.97 and summ[k]["kl64"] <= 0.01
    if "pre" in summ and "dec" in summ:
        ratio = summ["dec"]["kl64"] / max(summ["pre"]["kl64"], 1e-9)
        print(f"  decode/prefill KL ratio {ratio:.2f} (bar: dec KL <= max(1.5 x pre, 1e-3))")
        ok &= summ["dec"]["kl64"] <= max(1.5 * summ["pre"]["kl64"], 1e-3)
    if a.routing and os.path.exists(f"{a.out}/index.json"):
        idx = json.load(open(f"{a.out}/index.json"))
        loc = {q["id"]: q for q in idx["tasks"]}
        Ls = sorted(int(os.path.basename(f)[1:].split(".")[0]) for f in glob.glob(f"{a.trace}/L*.r{a.rank}of{W}.npz"))
        worst = 1.0
        for L in Ls:
            tr = np.load(f"{a.trace}/L{L}.r{a.rank}of{W}.npz")
            ti, tw = tr["ids"], tr["w"].astype(np.float32)
            agg = {"pre": [0, 0], "dec": [0, 0]}
            dw = []
            sh = {}
            for tid, t in tasks.items():
                if tid not in loc:
                    continue
                q = loc[tid]
                if q["g"] not in sh:
                    z = np.load(f"{a.out}/L{L}.r{q['g']}of{idx['W']}.npz")
                    sh[q["g"]] = (z["ids"], z["w"].astype(np.float32))
                di, dwt = sh[q["g"]][0][q["off"]:q["end"]], sh[q["g"]][1][q["off"]:q["end"]]
                rows = np.arange(t["rows0"], t["rows0"] + len(di))
                ci, cw = ti[rows], tw[rows]
                same = (np.sort(ci.astype(np.int64), 1) == np.sort(di.astype(np.int64), 1)).all(1)
                P = t["split"]
                agg["pre"][0] += same[:P].sum(); agg["pre"][1] += P
                agg["dec"][0] += same[P:].sum(); agg["dec"][1] += len(same) - P
                if same.any():
                    oc, od = np.argsort(ci[same], 1), np.argsort(di[same], 1)
                    dw.append(np.abs(np.take_along_axis(cw[same], oc, 1) - np.take_along_axis(dwt[same], od, 1)).max(1))
            fr = {k: v[0] / max(v[1], 1) for k, v in agg.items()}
            worst = min(worst, *fr.values())
            print(f"  L{L}: id-set agree pre {fr['pre']:.4f} dec {fr['dec']:.4f}; mean max|dw| "
                  f"{np.concatenate(dw).mean() if dw else float('nan'):.2e}")
        print(f"  routing worst-layer agreement {worst:.4f} (bar >= 0.97)")
        ok &= worst >= 0.97
    print("TF_CHECK", "PASS" if ok else "FAIL")
    json.dump(summ, open(f"{a.out}/tf_check.r{a.rank}.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "compare"])
    ap.add_argument("--rank", type=int, default=0, help="capture37g rank whose val segments are used")
    ap.add_argument("--cap", default=CAP)
    ap.add_argument("--trace", default=TRACE)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--max-segs", type=int, default=64)
    ap.add_argument("--tail", type=int, default=256, help="tokens per segment forced through the decode path")
    ap.add_argument("--no-routing", dest="routing", action="store_false")
    a = ap.parse_args()
    build(a) if a.cmd == "build" else compare(a)


if __name__ == "__main__":
    main()
