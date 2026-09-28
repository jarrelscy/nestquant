"""Distribution of NestQuant L2/L4 deltas vs EXL3 on 48 new experts (lead 2026-09-28).

  python nq_dist48.py select                              # CPU: results_dist48/selection.json
  CUDA_VISIBLE_DEVICES=4 python nq_dist48.py run --part 0/2
  python nq_dist48.py summary

Layers 3,9,16,23,30,36,43,49,56,62,69,76 x 4 experts from the T19 old-corpus stats0 sal: most used (n), median n,
~p10 n (with >= MIN_ROWS routed rows in the matched eval), and the highest boundary-weighted token REAP S_e
(fixed_set19 formula, 50/20/5/2) not already chosen; the 9 v1 experts are excluded.
H = Capture.glm_H (thread-08 recipe, boundary weight 1), EXL3-2/EXL3-4 on the same H, eval = T19 matched capture
(orbit docs: all/control/ood x forced/routed, same splits as results_v1).
Arms: (a) production p4126 single-pass inner0 (lam 0.3); lam0 (same, lam 0) encoded for every expert;
(b) = (a) with the fallback rule: lam0 where (a) all/routed L2 > EXL3-2 + 1.5 %.
"""
import os, sys, json, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "16")
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
T19 = "/home/coder/git/nestquant/threads/19-full-capture"
sys.path.insert(0, T19)
OUT = f"{HERE}/results_dist48"
ROOT = "/tmp/nestquant/19-capture"
LAYERS = (3, 9, 16, 23, 30, 36, 43, 49, 56, 62, 69, 76)
EXCL = {(16, 36), (16, 92), (16, 165), (49, 36), (49, 92), (49, 165), (66, 36), (66, 92), (66, 165)}
MIN_ROWS = 256
BW = {"d1": 50, "d2_4": 20, "d5_16": 5, "d17_32": 2}
PK = {"gate": 2.0, "up": 2.0, "down": 2.3125}
FALLBACK = 1.5


def select():
    import nq19_load, bnd19
    cap = nq19_load.Capture(root=ROOT)
    sel = {}
    for L in LAYERS:
        sal = cap.salience(L)["sums"]
        n = sal[:, 0, 0]
        S = sal[:, 0, 4].copy()
        for ki, _ in enumerate(bnd19.KINDS):
            for bi, b in enumerate(bnd19.BUCKET_NAMES):
                S += (BW[b] - 1) * sal[:, 1 + 4 * ki + bi, 4]
        ids = torch.load(cap.eval_path(L, "matched"), weights_only=True, mmap=True)["ids"]
        rows = np.bincount(ids.flatten().numpy(), minlength=256)
        ok = lambda e: (L, e) not in EXCL and rows[e] >= MIN_ROWS
        chosen = {}
        cand = [e for e in np.argsort(-n) if ok(e)]
        chosen["top_n"] = int(cand[0])
        med = np.median(n)
        chosen["median_n"] = int(min((e for e in range(256) if ok(e) and e not in chosen.values()), key=lambda e: abs(n[e] - med)))
        p10 = np.percentile(n, 10)
        chosen["p10_n"] = int(min((e for e in range(256) if ok(e) and e not in chosen.values()), key=lambda e: abs(n[e] - p10)))
        chosen["top_bnd_reap"] = int(next(e for e in np.argsort(-S) if ok(e) and e not in chosen.values()))
        sel[L] = {r: dict(expert=e, n=float(n[e]), n_pct=float((n < n[e]).mean() * 100), S_e=float(S[e]),
                          S_rank=int((S > S[e]).sum()), reap_mean=float(sal[e, 0, 4] / max(n[e], 1)),
                          eval_routed_rows=int(rows[e])) for r, e in chosen.items()}
        print(L, {r: (v["expert"], int(v["n"]), v["eval_routed_rows"]) for r, v in sel[L].items()}, flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(sel, open(f"{OUT}/selection.json", "w"), indent=1)


def run(part):
    import harness as h
    import nq_encode as NE
    import nq19_load
    i, m = map(int, part.split("/"))
    torch.cuda.set_per_process_memory_fraction(12 / 80)
    torch.backends.cuda.matmul.allow_tf32 = False
    cap = nq19_load.Capture(root=ROOT)
    sel = json.load(open(f"{OUT}/selection.json"))
    jobs = [(int(L), r, v) for L, d in sel.items() for r, v in d.items()]
    for L, role, v in jobs[i::m]:
        E = v["expert"]
        path = f"{OUT}/L{L}_E{E}.json"
        if os.path.exists(path):
            continue
        t0 = time.time()
        dm = cap.expert_data(L, E, "matched")
        HG = cap.glm_H(L, E)
        methods, tm = {}, {}
        for K in (2, 4):
            q = []
            for pi, pn in enumerate(NE.PROJ):
                Wq, _ = h.quantize_exl3_like(dm.teacher[pi], HG["H"][pi], K, count=1, sigma_reg=NE.PROD["sigma"][pn])
                q.append(Wq.cpu()); h.free_scratch()
            methods[f"EXL3-{K}"] = q
        bits = {}
        for tag, lam in (("nq", 0.3), ("nq_lam0", 0.0)):
            torch.cuda.synchronize(); t1 = time.time()
            art, dense = NE.encode_expert(dm.teacher, HG, res_K=PK, canonical_base=False, inner=0, lam=lam, lr=None)
            torch.cuda.synchronize(); tm[tag] = time.time() - t1
            bits[tag] = art["meta"]["rate"]
            for Lv in (2, 4):
                methods[f"{tag}/L{Lv}"] = dense[Lv]
            del art, dense; torch.cuda.empty_cache()
        ev = {}
        names = list(methods)
        for j in range(0, len(names), 3):
            grp = {n_: [w.cuda() for w in methods[n_]] for n_ in names[j:j + 3]}
            tb = h.table(h.evaluate(dm, grp, groups=True))
            ev.update({n_: tb[n_] for n_ in grp})
            del grp; torch.cuda.empty_cache()
        res = dict(layer=L, expert=E, role=role, sel=v, eval=ev, encode_s=tm, rate=bits,
                   hg_meta={k: (float(x) if torch.is_tensor(x) else x) for k, x in HG["meta"].items()},
                   total_s=time.time() - t0)
        json.dump(res, open(path + ".tmp", "w"), indent=1); os.replace(path + ".tmp", path)
        print(f"[L{L} E{E} {role}] {time.time()-t0:.0f}s enc {tm['nq']:.0f}s", flush=True)
        del methods, dm, HG; torch.cuda.empty_cache()


def run_lr(part):
    """Fix arms (low-rank plane NE.LR) on the same H; EXL3 rows reused from the first pass (deterministic, same seed)."""
    import harness as h
    import nq_encode as NE
    import nq19_load
    i, m = map(int, part.split("/"))
    torch.cuda.set_per_process_memory_fraction(12 / 80)
    torch.backends.cuda.matmul.allow_tf32 = False
    cap = nq19_load.Capture(root=ROOT)
    sel = json.load(open(f"{OUT}/selection.json"))
    jobs = [(int(L), r, v) for L, d in sel.items() for r, v in d.items()]
    for L, role, v in jobs[i::m]:
        E = v["expert"]
        path = f"{OUT}/L{L}_E{E}.json"
        R = json.load(open(path))
        if "nq_lr/L4" in R["eval"]:
            continue
        t0 = time.time()
        dm = cap.expert_data(L, E, "matched")
        HG = cap.glm_H(L, E)
        methods = {}
        R.setdefault("lr_meta", {})
        for tag, lam in (("nq_lr", 0.3),):
            torch.cuda.synchronize(); t1 = time.time()
            art, dense = NE.encode_expert(dm.teacher, HG, res_K=PK, canonical_base=False, inner=0, lam=lam, lr=dict(NE.LR))
            torch.cuda.synchronize()
            R["encode_s"][tag] = time.time() - t1
            R["rate"][tag] = art["meta"]["rate"]
            R["lr_meta"][tag] = dict(rank=art["meta"]["lr_rank"],
                                     nnz={p: art[p]["meta"].get("lr", {}).get("nnz") for p in NE.PROJ},
                                     bits={p: art["meta"]["info"][p]["bits"] for p in NE.PROJ},
                                     bitexact={p: art["meta"]["info"][p]["bitexact"] for p in NE.PROJ})
            for Lv in (2, 4):
                methods[f"{tag}/L{Lv}"] = dense[Lv]
            del art, dense; torch.cuda.empty_cache()
        names = list(methods)
        for j in range(0, len(names), 3):
            grp = {n_: [w.cuda() for w in methods[n_]] for n_ in names[j:j + 3]}
            tb = h.table(h.evaluate(dm, grp, groups=True))
            R["eval"].update({n_: tb[n_] for n_ in grp})
            del grp; torch.cuda.empty_cache()
        json.dump(R, open(path + ".tmp", "w"), indent=1); os.replace(path + ".tmp", path)
        print(f"[L{L} E{E} {role}] lr {time.time()-t0:.0f}s enc {R['encode_s']['nq_lr']:.0f}s rank {R['lr_meta']['nq_lr']['rank']}", flush=True)
        del methods, dm, HG; torch.cuda.empty_cache()


KEYS = ("all/routed", "ood/forced", "ood/routed")


def rel(a, b):
    return 100 * (a / b - 1)


def summary():
    import glob
    R = [json.load(open(f)) for f in sorted(glob.glob(f"{OUT}/L*_E*.json"))]
    rows = []
    for r in R:
        e = r["eval"]
        d = {}
        for arm in ("a", "b", "lam0"):
            src = "nq"
            if arm == "lam0" or (arm == "b" and rel(e["nq/L2"]["all/routed"], e["EXL3-2"]["all/routed"]) > FALLBACK):
                src = "nq_lam0"
            for Lv in (2, 4):
                for k in KEYS:
                    d[(arm, Lv, k)] = rel(e[f"{src}/L{Lv}"][k], e[f"EXL3-{Lv}"][k])
            d[(arm, "fb")] = src == "nq_lam0"
        rows.append((r, d))
    print(f"{'expert':<14}{'role':<13}{'n_pct':>6} | {'L2 a: r/oodF/oodR':>22} | {'L4 a':>22} | {'L2 b':>22} | {'L4 b':>22} | enc_s")
    for r, d in rows:
        f = lambda arm, Lv: "/".join(f"{d[(arm, Lv, k)]:+6.2f}" for k in KEYS)
        print(f"L{r['layer']:<2} E{r['expert']:<9}{r['role']:<13}{r['sel']['n_pct']:6.0f} | {f('a', 2)} | {f('a', 4)} | "
              f"{f('b', 2)}{'*' if d[('b', 'fb')] else ' '}| {f('b', 4)} | {r['encode_s']['nq']:.0f}")
    print(f"\nn = {len(rows)}; fallback experts (b): {sum(d[('b', 'fb')] for _, d in rows)}")
    for arm in ("a", "b", "lam0"):
        for Lv in (2, 4):
            for k in KEYS:
                x = np.array([d[(arm, Lv, k)] for _, d in rows])
                print(f"  arm {arm:<4} L{Lv} {k:<11} mean {x.mean():+6.2f} median {np.median(x):+6.2f} p90 {np.percentile(x, 90):+6.2f}"
                      f" worst {x.max():+6.2f}  #worse-than-EXL3 {int((x > 0).sum())}  #>+1.5 {int((x > 1.5).sum())}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("select", "run", "run_lr", "summary"))
    ap.add_argument("--part", default="0/1")
    a = ap.parse_args()
    {"select": select, "run": lambda: run(a.part), "run_lr": lambda: run_lr(a.part), "summary": summary}[a.cmd]()
