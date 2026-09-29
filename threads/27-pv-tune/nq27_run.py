"""T27 driver: tune one or more experts, write tuned artifacts + receipts (never touches nq-encode-v1).

  python nq27_run.py --arm NAME --objective act|H [--a 0.5] [--steps N] [--lr X] [--tune su,sv,U,V] L:E [L:E ...]

Out: /tmp/nestquant/27-pv-tune/arms/NAME/L{L}/experts/E{E}.pt (+ .json receipt: held-out before/after, history,
layout check, decode check).  Decode check: the saved artifact is re-loaded and decoded with nq_decode at L2/L4;
the result must equal a second decode bitwise (deterministic), and is compared (max rel) with the fp32 tuning model.
"""
import os, sys, json, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "8")
import torch

import nq_decode as D
import nq27_tune as T

SRC = "/tmp/nestquant/src/glm53-fp8"
ENC = "/tmp/nestquant/nq-encode-v1"
OUT = "/tmp/nestquant/27-pv-tune"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True); ap.add_argument("--objective", default="act", choices=["act", "H"])
    ap.add_argument("--a", type=float, default=0.5); ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--eval-every", type=int, default=25); ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--tune", default="su,sv,U,V"); ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--gpu-gb", type=float, default=12); ap.add_argument("--force", action="store_true")
    ap.add_argument("--train-cpu", action="store_true", help="keep train rows in pinned host memory")
    ap.add_argument("--norm", action="store_true", help="level-balanced loss (each level / its untuned error)")
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--gamma", type=float, default=0., help="forced regularizer weight (uniform chunk-0 rows)")
    ap.add_argument("--cap-q", type=float, default=0., help="robust loss: cap row weights at this energy quantile")
    ap.add_argument("--heavy", type=int, default=0, help="always-in-batch top-energy train rows (stratified)")
    ap.add_argument("pairs", nargs="+")
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(a.gpu_gb / 80)
    ap_tf32 = os.environ.get("NQ27_TF32", "0") == "1"      # speed option: TF32 matmuls while tuning (scoring is
    torch.backends.cuda.matmul.allow_tf32 = ap_tf32        # always the separate fp32/bf16 harness eval)
    from orbit_duet.source import weights
    cap = None
    if a.objective == "H":
        import nq19_load
        cap = nq19_load.Capture(root="/tmp/nestquant/19-capture-glmfmt")
    for pr in a.pairs:
        L, E = map(int, pr.split(":"))
        od = f"{OUT}/arms/{a.arm}/L{L}/experts"; os.makedirs(od, exist_ok=True)
        if os.path.exists(f"{od}/E{E}.json") and not a.force:
            continue
        t0 = time.time()
        art = torch.load(f"{ENC}/L{L}/experts/E{E}.pt", weights_only=False, map_location="cpu")
        teacher = [w.to("cuda", torch.float32) for w in weights(SRC, L, E)]
        data = T.ActData(f"{OUT}/rows/L{L}_E{E}.pt", "cuda", max_train=a.max_train or None,
                         train_dev="cpu" if a.train_cpu else None)
        fdata = None
        if a.gamma:
            fr = torch.load(f"{OUT}/frows/L{L}.pt", weights_only=True); fh = (fr["blk"] % 10) == 0
            fdata = (fr["x"][~fh].to("cuda"), fr["x"][fh].to("cuda")); del fr
        Hm = T.load_H(cap, L, E, "cuda") if a.objective == "H" else None
        print(f"[L{L} E{E}] train {data.n_train} held {data.n_hold} objective {a.objective} a {a.a}", flush=True)
        tries = []
        for lr in (a.lr, a.lr / 3, a.lr / 10):     # divergence fallback: held-out never improved -> smaller lr
            M, info = T.tune(art, teacher, data, objective=a.objective, Hm=Hm, a=a.a, steps=a.steps, lr=lr,
                             batch=a.batch, eval_every=a.eval_every, patience=a.patience,
                             tune_set=tuple(a.tune.split(",")), norm=a.norm, warmup=a.warmup, heavy=a.heavy, cap_q=a.cap_q, gamma=a.gamma, fdata=fdata, log=lambda m: print(m, flush=True))
            tries.append(dict(lr=lr, best_step=info["best_step"], held_best=info["held_best"]))
            if info["best_step"] > 0:
                break
            print(f"[L{L} E{E}] no held-out improvement at lr {lr:g}; retrying lower", flush=True)
        info["tries"] = tries; info["lr_used"] = lr
        new = M.write(art)
        T.layout_check(art, new)
        torch.save(new, f"{od}/E{E}.pt.tmp"); os.replace(f"{od}/E{E}.pt.tmp", f"{od}/E{E}.pt")
        re = torch.load(f"{od}/E{E}.pt", weights_only=False, map_location="cpu")
        T.layout_check(art, re)
        dchk = {}
        with torch.no_grad():
            for l in (2, 4):
                d1 = D.decode_expert(re, l, T.DECODE_DEV); d2 = D.decode_expert(re, l, T.DECODE_DEV)
                Wm = M.weights(l)
                dchk[l] = dict(deterministic=all(torch.equal(x, y) for x, y in zip(d1, d2)),
                               max_rel_vs_model=max(float((x.to(y.device) - y).norm() / y.norm()) for x, y in zip(d1, Wm)))
                del d1, d2, Wm
        info.update(layer=L, expert=E, arm=a.arm, objective=a.objective, a=a.a, lr=a.lr, steps=a.steps,
                    batch=a.batch, tune=a.tune, tf32=ap_tf32, decode_check=dchk, layout_identical=True,
                    changed={k: float(v.detach().abs().max()) for k, v in M.named_parameters()},
                    total_s=time.time() - t0)
        json.dump(info, open(f"{od}/E{E}.json", "w"), indent=1)
        print(f"[L{L} E{E}] held L2 {100*info['held0']['2']:.3f} -> {100*info['held_best']['2']:.3f}  "
              f"L4 {100*info['held0']['4']:.3f} -> {100*info['held_best']['4']:.3f}  best@{info['best_step']} "
              f"decode {dchk} ({info['total_s']:.0f}s)", flush=True)
        del M, data, teacher, art, new, re, Hm
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
