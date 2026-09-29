"""T27: effective sample size of the router-weighted output metric per expert (CPU only).

  python nq27_ess.py L:E [L:E ...]   -> /tmp/nestquant/27-pv-tune/val/ess/L{L}_E{E}.json
Row weight w_i = p_i^2 ||y_T(x_i)||^2 (the harness 'routed' metric's energy per row, teacher bf16);
ESS = (sum w)^2 / sum w^2; also the energy share of the top 1 / top 10 rows.  Computed on eval/val (the scoring rows)
and on the chunk-0 train / held-out split used by the tuner.
"""
import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import nq27_tune as T
from orbit_duet.source import weights

SRC = "/tmp/nestquant/src/glm53-fp8"; R = "/tmp/nestquant/nq-encode-v1"; OUT = "/tmp/nestquant/27-pv-tune"


def stats(x, p, tb):
    w = []
    for i in range(0, len(x), 4096):
        w.append(T.teacher_bf16(x[i:i + 4096], tb).double().square().sum(-1) * p[i:i + 4096].double().square())
    w = torch.cat(w); s = w.sort(descending=True).values
    return dict(n=len(w), ess=float(w.sum() ** 2 / w.square().sum()), top1=float(s[0] / s.sum()),
                top10=float(s[:10].sum() / s.sum()))


def main():
    torch.set_num_threads(16)
    for pr in sys.argv[1:]:
        L, E = map(int, pr.split(":"))
        o = f"{OUT}/val/ess/L{L}_E{E}.json"
        if os.path.exists(o):
            continue
        tb = [w.bfloat16() for w in weights(SRC, L, E, device="cpu")]
        cap = torch.load(f"{R}/_stats/eval/val/layer_{L}.pt", weights_only=True, mmap=True)
        routed, slots = torch.where(cap["ids"] == E)
        res = dict(val=stats(cap["x"][routed].bfloat16(), cap["p"][routed, slots], tb))
        r = torch.load(f"{OUT}/rows/L{L}_E{E}.pt", weights_only=True)
        hold = (r["blk"] % 10) == 0
        res["train"] = stats(r["x"][~hold], r["p"][~hold], tb); res["held"] = stats(r["x"][hold], r["p"][hold], tb)
        os.makedirs(os.path.dirname(o), exist_ok=True); json.dump(res, open(o, "w"), indent=1)
        print(L, E, {k: (v["n"], round(v["ess"], 1)) for k, v in res.items()}, flush=True)


if __name__ == "__main__":
    main()
