# Real EXL3 trellis (thread-05 harness, mul1, bit-exact) validation of the down act-order-shard +-d profile.
import sys, json, torch
sys.path.insert(0, "/home/coder/git/nestquant/threads/05-exl3-harness")
import harness as h

def shard_perm(H):
    srt = torch.argsort(H.diagonal()).cpu()
    return torch.cat([srt[j::8] for j in range(8)]).to(H.device)

def prof(Kb, d, nblk=128):  # per 16-block K: shard-local blocks 0-7 (low energy) Kb-d, 8-15 Kb+d
    return [Kb - d if (j % 16) < 8 else Kb + d for j in range(nblk)]

experts = [tuple(map(int, e.split("_"))) for e in (sys.argv[2] if len(sys.argv) > 2 else "16_36,16_92,16_165,49_36").split(",")]
Kbs = [int(k) for k in (sys.argv[3] if len(sys.argv) > 3 else "2,3").split(",")]
sig_list = [float(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["0.03"])]
res = {}
for L, E in experts:
    data = h.load_expert(L, E)
    Hd = data.H(2, normalized=False)
    perm = shard_perm(Hd); inv = torch.argsort(perm)
    Hp = Hd[perm][:, perm].contiguous(); Wd = data.teacher[2]; Wp = Wd[:, perm].contiguous()
    for sig in sig_list:
        for Kb in Kbs:
            (g, u, _), _ = h.quantize_expert_exl3_like(data, {"gate": Kb, "up": Kb, "down": Kb}, sigma_reg=sig)
            h.free_scratch()
            meth = {}; bpw = {}
            dq, inf = h.quantize_exl3_like(Wd, Hd, Kb, count=data.count, sigma_reg=sig); meth["uniform"] = dq; bpw["uniform"] = inf["bpw"]
            variants = [("ashard_uniform", 0), ("ashard_pm0.5", 0.5), ("ashard_pm1", 1)]
            if Kb == 2: variants.append(("ashard_rev_pm0.5", -0.5))
            for nm, d in variants:
                dq, inf = h.quantize_exl3_like(Wp, Hp, prof(Kb, d), count=data.count, sigma_reg=sig)
                meth[nm] = dq[:, inv].contiguous(); bpw[nm] = inf["bpw"]; h.free_scratch()
            ev = h.evaluate(data, {k: [g, u, v] for k, v in meth.items()})
            for nm, dq in meth.items():
                pl = h.proxy_losses(data, [g, u, dq])
                r = dict(bpw_down=bpw[nm], proxy_down=pl["down"],
                         **{f"{dom}_{m}": (ev[dom][m]["router_weighted_relative_l2"][nm] if ev[dom][m] else float("nan")) for dom in ("all", "control", "ood") for m in ("forced", "routed")})
                res[f"L{L}E{E}_s{sig}_K{Kb}_{nm}"] = r
                print(f"L{L}E{E} s{sig} K{Kb} {nm:18s} bpw {r['bpw_down']:.3f} proxy {r['proxy_down']:.5f} "
                      f"routed all {r['all_routed']:.4f} ctl {r['control_routed']:.4f} ood {r['ood_routed']:.4f} forced {r['all_forced']:.4f}", flush=True)
            del meth, g, u; h.free_scratch()
    del data, Hd, Hp, Wp; h.free_scratch()
json.dump(res, open(f"/home/coder/git/nestquant/threads/01-rate-allocation/results/trellis_validate_{'_'.join(map(str,sig_list))}_{sys.argv[2] if len(sys.argv)>2 else 'all'}.json", "w"), indent=1)
