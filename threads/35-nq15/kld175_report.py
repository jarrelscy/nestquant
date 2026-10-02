"""T35: report a b175 real-layer KLD pass (BF16 teacher, kvq) from nq_e2e per-rank results.
Per window KLD (win_kl), mean over windows, token-pooled KLD, and the hot share of each Adapt arm pooled over
layers (salience-weighted = sum l4_sal / sum sal_tot; routed calls = sum l4_slots / sum slots) plus the layer-mean
of the salience share.  Effective resident bpw at H hot experts/layer: base + H/256 (L4 - base).
  python kld175_report.py RESULTS_DIR [OUT.json]"""
import glob, json, sys

R = sys.argv[1]
parts = [json.load(open(p)) for p in sorted(glob.glob(f"{R}/r*.json"))]
BASE, L4 = 1.775757, 4.141464           # b175 sizing.json: base incl. scales; L4 incl. residual + lowrank
REF = {"TR3 3.42 (EXL3)": 0.0241, "fp8 (T34)": 0.01179, "nq4 2-4 (T34)": 0.01566, "jF128 2-4 (T34)": 0.02150,
       "jF77 2-4 (T34)": 0.02607}
out = {"ranks": len(parts), "kvq": parts[0].get("kvq"), "teacher": parts[0].get("teacher"), "arms": {}}
for arm in parts[0]["results"]:
    win, kls, nt = {}, 0.0, 0
    s = {"slots": 0, "l4_slots": 0, "sal_tot": 0.0, "l4_sal": 0.0}
    lay_share = {}
    for p in parts:
        r = p["results"][arm]
        wins = [w for _, ws in p["windows"] for w in ws]
        for g, w in zip(r["groups"], wins):
            win[w] = g["win_kl"][0] if len(g["win_kl"]) == 1 else g["klsum"] / g["ntok"]
            kls += g["klsum"]; nt += g["ntok"]
        for L, d in (r.get("extra") or {}).get("diag", {}).items():
            for k in s:
                s[k] += d.get(k, 0)
            a = lay_share.setdefault(L, [0.0, 0.0]); a[0] += d.get("l4_sal", 0); a[1] += d.get("sal_tot", 0)
    spec = r["spec"]
    H = next((int(x.split("=")[1]) for x in spec.split(",") if x.startswith("n_float=")), None)
    o = {"spec": spec, "win_kld": {str(k): win[k] for k in sorted(win)},
         "mean_kld": sum(win.values()) / len(win), "pooled_kld": kls / nt, "ntok": nt}
    if s["slots"]:
        o.update(H=H, hot_share_sal=s["l4_sal"] / s["sal_tot"], hot_share_routed=s["l4_slots"] / s["slots"],
                 hot_share_sal_layer_mean=sum(a / b for a, b in lay_share.values() if b) / len(lay_share),
                 eff_bpw=BASE + H / 256 * (L4 - BASE) if H is not None else None)
    out["arms"][arm] = o
out["ref"] = REF
for arm, o in out["arms"].items():
    print(f"{arm:10} mean KLD {o['mean_kld']:.5f} pooled {o['pooled_kld']:.5f} windows "
          + " ".join(f"{w}:{v:.5f}" for w, v in o["win_kld"].items())
          + (f" | H{o['H']} hot sal {o['hot_share_sal']:.3f} (layer-mean {o['hot_share_sal_layer_mean']:.3f})"
             f" routed {o['hot_share_routed']:.3f} eff {o['eff_bpw']:.3f} bpw" if "H" in o else ""))
for k, v in REF.items():
    print(f"  ref {k:20} {v:.5f}")
if len(sys.argv) > 2:
    json.dump(out, open(sys.argv[2], "w"), indent=1)
