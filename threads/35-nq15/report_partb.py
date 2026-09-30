"""T35 Part B table: per-arm 4-window KL(BF16 teacher || arm) (fp8 KV emulated), window SD, paired diff vs the b=2.0
matched-memory arm, sal-hot pooled / layer-mean and 4-bit slot share from the harness diag (salstat=1), memory from
sizing.json.  -> /tmp/nestquant/35-nq15/report_partb.json"""
import glob, json
import numpy as np
O = "/tmp/nestquant/35-nq15"
R = f"{O}/e2e/results"
SZ = json.load(open(f"{O}/sizing.json"))
REF = {"nq4": 0.01566, "jF77_hm07": 0.02607, "jF128": 0.02150}
T34 = json.load(open("/tmp/nestquant/34-tr3/report_kvq4.json"))
ARMS = {"e20_77": (2.0, 77), "b20_6": (2.0, 6), "b175_32": (1.75, 32), "b125_71": (1.25, 71), "b15_54": (1.5, 54),
        "b10_86": (1.0, 86), "b15_77": (1.5, 77)}


def load(tags):
    W, D = {}, {}
    for t in tags:
        for p in sorted(glob.glob(f"{R}/{t}/r*.json")):
            j = json.load(open(p)); gids = [int(i) for _, m in j["windows"] for i in m]
            for arm, r in j["results"].items():
                for g, k in zip(gids, r["groups"][0]["win_kl"]):
                    W.setdefault(arm, {})[g] = k
                dg = (r.get("extra") or {}).get("diag")
                if dg:
                    acc = D.setdefault(arm, {})
                    for L, d in dg.items():
                        a = acc.setdefault(int(L), [0.0, 0.0, 0, 0, 0.0, 0])
                        a[0] += d.get("l4_sal", 0); a[1] += d.get("sal_tot", 0); a[2] += d["l4_slots"]
                        a[3] += d["slots"]; a[4] += d["churn_sum"]; a[5] += d["churn_n"]
    return {a: np.array([w[i] for i in range(4)]) for a, w in W.items() if all(i in w for i in range(4))}, D


def mem(b, H):
    base = b + SZ["b2"] - 2.0
    return base * SZ["per_bit_all"] + H * (SZ["b4"] - base) * SZ["per_bit_slot"]


if __name__ == "__main__":
    K, D = load(["passT35B1", "passT35B2"])
    out = {}
    base = K.get("b20_6")
    print(f"{'arm':8} {'b':>5} {'H':>4} {'GiB':>6} {'salP':>6} {'salLm':>6} {'slot4':>6} {'churn':>5} {'KLD':>8} {'SE':>7}"
          f" {'winSD':>7}  {'d vs b20_6 (paired SE)':>24}  w0..w3")
    for a, v in K.items():
        b, H = ARMS.get(a, (None, None))
        d = D.get(a, {})
        salP = sum(x[0] for x in d.values()) / max(sum(x[1] for x in d.values()), 1e-30) if d else float("nan")
        salL = float(np.mean([x[0] / x[1] for x in d.values() if x[1] > 0])) if d else float("nan")
        slot = sum(x[2] for x in d.values()) / max(sum(x[3] for x in d.values()), 1) if d else float("nan")
        ch = float(np.mean([x[4] / max(x[5], 1) for x in d.values()])) if d else float("nan")
        m = mem(b, H) if b else float("nan")
        dd = v - base if base is not None else None
        ds = f"{dd.mean():+.5f} ({dd.std(ddof=1) / 2:.5f}) {(v.mean() / base.mean() - 1) * 100:+.1f}%" if dd is not None else ""
        print(f"{a:8} {b or '':>5} {H or '':>4} {m:6.1f} {salP:6.3f} {salL:6.3f} {slot:6.3f} {ch:5.2f} {v.mean():8.5f}"
              f" {v.std(ddof=1) / 2:7.5f} {v.std(ddof=1):7.5f}  {ds:>24}  " + " ".join(f"{x:.4f}" for x in v))
        out[a] = dict(b=b, H=H, mem_GiB=m, sal_pooled=salP, sal_lmean=salL, slot4=slot, churn=ch, kld=float(v.mean()),
                      se=float(v.std(ddof=1) / 2), win_sd=float(v.std(ddof=1)), win=v.tolist(),
                      d_vs_b20_6=None if dd is None else float(dd.mean()),
                      d_se=None if dd is None else float(dd.std(ddof=1) / 2))
    if "e20_77" in K:
        t = np.array(T34["jF77_hm07"]["kvq"])
        dd = K["e20_77"] - t
        print(f"SANITY e20_77 {K['e20_77'].mean():.5f} vs T34 jF77_hm07 {t.mean():.5f}: diff {dd.mean():+.6f}, "
              f"max |per-window| {np.abs(dd).max():.6f}")
        out["_sanity"] = dict(e20_77=K["e20_77"].tolist(), t34_jF77=t.tolist(), max_abs=float(np.abs(dd).max()))
    if "fp8" in K:
        print(f"fp8 (inline) {K['fp8'].mean():.5f} vs T34 {np.mean(T34['fp8']['kvq']):.5f}")
    json.dump(out, open(f"{O}/report_partb.json", "w"), indent=1)
