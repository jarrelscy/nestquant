"""Thread 31 stage 2: per-(layer, expert) upgrade-benefit table (schema nq-delta-v1) + sanity report.  CPU only.

    nq31_table.py [--out /tmp/nestquant/31-delta]

Inputs: stage-1 gains (nq31_gain.py -> OUT/gain/L{L}.npz) and the served layer manifests
(L3-6: T29 refit /tmp/nestquant/29-outlier-gap/nq-encode-h512/L{L}; L7-77: /tmp/nestquant/nq-encode-v1/L{L}).
Writes PUBLIC OUT/delta_table.json + OUT/delta_table.npz (per-expert aggregates e2 e4 drel G delta n_routed, the
definition and the manifest sha256s only; no paths / corpus names: they go to HF serving/predictor/) and PRIVATE
OUT/private/{delta_extra.npz, provenance.json, sanity.json, sanity.txt} (never uploaded).
"""
import argparse, hashlib, json, os, time
import numpy as np

V1 = "/tmp/nestquant/nq-encode-v1"
H512 = "/tmp/nestquant/29-outlier-gap/nq-encode-h512"
LAYERS = list(range(3, 78))
PROJ = ("gate", "up", "down")
BANDS = {"L3-6": range(3, 7), "L7-40": range(7, 41), "L41-77": range(41, 78)}
TOPK = 45
VERSION = 1

DEFINITION = (
    "Per routed expert e of MoE layer L (256 per layer, identical on all TP ranks). "
    "p_{proj,v} = the served layer manifest's per_expert[e].proj.{gate,up,down}.proxy_rot[v] = tr(dW H dW^T)/tr(W H W^T), "
    "the encoder's H-weighted relative error of the bit-exact level-v plane (manifests identified by manifest_sha256; "
    "L3-6 = the h512 refit). "
    "e2, e4 = sqrt(mean over gate/up/down of p_{proj,v}): relative RMS expert-output error at level 2 / level 4. "
    "drel = e2^2 - e4^2: drop in relative output-error energy from upgrading 2 -> 4 bit. "
    "G = E_p2||y||^2 / E_p2||x||^2 = tr(W_d D2 W_d^T) / tr(A2): energy gain of the FP8 teacher expert over the "
    "calibration tokens routed to it, each token weighted by p^2 (p = router weight incl. routed_scaling_factor 2.5, "
    "the same weighting as the serve's benefit sum_t w^2 ||x_t||^2); x = normalized MoE-layer input, h = SwiGLU "
    "output, y = W_d h (before p), A2 = sum_routed p^2 x x^T, D2 = sum_routed p^2 h h^T. Exact traces, no approximation. "
    "delta = drel * G: drop in absolute expert-output error energy per unit input energy. "
    "Serve: benefit_e ~ delta_e * sum_t w_{t,e}^2 ||x_t||^2, ranked within a layer. "
    "n_routed = number of calibration rows routed to e.")


EXTRA_DEF = (
    "Extra columns (private delta_extra.npz): G_plain = tr(W_d D0 W_d^T)/tr(A0) (unweighted routed tokens), "
    "delta_plain = drel*G_plain; G_approx = (sum_ynorm/n)^2 / (tr(C_all)/T_fit) (the mean-||y|| / layer-mean-||x||^2 "
    "fallback, for comparison only); composed variant ec_v^2 = p_down,v + kappa*(p_gate,v + p_up,v): gate/up proxies "
    "are output-weighted (G_gate/G_up) so each ~ the relative h-error energy it causes; that error reaches y through "
    "W_d with relative gain kappa = sum_i M_ii D2_ii / tr(M D2), M = W_d^T W_d (h-noise uncorrelated across channels "
    "with the signal's per-channel energy vs the signal); drel_comp = ec2^2 - ec4^2, delta_comp = drel_comp*G. "
    "reap = sum_routed p ||y|| (T19 sal col 4). Data: T19 text capture stats/L{L}.v25.")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def spearman(a, b):
    ra = np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(float)
    rb = np.argsort(np.argsort(b, kind="stable"), kind="stable").astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def topk(v, k=TOPK):
    return set(np.lexsort((np.arange(len(v)), -np.asarray(v)))[:k].tolist())


def pct(v):
    q = np.percentile(v, [1, 10, 50, 90, 99])
    return dict(zip(("p01", "p10", "p50", "p90", "p99"), [float(x) for x in q]), mean=float(np.mean(v)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/nestquant/31-delta")
    a = ap.parse_args()
    cols = {k: [] for k in ("e2", "e4", "drel", "G", "delta", "n_routed", "G_plain", "delta_plain", "kappa",
                            "e2_comp", "e4_comp", "drel_comp", "delta_comp", "reap", "ymean2", "sum_p2", "Ex2_p2", "Ey2_p2",
                            "Ex2", "Ey2", "G_approx", "p_gate2", "p_up2", "p_down2", "p_gate4", "p_up4", "p_down4")}
    src, Ex2_all = {}, {}
    for L in LAYERS:
        mp = f"{H512 if L <= 6 else V1}/L{L}/manifest.json"
        man = json.load(open(mp))
        assert man["layer"] == L and len(man["per_expert"]) == 256, mp
        g = np.load(f"{a.out}/gain/L{L}.npz")
        P = {v: np.array([[man["per_expert"][str(e)]["proj"][pn]["proxy_rot"][v] for pn in PROJ] for e in range(256)])
             for v in ("2", "4")}
        for v in ("2", "4"):
            assert man["per_expert"]["0"]["proj"]["down"]["bitexact"][v]
        e2, e4 = np.sqrt(P["2"].mean(1)), np.sqrt(P["4"].mean(1))
        kap = g["kap_p2"]
        ec = {v: np.sqrt(P[v][:, 2] + kap * (P[v][:, 0] + P[v][:, 1])) for v in ("2", "4")}
        G = g["Ey2_p2"] / g["Ex2_p2"]; Gp = g["Ey2"] / g["Ex2"]
        drel = e2 ** 2 - e4 ** 2; drc = ec["2"] ** 2 - ec["4"] ** 2
        row = dict(e2=e2, e4=e4, drel=drel, G=G, delta=drel * G, n_routed=g["n"], G_plain=Gp, delta_plain=drel * Gp,
                   kappa=kap, e2_comp=ec["2"], e4_comp=ec["4"], drel_comp=drc, delta_comp=drc * G,
                   reap=g["sum_p_ynorm"], ymean2=g["ymean"] ** 2, sum_p2=g["sp2"], Ex2_p2=g["Ex2_p2"], Ey2_p2=g["Ey2_p2"], Ex2=g["Ex2"],
                   Ey2=g["Ey2"], G_approx=g["ymean"] ** 2 / float(g["Ex2_all"]),
                   **{f"p_{pn}{v}": P[v][:, i] for v in ("2", "4") for i, pn in enumerate(PROJ)})
        for k in cols:
            cols[k].append(np.asarray(row[k], dtype=np.float64))
        Ex2_all[L] = float(g["Ex2_all"])
        src[L] = dict(manifest=mp, manifest_sha256=sha(mp), stats_dir=str(g["stats_dir"]))
        assert (np.isfinite(G) & (G > 0)).all() and (drel > 0).all(), L
    A = {k: np.stack(v) for k, v in cols.items()}                            # [75, 256]
    os.makedirs(f"{a.out}/private", exist_ok=True)

    # ---- table
    main_cols = ("e2", "e4", "drel", "G", "delta", "n_routed")
    # PUBLIC (uploaded to HF serving/predictor/): per-expert aggregates + definition + manifest hashes only.
    tab = dict(schema="nq-delta-v1", version=VERSION, definition=DEFINITION, layers=LAYERS, n_experts=256, columns=list(main_cols),
               manifest_sha256={str(L): src[L]["manifest_sha256"] for L in LAYERS},
               per_layer={str(L): {k: [float(x) for x in A[k][i]] if k != "n_routed" else [int(x) for x in A[k][i]]
                                   for k in main_cols} for i, L in enumerate(LAYERS)})
    json.dump(tab, open(f"{a.out}/delta_table.json.tmp", "w"))
    os.replace(f"{a.out}/delta_table.json.tmp", f"{a.out}/delta_table.json")
    np.savez(f"{a.out}/delta_table.npz", schema=np.array("nq-delta-v1"), version=np.array(VERSION), layers=np.array(LAYERS),
             definition=np.array(DEFINITION), manifest_sha256=np.array([src[L]["manifest_sha256"] for L in LAYERS]),
             **{k: A[k].astype(np.int64) if k == "n_routed" else A[k] for k in main_cols})
    # PRIVATE (not uploaded): extra columns, provenance paths
    np.savez(f"{a.out}/private/delta_extra.npz", layers=np.array(LAYERS), Ex2_all=np.array([Ex2_all[L] for L in LAYERS]),
             **A)
    json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M %Z"), sources=src,
                   teacher="/tmp/nestquant/src/glm53-fp8 (down_proj, bf16-cast)",
                   code="threads/31-delta/nq31_gain.py + nq31_table.py", extra_definition=EXTRA_DEF),
              open(f"{a.out}/private/provenance.json", "w"), indent=1, default=str)

    # ---- sanity
    li = {L: i for i, L in enumerate(LAYERS)}
    rep = dict(bands={}, per_layer={}, spot={})
    for b, Ls in BANDS.items():
        ix = [li[L] for L in Ls]
        rep["bands"][b] = {k: pct(A[k][ix].ravel()) for k in ("e2", "e4", "drel", "G", "delta", "kappa", "e2_comp", "e4_comp",
                                                              "G_plain", "n_routed")}
        rep["bands"][b]["G_over_G_plain"] = pct((A["G"][ix] / A["G_plain"][ix]).ravel())
        rep["bands"][b]["Ey2_over_ymean2"] = pct((A["Ey2"][ix] / A["ymean2"][ix]).ravel())
    for L in LAYERS:
        i = li[L]
        d, dr, reap, nr = A["delta"][i], A["drel"][i], A["reap"][i], A["n_routed"][i]
        stat_b = d * A["Ex2_p2"][i] * A["sum_p2"][i]                       # delta * sum_t p^2 ||x||^2 (static benefit)
        rep["per_layer"][L] = dict(
            sp_delta_drel=spearman(d, dr), sp_delta_reap=spearman(d, reap), sp_delta_nrouted=spearman(d, nr),
            sp_delta_Gplain=spearman(d, A["delta_plain"][i]), sp_delta_comp=spearman(d, A["delta_comp"][i]),
            sp_G_drel=spearman(A["G"][i], dr), sp_delta_Gapprox=spearman(d, dr * A["G_approx"][i]),
            top45_delta_vs_nrouted=TOPK - len(topk(d) & topk(nr)),
            top45_benefit_vs_nrouted=TOPK - len(topk(stat_b) & topk(nr)),
            top45_benefit_vs_reap=TOPK - len(topk(stat_b) & topk(reap)),
            top45_delta_vs_comp=TOPK - len(topk(d) & topk(A["delta_comp"][i])),
            G_cv=float(A["G"][i].std() / A["G"][i].mean()), drel_cv=float(dr.std() / dr.mean()))
    for b, Ls in BANDS.items():
        rep["bands"][b]["per_layer_summary"] = {k: dict(mean=float(np.mean([rep["per_layer"][L][k] for L in Ls])),
                                                        min=float(np.min([rep["per_layer"][L][k] for L in Ls])),
                                                        max=float(np.max([rep["per_layer"][L][k] for L in Ls])))
                                                for k in rep["per_layer"][3]}

    # ---- spot cross-check (measured routed rel-L2 % of the served planes on held-out rows)
    pts = []                                    # (L, e, meas2, meas4, source)
    for L in range(7, 78):
        sp = json.load(open(f"{V1}/spot/L{L}.json"))
        for role, r in sp["experts"].items():
            pts.append((L, r["expert"], r["eval"]["nq/L2"]["all/routed"], r["eval"]["nq/L4"]["all/routed"], "v1spot"))
    for L in range(3, 7):
        d29 = f"{H512}/spot29"
        for fn in sorted(os.listdir(d29)):
            if fn.startswith(f"L{L}_") and not fn.endswith("exl3.jsonl"):
                for ln in open(f"{d29}/{fn}"):
                    r = json.loads(ln)
                    pts.append((L, r["expert"], r["eval"]["new/L2"]["routed"], r["eval"]["new/L4"]["routed"], "spot29"))
    for srcn in ("v1spot", "spot29"):
        P_ = [p for p in pts if p[4] == srcn]
        idx = [(li[p[0]], p[1]) for p in P_]
        m2 = np.array([p[2] for p in P_]); m4 = np.array([p[3] for p in P_])
        o = dict(n=len(P_))
        for est in ("", "_comp"):
            p2 = np.array([100 * A["e2" + est][i, e] for i, e in idx]); p4 = np.array([100 * A["e4" + est][i, e] for i, e in idx])
            md = m2 ** 2 - m4 ** 2; pd = p2 ** 2 - p4 ** 2
            o["rms3" if not est else "comp"] = dict(
                spearman_e2=spearman(p2, m2), spearman_e4=spearman(p4, m4), spearman_drel=spearman(pd, md),
                ratio_meas_over_pred_e2=pct(m2 / p2), ratio_meas_over_pred_e4=pct(m4 / p4),
                ratio_meas_over_pred_drel=pct(md / pd))
            if srcn == "spot29":
                w = {}
                for L in range(3, 7):
                    s = [k for k, p in enumerate(P_) if p[0] == L]
                    w[L] = dict(sp_e2=spearman(p2[s], m2[s]), sp_e4=spearman(p4[s], m4[s]), sp_drel=spearman(pd[s], md[s]),
                                sp_delta=spearman(pd[s] * A["G"][li[L], [P_[k][1] for k in s]],
                                                  md[s] * A["G"][li[L], [P_[k][1] for k in s]]))
                o["rms3" if not est else "comp"]["within_layer"] = w
        rep["spot"][srcn] = o
    json.dump(rep, open(f"{a.out}/private/sanity.json", "w"), indent=1)

    # ---- text summary
    f = lambda x: f"{x:.3g}"
    lines = [f"nq-delta-v1 sanity  ({time.strftime('%Y-%m-%d %H:%M %Z')})", ""]
    for b in BANDS:
        B = rep["bands"][b]
        lines.append(f"[{b}]  p10/p50/p90")
        for k in ("e2", "e4", "drel", "G", "delta", "kappa", "e2_comp", "e4_comp", "G_over_G_plain", "Ey2_over_ymean2"):
            lines.append(f"  {k:16s} {f(B[k]['p10'])} / {f(B[k]['p50'])} / {f(B[k]['p90'])}   (p01 {f(B[k]['p01'])}, p99 {f(B[k]['p99'])})")
        S = B["per_layer_summary"]
        for k in ("sp_delta_drel", "sp_delta_reap", "sp_delta_nrouted", "sp_G_drel", "sp_delta_Gplain", "sp_delta_Gapprox",
                  "sp_delta_comp", "top45_delta_vs_nrouted", "top45_benefit_vs_nrouted", "top45_benefit_vs_reap",
                  "top45_delta_vs_comp", "G_cv", "drel_cv"):
            lines.append(f"  within-layer {k:26s} mean {f(S[k]['mean'])}  [min {f(S[k]['min'])}, max {f(S[k]['max'])}]")
        lines.append("")
    lines.append("per layer: L  sp(delta,drel)  sp(delta,reap)  top45 swaps delta-vs-nrouted  benefit-vs-nrouted")
    for L in LAYERS:
        r = rep["per_layer"][L]
        lines.append(f"  L{L:<3d} {r['sp_delta_drel']:+.3f}  {r['sp_delta_reap']:+.3f}  {r['top45_delta_vs_nrouted']:3d}  {r['top45_benefit_vs_nrouted']:3d}")
    lines.append("")
    for srcn, o in rep["spot"].items():
        for est in ("rms3", "comp"):
            s = o[est]
            lines.append(f"spot {srcn} (n={o['n']}) {est}: spearman e2 {s['spearman_e2']:.3f} e4 {s['spearman_e4']:.3f} "
                         f"drel {s['spearman_drel']:.3f}; meas/pred median e2 {s['ratio_meas_over_pred_e2']['p50']:.3f} "
                         f"e4 {s['ratio_meas_over_pred_e4']['p50']:.3f} drel {s['ratio_meas_over_pred_drel']['p50']:.3f} "
                         f"(e2 p10-p90 {s['ratio_meas_over_pred_e2']['p10']:.2f}-{s['ratio_meas_over_pred_e2']['p90']:.2f})")
            if "within_layer" in s:
                for L, w in s["within_layer"].items():
                    lines.append(f"    L{L} within-layer spearman e2 {w['sp_e2']:.3f} e4 {w['sp_e4']:.3f} drel {w['sp_drel']:.3f} delta {w['sp_delta']:.3f}")
    open(f"{a.out}/private/sanity.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
