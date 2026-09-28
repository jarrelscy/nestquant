"""Thread 25: GLOBAL re-score of the default 4-bit set (lead/user 2026-09-29).  No re-encode: only fixed_set.json changes.

    nq25_rescore.py [--out /tmp/nestquant/25-campaign/rescore/fixed_set_v3.json] [--allow-partial]

score_e = S_e * d_e over all 75 x 256 routed experts, where
  S_e = 0.75 * T_e / sum_all T + 0.25 * V_e / sum_all V       (GLOBAL totals, so layers share one absolute scale)
        T_e = T19 boundary-weighted REAP (fixed_set19 definition: sum_p_ynorm[all] + sum_c (w_c - 1) sum_p_ynorm[c],
              think/end x d1 50, d2_4 20, d5_16 5, d17_32 2), V_e = vision-capture sum_p_ynorm (all rows)
  d_e = e_L2 - e_L4, e_Lv = 100 * sqrt(mean_p proxy_rot_p[Lv])  (relative RMS output-error proxy in %, from the
        uploaded layers' manifest per_expert.proj.*.proxy_rot = tr(dW H dW^T) / tr(W H W^T), the encoder's own
        H-weighted relative error of the bit-exact L2 / L4 planes, H = the 0.75/0.25 text/vision blend it encoded with)
Selection: per-layer floor F (default 16) filled by top S_e in the layer (boundary coverage survives), then global
top-by-score up to BUDGET (default 26 * 75 = 1950) with a per-layer cap C (default 128).  floating_default = the same
rule as fixed_set19 (top-51 NON-fixed experts per layer by n_routed), recomputed so it stays disjoint.
Also reports: coverage (share of routes in the set) all / think_d1 / end_d1 / vision vs the current set, the predicted
error removed sum_{e in set} score_e vs the current set, the d_e cross-check vs the spot numbers (Spearman), and the
flagged layers before/after.  Writes schema nestquant-25-fixed-set-v3 (a superset of v2: fixed_set / floating_default
per layer + definition + the per-expert inputs).
"""
import argparse, glob, hashlib, json, os, sys, time
import numpy as np

T19 = "/home/coder/git/nestquant/threads/19-full-capture"
sys.path.insert(0, T19)
import bnd19

CUR = "/tmp/nestquant/19-capture-glmfmt/fixed_set.json"
TXT = "/tmp/nestquant/19-capture-glmfmt"
VIS = "/tmp/nestquant/19-capture-mm"
ROOT = "/tmp/nestquant/nq-encode-v1"
LAYERS = range(3, 78)
PROJ = ("gate", "up", "down")


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def est(pr, Lv, kind):
    p = np.array([pr[pn]["proxy_rot"][str(Lv)] for pn in PROJ])
    if kind == "rms3":
        return 100 * np.sqrt(p.mean())
    if kind == "down":
        return 100 * np.sqrt(p[2])
    if kind == "gu_down":            # gate/up errors feed down through SwiGLU: rough additive composition
        return 100 * np.sqrt(0.5 * (p[0] + p[1]) + p[2])
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cur", default=CUR)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default="/tmp/nestquant/25-campaign/rescore/fixed_set_v3.json")
    ap.add_argument("--budget", type=int, default=26 * 75)
    ap.add_argument("--floor", type=int, default=16)
    ap.add_argument("--cap", type=int, default=128)
    ap.add_argument("--vision-w", type=float, default=0.25)
    ap.add_argument("--floating", type=int, default=51)
    ap.add_argument("--estimator", default="rms3", choices=("rms3", "down", "gu_down"))
    ap.add_argument("--allow-partial", action="store_true", help="prep: layers without a manifest get d_e = layer-mean of others")
    ap.add_argument("--norm", default="layer", choices=("layer", "global"),
                    help="layer (default, lead 2026-09-29): each side normalized by its own layer total; global: v3-draft absolute units")
    ap.add_argument("--gamma", help="json {L: gamma_L} = RMS(MoE out)/RMS(residual in); default gamma_L = 1 (no norms in the T19 capture)")
    ap.add_argument("--extra-spot", default="/tmp/nestquant/25-campaign/early-lr/multi", help="val-set spot (base arm) jsons")
    ap.add_argument("--complete-only", action="store_true",
                    help="preview: only layers with a full manifest, budget 26 x that many layers (no filled d_e)")
    a = ap.parse_args()
    global LAYERS
    if a.complete_only:
        ok = []
        for L in LAYERS:
            mp = f"{a.root}/L{L}/manifest.json"
            if os.path.exists(mp) and len(json.load(open(mp)).get("per_expert", {})) == 256:
                ok.append(L)
        LAYERS = ok; a.budget = 26 * len(ok)
    cur = json.load(open(a.cur))
    cur_sha = hashlib.sha256(open(a.cur, "rb").read()).hexdigest()
    w = {k: float(v) for k, v in cur["weights"].items()}
    T, V, NR, SAL, VSAL, D, PR = {}, {}, {}, {}, {}, {}, {}
    missing = []
    for L in LAYERS:
        vd = os.path.realpath(f"{TXT}/stats/L{L}")
        assert os.path.basename(vd) == cur["stats_version"][str(L)], (L, vd)
        sal = np.load(f"{vd}/sal.npy")
        S = sal[:, 0, 4].copy()
        for ki, kind in enumerate(bnd19.KINDS):
            for bi, b in enumerate(bnd19.BUCKET_NAMES):
                S += (w[f"{kind}/{b}"] - 1) * sal[:, 1 + 4 * ki + bi, 4]
        assert np.allclose(S, cur["S_e"][str(L)], rtol=1e-5), f"L{L}: S_e recompute != v2 file"
        vv = os.path.realpath(f"{VIS}/stats/L{L}")
        assert os.path.basename(vv) == cur["vision_stats_version"][str(L)], (L, vv)
        vs = np.load(f"{vv}/sal.npy")
        T[L], V[L], NR[L], SAL[L], VSAL[L] = S, vs[:, 0, 4].copy(), sal[:, 0, 0].copy(), sal, vs
        mp = f"{a.root}/L{L}/manifest.json"
        man = json.load(open(mp)) if os.path.exists(mp) else None
        if man is None or len(man.get("per_expert", {})) != 256:
            missing.append(L); continue
        PR[L] = {int(e): v["proj"] for e, v in man["per_expert"].items()}
        D[L] = np.array([est(PR[L][e], 2, a.estimator) - est(PR[L][e], 4, a.estimator) for e in range(256)])
    if missing and not a.allow_partial:
        raise SystemExit(f"layers without a full manifest: {missing}")
    fill = float(np.mean([D[L].mean() for L in D]))
    for L in missing:
        D[L] = np.full(256, fill)
    if a.norm == "global":
        Tt = {L: sum(T[l].sum() for l in LAYERS) for L in LAYERS}; Vt = {L: sum(V[l].sum() for l in LAYERS) for L in LAYERS}
    else:
        Tt = {L: T[L].sum() for L in LAYERS}; Vt = {L: V[L].sum() for L in LAYERS}
    G = {L: 1.0 for L in LAYERS}
    if a.gamma:
        G.update({int(k): float(v) for k, v in json.load(open(a.gamma)).items()})
    SB = {L: (1 - a.vision_w) * T[L] / Tt[L] + a.vision_w * V[L] / Vt[L] for L in LAYERS}
    SC = {L: SB[L] * D[L] * G[L] for L in LAYERS}

    # ---- selection
    sel = {}
    for L in LAYERS:
        sel[L] = [int(e) for e in np.lexsort((np.arange(256), -SB[L]))[:a.floor]]
    pool = [(-SC[L][e], L, e) for L in LAYERS for e in range(256) if e not in set(sel[L])]
    pool.sort()
    n = sum(len(v) for v in sel.values())
    for _, L, e in pool:
        if n >= a.budget:
            break
        if len(sel[L]) < a.cap:
            sel[L].append(int(e)); n += 1
    fixed = {L: sorted(sel[L]) for L in LAYERS}
    old = {L: sorted(int(e) for e in cur["fixed_set"][str(L)]) for L in LAYERS}

    # ---- floating_default: same rule as fixed_set19, recomputed on the new fixed set (disjoint)
    floating = {}
    for L in LAYERS:
        nr = NR[L].copy(); nr[fixed[L]] = -1
        floating[L] = [int(e) for e in np.lexsort((np.arange(256), -nr))[:a.floating]]

    # ---- metrics
    def cov(sets, per_layer=True):
        out = {}
        for name, get in (("all", lambda L: SAL[L][:, 0, 0]), ("think_d1", lambda L: SAL[L][:, 1, 0]),
                          ("end_d1", lambda L: SAL[L][:, 5, 0]), ("vision", lambda L: VSAL[L][:, 0, 0])):
            num = [get(L)[sets[L]].sum() for L in LAYERS]; den = [max(get(L).sum(), 1) for L in LAYERS]
            out[name] = dict(layer_mean=float(np.mean(np.array(num) / np.array(den))), global_=float(sum(num) / sum(den)))
        return out
    err = lambda sets: float(sum(SC[L][sets[L]].sum() for L in LAYERS))
    tot_err = float(sum(SC[L].sum() for L in LAYERS))
    rep = dict(n_total=sum(len(v) for v in fixed.values()), per_layer={L: len(fixed[L]) for L in LAYERS},
               min=min(len(v) for v in fixed.values()), max=max(len(v) for v in fixed.values()),
               disjoint=all(not (set(fixed[L]) & set(floating[L])) for L in LAYERS),
               err_removed=dict(new=err(fixed), cur=err(old), ratio=err(fixed) / err(old), total_possible=tot_err,
                                new_frac=err(fixed) / tot_err, cur_frac=err(old) / tot_err),
               coverage=dict(new=cov(fixed), cur=cov(old)),
               changed_experts=sum(len(set(fixed[L]) ^ set(old[L])) for L in LAYERS) // 2,
               missing_layers=missing, estimator=a.estimator)
    # per-layer error removed + REAP mass kept
    rep["layer_detail"] = {L: dict(n=len(fixed[L]), err_new=float(SC[L][fixed[L]].sum()), err_cur=float(SC[L][old[L]].sum()),
                                   S_share=float(SB[L].sum()), d_mean=float(D[L].mean()), gamma=G[L],
                                   frac_new=float(SC[L][fixed[L]].sum() / SC[L].sum()),
                                   frac_cur=float(SC[L][old[L]].sum() / SC[L].sum())) for L in LAYERS}
    rep["norm"], rep["gamma"] = a.norm, (a.gamma or "1 (T19 capture has no residual-stream norms)")

    # ---- d_e cross-check vs spot numbers (production spot: matched eval; early-lr multi: val eval, base arm)
    xs = []
    for p in sorted(glob.glob(f"{a.root}/spot/L*.json")):
        s = json.load(open(p)); L = int(s["layer"])
        if L not in PR:
            continue
        for role, v in s["experts"].items():
            ev = v["eval"]; k = v.get("flag_key", "all/routed")
            if "nq/L2" in ev and k in ev["nq/L2"]:
                xs.append(dict(L=L, E=v["expert"], src="spot-matched", key=k, rows=v["routed_rows"],
                               spot_d=ev["nq/L2"][k] - ev["nq/L4"][k], est_d=float(D[L][v["expert"]]),
                               spot_L2=ev["nq/L2"][k], est_L2=float(est(PR[L][v["expert"]], 2, a.estimator))))
    for p in sorted(glob.glob(f"{a.extra_spot}/L*-E*.json")):
        s = json.load(open(p)); L = int(s["layer"]); E = int(s["expert"])
        if L in PR and "base/L2" in s["eval"]:
            ev = s["eval"]
            xs.append(dict(L=L, E=E, src="val-multi", key="all/routed", rows=s["routed_rows"],
                           spot_d=ev["base/L2"]["all/routed"] - ev["base/L4"]["all/routed"], est_d=float(D[L][E]),
                           spot_L2=ev["base/L2"]["all/routed"], est_L2=float(est(PR[L][E], 2, a.estimator))))
    if len(xs) >= 3:
        rep["xcheck"] = dict(n=len(xs), spearman_d=spearman([x["spot_d"] for x in xs], [x["est_d"] for x in xs]),
                             spearman_L2=spearman([x["spot_L2"] for x in xs], [x["est_L2"] for x in xs]),
                             pearson_d=float(np.corrcoef([x["spot_d"] for x in xs], [x["est_d"] for x in xs])[0, 1]),
                             ratio_median=float(np.median([x["est_d"] / x["spot_d"] for x in xs if x["spot_d"] > 0])),
                             rows=xs)
        for kind in ("rms3", "down", "gu_down"):
            e2 = [est(PR[x["L"]][x["E"]], 2, kind) - est(PR[x["L"]][x["E"]], 4, kind) for x in xs]
            rep["xcheck"][f"spearman_d_{kind}"] = spearman([x["spot_d"] for x in xs], e2)

    rule = (f"{rep['n_total']} experts across layers 3-77 always level 4 (default allocation): global top by "
            f"S_e x d_e (boundary-weighted REAP 0.75 text + 0.25 vision, global totals, x 2->4-bit relative error drop), "
            f"per-layer floor {a.floor} (top S_e) and cap {a.cap}; the rest level 2 until a runtime allocation overrides")
    doc = dict(schema="nestquant-25-fixed-set-v3", created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               K="variable", budget=a.budget, floor=a.floor, cap=a.cap, rule=rule, estimator=a.estimator,
               prev=dict(path=a.cur, sha256=cur_sha, schema=cur.get("schema")),
               root=cur.get("root"), stats=cur.get("stats"), require=cur.get("require"), shards=cur.get("shards"),
               vision=dict(cur.get("vision") or {}, w=a.vision_w), weights=cur["weights"],
               other_tokens_weight=cur.get("other_tokens_weight", 1.0),
               definition="GLOBAL default 4-bit set (T25 rescore, user-approved 2026-09-29). "
                          "T_e = sum_p_ynorm[all] + sum_c (w_c - 1) sum_p_ynorm[c] over the 8 disjoint boundary categories "
                          "(think/end x d1 50, d2_4 20, d5_16 5, d17_32 2; fixed_set19 definition, sal.npy col 4); "
                          "V_e = vision-capture sum_p_ynorm (all rows). S_e = (1-w) T_e / sum_{all layers,e} T + "
                          "w V_e / sum_{all layers,e} V (w = %g, GLOBAL totals). d_e = e_L2 - e_L4 with e_Lv = 100 sqrt(mean over "
                          "gate/up/down of proxy_rot[Lv]) (manifest per_expert proxy_rot = tr(dW H dW^T)/tr(W H W^T) of the "
                          "uploaded bit-exact planes). score_e = S_e d_e. fixed_set[L] = top-%d by S_e in L (floor), then "
                          "global top by score (ties -> lower layer, lower id) up to %d total with at most %d per layer. "
                          "floating_default = top-%d NON-fixed experts per layer by n_routed (fixed_set19 rule, disjoint)."
                          % (a.vision_w, a.floor, a.budget, a.cap, a.floating),
               fixed_set={str(L): fixed[L] for L in LAYERS}, n_fixed={str(L): len(fixed[L]) for L in LAYERS},
               floating_default={str(L): floating[L] for L in LAYERS},
               S_blend={str(L): [float(f"{v:.7g}") for v in SB[L]] for L in LAYERS},
               d_e={str(L): [float(f"{v:.6g}") for v in D[L]] for L in LAYERS},
               score={str(L): [float(f"{v:.7g}") for v in SC[L]] for L in LAYERS},
               S_e=cur["S_e"], vision_sal=cur["vision_sal"], n_routed=cur["n_routed"],
               stats_version=cur["stats_version"], vision_stats_version=cur["vision_stats_version"],
               coverage={str(L): dict(all=float(SAL[L][fixed[L], 0, 0].sum() / max(SAL[L][:, 0, 0].sum(), 1)),
                                      think_d1=float(SAL[L][fixed[L], 1, 0].sum() / max(SAL[L][:, 1, 0].sum(), 1)),
                                      end_d1=float(SAL[L][fixed[L], 5, 0].sum() / max(SAL[L][:, 5, 0].sum(), 1)),
                                      vision=float(VSAL[L][fixed[L], 0, 0].sum() / max(VSAL[L][:, 0, 0].sum(), 1)))
                         for L in LAYERS},
               report={k: v for k, v in rep.items() if k not in ("xcheck", "layer_detail")})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(doc, open(a.out + ".tmp", "w")); os.replace(a.out + ".tmp", a.out)
    rp = a.out.replace(".json", ".report.json")
    json.dump(rep, open(rp + ".tmp", "w"), indent=1); os.replace(rp + ".tmp", rp)
    print(json.dumps(dict(out=a.out, sha256=hashlib.sha256(open(a.out, "rb").read()).hexdigest(),
                          **{k: v for k, v in rep.items() if k not in ("layer_detail", "per_layer", "xcheck")},
                          xcheck={k: v for k, v in rep.get("xcheck", {}).items() if k != "rows"}), indent=1))


if __name__ == "__main__":
    main()
