"""Chunked-prefill / router-lookahead report (thread 18), from nq_e2e.py passes run with NQ_LOOKAHEAD=1.

  nq_prefill.py accuracy --tag TAG [--delta delta_table.json] [--parity-out DIR]
      router lookahead accuracy on the NQ_LA_STREAMS streams (default ref = FP8 and nqdef), per band / layer:
        per-token top-8 recall (1- and 2-ahead)
        chunk level (C = 2048 = one window, C = 8192 = 4 consecutive windows of one corpus on one rank), per
        ranking r in {count, w (sum gate weight), sal (sum w^2 |x|^2 delta_e)}, N in {45, 77}, non-fixed experts:
          recall   |top-N(pred) & top-N(act)| / |top-N(act)|        (experts with a positive score only)
          capture  share of the chunk's actual routed slots / gate weight / salience on fixed + top-N(pred),
                   next to the same for fixed + top-N(act) (upper bound)
      --parity-out: per-layer predicted (1-, 2-ahead) and actual top-45 sets (all rankings) for the first C = 2048
      chunk of wikitext and of github, FP8 and nqdef streams; expert ids + aggregate metrics only (public upload).
  nq_prefill.py arms --tag TAG --base nqadapt_chain
      per-arm level-4 share and per-chunk set changes (chunk-lag churn, lookahead upgrades): mean / p95 per layer
"""
import argparse
import glob
import json
import os

import numpy as np

OUT = os.environ.get("NQ_OUT", "/tmp/nestquant/18-e2e")
MAN = os.environ.get("SERVE_MAN", "/tmp/nestquant/28-serve-release/out/serving/tp4/manifest.json")
BANDS = {"L3-6": range(3, 7), "L7-40": range(7, 41), "L41-77": range(41, 78)}
KIND = {"count": 0, "w": 1, "sal": 2}
PARITY_CORPORA = ("wikitext", "github")


def load_parts(tag):
    return [json.load(open(p)) for p in sorted(glob.glob(f"{OUT}/results/{tag}/r*.json"))]


def load_delta(path):
    """T31 table -> {"sal": {L: delta_e}, "salrel": {L: drel_e}} (nq-delta-v1 per_layer columns; legacy {'delta': ..})."""
    if not path or not os.path.exists(path):
        return None
    j = json.load(open(path))
    if "per_layer" in j:
        return {r: {int(L): np.asarray(v[c], np.float64) for L, v in j["per_layer"].items()}
                for r, c in (("sal", "delta"), ("salrel", "drel"), ("cntdelta", "delta"))}
    d = j["delta"]
    it = d.items() if isinstance(d, dict) else enumerate(d)
    return {"sal": {int(L): np.asarray(v, np.float64) for L, v in it if v is not None}}


def vecs(arr, L, delta):
    """[3, ..., 256] (count, sum w, sum w^2|x|^2) -> {ranking: score}"""
    out = {"count": arr[0], "w": arr[1]}
    for r, D in (delta or {}).items():
        out[r] = (arr[0] if r == "cntdelta" else arr[2]) * D[L]
    return out


def topn(score, fixed, n):
    """top-n non-fixed experts with a positive score (ties -> lower id)."""
    s = np.where(fixed, -np.inf, score)
    o = np.argsort(-s, kind="stable")[:n]
    return o[s[o] > 0]


def cmd_accuracy(a):
    parts = load_parts(a.tag)
    m = json.load(open(MAN))
    fixed = {int(L): np.isin(np.arange(256), v) for L, v in m["default_allocation"].items()}
    delta = load_delta(a.delta)
    ranks = ["count", "w"] + sorted(delta or {})
    measures = ["slots"] + ranks[1:]
    streams = sorted({os.path.basename(f).rsplit("_r", 1)[0][3:]
                      for f in glob.glob(f"{OUT}/results/{a.tag}/la_*_r*.npz")})
    res = {"tag": a.tag, "streams": streams, "rankings": ranks, "measures": measures,
           "delta": a.delta if delta is not None else None, "definitions": __doc__}
    parity = {c: {} for c in PARITY_CORPORA}
    for st in streams:
        # per-token recall: from the rank json stats (ref: ref_stats; candidates: results[name].stats)
        tok = {}
        for p in parts:
            S = p["ref_stats"] if st == "ref" else p["results"][st]["stats"]
            ntok = sum(len(w) for _, w in p["windows"]) * p["seq"]
            for k in (1, 2):
                for L, v in S.get(f"la{k}_tok_recall", {}).items():
                    t = tok.setdefault(k, {}).setdefault(int(L), [0.0, 0])
                    t[0] += v * ntok
                    t[1] += ntok
        # chunk-level: acc[(k, C, r, N)][L] = [recall_sum, cap_pred{m}, cap_act{m}, n]
        acc = {}
        for p in parts:
            R = p["rank"]
            f = f"{OUT}/results/{a.tag}/la_{st}_r{R}.npz"
            if not os.path.exists(f):
                continue
            z = np.load(f)
            gpw = p["groups_per_window"]
            win_ids = [(n, w) for n, ws in p["windows"] for w in ws]   # rank-local window order
            layers = sorted({int(k.split("_")[0][1:]) for k in z.files})
            for L in layers:
                act = z[f"L{L}_act"].astype(np.float64)                 # [3, N, 256]
                fx = fixed[L]
                for k in (1, 2):
                    if f"L{L}_p{k}" not in z.files:
                        continue
                    pr = z[f"L{L}_p{k}"].astype(np.float64)
                    for C in (2048, 8192):
                        g = C // p["seq"]
                        chunks = []                                      # lists of window indices
                        w = 0
                        while w < len(gpw):
                            j = w
                            while j < len(gpw) and gpw[j] == gpw[w]:
                                j += 1
                            for c0 in range(w, j - g + 1, g):
                                chunks.append(list(range(c0, c0 + g)))
                            w = j
                        for ch in chunks:
                            A = vecs(act[:, ch].sum(1), L, delta)          # {ranking: [256]}
                            P = vecs(pr[:, ch].sum(1), L, delta)
                            A["slots"] = A["count"]
                            tot = {mm: A[mm].sum() for mm in measures}
                            for r in ranks:
                                for n in (45, 77):
                                    tp, ta = topn(P[r], fx, n), topn(A[r], fx, n)
                                    e = acc.setdefault((k, C, r, n), {}).setdefault(
                                        L, [0.0, {mm: 0.0 for mm in measures}, {mm: 0.0 for mm in measures}, 0])
                                    e[0] += len(np.intersect1d(tp, ta)) / max(len(ta), 1)
                                    for mm in measures:
                                        v = A[mm]
                                        sp = np.zeros(256, bool); sp[tp] = True
                                        sa = np.zeros(256, bool); sa[ta] = True
                                        e[1][mm] += v[fx | sp].sum() / max(tot[mm], 1e-30)
                                        e[2][mm] += v[fx | sa].sum() / max(tot[mm], 1e-30)
                                    e[3] += 1
                            if C == 2048 and win_ids[ch[0]][0] in PARITY_CORPORA and \
                                    win_ids[ch[0]][1] == 0:                  # first chunk of the corpus
                                pc = parity[win_ids[ch[0]][0]].setdefault(st, {})
                                d = pc.setdefault(str(L), {})
                                for r in ranks:
                                    d.setdefault(f"act_top45_{r}", topn(A[r], fx, 45).tolist())
                                    d[f"pred{k}_top45_{r}"] = topn(P[r], fx, 45).tolist()
        out = {"tok_recall": {}, "chunk": {}}
        for k, t in tok.items():
            pl = {L: v[0] / v[1] for L, v in sorted(t.items())}
            out["tok_recall"][f"{k}ahead"] = {
                "bands": {b: float(np.mean([pl[L] for L in rg if L in pl])) for b, rg in BANDS.items()
                          if any(L in pl for L in rg)},
                "per_layer": pl}
        for (k, C, r, n), per in sorted(acc.items()):
            def red(Ls):
                es = [per[L] for L in Ls if L in per]
                if not es:
                    return None
                nn = sum(e[3] for e in es)
                return {"recall": sum(e[0] for e in es) / nn,
                        "capture_pred": {mm: sum(e[1][mm] for e in es) / nn for mm in measures},
                        "capture_act_topN": {mm: sum(e[2][mm] for e in es) / nn for mm in measures},
                        "n_chunk_layers": nn}
            key = f"{k}ahead_C{C}_{r}_top{n}"
            out["chunk"][key] = {"bands": {b: red(rg) for b, rg in BANDS.items()},
                                 "L10_30_50_70": {L: red([L]) for L in (10, 30, 50, 70)}}
        res[st] = out
        print(f"== {st}: per-token top-8 recall "
              + "  ".join(f"{k}: " + " ".join(f"{b} {v:.3f}" for b, v in d["bands"].items())
                          for k, d in out["tok_recall"].items()))
        for key, d in out["chunk"].items():
            print(f"   {key:26s} " + " | ".join(
                f"{b} rec {v['recall']:.3f} cap " + "/".join(f"{v['capture_pred'][mm]:.3f}" for mm in measures)
                + " (act " + "/".join(f"{v['capture_act_topN'][mm]:.3f}" for mm in measures) + ")"
                for b, v in d["bands"].items() if v))
    os.makedirs(f"{OUT}/results", exist_ok=True)
    json.dump(res, open(f"{OUT}/results/{a.tag}_lookahead.json", "w"), indent=1)
    print(f"-> {OUT}/results/{a.tag}_lookahead.json")
    if a.parity_out:
        os.makedirs(a.parity_out, exist_ok=True)
        meta = {"what": "per-layer top-45 non-fixed expert sets for one C = 2048 prefill chunk (the corpus's first "
                        "2048-token window): actual routing vs router lookahead from h_out(L-1) (pred1) / h_out(L-2) "
                        "(pred2) with layer L's post_attention_layernorm + router (sigmoid + e_score_correction_bias, "
                        "noaux_tc top-8)",
                "rankings": {"count": "routed slots", "w": "sum of gate weights (normalised, x routed_scaling_factor)",
                             "sal": "sum w^2 |x|^2 delta_e (delta = drel * G: T31 nq-delta-v1)",
                             "salrel": "sum w^2 |x|^2 drel_e (T31 drel, no gain G)",
                             "cntdelta": "routed slots x delta_e",
                             "x": "normalised MoE input (post_attention_layernorm output, what the experts see)"},
                "streams": {"ref": "FP8 reference", "nqdef": "NestQuant nqdef (26 fixed level-4 / layer, rest "
                                                               "level-2; h512 L3-6)"},
                "fixed_excluded": "manifest default_allocation (26 / layer) excluded from every set",
                "ties": "lower expert id first; experts with zero score never listed", "delta": res["delta"]}
        for c, d in parity.items():
            json.dump({"corpus": c, "chunk": {"window": 0, "tokens": 2048}, **meta, "sets": d},
                      open(f"{a.parity_out}/parity_{c}_C2048.json", "w"), indent=0)
        json.dump(res, open(f"{a.parity_out}/lookahead_accuracy.json", "w"), indent=1)
        print(f"-> {a.parity_out}")


def cmd_arms(a):
    parts = load_parts(a.tag)
    names = list(parts[0]["results"].keys())
    res = {}
    for nm in names:
        S = L4 = 0
        ch, up = {}, {}
        for p in parts:
            ex = p["results"][nm].get("extra") or {}
            for L, v in (ex.get("diag") or {}).items():
                S += v["slots"]; L4 += v["l4_slots"]
                for key, dst in (("churn_hist", ch), ("upg_hist", up)):
                    for val, c in (v.get(key) or {}).items():
                        dst.setdefault(int(L), {}); dst[int(L)][int(val)] = dst[int(L)].get(int(val), 0) + c
        if not S:
            continue

        def summ(h):                                          # {L: {value: count}} -> mean / p95 over all chunks
            allv = {}
            for d in h.values():
                for v, c in d.items():
                    allv[v] = allv.get(v, 0) + c
            if not allv:
                return None
            vs = np.array(sorted(allv)); cs = np.array([allv[v] for v in vs])
            cum = np.cumsum(cs) / cs.sum()
            per_band = {}
            for b, rg in BANDS.items():
                bv = {}
                for L in rg:
                    for v, c in h.get(L, {}).items():
                        bv[v] = bv.get(v, 0) + c
                if bv:
                    x = np.array(sorted(bv)); y = np.array([bv[v] for v in x])
                    per_band[b] = {"mean": float((x * y).sum() / y.sum()),
                                   "p95": int(x[np.searchsorted(np.cumsum(y) / y.sum(), 0.95)])}
            return {"mean": float((vs * cs).sum() / cs.sum()), "p95": int(vs[np.searchsorted(cum, 0.95)]),
                    "n_chunk_layers": int(cs.sum()), "bands": per_band}
        res[nm] = {"l4_share": L4 / S, "set_changes_per_chunk": summ(ch), "upgrades_per_chunk": summ(up)}
        print(f"{nm:16s} l4 share {100 * L4 / S:.1f}%  set changes/chunk/layer "
              f"{res[nm]['set_changes_per_chunk']}  upgrades/chunk/layer {res[nm]['upgrades_per_chunk']}")
    json.dump(res, open(f"{OUT}/results/{a.tag}_arms.json", "w"), indent=1)
    print(f"-> {OUT}/results/{a.tag}_arms.json")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("accuracy")
    s.add_argument("--tag", required=True)
    s.add_argument("--delta", default="/tmp/nestquant/31-delta/delta_table.json")
    s.add_argument("--parity-out")
    s = sub.add_parser("arms")
    s.add_argument("--tag", required=True)
    a = ap.parse_args()
    {"accuracy": cmd_accuracy, "arms": cmd_arms}[a.cmd](a)


if __name__ == "__main__":
    main()
