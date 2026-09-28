"""NestQuant v1 production entry point: encode every routed expert of one GLM layer -> per-TP8-shard planes + manifest.

  python nq_layer.py --layer L --stats /tmp/nestquant/19-capture --out /tmp/nestquant/nq-encode [--experts 0:256]
                     [--res-k 2,2,2.3125 | --rate R] [--bnd 50] [--source $NQ19_SRC] [--check-decode] [--no-finalize]

Per expert (nq_encode.encode_expert): H/G from the thread-19 capture (nq19_load.Capture(root).glm_H, thread-08 recipe),
base = K2 mul1 + per-ring sign + two-sided G + blend 0.3 (production: ONE joint pass, inner 0; --canonical in nq_encode = 2-pass K2-reference base), then the
P4 plane re-fit on the frozen base with the production residual: pattern K per projection (default gate/up 2
(uniform K2), down 2.3125 = (2, 0x9248) -> 4.1263 bpw; --res-k; 1.9375 = (1, 0xFFFE) available), or --rate R = positional K2.5 on the last-processed
units of every shard. H/G boundary weighting per --bnd is OFF by default (weight 1 = thread-08 recipe; user decision 2026-09-29).
Resumable: finished experts are kept in OUT/L{L}/experts/E{E}.pt; the final step splits them into
  OUT/L{L}/tp{s}.pt   s = 0..7   {E: {proj: {base, var, p4, word, suh, svh}}}  (only the bytes shard s reads)
  OUT/L{L}/manifest.json         format, config, per-shard sha256 + bytes, per-expert bits / proxies / checks
Scales per shard: gate/up (sharded on out) keep the full input suh and the shard's svh slice; down (sharded on in) keeps
the shard's suh slice and the full svh. Level 2 reads base (+ its suh/svh); level 4 reads base + p4 (+ p4 suh/svh).
`assemble(dir, L, E)` rebuilds the nq_decode artifact from the shard files (decode check).
"""
import os, sys, json, time, hashlib, argparse
os.environ.setdefault("OMP_NUM_THREADS", "16")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nq_decode as D
import nq_encode as NE
import nq_bnd as NB
import harness as h

T19 = "/home/coder/git/nestquant/threads/19-full-capture"
NSH = 8


FIXED_RULE = dict(rule="top 26 experts per layer by token-weighted REAP = always level 4 (default allocation); "
                       "the rest level 2 until a runtime allocation overrides",
                  n=26, reap_token_weights={"d1": 50, "d2_4": 20, "d5_16": 5, "d17_32": 2, "other": 1},
                  window="32 tokens before a non-empty </think> or <|im_end|>; row -> nearer boundary, ties -> end")


def default_allocation(a, L):
    """Manifest default allocation from T19's fixed_set.json (schema reserved even when the file is not there yet)."""
    path = a.fixed_set or f"{a.stats}/fixed_set.json"
    out = dict(FIXED_RULE, source=path, level4_experts=None, status="pending (fixed_set.json not found)")
    if os.path.exists(path):
        fs = json.load(open(path))           # T19 fixed_set19.py schema nestquant-19-fixed-set-v1
        ent = fs.get("fixed_set", {}).get(str(L))
        out.update(level4_experts=sorted(int(e) for e in ent) if ent is not None else None, sha256=sha(path),
                   schema=fs.get("schema"), K=fs.get("K"), reap_token_weights_file=fs.get("weights"),
                   other_tokens_weight=fs.get("other_tokens_weight"), definition=fs.get("definition"),
                   stats_version=fs.get("stats_version", {}).get(str(L)),
                   status="ok" if ent is not None else f"layer {L} missing in {path}")
    return out


def teacher(source, L, E):
    from orbit_duet.source import weights
    return [w.float() for w in weights(source, L, E)]


def shard_slices(pn, k, n):
    """(suh slice, svh slice) of TP shard s for projection pn (k = in, n = out)."""
    if pn == "down":
        return lambda s: (slice(s * k // NSH, (s + 1) * k // NSH), slice(0, n))
    return lambda s: (slice(0, k), slice(s * n // NSH, (s + 1) * n // NSH))


def split_expert(art):
    """artifact -> [NSH] {proj: plane bytes of shard s}"""
    out = [dict() for _ in range(NSH)]
    for pn in NE.PROJ:
        P = art[pn]; m = P["meta"]
        sl = shard_slices(pn, m["k"], m["n"])
        for s in range(NSH):
            ks, ns = sl(s)
            d = dict(base=P["base"]["shards"][s], p4=P["p4"]["shards"][s], word=P["p4"]["word"][s].to(torch.int32),
                     suh2=P["base"]["suh"][ks], svh2=P["base"]["svh"][ns], suh4=P["p4"]["suh"][ks], svh4=P["p4"]["svh"][ns])
            if P["base"].get("var"):
                d["var"] = P["base"]["var"][s]
            out[s][pn] = d
    return out


def assemble(root, L, E):
    """Rebuild the nq_decode artifact of expert E from the TP shard files (+ manifest meta)."""
    man = json.load(open(f"{root}/L{L}/manifest.json"))
    parts = [torch.load(f"{root}/L{L}/tp{s}.pt", weights_only=False)[E] for s in range(NSH)]
    art = {}
    for pn in NE.PROJ:
        meta = dict(man["proj_meta"][pn])
        k, n = meta["k"], meta["n"]
        cat = lambda key: [p[pn][key] for p in parts]
        if pn == "down":
            suh2, svh2 = torch.cat(cat("suh2")), parts[0][pn]["svh2"]
            suh4, svh4 = torch.cat(cat("suh4")), parts[0][pn]["svh4"]
        else:
            suh2, svh2 = parts[0][pn]["suh2"], torch.cat(cat("svh2"))
            suh4, svh4 = parts[0][pn]["suh4"], torch.cat(cat("svh4"))
        base = dict(shards=cat("base"), suh=suh2, svh=svh2)
        if "var" in parts[0][pn]:
            base["var"] = cat("var")
        art[pn] = dict(base=base, p4=dict(shards=cat("p4"), word=cat("word"), suh=suh4, svh=svh4), meta=meta)
    return art


def sha(path):
    hh = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            hh.update(b)
    return hh.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--stats", default="/tmp/nestquant/19-capture", help="thread-19 capture root (nq19_load.Capture)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--experts", default="0:256")
    ap.add_argument("--rate", type=float, help="positional-K2.5 rate instead of the pattern default (e.g. 4.15625)")
    ap.add_argument("--source", default=os.environ.get("NQ19_SRC", "/tmp/nestquant/src/glm53-fp8"), help="full FP8 checkpoint or per-expert dir")
    ap.add_argument("--res-k", help="pattern residual 'gate,up,down' K; default nq_encode.PROD res_K = 2,2,2.3125 "
                                    "(4.1263 bpw); 1.9375,1.9375,2.3125 = 4.0846 (fails the 9-expert rule)")
    ap.add_argument("--bnd", default=str(NB.DEFAULT_BND),
                    help="boundary-window weight, default 1 = OFF (user 2026-09-29); flat ('50') or per bucket "
                         "'think:d1=50,end:d17_32=4,*=10' (kinds think/end, buckets d1,d2_4,d5_16,d17_32)")
    ap.add_argument("--bnd-k", type=float, default=NB.DEFAULT_K, help="ESS shrink: w_eff = 1+(w-1)ESS/(ESS+k)")
    ap.add_argument("--bnd-cap", type=float, default=NB.DEFAULT_CAP, help="max trace share of the boundary increment")
    ap.add_argument("--fixed-set", help="T19 fixed_set.json (default STATS/fixed_set.json): the always-4-bit default allocation")
    ap.add_argument("--check-decode", action="store_true", help="decode every expert back from the shard files")
    ap.add_argument("--no-finalize", action="store_true", help="only encode (e.g. several workers on disjoint ranges)")
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(12 / 80)
    torch.backends.cuda.matmul.allow_tf32 = False
    sys.path.insert(0, T19)
    import nq19_load
    cap = nq19_load.Capture(root=a.stats)
    L = a.layer
    bw = NB.parse_bnd(a.bnd)
    d = f"{a.out}/L{L}"; ed = f"{d}/experts"
    os.makedirs(ed, exist_ok=True)
    e0, e1 = map(int, a.experts.split(":"))
    for E in range(e0, e1):
        path = f"{ed}/E{E}.pt"
        if os.path.exists(path):
            continue
        t0 = time.time()
        HG = NB.glm_H_bnd(cap, L, E, bw, k=a.bnd_k, cap_frac=a.bnd_cap)
        flags = []
        for i, Hm in enumerate(HG["H"]):
            if not torch.isfinite(Hm).all() or float(Hm.diagonal().mean()) <= 0:      # unrouted expert: no stats
                HG["H"][i] = torch.eye(Hm.shape[0], device=Hm.device); flags.append(f"H{i}=I")
        for i, G in enumerate(HG["G"][:2]):
            if not torch.isfinite(G).all():
                HG["G"][i] = None; flags.append(f"G{i}=none")
        rk = dict(zip(NE.PROJ, map(float, a.res_k.split(",")))) if a.res_k else None
        art, _ = NE.encode_expert(teacher(a.source, L, E), HG, rate=a.rate, res_K=rk)
        art["meta"].update(layer=L, expert=E, flags=flags, bnd=NB.bnd_tag(bw), bnd_k=a.bnd_k, bnd_cap=a.bnd_cap, hg_meta={k: (float(v) if torch.is_tensor(v) else v)
                                                                   for k, v in HG.get("meta", {}).items()})
        torch.save(art, path + ".tmp"); os.replace(path + ".tmp", path)
        print(f"[L{L} E{E}] {time.time()-t0:.0f}s {flags} "
              f"{ {p: round(art['meta']['info'][p]['bits'][4], 4) for p in NE.PROJ} }", flush=True)
        del art, HG; torch.cuda.empty_cache()
    if a.no_finalize:
        return
    # ---- finalize: all experts of the layer present -> TP shard files + manifest
    have = sorted(int(f[1:-3]) for f in os.listdir(ed) if f.endswith(".pt"))
    shards = [dict() for _ in range(NSH)]
    per_exp, proj_meta = {}, None
    for E in have:
        art = torch.load(f"{ed}/E{E}.pt", weights_only=False)
        for s, part in enumerate(split_expert(art)):
            shards[s][E] = part
        m = art["meta"]
        per_exp[E] = dict(flags=m.get("flags"), rate=m["rate"], bnd=m.get("bnd"),
                          proj={p: {k: m["info"][p][k] for k in ("bits", "proxy_rot", "bitexact", "L2_equal_canonical")
                                    if k in m["info"][p]} for p in NE.PROJ})
        if proj_meta is None:
            proj_meta = {p: {k: v for k, v in art[p]["meta"].items()} for p in NE.PROJ}
            cfg = {k: m[k] for k in ("format", "rate", "base_var", "lam", "inner", "sigma", "canonical_base", "res_K") if k in m}
            cfg.update(bnd=m.get("bnd"), bnd_k=m.get("bnd_k"), bnd_cap=m.get("bnd_cap"))
    files = {}
    for s in range(NSH):
        f = f"{d}/tp{s}.pt"
        torch.save(shards[s], f + ".tmp"); os.replace(f + ".tmp", f)
        files[f"tp{s}.pt"] = dict(sha256=sha(f), bytes=os.path.getsize(f))
    exp_bytes = {}
    for pn in NE.PROJ:
        pr = shards[0][have[0]][pn]
        exp_bytes[pn] = {k: int(v.numel() * v.element_size()) if k != "word" else 2 * int(v.numel())
                         for k, v in pr.items()}
        if "var" in pr:
            exp_bytes[pn]["var"] = (pr["var"].numel() * D.variant_bits(cfg["base_var"]) + 7) // 8
    man = dict(format="nestquant-v1", layer=L, default_allocation=default_allocation(a, L), n_experts=len(have), experts=have, config=cfg, tp=NSH,
               proj_meta=proj_meta, files=files,
               packed_bytes_per_expert_per_shard=exp_bytes,
               note="word = u16 Mb | N<<8 per 16x128 unit (stored int32 here); var = 1-bit per-ring sign "
                    "(stored uint8 here); base/p4 = LSB-first ring streams, units in (strip, chunk) order per shard; "
                    "residual K per unit from proj_meta.res_rule (positional, no map). Decoder: nq_decode.py.",
               per_expert=per_exp, time=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(man, open(f"{d}/manifest.json", "w"), indent=1)
    print(f"[L{L}] finalized {len(have)} experts -> {d}", flush=True)
    if a.check_decode:
        bad = 0
        for E in have:
            art = torch.load(f"{ed}/E{E}.pt", weights_only=False)
            re = assemble(a.out, L, E)
            for Lv in (2, 4):
                bad += not all(torch.equal(x, y) for x, y in zip(D.decode_expert(art, Lv), D.decode_expert(re, Lv)))
        print(f"[L{L}] shard-file decode == artifact decode: {bad == 0} ({bad} mismatches)", flush=True)


if __name__ == "__main__":
    main()
