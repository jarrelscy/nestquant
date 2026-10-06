#!/usr/bin/env python3
"""T37 Mac sizing (CPU only, pure python; reads only safetensors headers).  Numbers for mac/SPEC.md.

  nice -n 10 python3 sizes37.py [--rel /tmp/nestquant/37-flash/release] [--out /tmp/nestquant/37-flash/mac/sizes37.json]

1. Re-implements sm120/moe.py rbits/proj_sizes and streaming/p4rec.layout, and checks them against the measured
   b175 TP4 artifact (rank0.json rec_bytes/seg, res/rank0/L10.pt per-expert planes; constants below).
2. T37 (H 4096, I 2048, base 1.5 = (1,0xAAAA), gu res 2.5 = (2,0xAAAA), dn res 2.8125 = (2,0xFBDE)) level-4
   record + resident bytes at TP1 (Mac) and TP8 (release shards).
3. Backbone at load from the release headers under Mac policies (as shipped / MLX q8 affine g64).
4. Budget tables at 96/104/112 (GB and GiB), solving for floating slots per layer.
5. KV + recurrent state; LMPF TTFT grid (plain / budget / full) over prefill rate P, SSD rate R, window W."""
import argparse
import json
import math
import os
import re
import struct

GB, GiB = 1e9, 2 ** 30
up = lambda n, a: (n + a - 1) // a * a          # noqa: E731
popc = lambda m: bin(m).count("1")              # noqa: E731
RKP = {0: (2, 0), 1: (1, 0xEEEE), 2: (2, 0xAAAA), 3: (2, 0x8888), 4: (3, 0), 5: (1, 0xAAAA), 6: (1, 0xFFFE),
       7: (2, 0x9248), 8: (1, 0xFEFE), 9: (2, 0xD5AA), 10: (2, 0xFBDE)}   # 9 = b1.75 dn res; 10 = T37 dn res (new)
RMAX, SEG_ALIGN = 4, 256


def rbits(rk):
    KA, M = RKP[rk]
    return 4 * (16 * KA + popc(M))               # bits per 64-weight record


def proj(N, K, rk_res, rk_base):
    S, C = N // 16, K // 128
    nrec = S * C * 32
    rb = rbits(rk_res)
    return dict(S=S, C=C, nrec=nrec, p4=nrec * rb // 8 + (4 if rb % 16 else 0), d4=S * C * 4,
                base=nrec * rbits(rk_base) // 8, var=S * C)


def record(H, Ish, rk_gu, rk_dn, rk_base, align):
    gu, dn = proj(2 * Ish, H, rk_gu, rk_base), proj(H, Ish, rk_dn, rk_base)
    n = {"gu.p4": gu["p4"], "gu.d4": gu["d4"], "dn.p4": dn["p4"], "dn.d4": dn["d4"], "lr4": RMAX * (2 * Ish + H) * 2}
    seg, o = {}, 0
    for k in ("gu.p4", "gu.d4", "dn.p4", "dn.d4", "lr4"):
        seg[k] = (o, n[k]); o = up(o + n[k], SEG_ALIGN)
    res = dict(gu_base=gu["base"], gu_var=gu["var"], dn_base=dn["base"], dn_var=dn["var"],
               sc2=(3 * H + 3 * Ish) * 2, sc4=(3 * H + 3 * Ish) * 2,
               lr=(RMAX * (H + 2 * Ish) + RMAX * (Ish + H)) * 2)
    return dict(seg=seg, rec_raw=o, rec_bytes=up(o, align), res=res, res_bytes=sum(res.values()))


def check_b175():
    """measured: /tmp/nestquant/35-nq15/release/repo/rank0.json + res/rank0/L10.pt (GLM-5.3 TP4, H 6144, I/4 512)."""
    r = record(6144, 512, 3, 9, 1, 4096)
    assert r["rec_bytes"] == 2854912, r["rec_bytes"]
    want = {"gu.p4": (0, 1769472), "gu.d4": (1769472, 12288), "dn.p4": (1781760, 1007620),
            "dn.d4": (2789632, 6144), "lr4": (2795776, 57344)}
    assert r["seg"] == want, r["seg"]
    assert r["res"]["gu_base"] == 344064 * 4 and r["res"]["dn_base"] == 172032 * 4
    assert r["res"]["gu_var"] == 3072 and r["res"]["dn_var"] == 1536
    assert r["res"]["sc2"] == 19968 * 2 and r["res"]["lr"] == 55296 * 2
    assert abs(r["res_bytes"] * 256 - 578427268) < 256 * 64, r["res_bytes"] * 256   # + pickle overhead
    return r


def headers(paths):
    out = {}
    for p in paths:
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
        h.pop("__metadata__", None)
        for k, v in h.items():
            out[k] = (v["dtype"], v["shape"], v["data_offsets"][1] - v["data_offsets"][0])
    return out


def backbone(rel):
    """bytes per category: shipped, and Mac q8 (MLX affine 8-bit, group 64 => 8.5 bpw) for every 2-D linear incl.
    embed/lm_head (QuantizedEmbedding); fp8 scale_inv dropped; router gate / mHC fn / norms / conv / fp32 kept."""
    fs = [os.path.join(rel, f) for f in sorted(os.listdir(rel)) if f.startswith(("nonexpert-", "vision_tower"))]
    T = headers(fs)
    cat = {}
    for k, (dt, sh, nb) in T.items():
        L = re.search(r"layers\.(\d+)\.", k)
        L = int(L.group(1)) if L else -1
        if k.startswith("model.visual"):
            c = "vision"
        elif L == 45:
            c = "mtp45_nonexpert"
        elif "embed_tokens" in k:
            c = "embed"
        elif k.startswith("lm_head"):
            c = "lm_head"
        elif ".self_attn." in k and L in KDA:
            c = "attn_kda"
        elif ".self_attn." in k:
            c = "attn_dsa"
        elif "shared_experts" in k:
            c = "shared_experts"
        elif ".mlp.gate." in k:
            c = "router"
        elif ".mlp." in k:
            c = "dense_mlp"
        elif ".hc_" in k:
            c = "mhc"
        else:
            c = "norms_misc"
        lin = (len(sh) == 2 and k.endswith(".weight") and min(sh) >= 64 and ".mlp.gate." not in k
               and ".hc_" not in k)
        q8 = 0 if k.endswith("weight_scale_inv") else (sh[0] * sh[1] * 8.5 / 8 if lin else nb)
        q4 = 0 if k.endswith("weight_scale_inv") else (sh[0] * sh[1] * 4.5 / 8 if lin else nb)
        d = cat.setdefault(c, dict(ship=0, q8=0, q4=0, n=0))
        d["ship"] += nb; d["q8"] += q8; d["q4"] += q4; d["n"] += 1
    return {c: {k: (int(v) if k != "n" else v) for k, v in d.items()} for c, d in sorted(cat.items())}


KDA = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37,
       38, 40, 41, 42, 44]
DSA = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]
NE, NMOE, FIXED, TOPK = 288, 42, 19, 8


def state(ntok, kv_dtype_bytes=2):
    kv = len(DSA) * ntok * (512 * kv_dtype_bytes + 128 * 2 / 4)     # MLA latent (NoPE) + kpool/4 indexer keys
    kda = len(KDA) * (64 * 128 * 128 * 4 + 3 * 3 * 8192 * 2)        # fp32 recurrent + conv tails
    return dict(kv=kv, kda=kda, mtp_kv=ntok * (512 * kv_dtype_bytes + 32), total=kv + kda)


def budget(total, rec, base_all, bb, vision, mtp, ctx, dedicated):
    fixed = FIXED * NMOE * rec
    items = dict(base_1p5=base_all, fixed_4bit=fixed, backbone=bb, vision=vision, mtp=mtp,
                 kv_state=state(ctx)["total"], **dedicated)
    used = sum(items.values())
    nfl = int((total - used) // (NMOE * rec))
    items["floating"] = nfl * NMOE * rec
    items["free"] = total - used - items["floating"]
    return dict(items=items, floating_per_layer=nfl, hot_per_layer=nfl + FIXED)


def ttft(n, P, R, W, rec, nres_layer, budget_s=2.0, moe_frac=0.5, bud_min=1024):
    """seconds.  plain: n/P (chunked, frozen set).  full: per window, per MoE layer, reads of the non-resident
    records (one sequential scan) overlap the layer's compute; the first group of the next layer is prefetched
    during this layer, so exposure per layer = max(0, t_r - t_c).  budget: reads only overlap the layer's own
    MoE (selection needs the router counts) plus budget_s spread over the layers."""
    tc_tok = 1.0 / P
    plain = n * tc_tok
    nw = max(1, math.ceil(n / W))
    tr = (NE - nres_layer) * rec / R
    full = 0.0
    for w in range(nw):
        tw = min(W, n - w * W)
        tcL = tw * tc_tok / 45
        full += tw * tc_tok + NMOE * max(0.0, tr - tcL)
    full += tr                                                       # prime: first MoE layer of window 0
    tcL = n * tc_tok / 45
    x = min(NE - nres_layer, int((moe_frac * tcL + budget_s / NMOE) * R / rec)) if n >= bud_min else 0
    return dict(plain=plain, full=full, budget=plain + (budget_s if n >= bud_min else 0), bud_x=x,
                windows=nw, reads_GB=nw * NMOE * (NE - nres_layer) * rec / GB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rel", default="/tmp/nestquant/37-flash/release")
    ap.add_argument("--out", default="/tmp/nestquant/37-flash/mac/sizes37.json")
    a = ap.parse_args()
    out = {"b175_check": "ok", "b175_rec": check_b175()["rec_bytes"]}
    t1 = {al: record(4096, 2048, 2, 10, 5, al) for al in (4096, 16384)}
    t8 = record(4096, 256, 2, 10, 5, 4096)
    rec = t1[16384]["rec_bytes"]
    base_all = t1[16384]["res_bytes"] * NE * NMOE
    out["tp1"] = {str(k): v for k, v in t1.items()}
    out["tp8"] = t8
    out["rank0_bin_bytes"] = rec * NE * NMOE
    out["base_all"] = base_all
    out["slot_per_layer_all"] = rec * NMOE
    bbc = backbone(a.rel)
    out["backbone_cats"] = bbc
    core = [c for c in bbc if c not in ("vision", "mtp45_nonexpert")]
    bb_ship = sum(bbc[c]["ship"] for c in core)
    bb_q8 = sum(bbc[c]["q8"] for c in core)
    bb_q8_bfemb = bb_q8 - bbc["embed"]["q8"] + bbc["embed"]["ship"]
    vis_ship, vis_q8 = bbc["vision"]["ship"], bbc["vision"]["q8"]
    mtp_ne = bbc["mtp45_nonexpert"]["q8"]
    mtp_exp_q4 = NE * 3 * 4096 * 2048 * 4.5 / 8
    out["backbone"] = dict(ship=bb_ship, q8=bb_q8, q8_bf16embed=bb_q8_bfemb, vision_ship=vis_ship, vision_q8=vis_q8,
                           mtp_nonexpert_q8=mtp_ne, mtp_experts_q4=mtp_exp_q4)
    # decode bytes/token (weights touched): backbone minus embed (one row) + 8 routed experts per MoE layer
    active_bb = bb_q8 - bbc["embed"]["q8"]
    out["decode"] = {}
    for h in (0.5, 0.65, 0.8):
        b = active_bb + TOPK * NMOE * (t1[16384]["res_bytes"] + h * t1[16384]["rec_raw"])
        out["decode"][str(h)] = dict(bytes_per_tok=b, toks_ceiling_614x0p7=614e9 * 0.7 / b)
    # dedicated runtime (not borrowable): MLX activations / command buffers, process, allocator slack, jF
    ded = dict(mlx_workspace=1.0 * GB, process_runtime=1.5 * GB, alloc_slack=1.0 * GB, jf=0.1 * GB)
    out["budgets"] = {}
    for unit, u in (("GB", GB), ("GiB", GiB)):
        for tot in (96, 104, 112):
            for ctx in (32768, 131072):
                mtp = (mtp_ne + mtp_exp_q4) if tot == 112 else 0
                out["budgets"][f"{tot}{unit}_{ctx // 1024}K"] = budget(tot * u, rec, base_all, bb_q8, vis_ship,
                                                                     mtp, ctx, ded)
    out["state"] = {f"{n // 1024}K": state(n) for n in (2048, 8192, 32768, 131072)}
    out["lmpf_buffers"] = dict(
        ring_layer_double=2 * (NE - FIXED) * rec, ring_groups_3x32=96 * rec,
        window_state_per_tok=4 * 4096 * 2, moe_window_per_tok=4096 * 2 + 4096 * 4,
        **{f"window_{w // 1024}K": w * (4 * 4096 * 2 + 4096 * 2 + 4096 * 4) for w in (16384, 32768, 65536)})
    grid = {}
    for P in (250, 400, 600):
        for R in (6e9, 10e9, 13e9):
            for W in (16384, 32768, 65536):
                for n in (2048, 8192, 32768, 131072):
                    for nres in (FIXED, FIXED + 48):
                        grid[f"P{P}_R{int(R / 1e9)}_W{W // 1024}K_n{n // 1024}K_res{nres}"] = ttft(n, P, R, W, rec,
                                                                                                    nres)
    out["ttft"] = grid
    # full-mode break-even window (reads fully hidden): W* = 45 * P * t_r
    out["breakeven_W"] = {f"P{P}_R{int(R / 1e9)}_res{nres}": 45 * P * (NE - nres) * rec / R
                          for P in (250, 400, 600) for R in (6e9, 10e9, 13e9) for nres in (FIXED, FIXED + 48)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    f = lambda x: f"{x / GB:.2f}"                                                    # noqa: E731
    print("b175 check ok; T37 TP1 rec", t1[4096]["rec_bytes"], t1[16384]["rec_bytes"], "raw", t1[4096]["rec_raw"],
          "seg", t1[16384]["seg"])
    print("TP1 res/expert", t1[16384]["res_bytes"], t1[16384]["res"], "base_all GB", f(base_all),
          "rank0.bin GB", f(rec * NE * NMOE), "TP8 rec", t8["rec_bytes"], "TP8 res", t8["res_bytes"])
    for c, d in bbc.items():
        print(f"  {c:18s} n={d['n']:5d} ship {f(d['ship'])} q8 {f(d['q8'])}")
    print("backbone ship/q8/q8+bf16emb", f(bb_ship), f(bb_q8), f(bb_q8_bfemb), "vision", f(vis_ship), f(vis_q8),
          "mtp ne/exp_q4", f(mtp_ne), f(mtp_exp_q4))
    for k, v in out["decode"].items():
        print("decode hit", k, f(v["bytes_per_tok"]), "GB/tok", f"{v['toks_ceiling_614x0p7']:.1f} tok/s")
    for k, v in out["budgets"].items():
        print(k, "float/layer", v["floating_per_layer"], "hot", v["hot_per_layer"],
              {i: f(x) for i, x in v["items"].items()})
    for k, v in out["state"].items():
        print("state", k, {i: f(x) for i, x in v.items()})
    print("lmpf", {k: f(v) if v > 1e6 else v for k, v in out["lmpf_buffers"].items()})
    for k, v in out["breakeven_W"].items():
        print("breakeven", k, int(v))
    for k, v in grid.items():
        if "_R10_" in k or "W16K" in k:
            print(k, {i: (round(x, 1) if isinstance(x, float) else x) for i, x in v.items()})


if __name__ == "__main__":
    main()
