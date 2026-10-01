"""T35 Stage 1 pilot: pattern-rate base (b1.5, b1.75) vs production b2.0 on thread 12's 9 pilot experts.

  CUDA_VISIBLE_DEVICES=g NQ_GLM_SOURCE=/tmp/nestquant/src/glm53-fp8 python pilot15.py 16:36,49:92 [--cfgs b20,b175,b15]

Per expert (same H = thread-08 recipe nq_run.glm_H, same eval capture/metric as threads/12 results_v1):
  * encode each config with the production encoder (single joint pass, inner 0, sign variants, lr plane) at the
    config's base K + matched residual (nq15.CONFIGS); encode_expert asserts bit-exact internal decode;
  * round trip: artifact saved, re-loaded from disk, decoded by nq_decode (+ this thread's base-K ring_levels) and
    compared bit-exactly with the encoder's dense levels; 32 units / projection cross-checked with ref15_spec;
  * EXL3 same-H anchors at the base rates (EXL3-1.5 native half rate, EXL3-1.75 via the (1, 0xEEEE) frac Viterbi);
    EXL3-2 / EXL3-4 / EXL3-4+4.125 copied from threads/12 results_v1 (same H, same protocol);
  * error direction vs FP8: E_b = W_b(L2) - W_fp8; ratio |E_b|^2/|E_2|^2 (plain + H-metric) and cos(E_b, E_2).
Results: results/pilot/L{L}_E{E}.json (rows nq15_{cfg}/L{2,4}); artifacts /tmp/nestquant/35-nq15/pilot/art.
"""
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nq15                                    # installs the base-K extension (must precede encode calls)
import torch
import nq_decode as D
import nq_encode as NE
import nq_run as R
import nq_patvit as PV
import harness as h

ART = "/tmp/nestquant/35-nq15/pilot/art"
RES = f"{HERE}/results/pilot"
R.SCR = "/tmp/nestquant/35-nq15/pilot/H"        # glm_H cache (thread 12's scratch dir is not used)
T12RES = "/home/coder/git/nestquant/threads/12-reference-encoder/results_v1"
PROJ = NE.PROJ


def hmetric(E, H):
    return float((E @ H * E).sum())


@torch.no_grad()
def direction(dense, W, HG):
    """per projection + total: ratio and cosine of each config's L2/L4 error vs b20's, plain and H-metric."""
    out = {}
    for cfg in dense:
        if cfg == "b20":
            continue
        row = {}
        for Lv in (2, 4):
            tot = dict(nb=0., n2=0., dot=0., hb=0., h2=0., hdot=0.)
            per = {}
            for pi, pn in enumerate(PROJ):
                Wt = W[pi].cuda().float(); Hm = HG["H"][pi].cuda().float()
                Eb = dense[cfg][Lv][pi].cuda() - Wt; E2 = dense["b20"][Lv][pi].cuda() - Wt
                nb, n2, dot = float(Eb.square().sum()), float(E2.square().sum()), float((Eb * E2).sum())
                hb, h2 = hmetric(Eb, Hm), hmetric(E2, Hm)
                hdot = float((Eb @ Hm * E2).sum())
                per[pn] = dict(ratio=nb / n2, cos=dot / (nb * n2) ** .5, ratio_H=hb / h2, cos_H=hdot / (hb * h2) ** .5,
                               rel_fp8=nb / float(Wt.square().sum()))
                for k, v in zip(tot, (nb, n2, dot, hb, h2, hdot)):
                    tot[k] += v
                del Wt, Hm, Eb, E2
            per["all"] = dict(ratio=tot["nb"] / tot["n2"], cos=tot["dot"] / (tot["nb"] * tot["n2"]) ** .5,
                              ratio_H=tot["hb"] / tot["h2"], cos_H=tot["hdot"] / (tot["hb"] * tot["h2"]) ** .5)
            row[f"L{Lv}"] = per
        out[cfg] = row
    torch.cuda.empty_cache()
    return out


def exl3_anchor(Ws, HG, K):
    q, bpw = [], []
    for pi, pn in enumerate(PROJ):
        # pattern K: frac Viterbi; a LUT-tensor codebook skips exllamav3 pack_trellis (half-integer K only)
        kw = dict(quantizer=PV.patq, codebook=h.codebook_lut("mul1")) if PV.is_pat(K) else {}
        Wq, inf = h.quantize_exl3_like(Ws[pi], HG["H"][pi], K, count=1, sigma_reg=R.SIG[pn], **kw)
        q.append(Wq.cpu()); bpw.append(inf["bpw"]); h.free_scratch()
    return q, sum(bpw) / 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experts")
    ap.add_argument("--cfgs", default="b20,b175,b15")
    ap.add_argument("--no-anchors", action="store_true")
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(16 / 80)
    torch.backends.cuda.matmul.allow_tf32 = False
    os.makedirs(ART, exist_ok=True); os.makedirs(R.SCR, exist_ok=True); os.makedirs(RES, exist_ok=True)
    for spec in a.experts.split(","):
        L, E = map(int, spec.split(":"))
        t0 = time.time()
        log = lambda m: print(f"[{L}:{E}] {m} {time.time() - t0:.0f}s", flush=True)
        book = R.Book(f"{RES}/L{L}_E{E}.json")
        ev = book.R.setdefault("eval", {})
        info_all = book.R.setdefault("info", {})
        t12 = json.load(open(f"{T12RES}/L{L}_E{E}.json"))
        for n in ("EXL3-2", "EXL3-4", "EXL3-4+4.125", "nq_p4126_c0i0lr/L2", "nq_p4126_c0i0lr/L4"):
            if n in t12["eval"]:
                ev.setdefault(n, t12["eval"][n]); info_all.setdefault(n, dict(copied_from=f"{T12RES}/L{L}_E{E}.json"))
        data = h.load_expert(L, E)
        HG = R.glm_H(data, L, E)
        log("loaded")
        methods, extra, dense = {}, {}, {}
        for cfg in a.cfgs.split(","):
            C = nq15.CONFIGS[cfg]
            NE.BASE_K = C["base_K"]
            torch.cuda.synchronize(); t1 = time.time()
            art, dn = NE.encode_expert(data.teacher, HG, res_K=C["res_K"], canonical_base=False, inner=0)
            dt = time.time() - t1
            NE.BASE_K = 2.0
            path = f"{ART}/L{L}_E{E}_{cfg}.pt"
            torch.save(art, path)
            # ---- decode round trip from disk (fresh load, CPU -> GPU decode), bit-exact vs encoder dense levels
            art2 = torch.load(path, weights_only=False, map_location="cpu")
            rt = {Lv: all(torch.equal(x.cpu(), y) for x, y in zip(D.decode_expert(art2, Lv), dn[Lv])) for Lv in (2, 4)}
            r15 = {p: nq15.xcheck_ref15(art2[p], 32) for p in PROJ}
            m = art["meta"]
            bits = {Lv: sum(m["info"][p]["bits"][Lv] for p in PROJ) / 3 for Lv in (2, 4)}
            for Lv in (2, 4):
                nm = f"nq15_{cfg}/L{Lv}"
                methods[nm] = dn[Lv]
                extra[nm] = dict(bpw=bits[Lv], base_K=C["base_K"], res_K=C["res_K"], time_s=dt, roundtrip_bitexact=rt,
                                 ref15_mismatch=r15, bitexact_internal={p: m["info"][p]["bitexact"] for p in PROJ},
                                 bits_proj={p: m["info"][p]["bits"] for p in PROJ}, lr_rank=m["lr_rank"])
            dense[cfg] = dn
            log(f"{cfg} base {C['base_K']} L2 {bits[2]:.4f} L4 {bits[4]:.4f} bpw  roundtrip {rt}  ref15 {r15}  {dt:.0f}s")
            assert all(rt.values()) and all(v == 0 for v in r15.values()), (cfg, rt, r15)
            del art, art2
            PV.free_tmp(); h.free_scratch()
        if "b20" in dense:
            book.R["direction"] = direction(dense, data.teacher, HG)
            for cfg, row in book.R["direction"].items():
                log(f"direction {cfg}: L2 ratio {row['L2']['all']['ratio']:.3f} ratio_H {row['L2']['all']['ratio_H']:.3f} "
                    f"cos {row['L2']['all']['cos']:.3f} cos_H {row['L2']['all']['cos_H']:.3f}")
        if not a.no_anchors:
            for K in (1.5, 1.75):
                nm = f"EXL3-{K}"
                if nm not in ev:
                    q, bpw = exl3_anchor(data.teacher, HG, K)
                    methods[nm] = q; extra[nm] = dict(bpw=bpw)
            log("anchors")
        R.evaluate_into(book, data, methods, extra)
        book.R["time_s"] = time.time() - t0
        book.save()
        log("evaluated")
        del data, HG, methods, dense
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
