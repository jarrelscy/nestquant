"""Inkling-2-Mini calibration capture and KLD eval (whole-model HF forward + MoE hook; T19 stats schema via cap_ink).

Mini is a looped model: layers 6-17 run 3 passes with shared weights (48 layer calls per token). A layer-sequential
re-implementation (cap_ink) cannot follow the cross-pass messages, so this script runs the official HF
Inkling2LoopForConditionalGeneration per segment and hooks every MoE block call.

  phase fwd   : per segment forward; each MoE call (layer L, pass p) writes its input rows x to
                OUT/tmp/x.r{R}.L{L}.p{p}.bf16 (memmap [N, D]); val top-64 logprobs + CE -> OUT/final; optional trace.
  phase stats : per layer, rows of all its passes (N * P) -> cap_ink Stats/moe_pass/write_group/write_meta/write_eval
                (teacher weights from inkdq, the encoder's source). P = 3 for loop layers, else 1.
  phase eval  : forward with experts replaced by predecoded NestQuant levels (--wdir, --wsel 2|4|hot:N|float), KLD vs --ref.

    torchrun --nproc-per-node 4 cap_mini.py --phase fwd --out /tmp/nq_mini/cap-txt [--trace DIR]
    torchrun --nproc-per-node 4 cap_mini.py --phase stats --out /tmp/nq_mini/cap-txt
    INK_MM=/tmp/nq_mini/corpus_mm ... (image windows; pixel values go through the HF vision tower)
Same world size for fwd and stats (rank-local row files)."""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

torch.set_grad_enabled(False)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inkdq as Q
import cap_ink as C
from transformers.models.inkling2_loop import modeling_inkling2_loop as LM

D, FF, NE, TOPK = Q.D, Q.F, Q.NE, 8
MM = C.MM
KLV = []                 # per-token KLD vectors (eval)
REC = {}                 # (layer) -> list of (x, ids, w) in call order for the current segment
EVAL = {}                # eval-mode expert override state


def passes(cfg, L):
    return cfg.loop_num_passes if cfg.loop_layer_index_start <= L < cfg.loop_layer_index_end else 1


def moe_forward(self, hidden_states):
    b, s, d = hidden_states.shape
    x = hidden_states.view(-1, d)
    _, w, ids = self.gate(x)
    k = len(REC.setdefault(self.layer_idx, []))
    REC[self.layer_idx].append((x, ids, w))
    if EVAL:
        y = eval_experts(self, x, ids, w, k)
    else:
        y = self.experts(x, ids, w)
    return y.reshape(b, s, d)


def eval_experts(moe, x, ids, w, k):
    """Experts with NestQuant levels: EVAL['lv'][L] = (gu2, dn2, gu4, dn4) bf16 on GPU; hot mask [n, NE] per row."""
    L = moe.layer_idx
    gu2, dn2, gu4, dn4 = EVAL["lv"][L]
    hot = EVAL["hot_fn"](L, k, x, ids, w)                     # bool [n, NE] (level 4 where True)
    out = torch.zeros(x.shape[0], D, dtype=torch.float32, device=x.device)
    flat = ids.reshape(-1)
    order = torch.argsort(flat, stable=True)
    cnt = torch.bincount(flat, minlength=NE).tolist()
    rows_all = order // TOPK
    starts = np.r_[0, np.cumsum(cnt)]
    for e in range(NE):
        if not cnt[e]:
            continue
        sl = order[starts[e]:starts[e + 1]]
        rows = rows_all[starts[e]:starts[e + 1]]
        p = w.reshape(-1)[sl]
        hm = hot[rows, e]
        for sel, Wgu, Wd in ((~hm, gu2, dn2), (hm, gu4, dn4)):
            if Wgu is None or not sel.any():
                continue
            r_ = rows[sel]
            _, y, _, _ = C.teacher(x[r_], Wgu[e], Wd[e])
            out.index_add_(0, r_, y.float() * p[sel][:, None].float())
    return out.to(x.dtype)


def load_model(dev):
    m = LM.Inkling2LoopForConditionalGeneration.from_pretrained(Q.CK, dtype=torch.bfloat16, device_map={"": dev},
                                                                attn_implementation="eager")
    m.eval()
    cfg = m.config.text_config
    for L, lay in enumerate(m.model.language_model.layers):
        lay.mlp.layer_idx = L
        lay.mlp.forward = moe_forward.__get__(lay.mlp)
    return m, cfg


def seg_inputs(data, s, dev, wl):
    """input_ids [1,n] and (pixel_values, image_grid_thw) of the images inside this segment."""
    ix = s["ix"]
    ids = data.tok[torch.from_numpy(ix)].to(dev)[None]
    kw = {}
    if MM:
        a0, a1 = int(ix[0]), int(ix[-1]) + 1
        pv, gr = [], []
        for o, root, w in data.woff:
            if not (o <= a0 < o + 2048):
                continue
            if root not in wl:
                wl[root] = [json.loads(l) for l in open(f"{root}/windows.jsonl")]
            for sl in wl[root][w].get("images", []):
                if a0 <= o + sl["start"] < a1:
                    assert o + sl["end"] <= a1, "image crosses a segment"
                    p = torch.load(f"{root}/{sl['pix_file']}", weights_only=True)
                    pv.append(p["pixel_values"]); gr.append(p["image_grid_thw"])
        if pv:
            kw = dict(pixel_values=torch.cat(pv).to(dev), image_grid_thw=torch.cat(gr).to(dev))
    return ids, kw


def score(a, cfg, lg, s, si, data, dev, ref, res, top, R):
    V = cfg.unpadded_vocab_size
    ix = torch.from_numpy(s["ix"])
    lp = F.log_softmax(lg[:, :V].float(), -1)
    tgt = data.tok[ix][1:].to(dev)
    ce = -lp[:-1].gather(1, tgt[:, None])[:, 0]; acc = (lp[:-1].argmax(-1) == tgt).float()
    pm = None
    if MM:
        pm = torch.zeros(s["n"], dtype=torch.bool, device=dev)
        pm[:-1] = torch.from_numpy(data.rk[ix.numpy()][1:] == 2).to(dev)
        if int(pm.sum()) < 1:
            return
        ce, acc = ce[pm[:-1]], acc[pm[:-1]]
    n_eff = int(pm.sum()) + 1 if pm is not None else s["n"]
    r = dict(rank=R, seg=si, split=s["split"], n=n_eff, ce=float(ce.mean()), acc=float(acc.mean()))
    if ref is not None and si in ref:
        tv, ti = ref[si]["v"].to(dev).float(), ref[si]["i"].to(dev).long()
        pt = tv.exp(); qs = lp.gather(1, ti)
        kl = (pt * (tv - qs)).sum(1)
        rt = (1 - pt.sum(1)).clamp_min(1e-12); rq = (1 - qs.exp().sum(1)).clamp_min(1e-12)
        kl = kl + rt * (rt.log() - rq.log())
        lpm = lp
        if pm is not None:
            kl, ti, lpm = kl[pm], ti[pm], lp[pm]
        r.update(kld=float(kl.mean()), kld_n=int(kl.numel()), top1=float((ti[:, 0] == lpm.argmax(-1)).float().mean()))
        KLV.append((si, kl.float().cpu()))
    res.append(r)
    if s["split"] == "val":
        v, i = lp.topk(64, -1); top[si] = dict(rows=ix, v=v.half().cpu(), i=i.int().cpu())


def summarize(a, R, W):
    dist.barrier()
    if R:
        return
    rr = sum((json.load(open(f"{a.out}/final/ce.r{r}.json")) for r in range(W)), [])
    for sp in ("fit", "val"):
        q = [r for r in rr if r["split"] == sp]
        nt = sum(r["n"] - 1 for r in q)
        if nt:
            print(f"[capmini] final {sp}: ppl {np.exp(sum(r['ce'] * (r['n'] - 1) for r in q) / nt):.3f} "
                  f"acc {sum(r['acc'] * (r['n'] - 1) for r in q) / nt:.4f} ({nt} tokens)", flush=True)
    k = [r for r in rr if "kld" in r]
    if k:
        nk = sum(r["kld_n"] for r in k)
        out = dict(kld=sum(r["kld"] * r["kld_n"] for r in k) / nk, top1=sum(r["top1"] * r["kld_n"] for r in k) / nk,
                   n=nk, wsel=a.wsel, wdir=a.wdir, ref=a.ref)
        json.dump(out, open(f"{a.out}/final/kld.json", "w"), indent=1)
        print(f"[capmini] KLD vs ref: {out['kld']:.5f} top1 {out['top1']:.4f} ({nk} val tokens)", flush=True)


def run_forward(a, data, dev, R, W, splits):
    """Forward every segment of the given splits. fwd: write MoE inputs (+trace). eval: score only."""
    m, cfg = load_model(dev)
    nl = cfg.num_hidden_layers
    if EVAL:                                  # predecoded levels replace the BF16 experts: free them
        for lay in m.model.language_model.layers:
            lay.mlp.experts.gate_up_proj.data = torch.empty(0, device=dev)
            lay.mlp.experts.down_proj.data = torch.empty(0, device=dev)
        torch.cuda.empty_cache()
        EVAL["lv"] = {L: tuple(None if w is None else w.to(dev) for w in ws) for L, ws in EVAL["lv"].items()}
    N = data.N
    store = {}
    if a.phase == "fwd":
        os.makedirs(f"{a.out}/tmp", exist_ok=True)
        for L in range(nl):
            for p in range(passes(cfg, L)):
                fn = f"{a.out}/tmp/x.r{R}.L{L}.p{p}.bf16"
                store[(L, p)] = np.lib.format.open_memmap(fn + ".npy", mode="w+", dtype=np.uint16, shape=(N, D))
    trace = {} if a.trace else None
    os.makedirs(f"{a.out}/final", exist_ok=True)
    ref = torch.load(f"{a.ref}/final/top64.r{R}.pt") if a.ref else None
    res, top, wl = [], {}, {}
    t0 = time.time()
    segs = [(si, s) for si, s in enumerate(data.segs) if s["split"] in splits]
    for k, (si, s) in enumerate(segs):
        REC.clear()
        ids, kw = seg_inputs(data, s, dev, wl)
        if EVAL.get("float"):
            float_prepass(m, cfg, ids, kw, dev)
            REC.clear()
        out = m(input_ids=ids, use_cache=False, **kw)
        if s["n"] >= 2:
            score(a, cfg, out.logits[0], s, si, data, dev, ref, res, top, R)
        if store or trace is not None:
            ix = s["ix"]
            for L, calls in REC.items():
                assert len(calls) == passes(cfg, L), (L, len(calls))
                for p, (x, ti, tw) in enumerate(calls):
                    if store:
                        store[(L, p)][ix] = x.view(torch.int16).cpu().numpy().view(np.uint16)
                    if trace is not None:
                        trace.setdefault((L, p), []).append((torch.from_numpy(ix), ti.short().cpu(), tw.half().cpu(),
                                                             x.float().square().sum(1).cpu()))
        del out
        if R == 0 and (k % 200 == 0 or k == len(segs) - 1):
            print(f"[capmini] {a.phase} seg {k + 1}/{len(segs)} {time.time() - t0:.0f}s "
                  f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G", flush=True)
    for v in store.values():
        v.flush()
    torch.save(top, f"{a.out}/final/top64.r{R}.pt")
    if KLV:
        torch.save(KLV, f"{a.out}/final/klv.r{R}.pt")
    json.dump(res, open(f"{a.out}/final/ce.r{R}.json", "w"))
    if trace is not None:
        os.makedirs(a.trace, exist_ok=True)
        json.dump(dict(wins=[(g, w, sp) for g, _, w, sp in data.wins], N=N), open(f"{a.trace}/windows.r{R}of{W}.json", "w"))
        for (L, p), t in trace.items():
            np.savez(f"{a.trace}/L{L}.p{p}.r{R}of{W}.npz", rows=torch.cat([q[0] for q in t]).numpy(),
                     ids=torch.cat([q[1] for q in t]).numpy().astype(np.uint16),
                     w=torch.cat([q[2] for q in t]).numpy(), xn=torch.cat([q[3] for q in t]).numpy())
    summarize(a, R, W)


class Tiled:
    """cap_ink.Data fields tiled over P passes (rows p*N + i = pass p of row i)."""

    def __init__(self, d, P):
        t = lambda v: torch.cat([v] * P) if torch.is_tensor(v) else np.concatenate([v] * P)
        self.fit, self.cat, self.val = t(d.fit), t(d.cat), t(d.val)
        self.docid, self.pos, self.th, self.en = t(d.docid), t(d.pos), t(d.th), t(d.en)
        self.dom = d.dom * P
        self.N = d.N * P


def run_stats(a, data, dev, R, W, cfg):
    N = data.N
    fit_idx = torch.nonzero(data.fit)[:, 0]
    T_r = len(fit_idx)
    gen = torch.Generator().manual_seed(C.CTX_SEED + R)
    ctx = fit_idx[torch.randperm(T_r, generator=gen)[: T_r // 4]]
    n_ctx_r = len(ctx); dc1 = torch.sort(ctx[: min(n_ctx_r, a.dc_cap)])[0]
    dc_scale = n_ctx_r / max(len(dc1), 1)
    ctxm1 = torch.zeros(N, dtype=torch.bool); ctxm1[ctx] = True
    G = a.groups; gsz = NE // G
    assert gsz % W == 0
    layers = range(*[int(v) + i for i, v in enumerate(a.layers.split("-"))])
    for L in layers:
        t0 = time.time()
        P = passes(cfg, L)
        td = Tiled(data, P)
        X = torch.empty(N * P, D, dtype=torch.bfloat16)
        for p in range(P):
            X[p * N:(p + 1) * N].view(torch.int16).copy_(torch.from_numpy(
                np.load(f"{a.out}/tmp/x.r{R}.L{L}.p{p}.bf16.npy", mmap_mode="r").view(np.int16)))
        ctxm = torch.cat([ctxm1] * P)
        dc = torch.cat([dc1 + p * N for p in range(P)])
        tot = torch.tensor([T_r * P, n_ctx_r * P, len(dc), int(td.val.sum()), N * P], dtype=torch.float64, device=dev)
        allt = [torch.zeros_like(tot) for _ in range(W)]; dist.all_gather(allt, tot)
        allt = [t.tolist() for t in allt]
        T_fit = int(sum(t[0] for t in allt)); n_ctx = int(sum(t[1] for t in allt))
        lay = SimpleNamespace(mlp=SimpleNamespace(gate=GATES[L]))
        Ex = Q.LayerExperts(L, dev)
        g_acc = torch.zeros(NE, 6, FF, dtype=torch.float64, device=dev)
        sal_acc = torch.zeros(NE, 9, 6, dtype=torch.float64, device=dev)
        Cc = torch.zeros(D, D, device=dev); Ca = torch.zeros(D, D, device=dev)
        ev = dict(x=[], ids=[], p=[])
        xc_all = X[dc].to(dev) if len(dc) else None
        NN = N * P
        for gi in range(G):
            e0, e1 = gi * gsz, (gi + 1) * gsz
            st = C.Stats(gsz, dev)
            Wc = {e: Ex.w(e) for e in range(e0, e1)}
            for c0 in range(0, NN, a.chunk):
                c1 = min(NN, c0 + a.chunk)
                x = X[c0:c1].to(dev)
                Yc = torch.zeros(c1 - c0, D, device=dev)
                fm = td.fit[c0:c1].to(dev)
                if gi == 0:
                    C.gram_(Ca, x[fm].float()); C.gram_(Cc, x[ctxm[c0:c1].to(dev)].float())
                C.moe_pass(lay, Wc, x, Yc, st, e0, e1, fm, td.cat[c0:c1].to(dev), None, 1.0, g_acc, sal_acc, None)
                if gi == 0:
                    vm = td.val[c0:c1]
                    if vm.any():
                        vmd = vm.to(dev)
                        tw, ti = C.route(lay, x[vmd])
                        ev["x"].append(x[vmd].cpu()); ev["ids"].append(ti.cpu()); ev["p"].append(tw.float().cpu())
            if xc_all is not None:
                for c0 in range(0, len(dc), 16384):
                    xc = xc_all[c0:c0 + 16384]
                    for e in range(e0, e1):
                        Wgu, Wd = Wc[e]
                        h, _, cg2, cu2 = C.teacher(xc, Wgu, Wd)
                        C.gram_(st.Dc[e - e0], h.float() * dc_scale ** 0.5)
                        g_acc[e, 4] += cg2.sum(0).double() * dc_scale; g_acc[e, 5] += cu2.sum(0).double() * dc_scale
            del Wc
            C.write_group(a, L, st, gi, gsz, R, W)
            del st
            torch.cuda.empty_cache()
        C.write_meta(a, L, g_acc, sal_acc, Cc, Ca, R, W, dict(T_fit=T_fit, n_ctx=n_ctx, allt=allt, dc_cap=a.dc_cap,
                                                                groups=G))
        if R == 0:                                           # protocol notes for the looped model
            mf = f"{a.out}/stats/L{L}.v1/meta.json"
            m = json.load(open(mf))
            m["protocol"]["model"] = "thinkingmachines/Inkling-2-Mini (native NVFP4 L0-19 / MXFP8 L20-23, dequantised)"
            m["protocol"]["passes"] = P
            m["protocol"]["rows"] = "rows of every pass of this layer (loop layers 6-17 run 3 passes per token)"
            json.dump(m, open(mf, "w"))
        val_rows = torch.nonzero(td.val)[:, 0]
        C.write_eval(a, L, ev, td, val_rows, R, W)
        del X, xc_all, Ex
        torch.cuda.empty_cache()
        if R == 0:
            print(f"[capmini] stats L{L} P{P} {time.time() - t0:.0f}s mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G "
                  f"{time.strftime('%H:%M:%S')}", flush=True)
        dist.barrier()


from types import SimpleNamespace  # noqa: E402
GATES = {}


def load_gates(dev, cfg):
    """Router modules only (HF TopKRouter per layer) for the stats phase."""
    from safetensors import safe_open
    for L in range(cfg.num_hidden_layers):
        g = LM.Inkling2LoopTopKRouter(cfg).to(dev)
        g.weight.data.copy_(Q.raw(f"model.llm.layers.{L}.mlp.gate.weight").float())
        g.e_score_correction_bias.data.copy_(Q.raw(f"model.llm.layers.{L}.mlp.gate.bias").float())
        GATES[L] = g.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["fwd", "stats", "eval"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="0-23")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--nfit", default="761,1200")
    ap.add_argument("--val-only", action="store_true")
    ap.add_argument("--chunk", type=int, default=32768)
    ap.add_argument("--groups", type=int, default=1)
    ap.add_argument("--dc-cap", type=int, default=65536)
    ap.add_argument("--trace", default="")
    ap.add_argument("--wdir", default="")
    ap.add_argument("--wsel", default="4")
    ap.add_argument("--hot-stats", default="/tmp/nq_mini/cap-txt/stats")
    ap.add_argument("--ref", default="")
    ap.add_argument("--pred", default="/tmp/nq_mini/pred/net.pt")
    ap.add_argument("--pred-H", type=int, default=56)
    ap.add_argument("--pred-hm", type=float, default=0.5)
    a = ap.parse_args()
    dist.init_process_group("nccl")
    R, W = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", R % torch.cuda.device_count()); torch.cuda.set_device(dev)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_num_threads(4)
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(Q.CK).text_config
    nfit = [int(v) for v in a.nfit.split(",")]
    data = C.Data(C.windows(a.smoke, nfit), R, W)
    if R == 0:
        print(f"[capmini] phase {a.phase} {len(data.wins)} windows/rank0 N {data.N} segs {len(data.segs)}", flush=True)
    if a.phase == "fwd":
        run_forward(a, data, dev, R, W, ("val",) if a.val_only else ("fit", "val"))
    elif a.phase == "stats":
        os.makedirs(f"{a.out}/stats", exist_ok=True); os.makedirs(f"{a.out}/eval/val", exist_ok=True)
        load_gates(dev, cfg)
        run_stats(a, data, dev, R, W, cfg)
    else:
        setup_eval(a, cfg, dev)
        run_forward(a, data, dev, R, W, ("val",))
    dist.barrier()
    dist.destroy_process_group()


def setup_eval(a, cfg, dev):
    lv = {}
    for L in range(cfg.num_hidden_layers):
        s = {v: torch.load(f"{a.wdir}/L{L}.lv{v}.pt", mmap=True) for v in (2, 4) if os.path.exists(f"{a.wdir}/L{L}.lv{v}.pt")}
        g = lambda v, k: s[v][k] if v in s else None
        lv[L] = (g(2, "gu"), g(2, "dn"), g(4, "gu"), g(4, "dn"))
    EVAL["lv"] = lv
    if a.wsel in ("2", "4"):
        full = a.wsel == "4"
        EVAL["hot_fn"] = lambda L, k, x, ids, w: torch.full((x.shape[0], NE), full, dtype=torch.bool, device=x.device)
    elif a.wsel.startswith("hot:"):
        n = int(a.wsel[4:])
        hs = {}
        for L in range(cfg.num_hidden_layers):
            nr = json.load(open(f"{a.hot_stats}/L{L}/meta.json"))["n_routed"]
            m = torch.zeros(NE, dtype=torch.bool, device=dev); m[sorted(range(NE), key=lambda e: -nr[e])[:n]] = True
            hs[L] = m
        EVAL["hot_fn"] = lambda L, k, x, ids, w: hs[L][None].expand(x.shape[0], NE)
    elif a.wsel == "float":
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pred"))
        import train as PT
        ck = torch.load(a.pred, map_location=dev)
        net = PT.Net(ck["args"]["d"], ck["args"]["nl"]).to(dev).eval(); net.load_state_dict(ck["state"])
        stats = PT.layer_stats()
        static = {}
        for L in range(cfg.num_hidden_layers):
            mL, prior = stats[L]
            m_ = torch.zeros(NE, dtype=torch.bool, device=dev); m_[torch.topk(prior.to(dev), a.pred_H).indices] = True
            static[L] = m_
        EVAL.update(float=True, PT=PT, net=net, stats=stats, static=static, H=a.pred_H, hm=a.pred_hm, sets={})

        def hot_fn(L, k, x, ids, w):
            if L not in EVAL["sets"]:                          # prepass: static top-H
                return static[L][None].expand(x.shape[0], NE)
            R_ = EVAL["sets"][L]                              # [nb, NE] set serving each 16-token block
            return R_[torch.arange(x.shape[0], device=x.device) // 16]
        EVAL["hot_fn"] = hot_fn
    else:
        raise SystemExit(f"wsel {a.wsel}?")


def float_prepass(m, cfg, ids, kw, dev):
    """Routing of this segment under the static sets -> predictor sets per layer (all passes summed per block)."""
    EVAL["sets"] = {}
    REC.clear()
    m(input_ids=ids, use_cache=False, **kw)
    n = ids.shape[1]; nb = (n + 15) // 16
    blk = torch.arange(n, device=dev) // 16
    PT = EVAL["PT"]
    sets = {}
    for L, calls in REC.items():
        cnt = torch.zeros(nb, NE, device=dev); sal = torch.zeros(nb, NE, device=dev)
        for x, ti, tw in calls:
            xn = x.float().square().sum(1)
            br = blk[:, None].expand(-1, TOPK).reshape(-1)
            cnt.index_put_((br, ti.reshape(-1)), torch.ones(br.numel(), device=dev), accumulate=True)
            sal.index_put_((br, ti.reshape(-1)), (tw.float() ** 2 * xn[:, None]).reshape(-1), accumulate=True)
        chain = torch.zeros(nb, dtype=torch.long, device=dev); pos = torch.arange(nb, device=dev)
        sets[L] = PT.sets_for_blocks(EVAL["net"], L, cnt, sal, chain, pos, EVAL["H"], EVAL["hm"], EVAL["stats"])
    EVAL["sets"] = sets


if __name__ == "__main__":
    main()
