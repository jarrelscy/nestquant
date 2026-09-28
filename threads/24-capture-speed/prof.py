"""T24: fine-grained profile of capture_fwd.py stage 1 (one MoE layer, one chunk) + bitwise-safety probes.

Usage: run.sh-env python prof.py --state <out>/state/state_X.bf16 --layer L --corpus C --fit-start F --rows N
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/coder/git/nestquant/threads/19-full-capture")
import nq19
from nq19 import D, NEXP
from capture_fwd import Corpus, ffn


def T():
    torch.cuda.synchronize()
    return time.time()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--fit-start", type=int, required=True)
    ap.add_argument("--windows", type=int, required=True, help="windows in the state file")
    ap.add_argument("--rows", type=int, default=65536)
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    nq19.gpu_cap()
    from benchmarks.ood.glm_reference import Layer
    src = nq19.Src()
    corpus = Corpus(a.windows, 0, a.fit_start, a.corpus)
    C = corpus.C
    n = a.rows
    S = torch.empty(corpus.T, D, dtype=torch.bfloat16)
    with open(a.state, "rb") as f:
        f.readinto(memoryview(S.view(torch.uint8).numpy().reshape(-1)))
    tim = {}

    def add(k, v):
        tim[k] = tim.get(k, 0.) + v
    t = T(); layer = Layer(src, a.layer); add("layer_init(attn weights)", T() - t)
    t = T(); cache = nq19.ExpertCache(src); add("expertcache_alloc", T() - t)
    t = T(); cache.load(a.layer); add("expertcache_load(disk->pinned)", T() - t)
    t = T(); shared = [src.weight(f"model.layers.{a.layer}.mlp.shared_experts.{p}.weight") for p in nq19.PROJ]; add("shared_load", T() - t)
    t = T(); st = S[:n].cuda().view(n // C, C, D); add("S_h2d_pageable", T() - t)
    Sp = torch.empty(n, D, dtype=torch.bfloat16, pin_memory=True); Sp.copy_(S[:n])
    t = T(); st2 = Sp.cuda(non_blocking=True); add("S_h2d_pinned", T() - t); del st2
    res = torch.empty_like(st); xs = torch.empty_like(st)
    ids = torch.empty(n, 8, dtype=torch.long, device="cuda"); gp = torch.empty(n, 8, device="cuda")
    tf = {"seg_h2d": 0., "front": 0., "route": 0., "copy": 0.}
    t0 = T()
    for k in range(n // C):
        t = T(); seg = corpus.seg(k).cuda(); tf["seg_h2d"] += T() - t
        t = T(); r, x = layer.front(st[k:k + 1], segments=seg); tf["front"] += T() - t
        t = T(); i_, g_ = layer.route(x.flatten(0, 1)); tf["route"] += T() - t
        t = T(); ids[k * C:(k + 1) * C] = i_; gp[k * C:(k + 1) * C] = g_.float(); res[k] = r[0]; xs[k] = x[0]; tf["copy"] += T() - t
    add("front_total(synced per op)", T() - t0)
    for k_, v in tf.items():
        add("  front." + k_, v)
    # unsynced front (as in capture_fwd)
    t0 = T()
    for k in range(n // C):
        seg = corpus.seg(k).cuda()
        r, x = layer.front(st[k:k + 1], segments=seg)
        i_, g_ = layer.route(x.flatten(0, 1))
        ids[k * C:(k + 1) * C] = i_; gp[k * C:(k + 1) * C] = g_.float(); res[k] = r[0]; xs[k] = x[0]
    add("front_total(unsynced, as capture_fwd)", T() - t0)
    del r, x
    res = res.view(n, D); xs = xs.view(n, D)
    # MoE
    tm = {"sort": 0., "w_h2d": 0., "dequant": 0., "ffn": 0., "scatter": 0., "shared+res": 0.}
    t0 = T()
    t = T()
    out = torch.zeros(n, D, dtype=torch.bfloat16, device="cuda")
    flat = ids.reshape(-1); order = torch.argsort(flat, stable=True); rows_all = order // 8
    p_all = gp.reshape(-1)[order]; cnt = torch.bincount(flat, minlength=NEXP); offs = [0] + cnt.cumsum(0).tolist()
    tm["sort"] += T() - t
    for e in range(NEXP):
        if offs[e] == offs[e + 1]:
            continue
        t = T()
        raw = [(cache.raw(e, p, ".weight"), cache.raw(e, p, ".weight_scale_inv")) for p in nq19.PROJ]
        tm["w_h2d"] += T() - t
        t = T(); w = [nq19.dequant(q, s, src.block).to(torch.bfloat16) for q, s in raw]; tm["dequant"] += T() - t
        for b in range(offs[e], offs[e + 1], 4096):
            t = T()
            sel = rows_all[b:min(b + 4096, offs[e + 1])]
            contrib = (ffn(xs[sel], w) * p_all[b:b + len(sel)][:, None]).to(torch.bfloat16)
            tm["ffn"] += T() - t
            t = T(); out[sel] = out[sel] + contrib; tm["scatter"] += T() - t
    t = T()
    for b in range(0, n, 4096):
        value = out[b:b + 4096] + ffn(xs[b:b + 4096], shared)
        out[b:b + 4096] = res[b:b + 4096] + value
    tm["shared+res"] += T() - t
    add("moe_total(synced per op)", T() - t0)
    for k_, v in tm.items():
        add("  moe." + k_, v)
    ref_out = out.clone()
    # unsynced moe as in capture_fwd
    t0 = T()
    out = torch.zeros(n, D, dtype=torch.bfloat16, device="cuda")
    for e in range(NEXP):
        if offs[e] == offs[e + 1]:
            continue
        w = cache.expert(e)
        for b in range(offs[e], offs[e + 1], 4096):
            sel = rows_all[b:min(b + 4096, offs[e + 1])]
            contrib = (ffn(xs[sel], w) * p_all[b:b + len(sel)][:, None]).to(torch.bfloat16)
            out[sel] = out[sel] + contrib
    for b in range(0, n, 4096):
        value = out[b:b + 4096] + ffn(xs[b:b + 4096], shared)
        out[b:b + 4096] = res[b:b + 4096] + value
    add("moe_total(unsynced, as capture_fwd)", T() - t0)
    assert torch.equal(out, ref_out)
    t = T(); xc, ic, pc = xs.cpu(), ids.to(torch.uint8).cpu(), gp.cpu(); add("acts_d2h_pageable", T() - t)
    t = T(); oc = out.cpu(); add("state_d2h_pageable", T() - t)
    xp = torch.empty(n, D, dtype=torch.bfloat16, pin_memory=True)
    t = T(); xp.copy_(xs, non_blocking=True); add("acts_d2h_pinned", T() - t)
    t = time.time(); xc.view(torch.uint8).numpy().tofile("/tmp/nestquant/24-capture-speed/probe_x.bf16"); os.sync(); add("disk_write_x(+sync)", time.time() - t)
    os.remove("/tmp/nestquant/24-capture-speed/probe_x.bf16")
    add("peak_cuda_gb", torch.cuda.max_memory_allocated() / 2**30)
    for k_, v in tim.items():
        print(f"{k_:45s} {v:8.3f}")
    print(json.dumps({k: round(v, 3) for k, v in tim.items()}))

    if a.probe:
        # 1) front batching: B windows in one call vs one-by-one
        for B in (2, 8):
            k0 = 0
            segB = torch.cat([corpus.seg(k0 + j) for j in range(B)]).cuda()
            rB, xB = layer.front(st[k0:k0 + B], segments=segB)
            ok_r = torch.equal(rB.view(-1, D), res[:B * C]); ok_x = torch.equal(xB.view(-1, D), xs[:B * C])
            iB, gB = layer.route(xB.flatten(0, 1))
            ok_i = torch.equal(iB, ids[:B * C]); ok_g = torch.equal(gB.float(), gp[:B * C])
            print(f"probe front batch B={B}: res {ok_r} x {ok_x} ids {ok_i} p {ok_g} "
                  f"maxdiff x {(xB.view(-1, D).float() - xs[:B * C].float()).abs().max().item():.3g}")
            # routing a whole batch of windows (router only), windows computed singly
            iR, gR = layer.route(xs[:B * C])
            print(f"probe route batch B={B}: ids {torch.equal(iR, ids[:B * C])} p {torch.equal(gR.float(), gp[:B * C])}")
        # 2) GEMM M-invariance for expert ffn rows (bf16, K=6144, N=2048)
        w = cache.expert(0)
        X = xs[:8192]
        full = ffn(X, w)
        for M in (1, 7, 64, 333, 1024, 4096):
            part = ffn(X[:M], w)
            print(f"probe ffn M={M}: bitwise {torch.equal(part, full[:M])}")


if __name__ == "__main__":
    main()
