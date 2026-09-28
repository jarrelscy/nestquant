"""Thread 19 stage 1: sequential FP8-reference forward of GLM-5.3 over the calibration windows.

Mirrors orbit-duet's pilot arithmetic (benchmarks/capture_glm_training.py + glm_calibration.py) so that the
first 128 windows reproduce the pilot's normalised MoE inputs:
  * attention/norm/router: orbit's glm_reference.Layer, one 512-token window per call, document-local mask;
  * dense MLP: residual + expert_forward(x) per window;
  * MoE: expert-index order, each expert's rows ascending in 4096-row batches, bf16 accumulation
    out = out + (ffn(x) * p).bf16, then out + shared (4096-row blocks), then residual + value.
    Token chunk 0 is exactly [0, 65536) (= the pilot corpus), so its batches are the pilot's batches.
The frozen matched evaluation documents are replayed alongside with capture_glm.py's arithmetic
(batch-1 docs, no mask, all 5120 rows routed at once, index_add_), giving harness-format captures per layer.

Per MoE layer writes  OUT/acts/L{l}/{x.bf16, ids.u8, p.f32, done.json}  (fit rows),
OUT/eval/val/layer_{l}.pt (held-out windows, harness capture format) and OUT/eval/matched/layer_{l}.pt.
Resumable at layer granularity via OUT/state/.
"""
import argparse
import json
import os
import resource
import time

import numpy as np
import torch
import torch.nn.functional as F

import nq19
from nq19 import D, NEXP, OUT, CORPUS, MATCHED, TAIL_FIRST_WINDOW, VAL_WINDOWS_RESERVED

CTX = 512


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def ffn(x, w):
    g, u, d = w
    return F.linear(F.silu(F.linear(x, g)) * F.linear(x, u), d)


def pending_acts_gb(shards_root):
    """Stage-1 activations on disk not yet merged into the cumulative stats (stage 2 writes acts/L*/merged)."""
    import glob
    tot = 0
    for f in glob.glob(f"{shards_root}/s*/acts/L*/x.bf16"):
        if not os.path.exists(os.path.join(os.path.dirname(f), "merged")):
            try:
                tot += os.path.getsize(f)
            except FileNotFoundError:
                pass
    return tot / 2**30


def free_gb(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 2**30


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


class Corpus:
    def __init__(self, fit_windows, val_windows, fit_start=0):
        self.tokens = np.load(f"{CORPUS}/tokens.npy", mmap_mode="r")
        self.segs = np.load(f"{CORPUS}/segments.npy", mmap_mode="r")
        if fit_start + fit_windows > TAIL_FIRST_WINDOW - VAL_WINDOWS_RESERVED:
            raise ValueError("fit windows overlap the held-out / nq-tail windows")
        self.windows = list(range(fit_start, fit_start + fit_windows)) + list(range(TAIL_FIRST_WINDOW - val_windows, TAIL_FIRST_WINDOW))
        self.n_fit = fit_windows * CTX
        self.T = len(self.windows) * CTX

    def seg(self, w):   # int32 like the pilot (np.array(segments[...]) -> torch)
        return torch.from_numpy(np.array(self.segs[self.windows[w]:self.windows[w] + 1], copy=True))

    def tok(self, w):
        return torch.from_numpy(np.array(self.tokens[self.windows[w]:self.windows[w] + 1], copy=True)).long()


def chunks(n_fit, T, step):
    out, r = [], 0
    first = min(65536, n_fit)
    if first:
        out.append((0, first)); r = first
    while r < n_fit:
        n = min(step, n_fit - r); out.append((r, n)); r += n
    while r < T:
        n = min(step, T - r); out.append((r, n)); r += n
    return out


def val_meta(corpus, fit_rows):
    """document_ids (window-local segments), token positions and category per doc for held-out rows."""
    wins = corpus.windows[fit_rows // CTX:]
    want = set(wins)
    cats = {}
    with open(f"{CORPUS}/windows.jsonl") as f:
        for i, line in enumerate(f):
            if i in want:
                cats[i] = [s["category"] for s in json.loads(line)["segments"]]
    doc_ids, pos, domains, keys = [], [], [], []
    for w in wins:
        s = np.asarray(corpus.segs[w])
        change = np.ones(CTX, bool); change[1:] = s[1:] != s[:-1]
        k = np.cumsum(change) - 1                       # segment ordinal within window
        if k[-1] + 1 != len(cats[w]):
            raise ValueError(f"window {w}: segment count mismatch with windows.jsonl")
        starts = np.maximum.accumulate(np.where(change, np.arange(CTX), 0))
        base = len(domains)
        doc_ids.append(base + k); pos.append(np.arange(CTX) - starts)
        domains += [f"val:{c}" for c in cats[w]]
        keys += [(w, int(s[np.argmax(k == j)])) for j in range(k[-1] + 1)]
    return (torch.from_numpy(np.concatenate(doc_ids)).long(), torch.from_numpy(np.concatenate(pos)).long(),
            domains, keys)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--fit-start", type=int, default=0, help="first fit window (shard k: 2048 k)")
    ap.add_argument("--fit-windows", type=int, default=2048)
    ap.add_argument("--acts-budget-gb", type=float, default=0,
                    help="wait before a layer while shards/*/acts/L*/x.bf16 not yet merged by stage 2 exceed this (0 = off)")
    ap.add_argument("--val-windows", type=int, default=128)
    ap.add_argument("--last-layer", type=int, default=77)
    ap.add_argument("--acts-layers", default="all", help="MoE layers whose acts/evals are written (all|list)")
    ap.add_argument("--chunk", type=int, default=131072)
    ap.add_argument("--no-matched", action="store_true")
    ap.add_argument("--max-layers", type=int, default=0, help="stop after this many new layers (0 = no limit)")
    a = ap.parse_args()
    nq19.gpu_cap()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "16")))
    from benchmarks.ood.glm_reference import Layer
    src = nq19.Src()
    cfg = src.config
    corpus = Corpus(a.fit_windows, a.val_windows, a.fit_start)
    n_fit, T = corpus.n_fit, corpus.T
    nwin = len(corpus.windows)
    plan = chunks(n_fit, T, a.chunk)
    moe_layers = [l for l in range(cfg["num_hidden_layers"]) if l >= cfg["first_k_dense_replace"]]
    acts_layers = set(moe_layers) if a.acts_layers == "all" else {int(v) for v in a.acts_layers.split(",")}
    for d in ("state", "acts", "eval/val", "eval/matched"):
        os.makedirs(f"{a.out}/{d}", exist_ok=True)
    protocol = dict(
        schema="nestquant-19-glm53-full-capture-v1", fit_start=a.fit_start, fit_windows=a.fit_windows, fit_tokens=n_fit,
        val_windows=[corpus.windows[a.fit_windows]] + [corpus.windows[-1]] if a.val_windows else [],
        corpus=CORPUS, corpus_tokens_sha256=json.load(open(f"{CORPUS}/manifest.json"))["tokens_sha256"],
        source=src.root, chunk0=plan[0], chunk=a.chunk,
        arithmetic="orbit-duet pilot: FP8 decoded to BF16; document-local attention per 512-window; "
                   "expert-index-ordered BF16 accumulation (4096-row expert batches) before shared + residual",
        torch=torch.__version__)
    prog_path = f"{a.out}/state/progress.json"
    if os.path.exists(prog_path):
        prog = json.load(open(prog_path))
        if prog["protocol"] != protocol:
            raise ValueError("protocol differs from the existing state; use a fresh --out")
    else:
        prog = dict(protocol=protocol, next_layer=0, layers={})
    write_json(f"{a.out}/protocol.json", protocol)

    # ---- matched evaluation docs (capture_glm.py semantics)
    docs = [json.loads(l) for l in open(f"{MATCHED}/documents.jsonl")]
    mtok = torch.tensor([r["tokens"] for r in docs], device="cuda", dtype=torch.long)
    mprotocol = dict(role="evaluation only", corpus=json.load(open(f"{MATCHED}/manifest.json")),
                     arithmetic="orbit-duet capture_glm.py replay (thread 19 re-implementation)",
                     source=src.root)
    vprotocol = dict(role="evaluation only", corpus=CORPUS,
                     windows=corpus.windows[a.fit_windows:], selection="last held-out windows below the nq-tail "
                     "(windows < 28784), document-local attention like the fit rows", arithmetic=protocol["arithmetic"])
    if a.val_windows:
        v_doc, v_pos, v_dom, _ = val_meta(corpus, n_fit)

    # ---- state
    S = torch.empty(T, D, dtype=torch.bfloat16)
    first_layer = prog["next_layer"]
    if first_layer == 0:
        for w in range(nwin):
            S[w * CTX:(w + 1) * CTX] = src.embed(corpus.tok(w).cuda()).flatten(0, 1).cpu()
        M = src.embed(mtok)
    else:
        sp = prog["state_file"]
        with open(f"{a.out}/state/{sp}", "rb") as f:
            if f.readinto(memoryview(S.view(torch.uint8).numpy().reshape(-1))) != T * D * 2:
                raise ValueError("short state file")
        M = torch.load(f"{a.out}/state/{sp}.matched.pt").cuda()
    cache = nq19.ExpertCache(src) if any(l >= first_layer for l in moe_layers) else None
    t_start = time.time()
    new_layers = 0
    for li in range(first_layer, a.last_layer + 1):
        while ((a.acts_budget_gb and pending_acts_gb(os.path.dirname(os.path.abspath(a.out))) > a.acts_budget_gb)
               or free_gb(a.out) < 200 + 50):
            time.sleep(30)                                   # back-pressure: let stage 2 merge / delete; disk floor
        t0 = time.time()
        layer = Layer(src, li)
        sparse = layer.sparse
        tim = dict(load=0., front=0., moe=0., write=0., matched=0.)
        if sparse:
            cache.load(li)
            shared = [src.weight(f"model.layers.{li}.mlp.shared_experts.{p}.weight") for p in nq19.PROJ]
        else:
            dense = [src.weight(f"model.layers.{li}.mlp.{p}.weight") for p in nq19.PROJ]
        tim["load"] = time.time() - t0
        acts = sparse and li in acts_layers
        if acts:
            ad = f"{a.out}/acts/L{li}"
            os.makedirs(ad, exist_ok=True)
            fx = open(f"{ad}/x.bf16", "wb"); fi = open(f"{ad}/ids.u8", "wb"); fp = open(f"{ad}/p.f32", "wb")
            vx, vi, vp = [], [], []
        cov = np.zeros(NEXP, np.int64)
        for (r0, n) in plan:
            ta = time.time()
            st = S[r0:r0 + n].cuda().view(n // CTX, CTX, D)
            res = torch.empty_like(st); xs = torch.empty_like(st)
            if sparse:
                ids = torch.empty(n, 8, dtype=torch.long, device="cuda"); gp = torch.empty(n, 8, device="cuda")
            for k in range(n // CTX):
                seg = corpus.seg(r0 // CTX + k).cuda()
                r, x = layer.front(st[k:k + 1], segments=seg)
                if sparse:
                    i_, g_ = layer.route(x.flatten(0, 1))
                    ids[k * CTX:(k + 1) * CTX] = i_; gp[k * CTX:(k + 1) * CTX] = g_.float()
                    res[k] = r[0]; xs[k] = x[0]
                else:
                    st[k] = (r + ffn(x, dense))[0]
            del r, x
            torch.cuda.synchronize(); tim["front"] += time.time() - ta
            if sparse:
                tb = time.time()
                res = res.view(n, D); xs = xs.view(n, D); out = torch.zeros(n, D, dtype=torch.bfloat16, device="cuda")
                flat = ids.reshape(-1)
                order = torch.argsort(flat, stable=True)
                rows_all = order // 8
                p_all = gp.reshape(-1)[order]
                cnt = torch.bincount(flat, minlength=NEXP)
                cov += cnt.cpu().numpy()
                offs = [0] + cnt.cumsum(0).tolist()
                for e in range(NEXP):
                    if offs[e] == offs[e + 1]:
                        continue
                    w = cache.expert(e)
                    for b in range(offs[e], offs[e + 1], 4096):
                        sel = rows_all[b:min(b + 4096, offs[e + 1])]
                        contrib = (ffn(xs[sel], w) * p_all[b:b + len(sel)][:, None]).to(torch.bfloat16)
                        out[sel] = out[sel] + contrib
                    del w
                for b in range(0, n, 4096):
                    value = out[b:b + 4096] + ffn(xs[b:b + 4096], shared)
                    out[b:b + 4096] = res[b:b + 4096] + value
                st = out
                torch.cuda.synchronize(); tim["moe"] += time.time() - tb
                if acts:
                    tc = time.time()
                    xc, ic, pc = xs.cpu(), ids.to(torch.uint8).cpu(), gp.cpu()
                    if r0 < n_fit:
                        xc.view(torch.uint8).numpy().tofile(fx); ic.numpy().tofile(fi); pc.numpy().tofile(fp)
                    else:
                        vx.append(xc); vi.append(ids.cpu()); vp.append(pc)
                    tim["write"] += time.time() - tc
                del res, xs, out, ids, gp
            S[r0:r0 + n] = st.reshape(n, D).cpu()
            del st
        # ---- matched eval docs
        if not a.no_matched:
            tm = time.time()
            rs, xs_ = [], []
            for doc in M.split(1):
                r, x = layer.front(doc); rs.append(r); xs_.append(x)
            residual = torch.cat(rs); x = torch.cat(xs_); flat = x.flatten(0, 1)
            if not sparse:
                M = residual + layer.mlp(x)
            else:
                mids, mg = layer.route(flat)
                if acts:
                    payload = dict(x=flat.cpu(), ids=mids.cpu(), p=mg.cpu(),
                                   document_ids=torch.arange(len(docs)).repeat_interleave(mtok.shape[1]),
                                   token_positions=torch.arange(mtok.shape[1]).repeat(len(docs)),
                                   domains=[r["domain"] for r in docs], layer=li, protocol=mprotocol)
                    torch.save(payload, f"{a.out}/eval/matched/layer_{li}.partial")
                    os.replace(f"{a.out}/eval/matched/layer_{li}.partial", f"{a.out}/eval/matched/layer_{li}.pt")
                result = torch.zeros_like(flat)
                for e in mids.unique().sort().values.tolist():
                    rows, slots = torch.where(mids == e)
                    value = ffn(flat[rows], cache.expert(e))
                    result.index_add_(0, rows, (value * mg[rows, slots, None]).to(result.dtype))
                result += ffn(flat, shared)
                M = residual + result.reshape_as(residual)
            del rs, xs_, residual, x, flat
            tim["matched"] = time.time() - tm
        if acts:
            fx.close(); fi.close(); fp.close()
            if a.val_windows:
                vpay = dict(x=torch.cat(vx), ids=torch.cat(vi), p=torch.cat(vp), document_ids=v_doc,
                            token_positions=v_pos, domains=v_dom, layer=li, protocol=vprotocol)
                torch.save(vpay, f"{a.out}/eval/val/layer_{li}.partial")
                os.replace(f"{a.out}/eval/val/layer_{li}.partial", f"{a.out}/eval/val/layer_{li}.pt")
                del vpay, vx, vi, vp
            write_json(f"{ad}/done.json", dict(layer=li, rows=n_fit, width=D, x="x.bf16 [rows, 6144] bf16",
                                               ids="ids.u8 [rows, 8] uint8", p="p.f32 [rows, 8] float32 (router weight incl. routed_scaling_factor)",
                                               coverage_all_rows=cov.tolist(), protocol=protocol))
        if not torch.isfinite(S[::997].float()).all():
            raise ValueError(f"nonfinite state after layer {li}")
        # ---- checkpoint
        tw = time.time()
        sp = f"state_{li % 2}.bf16"
        S.view(torch.uint8).numpy().tofile(f"{a.out}/state/{sp}")
        torch.save(M.cpu(), f"{a.out}/state/{sp}.matched.pt")
        prog["next_layer"] = li + 1; prog["state_file"] = sp
        prog["layers"][str(li)] = dict(seconds=round(time.time() - t0, 1), **{k: round(v, 1) for k, v in tim.items()},
                                       ckpt=round(time.time() - tw, 1), sparse=sparse,
                                       peak_cuda_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                                       peak_rss_gb=round(rss_gb(), 2))
        write_json(prog_path, prog)
        print(json.dumps(dict(layer=li, **prog["layers"][str(li)], elapsed=round(time.time() - t_start))), flush=True)
        del layer
        new_layers += 1
        if a.max_layers and new_layers >= a.max_layers:
            break


if __name__ == "__main__":
    main()
