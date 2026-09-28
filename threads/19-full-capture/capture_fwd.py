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
    """Old corpus (orbit glm53_training_15m_v2, C = 512): fit windows [fit_start, +fit_windows) < 28656, val = the
    last val_windows windows below the nq-tail.  Group corpus (thread 21 layout: tokens/segments [W, C] +
    split.json {fit: [0, Nf), val: [Nf, W)}): fit windows inside the fit split, val = the first val_windows windows
    of the val split (-1 = all)."""

    def __init__(self, fit_windows, val_windows, fit_start=0, root=CORPUS):
        self.root = root
        self.tokens = np.load(f"{root}/tokens.npy", mmap_mode="r")
        self.segs = np.load(f"{root}/segments.npy", mmap_mode="r")
        self.C = self.tokens.shape[1]
        self.group = os.path.exists(f"{root}/split.json")
        if self.group:
            sp = json.load(open(f"{root}/split.json"))
            (f0, f1), (v0, v1) = sp["fit"], sp["val"]
            if fit_start < f0 or fit_start + fit_windows > f1:
                raise ValueError(f"fit windows outside the fit split {sp['fit']}")
            nv = v1 - v0 if val_windows < 0 else min(val_windows, v1 - v0)
            self.windows = list(range(fit_start, fit_start + fit_windows)) + list(range(v0, v0 + nv))
        else:
            if self.C != CTX:
                raise ValueError("old corpus must have 512-token windows")
            if fit_start + fit_windows > TAIL_FIRST_WINDOW - VAL_WINDOWS_RESERVED:
                raise ValueError("fit windows overlap the held-out / nq-tail windows")
            self.windows = list(range(fit_start, fit_start + fit_windows)) + list(range(TAIL_FIRST_WINDOW - val_windows, TAIL_FIRST_WINDOW))
        self.n_fit = fit_windows * self.C
        self.T = len(self.windows) * self.C
        self.n_val_windows = len(self.windows) - fit_windows

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
    CTX = corpus.C
    wins = corpus.windows[fit_rows // CTX:]
    want = set(wins)
    cats = {}
    with open(f"{corpus.root}/windows.jsonl") as f:
        for i, line in enumerate(f):
            if i in want:
                cats[i] = json.loads(line)["segments"]
    doc_ids, pos, domains, keys = [], [], [], []
    for w in wins:
        s = np.asarray(corpus.segs[w])
        change = np.ones(CTX, bool); change[1:] = s[1:] != s[:-1]
        k = np.cumsum(change) - 1                       # segment ordinal within window
        ent = cats[w]
        if k[-1] + 1 == len(ent):
            wc = [e["category"] for e in ent]
        elif all("doc_index" in e for e in ent):
            # thread-21 groups: non-contiguous pieces of one document share its segment id (= doc_index)
            byd = {e["doc_index"]: e["category"] for e in ent}
            wc = [byd[int(s[np.argmax(k == j)])] for j in range(k[-1] + 1)]
        else:
            raise ValueError(f"window {w}: segment count mismatch with windows.jsonl")
        starts = np.maximum.accumulate(np.where(change, np.arange(CTX), 0))
        base = len(domains)
        doc_ids.append(base + k); pos.append(np.arange(CTX) - starts)
        domains += [f"val:{c}" for c in wc]
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
    ap.add_argument("--val-windows", type=int, default=128, help="old corpus: last N below the nq-tail; group corpus: "
                    "first N of the val split (-1 = all)")
    ap.add_argument("--corpus", default=CORPUS, help="old orbit corpus or a thread-21 group dir (tokens/segments [W, C], split.json)")
    ap.add_argument("--shard-id", type=int, help="stats shard id (default fit_start // 2048 on the old corpus; required for groups)")
    ap.add_argument("--last-layer", type=int, default=77)
    ap.add_argument("--acts-layers", default="all", help="MoE layers whose acts/evals are written (all|list)")
    ap.add_argument("--chunk", type=int, default=131072)
    ap.add_argument("--ckpt-every", type=int, default=1, help="state checkpoint every N layers (13.7 GB each)")
    ap.add_argument("--no-matched", action="store_true")
    ap.add_argument("--max-layers", type=int, default=0, help="stop after this many new layers (0 = no limit)")
    a = ap.parse_args()
    nq19.gpu_cap()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "16")))
    from benchmarks.ood.glm_reference import Layer
    src = nq19.Src()
    cfg = src.config
    corpus = Corpus(a.fit_windows, a.val_windows, a.fit_start, a.corpus)
    CTX = corpus.C
    if corpus.group and a.shard_id is None:
        raise ValueError("--shard-id is required for group corpora")
    n_fit, T = corpus.n_fit, corpus.T
    nwin = len(corpus.windows)
    plan = chunks(n_fit, T, a.chunk)
    moe_layers = [l for l in range(cfg["num_hidden_layers"]) if l >= cfg["first_k_dense_replace"]]
    acts_layers = set(moe_layers) if a.acts_layers == "all" else {int(v) for v in a.acts_layers.split(",")}
    for d in ("state", "acts", "eval/val", "eval/matched"):
        os.makedirs(f"{a.out}/{d}", exist_ok=True)
    protocol = dict(
        schema="nestquant-19-glm53-full-capture-v1", fit_start=a.fit_start, fit_windows=a.fit_windows, fit_tokens=n_fit,
        val_windows=[corpus.windows[a.fit_windows]] + [corpus.windows[-1]] if corpus.n_val_windows else [],
        corpus=a.corpus, corpus_tokens_sha256=json.load(open(f"{a.corpus}/manifest.json")).get("tokens_sha256"),
        source=src.root, chunk0=plan[0], chunk=a.chunk,
        arithmetic="orbit-duet pilot: FP8 decoded to BF16; document-local attention per 512-window; "
                   "expert-index-ordered BF16 accumulation (4096-row expert batches) before shared + residual",
        torch=torch.__version__)
    if corpus.group:                                     # (old-corpus protocol kept byte-identical for resumes)
        protocol.update(shard_id=a.shard_id, window_tokens=CTX, group=os.path.basename(os.path.normpath(a.corpus)),
                        arithmetic=protocol["arithmetic"].replace("per 512-window", f"per {CTX}-token window"))
    elif a.shard_id is not None and a.shard_id != a.fit_start // nq19.SHARD_WINDOWS:
        raise ValueError("old corpus: shard id is fit_start // 2048")
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
    vprotocol = dict(role="evaluation only", corpus=a.corpus,
                     windows=corpus.windows[a.fit_windows:], selection="last held-out windows below the nq-tail "
                     "(windows < 28784), document-local attention like the fit rows", arithmetic=protocol["arithmetic"])
    if corpus.group:
        vprotocol["selection"] = "val split of the group corpus (whole documents), document-local attention like the fit rows"
    if corpus.n_val_windows:
        v_doc, v_pos, v_dom, _ = val_meta(corpus, n_fit)
        try:
            import bnd19
            th, en = bnd19.corpus_flags(a.corpus)
            vw = corpus.windows[a.fit_windows:]
            v_bnd = dict(bnd_think=torch.from_numpy(np.concatenate([np.asarray(th[w]) for w in vw]).astype(np.int8)),
                         bnd_end=torch.from_numpy(np.concatenate([np.asarray(en[w]) for w in vw]).astype(np.int8)))
            vprotocol["boundary"] = ("bnd_think / bnd_end: distance 1..32 to the next think / end boundary in the same "
                                     "segment, 0 = none; exclusive (nearer, ties -> end)")
        except FileNotFoundError:
            v_bnd = {}

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
    import queue, threading
    wq, werr = queue.Queue(maxsize=6), []

    def writer():                                        # background x/ids/p appends (overlap with compute)
        while True:
            f, arr = wq.get()
            try:
                arr.tofile(f)
            except Exception as ex:
                werr.append(ex)
            wq.task_done()
    threading.Thread(target=writer, daemon=True).start()
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
        acts = (sparse and li in acts_layers and not os.path.exists(f"{a.out}/acts/L{li}/merged")
                and not os.path.exists(f"{a.out}/acts/L{li}/done.json"))      # complete acts are deterministic: never rewrite
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
                        wq.put((fx, xc.view(torch.uint8).numpy())); wq.put((fi, ic.numpy())); wq.put((fp, pc.numpy()))
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
            wq.join()
            if werr:
                raise werr[0]
            fx.close(); fi.close(); fp.close()
            if corpus.n_val_windows:
                vpay = dict(x=torch.cat(vx), ids=torch.cat(vi), p=torch.cat(vp), document_ids=v_doc,
                            token_positions=v_pos, domains=v_dom, layer=li, protocol=vprotocol, **v_bnd)
                torch.save(vpay, f"{a.out}/eval/val/layer_{li}.partial")
                os.replace(f"{a.out}/eval/val/layer_{li}.partial", f"{a.out}/eval/val/layer_{li}.pt")
                del vpay, vx, vi, vp
            write_json(f"{ad}/done.json", dict(layer=li, rows=n_fit, width=D, x="x.bf16 [rows, 6144] bf16",
                                               ids="ids.u8 [rows, 8] uint8", p="p.f32 [rows, 8] float32 (router weight incl. routed_scaling_factor)",
                                               coverage_all_rows=cov.tolist(), protocol=protocol))
        if not torch.isfinite(S[::997].float()).all():
            raise ValueError(f"nonfinite state after layer {li}")
        # ---- checkpoint (every --ckpt-every layers and at the end; progress.json records only checkpointed layers)
        tw = time.time()
        stop_now = a.max_layers and new_layers + 1 >= a.max_layers
        if li % a.ckpt_every == 0 or li == a.last_layer or stop_now:
            sp = "state_1.bf16" if prog.get("state_file") == "state_0.bf16" else "state_0.bf16"
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
