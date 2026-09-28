"""T24 fast variant of capture_fwd.py (same outputs, byte for byte; same protocol.json, so it can resume a shard
started by capture_fwd.py and vice versa).  Speedups, none of which changes the arithmetic:
  * super-chunks (--super-rows): consecutive original chunks share one pass over the 256 experts, so each expert's
    FP8 weights are copied + dequantised once per super-chunk instead of once per 131072-row chunk; inside, every
    original chunk keeps its own argsort and 4096-row expert batches (the expert GEMMs are M-dependent);
  * expert weights: one H2D copy per expert, double-buffered on a side stream (prefetch e+1 while e computes);
    dequant via a broadcast block view (the same fp32 products as repeat_interleave, no 50 MB scale temporaries);
  * pinned host state S, async D2H of the residual/state and of the acts into a pinned ring (disk writes in a thread);
  * the state checkpoint is written by a background thread chunk by chunk (the next layer waits per chunk before
    overwriting it; progress.json advances only when the file is complete);
  * window segments preloaded on the GPU.  Front stays one window per call (batching windows changes router bits).

Original docstring:
Thread 19 stage 1: sequential FP8-reference forward of GLM-5.3 over the calibration windows.

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


class Prefetch:
    """Double-buffered per-expert H2D of one layer's raw FP8 + scales (ExpertCache.buf is pinned, experts contiguous)."""

    def __init__(self, cache, nslot=2):
        self.cache = cache
        self.stream = torch.cuda.Stream()
        self.esz = cache.size // NEXP
        self.bufs = [torch.empty(self.esz, dtype=torch.uint8, device="cuda") for _ in range(nslot)]
        self.free = [None] * nslot
        self.ready = [None] * nslot

    def bind(self, layer):
        c = self.cache
        self.layer = layer
        self.start_of = []
        for e in range(NEXP):
            p0 = c.views[f"model.layers.{layer}.mlp.experts.{e}.gate_proj.weight"][0]
            if p0 != e * self.esz:
                raise ValueError("experts not contiguous / equal-sized in ExpertCache")
            self.start_of.append(p0)

    def start(self, e, slot):
        with torch.cuda.stream(self.stream):
            if self.free[slot] is not None:
                self.stream.wait_event(self.free[slot])
            self.bufs[slot].copy_(self.cache.buf[e * self.esz:(e + 1) * self.esz], non_blocking=True)
            ev = torch.cuda.Event(); ev.record(self.stream)
        self.ready[slot] = ev

    def weights(self, e, slot):
        from nq_io import DT
        torch.cuda.current_stream().wait_event(self.ready[slot])
        buf, base, out = self.bufs[slot], e * self.esz, []
        for p in nq19.PROJ:
            t = []
            for suf in (".weight", ".weight_scale_inv"):
                pos, nb, dt, shape = self.cache.views[f"model.layers.{self.layer}.mlp.experts.{e}.{p}{suf}"]
                t.append(buf[pos - base:pos - base + nb].view(DT[dt]).view(shape))
            out.append(dequant_bf16(t[0], t[1], self.cache.src.block))
        ev = torch.cuda.Event(); ev.record()
        self.free[slot] = ev
        return out


def dequant_bf16(w, s, block):
    """== orbit dequantize(w, s, block).to(bf16) bitwise (same fp32 product per element), without the expanded scale."""
    r, c = w.shape
    br, bc = block
    if r % br or c % bc or s.dtype != torch.float32 or tuple(s.shape) != (r // br, c // bc):
        return nq19.dequant(w, s, block).to(torch.bfloat16)
    return (w.view(r // br, br, c // bc, bc).float() * s[:, None, :, None]).view(r, c).to(torch.bfloat16)


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
    ap.add_argument("--super-rows", type=int, default=262144, help="T24: rows per super-chunk (GPU: 24.6 KB/row for XS+OUT)")
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
        if prog["protocol"] != json.loads(json.dumps(protocol)):
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

    # ---- state (T24: pinned host state -> 4-10x faster H2D/D2H than pageable)
    S = torch.empty(T, D, dtype=torch.bfloat16, pin_memory=True)
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
    pf = Prefetch(cache) if cache is not None else None
    # segments of every window preloaded on the GPU (same dtype/shape as corpus.seg(w))
    seg0 = corpus.seg(0)
    segs_gpu = torch.from_numpy(np.stack([np.asarray(corpus.segs[w]) for w in corpus.windows]).astype(seg0.numpy().dtype)).cuda()
    # super-chunks: consecutive plan chunks (never mixing fit / val rows); each expert is loaded once per super-chunk
    groups, cur = [], []
    for (r0, n) in plan:
        if cur and (sum(c[1] for c in cur) + n > a.super_rows or (r0 >= n_fit) != (cur[0][0] >= n_fit)):
            groups.append(cur); cur = []
        cur.append((r0, n))
    groups.append(cur)
    gmax = max(sum(c[1] for c in g) for g in groups)
    smax = max(n for _, n in plan)
    XS = torch.empty(gmax, D, dtype=torch.bfloat16, device="cuda")        # MoE inputs of the super-chunk
    OB = torch.empty(gmax, D, dtype=torch.bfloat16, device="cuda")       # front scratch (state -> residual), then MoE out
    IDS = torch.empty(gmax, 8, dtype=torch.long, device="cuda"); GP = torch.empty(gmax, 8, device="cuda")
    chunk_idx = {r0: i for i, (r0, _) in enumerate(plan)}
    t_start = time.time()
    new_layers = 0
    import queue, threading
    # acts: ring of pinned host slots, async D2H on the compute stream, written by a background thread
    NSLOT = 3
    slots = [(torch.empty(smax, D, dtype=torch.bfloat16, pin_memory=True), torch.empty(smax, 8, dtype=torch.uint8, pin_memory=True),
              torch.empty(smax, 8, dtype=torch.float32, pin_memory=True)) for _ in range(NSLOT)]
    free_slots = queue.Queue()
    for i in range(NSLOT):
        free_slots.put(i)
    wq, werr = queue.Queue(), []

    def writer():
        while True:
            ev, s, n, files = wq.get()
            try:
                ev.synchronize()
                for f, t in zip(files, slots[s]):
                    t[:n].view(torch.uint8).numpy().tofile(f)
            except Exception as ex:
                werr.append(ex)
            free_slots.put(s)
            wq.task_done()
    threading.Thread(target=writer, daemon=True).start()
    # background state checkpoint: writes S chunk by chunk; the next layer waits per chunk before overwriting it
    ck = dict(thread=None, events=None, sp=None, li=None, err=None)

    def ckpt_writer(path, M_cpu, events):
        try:
            with open(path, "wb") as f:
                for (r0, n), ev in zip(plan, events):
                    f.write(memoryview(S[r0:r0 + n].view(torch.uint8).numpy().reshape(-1)))
                    ev.set()
            torch.save(M_cpu, path + ".matched.pt")
        except Exception as ex:
            ck["err"] = ex
            for ev in events:
                ev.set()

    def ck_commit(block):
        th = ck["thread"]
        if th is None or (not block and th.is_alive()):
            return 0.
        tw_ = time.time()
        th.join()
        if ck["err"]:
            raise ck["err"]
        prog["next_layer"] = ck["li"] + 1; prog["state_file"] = ck["sp"]
        ck.update(thread=None, events=None)
        return time.time() - tw_

    def wait_slice(r0):
        evs = ck["events"]
        if evs is not None:
            evs[chunk_idx[r0]].wait()

    for li in range(first_layer, a.last_layer + 1):
        while ((a.acts_budget_gb and pending_acts_gb(os.path.dirname(os.path.abspath(a.out))) > a.acts_budget_gb)
               or free_gb(a.out) < 200 + 50):
            time.sleep(30)                                   # back-pressure: let stage 2 merge / delete; disk floor
        t0 = time.time()
        layer = Layer(src, li)
        sparse = layer.sparse
        tim = dict(load=0., front=0., moe=0., write=0., matched=0., ckpt_wait=0.)
        if sparse:
            cache.load(li); pf.bind(li)
            shared = [src.weight(f"model.layers.{li}.mlp.shared_experts.{p}.weight") for p in nq19.PROJ]
        else:
            dense = [src.weight(f"model.layers.{li}.mlp.{p}.weight") for p in nq19.PROJ]
        tim["load"] = time.time() - t0
        acts = (sparse and li in acts_layers and not os.path.exists(f"{a.out}/acts/L{li}/merged")
                and not os.path.exists(f"{a.out}/acts/L{li}/done.json"))      # complete acts are deterministic: never rewrite
        if acts:
            ad = f"{a.out}/acts/L{li}"
            os.makedirs(ad, exist_ok=True)
            files = (open(f"{ad}/x.bf16", "wb"), open(f"{ad}/ids.u8", "wb"), open(f"{ad}/p.f32", "wb"))
            vx, vi, vp = [], [], []
        cov = np.zeros(NEXP, np.int64)
        for grp in groups:
            ta = time.time()
            subs, off = [], 0
            for (r0, n) in grp:
                buf = OB[off:off + n]
                buf.copy_(S[r0:r0 + n], non_blocking=True)
                st = buf.view(n // CTX, CTX, D)
                if sparse:
                    xs = XS[off:off + n].view(n // CTX, CTX, D); ids = IDS[off:off + n]; gp = GP[off:off + n]
                for k in range(n // CTX):
                    w0 = r0 // CTX + k
                    r, x = layer.front(st[k:k + 1], segments=segs_gpu[w0:w0 + 1])
                    if sparse:
                        i_, g_ = layer.route(x.flatten(0, 1))
                        ids[k * CTX:(k + 1) * CTX] = i_; gp[k * CTX:(k + 1) * CTX] = g_.float()
                        st[k] = r[0]; xs[k] = x[0]
                    else:
                        st[k] = (r + ffn(x, dense))[0]
                del r, x
                tw_ = time.time(); wait_slice(r0); tim["ckpt_wait"] += time.time() - tw_
                S[r0:r0 + n].copy_(buf, non_blocking=True)        # sparse: residual parked on the host; dense: new state
                if acts:
                    tc = time.time()
                    if r0 < n_fit:
                        s = free_slots.get()
                        hx, hi, hp = slots[s]
                        hx[:n].copy_(XS[off:off + n], non_blocking=True)
                        hi[:n].copy_(ids.to(torch.uint8), non_blocking=True)
                        hp[:n].copy_(gp, non_blocking=True)
                        ev = torch.cuda.Event(); ev.record()
                        wq.put((ev, s, n, files))
                    else:
                        vx.append(XS[off:off + n].cpu()); vi.append(ids.cpu()); vp.append(gp.cpu())
                    tim["write"] += time.time() - tc
                subs.append((r0, n, off)); off += n
            G = off
            torch.cuda.synchronize(); tim["front"] += time.time() - ta
            if not sparse:
                continue
            tb = time.time()
            tabs = []
            for (r0, n, off_) in subs:                      # per original chunk: identical sort / batches
                flat = IDS[off_:off_ + n].reshape(-1)
                order = torch.argsort(flat, stable=True)
                rows_all = order // 8
                p_all = GP[off_:off_ + n].reshape(-1)[order]
                cnt = torch.bincount(flat, minlength=NEXP)
                cov += cnt.cpu().numpy()
                tabs.append((off_, n, rows_all, p_all, [0] + cnt.cumsum(0).tolist()))
            OB[:G].zero_()
            active = [e for e in range(NEXP) if any(t[4][e] != t[4][e + 1] for t in tabs)]
            if active:
                pf.start(active[0], 0)
            for j, e in enumerate(active):
                if j + 1 < len(active):
                    pf.start(active[j + 1], (j + 1) % 2)
                w = pf.weights(e, j % 2)
                for (off_, n, rows_all, p_all, offs) in tabs:
                    xs = XS[off_:off_ + n]; out = OB[off_:off_ + n]
                    for b in range(offs[e], offs[e + 1], 4096):
                        sel = rows_all[b:min(b + 4096, offs[e + 1])]
                        contrib = (ffn(xs[sel], w) * p_all[b:b + len(sel)][:, None]).to(torch.bfloat16)
                        out[sel] = out[sel] + contrib
                del w
            for (r0, n, off_) in subs:
                xs = XS[off_:off_ + n]; out = OB[off_:off_ + n]
                for b in range(0, n, 4096):
                    rb = S[r0 + b:r0 + min(b + 4096, n)].to("cuda", non_blocking=True)
                    value = out[b:b + 4096] + ffn(xs[b:b + 4096], shared)
                    out[b:b + 4096] = rb + value
                S[r0:r0 + n].copy_(out, non_blocking=True)
            torch.cuda.synchronize(); tim["moe"] += time.time() - tb
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
            tc = time.time()
            wq.join()
            if werr:
                raise werr[0]
            for f in files:
                f.close()
            tim["write"] += time.time() - tc
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
        # ---- checkpoint in the background (every --ckpt-every layers and at the end). progress.json's next_layer /
        # state_file advance only once the file is fully written (alternating files, like the original).
        tw = time.time()
        stop_now = a.max_layers and new_layers + 1 >= a.max_layers
        if li % a.ckpt_every == 0 or li == a.last_layer or stop_now:
            tim["ckpt_wait"] += ck_commit(block=True)
            sp = "state_1.bf16" if prog.get("state_file") == "state_0.bf16" else "state_0.bf16"
            events = [threading.Event() for _ in plan]
            ck.update(events=events, sp=sp, li=li, err=None,
                      thread=threading.Thread(target=ckpt_writer, args=(f"{a.out}/state/{sp}", M.cpu(), events), daemon=True))
            ck["thread"].start()
            if li == a.last_layer or stop_now:
                tim["ckpt_wait"] += ck_commit(block=True)
        else:
            tim["ckpt_wait"] += ck_commit(block=False)
        prog["layers"][str(li)] = dict(seconds=round(time.time() - t0, 1), **{k: round(v, 1) for k, v in tim.items()},
                                       ckpt=round(time.time() - tw, 1), sparse=sparse, impl="capture_fwd_fast",
                                       peak_cuda_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                                       peak_rss_gb=round(rss_gb(), 2))
        write_json(prog_path, prog)
        print(json.dumps(dict(layer=li, **prog["layers"][str(li)], elapsed=round(time.time() - t_start))), flush=True)
        del layer
        new_layers += 1
        if a.max_layers and new_layers >= a.max_layers:
            break
    ck_commit(block=True)
    write_json(prog_path, prog)


if __name__ == "__main__":
    main()
