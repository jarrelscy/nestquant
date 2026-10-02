"""Block tensors for training the nq-tfpred predictor.

A stream (one task: its requests' decode rows in order) becomes per-16-token-block arrays:
  cnt  [nb, NL, NE] u8    routed hits per expert in the block (slot-weighted)
  sal  [nb, NL, NE] f16   salience-weighted hits ((rsf*w)^2 * xn summed over the block's slots; capture only)
  ans  [nb] u8            answer-phase tokens in the block (after </think>)
  rq   [nb] i32           request of the block's last token; rpos [nb] blocks since that request's first decode block
  pf   [R, NL, NE] f32    request prefill expert counts / prefill length (fraction per token); pfn [R]
  tok  [nb, 16] i32       token ids; hp [nb, P, 256] f16 block-mean MoE-input projections (capture only)
History inputs are raw block counts at two resolutions (no EMAs): the last F=32 blocks and the last C=32 chunks of 8 blocks.
Targets: hits in windows WIN after the block end (sim.WIN)."""
import numpy as np, torch, glob, json

G = 16; F = 32; C = 32; CK = 8
WIN = [0, 16, 64, 128, 256, 512, 1024]
TASKS_IDS = ['embedding-drift-monitor', 'fin-saccr-rwa', 'formal-crypto', 'freight-dispatch-shift', 'pretrain-shard-corruption', 'sound-change-cascade']
TEST_TASKS = ['freight-dispatch-shift', 'sound-change-cascade', 'embedding-drift-monitor']   # = nq-algo held-out (not in GBDT v1 / jF training)
VAL_TASKS = ['formal-crypto']
CAP_HOLDOUT = ['satb-audio-transcription']   # reserved TRUE held-out capture (long prefills 67K/96K), never train/val; final-window live load part 2


def _block_hist(ex, nb, NL, wts=None, chunk=4096):
    """per-block hit (or weight-sum) histogram [nb, NL, 256], computed chunk by chunk with bincount (low RAM)."""
    out = np.zeros((nb, NL, 256), np.uint8 if wts is None else np.float32)
    off = (np.arange(NL, dtype=np.int64) * 256)[None, :, None]
    for b0 in range(0, nb, chunk):
        b1 = min(nb, b0 + chunk); n = b1 - b0
        ix = ex[b0 * G:b1 * G].astype(np.int64).reshape(n, G, NL, 8) + off[:, None] + (np.arange(n, dtype=np.int64) * NL * 256)[:, None, None, None]
        w = None if wts is None else wts(b0 * G, b1 * G).reshape(-1)
        h = np.bincount(ix.reshape(-1), weights=w, minlength=n * NL * 256)
        out[b0:b1] = h.reshape(n, NL, 256).astype(out.dtype)
    return out


def blocks_from_stream(z, rsf=2.5, cache=None):
    ex = z['ex']; N = (len(ex) // G) * G; nb = N // G; NL = ex.shape[1]
    import os
    if cache and os.path.exists(cache):
        cnt = np.load(cache, mmap_mode="r")
    else:
        cnt = _block_hist(ex, nb, NL)
        if cache: np.save(cache, cnt)
    out = dict(cnt=cnt)
    if 'w' in z:
        W, XN = z['w'], z['xn']
        sal = _block_hist(ex, nb, NL, wts=lambda a, b: (rsf * W[a:b].astype(np.float32)) ** 2 * XN[a:b, :, None].astype(np.float32))
        out['sal'] = sal.astype(np.float16)
    th = z['think'][:N].reshape(nb, G); out['ans'] = (~th).sum(1).astype(np.uint8)
    rq = z['rq'][:N].reshape(nb, G)[:, -1]; out['rq'] = rq.astype(np.int32)
    first = np.full(len(z['pfn']), -1)
    for b in range(nb):
        if first[rq[b]] < 0: first[rq[b]] = b
    out['rpos'] = (np.arange(nb) - first[rq]).astype(np.int32)
    pfc = cache.replace('cnt-', 'pf16-') if cache else None
    if pfc and os.path.exists(pfc):
        out['pf'] = np.load(pfc, mmap_mode='r')
    else:
        pf = z['pf']; out['pf'] = np.empty(pf.shape, np.float16); den = np.maximum(z['pfn'], 1)
        for i in range(0, len(pf), 4096):
            out['pf'][i:i + 4096] = pf[i:i + 4096] / den[i:i + 4096, None, None]
        del pf
        if pfc: np.save(pfc, out['pf'])
    out['pfn'] = z['pfn']
    if 'tok' in z: out['tok'] = z['tok'][:N].reshape(nb, G).astype(np.int32)
    if 'hp' in z: out['hp'] = z['hp'][:N].astype(np.float32).reshape(nb, G, *z['hp'].shape[1:]).mean(1).astype(np.float16)
    return out


class Blocks:
    """GPU-resident block tensors of several streams, concatenated with F*C padding so history/target gathers never cross
    streams (gathers clamp to the stream's range; out-of-range history reads zeros, out-of-range targets are masked)."""
    def __init__(s, blist, dev, use_sal=False, cdev=None, mmap_dir=None):
        """mmap_dir: (CPU data only) the big arrays (cnt/sal/pf/chunk sums) are concatenated chunk-wise into
        <mmap_dir>/*.npy and memory-mapped (page cache, reclaimable) instead of held in anonymous RAM."""
        s.dev = dev; s.use_sal = use_sal; s.cdev = cdev if cdev is not None else dev   # data device / compute device
        pads = []; off = 0; s.seg = []
        cn, sl, an, rq, rp, tk, hp, pf = [], [], [], [], [], [], [], []
        roff = 0
        for d in blist:
            nb = len(d['cnt']); s.seg.append((off, off + nb)); off += nb
            cn.append(d['cnt']); an.append(d['ans']); rq.append(d['rq'] + roff); rp.append(d['rpos']); pf.append(d['pf']); roff += len(d['pf'])
            if use_sal: sl.append(d['sal'])
            if 'tok' in d: tk.append(d['tok'])
            if 'hp' in d: hp.append(d['hp'])
        s.nb = off
        def cat(name, parts, dt=None):
            if mmap_dir is None:
                return torch.from_numpy(np.concatenate(parts)).to(dev)
            import os
            os.makedirs(mmap_dir, exist_ok=True); f = f'{mmap_dir}/{name}.npy'
            shp = (sum(len(p) for p in parts),) + parts[0].shape[1:]; dt = dt or parts[0].dtype
            if not os.path.exists(f):
                o = np.lib.format.open_memmap(f + '.tmp.npy', 'w+', dt, shp); i = 0
                for p in parts:
                    for j in range(0, len(p), 2048): o[i + j:i + min(len(p), j + 2048)] = p[j:j + 2048]
                    i += len(p)
                o.flush(); del o; os.rename(f + '.tmp.npy', f)
            m = np.load(f, mmap_mode='c'); assert m.shape == shp, (f, m.shape, shp)
            return torch.from_numpy(m)
        s.cnt = cat('cnt', cn)
        s.sal = cat('sal', sl) if use_sal else None
        s.ans = torch.from_numpy(np.concatenate(an)).to(dev)
        s.rq = torch.from_numpy(np.concatenate(rq)).long().to(dev); s.rpos = torch.from_numpy(np.concatenate(rp)).to(dev)
        s.pf = cat('pf', pf)
        s.tok = torch.from_numpy(np.concatenate(tk)).to(dev) if len(tk) == len(blist) else None
        s.hp = torch.from_numpy(np.concatenate(hp)).to(dev) if len(hp) == len(blist) else None
        s.lo = torch.zeros(s.nb, dtype=torch.long, device=dev); s.hi = torch.zeros_like(s.lo)
        for a, b in s.seg: s.lo[a:b] = a; s.hi[a:b] = b
        val = s.sal if use_sal else s.cnt
        # chunk sums of CK blocks aligned to absolute block index (per stream alignment is irrelevant: chunks never mix
        # streams because a chunk touching a boundary is masked by lo)
        nc = (s.nb + CK - 1) // CK
        if mmap_dir is None:
            vv = torch.zeros(nc * CK, *val.shape[1:], dtype=torch.float16 if use_sal else torch.uint8, device=dev); vv[:s.nb] = val
            s.ch = vv.view(nc, CK, *val.shape[1:]).sum(1, dtype=torch.float32).to(torch.float16)
        else:
            import os
            f = f'{mmap_dir}/ch_{"sal" if use_sal else "cnt"}.npy'
            if not os.path.exists(f):
                o = np.lib.format.open_memmap(f + '.tmp.npy', 'w+', np.float16, (nc,) + tuple(val.shape[1:]))
                for c0 in range(0, nc, 256):
                    c1 = min(nc, c0 + 256); v = val[c0 * CK:min(s.nb, c1 * CK)].float()
                    if len(v) < (c1 - c0) * CK: v = torch.cat([v, v.new_zeros((c1 - c0) * CK - len(v), *v.shape[1:])])
                    o[c0:c1] = v.view(c1 - c0, CK, *v.shape[1:]).sum(1).half().numpy()
                o.flush(); del o; os.rename(f + '.tmp.npy', f)
            s.ch = torch.from_numpy(np.load(f, mmap_mode='c'))
        s.clo = s.lo // CK

    def sample_index(s, n, gen, min_hist=0):
        b = torch.randint(0, s.nb, (n,), generator=gen, device='cpu').to(s.dev)
        return b

    def _g(s, t, i):
        return t[i].to(s.cdev, non_blocking=True)

    def inputs(s, b):
        """b [B] block indices (the refresh at the end of block b). returns dict of model inputs (on cdev)."""
        val = s.sal if s.use_sal else s.cnt
        b = b.to(s.dev)
        k = torch.arange(F, device=s.dev)
        fi = b[:, None] - (F - 1) + k[None]                          # [B,F] oldest..newest
        fm = (fi >= s.lo[b][:, None]).to(s.cdev)
        fh = s._g(val, fi.clamp(min=0)).float() * fm[:, :, None, None]   # [B,F,NL,NE]
        # coarse: chunks fully before the fine window
        c_end = (b - F + 1) // CK                                   # first chunk index not fully covered... chunks < c_end
        ci = c_end[:, None] - C + torch.arange(C, device=s.dev)[None]
        cm = ((ci * CK >= s.lo[b][:, None]) & (ci >= 0)).to(s.cdev)
        chh = s._g(s.ch, ci.clamp(min=0)).float() * cm[:, :, None, None]   # [B,C,NL,NE]
        # answer tokens in the fine window, prefill of the current request, request position
        an = s._g(s.ans, fi.clamp(min=0)).float() * fm
        rq = s.rq[b]
        out = dict(fine=fh, coarse=chh, ans=an / G, pf=s._g(s.pf, rq).float(), rpos=s._g(s.rpos, b).float())
        if s.tok is not None:
            ti = b[:, None] - 3 + torch.arange(4, device=s.dev)[None]
            out['tok'] = s._g(s.tok, ti.clamp(min=0)).reshape(len(b), 64).long()
            out['tokm'] = (ti >= s.lo[b][:, None]).repeat_interleave(G, 1).to(s.cdev)
        if s.hp is not None:
            hi_ = (b[:, None] - 3 + torch.arange(4, device=s.dev)[None])
            out['hp'] = s._g(s.hp, hi_.clamp(min=0)).float() * (hi_ >= s.lo[b][:, None]).to(s.cdev)[:, :, None, None]   # [B,4,P,256]
        return out

    def targets(s, b, wins=WIN):
        """hits (or salience) per window after the end of block b: [B, NL, NE, W], mask [B, W] (window inside stream)."""
        val = s.sal if s.use_sal else s.cnt
        b = b.to(s.dev)
        nwb = wins[-1] // G
        fi = b[:, None] + 1 + torch.arange(nwb, device=s.dev)[None]
        fm = (fi < s.hi[b][:, None]).to(s.cdev)
        fv = s._g(val, fi.clamp(max=s.nb - 1)).float() * fm[:, :, None, None]     # [B,64,NL,NE]
        cs = torch.cat([torch.zeros_like(fv[:, :1]), fv.cumsum(1)], 1)
        e = [w // G for w in wins]
        y = torch.stack([cs[:, e[i + 1]] - cs[:, e[i]] for i in range(len(e) - 1)], -1)
        m = torch.stack([fm[:, e[i + 1] - 1] for i in range(len(e) - 1)], -1)
        return y, m


def npz_mmap(f):
    """zero-copy read-only memmaps of an UNCOMPRESSED npz's members (falls back to lazy np.load for compressed members)."""
    import zipfile
    out = {}; lz = None
    with zipfile.ZipFile(f) as zf:
        for i in zf.infolist():
            k = i.filename[:-4]
            if i.compress_type != 0:
                lz = lz or np.load(f); out[k] = lz[k]; continue
            with open(f, 'rb') as fh:
                fh.seek(i.header_offset); h = fh.read(30)
                n, m = int.from_bytes(h[26:28], 'little'), int.from_bytes(h[28:30], 'little')
                fh.seek(i.header_offset + 30 + n + m)
                v = np.lib.format.read_magic(fh)
                sh, fo, dt = (np.lib.format.read_array_header_1_0 if v == (1, 0) else np.lib.format.read_array_header_2_0)(fh)
                off = fh.tell()
            out[k] = np.memmap(f, dt, 'r', off, sh, 'F' if fo else 'C') if int(np.prod(sh)) else np.zeros(sh, dt)
    return out


def ids_blocks(task, kind='ids'):
    f = f'/rawdata/Jarrel/nq-tfpred/ds/{kind}-{task}.npz'
    return blocks_from_stream(npz_mmap(f), cache=f'/rawdata/Jarrel/nq-tfpred/ds/cnt-{kind}-{task}.npy')


def load_ids(task):
    z = dict(np.load(f'/rawdata/Jarrel/nq-tfpred/ds/ids-{task}.npz'))
    return z
