"""Thread 09: continuous post-tuning of fixed-code EXL3 artifacts against real SwiGLU expert outputs."""
import json, math, os
import torch
import torch.nn.functional as F

ROOT = '/home/coder/git/orbit-duet'
SCRATCH = '/tmp/nestquant/09-post-tuning'
CFG = {
    'glm': dict(L=16, E=36, source='/tmp/orbit-duet-glm53-fp8',
                sample=f'{ROOT}/runs/glm53_pilot_matched_l16/statistics/l16_e36_training_sample.pt',
                exl3=f'{ROOT}/runs/glm53_pilot_matched_l16/exl3_e36',
                captures={'pilot': f'{ROOT}/runs/glm53_matched_context_pilot_v1_capture/layer_16.pt'}),
    'mimo': dict(L=55, E=70, source='/tmp/mimo-a100/data/jarrel/mimo-exl3-sequential/source',
                 sample=f'{ROOT}/runs/full55_statistics/l55_e70_training_sample.pt',
                 exl3=f'{ROOT}/runs/exl3_statistics_only_check_l55_e70',
                 captures={'control': f'{ROOT}/runs/native_id_control_v1_capture/layer_55.pt',
                           'ood': f'{ROOT}/runs/ood_controlled_v1_capture/layer_55.pt'}),
}
SHAPES = [(2048, 6144), (2048, 6144), (6144, 2048)]  # (out, in) for gate, up, down


def setup():
    torch.cuda.set_per_process_memory_fraction(12 / 80)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(8)


def teacher_weights(model):
    from orbit_duet.source import weights
    from orbit_duet.statistics import tensor_hash
    c = CFG[model]
    w = weights(c['source'], c['L'], c['E'])
    s = torch.load(c['sample'], mmap=True, weights_only=True)
    assert [tensor_hash(t) for t in w] == list(s['teacher_tensor_sha256']), 'teacher hash mismatch'
    return w


def load_exl3(model, bits):
    """Return list of dicts (Q rotated inner weight [in,out] fp32, suh [in], svh [out]) + raw EXL3Expert."""
    from orbit_duet.exl3_adapter import EXL3Expert
    e = EXL3Expert(f"{CFG[model]['exl3']}/expert_{bits}.bin")
    out = []
    for o in e.objects:
        Q = o.get_inner_weight_tensor().float()
        out.append(dict(Q=Q, suh=o.suh.float().clone(), svh=o.svh.float().clone()))
    ref = e.decoded_weights()
    return out, ref


_H = {}
def had(dev):
    if dev not in _H:
        from exllamav3.modules.quant.exl3_lib.quantize import get_hadamard_dt
        _H[dev] = get_hadamard_dt(128, dev, torch.float32, 1 / math.sqrt(128))
    return _H[dev]


def had_l(x):
    k, n = x.shape
    return (had(x.device) @ x.view(-1, 128, n)).view(k, n)


def had_r(x):
    k, n = x.shape
    return (x.view(k, -1, 128) @ had(x.device)).view(k, n)


class Tuned(torch.nn.Module):
    """W^T[in,out] = diag(suh e^a) H [ remap(Q) * e^(r_in x r_out) * e^tile ] H diag(svh e^b) + U V ; y += bias."""
    def __init__(self, parts, groups, rank=0, lut_knots=33, tile=16):
        super().__init__()
        self.groups = set(groups)
        self.rank = rank
        self.tile = tile
        for i, p in enumerate(parts):
            k, n = p['Q'].shape
            self.register_buffer(f'Q{i}', p['Q'])
            self.register_buffer(f'su{i}', p['suh'])
            self.register_buffer(f'sv{i}', p['svh'])
            z = lambda *s: torch.nn.Parameter(torch.zeros(*s, device=p['Q'].device))
            if 'suv' in self.groups:
                setattr(self, f'a{i}', z(k)); setattr(self, f'b{i}', z(n))
            if 'rot' in self.groups:
                setattr(self, f'ri{i}', z(k)); setattr(self, f'ro{i}', z(n))
            if 'tile' in self.groups:
                setattr(self, f't{i}', z(k // tile, n // tile))
            if 'lut' in self.groups or 'vlut' in self.groups:
                u, inv = torch.unique(p['Q'], return_inverse=True)
                self.register_buffer(f'lu{i}', u); self.register_buffer(f'li{i}', inv.to(torch.int32))
                if 'lut' in self.groups:  # smooth piecewise-linear remap, lut_knots knots
                    lo, hi = float(u.min()), float(u.max())
                    pos = (u - lo) / (hi - lo) * (lut_knots - 1)
                    idx = pos.floor().clamp(0, lut_knots - 2)
                    self.register_buffer(f'lk{i}', idx.long()); self.register_buffer(f'lf{i}', pos - idx)
                    setattr(self, f'ld{i}', z(lut_knots))
                else:  # free per-distinct-value table
                    setattr(self, f'ld{i}', z(len(u)))
            if 'bias' in self.groups:
                setattr(self, f'bias{i}', z(n))
            if rank:
                setattr(self, f'U{i}', z(k, rank))
                g = torch.Generator(device='cpu').manual_seed(1234 + i)
                V = torch.randn(rank, n, generator=g).to(p['Q'].device) * 1e-2
                setattr(self, f'V{i}', torch.nn.Parameter(V))

    def weight(self, i):
        g = self.groups
        Q = getattr(self, f'Q{i}')
        k, n = Q.shape
        if 'lut' in g or 'vlut' in g:
            d = getattr(self, f'ld{i}')
            if 'lut' in g:
                k_ = getattr(self, f'lk{i}'); f_ = getattr(self, f'lf{i}')
                du = d[k_] * (1 - f_) + d[k_ + 1] * f_
            else:
                du = d
            Q = Q + du[getattr(self, f'li{i}').long()]
        if 'rot' in g:
            Q = Q * getattr(self, f'ri{i}').exp()[:, None] * getattr(self, f'ro{i}').exp()[None, :]
        if 'tile' in g:
            t = self.tile
            Q = (Q.view(k // t, t, n // t, t) * getattr(self, f't{i}').exp()[:, None, :, None]).view(k, n)
        su = getattr(self, f'su{i}'); sv = getattr(self, f'sv{i}')
        if 'suv' in g:
            su = su * getattr(self, f'a{i}').exp(); sv = sv * getattr(self, f'b{i}').exp()
        W = had_r(had_l(Q) * su[:, None]) * sv[None, :]
        if self.rank:
            W = W + getattr(self, f'U{i}') @ getattr(self, f'V{i}')
        return W  # [in, out]

    def biases(self):
        if 'bias' not in self.groups:
            return [None] * 3
        return [getattr(self, f'bias{i}') for i in range(3)]

    def dense(self):
        """Evaluator-layout weights (out,in) fp32 and biases."""
        with torch.no_grad():
            return [self.weight(i).T.contiguous() for i in range(3)], [None if b is None else b.detach().clone() for b in self.biases()]

    def forward(self, x):
        Ws = [self.weight(i).T.bfloat16() for i in range(3)]
        bs = [None if b is None else b.bfloat16() for b in self.biases()]
        return expert(x, Ws, bs)

    def extra_bits(self):
        """Extra stored bits beyond native EXL3 (su/sv tuning is free)."""
        bits = 0
        for i, (o, inn) in enumerate(SHAPES):
            k, n = inn, o
            if 'rot' in self.groups: bits += 16 * (k + n)
            if 'tile' in self.groups: bits += 8 * (k // self.tile) * (n // self.tile)  # 8-bit log-scale code
            if 'lut' in self.groups or 'vlut' in self.groups: bits += 16 * getattr(self, f'ld{i}').numel()
            if 'bias' in self.groups: bits += 16 * n
            if self.rank: bits += 16 * self.rank * (k + n)
        return bits / (3 * 2048 * 6144)


def expert(x, Ws, bs=(None, None, None)):
    """Same arithmetic as orbit_duet.evaluate.teacher (BF16), optional biases."""
    x = x.bfloat16()
    g = F.linear(x, Ws[0].bfloat16(), None if bs[0] is None else bs[0].bfloat16())
    u = F.linear(x, Ws[1].bfloat16(), None if bs[1] is None else bs[1].bfloat16())
    return F.linear(F.silu(g) * u, Ws[2].bfloat16(), None if bs[2] is None else bs[2].bfloat16()).float()


@torch.no_grad()
def eval_captures(model, Wt, cand):
    """cand: dict name -> (Ws, bs). Returns nested dict of relative L2 (%) for forced/routed per group."""
    c = CFG[model]; E = c['E']
    res = {}
    for cname, path in c['captures'].items():
        cap = torch.load(path, weights_only=True, mmap=True)
        assert cap['layer'] == c['L']
        doms = cap['domains']
        docdom = torch.tensor([0 if not d.startswith('ood:') else 1 for d in doms]) if any(d.startswith('ood:') for d in doms) else None
        N = len(cap['x'])
        groups = {'all': torch.arange(N)}
        if docdom is not None:
            dd = docdom[cap['document_ids']]
            groups['control'] = (dd == 0).nonzero().flatten(); groups['ood'] = (dd == 1).nonzero().flatten()
        routed, slots = torch.where(cap['ids'] == E)
        for gname, rows in groups.items():
            take = torch.isin(routed, rows)
            for mode, (rr, pr) in {'forced': (rows, torch.ones(len(rows))),
                                   'routed': (routed[take], cap['p'][routed[take], slots[take]])}.items():
                if len(rr) == 0: continue
                num = {k: 0. for k in cand}; den = 0.
                for s in range(0, len(rr), 256):
                    ids = rr[s:s + 256]; x = cap['x'][ids].cuda(); p2 = pr[s:s + 256].cuda().double().square()
                    t = expert(x, Wt).double(); den += float((t.square().sum(-1) * p2).sum())
                    for k, (Ws, bs) in cand.items():
                        num[k] += float(((expert(x, Ws, bs).double() - t).square().sum(-1) * p2).sum())
                res[f'{cname}/{gname}/{mode}'] = dict(rows=len(rr), **{k: 100 * (v / den) ** .5 for k, v in num.items()})
    return res

# GLM EXL3 refit on the 90% tuning split only (refit90.py): the 10% held-out split is clean for codes and tuning
CFG['glm90'] = dict(CFG['glm'], exl3=f'{SCRATCH}/exl3_90_glm')
# GLM EXL3 refit on the full saved statistics with sigma_reg=0.3 (refit_sweep.py, KEEP=1)
CFG['glms03'] = dict(CFG['glm'], exl3=f'{SCRATCH}/exl3_s03_glm')
# other GLM pilot experts (for the sigma_reg generalisation check)
for _L in (16, 49, 66):
    for _E in (36, 92, 165):
        CFG[f'glm_{_L}_{_E}'] = dict(L=_L, E=_E, source='/tmp/orbit-duet-glm53-fp8',
            sample=f'{ROOT}/runs/glm53_pilot_matched_l{_L}/statistics/l{_L}_e{_E}_training_sample.pt',
            exl3=f'{ROOT}/runs/glm53_pilot_matched_l{_L}/exl3_e{_E}',
            captures={'pilot': f'{ROOT}/runs/glm53_matched_context_pilot_v1_capture/layer_{_L}.pt'})
