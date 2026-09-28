"""06-expert-objective: EXL3 quantizer (upstream quantize_exl3, custom H / H_out) under different expert-level objectives.
Fitting uses only training statistics + training_sample rows; evaluation captures are used only for final scoring."""
import json, math, time, sys
from pathlib import Path
import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = False
OD = '/home/coder/git/orbit-duet'
sys.path.insert(0, OD)
from orbit_duet.source import weights as load_weights
from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3, get_temp_buffers
from exllamav3.modules.quant.exl3_lib import quantize as qz

CFG = {
    'glm': dict(source='/tmp/orbit-duet-glm53-fp8', layer=16, expert=36,
                stats=f'{OD}/runs/glm53_pilot_matched_l16/statistics/l16_e36.pt',
                sample=f'{OD}/runs/glm53_pilot_matched_l16/statistics/l16_e36_training_sample.pt',
                exl3=f'{OD}/runs/glm53_pilot_matched_l16/exl3_e36',
                evals=[('ctx', f'{OD}/runs/glm53_matched_context_pilot_v1_capture/layer_16.pt')]),
    'mimo': dict(source=f'{OD}/runs/source_mimo', layer=55, expert=70,
                 stats=f'{OD}/runs/full55_statistics/l55_e70.pt',
                 sample=f'{OD}/runs/full55_statistics/l55_e70_training_sample.pt',
                 exl3=f'{OD}/runs/full55_exl3_e70',
                 evals=[('control', f'{OD}/runs/native_id_control_v1_capture/layer_55.pt'),
                        ('ood', f'{OD}/runs/ood_controlled_v1_capture/layer_55.pt')]),
}


def setup():
    torch.cuda.set_per_process_memory_fraction(12 / 80)
    torch.set_num_threads(8)


def teacher(x, w):
    g, u, d = w
    return F.linear(F.silu(F.linear(x.bfloat16(), g.bfloat16())) * F.linear(x.bfloat16(), u.bfloat16()), d.bfloat16()).float()


class Data:
    def __init__(self, model):
        c = CFG[model]; self.c = c; self.model = model
        self.w = [t.cuda().float() for t in load_weights(c['source'], c['layer'], c['expert'])]
        st = torch.load(c['stats'], weights_only=True, mmap=True)
        self.grams = [t.float() for t in st['grams']]; self.outputs = [t.float() for t in st['outputs']]
        self.meta = st['metadata']; self.count = self.meta['training_rows']
        s = torch.load(c['sample'], weights_only=True, mmap=True)
        self.sx = s['x']; self.sp = s['p'].float()
        self.full_scale = self.meta['mass'] / float(self.sp.square().sum())  # sample -> full-stat mass ratio

    # --- sample-row helpers (training only) ---
    def batches(self, bs=2048):
        for i in range(0, len(self.sx), bs):
            yield self.sx[i:i + bs].cuda(), self.sp[i:i + bs].cuda()

    @torch.no_grad()
    def hidden(self, x, g, u):
        return (F.silu(F.linear(x.bfloat16(), g.bfloat16())) * F.linear(x.bfloat16(), u.bfloat16())).float()

    @torch.no_grad()
    def sample_grams(self, pw=2.0, gq=None, uq=None):
        """X-gram and A-grams on sample rows weighted p**pw. If gq/uq given also A_q grams and cross A^T A_q."""
        g, u, d = self.w
        Hx = torch.zeros(6144, 6144, device='cuda'); Ha = torch.zeros(2048, 2048, device='cuda')
        Hq = torch.zeros_like(Ha) if gq is not None else None; C = torch.zeros_like(Ha) if gq is not None else None
        Og = torch.zeros_like(Ha); Ou = torch.zeros_like(Ha)
        for x, p in self.batches():
            r = p.pow(pw / 2)[:, None]
            xf = x.float(); Hx.addmm_((xf * r).T, xf * r)
            a = self.hidden(x, g, u) * r; Ha.addmm_(a.T, a)
            gx, ux = F.linear(xf, g), F.linear(xf, u); sig = gx.sigmoid()
            dg = ux * sig * (1 + gx * (1 - sig)) * r; du = F.silu(gx) * r
            Og.addmm_(dg.T, dg); Ou.addmm_(du.T, du)
            if gq is not None:
                aq = self.hidden(x, gq, uq) * r
                Hq.addmm_(aq.T, aq); C.addmm_(a.T, aq)
        return dict(Hx=Hx, Ha=Ha, Hq=Hq, C=C, Og=Og, Ou=Ou)

    @torch.no_grad()
    def train_error(self, wq, pw=2.0):
        num = den = 0.
        for x, p in self.batches():
            t = teacher(x, self.w).double(); e = (teacher(x, wq).double() - t).square().sum(-1)
            q = p.double().pow(pw); num += float((e * q).sum()); den += float((t.square().sum(-1) * q).sum())
        return (num / den) ** .5

    @torch.no_grad()
    def evaluate(self, wq):
        """forced (p=1 all rows), routed (actual routed rows weighted p^2), separated by control/ood domain."""
        res = {}
        E = self.c['expert']
        for name, path in self.c['evals']:
            cap = torch.load(path, weights_only=True, mmap=True)
            domains = cap['domains']; docs = cap['document_ids']
            is_ood = torch.tensor([(name == 'ood') or domains[int(d)].startswith('ood') for d in docs])
            rid, slot = torch.where(cap['ids'] == E); rp = torch.zeros(len(cap['x'])); rp[rid] = cap['p'][rid, slot]
            acc = {}
            for i in range(0, len(cap['x']), 256):
                x = cap['x'][i:i + 256].cuda(); t = teacher(x, self.w).double()
                e = (teacher(x, wq).double() - t).square().sum(-1).cpu(); en = t.square().sum(-1).cpu()
                o = is_ood[i:i + 256]; pr = rp[i:i + 256].double().square()
                for key, m, wt in [('forced_id', ~o, None), ('forced_ood', o, None), ('routed_id', ~o, pr), ('routed_ood', o, pr),
                                   ('forced_all', torch.ones_like(o), None), ('routed_all', torch.ones_like(o), pr)]:
                    ww = m.double() if wt is None else m.double() * wt
                    a = acc.setdefault(key, [0., 0., 0]); a[0] += float((e * ww).sum()); a[1] += float((en * ww).sum()); a[2] += int((ww > 0).sum())
            for k, (n, dd, cnt) in acc.items():
                if cnt: res[f'{name}:{k}'] = dict(rel=(n / dd) ** .5, rows=cnt)
        return res


QA = dict(devices=['cuda:0'], seed=91426, sigma_reg=.03, apply_out_scales=None, mul1=True)


def quant(W, H, count, bits, H_out=None, **extra):
    """W: (out, in) float cuda. H: (in,in) sum-gram (divided by count inside). Returns decoded (out,in) and proxy."""
    from exllamav3.modules.quant.exl3 import LinearEXL3
    qa = dict(QA, K=bits, **extra)
    if H_out is not None: qa['H_out'] = H_out.cuda().float()
    hd = dict(H=H.cuda().float().clone(), count=count, finalized=False, device=torch.device('cuda:0'))
    with torch.no_grad():
        _, proxy, val = quantize_exl3(W.T.contiguous().float(), hd, qa, False, verbose=False)
    lin = LinearEXL3(None, W.shape[1], W.shape[0], **{k: v for k, v in val.items() if torch.is_tensor(v)})
    Wq = lin.get_weight_tensor().T.float().contiguous()
    get_temp_buffers.cache_clear(); del hd, lin, val; torch.cuda.empty_cache()
    return Wq, proxy


def rotated_spread(G):
    """AM/GM of diag(G), and of LDL-D of G after EXL3's 128-blockwise output Hadamard (random signs)."""
    G = G.cuda().float(); G = G / G.diagonal().mean()
    d0 = G.diagonal(); s = (torch.randn(G.shape[0], device='cuda').sign())
    R = G * s[None] * s[:, None]; qz.blockwise_preapply_had_r_(R, 128); qz.blockwise_preapply_had_l_(R, 128)
    R.diagonal().add_(0.03 * R.diagonal().mean()); Gd = G.clone(); Gd.diagonal().add_(0.03 * Gd.diagonal().mean())
    def amgm(v): v = v.double().clamp_min(1e-30); return float(v.mean() / v.log().mean().exp())
    Dr = torch.linalg.cholesky(R.double()).diagonal().square(); D0 = torch.linalg.cholesky(Gd.double()).diagonal().square()
    return dict(diag_amgm=amgm(d0), rot_diag_amgm=amgm(R.diagonal()), ldl_amgm=amgm(D0), rot_ldl_amgm=amgm(Dr))
