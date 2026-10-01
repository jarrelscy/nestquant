"""nq-res-v2 real-artifact check + kernel test vectors (CPU). Spec: sm120/NQ_RES_V2.md.

For one expert E of layer L, TP4 rank r (shards 2r, 2r+1) of a nestquant-v1 root (campaign or shipped v1):
  1. kernel planes (sm120/nqload.kernel_expert) -> host dense decode (moe.dense_W, rotated domain = what the kernel's
     NQ_WDUMP build dumps) == the encoder-side decoder (nq_decode.ring_levels; threads/35 nq15 for base_K != 2), bitwise,
     gate|up and down at level 2 and 4
  2. serve-format round trip: streaming/resident.save -> load (res .pt) + p4rec.pack (record bytes) -> planes re-parsed
     from those bytes -> dense_W again bitwise equal (base read from the PACKED res plane, P4 from the record)
  3. writes OUT/L{L}_E{E}_r{r}.pt: the record bytes, the res fields of the expert, the expected dense weights and meta
Usage: res2_testvec.py --root ROOT --layer L --expert E [--rank 0] [--out /tmp/nestquant/35-nq15/testvec] [--code REPO]"""
import os, sys, json, types, argparse, hashlib, tempfile
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--root', required=True); ap.add_argument('--layer', type=int, required=True)
ap.add_argument('--expert', type=int, required=True); ap.add_argument('--rank', type=int, default=0)
ap.add_argument('--out', default='/tmp/nestquant/35-nq15/testvec'); ap.add_argument('--tag', default='')
ap.add_argument('--code', default='/home/coder/git/nestquant')
a = ap.parse_args()
sys.modules.setdefault('build', types.SimpleNamespace(get=lambda *x, **k: None, get_sal=lambda *x, **k: None))
sys.path[:0] = [f'{a.code}/sm120', f'{a.code}/streaming', os.path.dirname(os.path.abspath(__file__))]
torch.set_num_threads(int(os.environ.get('NT', '8')))
import nq15                                   # noqa: E402,F401  base_K-aware nq_decode.ring_levels (identity for K=2)
import moe, nqload as NQ, resident as RS, p4rec as PR   # noqa: E402

L, E, r = a.layer, a.expert, a.rank
d = NQ.layer_dir(a.root, L); man = json.load(open(f'{d}/manifest.json'))
ss = [2 * r, 2 * r + 1]
parts = {i: NQ.load_part(d, i) for i in ss}
art = NQ.group_art(parts, man, E, ss)
ex, _, (Qg, Qu, Qd) = NQ.kernel_expert(art, 'cpu', want_Q=True)
ex.had_dn = NQ.had_width(man)
H, I = ex.H, ex.I
W = {}
for lv in (2, 4):
    W[f'gu{lv}'] = moe.dense_W(ex.gu, lv, 4, torch.float16); W[f'dn{lv}'] = moe.dense_W(ex.dn, lv, 4, torch.float16)
    assert torch.equal(W[f'gu{lv}'], torch.cat([Qg[lv], Qu[lv]])), f'gu L{lv} != nq_decode'
    assert torch.equal(W[f'dn{lv}'], Qd[lv]), f'dn L{lv} != nq_decode'
print(f'1. L{L} E{E} rank{r}: kernel planes dense decode == nq_decode (gate|up, down x L2, L4) bk gu/dn '
      f'{ex.gu.bk}/{ex.dn.bk} rk {ex.gu.rk}/{ex.dn.rk} had_dn {ex.had_dn}', flush=True)

# 2. serve-format round trip
RL = types.SimpleNamespace(L=L, rank=r, tp=4, H=H, I=I, experts=[E], ex={E: ex}, had_dn=ex.had_dn)
lay = PR.layout(ex, H, I); rec = PR.pack(ex, lay)
with tempfile.TemporaryDirectory(dir='/tmp/nestquant/35-nq15') as td:
    rp = f'{td}/L{L}.pt'; RS.save(RL, rp); raw = torch.load(rp, weights_only=False)
    xs, H2, I2 = RS.load(rp, 'cpu'); x = xs[E]
fmt = raw['format']
rb = torch.frombuffer(bytearray(rec), dtype=torch.uint8)
seg = lambda k: rb[lay['seg'][k][0]:lay['seg'][k][0] + lay['seg'][k][1]].clone().view(torch.int32)
for pn, src in (('gu', x.gu), ('dn', x.dn)):
    p0 = getattr(ex, pn); d4 = seg(f'{pn}.d4').long()
    q = types.SimpleNamespace(N=p0.N, K=p0.K, rk=src.rk, bk=src.bk, z=moe.proj_sizes(p0.N, p0.K, None, src.rk, src.bk),
                              base=src.base, var=src.var, p4=seg(f'{pn}.p4'), Mb=d4 & 255, Nn=(d4 >> 8) & 255, fl=None)
    for lv in (2, 4):
        assert torch.equal(moe.dense_W(q, lv, 4, torch.float16), W[f'{pn}{lv}']), f'{pn} L{lv} round trip'
print(f'2. res ({fmt}) + record ({lay["rec_bytes"]} B) round trip == decode, bitwise', flush=True)

# 3. test vector
os.makedirs(a.out, exist_ok=True)
i = raw['experts'].index(E)
res = {k: raw[k][i].clone() for k in ('gu_base', 'gu_var', 'dn_base', 'dn_var', 'sc2', 'sc4', 'lr')}
meta = dict(spec='sm120/NQ_RES_V2.md', root=a.root, layer=L, expert=E, rank=r, tp=4, H=H, I=I, res_format=fmt,
            rk_gu=ex.gu.rk, rk_dn=ex.dn.rk, bk_gu=ex.gu.bk, bk_dn=ex.dn.bk, rg=ex.rg, rd=ex.rd, in_had_down=ex.had_dn,
            base_K=man['proj_meta']['down'].get('base_K', 2),
            res_K={p: man['proj_meta'][p]['res_rule']['K'] for p in ('gate', 'up', 'down')},
            seg={k: list(v) for k, v in lay['seg'].items()}, rec_bytes=lay['rec_bytes'],
            W='dense decoded weights, rotated domain [N, K] fp16 (moe.dense_W == kernel NQ_WDUMP): gu = gate rows 0..I-1 '
              '| up rows I..2I-1, K = H; dn N = H, K = I; level 2 and 4',
            sha256={k: hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for k, v in W.items()})
p = f'{a.out}/L{L}_E{E}_r{r}{a.tag}.pt'
torch.save(dict(meta=meta, record=rb.clone(), res=res, W=W), p)
json.dump(meta, open(p[:-3] + '.json', 'w'), indent=1)
print(f'3. wrote {p} ({os.path.getsize(p) / 2**20:.1f} MiB)', flush=True)
