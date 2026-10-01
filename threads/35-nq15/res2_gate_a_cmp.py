"""gate (a) compare: OLD/NEW repack dirs (res2_gate_a_run.py with origin/main code vs nq-res-v2 code) over v1 L3 + L10: rank json, record regions, res files byte-identical. argv: OLD NEW"""
import os, json, hashlib, filecmp
import sys; O, N = sys.argv[1], sys.argv[2]; ok = True; out = {}
for r in range(4):
    jo, jn = open(f'{O}/rank{r}.json','rb').read(), open(f'{N}/rank{r}.json','rb').read()
    e = jo == jn; ok &= e
    idx = json.loads(jo); rb = idx['rec_bytes']
    so, sn = os.path.getsize(f'{O}/rank{r}.bin'), os.path.getsize(f'{N}/rank{r}.bin'); ok &= so == sn
    with open(f'{O}/rank{r}.bin','rb') as fo, open(f'{N}/rank{r}.bin','rb') as fn:
        for L in (3, 10):
            off = (L - 3) * 256 * rb; n = 256 * rb
            fo.seek(off); fn.seek(off); a = fo.read(n); b = fn.read(n)
            eq = a == b and len(a) == n; ok &= eq
            out[f'r{r} L{L} rec'] = (eq, hashlib.sha256(a).hexdigest()[:16])
        for L in (3, 10):
            eq = filecmp.cmp(f'{O}/res/rank{r}/L{L}.pt', f'{N}/res/rank{r}/L{L}.pt', shallow=False); ok &= eq
            out[f'r{r} L{L} res'] = (eq, hashlib.sha256(open(f'{N}/res/rank{r}/L{L}.pt','rb').read()).hexdigest()[:16])
    out[f'r{r} json'] = (e, sorted(idx['layers']), idx['format'], rb, so)
for k, v in out.items(): print(k, v)
import torch
d = torch.load(f'{N}/res/rank0/L3.pt', weights_only=False, mmap=True); print('L3 res format', d['format'], 'in_had_down', d.get('in_had_down'))
print('GATE_A', 'PASS' if ok else 'FAIL')
