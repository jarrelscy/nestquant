import json, hashlib, filecmp, os
L = int(__import__('sys').argv[1]); ok = True; S = 'shipped'; G = '/tmp/nestquant/35-nq15/gate_a'
for r in range(4):
    io, inn, isp = (json.load(open(f'{d}/rank{r}.json')) for d in (f'{G}/old', f'{G}/new', S))
    rb = io['rec_bytes']; e_idx = io['layers'][str(L)] == inn['layers'][str(L)]
    e_sp = json.loads(json.dumps(inn['layers'][str(L)])) == {k: isp['layers'][str(L)][k] for k in inn['layers'][str(L)]}
    off, n = (L - 3) * 256 * rb, 256 * rb
    reg = []
    for d in (f'{G}/old', f'{G}/new'):
        with open(f'{d}/rank{r}.bin', 'rb') as f: f.seek(off); reg.append(f.read(n))
    reg.append(open(f'{S}/rank{r}.L{L}.rec', 'rb').read())
    e_rec = reg[0] == reg[1] == reg[2] and len(reg[0]) == n
    e_res = filecmp.cmp(f'{G}/old/res/rank{r}/L{L}.pt', f'{G}/new/res/rank{r}/L{L}.pt', shallow=False) and \
            filecmp.cmp(f'{G}/new/res/rank{r}/L{L}.pt', f'{S}/res/rank{r}/L{L}.pt', shallow=False)
    ok &= e_idx and e_sp and e_rec and e_res
    print(f'r{r} rb={rb} rb_shipped={isp["rec_bytes"]} idx old==new {e_idx} new==shipped {e_sp} rec old==new==shipped {e_rec} '
          f'{hashlib.sha256(reg[1]).hexdigest()[:16]} res {e_res}')
print(f'GATE_A L{L}', 'PASS' if ok else 'FAIL')
