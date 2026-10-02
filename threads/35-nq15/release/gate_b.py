"""T35 gate (b): for sampled (L, E, rank) of the b175 release, res2_testvec.py (bitwise: kernel-plane dense decode ==
encoder nq_decode at L2/L4, and res+record serve-format round trip == decode) must PASS, and its record bytes / res fields
must equal what the release build wrote (repo/rank{r}.bin at ((L-3)*256+E)*rec_bytes, repo/res/rank{r}/L{L}.pt).
argv: L:E:r ...   writes gate_b/L{L}_E{E}_r{r}.json"""
import sys, os, json, subprocess, torch
G = '/tmp/nestquant/35-nq15/gate_b'; REPO = '/tmp/nestquant/35-nq15/release/repo'
PY = '/home/coder/git/glm52/.venv/bin/python'; TV = '/home/coder/git/nestquant/threads/35-nq15/res2_testvec.py'
allok = True
for s in sys.argv[1:]:
    L, E, r = map(int, s.split(':'))
    lg = f'{G}/tv/L{L}_E{E}_r{r}.log'
    rc = subprocess.run(['nice', '-n', '10', PY, TV, '--root', '/tmp/nestquant/35-nq15/enc_b175', '--layer', str(L), '--expert', str(E),
                         '--rank', str(r), '--out', f'{G}/tv', '--code', '/tmp/nestquant/35-nq15/wt_main'],
                        stdout=open(lg, 'w'), stderr=subprocess.STDOUT, env=dict(os.environ, NT='4')).returncode
    res = dict(L=L, E=E, rank=r, testvec_rc=rc)
    if rc == 0:
        tv = torch.load(f'{G}/tv/L{L}_E{E}_r{r}.pt', weights_only=False)
        rb = json.load(open(f'{REPO}/rank{r}.json'))['rec_bytes']
        with open(f'{REPO}/rank{r}.bin', 'rb') as f:
            f.seek(((L - 3) * 256 + E) * rb); b = f.read(rb)
        rec = bytes(tv['record'].numpy().tobytes())
        res['record_eq'] = b[:len(rec)] == rec and (len(rec) == rb or not any(b[len(rec):]))
        raw = torch.load(f'{REPO}/res/rank{r}/L{L}.pt', weights_only=False, map_location='cpu')
        i = list(raw['experts']).index(E)
        res['res_eq'] = {k: bool(torch.equal(raw[k][i], v)) for k, v in tv['res'].items()}
        res['res_format'] = raw.get('format'); res['in_had_down'] = raw.get('in_had_down')
        res['ok'] = res['record_eq'] and all(res['res_eq'].values())
    else:
        res['ok'] = False
    allok &= res['ok']
    json.dump(res, open(f'{G}/L{L}_E{E}_r{r}.json', 'w'), indent=1)
    print(json.dumps(res), flush=True)
print('GATE_B', 'PASS' if allok else 'FAIL', flush=True)
