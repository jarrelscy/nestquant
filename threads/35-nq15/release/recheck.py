"""Recheck a b175 repack dir against the build's done/L{L}.json markers: every (layer, rank) record region of
rank{r}.bin and every res/rank{r}/L{L}.pt by sha256; rank{r}.json covers L3-77 x 256 experts at rec_bytes; bin size
= 75*256*rec_bytes.  Optional REF dir: rank*.json and artifact_stamp.json must be byte-identical to REF's.
  python recheck.py DIR [REF]   (4 procs, one per rank)"""
import hashlib, json, os, sys
from multiprocessing import Pool
D = sys.argv[1]; REF = sys.argv[2] if len(sys.argv) > 2 else None
DONE = '/tmp/nestquant/35-nq15/release/done'; L0, NL, NE = 3, 75, 256
ONLY = [int(x) for x in os.environ.get('ONLY', '').split(',') if x]   # test: check only these layers


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def rank(r):
    bad = []
    idx = json.load(open(f'{D}/rank{r}.json')); rb = idx['rec_bytes']
    if os.path.getsize(f'{D}/rank{r}.bin') != NL * NE * rb:
        bad.append((r, 'bin size'))
    with open(f'{D}/rank{r}.bin', 'rb') as f:
        for L in (ONLY or range(L0, L0 + NL)):
            m = json.load(open(f'{DONE}/L{L}.json'))['ranks'][str(r)]
            if rb != json.load(open(f'{DONE}/L{L}.json'))['rec_bytes']:
                bad.append((r, L, 'rec_bytes'))
            e = idx['layers'].get(str(L))
            if e is None or sorted(map(int, e['experts'])) != list(range(NE)):
                bad.append((r, L, 'index'))
            f.seek((L - L0) * NE * rb); h = hashlib.sha256()
            left = NE * rb
            while left:
                b = f.read(min(left, 1 << 24)); h.update(b); left -= len(b)
                if not b:
                    break
            if h.hexdigest() != m['rec_sha256']:
                bad.append((r, L, 'rec sha'))
            rp = f'{D}/res/rank{r}/L{L}.pt'
            if not os.path.exists(rp) or os.path.getsize(rp) != m['res_bytes'] or sha_file(rp) != m['res_sha256']:
                bad.append((r, L, 'res'))
    if REF:
        for fn in (f'rank{r}.json',) + (('artifact_stamp.json',) if r == 0 else ()):
            if open(f'{D}/{fn}', 'rb').read() != open(f'{REF}/{fn}', 'rb').read():
                bad.append((r, fn, 'differs from REF'))
    print(f'rank{r}: {len(idx["layers"])} layers, rec_bytes {rb}, bad {bad}', flush=True)
    return bad


if __name__ == '__main__':
    with Pool(4) as p:
        bad = sum(p.map(rank, range(4)), [])
    print('RECHECK', D, 'PASS' if not bad else f'FAIL {bad}')
