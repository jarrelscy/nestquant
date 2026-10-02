"""captured traces (/rawdata/Jarrel/nq-tfpred/traces/cap-<boot>-<chunk>.npz) -> per-task decode streams
/rawdata/Jarrel/nq-tfpred/ds/cap-<task>.npz (same layout as prep_ids + w, xn, hp).

Rows -> steps: step k covers absolute rows [end-T, end) of request q.  Decode steps have T <= 4 (MTP ns=3 verify rows,
incl. rejected drafts); steps with a prefilling request (step_T < 0, short prefill tails) are dropped.  Per request, rows are deduped by
position keeping the last (an accepted token's row supersedes its rejected-draft row); pos 0 / padding rows dropped.
Think phase = from the first decode row until the </think> token (inclusive)."""
import numpy as np, glob, json, re, collections, sys, os
TH, ETH = 154841, 154842
TR = '/rawdata/Jarrel/nq-tfpred/traces'; OUT = '/rawdata/Jarrel/nq-tfpred/ds'
MAXT = 4
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
BIG = ('ids', 'w', 'xn', 'hp')


class Lazy:
    """row-concatenation of per-file memmaps with an optional row permutation; R[k][ix] gathers only the rows asked for"""
    def __init__(s, parts, o=None):
        s.parts = parts; s.off = np.cumsum([0] + [len(p) for p in parts]); s.o = o
    def perm(s, o): return Lazy(s.parts, o)
    def __len__(s): return int(s.off[-1])
    def __getitem__(s, ix):
        ix = np.asarray(ix); g = s.o[ix] if s.o is not None else ix
        fi = np.searchsorted(s.off, g, 'right') - 1
        out = np.empty((len(g),) + s.parts[0].shape[1:], s.parts[0].dtype)
        for f in np.unique(fi):
            m = fi == f; loc = g[m] - s.off[f]; so = np.argsort(loc, kind='stable')
            tmp = np.asarray(s.parts[f][loc[so]]); r = np.empty_like(tmp); r[so] = tmp; out[m] = r
        return out


def task_of(name):
    m = re.match(r'(?:chatcmpl-)?tfcap-(.+?)__', name)
    return m.group(1) if m else None


def main():
    files = sorted(glob.glob(f'{TR}/cap-*.npz'))
    boots = collections.defaultdict(list)
    for f in files:
        boots[os.path.basename(f).split('-')[1]].append(f)
    reqs = {}                                       # (boot, q) -> dict
    stats = collections.Counter()
    for boot, fl in boots.items():
        rows = collections.defaultdict(list); steps = []; pfs = []; names = {}
        for f in fl:
            z = np.load(f); zm = D.npz_mmap(f)
            for k in ('ids', 'w', 'xn', 'hp', 'tok', 'pos', 'abs'):
                rows[k].append(zm[k] if k in BIG else z[k])
            steps.append(np.stack([z['step_req'], z['step_T'], z['step_end'], z['step_wall'].astype(np.int64)], 1))
            for i in range(len(z['pf_req'])):
                pfs.append((int(z['pf_req'][i]), int(z['pf_T'][i]), float(z['pf_wall'][i]), z['pf_counts'][i]))
            names.update(json.loads(str(z['req_names'])))
            stats['lost'] = int(z['lost'])
        R = {k: (Lazy(v) if k in BIG else np.concatenate(v)) for k, v in rows.items()}   # big members stay memory-mapped
        o = np.argsort(R['abs'], kind='stable'); R = {k: (v.perm(o) if k in BIG else v[o]) for k, v in R.items()}
        a0 = R['abs']
        S = np.concatenate(steps)
        for q, T, end, wall in S:
            if T < 0 or T > 16:
                stats['prefill_steps'] += 1; continue
            lo = np.searchsorted(a0, end - T); hi = np.searchsorted(a0, end)
            if hi - lo != T:
                stats['missing_rows'] += T - (hi - lo)
            key = (boot, int(q)); r = reqs.setdefault(key, dict(rows=[], wall=wall, pf=None, pfn=0, name=names.get(str(q), '')))
            r['rows'].append(np.arange(lo, hi))
        for q, T, wall, c in pfs:
            key = (boot, q); r = reqs.setdefault(key, dict(rows=[], wall=wall, pf=None, pfn=0, name=names.get(str(q), '')))
            r['pf'] = c.astype(np.int64) if r['pf'] is None else r['pf'] + c; r['pfn'] += T; r['wall'] = min(r['wall'], wall)
        for key, r in reqs.items():
            if key[0] == boot: r['R'] = R
    bytask = collections.defaultdict(list)
    for key, r in reqs.items():
        t = task_of(r['name'])
        if t is None or not r['rows']:
            stats['unnamed_or_empty'] += 1; continue
        bytask[t].append(r)
    for t, rl in sorted(bytask.items()):
        rl.sort(key=lambda r: r['wall'])
        cols = collections.defaultdict(list); PF, PFN, RS = [], [], []; n = 0
        for qi, r in enumerate(rl):
            R = r['R']; ix = np.concatenate(r['rows']); pos = R['pos'][ix]
            keep = pos > 0; ix, pos = ix[keep], pos[keep]
            # last row per position, in position order
            o = np.lexsort((np.arange(len(ix)), pos)); ix, pos = ix[o], pos[o]
            last = np.r_[pos[1:] != pos[:-1], True]; ix = ix[last]
            stats['rows_dedup_dropped'] += int((~last).sum())
            if not len(ix): continue
            tk = R['tok'][ix]; en = np.cumsum(tk == ETH); th = (en - (tk == ETH)) == 0
            for k in ('ids', 'w', 'xn', 'hp', 'tok'):
                cols[k].append(R[k][ix])
            cols['think'].append(th); cols['rq'].append(np.full(len(ix), len(PF), np.int32)); cols['pos'].append(pos)
            PF.append(r['pf'] if r['pf'] is not None else np.zeros((75, 256), np.int64)); PFN.append(r['pfn']); RS.append(n); n += len(ix)
        if not n: continue
        out = {k: np.concatenate(v) for k, v in cols.items()}; del cols
        out['ex'] = out.pop('ids')
        np.savez(f'{OUT}/cap-{t}.npz', **out, pf=np.stack(PF).astype(np.int32), pfn=np.array(PFN), rstart=np.array(RS))
        print(t, 'rows', n, 'reqs', len(PF), 'think %.3f' % out['think'].mean(), flush=True)
    print(json.dumps(stats))


if __name__ == '__main__':
    main()
