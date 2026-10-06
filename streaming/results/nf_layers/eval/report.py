# nq-lalloc results table: per arm registry live KLD (+se over contexts), paired delta vs U, top-1, per-domain, decode hot share
# (rank-0 lay_hot/lay_tot deltas per context, all + 5 bands of 15 layers), tps, confirmation windows w0-3.
import json,numpy as np,collections,sys
O='/data/Jarrel/nq-lalloc';dom=[r['domain'] for r in json.load(open('/data/Jarrel/nq-kld/reg/root/panel/panel.json'))['records']]
S=collections.defaultdict(dict)
for l in open(f'{O}/scores.jsonl'):x=json.loads(l);S[x['tag'][3:]][x['ctx']]=x
CW=collections.defaultdict(dict)
try:
    for l in open('/data/Jarrel/nq-kld/runs/scores.jsonl'):
        x=json.loads(l)
        if x['tag'].startswith('la_'):CW[x['tag'][3:].rsplit('_w',1)[0]][x['win']]=x['kld']
except FileNotFoundError:pass
def hot(arm):
    try:R=[json.loads(l) for l in open(f'{O}/la_{arm}.jsonl')]
    except FileNotFoundError:return None,None,None
    H=np.zeros(75);T=np.zeros(75);tps=[]
    for r in R:
        a,b=r.get('io_before'),r.get('io_after');tps.append(r['tps'])
        if a and b and a.get('lay_hot') and b.get('lay_hot'):H+=np.array(b['lay_hot'])-a['lay_hot'];T+=np.array(b['lay_tot'])-a['lay_tot']
    if T.sum()==0:return None,None,np.mean(tps)
    return H.sum()/T.sum(),[H[i:i+15].sum()/T[i:i+15].sum() for i in range(0,75,15)],np.mean(tps)
arms=[a for a in ['U','R1','R1r','R2','R2r','S','Sr','D','Dr','U2'] if len(S.get(a,{}))==25]
U=np.array([S['U'][c]['kld'] for c in range(25)]) if 'U' in arms else None
UU=(U+np.array([S['U2'][c]['kld'] for c in range(25)]))/2 if 'U2' in arms else None
print('| arm | KLD ± se | Δ vs U ± se (paired) | Δ vs mean(U,U2) | top-1 | code | ency | lit | multi | sci | hot (dec) | hot by band L3-17/18-32/33-47/48-62/63-77 | tok/s | conf w0/w1/w2/w3 = mean |')
print('|'+'---|'*14)
for a in arms:
    k=np.array([S[a][c]['kld'] for c in range(25)]);t1=np.mean([S[a][c]['top1'] for c in range(25)])
    dd=' '.join(f'{np.mean([k[c] for c in range(25) if dom[c]==d]):.4f}' for d in ['code','encyclopedic','literary','multilingual','scientific']).split()
    if U is not None and a!='U':d=k-U;ds=f'{d.mean():+.4f} ± {d.std(ddof=1)/5:.4f}'
    else:ds='—'
    du=f'{(k-UU).mean():+.4f} ± {(k-UU).std(ddof=1)/5:.4f}' if UU is not None and a not in('U','U2') else '—'
    h,hb,tp=hot(a);cw=CW.get(a,{})
    cws='/'.join(f'{cw[w]:.4f}' for w in range(4))+f' = {np.mean([cw[w] for w in range(4)]):.4f}' if len(cw)==4 else '—'
    print(f'| {a} | {k.mean():.4f} ± {k.std(ddof=1)/5:.4f} | {ds} | {du} | {t1:.4f} | '+' | '.join(dd)+f' | {h if h is None else round(h,4)} | {"/".join(f"{x:.3f}" for x in hb) if hb else "—"} | {tp if tp is None else round(tp,1)} | {cws} |')
