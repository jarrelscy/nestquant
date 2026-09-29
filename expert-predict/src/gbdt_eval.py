"""Score every 16-token refresh with the GBDT (candidates rank 20-120 by EMA256; ranks <20 kept), then run the C sim at several caps."""
import sys, os, time, json; sys.path.insert(0, '/data/Jarrel/expert-predict/src'); from gbdt_feats import *
import lightgbm as lgb
tag, tgt = sys.argv[1], sys.argv[2]; tasks = sys.argv[3].split(','); CAPS = [float(x) for x in os.environ.get('CAPS', '6,12,24').split(',')]
HMS = [float(x) for x in os.environ.get('HMS', '0,0.1,0.25').split(',')]
Ps = np.load(f'{W}/feat/coact_{tag}.npy'); TG = tgt.split(','); BST = [lgb.Booster(model_file=f'{W}/feat/gbdt_{tag}_{g}.txt') for g in TG]
out = open(f'{W}/results/gbdt_eval.jsonl', 'a'); FXf = FIXED.astype(np.float32)
import glob
def loadx(t):
    if not t.startswith('probe:'): return load(t)
    fs = sorted(glob.glob(f'{DATA}/probe-{t[6:]}-[0-9].npz')); zs = [np.load(f) for f in fs]
    ex = np.ascontiguousarray(np.concatenate([z['ex'] for z in zs])); tok = np.concatenate([z['tok'] for z in zs])
    req = np.concatenate([np.full(len(z['tok']), i) for i, z in enumerate(zs)])
    return dict(ex=ex, tok=tok, req=req, full_idx=np.arange(len(ex)), dec_all=np.ones(len(ex), bool))
for t in tasks:
    d = loadx(t); ex = d['ex']; N = len(ex); nref = -(-N // 16)
    SS = [np.zeros((nref + 1, NL, NE), np.float32) for _ in TG]; tpred = 0; hitg = [0.0] * len(TG); hite = tot = 0.0; t0 = time.time(); B = []
    def flush(B):
        global tpred, hite, tot
        Xc = np.concatenate([x[2] for x in B]); nc = RHI - RLO
        for k, bst in enumerate(BST):
            cols = [FNAMES.index(f) for f in bst.feature_name()]; t1 = time.time(); pr = bst.predict(Xc[:, cols] if len(cols) < len(FNAMES) else Xc, num_threads=16); tpred += time.time() - t1; pr = pr.reshape(len(B), NL, nc)
            for i, (b, cand, _, top, e256, y) in enumerate(B):
                s = SS[k][b]; np.put_along_axis(s, cand, pr[i].astype(np.float32), 1); np.put_along_axis(s, top, 1e3 + e256[np.arange(NL)[:, None], top], 1)
                if y is not None:
                    sg = np.where(FIXED.astype(bool), -np.inf, s); tg = np.argpartition(-sg, NF, 1)[:, :NF]; hitg[k] += (y * FXf).sum() + np.take_along_axis(y, tg, 1).sum()
        for i, (b, cand, _, top, e256, y) in enumerate(B):
            if y is not None:
                fx = (y * FXf).sum(); tot += y.sum()
                se = np.where(FIXED.astype(bool), -np.inf, e256); te = np.argpartition(-se, NF, 1)[:, :NF]; hite += fx + np.take_along_axis(y, te, 1).sum()
    for b, cand, X, _, _, e256, top, y64 in gen(d, Ps, 1):
        B.append((b, cand, X, top, e256, y64 if b % 4 == 0 else None))
        if len(B) == 256: flush(B); B = []
    if B: flush(B)
    st, _ = seg_state(d['tok'], d['req']); feat_secs = time.time() - t0 - tpred
    if t.startswith('probe:'):
        Cb = block_counts(ex)
        for nm, R, hl, hm in (('ema512_R64', 64, 512, 0.0), ('ema256_R16_hm0.1', 16, 256, 0.1), ('ema128_R16_hm0.1', 16, 128, 0.1)):
            Se = ema_scores(Cb, R, hl)
            for cap in CAPS:
                o = sim(ex, Se, R=R, hm=hm, cap_gbps=cap); o.pop('hits')
                r = dict(task=t, tag='ema', tgt=nm, cap=cap, hm=hm, share=o['share'], gbps=o['gbps'], sal=o['sal_share'], N=N)
                print(json.dumps(r), flush=True); out.write(json.dumps(r) + '\n'); out.flush()
    for k, g in enumerate(TG):
      S = SS[k]
      base = dict(task=t, tag=tag, tgt=g, recall_gbdt=hitg[k] / tot, recall_ema256=hite / tot, feat_secs=feat_secs,
                pred_us_per_refresh_16thr=tpred / len(TG) / nref * 1e6)
      print(json.dumps(base), flush=True)
      for cap in CAPS:
        for hm in HMS:
            for lazy in (0,):
                o = sim(ex, S, R=16, hm=hm, lazy=lazy, cap_gbps=cap); h = o.pop('hits') / 600.
                r = dict(base, cap=cap, hm=hm, lazy=lazy, share=o['share'], think=float(h[st == 0].mean()), answer=float(h[st == 1].mean()),
                         gbps=o['gbps'], sal=o['sal_share'], N=N)
                print(json.dumps(r), flush=True); out.write(json.dumps(r) + '\n'); out.flush()
                if os.environ.get('LATE') and hm == float(os.environ['LATE']):   # prediction applied one refresh late
                    SL = np.empty_like(S); SL[1:] = S[:-1]; SL[0] = 0
                    o = sim(ex, SL, R=16, hm=hm, lazy=0, cap_gbps=cap); o.pop('hits'); del SL
                    r = dict(base, tgt=base['tgt'] + '_late1', cap=cap, hm=hm, lazy=0, share=o['share'], gbps=o['gbps'], sal=o['sal_share'], N=N)
                    print(json.dumps(r), flush=True); out.write(json.dumps(r) + '\n'); out.flush()
      del S
    del SS
