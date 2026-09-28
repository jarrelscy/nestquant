"""Pass 1: uniform 2+2 fit recording per-128x128-unit cost curves.  Allocate (per TP8 shard, constant bytes) and
refit (pass 2).  Variants: perm = none | act (input channels sorted by diag H; for down: act-order-shard =
sort, deal round-robin to 8 shards of 256, sort within shard).  Saves dequantized weights + receipts."""
import sys, json, time
from alloc_lib import *
L, E = int(sys.argv[1]), int(sys.argv[2]); perms = sys.argv[3].split(',') if len(sys.argv) > 3 else ['none']
data, Hs, G = load(L, E, want_G=True)
D = f'{SCR}/L{L}E{E}'; rec_path = f'results/alloc_L{L}E{E}.json'
rec = json.load(open(rec_path)) if os.path.exists(rec_path) else {}
LAM = 0.3
for mi, pn in enumerate(PROJ):
    W, H = data.teacher[mi], Hs[mi]
    g = G[mi]
    if g is not None:   # 128-group spread of thread-06 downstream weights (diagnostic only; Hadamard mixes within 128)
        rec[f'{pn}/G_group_cv'] = float(g.reshape(-1, 128).square().mean(1).std() / g.reshape(-1, 128).square().mean(1).mean())
    for perm in perms:
        n = W.shape[1]
        if perm == 'act':
            order = torch.argsort(H.diagonal(), descending=True).cpu()
            if mi == 2:
                sh = [order[s::8] for s in range(8)]       # already sorted within shard
                order = torch.cat(sh)
            pi = order.cuda()
        else:
            pi = torch.arange(n, device='cuda')
        inv = torch.argsort(pi)
        Wp, Hp = W[:, pi].contiguous(), H[pi][:, pi].contiguous()
        P = problem(Wp, Hp, mi)
        m = P.Wn.shape[0]; nc, nr = n // U, m // U; sh = shard_of(nc, nr, mi == 2)
        two = torch.full((nc, nr), 2.0)
        unperm = lambda o: {k: (P.dequant(o[k])[:, inv] if k in ('Q2', 'Q4') else o[k]) for k in o}
        def save(o, tag):
            os.makedirs(f'{D}/{pn}', exist_ok=True)
            torch.save(dict(w2=P.dequant(o['Q2'])[:, inv].bfloat16().cpu(), w4=P.dequant(o['Q4'])[:, inv].bfloat16().cpu()), f'{D}/{pn}/{tag}.pt')
        t0 = time.time()
        o = fit(P, LAM, two, two, cand_b=(1.5, 2, 2.5), cand_r=(0, 1.5, 2, 2.5, 3))
        c2, c4 = o['c2'], o['c4']
        torch.save(dict(c2=c2, c4=c4), f'{D}/{pn}/curves_{perm}.pt')
        save(o, f'uni_{perm}'); rec[f'{pn}/uni_{perm}'] = dict(l2=o['l2'], l4=o['l4'], bpw=bits(m, n, two, two, mi == 2))
        print(pn, perm, 'uni', o['l2'], o['l4'], round(time.time() - t0), flush=True)
        # allocations
        Kr4 = allocate({k: c4[k] for k in (0, 1.5, 2, 2.5, 3)}, 2.0, sh, (0, 1.5, 2, 2.5, 3))
        Kb2 = allocate(c2, 2.0, sh, (1.5, 2, 2.5))
        # forced +-0.5 by benefit rank per shard (sanity: what a real spread does)
        ben = (c4[1.5] - c4[2.5])
        Kf = two.clone()
        for s in sh.unique():
            idx = (sh == s).nonzero(); b = ben[idx[:, 0], idx[:, 1]]; o_ = torch.argsort(b, descending=True)
            hi = idx[o_[:len(o_) // 2]]; lo = idx[o_[len(o_) // 2:]]
            Kf[hi[:, 0], hi[:, 1]] = 2.5; Kf[lo[:, 0], lo[:, 1]] = 1.5
        ben2 = (c2[1.5] - c2[2.5]); Kf2 = two.clone()
        for s in sh.unique():
            idx = (sh == s).nonzero(); b = ben2[idx[:, 0], idx[:, 1]]; o_ = torch.argsort(b, descending=True)
            hi = idx[o_[:len(o_) // 2]]; lo = idx[o_[len(o_) // 2:]]
            Kf2[hi[:, 0], hi[:, 1]] = 2.5; Kf2[lo[:, 0], lo[:, 1]] = 1.5
        # proxy-estimated gains (from pass-1 curves)
        est = lambda cur, K: float(sum(cur[k][K == k].sum() for k in cur if (K == k).any()))
        rec[f'{pn}/est_{perm}'] = dict(l4_uni=est(c4, two), l4_greedy=est(c4, Kr4), l4_pm05=est(c4, Kf),
                                        l2_uni=est(c2, two), l2_greedy=est(c2, Kb2), l2_pm05=est(c2, Kf2),
                                        colcv4=float(c4[2].sum(1).std() / c4[2].sum(1).mean()), rowcv4=float(c4[2].sum(0).std() / c4[2].sum(0).mean()),
                                        colcv2=float(c2[2].sum(1).std() / c2[2].sum(1).mean()),
                                        hist4={str(k): int((Kr4 == k).sum()) for k in (0, 1.5, 2, 2.5, 3)}, hist2={str(k): int((Kb2 == k).sum()) for k in (1.5, 2, 2.5)})
        print(pn, perm, rec[f'{pn}/est_{perm}'], flush=True)
        for tag, Kb, Kr in ((f'A4g_{perm}', two, Kr4), (f'A4pm_{perm}', two, Kf), (f'A2g_{perm}', Kb2, two), (f'A2pm_{perm}', Kf2, two)):
            if tag.startswith('A4g') and bool((Kr4 == 2).all()) or tag.startswith('A2g') and bool((Kb2 == 2).all()):
                rec[f'{pn}/{tag}'] = 'uniform'; continue
            o = fit(P, LAM, Kb, Kr)
            save(o, tag); rec[f'{pn}/{tag}'] = dict(l2=o['l2'], l4=o['l4'], bpw=bits(m, n, Kb, Kr, mi == 2))
            print(pn, tag, o['l2'], o['l4'], rec[f'{pn}/{tag}']['bpw'], flush=True)
        json.dump(rec, open(rec_path, 'w'), indent=1)
        del P, o; torch.cuda.empty_cache()
