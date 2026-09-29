import sys; sys.path.insert(0, '/data/Jarrel/expert-predict/src'); from feats import *
for task in sys.argv[1].split(','):
    d = load(task); ex = d['ex']; C = block_counts(ex); S = ema_scores(C, 64, 512); o = sim(ex, S); h = o['hits'] / 600.
    st, newreq = seg_state(d['tok'], d['req'])
    rs = np.nonzero(newreq)[0]; since = np.arange(len(h)) - rs[np.searchsorted(rs, np.arange(len(h)), 'right') - 1]
    ev = np.nonzero(d['tok'] == SPECIAL['ethink'])[0]; j = np.searchsorted(ev, np.arange(len(h)), 'right') - 1
    sa = np.where((j >= 0) & (st == 1), np.arange(len(h)) - ev[np.maximum(j, 0)], -1)
    print(task, 'all %.4f' % h.mean(), 'think %.4f answer %.4f (answer tok share %.3f)' % (h[st == 0].mean(), h[st == 1].mean(), (st == 1).mean()))
    for a, b in [(0, 16), (16, 64), (64, 256), (256, 1024), (1024, 4096), (4096, 1 << 30)]:
        m = (since >= a) & (since < b); print('  since req start [%d,%d): share %.4f  tok %.3f' % (a, b, h[m].mean(), m.mean()))
    for a, b in [(0, 16), (16, 64), (64, 256), (256, 100000)]:
        m = (sa >= a) & (sa < b); print('  since </think> [%d,%d): share %.4f  tok %.3f' % (a, b, h[m].mean() if m.any() else 0, m.mean()))
