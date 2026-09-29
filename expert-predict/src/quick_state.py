import sys; sys.path.insert(0, '/data/Jarrel/expert-predict/src'); from feats import *
d = load('embedding-drift-monitor'); P = prep(d)
print('N', P['N'], 'answer share', (P['st'] == 1).mean(), 'Cth+Can==C', bool((P['Cth'].astype(int) + P['Can'] == P['C']).all()))
pr = np.random.rand(2, NL, NE).astype(np.float32)
import time; t0 = time.time(); n = 0
for r, s, X in iter_feats(P, 64, pr): n += 1
print(n, 'refreshes', time.time() - t0, 's', X.shape)
