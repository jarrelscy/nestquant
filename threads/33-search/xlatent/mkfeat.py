import sys, time, numpy as np, torch
import xlib as X, xnet as N
torch.set_num_threads(8)
c = sys.argv[1]
t0 = time.time()
f = N.own_features(X.load(c, "bsal"), X.load(c, "bcnt"), X.load(c, "S_v2"), X.chains(c), torch.device("cpu"))
np.save(f"{X.D}/{c}/feat7.npy", f.numpy()); print(c, f.shape, time.time() - t0)
