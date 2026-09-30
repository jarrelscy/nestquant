import time, os
os.environ["LAYOUT"] = "k0"
import numpy as np, plib as P, lightgbm as lgb
n = 20000
X = np.random.rand(n * 256, 30).astype(np.float32)
b = lgb.Booster(model_file="/tmp/nestquant/33-search/process/models/k0_v2proc.txt")
t = time.time(); b.predict(X[:, :b.num_feature()], num_threads=1); print("predict 20k blocks", round(time.time() - t, 1))
