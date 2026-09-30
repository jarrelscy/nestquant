"""infer_sm.py RUN [CORPUS=sm120tf]: load runs/RUN/model.pt, causal inference on CORPUS -> runs/RUN/S_CORPUS.npy (f16)."""
import os, sys
import numpy as np, torch
run = sys.argv[1]; corpus = sys.argv[2] if len(sys.argv) > 2 else "sm120tf"
ck = torch.load(f"/tmp/nestquant/33-search/seq/runs/{run}/model.pt", map_location="cpu")
args = ck["args"]
argv = [run]
for k, v in args.items():
    if k != "name":
        argv += [f"--{k}", str(v)]
sys.argv = ["train.py"] + argv
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")).read().split('if __name__ == "__main__":')[0]
exec(src)
model = TCN(3 + a.v2, a.H, a.dil).to(dev); model.load_state_dict(ck["sd"])
c_, s_, v_ = load(corpus)
S = infer(model, corpus, c_, s_, v_)
np.save(f"{OUTD}/S_{corpus}.npy", S.astype(np.float16))
print("saved", run, corpus, S.shape, flush=True)
