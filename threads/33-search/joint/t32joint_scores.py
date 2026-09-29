"""T32 jbase_s2 heldout / calib-fit scores (exp log mu) -> $OUT/scores/jbase_s2_STREAM/L.npy"""
import os, sys
sys.path.insert(0, "/home/coder/git/nestquant/threads/32-gbdt-sal")
import numpy as np, torch
import joint as JT
import jlib as J
s = sys.argv[1]; od = f"{J.OUT}/scores/jbase_s2_{s}"; os.makedirs(od, exist_ok=True)
net = JT.build_model("base", 2); net.load_state_dict(torch.load(f"{JT.MD}/jbase_s2.pt", map_location="cpu"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"; torch.set_num_threads(20); net = net.to(DEV).eval()
with torch.no_grad():
    for i, L in enumerate(J.LAYERS):
        x = torch.from_numpy(np.load(f"{JT.JD}/{s}/L{L}.npz")["X"]).to(DEV)
        lt = torch.full((x.shape[0],), i, dtype=torch.long, device=DEV)
        lm = torch.cat([net(x[j:j + 1024], lt[j:j + 1024]) for j in range(0, x.shape[0], 1024)])
        np.save(f"{od}/L{L}.npy", torch.exp(lm.double()).float().cpu().numpy())
print("done")
