"""GPU: score_stream.py STREAM TMPDIR NAME1,NAME2 -> scores/NAME_STREAM/L.npy for each model; consumes TMPDIR/L.npz
as featjob.py produces them (deletes after scoring)."""
import os, sys, time
import numpy as np, torch
import train as TR
stream, td, names = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
dev = os.environ.get("DEV", "cuda")
nets = {}
for n in names:
    ck = torch.load(f"{TR.OUT}/models/{n}.pt", map_location="cpu")
    a = ck["args"]
    net = TR.Net(ck["K"], a["arch"], a["d"], a["nl"]).to(dev); net.load_state_dict(ck["state"]); net.eval()
    nets[n] = net
    os.makedirs(f"{TR.OUT}/scores/{n}_{stream}", exist_ok=True)
fx_np, _ = TR.sets(); fx = torch.from_numpy(fx_np).to(dev)
order = list(enumerate(TR.LAYERS))
if os.environ.get("REV"):
    order = order[::-1]
for i, L in order:
    if all(os.path.exists(f"{TR.OUT}/scores/{n}_{stream}/L{L}.npy") for n in nets):
        continue
    f = f"{td}/L{L}.npz"
    while not os.path.exists(f):
        time.sleep(5)
    z = np.load(f)
    X = torch.from_numpy(z["X"]); P = torch.from_numpy(np.log(np.maximum(z["P"], 1e-30)).astype(np.float32))
    for n, net in nets.items():
        out = []
        with torch.no_grad():
            for j in range(0, X.shape[0], 2048):
                x = X[j:j + 2048].to(dev); lp = P[j:j + 2048].to(dev)
                li = torch.full((x.shape[0],), i, dtype=torch.long, device=dev)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                    r = net(x, lp, li, fx[li]).float()
                out.append(torch.exp((lp + r).clamp(max=30)).cpu())
        np.save(f"{TR.OUT}/scores/{n}_{stream}/L{L}.npy", torch.cat(out).numpy())
    if not os.environ.get("KEEP"):
        os.remove(f); open(f + ".done", "w").close()
    print(L, end=" ", flush=True)
print("done")
