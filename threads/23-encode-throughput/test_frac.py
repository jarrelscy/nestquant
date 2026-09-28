import sys, time, os
sys.path.insert(0, "/home/coder/git/nestquant/threads/23-encode-throughput")
import common as C
import torch, frac23
C.setup()
import nq_patvit as PV
K = float(os.environ.get("K", "2.3125"))
torch.manual_seed(0)
n = int(os.environ.get("N", "1024"))
cases = {
 "normal": torch.randn(n, 256, device="cuda"),
 "normal*0.5": torch.randn(n, 256, device="cuda") * 0.5,
 "normal*3": torch.randn(n, 256, device="cuda") * 3,
 "zeros": torch.zeros(64, 256, device="cuda"),
 "const": torch.full((64, 256), 0.37, device="cuda"),
 "quantized-ties": (torch.randn(n, 256, device="cuda") * 4).round() / 4,
 "huge": torch.randn(64, 256, device="cuda") * 1e4,
 "inf": torch.where(torch.rand(64, 256, device="cuda") < 0.05, float("inf"), torch.randn(64, 256, device="cuda")),
 "-inf": torch.where(torch.rand(64, 256, device="cuda") < 0.05, float("-inf"), torch.randn(64, 256, device="cuda")),
 "nan": torch.where(torch.rand(64, 256, device="cuda") < 0.02, float("nan"), torch.randn(64, 256, device="cuda")),
 "allnan": torch.full((8, 256), float("nan"), device="cuda"),
 "sparse": torch.randn(n, 256, device="cuda") * (torch.rand(n, 256, device="cuda") < 0.1),
}
if os.environ.get("RINGS"):
    for f in os.environ["RINGS"].split(","):
        cases[os.path.basename(f)] = torch.load(f).float().cuda()
ok = True
VS = [int(v) for v in os.environ.get("VARS", "0,1,2,3").split(",")]
for k, r in cases.items():
    qa, ia = PV.patq_cuda(r, K)
    for v in VS:
        qb, ib = frac23.patq_cuda(r, K, v)
        bad = (ia != ib).any(1).sum().item()
        bq = (qa.view(torch.int32) != qb.view(torch.int32)).any(1).sum().item()
        ok &= bad == 0 and bq == 0
        print(f"{k:16s} var {v} rings {r.shape[0]:5d} mismatching rings idx {bad} val {bq}", flush=True)
r = cases["normal"]
for f, nm in [((lambda x: PV.patq_cuda(x, K)), "exl3 frac")] + [((lambda x, v=v: frac23.patq_cuda(x, K, v)), f"frac23 v{v}") for v in VS]:
    f(r); torch.cuda.synchronize(); t = time.time()
    for _ in range(3): f(r)
    torch.cuda.synchronize(); print(f"{nm}: {(time.time()-t)/3/r.shape[0]*1e6:.2f} us/ring", flush=True)
print("OK" if ok else "FAIL")
