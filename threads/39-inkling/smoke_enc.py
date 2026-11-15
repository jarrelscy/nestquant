import os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nq_ink_env as E
NE, ND = E.NE, E.ND
torch.manual_seed(0)
L, Ex = int(sys.argv[1]), int(sys.argv[2])
W = E.OS.weights(None, L, Ex)
X = torch.randn(8192, E.D, device="cuda") * (1 + 3 * (torch.rand(E.D, device="cuda") < 0.01))
Hx = X.T @ X / X.shape[0]
g, u = (X.bfloat16() @ W[0].T.bfloat16()).float(), (X.bfloat16() @ W[1].T.bfloat16()).float()
h = 4 * torch.tanh(g / 4) * torch.sigmoid(g) * 25 * torch.tanh(u / 25)
Ha = h.T @ h / h.shape[0]
HG = dict(H=[Hx, Hx, Ha], G=[None, None, None])
t0 = time.time()
art, dense = NE.encode_expert(W, HG, res_K=dict(gate=2.25, up=2.25, down=2.5625), lr=dict(NE.LR))
print("encode s", round(time.time() - t0, 1), "rate", art["meta"]["rate"], {p: art["meta"]["info"][p]["bits"] for p in NE.PROJ})
for lv in (2, 4):
    gq, uq, dq = [w.cuda() for w in dense[lv]]
    def mo(Wg, Wu, Wd, x):
        gg, uu = x @ Wg.T, x @ Wu.T
        return (4 * torch.tanh(gg / 4) * torch.sigmoid(gg) * 25 * torch.tanh(uu / 25)) @ Wd.T
    Xe = torch.randn(2048, E.D, device="cuda")
    yr, yq = mo(*W, Xe), mo(gq, uq, dq, Xe)
    print(f"L{lv} rel out err {( (yq - yr).norm() / yr.norm()).item():.4f}  weight rel err",
          [round(((a - b).norm() / b.norm()).item(), 4) for a, b in zip((gq, uq, dq), W)])
print("shards", {p: len(art[p]["base"]["shards"]) for p in NE.PROJ})
