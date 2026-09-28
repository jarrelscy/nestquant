"""Anisotropy of the 16x16 LDL D blocks (rotated, damped thread-08 H): within-block AM/GM of diag and eigenvalues."""
from core17 import *
X = Exp(16, 36)
for p in range(3):
    st = {}
    def hook(weight, Hr, Lf): st["Hr"] = Hr.clone()
    o = quantize(X.W[p], X.H(p), sigma=SIG[p], prep_hook=hook)
    Hr = st["Hr"].double(); k = Hr.shape[0]
    # Hr from block_ldl is the damped rotated H; D blocks: D = L^-1 H L^-T with block-unit-lower L
    Lf, _ = Qm.block_ldl(Hr.float().clone(), 16, {"sigma_reg": SIG[p]}, False)
    Lf = Lf.double().cuda()
    Linv = torch.linalg.solve_triangular(Lf, torch.eye(k, device="cuda", dtype=torch.float64), upper=False)
    # H = L D L^T with L block-lower (block_ldl convention?) test both
    D = Linv @ Hr @ Linv.T
    off = D.clone()
    for b in range(0, k, 16): off[b:b+16, b:b+16] = 0
    rd, re, tr = [], [], []
    for b in range(0, k, 16):
        Db = D[b:b+16, b:b+16]; d = Db.diagonal(); e = torch.linalg.eigvalsh(Db).clamp_min(1e-30)
        rd.append(float(d.mean() / d.log().mean().exp())); re.append(float(e.mean() / e.log().mean().exp())); tr.append(float(d.sum()))
    rd, re, tr = torch.tensor(rd), torch.tensor(re), torch.tensor(tr)
    w = tr / tr.sum()
    print(["gate","up","down"][p], "offblock rel", float(off.norm() / D.norm()), "diag AM/GM (tr-weighted)", float((w*rd).sum()),
          "eig AM/GM", float((w*re).sum()), "across-block AM/GM of trD", float(tr.mean() / tr.log().mean().exp()), flush=True)
