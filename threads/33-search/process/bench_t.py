import time, torch
torch.set_num_threads(8)
Lr, NE, R, K = 75, 256, 128, 4
g = torch.Generator().manual_seed(0)
cnts = torch.poisson(torch.full((Lr, R, NE), 0.5)); scs = cnts * 1.1
logp = torch.log(torch.full((Lr, R), 1.0 / R)); a0 = 0.5
x = (torch.rand(Lr, NE) < 0.25).float() * torch.randint(1, 5, (Lr, NE)).float(); sal = x * 1.2
post = torch.full((Lr, NE, K), 0.25); A = torch.eye(K) * 0.9 + 0.025; lam = torch.tensor([.03, .3, 1.4, 5.])
rl = torch.arange(1, R + 1).float()
def step():
    al = a0 + torch.cat([torch.zeros(Lr, 1, NE), cnts[:, :-1]], 1)
    As = al.sum(2); n = x.sum(1, keepdim=True)
    lp = torch.lgamma(As) - torch.lgamma(As + n) + (torch.lgamma(al + x[:, None]) - torch.lgamma(al)).sum(2)
    lg = torch.cat([torch.logsumexp(logp, 1, keepdim=True) + torch.log(torch.tensor(1 / 32)) + lp[:, :1], logp[:, :-1] + torch.log(torch.tensor(31 / 32)) + lp[:, 1:]], 1)
    lg = lg - torch.logsumexp(lg, 1, keepdim=True)
    c2 = torch.cat([torch.zeros(Lr, 1, NE), cnts[:, :-1]], 1) + x[:, None]
    s2 = torch.cat([torch.zeros(Lr, 1, NE), scs[:, :-1]], 1) + sal[:, None]
    p = lg.exp(); alp = a0 + c2
    rate = torch.einsum("lr,lre->le", p, alp / alp.sum(2, keepdim=True)); srate = torch.einsum("lr,lre->le", p / rl, s2)
    B = torch.exp(x[..., None] * torch.log(lam) - lam - torch.lgamma(x[..., None] + 1))
    q = (post @ A) * B; q = q / q.sum(2, keepdim=True)
    return rate, srate, q
for R_ in (128, 32):
    R = R_; cnts = cnts[:, :R]; scs = scs[:, :R]; logp = logp[:, :R]; rl = rl[:R]
    step(); t = time.time(); [step() for _ in range(30)]; print(f"torch CPU 8 thr R={R}: {(time.time() - t) / 30 * 1e3:.1f} ms/refresh")
