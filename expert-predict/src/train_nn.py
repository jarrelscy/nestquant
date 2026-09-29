"""Low-rank cross-expert future-rate model (torch CPU). usage: train_nn.py TRAIN(comma) EVAL(comma) F k tag
pred = softplus(diag(x) + W2 relu(W1 sqrt(x))) ; diag = per-layer weights on 4 EMA rates + per-(l,e,state) bias.
Poisson loss on future counts. Writes S for eval tasks to feat/S_<tag>_<task>.npy (float16)."""
import os; os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys, time; sys.path.insert(0, '/data/Jarrel/expert-predict/src'); from common import *
import torch; torch.set_num_threads(12); torch.manual_seed(0)
train, evl = sys.argv[1].split(','), sys.argv[2].split(','); F = int(sys.argv[3]); KH = int(sys.argv[4]); tag = sys.argv[5]
EPOCHS = int(os.environ.get('EPOCHS', 3))
M_ = NL * NE
def ld(t):
    z = np.load(f'{W}/feat/{t}_R64.npz'); m = z[f'n{F}'] == F // 16
    return torch.from_numpy(z['X'][m]), torch.from_numpy(z['st'][m].astype(np.int64)), torch.from_numpy(z[f'Y{F}'][m])
Xs, Ss, Ys = zip(*[ld(t) for t in train]); X = torch.cat(Xs); S_ = torch.cat(Ss); Y = torch.cat(Ys); del Xs, Ys
print('train samples', len(X), flush=True)
class Net(torch.nn.Module):
    def __init__(s):
        super().__init__()
        s.a = torch.nn.Parameter(torch.zeros(2, 4, NL, 1)); s.b = torch.nn.Parameter(torch.full((2, M_), -4.0))
        s.W1 = torch.nn.Linear(4 * M_, KH); s.W2 = torch.nn.Linear(KH, M_)
        torch.nn.init.zeros_(s.W2.weight); torch.nn.init.zeros_(s.W2.bias)
    def forward(s, x, st):
        x = x.float(); sq = x.sqrt()                                     # [B,4,M]
        a = s.a[st]                                                      # [B,4,NL,1]
        diag = (a * sq.view(-1, 4, NL, NE)).sum(1).view(-1, M_) + s.b[st]
        z = torch.relu(s.W1(sq.view(-1, 4 * M_)))
        return torch.nn.functional.softplus(diag + s.W2(z)) + 1e-6
net = Net()
if KH == 0:
    net.W1 = torch.nn.Linear(4 * M_, 1); torch.nn.init.zeros_(net.W1.weight)
# init diag from a log-rate-ish mapping: softplus(sum a sqrt(x) + b) ~ x
with torch.no_grad(): net.a[:, 1] = 2.0
opt = torch.optim.Adam(net.parameters(), lr=2e-3)
B = 64; n = len(X); steps = EPOCHS * (n // B); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
t0 = time.time(); k = 0
for ep in range(EPOCHS):
    perm = torch.randperm(n); tot = 0
    for i in range(0, n - B + 1, B):
        idx = perm[i:i + B]; x, st, y = X[idx], S_[idx], Y[idx].float()
        p = net(x, st); loss = (p - y * torch.log(p)).mean() * F     # Poisson NLL on counts (rate * F)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item(); k += 1
        if k % 200 == 0: print(f'ep {ep} step {k}/{steps} loss {tot / (i // B + 1):.5f} {time.time() - t0:.0f}s', flush=True)
del X, Y
torch.save(net.state_dict(), f'{W}/feat/nn_{tag}.pt')
net.eval()
for t in evl:
    z = np.load(f'{W}/feat/{t}_R64.npz'); Xe = torch.from_numpy(z['X']); Se = torch.from_numpy(z['st'].astype(np.int64))
    out = np.zeros((len(Xe), M_), np.float16)
    with torch.no_grad():
        for i in range(0, len(Xe), 256): out[i:i + 256] = net(Xe[i:i + 256], Se[i:i + 256]).numpy()
    # held-out Poisson loss vs EMA baselines
    m = z[f'n{F}'] == F // 16; y = z[f'Y{F}'][m].astype(np.float32)
    def pl(p): p = np.maximum(p.astype(np.float32), 1e-6); return float((p - y * np.log(p)).mean() * F)
    print(t, 'heldout poisson nn', pl(out[m]), 'ema128', pl(z['X'][m][:, 1]), 'ema512', pl(z['X'][m][:, 2]), flush=True)
    np.save(f'{W}/feat/S_{tag}_{t}.npy', out)
