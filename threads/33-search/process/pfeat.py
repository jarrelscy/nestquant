"""T33h stochastic-process features of expert activity (all causal at the end of block k, reset per chain).
Per layer L, fitted on calib-fit TRAIN chains (VAL = chains 4, 9, 14, ... held out):
  Hawkes   lambda_e(k+1) = mu_e + sum_m a_m E_m,e(k)   (exp-kernel mixture, block counts, Poisson MLE; mu_e = static
           per-expert immigrant rate). hk_pred = E[count over next 4 blocks] propagating the expected self-excitation.
  HMM      K-state Poisson-emission HMM on block counts, shared by the layer's experts (Baum-Welch). Forward-filtered
           posterior -> hmm_pred (E next-4 count), hmm_hot (P top two states), hmm_rise (E rate in 4 blocks - E rate now).
  BOCPD    Adams-MacKay on the layer's 256-bin count histogram, Dirichlet-multinomial UPM (prior = layer's calib usage
           x kappa), constant hazard, run length truncated at R. bo_cp = P(r <= 4), bo_er = E[r] (blocks),
           bo_rate = posterior-predictive per-expert share x 128, bo_rmap = rate since the MAP changepoint.
  Kalman   local-linear-trend filter per expert on y = log(sal16 + c) (shared P / gain within layer): kf_lvl, kf_slp,
           kf_fc = lvl + 4 slp (log units)."""
import numpy as np
from scipy.special import gammaln
from scipy.optimize import minimize

NE = 256
HAWKES_HL = (1, 4, 16, 64)          # EMA half-lives in blocks (16 / 64 / 256 / 1024 tokens)


def val_chain(ci):
    return ci % 5 == 4


def ema_blocks(M, hl, segs):
    """plain block EMA E(k) = d E(k-1) + x(k) (x includes block k), reset per chain."""
    from scipy.signal import lfilter
    d = 0.5 ** (1.0 / hl)
    E = np.empty(M.shape, np.float64)
    for s, e in segs:
        E[s:e] = lfilter([1.0], [1.0, -d], M[s:e], axis=0)
    return E


# ------------------------------------------------------------------------------------------ Hawkes
def fit_hawkes(C, segs, fx):
    """C [nb, NE] counts. Poisson MLE of lambda(k+1) = mu_e + sum a_m (1-d_m) E_m(k). -> dict(mu, a)."""
    nf = ~fx
    Es = [ema_blocks(C, h, segs) * (1 - 0.5 ** (1.0 / h)) for h in HAWKES_HL]
    rows = np.concatenate([np.arange(s, e - 1) for s, e in segs])
    Z = np.stack([E[rows][:, nf] for E in Es], -1)           # [n, nnf, M]
    y = C[rows + 1][:, nf]
    n, ne_, M = Z.shape
    Zf = Z.reshape(-1, M); yf = y.ravel()
    eidx = np.tile(np.arange(ne_), n)

    def f(p):
        mu = np.exp(p[:ne_]); a = np.exp(p[ne_:])
        lam = mu[eidx] + Zf @ a
        ll = (yf * np.log(lam) - lam).sum()
        r = yf / lam - 1
        gmu = np.bincount(eidx, r, ne_) * mu
        ga = (Zf * r[:, None]).sum(0) * a
        return -ll / len(yf), -np.r_[gmu, ga] / len(yf)
    p0 = np.r_[np.log(y.mean(0) * 0.2 + 1e-4), np.log(np.full(M, 0.2))]
    r = minimize(f, p0, jac=True, method="L-BFGS-B", options=dict(maxiter=200))
    mu = np.zeros(NE); mu[nf] = np.exp(r.x[:ne_])
    return dict(mu=mu, a=np.exp(r.x[ne_:]), nll=float(r.fun))


def hawkes_feats(C, segs, P):
    """hk_mu (static), hk_lam (next-block intensity), hk_pred (expected next-4 count with expected self-excitation)."""
    ds = np.array([0.5 ** (1.0 / h) for h in HAWKES_HL]); a = P["a"]; mu = P["mu"]
    Es = [ema_blocks(C, h, segs) * (1 - 0.5 ** (1.0 / h)) for h in HAWKES_HL]
    lam = mu[None] + sum(a[m] * Es[m] for m in range(len(ds)))
    # expected propagation: state E_m(k+j) = d_m E_m(k+j-1) + (1-d_m) lam(k+j)
    St = [E.copy() for E in Es]
    tot = np.zeros(C.shape)
    lj = lam
    for j in range(4):
        tot += lj
        St = [ds[m] * St[m] + (1 - ds[m]) * lj for m in range(len(ds))]
        lj = mu[None] + sum(a[m] * St[m] for m in range(len(ds)))
    return dict(hk_mu=np.broadcast_to(mu, C.shape).astype(np.float32), hk_lam=lam.astype(np.float32),
                hk_pred=tot.astype(np.float32))


# ------------------------------------------------------------------------------------------ HMM
def _pois_logB(c, lam):
    return c[..., None] * np.log(lam) - lam - gammaln(c[..., None] + 1)


def fit_hmm(C, segs, fx, K=4, iters=25):
    nf = ~fx
    seqs = [C[s:e][:, nf] for s, e in segs]                   # list of [T, N]
    lam = np.array([0.02, 0.3, 1.5, 5.0])[:K] if K == 4 else np.geomspace(0.02, 5, K)
    A = np.full((K, K), 0.02 / (K - 1)); np.fill_diagonal(A, 0.98)
    pi = np.full(K, 1.0 / K); pi[0] = 0.7; pi[1:] = 0.3 / (K - 1)
    for it in range(iters):
        num_l = np.zeros(K); den_l = np.zeros(K); xi = np.zeros((K, K)); g0 = np.zeros(K); ll = 0.0
        for c in seqs:
            T, N = c.shape
            B = np.exp(_pois_logB(c, lam))                     # [T, N, K]
            al = np.empty((T, N, K)); sc = np.empty((T, N))
            a = pi[None] * B[0]; sc[0] = a.sum(1); al[0] = a / sc[0][:, None]
            for t in range(1, T):
                a = (al[t - 1] @ A) * B[t]; sc[t] = a.sum(1); al[t] = a / sc[t][:, None]
            ll += np.log(sc).sum()
            be = np.ones((N, K))
            for t in range(T - 1, -1, -1):
                g = al[t] * be
                num_l += (g * c[t][:, None]).sum(0); den_l += g.sum(0)
                if t == 0:
                    g0 += g.sum(0)
                else:
                    bb = B[t] * be / sc[t][:, None]
                    xi += np.einsum("nk,nl->kl", al[t - 1], bb) * A
                    be = bb @ A.T
        lam = np.maximum(num_l / den_l, 1e-4)
        A = xi / xi.sum(1, keepdims=True); pi = g0 / g0.sum()
        o = np.argsort(lam); lam, A, pi = lam[o], A[np.ix_(o, o)], pi[o]
    return dict(lam=lam, A=A, pi=pi, ll=float(ll))


def hmm_feats(C, segs, P):
    lam, A, pi = P["lam"], P["A"], P["pi"]
    K = len(lam)
    nb = C.shape[0]
    post = np.empty((nb, NE, K), np.float32)
    B = np.exp(_pois_logB(C, lam))
    for s, e in segs:
        a = pi[None] * B[s]; a /= a.sum(1, keepdims=True); post[s] = a
        for t in range(s + 1, e):
            a = (a @ A) * B[t]; a /= a.sum(1, keepdims=True); post[t] = a
    A4 = [np.linalg.matrix_power(A, j) for j in range(1, 5)]
    v = sum(Aj @ lam for Aj in A4)                            # E next-4 count per current state
    pf = post.reshape(-1, K)
    pred = (pf @ v).reshape(nb, NE)
    now = (pf @ lam).reshape(nb, NE)
    rise = (pf @ (A4[3] @ lam)).reshape(nb, NE) - now
    return dict(hmm_pred=pred.astype(np.float32), hmm_hot=post[..., K - 2:].sum(-1),
                hmm_rise=rise.astype(np.float32), hmm_p0=post[..., 0])


# ------------------------------------------------------------------------------------------ BOCPD
def bocpd_feats(C, segs, prior, kappa=64.0, hazard=1 / 128, R=128, tau=1.0):
    """C [nb, NE] counts (fixed experts included: histogram of the whole layer). prior [NE] usage share."""
    nb = C.shape[0]
    out = {k: np.zeros((nb, NE) if k in ("bo_rate", "bo_rmap") else nb, np.float32)
           for k in ("bo_cp", "bo_er", "bo_rate", "bo_rmap")}
    a0 = kappa * prior + 1e-3
    lh, l1h = np.log(hazard), np.log1p(-hazard)
    evid = 0.0
    for s, e in segs:
        # hypotheses: row r = current run holds the last r+1 blocks (row 0: a change right before this block)
        logp = None
        cnts = np.zeros((0, NE))
        for k in range(s, e):
            x = C[k]
            nz = np.flatnonzero(x)
            n = x.sum()
            al = a0[None] + np.vstack([np.zeros((1, NE)), cnts])       # row 0 = prior (fresh run)
            Asum = al.sum(1)
            lpred = (gammaln(Asum) - gammaln(Asum + n) + (gammaln(al[:, nz] + x[nz]) - gammaln(al[:, nz])).sum(1)) / tau
            if logp is None:
                lg = lpred[:1]
            else:
                cp = np.logaddexp.reduce(logp) + lh
                lg = np.r_[cp + lpred[0], logp + l1h + lpred[1:]]
            z = np.logaddexp.reduce(lg)
            evid += z
            logp = lg - z
            cnts = np.vstack([np.zeros((1, NE)), cnts]) + x[None]
            if len(logp) > R:                                # truncate: merge tail into the last kept hypothesis
                logp = np.r_[logp[:R - 1], np.logaddexp.reduce(logp[R - 1:])]
                cnts = cnts[:R]
            p = np.exp(logp)
            rl = np.arange(1, len(p) + 1)
            out["bo_cp"][k] = p[:4].sum()
            out["bo_er"][k] = (p * rl).sum()
            alp = a0[None] + cnts
            out["bo_rate"][k] = (p @ (alp / alp.sum(1, keepdims=True))) * 128.0
            m = int(np.argmax(p))
            out["bo_rmap"][k] = cnts[m] / (m + 1)
    out["evid"] = evid
    return out


# ------------------------------------------------------------------------------------------ Kalman LLT
def kalman_feats(Y, segs, q_l, q_s, r=1.0, p0=4.0):
    """local-linear-trend Kalman on Y [nb, NE] per expert; common covariance (same obs model for all experts)."""
    nb = Y.shape[0]
    lvl = np.empty(Y.shape, np.float32); slp = np.empty(Y.shape, np.float32)
    F = np.array([[1.0, 1.0], [0.0, 1.0]]); Q = np.diag([q_l, q_s]); H = np.array([1.0, 0.0])
    for s, e in segs:
        x = np.stack([Y[s], np.zeros(Y.shape[1])])                   # [2, NE]
        Pm = np.diag([r, p0 * q_s + 1e-3])
        lvl[s] = x[0]; slp[s] = x[1]
        for k in range(s + 1, e):
            x = F @ x; Pm = F @ Pm @ F.T + Q
            S = H @ Pm @ H + r
            Kg = Pm @ H / S
            x = x + Kg[:, None] * (Y[k] - x[0])[None]
            Pm = Pm - np.outer(Kg, H @ Pm)
            lvl[k] = x[0]; slp[k] = x[1]
    return dict(kf_lvl=lvl, kf_slp=slp, kf_fc=(lvl + 4 * slp).astype(np.float32))
