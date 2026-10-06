"""T37 end-to-end KLD of NestQuant 1.5/4 GLM-5.3-Flash vs the fp8 teacher on the held-out capture val windows.

    python kld37.py prep                                   # CPU: segment plan (PRIVATE: tokens) from cap-txt
    torchrun --nproc-per-node 8 kld37.py run [--arms fp8,all2,mac63,mac59,stat63,all4]

Forward = capture37g (fp8 block-dequant -> bf16 linears, TF32, segment-local attention, positions restart), imported
from capture37o (same code + exact KDA head-split / eager q-block memory patches).  Layer-sequential over the whole
model; every arm keeps its own mHC hidden state (on GPU), so the routing of every arm is ON-POLICY (its own hidden
states).  Experts per MoE layer:
  fp8     original fp8 experts (inline teacher; validation (b): must reproduce the stored teacher top-64)
  all2    every routed expert at NestQuant level 2 (1.5 bpw base)                      [floor]
  all4    every routed expert at level 4 (base + residual)                             [ceiling]
  macN    Mac 96 GiB serve emulation: 19 fixed (fixed_set.json) + N floating at level 4, rest level 2;
          floating set from the jF predictor run STREAMING on the arm's own routing, per request (= segment):
            prompt = rows [0, P), P = first <think> + 1 (no <think>: P = 0, whole segment = decode, cold start)
              P < 1024  plain path: floating set frozen = floating_default_N (new request)            (SPEC 2.6)
              1024..4K  budget mode: + top-x (x=125) non-resident experts by the window's own router counts at L4
            handoff  : P >= 16 -> step_chunk over the last min(256, P//16) prompt blocks, target_seed() (hm 0);
                       layers with no score -> floating_default_N
            decode   : 16-token blocks from P; after each full block: block counts/sal -> close/score ->
                       target(resident) at hm 0.7; swaps land instantly (T36: landing ~ +0.0003 KLD)
  statN   19 fixed + floating_default_N frozen (no predictor) -- isolates the predictor's value
  orcN / lagN  floating set = top-N by salience of the same / previous 16-row block (ceiling / naive history)
  emaN    T37b predictor fix: floating set = top-N non-fixed by an EMA of per-token salience (half-life hl tokens),
          refreshed every G rows with hysteresis hm (NQ37_PG="G,hm,hl,SC", default 16,0.5,64,8)
  pgNhKtT emaN residency on N-SC slots + SC scratch slots filled per token just-in-time (T/100 loads per layer per
          token on average, credit cap 4, LRU eviction) with the most salient experts PREDICTED by a pre-gate:
          layer L's ffn_hc + norm + router applied to the hidden state K half-layers (attention / MoE blocks) before
          layer L's MoE input (K=1: entering layer L, skips its attention; K=2: after layer L-1's attention, skips
          L-1's MoE + L's attention; ...); loads must land within those K blocks
  poNtT   as pgN with the TRUE current-token routing choosing the loads (oracle pre-gate, ceiling of pgN)
floating_default_N = top-N non-fixed experts by fixed_set.json n_routed (= the floating_default rule, verified for 48).
Metrics (aggregates only): KL64 = top-64 + tail KL(teacher || arm) vs the STORED teacher (cap-txt/final/top64, a lower
bound of the full KL), full-vocab KL vs the inline fp8 arm, top-1 agreement, ppl; split all / prompt / decode /
decode-seeded / decode-cold; fraction of routed slots served at level 4.
PRIVATE per-segment detail -> /tmp/nestquant/37-flash/private/kld37/.  Aggregates -> /tmp/nestquant/37-flash/kld/."""
import argparse
import copy
import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
T37 = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, T37); sys.path.insert(0, f"{T37}/jf")
import capture37o as C  # noqa: E402  (patches HF glm5_next; teacher helpers)

M = C.M
torch.set_grad_enabled(False)
KVQ = os.environ.get("NQ37_KVQ", "")            # "" | absorbed | kvq   (T34 fp8_ds_mla serve emulation, Flash port)


def _attn_kvq(self, hidden_states, attention_mask, past_key_values=None, prev_topk_indices=None, **kw):
    """Glm5NextTextAttention.forward as absorbed MLA over an emulated fp8_ds_mla cache (threads/34-tr3/kvq.py): latent
    e4m3 + 4 pow2 tile scales, k_rot 16-bit verbatim, 512-d Q latent e4m3 per (token, head, 128-tile), fp32 attention,
    o_lat -> bf16 -> W_UV -> o_proj.  Flash's MLA forward applies no RoPE; the indexer top-k mask is kept as is.
    KVQ=absorbed: same path without quantisation (numerics control)."""
    import kvq as KQ
    B, T = hidden_states.shape[:-1]
    H, dn, dr, dv, R = self.num_heads, self.qk_nope_head_dim, self.qk_rope_head_dim, self.v_head_dim, self.kv_lora_rank
    q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
    q = self.q_b_proj(q_resid).view(B, T, H, dn + dr).transpose(1, 2)
    q_pass, q_rot = torch.split(q, [dn, dr], dim=-1)
    kv_pass, k_rot = torch.split(self.kv_a_proj_with_mqa(hidden_states), [R, dr], dim=-1)
    kv_c = self.kv_a_layernorm(kv_pass)                                                   # [B,T,R] bf16
    if self.indexer is not None:
        topk = self.indexer(hidden_states=hidden_states, q_resid=q_resid, attention_mask=attention_mask,
                            past_key_values=None)
    else:
        assert prev_topk_indices is not None
        topk = prev_topk_indices
    m = self.build_attention_mask_from_topk(topk_indices=topk, query_states=q, kv_length=T)
    m = m if m.dtype == torch.bool else (m == 0)                                          # [B,1,T,T] visible
    W = self.kv_b_proj.weight.view(H, dn + dv, R)
    q_lat = torch.matmul(q_pass, W[:, :dn])                                               # [B,H,T,R] bf16
    quant = KVQ == "kvq"
    ql = KQ.quant_q_latent(q_lat) if quant else q_lat.float()
    kv_d = (KQ.ds_mla_roundtrip(kv_c) if quant else kv_c).float()
    k = torch.cat((kv_d, k_rot.float()), -1)[:, None].expand(B, H, T, R + dr)
    o_lat = F.scaled_dot_product_attention(torch.cat((ql, q_rot.float()), -1), k, kv_d[:, None].expand(B, H, T, R),
                                           attn_mask=m, scale=self.scaling)
    o = torch.matmul(o_lat.to(hidden_states.dtype), W[:, dn:].transpose(1, 2))             # [B,H,T,dv]
    return self.o_proj(o.transpose(1, 2).reshape(B, T, H * dv).contiguous()), None, (topk if self.next_skip_topk else None)


if KVQ:
    assert KVQ in ("absorbed", "kvq"), KVQ
    sys.path.insert(0, os.path.join(os.path.dirname(T37), "34-tr3"))
    M.Glm5NextTextAttention.forward = _attn_kvq
NE, TOPK, D, FF = C.NE, C.TOPK, C.D, C.FF
G, THINK, ETHINK = 16, 154841, 154842
CAP = "/tmp/nestquant/37-flash/cap-txt"
ENC = os.environ.get("NQ37_ENC", "/tmp/nestquant/37-flash/enc_b15")
PRIV = "/tmp/nestquant/37-flash/private/kld37"
OUTD = "/tmp/nestquant/37-flash/kld"
PLAN = f"{PRIV}/plan.pt"
BMT = "/tmp/nestquant/37-flash/bmteach"     # brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits (confirmation 0000-0003)
FIXED = f"{CAP}/fixed_set.json"
NET63 = "/tmp/nestquant/37-flash/jf/models/jF63.pt"
NET48 = "/tmp/nestquant/37-flash/release/serving/predictor/joint/jF.pt"
V2 = "/tmp/nestquant/37-flash/release/serving/predictor/joint/v2_sal_tweedie1.5.txt"
CATS = ["all", "prompt", "decode", "dec_seeded", "dec_cold", "think", "answer", "asst", "ctx", "raw"]
ROLE_ASST, ROLE_CTX = 154828, (154826, 154827, 154829)   # <|assistant|>; <|system|>, <|user|>, <|observation|>


def log(*a):
    print(f"[kld37 {time.strftime('%H:%M:%S')}]", *a, flush=True)


# ----------------------------------------------------------------------------------------------------------- plan
def prep_bm(a):
    """plan from the brandonmusic Flash BF16 teacher panel: one segment per window, P = 0 (whole window = decode, cold
    start, as the T34 jF arms); final() scores full-vocab KL(BF16 teacher || arm) from the stored fp32 logits."""
    plan = []
    for w in a.bm_windows.split(","):
        tok = np.load(f"{BMT}/calibration/panel-v1/arrays/{w}.tokens.npy").astype(np.int32)
        plan.append(dict(capR=-1, si=len(plan), win=w, n=len(tok), tok=tok, P=0, ix=None, ce_teacher=0.0, acc_teacher=0.0,
                         teach=f"{BMT}/logits/window-{w.rsplit('-', 1)[1]}.safetensors" if w.startswith("final-")
                         else f"{BMT}/logits/full-panel/{w.rsplit('-', 1)[0]}/{w}.safetensors"))
    os.makedirs(PRIV, exist_ok=True)
    torch.save(plan, a.plan)
    log(f"bm plan: {len(plan)} windows -> {a.plan}")


def prep(a):
    wins = C.windows("txt", 0)
    plan = []
    for R in range(8):
        data = C.Data(wins, R, 8)
        top = torch.load(f"{CAP}/final/top64.r{R}.pt", weights_only=False)
        ce = {r["seg"]: r for r in json.load(open(f"{CAP}/final/ce.r{R}.json"))}
        for si in sorted(top):
            s = data.segs[si]
            assert s["split"] == "val" and torch.equal(top[si]["rows"], torch.from_numpy(s["ix"])), (R, si)
            tok = data.tok[s["ix"]].numpy().astype(np.int32)
            th = np.flatnonzero(tok == THINK)
            P = int(th[0]) + 1 if len(th) else 0
            plan.append(dict(capR=R, si=int(si), n=int(s["n"]), tok=tok, P=P, ix=s["ix"].astype(np.int64),
                             ce_teacher=float(ce[si]["ce"]), acc_teacher=float(ce[si]["acc"])))
    os.makedirs(PRIV, exist_ok=True)
    torch.save(plan, PLAN)
    n = np.array([p["n"] for p in plan]); P = np.array([p["P"] for p in plan])
    log(f"plan: {len(plan)} segments, {n.sum()} rows; P==0 {int((P == 0).sum())}, 0<P<16 {int(((P > 0) & (P < 16)).sum())}, "
        f"16<=P<1024 {int(((P >= 16) & (P < 1024)).sum())}, P>=1024 {int((P >= 1024).sum())}; decode rows "
        f"{int((n - P).sum())} (seeded {int((n - P)[P >= 16].sum())})")


def assign(plan, R, W, nseg=0, order="plan"):
    idx = list(range(len(plan)))
    if order == "short":
        idx.sort(key=lambda i: (plan[i]["n"], i))
    if nseg:
        idx = idx[:nseg]
    bins = [[] for _ in range(W)]; tot = [0] * W
    for i in sorted(idx, key=lambda i: (-plan[i]["n"], i)):        # LPT by rows
        j = int(np.argmin(tot)); bins[j].append(i); tot[j] += plan[i]["n"]
    return sorted(bins[R])


# ------------------------------------------------------------------------------------------------ quantized experts
class QExperts:
    """Decoded NestQuant experts of one layer, dense bf16 (gu [2F, D] = cat(gate, up), dn [D, F]) per level.
    world > 1: each rank decodes its NE/W block on its GPU and all-gathers into [NE, ...] GPU buffers.
    world == 1 (smokes): decode only `need` (or all) experts, kept in host RAM, moved per use (GPU-7 < 15 GB)."""

    def __init__(self, L, levels, dev, R, W, need=None):
        import nqdec37 as Q
        self.dev, self.host = dev, W == 1
        dec = lambda e: Q.decode_expert(torch.load(f"{ENC}/L{L}/experts/E{e}.pt", weights_only=False,
                                                   map_location="cpu"), tuple(levels), dev)   # noqa: E731
        if self.host:
            self.cache = {}
            for e in (sorted(need) if need is not None else range(NE)):
                w = dec(e)
                self.cache[e] = {lv: (w[lv][0].cpu(), w[lv][1].cpu()) for lv in levels}
            return
        blk = NE // W
        lgu = {lv: torch.empty(blk, 2 * FF, D, dtype=torch.bfloat16, device=dev) for lv in levels}
        ldn = {lv: torch.empty(blk, D, FF, dtype=torch.bfloat16, device=dev) for lv in levels}
        for j, e in enumerate(range(R * blk, (R + 1) * blk)):
            w = dec(e)
            for lv in levels:
                lgu[lv][j] = w[lv][0]; ldn[lv][j] = w[lv][1]
            del w
        self.gu, self.dn = {}, {}
        for lv in levels:
            self.gu[lv] = torch.empty(NE, 2 * FF, D, dtype=torch.bfloat16, device=dev)
            dist.all_gather_into_tensor(self.gu[lv], lgu[lv]); del lgu[lv]
            self.dn[lv] = torch.empty(NE, D, FF, dtype=torch.bfloat16, device=dev)
            dist.all_gather_into_tensor(self.dn[lv], ldn[lv]); del ldn[lv]
        e = ((R + 1) % W) * blk + (L % blk)                       # gather layout check: another rank's expert
        w = dec(e)
        for lv in levels:
            assert torch.equal(self.gu[lv][e], w[lv][0]) and torch.equal(self.dn[lv][e], w[lv][1]), (L, e, lv)

    def w(self, e, lv):
        if self.host:
            gu, dn = self.cache[e][lv]
            return gu.to(self.dev, non_blocking=True), dn.to(self.dev, non_blocking=True)
        return self.gu[lv][e], self.dn[lv][e]


# ------------------------------------------------------------------------------------------------- jF streaming
class PredPool:
    """One GPUJointPredictor template (trees + net built once); fresh per-(request, layer) state via fork(L)."""

    def __init__(self, net, nf, fixed, dev, hm=0.7):
        import gpu_predictor37 as GP
        self.GP = GP
        self.fixed = fixed
        self.tmpl = GP.GPUJointPredictor(list(range(3, 45)), {L: fixed[L] for L in range(3, 45)}, net, n_float=nf, hm=hm,
                                         device=dev, v2_model=V2, bf16=True)
        self.dev = dev

    def fork(self, L):
        t = self.tmpl; p = copy.copy(t); dev = self.dev; NE_ = t.NE
        p.layers = [L]; p.NL = 1
        p.fixed = np.zeros((1, NE_), bool); p.fixed[0, list(self.fixed[L])] = True
        p.lidx = t.lidx[t.layers.index(L):t.layers.index(L) + 1]
        p.fxm = torch.from_numpy(p.fixed).to(dev)
        z32 = lambda: torch.zeros(1, NE_, dtype=torch.float32, device=dev)    # noqa: E731
        z64 = lambda: torch.zeros(1, NE_, dtype=torch.float64, device=dev)    # noqa: E731
        p.E = [z32(), z32()]; p.Et, p.Ea = z32(), z32(); p.wt = p.wa = 0.0
        p.Hc = {h: z64() for h in self.GP.HC}; p.Hs = {h: z64() for h in self.GP.HC}
        p.last = torch.full((1, NE_), -10 ** 6, dtype=torch.long, device=dev)
        p.bc, p.bca, p.bs = z32(), z32(), z64()
        p.h16, p.s16 = z32(), z64()
        p.btok = p.bans = p.nblk = 0; p.seg = 0; p.bst_state = 0
        p.S = None
        p.b_wt = torch.ones((), dtype=torch.float64, device=dev); p.b_wa = torch.ones_like(p.b_wt)
        p.b_state = torch.zeros((), dtype=torch.bool, device=dev)
        p.b_nblk = torch.zeros((), dtype=torch.long, device=dev)
        p.b_pos = torch.zeros((), dtype=torch.float32, device=dev)
        p.graph = None
        return p


def seg_state(tok):
    """post-token think/answer state (0 think, 1 answer after </think>), new request starts at 0 (= step())."""
    s = np.zeros(len(tok), np.uint8); cur = 0
    for j, t in enumerate(tok):
        if t == THINK: cur = 0
        elif t == ETHINK: cur = 1
        s[j] = cur
    return s


def blocks(ids, w, xn, seg, r0, r1):
    """rows [r0, r1) (multiple of 16) -> per-block cnt, cnta, nans, sal (f32), seg_last (= blocks37.block_mats)."""
    nb = (r1 - r0) // G
    ii = ids[r0:r1].astype(np.int64); ww = w[r0:r1]; xx = xn[r0:r1]; sg = seg[r0:r1]
    b = (np.arange(nb * G) // G)[:, None].repeat(TOPK, 1)
    idx = (b * NE + ii).ravel()
    cnt = np.bincount(idx, minlength=nb * NE).reshape(nb, 1, NE).astype(np.float32)
    am = np.repeat(sg[:, None], TOPK, 1).ravel().astype(bool)
    cnta = np.bincount(idx[am], minlength=nb * NE).reshape(nb, 1, NE).astype(np.float32)
    v = (ww.astype(np.float64) ** 2 * xx.astype(np.float64)[:, None]).ravel()
    sal = np.bincount(idx, weights=v, minlength=nb * NE).reshape(nb, 1, NE).astype(np.float32)
    return cnt, cnta, sg.reshape(nb, G).sum(1), sal, sg.reshape(nb, G)[:, -1]


def sal_l4(fx, nf, default, ids, w, xn, lag):
    """orcN / lagN: per 16-row block, the nf non-fixed experts with the highest salience (sum w^2 |x|^2) in the SAME
    block (oracle, perfect foresight = ceiling of any predictor at this budget) or in the PREVIOUS block (lag; block 0
    uses floating_default_N)."""
    n = len(ids); nb = (n + G - 1) // G
    m = np.zeros((n, NE), bool)
    idx = ((np.arange(n) // G)[:, None] * NE + ids.astype(np.int64)).ravel()
    v = (w.astype(np.float64) ** 2 * xn.astype(np.float64)[:, None]).ravel()
    sal = np.bincount(idx, weights=v, minlength=nb * NE).reshape(nb, NE)
    sal[:, fx] = -1
    top = np.argsort(-sal, 1, kind="stable")[:, :nf]
    for b in range(nb):
        cur = np.zeros(NE, bool)
        if lag and b == 0:
            cur[default] = True
        else:
            t = top[b - 1 if lag else b]; cur[t[sal[b - 1 if lag else b, t] > 0]] = True
        m[b * G:(b + 1) * G] = fx | cur
    return m


PGARM = re.compile(r"(ema|po|pg)(\d+)(?:h(\d+))?(?:t(\d+))?")
PGCFG = [float(v) for v in os.environ.get("NQ37_PG", "16,0.5,64,8").split(",")]


def dense_sal(ids, w, xn):
    """[n, NE] per-row salience w^2 |x|^2 of the routed experts."""
    n = len(ids); S = np.zeros(n * NE)
    idx = (np.arange(n)[:, None] * NE + ids.astype(np.int64)).ravel()
    np.add.at(S, idx, (w.astype(np.float64) ** 2 * xn.astype(np.float64)[:, None]).ravel())
    return S.reshape(n, NE)


def pg_l4(fx, default, S, Sp, nf, tb):
    """emaN / pgN / poN per-row level-4 mask [n, NE] for one request (P = 0: every row decodes).  S true salience
    [n, NE] (the EMA sees rows < t only), Sp predicted salience of row t (None: no top-up).  Returns (mask, swaps)."""
    G, hm, hl, SC = int(PGCFG[0]), PGCFG[1], PGCFG[2], int(PGCFG[3]) if Sp is not None else 0
    n = len(S); a = 0.5 ** (1.0 / hl); nres = nf - SC
    m = np.zeros((n, NE), bool)
    cur = np.zeros(NE, bool); cur[default[:nres]] = True
    scr = np.zeros(NE, bool); last = np.zeros(NE); state = np.zeros(NE); credit = 0.0; swaps = 0
    for t in range(n):
        if t and t % G == 0:
            v = np.where(fx, -np.inf, state * np.where(cur, 1 + hm, 1.0))
            new = np.zeros(NE, bool); new[np.lexsort((~cur, -v))[:nres]] = True
            swaps += int((new & ~cur & ~scr).sum())
            scr &= ~new
            dem = np.where(cur & ~new)[0]; free = SC - int(scr.sum())
            if free > 0 and len(dem):
                d = dem[np.argsort(-state[dem])][:free]; scr[d] = True; last[d] = t
            cur = new
        if Sp is not None and tb > 0:
            credit = min(credit + tb, 4.0)
            sp = Sp[t]; miss = np.where(~(fx | cur | scr) & (sp > 0))[0]
            k = min(int(credit), len(miss))
            for e in miss[np.argsort(-sp[miss])[:k]]:
                if scr.sum() >= SC:
                    s_ = np.where(scr)[0]; scr[s_[np.argmin(last[s_])]] = False
                scr[e] = True; last[e] = t
            credit -= k; swaps += k
        m[t] = fx | cur | scr
        last[scr & (S[t] > 0)] = t
        state *= a; state += S[t]
    return m, swaps


def mac_l4(pool, L, nf, default, ids, w, xn, tok, P, seg, budget_x=125, full_from=4096):
    """per-row level-4 expert mask [n, NE] bool for one request at layer L (streaming jF emulation).
    Returns (mask, info)."""
    n = len(ids)
    fx = np.zeros(NE, bool); fx[list(pool.fixed[L])] = True
    dflt = np.zeros(NE, bool); dflt[default] = True
    m = np.zeros((n, NE), bool)
    info = dict(refresh=0, swaps=0, seeded=0)
    # prompt
    if P > 0:
        if P >= full_from:
            m[:P] = True
        else:
            pm = fx | dflt
            if P >= 1024:                                         # budget mode: top-x by the window's own counts
                c = np.bincount(ids[:P].astype(np.int64).ravel(), minlength=NE).astype(np.float64)
                c[pm] = -1
                top = np.argsort(-c, kind="stable")[:budget_x]
                pm = pm.copy(); pm[top[c[top] > 0]] = True
            m[:P] = pm
    p = pool.fork(L)
    cur = dflt.copy()
    nb0 = min(256, P // G)
    if nb0 > 0:
        cnt, cnta, nans, sal, segl = blocks(ids, w, xn, seg, P - nb0 * G, P)
        p.step_chunk(cnt, sal, cnta, nans, segl)
        t = p.target_seed()
        if t is not None and t[0].any():
            cur = t[0] & ~fx
            info["seeded"] = 1
    r = P
    while r < n:
        r1 = min(n, r + G)
        m[r:r1] = fx | cur
        if r1 - r == G and r1 < n:
            cnt, cnta, nans, sal, segl = blocks(ids, w, xn, seg, r, r1)
            p._load_block(cnt[0], sal[0], cnta[0], int(nans[0]), int(segl[0]))
            p._close_block(); p.S = p._score()
            t = p.target(cur[None])
            if t is not None:
                new = t[0] & ~fx
                info["swaps"] += int((new & ~cur).sum()); cur = new
            info["refresh"] += 1
        r = r1
    return m, info


# ---------------------------------------------------------------------------------------------------- forward
def moe_arm(lay, x, src, l4row=None):
    """routed (+ shared) MoE of one arm.  src: ('fp8', Ex) | ('q', QE, level) | ('mix', QE) with l4row [n, NE] bool
    (True -> level 4 else level 2).  Returns (out, ids, w, l4slots)."""
    logits, tw, ti = lay.mlp.gate(x)
    out = torch.zeros_like(x)
    flat = ti.reshape(-1)
    order = torch.argsort(flat, stable=True)
    cnt = torch.bincount(flat, minlength=NE).tolist()
    rows_all = order // TOPK
    starts = np.r_[0, np.cumsum(cnt)]
    pflat = tw.reshape(-1)
    l4slots = 0
    if src[0] == "mix":
        hit4 = torch.gather(l4row, 1, ti).reshape(-1)              # per slot
        l4slots = int(hit4.sum())
    elif src[0] == "q" and src[2] == 4:
        l4slots = int(flat.numel())
    for e in range(NE):
        if cnt[e] == 0:
            continue
        sl = order[starts[e]:starts[e + 1]]
        rows = rows_all[starts[e]:starts[e + 1]]
        p = pflat[sl]
        if src[0] == "fp8":
            parts = [(rows, p, src[1].w(e))]
        elif src[0] == "q":
            parts = [(rows, p, src[1].w(e, src[2]))]
        else:
            h4 = hit4[sl]
            parts = [(rows[h4], p[h4], src[1].w(e, 4)), (rows[~h4], p[~h4], src[1].w(e, 2))]
        for rr, pp, (Wgu, Wd) in parts:
            if rr.numel() == 0:
                continue
            _, y, _, _ = C.teacher(x[rr], Wgu, Wd)
            out.index_add_(0, rr, (y * pp[:, None].to(y.dtype)).to(out.dtype))
    return out + lay.mlp.shared_experts(x), ti, tw, l4slots


def main_run(a):
    dist.init_process_group("nccl")
    R, W = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", R % torch.cuda.device_count()); torch.cuda.set_device(dev)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    cfg = C.AutoConfig.from_pretrained(C.CK).text_config
    cfg.num_local_experts = cfg.n_routed_experts
    arms = a.arms.split(",")
    qlayers = set(range(3, 45)) if a.qlayers == "all" else set(int(x) for x in a.qlayers.split(","))
    plan = torch.load(a.plan, weights_only=False)
    if a.segs:
        want = [tuple(int(v) for v in x.split(":")) for x in a.segs.split(",")]
        sel = [i for i, p in enumerate(plan) if (p["capR"], p["si"]) in want]
        mine = sel[R::W]
    else:
        mine = assign(plan, R, W, a.nseg, a.order)
    segs = [plan[i] for i in mine]
    off = np.r_[0, np.cumsum([s["n"] for s in segs])]
    N = int(off[-1])
    fixed_js = json.load(open(FIXED))
    fixed = {int(L): [int(e) for e in v] for L, v in fixed_js["fixed_set"].items()}
    def default_n(L, nf):
        nr = np.array(fixed_js["n_routed"][str(L)]); fx = set(fixed[L])
        return [int(e) for e in np.argsort(-nr, kind="stable") if int(e) not in fx][:nf]
    net = a.net or (NET63 if os.path.exists(NET63) else NET48)
    pools = {}
    for arm in arms:
        if arm.startswith("mac"):
            nf = int(arm[3:])
            pools[arm] = PredPool(net, nf, fixed, dev)
    segstate = [seg_state(s["tok"]) for s in segs]
    tot = torch.tensor([N, len(segs)], dtype=torch.float64, device=dev)
    allt = [torch.zeros_like(tot) for _ in range(W)]; dist.all_gather(allt, tot)
    if R == 0:
        log(f"world {W} arms {arms} qlayers {a.qlayers} net {net} | rows/segs per rank {[t.tolist() for t in allt]}")
    tok = torch.from_numpy(np.concatenate([s["tok"] for s in segs]).astype(np.int64)).to(dev)
    emb = C.get(C.PFX + "embed_tokens.weight", dev)
    H0 = emb[tok][:, None].expand(-1, cfg.hc_mult, -1).contiguous()
    del emb
    H = {arm: H0.clone() for arm in arms}
    del H0
    l4stat = {arm: torch.zeros(len(CATS), 4, dtype=torch.float64) for arm in arms}   # (l4 slots, slots, l4 sal, sal) per cat
    pinfo = {arm: dict(refresh=0, swaps=0, seeded=0, reqlayers=0) for arm in arms}
    pgk = {arm: int(PGARM.fullmatch(arm).group(3) or 0) for arm in arms if PGARM.fullmatch(arm)}
    Hhist = {arm: {} for arm, k in pgk.items() if k}         # pg arms: half-step index -> H (2L in, 2L+1 post-attn)
    catrow = row_cats(segs, segstate)                                                  # [N, len(CATS)] bool
    t_start = time.time()
    for L in range(a.L0, a.layers_max + 1):
        t0 = time.time()
        lay = C.build_layer(cfg, L, dev)
        sparse = cfg.mlp_layer_types[L] == "sparse"
        for arm in Hhist:
            Hhist[arm][2 * L] = H[arm]
            for LL in [LL for LL in Hhist[arm] if LL < 2 * L + 1 - pgk[arm]]:
                del Hhist[arm][LL]
        for arm in arms:                                        # attention half
            Hn = torch.empty_like(H[arm])
            for k, s in enumerate(segs):
                sl = slice(int(off[k]), int(off[k + 1]))
                h = H[arm][sl][None]
                post, comb, hs = lay.attn_hc(h)
                hs = lay.input_layernorm(hs)
                n = s["n"]
                am = torch.ones(1, n, dtype=torch.bool, device=dev)
                if lay.block_type == "linear_attention":
                    hs = lay.self_attn(hidden_states=hs, attention_mask=am)
                else:
                    hs, _, _ = lay.self_attn(hidden_states=hs, attention_mask=am,
                                             position_ids=torch.arange(n, device=dev)[None], position_embeddings=None)
                Hn[sl] = (post.to(h.dtype).unsqueeze(-1) * hs.unsqueeze(-2)
                          + torch.matmul(comb.to(h.dtype).transpose(-1, -2), h))[0]
            H[arm] = Hn
            if arm in Hhist:
                Hhist[arm][2 * L + 1] = Hn
        torch.cuda.synchronize(); ta = time.time() - t0; t0 = time.time()
        quant = sparse and L in qlayers and any(arm != "fp8" for arm in arms)
        Ex = C.Experts(L, dev) if sparse and (not quant or "fp8" in arms) else None
        QE = None
        if quant:
            lv = sorted({2, 4} & set(lvl for arm in arms for lvl in arm_levels(arm)))
            need = None
            if W == 1 and a.lazy:
                need = set()
                for arm in arms:
                    if arm != "fp8":
                        _, _, ti = lay.mlp.gate(lay.post_attention_layernorm(lay.ffn_hc(H[arm][None])[2])[0])
                        need |= set(torch.unique(ti).tolist())
            QE = QExperts(L, lv, dev, R, W, need)
        torch.cuda.synchronize(); tq = time.time() - t0; t0 = time.time(); tp = 0.0
        for arm in arms:                                        # MoE half
            h = H[arm][None]
            post, comb, hs = lay.ffn_hc(h)
            x = lay.post_attention_layernorm(hs)[0]
            l4row = None
            if not sparse:
                y = lay.mlp(x)
            else:
                if arm == "fp8" or not quant:
                    src = ("fp8", Ex)
                elif arm in ("all2", "all4"):
                    src = ("q", QE, int(arm[3]))
                elif PGARM.fullmatch(arm):
                    tp0 = time.time()
                    kind, nf, kk, tt = PGARM.fullmatch(arm).groups()
                    nf = int(nf); tb = int(tt or 0) / 100
                    _, tw, ti = lay.mlp.gate(x)
                    ids = ti.cpu().numpy(); ww = tw.float().cpu().numpy(); xn = x.float().square().sum(1).cpu().numpy()
                    Sp = None
                    if kind == "pg":
                        _, _, hp = lay.ffn_hc(Hhist[arm][2 * L + 1 - int(kk)][None])
                        xp = lay.post_attention_layernorm(hp)[0]
                        _, twp, tip = lay.mlp.gate(xp)
                        Sp = dense_sal(tip.cpu().numpy(), twp.float().cpu().numpy(), xp.float().square().sum(1).cpu().numpy())
                    fxm = np.zeros(NE, bool); fxm[fixed[L]] = True
                    rows = []
                    for k in range(len(segs)):
                        sl = slice(int(off[k]), int(off[k + 1]))
                        S = dense_sal(ids[sl], ww[sl], xn[sl])
                        mk, sw = pg_l4(fxm, default_n(L, nf), S, S if kind == "po" else (Sp[sl] if Sp is not None else None),
                                       nf, tb)
                        rows.append(mk); pinfo[arm]["swaps"] += sw; pinfo[arm]["refresh"] += len(mk)
                        pinfo[arm]["reqlayers"] += 1
                    l4row = torch.from_numpy(np.concatenate(rows)).to(dev)
                    tp += time.time() - tp0
                    src = ("mix", QE)
                else:
                    nf = int(arm[4:]) if arm.startswith("stat") else int(arm[3:])
                    oracle = arm.startswith(("orc", "lag"))
                    dflt = default_n(L, nf)
                    _, tw, ti = lay.mlp.gate(x)
                    if arm.startswith("stat"):
                        l4 = np.zeros(NE, bool); l4[fixed[L]] = True; l4[dflt] = True
                        l4row = torch.from_numpy(l4).to(dev)[None].expand(N, -1)
                    elif oracle:
                        ids = ti.short().cpu().numpy(); ww = tw.float().cpu().numpy()
                        xn = x.float().square().sum(1).cpu().numpy()
                        fxm = np.zeros(NE, bool); fxm[fixed[L]] = True
                        rows = [sal_l4(fxm, nf, dflt, ids[int(off[k]):int(off[k + 1])], ww[int(off[k]):int(off[k + 1])],
                                       xn[int(off[k]):int(off[k + 1])], arm.startswith("lag")) for k in range(len(segs))]
                        l4row = torch.from_numpy(np.concatenate(rows)).to(dev)
                    else:
                        tp0 = time.time()
                        ids = ti.short().cpu().numpy(); ww = tw.half().cpu().numpy()
                        xn = x.float().square().sum(1).cpu().numpy()
                        rows = []
                        for k, s in enumerate(segs):
                            sl = slice(int(off[k]), int(off[k + 1]))
                            mk, inf = mac_l4(pools[arm], L, nf, dflt, ids[sl], ww[sl], xn[sl], s["tok"], s["P"],
                                             segstate[k], a.budget_x)
                            rows.append(mk)
                            for kk in ("refresh", "swaps", "seeded"):
                                pinfo[arm][kk] += inf[kk]
                            pinfo[arm]["reqlayers"] += 1
                        l4row = torch.from_numpy(np.concatenate(rows)).to(dev)
                        tp += time.time() - tp0
                    src = ("mix", QE)
                y, ti, tw, _ = moe_arm(lay, x, src, l4row)
                if a.dump and arm == "fp8":
                    os.makedirs(f"{PRIV}/{a.tag}/dump", exist_ok=True)
                    torch.save(dict(ids=ti.short().cpu(), w=tw.half().cpu(), xn=x.float().square().sum(1).cpu(),
                                    x=x.cpu() if L <= 4 else None, segs=[(s["capR"], s["si"]) for s in segs]), f"{PRIV}/{a.tag}/dump/L{L}.r{R}.pt")
                # level-4 slot accounting per category (quantized layers, quantized arms only)
                if arm == "fp8" or not quant:
                    hit = None
                elif arm in ("all2", "all4"):
                    hit = torch.full_like(ti, arm == "all4", dtype=torch.bool)
                else:
                    hit = torch.gather(l4row, 1, ti)
                hs_ = hit.sum(1).double().cpu() if hit is not None else None
                if hit is not None:
                    sv = tw.double() ** 2 * x.double().square().sum(1, keepdim=True)
                    ss_ = (sv * hit).sum(1).cpu(); st_ = sv.sum(1).cpu()
                for ci in range(len(CATS) if hit is not None else 0):
                    mm = catrow[:, ci]
                    l4stat[arm][ci, 0] += float(hs_[mm].sum()); l4stat[arm][ci, 1] += float(mm.sum()) * TOPK
                    l4stat[arm][ci, 2] += float(ss_[mm].sum()); l4stat[arm][ci, 3] += float(st_[mm].sum())
            H[arm] = (post.to(h.dtype).unsqueeze(-1) * y[None].unsqueeze(-2)
                      + torch.matmul(comb.to(h.dtype).transpose(-1, -2), h))[0]
        if a.dump:
            hm_ = {}
            for arm in arms:
                hm_[arm] = [float(H[arm][int(off[k]):int(off[k + 1])].float().abs().amax()) for k in range(len(segs))]
            log(f"L{L} max|H| per seg " + " ".join(f"{arm}:" + ",".join(f"{v:.3g}" for v in hm_[arm]) for arm in arms))
        del Ex, QE, lay
        torch.cuda.empty_cache()
        torch.cuda.synchronize(); tm = time.time() - t0
        if R == 0:
            log(f"L{L} {cfg.layer_types[L][:6]} {cfg.mlp_layer_types[L]}{' Q' if quant else ''} attn {ta:.1f}s "
                f"load/dec {tq:.1f}s moe {tm:.1f}s (pred {tp:.1f}s) mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G "
                f"elapsed {(time.time() - t_start) / 60:.1f} min")
        dist.barrier()
    final(a, cfg, H, segs, off, catrow, l4stat, pinfo, arms, net, dev, R, W)
    dist.barrier()
    dist.destroy_process_group()


def arm_levels(arm):
    if arm == "fp8":
        return ()
    if arm in ("all2", "all4"):
        return (int(arm[3]),)
    return (2, 4)


def row_cats(segs, segstate):
    rows = []
    for s, st in zip(segs, segstate):
        n, P = s["n"], s["P"]
        c = np.zeros((n, len(CATS)), bool)
        j = np.arange(n)
        c[:, 0] = True
        c[:, 1] = j < P
        c[:, 2] = j >= P
        c[:, 3] = (j >= P) & (P >= G)
        c[:, 4] = (j >= P) & (P < G)
        c[:, 5] = st == 0
        c[:, 6] = st == 1
        # role of the context at row j (row j predicts token j+1): last role token at or before j
        rl = np.zeros(n, np.int8); cur = 0                     # 0 raw (no role token yet), 1 assistant, 2 other role
        for jj, t in enumerate(s["tok"]):
            if t == ROLE_ASST: cur = 1
            elif t in ROLE_CTX: cur = 2
            rl[jj] = cur
        c[:, 7] = rl == 1
        c[:, 8] = rl == 2
        c[:, 9] = rl == 0
        rows.append(c)
    return torch.from_numpy(np.concatenate(rows))


def final(a, cfg, H, segs, off, catrow, l4stat, pinfo, arms, net, dev, R, W):
    norm = M.Glm5NextTextRMSNorm(D, cfg.rms_norm_eps).to(dev)
    norm.weight.copy_(C.get(C.PFX + "norm.weight", dev, torch.float32))
    norm = norm.bfloat16()
    head = C.get("lm_head.weight", dev)
    hh = M.Glm5NextTextHyperHead()
    # sums per arm per cat: rows, kl64, klfull(vs fp8 arm), top1 vs teacher, top1 vs fp8, ce, ce_fp8
    K = 7
    S = {arm: torch.zeros(len(CATS), K, dtype=torch.float64) for arm in arms}
    tops = {}
    for R_ in sorted({s["capR"] for s in segs} - {-1}):
        tops[R_] = torch.load(f"{CAP}/final/top64.r{R_}.pt", weights_only=False)
    per = []
    for k, s in enumerate(segs):
        sl = slice(int(off[k]), int(off[k + 1]))
        n = s["n"]
        if s.get("teach"):
            from safetensors import safe_open
            with safe_open(s["teach"], "pt") as f_:
                lT = F.log_softmax(f_.get_tensor("logits").to(dev).float(), -1)
            assert lT.shape[0] == n - 1, (lT.shape, n)
            ti = lT.argmax(-1, keepdim=True)
            s["ce_teacher"] = float(-lT.gather(1, torch.from_numpy(s["tok"][1:].astype(np.int64)).to(dev)[:, None]).mean())
        else:
            lT = None
            tt = tops[s["capR"]][s["si"]]
            tv = tt["v"][:n - 1].to(dev).float(); ti = tt["i"][:n - 1].to(dev).long()
        tgt = torch.from_numpy(s["tok"][1:].astype(np.int64)).to(dev)
        cr = catrow[int(off[k]):int(off[k + 1]) - 1].to(dev).double()          # rows 0..n-2
        lps = {}
        for arm in arms:
            z = F.linear(norm(hh(H[arm][sl][None]))[0], head).float()[:n - 1]
            lps[arm] = F.log_softmax(z, -1); del z
        rec = dict(capR=s["capR"], si=s["si"], n=n, P=s["P"], win=s.get("win"))
        for arm in arms:
            lp = lps[arm]
            if lT is not None:                                             # full-vocab KL(BF16 teacher || arm)
                kl64 = (lT.exp() * (lT - lp)).sum(1)
            else:
                pT = tv.exp(); lQ = lp.gather(1, ti)
                tailT = (1 - pT.sum(1)).clamp(min=1e-12); tailQ = (1 - lQ.exp().sum(1)).clamp(min=1e-12)
                kl64 = (pT * (tv - lQ)).sum(1) + tailT * (tailT.log() - tailQ.log())
            if "fp8" in lps:
                lf = lps["fp8"]
                klf = (lf.exp() * (lf - lp)).sum(1)
                t1f = (lp.argmax(-1) == lf.argmax(-1)).double()
                cef = -lf.gather(1, tgt[:, None])[:, 0]
            else:
                klf = torch.zeros_like(kl64); t1f = torch.zeros_like(kl64); cef = torch.zeros_like(kl64)
            t1 = (lp.argmax(-1) == ti[:, 0]).double()
            ce = -lp.gather(1, tgt[:, None])[:, 0]
            v = torch.stack([torch.ones_like(kl64).double(), kl64.double(), klf.double(), t1, t1f, ce.double(), cef.double()], 1)
            S[arm] += (cr.T @ v).cpu()
            if a.dump:
                os.makedirs(f"{PRIV}/{a.tag}/dump", exist_ok=True)
                ent = -(lp.exp() * lp).sum(1)
                torch.save(dict(kl64=kl64.cpu(), klf=klf.cpu(), ce=ce.cpu(), ent=ent.cpu(), t1=t1.cpu(), t1f=t1f.cpu(),
                                top=[x.cpu() for x in lp.topk(8, -1)]), f"{PRIV}/{a.tag}/dump/final_{arm}_{s['capR']}_{s['si']}.pt")
            rec[arm] = dict(kl64=float(kl64.mean()), klf=float(klf.mean()), top1=float(t1.mean()), ce=float(ce.mean()))
        per.append(rec)
        del lps, lT
    for arm in arms:
        t = S[arm].to(dev); dist.all_reduce(t); S[arm] = t.cpu()
        t = l4stat[arm].to(dev); dist.all_reduce(t); l4stat[arm] = t.cpu()
        t = torch.tensor([pinfo[arm][k] for k in ("refresh", "swaps", "seeded", "reqlayers")], dtype=torch.float64, device=dev)
        dist.all_reduce(t); pinfo[arm] = dict(zip(("refresh", "swaps", "seeded", "reqlayers"), t.tolist()))
    os.makedirs(f"{PRIV}/{a.tag}", exist_ok=True)
    json.dump(per, open(f"{PRIV}/{a.tag}/perseg.r{R}.json", "w"))
    tce = torch.tensor([sum(s["ce_teacher"] * (s["n"] - 1) for s in segs), sum(s["n"] - 1 for s in segs)],
                       dtype=torch.float64, device=dev)
    dist.all_reduce(tce)
    if R == 0:
        res = dict(tag=a.tag, kvq=KVQ or None, time=time.strftime("%Y-%m-%d %H:%M:%S %Z"), arms=arms, qlayers=a.qlayers, net=net,
                   world=W, nseg=a.nseg, budget_x=a.budget_x,
                   teacher_stored=dict(rows=int(tce[1]), ppl=float(np.exp(float(tce[0]) / float(tce[1])))),
                   notes="kl64 = top-64+tail KL(stored fp8 teacher || arm) (lower bound of full KL); klfull = full-vocab "
                         "KL(inline fp8 arm || arm); top1 = argmax agreement with the stored teacher; ppl over next-token "
                         "targets; rows = positions 0..n-2 of each segment; l4_frac = routed slots served at level 4.",
                   cats={})
        for ci, c in enumerate(CATS):
            d = {}
            for arm in arms:
                s = S[arm][ci]; nrow = float(s[0])
                if nrow == 0:
                    continue
                d[arm] = dict(rows=int(nrow), kl64=float(s[1] / nrow), klfull=float(s[2] / nrow),
                              top1=float(s[3] / nrow), top1_vs_fp8=float(s[4] / nrow), ppl=float(np.exp(float(s[5] / nrow))),
                              ppl_fp8=float(np.exp(float(s[6] / nrow))),
                              l4_frac=float(l4stat[arm][ci, 0] / max(float(l4stat[arm][ci, 1]), 1)),
                              l4_sal_frac=float(l4stat[arm][ci, 2] / max(float(l4stat[arm][ci, 3]), 1e-30)))
            res["cats"][c] = d
        res["predictor"] = {arm: pinfo[arm] for arm in arms if arm.startswith("mac") or PGARM.fullmatch(arm)}
        res["pg_cfg"] = dict(zip(("G", "hm", "hl", "SC"), PGCFG))
        os.makedirs(OUTD, exist_ok=True)
        fn = f"{OUTD}/results_{a.tag}.json"
        json.dump(res, open(fn, "w"), indent=1)
        log(f"wrote {fn}")
        for c in ("all", "decode", "asst", "ctx", "raw"):
            for arm, d in res["cats"].get(c, {}).items():
                log(f"{c:10s} {arm:7s} rows {d['rows']:6d} KL64 {d['kl64']:.5f} KLfull {d['klfull']:.5f} top1 {d['top1']:.4f} "
                    f"ppl {d['ppl']:.4f} (fp8 {d['ppl_fp8']:.4f}) L4 {d['l4_frac']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prep", "prep_bm", "run"])
    ap.add_argument("--plan", default=PLAN)
    ap.add_argument("--bm-windows", default="confirmation-0000,confirmation-0001,confirmation-0002,confirmation-0003")
    ap.add_argument("--arms", default="fp8,all2,mac63,mac59,stat63,all4")
    ap.add_argument("--qlayers", default="all", help='"all" or comma list of MoE layers that are quantized (others fp8)')
    ap.add_argument("--nseg", type=int, default=0, help="smoke: only N segments")
    ap.add_argument("--order", default="plan", choices=["plan", "short"])
    ap.add_argument("--net", default="", help="jF checkpoint (default jf/models/jF63.pt if present else release nf48 jF.pt)")
    ap.add_argument("--budget-x", type=int, default=125)
    ap.add_argument("--lazy", type=int, default=1, help="world 1: decode only experts routed by some arm")
    ap.add_argument("--L0", type=int, default=0)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--dump", type=int, default=0, help="PRIVATE: fp8-arm routing per layer (debug vs teacher trace)")
    ap.add_argument("--layers-max", type=int, default=44)
    ap.add_argument("--segs", default="", help="capR:si,... -- only these segments (debug)")
    a = ap.parse_args()
    {"prep": prep, "prep_bm": prep_bm, "run": main_run}[a.mode](a)
