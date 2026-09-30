"""Candidate-quantiser plug-ins for nq_e2e.py (thread 18).

Every candidate is one object with this interface:

    class Quantiser:
        name: str
        def begin_layer(self, layer: int, device) -> None          # optional; load this layer's artefacts
        def expert(self, layer: int, expert: int, ref) -> dict | None
            # ref() -> {"gate_proj": [2048,6144], "up_proj": [2048,6144], "down_proj": [6144,2048]}
            #          bf16 on device = FP8 reference dequant (lazy; only read if you call it)
            # return the same keys (any float dtype, [out, in], ORIGINAL un-rotated basis, on device)
            # or None -> the reference expert is used (counted as "fallback" in the report)
        def end_layer(self, layer: int) -> None                    # optional; free memory

Spec strings (``--cand NAME=SPEC``):
    ref                               identity (sanity: KLD must be exactly 0)
    rtn:bits=4,group=128              round-to-nearest asym min/max per (row, group) of the FP8 reference
    dir:/path                         dequantised safetensors: any *.safetensors under /path with keys
                                      model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.weight  or
                                      layer{L}.expert{E}.{gate,up,down}_proj ; missing -> reference
    mix:lo=DIR,hi=DIR,set=F.json[,hi_layers=3-6]   per-expert choice between two dequantised dirs (nqdef)
    nestquant:root=/path,level=4      artefacts {root}/L{L}/experts/E{E}.pt (thread-25 campaign) or
                                      thread-12 {root}/layer_{L:03d}/expert_{E:03d}.pt
                                      decoded with threads/12-reference-encoder/nq_decode.decode_expert
    exl3:root=/path,bits=4            {root}/layer_{L:03d}/expert_{E:03d}/expert_{bits}.bin (orbit-duet
                                      legacy .bin; decoded by exllamav3 via orbit_duet.exl3_adapter)
    nvfp4:root=/path                  {root}/layer_{L:03d}/expert_{E:03d}/weights.pt (orbit-duet ModelOpt
                                      payload; decoded by orbit_duet.nvfp4_reference.decode)
    py:/file.py:Class[:k=v,...]       any external class implementing the interface
Optional ``layers=a-b`` in any spec restricts quantisation to those layers (others -> reference).
"""
import importlib
import importlib.util
import json
import numpy as np
import os
import sys

import torch

PROJ = ("gate_proj", "up_proj", "down_proj")
NQ12 = "/home/coder/git/nestquant/threads/12-reference-encoder"
ORBIT = "/home/coder/git/orbit-duet"


def _kv(s):
    out = {}
    for part in filter(None, s.split(",")):
        k, _, v = part.partition("=")
        out[k] = v
    return out


def _layers(kv):
    if "layers" not in kv:
        return None
    a, _, b = kv.pop("layers").partition("-")
    return set(range(int(a), int(b or a) + 1))


class Base:
    name = "?"
    only_layers = None

    def begin_layer(self, layer, device):
        self.dev = device

    def end_layer(self, layer):
        pass

    def active(self, layer):
        return self.only_layers is None or layer in self.only_layers


class Ref(Base):
    is_ref = True

    def expert(self, layer, expert, ref):
        return None


class RTN(Base):
    """Asymmetric min/max RTN per (row, group-of-inputs). Smoke-test quantiser only."""

    def __init__(self, bits=4, group=128):
        self.bits, self.group = int(bits), int(group)

    def expert(self, layer, expert, ref):
        if not self.active(layer):
            return None
        out = {}
        for p, w in ref().items():
            n, k = w.shape
            g = w.float().view(n, k // self.group, self.group)
            lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
            qmax = 2 ** self.bits - 1
            s = (hi - lo).clamp_min(1e-12) / qmax
            q = ((g - lo) / s).round().clamp(0, qmax)
            out[p] = (q * s + lo).view(n, k).to(torch.bfloat16)
        return out


class Dir(Base):
    """Directory of dequantised safetensors (any sharding; header-indexed, pread, no mmap)."""

    def __init__(self, path):
        from nq_io import SafeIndex
        self.idx = SafeIndex(path)

    def _key(self, layer, expert, p):
        for k in (f"model.layers.{layer}.mlp.experts.{expert}.{p}.weight",
                  f"layer{layer}.expert{expert}.{p}"):
            if k in self.idx:
                return k
        return None

    # Shared across Dir instances (and Mix's inner Dirs) for the current (layer, expert): several streams that
    # read the same predecoded dir (nq2, nqdef, nq2_early, ...) transfer each expert once.
    _cur = {"le": None, "w": {}}

    def expert(self, layer, expert, ref):
        if not self.active(layer):
            return None
        c = Dir._cur
        if c["le"] != (layer, expert, str(self.dev)):
            c["le"], c["w"] = (layer, expert, str(self.dev)), {}
        root = os.path.realpath(self.idx.root)
        if root in c["w"]:
            return c["w"][root]
        keys = [self._key(layer, expert, p) for p in PROJ]
        if any(k is None for k in keys):
            W = None
        else:
            W = {p: self.idx.get(k, self.dev) for p, k in zip(PROJ, keys)}
        c["w"][root] = W
        return W

    def end_layer(self, layer):
        Dir._cur.update(le=None, w={})


def nq_artifact_path(root, layer, expert):
    """Thread-25 campaign layout {root}/L{L}/experts/E{E}.pt, thread-12 {root}/layer_LLL/expert_EEE.pt, or a
    single-layer dir {root}/L{L}/E{E}.pt / {root}/experts/E{E}.pt / {root}/E{E}.pt (tuned re-encodes, --expert-override)."""
    for f in (f"{root}/L{layer}/experts/E{expert}.pt", f"{root}/layer_{layer:03d}/expert_{expert:03d}.pt",
              f"{root}/L{layer}/E{expert}.pt", f"{root}/experts/E{expert}.pt", f"{root}/E{expert}.pt"):
        if os.path.exists(f):
            return f
    return None


def load_level4_set(path):
    """{"layers": {"L": [experts at level 4]}, ...} (written by nq_defset.py) -> {int L: set(int e)}."""
    d = json.load(open(path))
    return {int(L): set(map(int, v)) for L, v in d["layers"].items()}


class Mix(Base):
    """Per-expert mix of two dequantised dirs: hi (level-4) for experts in set[L] or any expert of a layer in
    hi_layers, lo (level-2) otherwise.  mix:lo=DIR,hi=DIR,set=FILE.json[,hi_layers=3-6][,layers=a-b]"""

    def __init__(self, lo, hi, set=None, hi_layers=None):
        self.lo, self.hi = Dir(lo), Dir(hi)
        self.l4 = load_level4_set(set) if set else {}
        self.hi_layers = set_from_range(hi_layers) if hi_layers else set_from_range("")
        self.n_hi = self.n_lo = 0

    def begin_layer(self, layer, device):
        self.dev = self.lo.dev = self.hi.dev = device

    def end_layer(self, layer):
        Dir._cur.update(le=None, w={})

    def level_of(self, layer, expert):
        return 4 if (layer in self.hi_layers or expert in self.l4.get(layer, ())) else 2

    def expert(self, layer, expert, ref):
        if not self.active(layer):
            return None
        use_hi = layer in self.hi_layers or expert in self.l4.get(layer, ())
        W = (self.hi if use_hi else self.lo).expert(layer, expert, ref)
        if W is not None:
            if use_hi:
                self.n_hi += 1
            else:
                self.n_lo += 1
        return W


class Noise(Base):
    """FP8 reference weights; nq_e2e.moe_multi adds N(0, sigma * rms(y)) per element to every routed-expert output y
    (fp32, then rounded to the activation dtype like any expert output).  Noise floor for route agreement / KLD.
    noise:sigma=1e-3[,seed=0]"""

    def __init__(self, sigma="1e-3", seed="0"):
        self.out_noise, self.seed = float(sigma), int(seed)

    def expert(self, layer, expert, ref):
        return ref() if self.active(layer) else None


DELTA_DEFAULT = "/tmp/nestquant/31-delta/delta_table.json"


def load_delta(path, col="delta"):
    """T31 per-expert salience factor: nq-delta-v1 json {'per_layer': {L: {col: [256]}}} (col delta = drel * G | drel)
    or legacy {'delta': {L: [256]} | [[256] per layer]} -> {int L: tensor float64}."""
    j = json.load(open(path or DELTA_DEFAULT))
    if "per_layer" in j:
        return {int(L): torch.tensor(v[col], dtype=torch.float64) for L, v in j["per_layer"].items()}
    assert col == "delta", col
    d = j["delta"]
    it = d.items() if isinstance(d, dict) else enumerate(d)
    return {int(L): torch.tensor(v, dtype=torch.float64) for L, v in it if v is not None}


KINDS = ("count", "w", "sal", "salrel", "cntdelta", "wx2")   # per-slot value: 1 | gate w | w^2|x|^2 delta_e |
#                                                                 w^2|x|^2 drel_e | delta_e | w^2|x|^2 (no delta)


class Adapt(Base):
    """Causal replay of the serving level scheduler (streaming/scheduler.py defaults) per layer, per sequence:
    fixed = manifest default_allocation (always level 4); floating = n_float experts, starting at floating_default,
    re-chosen every `refresh` tokens as the top-n_float non-fixed experts by routing counts decayed with half-life
    `half_life` tokens (the stream's own top-8 routing), ties -> lower id (np.argsort stable).  One-refresh lag:
    the set chosen at refresh r (score over tokens < refresh*r) serves chunk r+1; chunks 0 and 1 use floating_default.
    Ignored vs scheduler.py: the SSD byte budget deferral (every upgrade lands exactly one refresh later) and the
    big_frac guard (never triggers in per-token decode: 8 experts/token); downgrades are applied with the same lag.
    adapt:lo=DIR,hi=DIR[,hi2=DIR],manifest=serving/tp4/manifest.json[,half_life=512,refresh=64,n_float=51,lag=1]
    (hi2: second level-4 dir for experts missing from hi, e.g. the complement of a partial predecode)"""

    def __init__(self, lo, hi, manifest, hi2=None, half_life=512, refresh=64, n_float=51, lag=1, chain=0, NE=256,
                 predictor="ema", hm=0.5, chunk=None, up=45, ahead=None, rank="count", delta=None, score="count",
                 oracle=None, horizon=64, block=16, gscale=None, gkeep=0, sal_hl=128, gbdt_model=None,
                 gbdt_scale=None, fb=0, gmode="next_refresh", grlo=20, grhi=121, salstat=0, nf_map=None,
                 joint=None, joint_map=None):
        # T32 nf_map=JSON {layer: n_float}: per-layer floating slot count (T33k allocation B; layers absent: n_float);
        # diag then also records nf and the min/max floating-set size actually served per layer.
        self.nf_map = {int(k): int(v) for k, v in json.load(open(nf_map)).items()} if nf_map else None
        # T32: gmode=sync (score of block k applied at its end) | next_refresh (serve default, one-block lag);
        # grlo/grhi: GBDT candidate band over EMA256 ranks (default 20..120 + forced top-20; grlo=0,grhi=256: all)
        self.gmode, self.grlo, self.grhi = gmode, int(grlo), int(grhi)
        assert gmode in ("next_refresh", "sync"), gmode
        # T32 chain=map: chain=1 but also break where the corpus's NQ_CORPUS_DIR/NAME.map.npz "chain" (else "task")
        # changes between consecutive windows (one sequence per task; corpora without a map: whole run)
        self.chainmap = str(chain) == "map"
        self.chain = 1 if self.chainmap else int(chain)   # 1: carry scores + floating set across consecutive windows of one corpus
        self.predictor, self.hm = predictor, float(hm)   # gbdt: streaming/gbdt_predictor.GBDTPredictor (see _core_gbdt)
        assert predictor in ("ema", "gbdt"), predictor
        # T32: gbdt_model=PATH swaps the LightGBM tree file (same 5 features; e.g. the salience-target retrain)
        self.gbdt_model = gbdt_model
        assert gbdt_model is None or (predictor == "gbdt" and os.path.exists(gbdt_model)), gbdt_model
        # T32 v2 (streaming/gbdt_predictor_v2.py): 9-feature salience model, or gbdt_scale=mps (hits x EMA128 sal/hit);
        # fed per-step sal[e] = sum w^2 |x|^2 from the stream's own routing (act_full)
        self.gbdt_v2 = False
        if gbdt_model is not None:
            with open(gbdt_model) as fh:
                for ln in fh:
                    if ln.startswith("feature_names="):
                        self.gbdt_v2 = len(ln.split("=", 1)[1].split()) > 5
                        break
        # T33i joint=PATH (.pt): threads/33-search/joint JointPredictor = v2 over all 256 experts (its own tree file
        # 32-gbdt-sal/models/v2_sal_tweedie1.5.txt, the net's training input) + a 2-layer transformer residual; same
        # step/target interface as GBDTPredictorV2 (salience fed per step), net on the layer's device.  Needs
        # predictor=gbdt; gbdt_model / grlo / grhi are ignored by the joint arm.  Default off.
        # joint_map=JSON {"corpus": NAME | [NAMES], "map": .map.npz | [one per corpus] (task, names per window), "by_task": {task: .pt},
        # "default": .pt}: task = int index into names, or the task_id string itself (fp8dec maps); per-chain net by the chain's task (out-of-sample fold models on decode corpora); windows
        # of other corpora use "default"; a chain must not span tasks.  Window ids from nq_e2e.WIN_IDS (__main__).
        self.joint, self.jmap = joint, None
        if joint_map is not None:
            jm = json.load(open(joint_map))
            cs, ms = (jm["corpus"], jm["map"]) if isinstance(jm["corpus"], list) else ([jm["corpus"]], [jm["map"]])
            task = {}
            for c_, m_ in zip(cs, ms):
                z = np.load(m_)
                task[c_] = ([str(t) for t in z["task"]] if z["task"].dtype.kind in "USO"
                            else [str(z["names"][t]) for t in z["task"]])
            self.jmap = dict(task=task, by=jm["by_task"], default=jm.get("default"))
            for p_ in list(self.jmap["by"].values()) + [self.jmap["default"]]:
                assert p_ is None or os.path.exists(p_), p_
            joint = joint or self.jmap["default"] or "map"
            self.joint = joint
        if joint is not None:
            assert predictor == "gbdt" and (joint == "map" or os.path.exists(joint)), joint
            self.gbdt_v2 = True
        self.gbdt_scale = gbdt_scale
        assert gbdt_scale in (None, "mps") and (gbdt_scale is None or predictor == "gbdt"), gbdt_scale
        if chunk is not None:            # chunked prefill: tokens [kC,(k+1)C) served by the EMA set through chunk k-1
            refresh, lag = int(chunk), 0
        # per-chunk upgrades (chunked-prefill lookahead arms): chunk k additionally serves at level 4 the top-`up`
        # non-fixed experts of chunk k ranked by `rank` (count | w = sum of gate weights | sal = sum w^2 |x|^2 delta_e)
        # from the router lookahead `ahead` layers back (1 | 2; set by nq_e2e.moe_multi as self.la_full) or, ahead=0,
        # from the chunk's actual routing (oracle upper bound).  Resident floating set = the chunk-lag set.
        self.ahead = None if ahead is None else int(ahead)
        self.up, self.rank = int(up), rank
        assert rank in KINDS, rank
        # decode salience arms (own routing via moe_multi -> self.act_full = (ids, w, |x|^2 of the normalised input)):
        #  score=K     EMA scheduler scores per-token value K instead of counts (ema_sal: K = sal)
        #  oracle=K    floating set of each `block`-token block = top-n_float non-fixed by ACTUAL value K summed over
        #              the next `horizon` tokens from the block start (causal ceiling; no lag, no hysteresis)
        #  gscale=K    predictor=gbdt: GBDT predicted next-64 hits x delta_K,e x EMA(sal_hl) of the expert's mean
        #              w^2 |x|^2 per hit, then the predictor's own hysteresis/top-51 (gkeep=1: keep the always-kept
        #              EMA256 top-20 forced; else they get 64 x their EMA256 rate as predicted hits)
        self.score, self.oracle, self.H, self.B = score, oracle, int(horizon), int(block)
        self.gscale, self.gkeep, self.sal_a = gscale, int(gkeep), 0.5 ** (1 / float(sal_hl))
        for k in (score, oracle or "count", gscale or "count"):
            assert k in KINDS, k
        assert gscale is None or predictor == "gbdt"
        kinds = {rank if ahead is not None else None, score, oracle, gscale}
        self.dfile = delta
        self.dtab = {c: load_delta(delta, c) for k, c in (("sal", "delta"), ("salrel", "drel")) if k in kinds}
        if "cntdelta" in kinds and "delta" not in self.dtab:
            self.dtab["delta"] = load_delta(delta, "delta")
        self.delta = self.dtab.get("delta")
        if rank == "salrel" and ahead is not None:
            self.delta = self.dtab["drel"]
        # fb=d (ahead arms, SM120 d039002 parity): the first d MoE layers have no lookahead source and take their
        # upgrades from the previous chunk's actual routing (chunk 0: none)
        self.fb = int(fb)
        # T32 salstat=1: diag also gets l4_sal / sal_tot = sum w^2|x|^2 over hot / all slots of the stream's routing
        self.salstat = int(salstat)
        self.needs_act = score != "count" or oracle is not None or gscale is not None or self.gbdt_v2 or \
            gbdt_scale is not None or self.fb > 0 or self.salstat > 0
        self.act_full = None
        self.la_full = None
        self.lo, self.hi = Dir(lo), Dir(hi)
        self.hi2 = Dir(hi2) if hi2 else None
        m = json.load(open(manifest))
        self.fixed = {int(L): sorted(map(int, v)) for L, v in m["default_allocation"].items()}
        self.fdef = {int(L): [int(e) for e in v] for L, v in m["floating_default"].items()}
        self.a = 0.5 ** (1 / float(half_life))
        self.R, self.nf, self.lag, self.NE = int(refresh), int(n_float), int(lag), NE
        self.nf0 = self.nf
        assert self.lag in (0, 1), "lag 0 (chunked prefill) or 1 (decode refresh) only"
        assert self.ahead is None or self.lag == 0, "lookahead upgrades are per prefill chunk (chunk=C)"
        self.n_hi = self.n_lo = 0
        self.diag = {}

    def begin_layer(self, layer, device):
        self.dev = self.lo.dev = self.hi.dev = device
        if self.hi2 is not None:
            self.hi2.dev = device

    def end_layer(self, layer):
        Dir._cur.update(le=None, w={})

    @torch.no_grad()
    def schedule(self, layer, ids, seq, groups=None):
        """chain=0: every window is its own sequence (state reset); chain=1: each run of consecutive windows of the
        same corpus (groups[w]) is one sequence (needs contiguous sharding, NQ_SHARD=contig, for document order)."""
        N = ids.shape[0] // seq
        if self.nf_map is not None:
            self.nf = self.nf_map.get(int(layer), self.nf0)
        if not self.chain:
            runs = [(0, N)]
            L = seq
        else:
            g = list(groups) if groups is not None else [0] * N
            runs, w0 = [], 0
            for w in range(1, N + 1):
                if w == N or g[w] != g[w0]:
                    runs.append((w0, w)); w0 = w
            if self.chain > 1:       # T32 chain=K>1: one sequence per K windows (fp8dec: one task = K windows)
                runs = [(a, min(a + self.chain, b)) for r0, b in runs for a in range(r0, b, self.chain)]
            if self.chainmap:
                runs = self._split_by_map(runs, g)
        hi, serves, tot = [], [], {}
        for w0, w1 in runs:
            self._la = None if self.la_full is None else tuple(t[w0 * seq:w1 * seq] for t in self.la_full)
            self._act = None if self.act_full is None else tuple(t[w0 * seq:w1 * seq] for t in self.act_full)
            self._run = (w0, w1)
            h, sv, st = self._core(layer, ids[w0 * seq:w1 * seq], seq if not self.chain else (w1 - w0) * seq)
            if getattr(self, "salstat", 0) and self._act is not None:
                ai, aw, ax = self._act
                assert torch.equal(ai.long(), ids[w0 * seq:w1 * seq].long()), "act routing != scheduled routing"
                v = aw.double().pow(2) * ax.double()[:, None]
                st = dict(st, sal_tot=float(v.sum()), l4_sal=float((v * h.to(v.device).double()).sum()))
            hi.append(h); serves.append(sv)
            for k, v in st.items():
                if isinstance(v, dict):          # histograms {value: count}
                    h_ = tot.setdefault(k, {})
                    for kk, vv in v.items():
                        h_[kk] = h_.get(kk, 0) + vv
                else:
                    tot[k] = tot.get(k, 0) + v
        tot["chains"] = [w1 - w0 for w0, w1 in runs] if self.chain else None
        if self.nf_map is not None:
            szs = [np.asarray(torch.as_tensor(sv).sum(-1).cpu()) for sv in serves]
            tot.update(nf=self.nf, float_max=int(max(z.max() for z in szs)), float_min=int(min(z.min() for z in szs)))
        self.diag[layer] = tot
        return torch.cat(hi), serves

    def _slot_value(self, layer, kind, ids, w=None, xn=None):
        """per routed slot [T, 8] float64 value of `kind` (see KINDS)."""
        if kind == "count":
            return torch.ones(ids.shape, dtype=torch.float64, device=ids.device)
        if w is None:
            ids_, w, xn = self._act
            assert torch.equal(ids_.long(), ids.long()), "act_full routing != scheduled routing"
        if kind == "w":
            return w.double()
        if kind == "wx2":
            return w.double().pow(2) * xn.double()[:, None]
        if kind == "cntdelta":
            return self.dtab["delta"][layer].to(ids.device)[ids.long()]
        D = self.dtab["delta" if kind == "sal" else "drel"][layer].to(ids.device)
        return w.double().pow(2) * xn.double()[:, None] * D[ids.long()]

    def _tok_matrix(self, ids, v):
        c = torch.zeros(ids.shape[0], self.NE, dtype=torch.float64, device=ids.device)
        c.scatter_add_(1, ids.long(), v)
        return c

    def _core_oracle(self, layer, ids, seq):
        dev, NE, B, H = ids.device, self.NE, self.B, self.H
        T, K = ids.shape
        N, nb = T // seq, seq // B
        assert N * seq == T and nb * B == seq
        c = self._tok_matrix(ids, self._slot_value(layer, self.oracle, ids)).view(N, seq, NE)
        cs = torch.zeros(N, seq + 1, NE, dtype=torch.float64, device=dev)
        cs[:, 1:] = c.cumsum(1)
        s0 = torch.arange(nb, device=dev) * B
        s1 = (s0 + H).clamp_max(seq)
        sc = (cs[:, s1] - cs[:, s0])                                   # [N, nb, NE] actual value in [s, s + H)
        fixed = torch.zeros(NE, dtype=torch.bool, device=dev)
        fixed[self.fixed[layer]] = True
        sc = sc.masked_fill(fixed, float("-inf"))
        top = torch.sort(-sc, dim=-1, stable=True).indices[..., :self.nf]
        want = torch.zeros(N, nb, NE, dtype=torch.bool, device=dev)
        want.scatter_(2, top, True)
        n = torch.arange(T, device=dev) // seq
        k = (torch.arange(T, device=dev) % seq) // B
        hi = (want | fixed)[n.unsqueeze(1), k.unsqueeze(1), ids.long()]
        fdef = torch.zeros(NE, dtype=torch.bool, device=dev)
        fdef[[e for e in self.fdef[layer] if e not in set(self.fixed[layer])][:self.nf]] = True
        stat = fixed | fdef
        churn = (want[:, 1:] & ~want[:, :-1]).sum(-1).double()
        d = dict(slots=T * K, l4_slots=int(hi.sum()), fixed_slots=int(fixed[ids.long()].sum()),
                 float0_slots=int(stat[ids.long()].sum()), churn_sum=float(churn.sum()), churn_n=int(churn.numel()),
                 churn_first_sum=0.0, churn_first_n=0)
        return hi, want, d

    def _core_gbdt(self, layer, ids, seq):
        """predictor=gbdt: SM120's production floating-set predictor (streaming/gbdt_predictor.py, commit 3f7dcda,
        mode next_refresh, hysteresis hm, 16-token blocks) driven token by token exactly as scheduler.Scheduler does
        with an unbounded byte budget: the level of token t's slots is fixed before its counts are seen; when
        P.step() applies a new score matrix, want = P.target(resident = current floating set) and every upgrade lands
        before the next token (downgrades too).  One predictor per sequence (chain: per run of windows).  token_ids
        None (plain-text corpora: think/answer state stays 'think')."""
        if "/home/coder/git/nestquant/streaming" not in sys.path:
            sys.path.insert(0, "/home/coder/git/nestquant/streaming")
        from gbdt_predictor import GBDTPredictor
        dev, NE = ids.device, self.NE
        T, K = ids.shape
        N = T // seq
        G = 16
        fixed = np.zeros(NE, bool); fixed[self.fixed[layer]] = True
        fdef = np.zeros(NE, bool)
        fdef[[e for e in self.fdef[layer] if e not in set(self.fixed[layer])][:self.nf]] = True
        idn = ids.long().cpu().numpy()
        serve = np.zeros((N, seq // G, NE), bool)
        if self.gscale == "cntdelta":            # predicted hits x delta_e only
            mean_sal = self.dtab["delta"][layer].numpy()[None, None].repeat(N, 0).repeat(seq // G, 1)
        elif self.gscale is not None:            # causal per-expert mean w^2|x|^2 per hit, EMA over tokens, at block ends
            ids_, w_, xn_ = self._act
            sv = self._tok_matrix(ids, w_.double().pow(2) * xn_.double()[:, None]).view(N, seq // G, G, NE)
            hv = self._tok_matrix(ids, torch.ones(ids.shape, dtype=torch.float64, device=ids.device)).view(sv.shape)
            wpos = self.sal_a ** torch.arange(G - 1, -1, -1, device=ids.device, dtype=torch.float64)
            Cs, Ch = torch.einsum("nkje,j->nke", sv, wpos), torch.einsum("nkje,j->nke", hv, wpos)
            aG = self.sal_a ** G
            for b in range(1, Cs.shape[1]):
                Cs[:, b] += Cs[:, b - 1] * aG
                Ch[:, b] += Ch[:, b - 1] * aG
            prior = Cs.sum(-1, keepdim=True) / Ch.sum(-1, keepdim=True).clamp_min(1e-30)
            mean_sal = torch.where(Ch > 0, Cs / Ch.clamp_min(1e-30), prior)          # [N, nblk, NE] through block b
            if self.gscale != "wx2":
                mean_sal = mean_sal * self.dtab["delta" if self.gscale == "sal" else "drel"][layer].to(ids.device)
            mean_sal = mean_sal.cpu().numpy()
            del sv, hv, Cs, Ch
        v2 = getattr(self, "gbdt_v2", False) or getattr(self, "gbdt_scale", None) is not None
        if v2:
            from gbdt_predictor_v2 import GBDTPredictorV2
            ids_, w_, xn_ = self._act
            assert torch.equal(ids_.long(), ids.long()), "act_full routing != scheduled routing"
            sv_ = (w_.double().pow(2) * xn_.double()[:, None]).cpu().numpy()
        churn = []
        if getattr(self, "joint", None):
            if "/home/coder/git/nestquant/threads/33-search/joint" not in sys.path:
                sys.path.insert(0, "/home/coder/git/nestquant/threads/33-search/joint")
            from joint_predictor import JointPredictor
            net = self._joint_net()
        for n in range(N):
            if getattr(self, "joint", None):
                P = JointPredictor([layer], {layer: self.fixed[layer]}, net, n_float=self.nf, hm=self.hm,
                                   device=str(dev), mode=getattr(self, "gmode", "next_refresh"),
                                   num_threads=int(os.environ.get("NQ_GBDT_THREADS", "4")))
            elif v2:
                P = GBDTPredictorV2([layer], {layer: self.fixed[layer]}, model_path=self.gbdt_model,
                                    scale=self.gbdt_scale, n_float=self.nf, hm=self.hm,
                                    mode=getattr(self, "gmode", "next_refresh"), rlo=getattr(self, "grlo", 20),
                                    rhi=getattr(self, "grhi", 121),
                                    num_threads=int(os.environ.get("NQ_GBDT_THREADS", "4")))
            else:
                P = GBDTPredictor([layer], {layer: self.fixed[layer]}, model_path=getattr(self, "gbdt_model", None),
                              n_float=self.nf, hm=self.hm, mode=getattr(self, "gmode", "next_refresh"),
                              rlo=getattr(self, "grlo", 20), rhi=getattr(self, "grhi", 121),
                              num_threads=int(os.environ.get("NQ_GBDT_THREADS", "4")))
            want = fdef.copy()
            try:
                for t in range(seq):
                    if t % G == 0:
                        serve[n, t // G] = want
                    c = np.bincount(idn[n * seq + t], minlength=NE).astype(np.float64)[None]
                    kw = {"sal": np.bincount(idn[n * seq + t], weights=sv_[n * seq + t], minlength=NE)[None]} \
                        if v2 else {}
                    if P.step(c, 1, None, t == 0, **kw):
                        if self.gscale is None:
                            w = P.target(want[None])
                        else:
                            w = self._gbdt_scaled_target(P, want, mean_sal[n, t // G], fixed)
                        if w is not None:
                            nw = w[0] & ~fixed
                            churn.append(int((nw & ~want).sum()))
                            want = nw
                        assert t % G == G - 1
            finally:
                P.close()
        serve_t = torch.from_numpy(serve).to(dev)
        hi_e = serve_t | torch.from_numpy(fixed).to(dev)
        nn = torch.arange(T, device=dev) // seq
        kk = (torch.arange(T, device=dev) % seq) // G
        hi = hi_e[nn.unsqueeze(1), kk.unsqueeze(1), ids.long()]
        stat = torch.from_numpy(fixed | fdef).to(dev)
        fx = torch.from_numpy(fixed).to(dev)
        d = dict(slots=T * K, l4_slots=int(hi.sum()), fixed_slots=int(fx[ids.long()].sum()),
                 float0_slots=int(stat[ids.long()].sum()), churn_sum=float(sum(churn)), churn_n=len(churn),
                 churn_first_sum=0.0, churn_first_n=0)
        return hi, serve_t, d

    def _split_by_map(self, runs, g):
        """chain=map: split each corpus run (one per corpus in rank order, nq_e2e.WIN_IDS) at chain/task changes."""
        M = sys.modules["__main__"]
        out = []
        for w0, w1 in runs:
            name, mine = M.WIN_IDS[g[w0]]
            assert len(mine) == w1 - w0, (name, len(mine), w0, w1)
            f = f"{M.CORPUS_DIR}/{name}.map.npz"
            if not os.path.exists(f):
                out.append((w0, w1)); continue
            z = np.load(f)
            key = np.asarray(z["chain"] if "chain" in z else z["task"])[list(mine)]
            a = w0
            for j in range(1, len(mine) + 1):
                if j == len(mine) or key[j] != key[j - 1]:
                    out.append((a, w0 + j)); a = w0 + j
        return out

    def _joint_net(self):
        """joint arm: net path for the current run of windows (joint_map: by the run's task)."""
        if self.jmap is None:
            return self.joint
        wins = [(n, g) for n, mine in sys.modules["__main__"].WIN_IDS for g in mine]   # local window -> (corpus, id)
        w0, w1 = self._run
        tasks = {self.jmap["task"][n][g] if n in self.jmap["task"] else None for n, g in wins[w0:w1]}
        assert len(tasks) == 1, f"chain spans tasks {tasks}"
        t = tasks.pop()
        p_ = self.jmap["by"].get(t, self.jmap["default"]) if t is not None else self.jmap["default"]
        assert p_ is not None, f"no joint net for task {t}"
        self.jlog = getattr(self, "jlog", {}); self.jlog[(w0, w1)] = (t, p_)
        return p_

    def _gbdt_scaled_target(self, P, resident, ms, fixed):
        """P.target with the applied score matrix rescaled: predicted hits x delta_e x mean sal per hit."""
        if P.S is None:
            return None
        S = P.S[0].astype(np.float64)
        forced = S >= 1e3
        hits = np.where(forced, 64.0 * (S - 1e3), S)
        v = hits * ms
        if self.gkeep:
            v = np.where(forced, 1e30 + v, v)
        v = np.where(fixed, -np.inf, v)
        r = resident & ~fixed
        v = np.where(r, v * (1 + P.hm), v)
        if np.where(fixed, 0, np.maximum(S, 0)).sum() <= 0:
            return r[None]
        want = np.zeros(self.NE, bool)
        want[np.argsort(-v, kind="stable")[:self.nf]] = True
        return want[None]

    @torch.no_grad()
    def _core(self, layer, ids, seq):
        """ids [T,8] (T = n*seq, sequences back to back) -> (hi [T,8] bool per routed slot, serve [n, nchunks, NE]
        bool floating set serving each chunk, diagnostics)."""
        if self.predictor == "gbdt":
            return self._core_gbdt(layer, ids, seq)
        if self.oracle is not None:
            return self._core_oracle(layer, ids, seq)
        dev, NE, R = ids.device, self.NE, self.R
        T, K = ids.shape
        N, nc = T // seq, seq // R
        assert N * seq == T and nc * R == seq
        c = self._tok_matrix(ids, self._slot_value(layer, self.score, ids)).view(N, nc, R, NE)
        wpos = self.a ** torch.arange(R - 1, -1, -1, device=dev, dtype=torch.float64)
        C = torch.einsum("nkje,j->nke", c, wpos)
        aR = self.a ** R
        S = torch.zeros(N, nc + 1, NE, dtype=torch.float64, device=dev)
        for k in range(nc):
            S[:, k + 1] = S[:, k] * aR + C[:, k]                      # score at tok = R*(k+1)
        fixed = torch.zeros(NE, dtype=torch.bool, device=dev)
        fixed[self.fixed[layer]] = True
        fdef = torch.zeros(NE, dtype=torch.bool, device=dev)
        fdef[[e for e in self.fdef[layer] if e not in set(self.fixed[layer])][:self.nf]] = True
        want = torch.zeros(N, nc, NE, dtype=torch.bool, device=dev)   # want[:, r] = set chosen at refresh r
        want[:, 0] = fdef
        sc = S[:, 1:nc].masked_fill(fixed, float("-inf"))
        top = torch.sort(-sc, dim=-1, stable=True).indices[..., :self.nf]
        w = torch.zeros(N, nc - 1, NE, dtype=torch.bool, device=dev)
        w.scatter_(2, top, True)
        has = S[:, 1:nc].sum(-1, keepdim=True) > 0                     # no counts yet -> keep the previous want
        for r in range(1, nc):
            want[:, r] = torch.where(has[:, r - 1], w[:, r - 1], want[:, r - 1])
        if self.lag == 1:                                              # chunk k served by want[k - 1]
            serve = torch.empty_like(want)
            serve[:, :1] = fdef
            serve[:, 1:] = want[:, :-1]
        else:                                                          # lag 0: want[k] = EMA through chunk k-1
            serve = want.clone()
        n = torch.arange(T, device=dev) // seq
        k = (torch.arange(T, device=dev) % seq) // R
        upg = None
        if self.ahead is not None:                                     # per-chunk upgrades from lookahead / oracle
            fbl = self.fb > 0 and layer < min(self.fixed) + self.fb
            li_, lw, lx = self._act if fbl else self._la
            li_ = li_.long()
            v = self._slot_value(layer, self.rank, li_, lw, lx)
            sc = torch.zeros(N * nc, NE, dtype=torch.float64, device=dev)
            sc.index_put_(((n * nc + k)[:, None].expand_as(li_), li_), v, accumulate=True)
            sc = sc.view(N, nc, NE).masked_fill(fixed, 0.0)
            if fbl:                                                    # previous chunk's routing
                sc = torch.cat([torch.zeros_like(sc[:, :1]), sc[:, :-1]], 1)
            top = torch.sort(-sc, dim=-1, stable=True).indices[..., :self.up]
            U = torch.zeros_like(serve)
            U.scatter_(2, top, True)
            U &= sc > 0                                                # never upgrade an expert with no score
            upg = (U & ~serve).sum(-1)                                 # upgrades beyond the resident set per chunk
            serve = serve | U
        hi_e = serve | fixed                                           # [N, nc, NE]
        hi = hi_e[n.unsqueeze(1), k.unsqueeze(1), ids.long()]
        # diagnostics (on this stream's routing)
        stat = fixed | fdef
        churn = (want[:, 1:] & ~want[:, :-1]).sum(-1).double()        # experts entering the set per refresh
        hist = lambda t: {int(a): int(b) for a, b in zip(*torch.unique(t.long(), return_counts=True))}  # noqa: E731
        d = dict(slots=T * K, l4_slots=int(hi.sum()), fixed_slots=int(fixed[ids.long()].sum()),
                 float0_slots=int(stat[ids.long()].sum()), churn_sum=float(churn.sum()), churn_n=int(churn.numel()),
                 churn_first_sum=float(churn[:, 0].sum()), churn_first_n=int(churn[:, 0].numel()),
                 churn_hist=hist(churn.flatten()) if churn.numel() else {})
        if upg is not None:
            d.update(upg_hist=hist(upg.flatten()), upg_sum=float(upg.sum()), upg_n=int(upg.numel()))
        return hi, serve, d

    def level_mask(self, layer, ids, seq, groups=None):
        return self.schedule(layer, ids, seq, groups)[0]

    def expert_level(self, layer, expert, ref, level):
        W = (self.hi if level == 4 else self.lo).expert(layer, expert, ref)
        if W is None and level == 4 and self.hi2 is not None:
            W = self.hi2.expert(layer, expert, ref)
        if W is not None:
            if level == 4:
                self.n_hi += 1
            else:
                self.n_lo += 1
        return W

    def expert(self, layer, expert, ref):
        raise RuntimeError("Adapt needs per-token levels (moe_multi level_mask path)")


def dir_sha(d, layer):
    """sha256 over (name, sha256(file)) of every artifact E{E}.pt this layer would read from d."""
    import hashlib
    h = hashlib.sha256()
    for e in range(256):
        f = nq_artifact_path(d, layer, e)
        if f is None:
            continue
        fh = hashlib.sha256()
        with open(f, "rb") as x:
            for b in iter(lambda: x.read(1 << 24), b""):
                fh.update(b)
        h.update(f"{os.path.basename(f)}:{fh.hexdigest()}\n".encode())
    return h.hexdigest()


class Override(Base):
    """Wraps a stream: for layers in `dirs`, routed experts come from DIR's E{E}.pt artifacts (tuned re-encodes),
    decoded with the batched nq_fastdec decoder at the level the wrapped stream would use (inner.level_of, e.g. Mix),
    rounded to fp16 like the predecode.  Other layers pass through to the wrapped stream."""

    def __init__(self, inner, dirs, batch=4):
        self.inner, self.dirs, self.batch = inner, dirs, batch
        self.name, self.spec = inner.name, inner.spec + f" +override{sorted(dirs)}"
        self.only_layers = getattr(inner, "only_layers", None)
        self.n_override = 0
        self.cache = {}
        if not hasattr(inner, "level_of"):
            raise SystemExit(f"--expert-override: stream {inner.name} has no per-expert level (use a mix: stream)")

    def __getattr__(self, k):                    # n_hi / n_lo / active ... of the wrapped stream
        return getattr(self.__dict__["inner"], k)

    def begin_layer(self, layer, device):
        self.dev = device
        self.inner.begin_layer(layer, device)
        self.cache = {}

    def end_layer(self, layer):
        self.cache = {}
        self.inner.end_layer(layer)

    def expert(self, layer, expert, ref):
        if layer not in self.dirs:
            return self.inner.expert(layer, expert, ref)
        if expert not in self.cache:
            import nq_fastdec as F
            self.cache = {}
            chunk = [e for e in range(expert, min(expert + self.batch, 256))]
            arts = {}
            for e in chunk:
                f = nq_artifact_path(self.dirs[layer], layer, e)
                if f is None:
                    raise SystemExit(f"override: missing E{e} for L{layer} under {self.dirs[layer]}")
                arts[e] = torch.load(f, map_location="cpu", weights_only=False)
            for lv in (2, 4):
                es = [e for e in chunk if self.inner.level_of(layer, e) == lv]
                if not es:
                    continue
                perms = [arts[e].get("meta", {}).get("inter_perm") for e in es]
                out = F.decode_experts([arts[e] for e in es], (lv,), self.dev, batch_had=True, perms=perms)[lv]
                for e, W in zip(es, out):
                    self.cache[e] = {pn: w.half() for pn, w in zip(PROJ, W)}
            if self.inner.level_of(layer, expert) == 4:
                self.inner.n_hi += 1
            else:
                self.inner.n_lo += 1
        else:
            if self.inner.level_of(layer, expert) == 4:
                self.inner.n_hi += 1
            else:
                self.inner.n_lo += 1
        self.n_override += 1
        return self.cache[expert]


def set_from_range(s):
    if not s:
        return set()
    a, _, b = s.partition("-")
    return set(range(int(a), int(b or a) + 1))


class NestQuant(Base):
    """Rotated-basis reconstructions are shared across levels: with nq2 and nq4 streams in one pass the
    expensive nq_decode.rotated_levels runs once per expert (0.37 s), each extra level costs ~0.02 s."""
    _shared = {}                       # class-level: {"key": (file, dev), "art": ..., "rot": {proj: rot}}

    def __init__(self, root, level=4):
        self.root, self.level = root, int(level)
        if NQ12 not in sys.path:
            sys.path.insert(0, NQ12)
        import nq_decode
        self.nqd = nq_decode

    def expert(self, layer, expert, ref):
        if not self.active(layer):
            return None
        f = nq_artifact_path(self.root, layer, expert)
        if f is None:
            return None
        c = NestQuant._shared
        if c.get("key") != (f, str(self.dev)):
            c.clear()
            art = torch.load(f, map_location="cpu", weights_only=False)
            c.update(key=(f, str(self.dev)), art=art,
                     rot={p: self.nqd.rotated_levels(art[p], self.dev) for p in ("gate", "up", "down")})
        art, rot = c["art"], c["rot"]
        W = [self.nqd.decode_matrix(art[p], self.level, self.dev, rot=rot[p]) for p in ("gate", "up", "down")]
        perm = art.get("meta", {}).get("inter_perm")          # same un-permute as nq_decode.decode_expert
        if perm is not None:
            inv = torch.argsort(torch.as_tensor(perm, device=self.dev))
            W = [W[0][inv], W[1][inv], W[2][:, inv]]
        return {"gate_proj": W[0], "up_proj": W[1], "down_proj": W[2]}

    def end_layer(self, layer):
        NestQuant._shared.clear()


class EXL3(Base):
    def __init__(self, root, bits=4):
        self.root, self.bits = root, int(bits)
        if ORBIT not in sys.path:
            sys.path.insert(0, ORBIT)
        # exllamav3_ext needs libcudart.so.12 (thread 05 ships one)
        import ctypes
        lib = "/home/coder/git/nestquant/threads/05-exl3-harness/lib/libcudart.so.12"
        if os.path.exists(lib):
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
        from orbit_duet.exl3_adapter import EXL3Expert
        self.cls = EXL3Expert

    def expert(self, layer, expert, ref):
        if not self.active(layer):
            return None
        f = f"{self.root}/layer_{layer:03d}/expert_{expert:03d}/expert_{self.bits}.bin"
        if not os.path.exists(f):
            return None
        g, u, d = self.cls(f).decoded_weights()
        return {"gate_proj": g, "up_proj": u, "down_proj": d}


class NVFP4(Base):
    def __init__(self, root):
        self.root = root
        if ORBIT not in sys.path:
            sys.path.insert(0, ORBIT)
        from orbit_duet.nvfp4_reference import decode
        self.decode = decode

    def expert(self, layer, expert, ref):
        if not self.active(layer):
            return None
        f = f"{self.root}/layer_{layer:03d}/expert_{expert:03d}/weights.pt"
        if not os.path.exists(f):
            return None
        pl = torch.load(f, weights_only=True, map_location="cpu")
        ws = [self.decode({k: v.to(self.dev) for k, v in q.items()}) for q in pl["weights"]]
        return dict(zip(PROJ, ws))


def make(name, spec):
    kind, _, rest = spec.partition(":")
    if kind == "py":
        path, _, tail = rest.partition(":")
        cls, _, args = tail.partition(":")
        kv = _kv(args)
        only = _layers(kv)
        if path.endswith(".py"):
            sp = importlib.util.spec_from_file_location(f"nqplug_{name}", path)
            mod = importlib.util.module_from_spec(sp)
            sp.loader.exec_module(mod)
        else:
            mod = importlib.import_module(path)
        q = getattr(mod, cls)(**kv)
    else:
        kv = _kv(rest) if kind != "dir" else {}
        if kind == "dir":
            path, _, tail = rest.partition(",")
            kv = _kv(tail)
        only = _layers(kv)
        route = kv.pop("route", None)            # oracle | oracle_ids: FP8 reference routing (nq_e2e.moe_multi)
        if kind == "ref":
            q = Ref()
        elif kind == "rtn":
            q = RTN(**kv)
        elif kind == "dir":
            q = Dir(path)
        elif kind == "mix":
            q = Mix(**kv)
        elif kind == "adapt":
            q = Adapt(**kv)
        elif kind == "noise":
            q = Noise(**kv)
        elif kind == "nestquant":
            q = NestQuant(**kv)
        elif kind == "exl3":
            q = EXL3(**kv)
        elif kind == "nvfp4":
            q = NVFP4(**kv)
        else:
            raise SystemExit(f"unknown quantiser kind {kind!r}")
    q.name = name
    q.spec = spec
    q.only_layers = only
    q.route_mode = locals().get("route")
    for m in ("begin_layer", "end_layer"):
        if not hasattr(q, m):
            setattr(q, m, (lambda *a, **k: None))
    return q
