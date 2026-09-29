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

    def __init__(self, lo, hi, manifest, hi2=None, half_life=512, refresh=64, n_float=51, lag=1, chain=0, NE=256):
        self.chain = int(chain)          # 1: carry scores + floating set across consecutive windows of one corpus
        self.lo, self.hi = Dir(lo), Dir(hi)
        self.hi2 = Dir(hi2) if hi2 else None
        m = json.load(open(manifest))
        self.fixed = {int(L): sorted(map(int, v)) for L, v in m["default_allocation"].items()}
        self.fdef = {int(L): [int(e) for e in v] for L, v in m["floating_default"].items()}
        self.a = 0.5 ** (1 / float(half_life))
        self.R, self.nf, self.lag, self.NE = int(refresh), int(n_float), int(lag), NE
        assert self.lag == 1, "only the one-refresh lag is implemented"
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
        if not self.chain:
            runs = [(0, N)]
            L = seq
        else:
            g = list(groups) if groups is not None else [0] * N
            runs, w0 = [], 0
            for w in range(1, N + 1):
                if w == N or g[w] != g[w0]:
                    runs.append((w0, w)); w0 = w
        hi, serves, tot = [], [], {}
        for w0, w1 in runs:
            h, sv, st = self._core(layer, ids[w0 * seq:w1 * seq], seq if not self.chain else (w1 - w0) * seq)
            hi.append(h); serves.append(sv)
            for k, v in st.items():
                tot[k] = tot.get(k, 0) + v
        tot["chains"] = [w1 - w0 for w0, w1 in runs] if self.chain else None
        self.diag[layer] = tot
        return torch.cat(hi), serves

    @torch.no_grad()
    def _core(self, layer, ids, seq):
        """ids [T,8] (T = n*seq, sequences back to back) -> (hi [T,8] bool per routed slot, serve [n, nchunks, NE]
        bool floating set serving each chunk, diagnostics)."""
        dev, NE, R = ids.device, self.NE, self.R
        T, K = ids.shape
        N, nc = T // seq, seq // R
        assert N * seq == T and nc * R == seq
        c = torch.zeros(T, NE, dtype=torch.float64, device=dev)
        c.scatter_add_(1, ids.long(), torch.ones(T, K, dtype=torch.float64, device=dev))
        c = c.view(N, nc, R, NE)
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
        serve = torch.empty_like(want)                                 # chunk k served by want[k - lag]
        serve[:, :1] = fdef
        serve[:, 1:] = want[:, :-1]
        hi_e = serve | fixed                                           # [N, nc, NE]
        n = torch.arange(T, device=dev) // seq
        k = (torch.arange(T, device=dev) % seq) // R
        hi = hi_e[n.unsqueeze(1), k.unsqueeze(1), ids.long()]
        # diagnostics (on this stream's routing)
        stat = fixed | fdef
        churn = (want[:, 1:] & ~want[:, :-1]).sum(-1).double()        # experts entering the set per refresh
        d = dict(slots=T * K, l4_slots=int(hi.sum()), fixed_slots=int(fixed[ids.long()].sum()),
                 float0_slots=int(stat[ids.long()].sum()), churn_sum=float(churn.sum()), churn_n=int(churn.numel()),
                 churn_first_sum=float(churn[:, 0].sum()), churn_first_n=int(churn[:, 0].numel()))
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
    for m in ("begin_layer", "end_layer"):
        if not hasattr(q, m):
            setattr(q, m, (lambda *a, **k: None))
    return q
