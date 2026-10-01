"""T35 early check: AdaptEmu with REAL b175 artifacts on the finished layers.

AdaptReal = emu_adapt.AdaptEmu (jF floating set, per-token levels; cold = W_ref + s (W_nq2 - W_ref), hot = real v1 L4)
except on layers rl: there BOTH levels come from the campaign's own E{E}.pt (pattern-rate base, nq15 decoder), cold =
its L2 decode, hot = its L4 decode, rounded to the predecode dtype.  The level decision (Adapt / jF) is unchanged;
Adapt's own lo/hi lookup still runs first, so counters and "no artifact -> FP8" semantics are identical.
  --cand NAME=py:/home/coder/git/nestquant/threads/35-nq15/real_adapt.py:AdaptReal:real=ROOT,rl=3-10+40-50,s=...,<Adapt args>
rl: '+'-separated layer ranges ("," separates the plug-in args).  Logs per real layer the cold-error ratio
||W_real2 - W_ref|| / ||W_emu2 - W_ref|| over the first experts it decodes (sanity: emulation vs real magnitude).
"""
import os, sys
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/home/coder/git/nestquant/threads/12-reference-encoder")
from emu_adapt import AdaptEmu   # noqa: E402

PROJ = ("gate_proj", "up_proj", "down_proj")


def parse_layers(s):
    out = set()
    for part in (s or "").split("+"):
        if part:
            a, _, b = part.partition("-")
            out |= set(range(int(a), int(b or a) + 1))
    return out


class AdaptReal(AdaptEmu):
    def __init__(self, real=None, rl="", **kw):
        super().__init__(**kw)
        import nq15                                   # noqa: F401  base-K decoder over nq_decode
        import nq_decode as D
        self.D, self.real, self.rl = D, real, parse_layers(rl)
        self.c_key, self.c_rot, self.c_art = None, None, None
        self.n_real = 0
        self.ratio = {}
        for L in sorted(self.rl):
            miss = [e for e in range(256) if not os.path.exists(f"{real}/L{L}/experts/E{e}.pt")]
            if miss:
                raise SystemExit(f"AdaptReal: L{L} missing {len(miss)} experts under {real}")
        print(f"AdaptReal real={real} layers={sorted(self.rl)} s={self.s}", flush=True)

    def _real(self, layer, expert, level, dev):
        if self.c_key != (layer, expert, str(dev)):
            art = torch.load(f"{self.real}/L{layer}/experts/E{expert}.pt", map_location="cpu", weights_only=False)
            self.c_art = art
            self.c_rot = {p: self.D.rotated_levels(art[p], dev) for p in ("gate", "up", "down")}
            self.c_key = (layer, expert, str(dev))
        art = self.c_art
        import nq35_t29 as T29                        # T29 layers: down decodes with the Had512 k side (meta)
        W = [self.D.decode_matrix(art[p], level, dev, rot=self.c_rot[p]) for p in ("gate", "up")]
        with T29.down_scope(art):
            W.append(self.D.decode_matrix(art["down"], level, dev, rot=self.c_rot["down"]))
        perm = art.get("meta", {}).get("inter_perm")
        if perm is not None:
            inv = torch.argsort(torch.as_tensor(perm, device=dev))
            W = [W[0][inv], W[1][inv], W[2][:, inv]]
        return dict(zip(PROJ, W))

    def expert_level(self, layer, expert, ref, level):
        if layer not in self.rl:
            return super().expert_level(layer, expert, ref, level)
        W0 = super(AdaptEmu, self).expert_level(layer, expert, ref, level)       # Adapt: counters / None semantics
        if W0 is None:
            return None
        dev = next(iter(W0.values())).device
        Wr = self._real(layer, expert, level, dev)
        out = {}
        for p, w in W0.items():
            if Wr[p].shape != w.shape:
                raise SystemExit(f"AdaptReal L{layer} E{expert} {p}: shape {tuple(Wr[p].shape)} vs {tuple(w.shape)}")
            out[p] = Wr[p].to(w.dtype)
        if level != 4 and len(self.ratio.setdefault(layer, [])) < 8:
            R = ref()
            num = sum(float((out[p].float() - R[p].float()).norm() ** 2) for p in out)
            den = sum(float((self.s * (W0[p].float() - R[p].float())).norm() ** 2) for p in out)
            self.ratio[layer].append((num / max(den, 1e-30)) ** 0.5)
            if len(self.ratio[layer]) == 8:
                r = self.ratio[layer]
                print(f"AdaptReal L{layer} cold |real-ref|/|emu-ref| mean {sum(r) / len(r):.4f} "
                      f"[{min(r):.4f},{max(r):.4f}] (8 experts)", flush=True)
        self.n_real += 1
        return out
