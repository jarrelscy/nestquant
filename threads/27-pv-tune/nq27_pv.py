"""T27 band pilot arm (b), the V step: re-encode (re-search codes) of one layer with Hessians corrected to the
PROPAGATED inputs, through the pinned campaign encoder path (T23 nq_layer_batch / nq_encode_batch.encode_group,
imported read-only, same production config, same text/vision blend as nq-encode-v1).

  python nq27_pv.py --layer L --arm NAME [--dump DIR] [--no-prop] [--w W --nw NW] [--experts a:b] [--group 2]

H (per expert, text side; the vision side and G are the campaign's):
  H_text' = nt(H_T19) + nt(H^_nqdef) - nt(H^_fp8), PSD-projected (eigen clamp at 1e-6 mean eig)
where H^_s is thread 08's recipe (nq19.recipe_H, alpha 0.25, ctx_mass 0.25) evaluated on the T18 dump of stream s
(x_s, ids_s, p_s; context = a fixed uniform 25 % token sample, n_ctx / T_fit = 0.25 as in T19) -- a control-variate
estimate of the Hessian under the nqdef-propagated input distribution that keeps T19's 15M-token sample for the
bulk and only takes the fp8 -> nqdef shift from the 262k dump tokens.  Down: the same on
h = bf16 silu(x g) * (x u) of the FP8 teacher (T19 arithmetic).
--no-prop: campaign H unchanged (sanity: must reproduce nq-encode-v1 bit-exactly).
Out: OUT/band/{arm}/L{L}/experts/E{E}.pt (+ .json: time, relative H shift, bit-identity vs nq-encode-v1).
"""
import os, sys, json, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "8")
T23 = "/home/coder/git/nestquant/threads/23-encode-throughput"
sys.path.insert(0, T23)
import common as C                        # T12 / T19 import paths (as the campaign's t23b encoder)
import torch
import torch.nn.functional as F
import nq_encode as NE
import nq_layer as NL
import nq_bnd as NBND
import nq_encode_batch as NBAT
import nq19
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nq27_band as B

ENC = "/tmp/nestquant/nq-encode-v1"
OUT = "/tmp/nestquant/27-pv-tune"


class PropText:
    """T19 text capture with glm_H's H replaced by the propagated control-variate estimate (G, meta kept)."""

    def __init__(self, text, delta):
        self.text, self.delta = text, delta

    def __getattr__(self, k):
        return getattr(self.__dict__["text"], k)

    def glm_H(self, L, E, device="cuda", **kw):
        ht = self.text.glm_H(L, E, device=device, **kw)
        dH, info = self.delta(E)
        if dH is None:
            return ht
        H = []
        for Hm, d in zip(ht["H"], dH):
            Hn = nq19.nt(Hm.double()) + d.to(Hm.device)
            ev, U = torch.linalg.eigh(Hn)
            floor = 1e-6 * float(ev.clamp_min(0).mean())
            info.setdefault("neg_eigs", []).append(int((ev < floor).sum()))
            Hn = (U * ev.clamp_min(floor)) @ U.T
            H.append(((Hn + Hn.T) / 2).float())
        return dict(ht, H=H, meta=dict(ht["meta"], prop=info))


def recipe(x, p, ctx, a_fn):
    """thread-08 recipe (alpha .25, ctx_mass .25) for gate/up (x) and down (h = a_fn(x)) -> (nt Hx, nt Ha) fp64."""
    out = []
    for f in (lambda z: z, a_fn):
        z = f(x).float(); zc = f(ctx).float()
        A2 = (z * p[:, None].square()).T @ z; A0 = z.T @ z; Cc = zc.T @ zc
        cp2 = 0.25 * float(p.double().square().sum()) / len(ctx)
        out.append(nq19.nt(nq19.recipe_H(A2, A0, Cc, cp2, 0.25).double()))
        del z, zc, A2, A0, Cc
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True); ap.add_argument("--arm", required=True)
    ap.add_argument("--dump", default=B.DUMP); ap.add_argument("--no-prop", action="store_true")
    ap.add_argument("--w", type=int, default=0); ap.add_argument("--nw", type=int, default=1)
    ap.add_argument("--experts", default="0:256"); ap.add_argument("--group", type=int, default=2)
    ap.add_argument("--gpu-gb", type=float, default=10.5); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    C.setup(cap_gb=a.gpu_gb)
    L = a.layer
    cap = NL.open_stats(f"{ENC}/_stats", f"{ENC}/_stats_mm", 0.25)           # the campaign's blend (nq-encode-v1)
    from orbit_duet.source import weights
    if not a.no_prop:
        S = {}
        for s in ("fp8", "nqdef"):
            S[s] = dict(x=B.load(a.dump, L, f"x_{s}"), ids=B.load(a.dump, L, f"ids_{s}"), p=B.load(a.dump, L, f"p_{s}"))
        g = torch.Generator().manual_seed(2791 + L)
        ci = torch.randperm(len(S["fp8"]["x"]), generator=g)[:len(S["fp8"]["x"]) // 4]      # 25 % context sample

        def delta(E):
            tb = [w.to("cuda").bfloat16() for w in weights(B.SRC, L, E)]
            a_fn = lambda z: F.silu(F.linear(z.bfloat16(), tb[0])) * F.linear(z.bfloat16(), tb[1])
            R, info = {}, {}
            for s, d in S.items():
                rows, sl = torch.where(d["ids"] == E)
                info[f"n_{s}"] = len(rows)
                if len(rows) < 16:
                    return None, dict(info, skipped="too few routed dump rows")
                R[s] = recipe(d["x"][rows].cuda(), d["p"][rows, sl].cuda(), d["x"][ci].cuda(), a_fn)
            dH = [R["nqdef"][0] - R["fp8"][0], R["nqdef"][0] - R["fp8"][0], R["nqdef"][1] - R["fp8"][1]]
            info["rel_shift"] = [float(v.norm() / R["fp8"][i].norm()) for i, v in zip((0, 0, 1), dH)]
            return dH, info
        cap.text = PropText(cap.text, delta)
    od = f"{OUT}/band/{a.arm}/L{L}/experts"; os.makedirs(od, exist_ok=True)
    e0, e1 = map(int, a.experts.split(":"))
    todo = [E for E in range(e0, e1) if E % a.nw == a.w and (a.force or not os.path.exists(f"{od}/E{E}.json"))]
    lrc = dict(NE.LR, tau=NE.LR["tau"], rmax=NE.LR["rmax"])
    bw = NBND.parse_bnd(str(NBND.DEFAULT_BND))
    seg = NBAT.Seg(True)
    with NBAT.patched(NBAT.DEFAULT_OPTS):
        for i in range(0, len(todo), a.group):
            g = todo[i:i + a.group]
            t0 = time.time()
            data = []
            for E in g:
                HG, flags = NL.expert_HG(cap, L, E, str(NBND.DEFAULT_BND), NBND.DEFAULT_K, NBND.DEFAULT_CAP)
                data.append((NL.teacher(B.SRC, L, E), HG, flags))
            th = time.time() - t0
            arts = NBAT.encode_group([(W, HG) for W, HG, _ in data], rate=None, res_K=None, seg=seg, lr=lrc)
            te = time.time() - t0 - th
            for E, (_, HG, flags), art in zip(g, data, arts):
                hm = HG.get("meta", {})
                art["meta"].update(layer=L, expert=E, flags=flags, bnd=NBND.bnd_tag(bw), bnd_k=None, bnd_cap=None,
                                   stats=f"{ENC}/_stats", stats_mm=f"{ENC}/_stats_mm", mm_w=0.25,
                                   hg_meta={k: (float(v) if torch.is_tensor(v) else v) for k, v in hm.items()
                                            if not isinstance(v, dict)})
                p = f"{od}/E{E}.pt"
                torch.save(art, p + ".tmp"); os.replace(p + ".tmp", p)
                ref = torch.load(f"{ENC}/L{L}/experts/E{E}.pt", weights_only=False, map_location="cpu")
                same = {pn: all(torch.equal(u, v) for u, v in zip(_flat(art[pn]), _flat(ref[pn]))) for pn in NE.PROJ}
                rc = dict(layer=L, expert=E, arm=a.arm, prop=not a.no_prop, prop_info=hm.get("prop"), flags=flags,
                          bitidentical_vs_campaign=same, h_s=th / len(g), enc_s=te / len(g), dump=a.dump)
                json.dump(rc, open(f"{od}/E{E}.json", "w"), indent=1)
                print(f"[L{L} E{E}] prop {not a.no_prop} {hm.get('prop')} same-as-campaign {same} "
                      f"H {th/len(g):.1f}s enc {te/len(g):.1f}s", flush=True)
            del data, arts
            torch.cuda.empty_cache()


def _flat(x):
    if isinstance(x, dict):
        for k in sorted(x):
            if k != "info":
                yield from _flat(x[k])
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _flat(v)
    elif torch.is_tensor(x):
        yield x


if __name__ == "__main__":
    main()
