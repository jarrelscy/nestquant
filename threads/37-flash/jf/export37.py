#!/usr/bin/env python3
"""T37 jF serve export -> $J37_REL (default /tmp/nestquant/37-flash/release/serving/predictor), mirroring the GLM-5.3
HF layout serving/predictor/{predictor.json, joint/...}.  Ships ONLY weights / configs / code / aggregate eval numbers:
no traces, token ids, routing, features, per-window scores (whitelist + checked at the end).
  export37.py NAME [--rel DIR] [--hm 0.7]"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time

import torch

import jlib37 as J

HERE = os.path.dirname(os.path.abspath(__file__))
# repo file -> release name (import names rewritten to the release layout)
CODE = {"gpu_predictor37.py": "gpu_predictor.py", "joint_predictor37.py": "joint_predictor.py",
        "train37.py": "train.py", "jlib37.py": "jlib.py", "parity37.py": "parity_stream.py"}
REN = [(r"\btrain37\b", "train"), (r"\bjlib37\b", "jlib"), (r"\bjoint_predictor37\b", "joint_predictor"),
       (r"\bgpu_predictor37\b", "gpu_predictor"), (r"\bparity37\b", "parity_stream")]
ALLOWED = {"jF.pt", "jF.json", "v2_sal_tweedie1.5.txt", "README.md", "SERVE.md"} | set(CODE.values())


def sha(f):
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rng(Ls):
    return f"{min(Ls)}-{max(Ls)}" if list(Ls) == list(range(min(Ls), max(Ls) + 1)) else ",".join(map(str, Ls))


SELFTEST = r"""
import json, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import gpu_predictor as GP, joint_predictor as JP
Ls = json.loads(sys.argv[2]); fixed = {int(k): v for k, v in json.loads(sys.argv[3]).items()}
kw = dict(hm=0.7, device="cpu", v2_model=sys.argv[1] + "/v2_sal_tweedie1.5.txt")
g = GP.GPUJointPredictor(Ls, fixed, sys.argv[1] + "/jF.pt", **kw); j = JP.JointPredictor(Ls, fixed, sys.argv[1] + "/jF.pt", **kw)
rng = np.random.default_rng(0); NE = g.NE; res = None; worst = 0.0; nref = 0
for t in range(96):
    c = np.zeros((len(Ls), NE), np.float32)
    for i in range(len(Ls)):
        c[i, rng.choice(NE, 8, replace=False, p=np.r_[np.full(32, 8.0), np.ones(NE - 32)] / (256 + NE - 32))] = 1
    sal = c * rng.gamma(2.0, 0.5, c.shape)
    tok = [154842] if t == 40 else [1]
    a, b = g.step(c, 1, tok, t == 0, sal), j.step(c, 1, tok, t == 0, sal)
    assert a == b
    if a:
        nref += 1; worst = max(worst, float(np.abs(np.log(g.S) - np.log(j.S)).max()))
        w = g.target(res if res is not None else np.zeros((len(Ls), NE), bool)); res = w
        assert (w.sum(1) == g.nf).all() and not (w & g.fixed).any()
assert nref == 6 and worst < 0.05, (nref, worst)
print(f"[export37] release self-test OK: {nref} refreshes, torch vs numpy max|dlog S| {worst:.2g}, NE {NE} nf {g.nf}")
"""


def selftest(jd, Ls):
    import subprocess
    fixed, _ = J.serve_sets()
    subprocess.run([sys.executable, "-I", "-B", "-c", SELFTEST, jd, json.dumps(list(Ls)),
                    json.dumps({str(L): fixed[L] for L in Ls})], check=True, cwd="/tmp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", default="jF")
    ap.add_argument("--rel", default=os.environ.get("J37_REL", "/tmp/nestquant/37-flash/release/serving/predictor"))
    ap.add_argument("--hm", type=float, default=0.7)
    a = ap.parse_args()
    jd = f"{a.rel}/joint"
    os.makedirs(jd, exist_ok=True)
    src_pt = f"{J.OUT}/models/{a.name}.pt"
    ck = torch.load(src_pt, map_location="cpu", weights_only=False)
    assert ck["ne"] == J.NE and ck["nf"] == J.NF, (ck["ne"], ck["nf"])
    # re-save with an explicit, minimal key set (weights + settings only)
    keep = ("state", "args", "K", "nf", "ne", "layers", "n_fixed", "fixed_set_sha256", "inputs", "extra")
    torch.save({k: ck[k] for k in keep}, f"{jd}/jF.pt")
    shutil.copyfile(J.V2, f"{jd}/v2_sal_tweedie1.5.txt")
    for s, d in CODE.items():
        t = open(f"{HERE}/{s}").read()
        for p, r in REN:
            t = re.sub(p, r, t)
        open(f"{jd}/{d}", "w").write(t)
    hist = json.load(open(f"{J.OUT}/models/{a.name}.json"))
    v2m = json.load(open(J.V2 + ".meta.json"))
    ev = json.load(open(f"{J.OUT}/eval/{a.name}.json")) if os.path.exists(f"{J.OUT}/eval/{a.name}.json") else None
    agg = None
    one = lambda r: {k: dict(cref_churn=r[k]["cref"], v2=r[k]["at_cref"]["v2"], jF=r[k]["at_cref"][a.name],  # noqa: E731
                             v2_1p5=r[k]["at_1p5cref"]["v2"], jF_1p5=r[k]["at_1p5cref"][a.name], static=r[k]["static"])
                     for k in ("pooled", "lmean")}
    if ev:      # top level = decode (primary); by_kind = decode / prefill / prefill_c<chunk> (aggregates only)
        agg = {sp: dict(one(r), by_kind={g: one(v) for g, v in r.get("by_kind", {}).items()},
                        handoff={n: r["handoff"][n] for n in ("v2", a.name) if n in r["handoff"]} if r.get("handoff") else None)
               for sp, r in ev["results"].items()}
    chj = json.load(open(f"{J.OUT}/chains.json"))
    chs = chj.get("splits", {})
    data = dict(sources=[s["type"] for s in chj.get("sources", [])], mix=chj.get("mix"),
                dec_prefill=chj.get("dec_prefill", False))
    nparam = sum(v.numel() for v in ck["state"].values())
    Ls = ck["layers"]
    cfg = dict(
        model="GLM-5.3-Flash", args=ck["args"], K=ck["K"], params=nparam, inputs=ck["inputs"], extra=ck["extra"],
        sha256={"jF.pt": sha(f"{jd}/jF.pt"), "v2_sal_tweedie1.5.txt": sha(f"{jd}/v2_sal_tweedie1.5.txt")},
        fixed_set_sha256=ck["fixed_set_sha256"],
        serve=dict(n_float=ck["nf"], n_fixed=ck["n_fixed"], hm=a.hm, refresh_tokens=J.G, mode="sync",
                   layout=f"k{ck['n_fixed']}f{ck['nf']}", layers=rng(Ls), NE=ck["ne"], top_k=J.TOPK,
                   routed_scaling_factor=2.5, think_id=J.THINK_ID, ethink_id=J.ETHINK_ID,
                   sal="sum over routed slots of w^2 * xn, w incl. routed_scaling_factor, xn = fp32 sum x^2 of the "
                       "normalised MoE input"),
        v2=dict(best_iteration=v2m["best_iteration"], band=v2m["band"], objective=v2m["obj"]),
        train=dict(best_val_pooled_sal_hot=hist["best_key"], cref_churn=hist["cref"],
                   selection=hist.get("selection", "val pooled sal-hot"), crefs=hist.get("crefs"),
                   data=data, train_prefill_frac=hist.get("train_prefill_frac"),
                   splits={k: {kk: v[kk] for kk in v if kk in ("chains", "blocks", "tokens", "docs", "by_kind")}
                           for k, v in chs.items()} if isinstance(chs, dict) else {}),
        eval=agg, exported=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    json.dump(cfg, open(f"{jd}/jF.json", "w"), indent=1)
    ft = lambda v: "n/a" if v is None else f"{100 * v:.2f}%"  # noqa: E731
    ln, hln = [], []
    if agg:
        for sp in agg:
            p = agg[sp]["pooled"]
            ln.append(f"| {sp} | decode | {p['cref_churn']:.2f} | {ft(p['v2'])} | {ft(p['jF'])} | {ft(p['static'])} |")
            for g, q in agg[sp]["by_kind"].items():
                if g != "decode":
                    q = q["pooled"]
                    ln.append(f"| {sp} | {g} | {q['cref_churn']:.2f} | {ft(q['v2'])} | {ft(q['jF'])} | {ft(q['static'])} |")
        for sp in agg:
            h = agg[sp].get("handoff")
            if h and a.name in h:
                for n in h[a.name]["gain"]:
                    hv, hj = h["v2"], h[a.name]
                    hln.append(f"| {sp} | {n} | {hj['seeded'][n]['chains']} | {ft(hv['seeded'][n]['pooled'])} | "
                               f"{ft(hv['cold'][n]['pooled'])} | {ft(hj['seeded'][n]['pooled'])} | {ft(hj['cold'][n]['pooled'])} |")
    open(f"{jd}/README.md", "w").write(f"""# Joint floating-set predictor (jF), GLM-5.3-Flash

Chooses, every 16 decode tokens, which {ck['nf']} of the {ck['ne'] - ck['n_fixed']} non-fixed routed experts of each
MoE layer ({rng(Ls)}) are held at 4 bit; the {ck['n_fixed']} fixed experts per layer (fixed_set.json, sha256
{ck['fixed_set_sha256'][:12]}) are always 4 bit.  Port of the GLM-5.3 jF predictor (same features, same net shape).

Score = exp(log v2 + r): v2 = the LightGBM GBDT `v2_sal_tweedie1.5.txt` ({v2m['best_iteration']} trees, 9 features)
applied to all {ck['ne']} experts; r = a 2-layer transformer (d {ck['args']['d']}, {nparam / 1e6:.2f}M parameters,
no expert-identity embedding) over the 22 routing-history inputs of the experts of a layer.  Top-{ck['nf']} with
hysteresis (resident scores x {1 + a.hm}) among the non-fixed experts.

The checkpoint holds only network weights and settings; no text, token ids or routing traces.

## Files

- `jF.pt` weights, `jF.json` settings + aggregate offline eval; `v2_sal_tweedie1.5.txt` the v2 GBDT (lightgbm).
- `gpu_predictor.py`: `GPUJointPredictor`, torch serve path (device cuda / cpu / mps; trees as torch tables).
- `joint_predictor.py`: numpy + lightgbm reference; `parity_stream.py`: offline-vs-streaming parity (needs the
  private offline data, not shipped).  `train.py` / `jlib.py`: network and feature definitions imported by the above.

## Use

    import json, gpu_predictor as GP
    fixed = {{int(L): v for L, v in json.load(open("fixed_set.json"))["fixed_set"].items()}}
    p = GP.GPUJointPredictor(range({min(Ls)}, {max(Ls) + 1}), fixed, "jF.pt", hm={a.hm}, device="cpu",
                             v2_model="v2_sal_tweedie1.5.txt")
    p.step(counts, ntok, token_ids, new_request, sal)   # counts / sal [{len(Ls)}, {ck['ne']}] per decode step
    want = p.target(resident)                           # bool [{len(Ls)}, {ck['ne']}]: floating experts at 4 bit

Build a new instance for each request; the resident floating set starts at floating_default (fixed_set.json).

## Offline eval (sync lag-0 replay, pooled salience share served at 4 bit, at v2's churn for hm {a.hm})

| split | kind | churn (new floating / block / layer) | v2 | jF | static (fixed + floating_default) |
|---|---|---|---|---|---|
""" + "\n".join(ln) + f"\n\nTraining data: {', '.join(data['sources'])}; prefill share of train rows "
      f"{ft(hist.get('train_prefill_frac'))}.  Model selection on decode validation chains only.  `prefill` = "
      f"prompt / teacher-forced chains at a 16-token refresh, `prefill_c<N>` = the set frozen per N-token prefill "
      "chunk (upper bound for short follow-up turns only).  No KLD measurement yet.\n"
      + ("\n## Prefill -> decode handoff (primary prefill-related number)\n\nDecode salience share at 4 bit over the "
         "first n decode blocks after a prompt: seeded = `step_chunk` over the last 4096 prompt tokens + one refresh at "
         "hm 0; cold = fresh predictor at floating_default.\n\n| split | n blocks | seqs | v2 seeded | v2 cold | jF seeded "
         "| jF cold |\n|---|---|---|---|---|---|---|\n" + "\n".join(hln) + "\n" if hln else ""))
    open(f"{jd}/SERVE.md", "w").write(f"""# jF serve note (GLM-5.3-Flash)

Interface = GLM-5.3 joint predictor (`mode='sync'`), only the shapes change: NL = {len(Ls)} (layers {rng(Ls)}),
NE = {ck['ne']}, n_float = {ck['nf']}, n_fixed = {ck['n_fixed']}, top-k {J.TOPK}, routed_scaling_factor 2.5.

Per decode step (or chunk of <= 16 tokens; `step()` ignores larger steps):
- `counts [NL, {ck['ne']}]`: routed-slot hits of the step's top-{J.TOPK} routing.
- `sal [NL, {ck['ne']}]`: sum over the step's routed slots of `w^2 * xn`, w = final combine weight INCLUDING the 2.5
  routed scaling, xn = fp32 sum(x^2) of the normalised MoE input (hidden 4096).
- `token_ids`: only for the think / answer state ({J.THINK_ID} `<think>`, {J.ETHINK_ID} `</think>`);
  `new_request=True` resets it.
Every 16 tokens: 22 inputs per (layer, expert), v2 trees on all experts, the net (one batch of NL x {ck['ne']}), then
`target(resident)` = top-{ck['nf']} non-fixed experts, resident scores multiplied by {1 + a.hm}.
## Prefill -> decode handoff (`step_chunk`, the handoff seeder)

The Mac runs every prompt of >= 1K new tokens layer-major and never refreshes the floating set inside a prefill
(threads/37-flash/mac/SPEC.md sec. 4); the prefill runs on the resident set (floating_default for a new request).
`step_chunk` only SEEDS the predictor for the handoff; it does not serve a per-chunk set:
1. after the prefill, per layer, bin the routing of the last ~4096 prompt tokens (256 16-token blocks; the longest
   EMA half-life is 2048) into `counts`, `sal` [nb, NL, {ck['ne']}] (optional answer-segment split
   `counts_ans`, `n_ans`, `seg_last`, as in the offline block format);
2. `p.step_chunk(counts, sal, ...)` folds them into the state block by block with no scoring and no set changes,
   then scores once;
3. `want = p.target_seed()` = top-{ck['nf']} non-fixed experts at hm 0 (no resident bonus; a layer with no score keeps
   floating_default) -> swap in before the first decode block;
4. decode continues with `step()` / `target(resident)` at hm {a.hm}.
The prompt length must be a multiple of 16 at the seeded tail's start (`step_chunk` asserts a block boundary): use
the last floor(len/16)*16 tokens ending at the prompt end.  Offline: eval37 "handoff" (seeded vs cold decode sal-hot
over the first 1 / 4 / 16 decode blocks).  Short (< 1K) follow-up prefills: `prefill_c1024` is only an upper bound.

State ~164 B per (layer, expert) = {164 * len(Ls) * ck['ne'] / 1e6:.1f} MB; weights {4 * nparam / 1e6:.1f} MB fp32.
""")
    top = dict(format="nq-predictor-v1", type="joint", default="joint", model="GLM-5.3-Flash",
               config="joint/jF.json", code="joint/gpu_predictor.py", model_file="joint/jF.pt",
               model_sha256=cfg["sha256"]["jF.pt"], v2_model="joint/v2_sal_tweedie1.5.txt",
               v2_sha256=cfg["sha256"]["v2_sal_tweedie1.5.txt"], code_sha256=sha(f"{jd}/gpu_predictor.py"),
               params=cfg["serve"], applies_to=f"floating set ({ck['nf']}) of the {ck['n_fixed']}-fixed layout; "
               "fixed experts from fixed_set.json (sha256 in joint/jF.json)")
    json.dump(top, open(f"{a.rel}/predictor.json", "w"), indent=1)
    selftest(jd, Ls)
    shutil.rmtree(f"{jd}/__pycache__", ignore_errors=True)
    # privacy whitelist check
    bad = [f for f in os.listdir(jd) if f not in ALLOWED]
    bad += [f for f in os.listdir(a.rel) if f not in ("joint", "predictor.json")]
    if bad:
        sys.exit(f"[export37] unexpected files in release: {bad}")
    tot = sum(os.path.getsize(f"{jd}/{f}") for f in os.listdir(jd))
    print(f"[export37] {a.rel}: predictor.json + joint/ ({len(os.listdir(jd))} files, {tot / 1e6:.1f} MB) "
          f"jF.pt {cfg['sha256']['jF.pt'][:12]} serve {cfg['serve']['layout']} L{cfg['serve']['layers']}", flush=True)


if __name__ == "__main__":
    main()
