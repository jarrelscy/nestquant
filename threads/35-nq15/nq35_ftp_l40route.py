"""T35 ft pilot helper (CPU): layer-L routing of the held-out windows (identical for every arm: its input is the exact
FP8 upstream cached by nq35_ftp.py) -> OUT/routing_L{L}.pt (layers=[L], ids, w, xn2, dec; same layout as routing/*.pt)
so nq35_ftp_jf.py --extra can put layer L in the jF replay (how often the tuned experts would be served cold = L2).
  python nq35_ftp_l40route.py --out /tmp/nestquant/35-nq15/ftp [--n-win 64 --n-held 0]
"""
import os, sys, json, argparse
T = "/home/coder/git/nestquant/threads"
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--layer", type=int, default=40)
ap.add_argument("--seq", type=int, default=512); ap.add_argument("--n-win", type=int, default=64)
ap.add_argument("--n-held", type=int, default=0)
ap.add_argument("--cache", default="/tmp/nestquant/35-nq15/private/ftp_cache")
ap.add_argument("--ft", default="/tmp/nestquant/35-nq15/private/ft")
a = ap.parse_args()
os.environ["NQ_SEQ"] = str(a.seq)
for p in (f"{T}/05-exl3-harness", "/home/coder/git/orbit-duet", f"{T}/25-campaign", f"{T}/27-pv-tune",
          f"{T}/18-e2e-eval", f"{T}/12-reference-encoder"):
    sys.path.insert(0, p)
import numpy as np                  # noqa: E402
import torch                        # noqa: E402
import nq_e2e as E2E                # noqa: E402
import nq_io                        # noqa: E402
torch.set_num_threads(8)
L = a.layer
C = torch.load(f"{a.cache}/L{L}_s{a.seq}_tr{a.n_win}_ho{a.n_held}.pt", weights_only=False)
meta = [json.loads(x) for x in open(f"{a.ft}/heldout.meta.jsonl")]
rows = [k for k, m in enumerate(meta) if m["src"] == "fp8dec"]
if a.n_held and a.n_held < len(rows):
    rows = [rows[k] for k in np.linspace(0, len(rows) - 1, a.n_held).round().astype(int)]
NH = len(rows)
tok = np.load(f"{a.ft}/heldout.tok.npy", mmap_mode="r"); dec = np.load(f"{a.ft}/heldout.dec.npy", mmap_mode="r")
t = torch.as_tensor(np.asarray(tok[rows, -a.seq:]), dtype=torch.long)
dm = torch.as_tensor(np.asarray(dec[rows, -a.seq:])).clone(); dm[:, -1] = False
assert torch.equal(C["tok"][-NH:], t)
h = C["hmid"].reshape(-1, a.seq, C["hmid"].shape[-1])[-NH:].reshape(-1, C["hmid"].shape[-1])
cfg = E2E.load_config()
fp8 = nq_io.FP8Model(E2E.FP8_DIR)
layer, sparse = E2E.Backbone(cfg, fp8, "cpu").build(L)
assert sparse
with torch.no_grad():
    x = layer.post_attention_layernorm(h)
    _, w, i = layer.mlp.gate(x)
torch.save(dict(layers=[L], ids=i.to(torch.int16)[None], w=w.half()[None], xn2=x.float().pow(2).sum(-1)[None], dec=dm),
           f"{a.out}/routing_L{L}.pt")
print(f"wrote {a.out}/routing_L{L}.pt: {NH} windows, {int(dm.sum())} decode rows", flush=True)
