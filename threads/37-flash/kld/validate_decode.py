"""T37 KLD validation (a): the harness decoder (nqdec37, /tmp/venv-t37g) == the production reference decoder
(threads/12 nq_decode.decode_expert under nq37_env/nq15, glm52 .venv; the encoder asserted bit-exactness of that
decode against its internal dense W at encode time: meta.info[proj].bitexact).  CPU only.

  # 1. official reference (glm52 venv, encode env):
  LD_LIBRARY_PATH=$NQ/threads/06-expert-objective/lib CUDA_VISIBLE_DEVICES= \
      /home/coder/git/glm52/.venv/bin/python validate_decode.py ref 3:10,3:235,22:0
  # 2. harness decoder + compare + weight-error sanity vs the fp8 teacher (venv-t37g):
  CUDA_VISIBLE_DEVICES= /tmp/venv-t37g/bin/python validate_decode.py cmp 3:10,3:235,22:0
Writes aggregate numbers to /tmp/nestquant/37-flash/kld/validate_decode.json."""
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ENC = os.environ.get("NQ37_ENC", "/tmp/nestquant/37-flash/enc_b15")
OUT = "/tmp/nestquant/37-flash/kld/val_decode"
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "12")))


def pairs(s):
    return [tuple(map(int, p.split(":"))) for p in s.split(",")]


def ref(ps):
    sys.path.insert(0, os.path.dirname(HERE))
    import nq37_env  # noqa: F401  (installs nq15: base_K-aware ring_levels)
    import nq_decode as D
    os.makedirs(OUT, exist_ok=True)
    for L, E in ps:
        art = torch.load(f"{ENC}/L{L}/experts/E{E}.pt", weights_only=False, map_location="cpu")
        t0 = time.time()
        r = {lv: D.decode_expert(art, lv, device="cpu") for lv in (2, 4)}
        torch.save(r, f"{OUT}/ref_L{L}_E{E}.pt")
        print(f"ref L{L} E{E} {time.time() - t0:.1f}s", flush=True)


def cmp(ps):
    sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
    import nqdec37 as Q
    from safetensors import safe_open
    CK = "/tmp/nestquant/37-flash/fp8"
    idx = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]

    def fp8w(k):
        with safe_open(f"{CK}/{idx[k + '.weight']}", "pt") as f:
            w = f.get_tensor(k + ".weight")
        with safe_open(f"{CK}/{idx[k + '.weight_scale_inv']}", "pt") as f:
            s = f.get_tensor(k + ".weight_scale_inv")
        s = s.float().repeat_interleave(128, 0).repeat_interleave(128, 1)[: w.shape[0], : w.shape[1]]
        return (w.float() * s).bfloat16().float()                  # = capture37 get(): the teacher's bf16 weights

    res = []
    for L, E in ps:
        art = torch.load(f"{ENC}/L{L}/experts/E{E}.pt", weights_only=False, map_location="cpu")
        t0 = time.time()
        mine = Q.decode_expert_f32(art, (2, 4), "cpu")
        td = time.time() - t0
        mbf = Q.decode_expert(art, (2, 4), "cpu")
        r = torch.load(f"{OUT}/ref_L{L}_E{E}.pt", weights_only=False)
        T = [fp8w(f"model.language_model.layers.{L}.mlp.experts.{E}.{p}") for p in ("gate_proj", "up_proj", "down_proj")]
        row = dict(L=L, E=E, decode_s=round(td, 2))
        for lv in (2, 4):
            row[f"L{lv}_maxabs_vs_ref_f32"] = max(float((a - b).abs().max()) for a, b in zip(mine[lv], r[lv]))
            row[f"L{lv}_bitexact_f32"] = all(torch.equal(a, b) for a, b in zip(mine[lv], r[lv]))
            gu, dn = mbf[lv]
            row[f"L{lv}_bf16_eq_ref_cast"] = bool(torch.equal(gu, torch.cat([r[lv][0], r[lv][1]]).bfloat16())
                                                 and torch.equal(dn, r[lv][2].bfloat16()))
            row[f"L{lv}_relerr_vs_fp8"] = {p: round(float((a - t).square().sum() / t.square().sum()), 5)
                                          for p, a, t in zip(("gate", "up", "down"), mine[lv], T)}
        row["encoder_proxy_rot"] = {p: art["meta"]["info"][p]["proxy_rot"] for p in ("gate", "up", "down")}
        row["encoder_bitexact"] = {p: art["meta"]["info"][p]["bitexact"] for p in ("gate", "up", "down")}
        print(json.dumps(row), flush=True)
        res.append(row)
    ok = all(r["L2_bitexact_f32"] and r["L4_bitexact_f32"] and r["L2_bf16_eq_ref_cast"] and r["L4_bf16_eq_ref_cast"]
             for r in res)
    json.dump(dict(ok=ok, rows=res, time=time.strftime("%Y-%m-%d %H:%M:%S %Z")),
              open("/tmp/nestquant/37-flash/kld/validate_decode.json", "w"), indent=1)
    print("DECODE CHECK", "PASS" if ok else "FAIL", flush=True)


if __name__ == "__main__":
    {"ref": ref, "cmp": cmp}[sys.argv[1]](pairs(sys.argv[2]))
