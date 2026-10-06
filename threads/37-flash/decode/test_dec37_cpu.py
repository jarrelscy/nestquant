#!/usr/bin/env python3
"""CPU correctness tests for dec37.py (no GPU).  Run with nice 10, <= 16 threads:
  OMP_NUM_THREADS=16 nice -n 10 /tmp/venv-t37g/bin/python test_dec37_cpu.py [--skip-real] [--only NAME]

  t_quant   fp8 block dequant (block path == capture37g _dq, bitwise) + NVFP4 pack/unpack round trip
  t_tiny    tiny fake Flash checkpoint (8 layers, DSA at 3/7, 16 experts top-4, index_topk 16 / kpool 4 so the
            indexer prunes from position 19 on), FP8 experts + FP8 DSA/dense linears, and an NVFP4-expert variant:
            HF Glm5NextTextModel (fp32, dequantised weights, no cache) vs dec37 ragged prefill wave
            (KDA piece chaining, batched/padded) + incremental ragged decode with a free dummy slot,
            final logits + routing records (ids / w / xn) at every row
  t_driver  full dec37 driver on the tiny checkpoint (greedy, 8 smoke prompts through 3 slots = refills):
            shard / jF-layout checks (blocks37 asserts) + every greedy token == HF argmax + routing == HF
  t_real    real GLM-5.3-Flash weights, fp32: L0 KDA attention (prefill 1300 tok + 8 decode steps) and L3 DSA attention
            (prefill 2600 tok = 650 pools > 512, pruned + 8 decode steps) vs the HF modules; L3 MoE vs the
            capture37g teacher math (64 tokens)
"""
import argparse
import copy
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dec37 as D  # noqa: E402

torch.set_grad_enabled(False)
REAL = D.CK_DEFAULT
TMP = "/tmp/nestquant/37-flash/dec37_test"
from transformers.models.glm5_next import modeling_glm5_next as M  # noqa: E402
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig  # noqa: E402

RES = {}


def report(name, ok, msg):
    RES[name] = (ok, msg)
    print(("PASS " if ok else "FAIL ") + name + ": " + msg, flush=True)


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


# ------------------------------------------------------------------ quantisers (test side)
def q_fp8(w):
    o, i = w.shape
    b = w.view(o // 128, 128, i // 128, 128)
    s = b.abs().amax((1, 3)).clamp_min(1e-12) / 448.0
    q = (b / s[:, None, :, None]).to(torch.float8_e4m3fn).view(o, i)
    return q, s.float()


E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def q_nvfp4(w):
    """-> packed u8 [o, i/2], scale fp8 [o, i/16], global fp32 scalar, dequant fp32 (independent of dec37)."""
    o, i = w.shape
    g = w.abs().max().clamp_min(1e-12) / (448.0 * 6.0)
    b = w.view(o, i // 16, 16)
    s = (b.abs().amax(-1) / 6.0 / g).clamp(max=448).to(torch.float8_e4m3fn)
    sf = s.float() * g
    x = b / sf.clamp_min(1e-30)[..., None]
    mag = (x.abs()[..., None] - E2M1).abs().argmin(-1)
    code = (mag + 8 * (x < 0).long()).view(o, i)
    deq = (E2M1[mag] * torch.where(x < 0, -1.0, 1.0) * sf[..., None]).view(o, i)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).to(torch.uint8)
    return packed, s, g.reshape(1).float(), deq


def t_quant():
    torch.manual_seed(0)
    w = torch.randn(256, 384) * 0.05
    q, s = q_fp8(w)
    ref = D.dq_ref(q, s)
    sc = D.Scratch(1, 384, 128, torch.bfloat16, "cpu")
    out = sc.gu[0, :256]
    out.copy_(q)
    out.view(2, 128, 3, 128).mul_(s[:, None, :, None])
    ok1 = torch.equal(out, ref)
    pk, s4, g, deq = q_nvfp4(w)
    d4 = D.nvfp4_dq(pk, s4, g, torch.float32)
    ok2 = torch.equal(d4, deq)
    u = D.nvfp4_unpack(torch.tensor([[0x21, 0xF8]], dtype=torch.uint8), torch.float32)
    ok3 = u.tolist() == [[0.5, 1.0, -0.0, -6.0]]
    report("t_quant", ok1 and ok2 and ok3, f"fp8 block==_dq {ok1}; nvfp4 round trip exact {ok2}; nibble order {ok3}; "
           f"nvfp4 rel err vs fp32 {rel(deq, w):.3f}")


# ------------------------------------------------------------------ tiny checkpoint
def tiny_cfg(swiglu=1.0):
    nL = 8
    return dict(
        model_type="glm5_next_text", vocab_size=154880, hidden_size=256, intermediate_size=256, moe_intermediate_size=128,
        num_hidden_layers=nL, num_attention_heads=4, num_key_value_heads=4, kv_lora_rank=128, q_lora_rank=128,
        qk_nope_head_dim=64, qk_rope_head_dim=0, v_head_dim=64, qk_head_dim=64, head_dim=0, n_routed_experts=16,
        num_experts_per_tok=4, n_shared_experts=1, routed_scaling_factor=2.5, n_group=1, topk_group=1,
        norm_topk_prob=True, first_k_dense_replace=3, index_n_heads=16, index_head_dim=32, index_topk=16, index_kpool=4,
        index_kpool_always_select_tail=True, index_kpool_compress=True, hc_mult=4, hc_eps=1e-6, hc_sinkhorn_iters=20,
        rms_norm_eps=1e-5, swiglu_limit=swiglu, hidden_act="silu", attention_bias=False, pad_token_id=154820,
        linear_attn_config=dict(num_heads=4, head_dim=32, short_conv_kernel_size=4, gate_lower_bound=-5.0,
                                kda_layers=[i for i in range(nL) if i % 4 != 3],
                                full_attn_layers=[i for i in range(nL) if i % 4 == 3]),
        max_position_embeddings=4096, scoring_func="sigmoid", topk_method="noaux_tc", tie_word_embeddings=False,
        mlp_layer_types=["dense"] * 3 + ["sparse"] * (nL - 3), moe_router_dtype="float32")


FP8_LIN = ("self_attn.q_a_proj", "self_attn.q_b_proj", "self_attn.kv_a_proj_with_mqa", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def make_tiny(path, expert_fmt="fp8", seed=0):
    """random tiny model -> (HF Glm5NextTextModel fp32 with the DEQUANTISED weights, lm_head) + checkpoint at path."""
    from safetensors.torch import save_file
    torch.manual_seed(seed)
    tc = tiny_cfg()
    cfg = Glm5NextTextConfig(**{k: v for k, v in tc.items() if k != "model_type"})
    cfg._attn_implementation = "eager"
    cfg.num_local_experts = cfg.n_routed_experts
    model = M.Glm5NextTextModel(cfg).float().eval()
    sd = model.state_dict()
    for k, v in sd.items():
        if not v.is_floating_point():
            continue
        if k.endswith(("layernorm.weight", "norm.weight", "o_norm.weight", "k_norm.weight")):
            v.copy_(1 + 0.1 * torch.randn_like(v))
        elif k.endswith("k_norm.bias"):
            v.copy_(0.05 * torch.randn_like(v))
        elif k.endswith("A_log"):
            v.copy_(0.3 * torch.randn_like(v))
        elif k.endswith(("dt_bias", "e_score_correction_bias", "compress_ape")):
            v.copy_(0.2 * torch.randn_like(v))
        elif k.endswith("_hc.fn"):
            v.copy_(torch.randn_like(v) / v.shape[1] ** 0.5)
        elif k.endswith("_hc.base"):
            v.copy_(0.5 * torch.randn_like(v))
        elif k.endswith("_hc.scale"):
            v.copy_(1 + 0.2 * torch.randn_like(v))
        elif k.endswith("embed_tokens.weight"):
            v.copy_(torch.randn_like(v))
        elif v.dim() == 3:                                       # experts [NE, out, in]
            v.copy_(torch.randn_like(v) / v.shape[2] ** 0.5)
        elif v.dim() == 2:
            v.copy_(torch.randn_like(v) / v.shape[1] ** 0.5)
        elif k.endswith("conv1d.weight") or v.dim() == 3:
            v.copy_(torch.randn_like(v) * 0.5)
    for k, v in sd.items():                                      # conv1d [C, 1, K]
        if k.endswith("conv1d.weight"):
            v.copy_(torch.randn_like(v) * 0.5)
    head = torch.randn(cfg.vocab_size, cfg.hidden_size) / cfg.hidden_size ** 0.5
    ck = {}
    P = D.PFX
    for k, v in sd.items():
        m = k.split(".")
        if k.startswith("layers.") and ".mlp.experts." in k:
            L = int(m[1])
            NE, Fm = cfg.n_routed_experts, cfg.moe_intermediate_size
            for e in range(NE):
                for nm, w in (("gate_proj", v[e, :Fm]), ("up_proj", v[e, Fm:])) if k.endswith("gate_up_proj") else \
                        (("down_proj", v[e]),):
                    base = f"{P}layers.{L}.mlp.experts.{e}.{nm}."
                    if expert_fmt == "fp8":
                        q, s = q_fp8(w)
                        ck[base + "weight"], ck[base + "weight_scale_inv"] = q, s
                        w.copy_(D.dq_ref(q, s, torch.float32))
                    else:
                        pk, s4, g, deq = q_nvfp4(w)
                        ck[base + "weight"], ck[base + "weight_scale"], ck[base + "weight_scale_2"] = pk, s4, g
                        w.copy_(deq)
            continue
        if k.endswith("conv1d.weight"):
            L = int(m[1])
            C = v.shape[0] // 3
            for j, n in enumerate("qkv"):
                ck[f"{P}layers.{L}.self_attn.{n}_conv1d.weight"] = v[j * C:(j + 1) * C].bfloat16().clone()
            v.copy_(v.bfloat16().float())
            continue
        r = k
        r = r.replace("self_attn.forget_gate.", "self_attn.")
        r = r.replace("attn_hc.", "hc_attn_").replace("ffn_hc.", "hc_ffn_")
        if any(r.endswith(x + ".weight") for x in FP8_LIN) and v.shape[0] % 128 == 0 and v.shape[1] % 128 == 0 \
                and ("self_attn.q_a" in r or "self_attn.q_b" in r or "kv_a_proj" in r or ".mlp." in r
                     or ("o_proj" in r and int(m[1]) % 4 == 3)):
            q, s = q_fp8(v)
            ck[P + r], ck[P + r[:-len("weight")] + "weight_scale_inv"] = q, s
            v.copy_(D.dq_ref(q, s, torch.float32))
            continue
        fp32 = r.endswith(D.FP32_KEYS) or "hc_" in r
        t = v.float() if fp32 else v.bfloat16()
        ck[P + r] = t.clone()
        v.copy_(t.float())
    ck["lm_head.weight"] = head.bfloat16()
    head = head.bfloat16().float()
    os.makedirs(path, exist_ok=True)
    save_file({k: v.contiguous() for k, v in ck.items()}, f"{path}/model-00001-of-00001.safetensors")
    json.dump(dict(weight_map={k: "model-00001-of-00001.safetensors" for k in ck}),
              open(f"{path}/model.safetensors.index.json", "w"))
    real = json.load(open(f"{REAL}/config.json"))
    real["text_config"] = tc
    real.pop("quantization_config", None)
    json.dump(real, open(f"{path}/config.json", "w"))
    for f in glob.glob(f"{REAL}/*token*") + glob.glob(f"{REAL}/chat_template*") + [f"{REAL}/generation_config.json"]:
        shutil.copy(f, path)
    model.load_state_dict(sd)
    return model, head


class Hooks:
    """records (ids, w, xn) of every HF MoE router call."""

    def __init__(self, model):
        self.rec, self.h = {}, []
        for L, lay in enumerate(model.layers):
            if isinstance(lay.mlp, M.Glm5NextTextMoE):
                self.h.append(lay.mlp.register_forward_hook(self._mk(L)))

    def _mk(self, L):
        def f(mod, inp, out):
            x = inp[0].reshape(-1, inp[0].shape[-1])
            _, tw, ti = mod.gate(x)
            self.rec[L] = (ti, tw, x.float().square().sum(1))
        return f

    def close(self):
        for h in self.h:
            h.remove()


def hf_ref(model, head, ids):
    hk = Hooks(model)
    h = model(input_ids=torch.tensor(ids)[None]).last_hidden_state[0]
    hk.close()
    return F.linear(h, head), hk.rec


def eng_args(ck, **kw):
    a = D.parse_args(["--ckpt", ck, "--devices", "cpu", "--dtype", "fp32", "--smoke", "--expert-block", "4",
                      "--kda-piece", "64", "--kda-tokens", "128", "--attn-budget-gb", "0.001", "--tok-chunk", "50"])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def cmp_routing(rec_e, rec_h, n, tag):
    """dec37 per-row records vs HF hook records for one sequence: id SETS equal, w / xn close."""
    worst, mism = 0.0, 0
    for L, (ti, tw, xn) in rec_h.items():
        i_e, w_e, x_e = rec_e[L]
        ih = ti[:n].numpy()
        oe, oh = np.argsort(i_e.astype(np.int64), 1), np.argsort(ih, 1)
        se, sh = np.take_along_axis(i_e.astype(np.int64), oe, 1), np.take_along_axis(ih, oh, 1)
        mism += int((se != sh).any(1).sum())
        we, wh = np.take_along_axis(w_e.astype(np.float32), oe, 1), np.take_along_axis(tw[:n].numpy(), oh, 1)
        good = (se == sh).all(1)
        if good.any():
            worst = max(worst, float(np.abs(we[good] - wh[good]).max()))
        worst = max(worst, float(np.abs(x_e - xn[:n].numpy()).max() / max(1e-9, float(xn[:n].abs().max()))))
    return mism, worst


def t_tiny(fmt):
    name = f"t_tiny_{fmt}"
    ck = f"{TMP}/tiny_{fmt}"
    shutil.rmtree(ck, ignore_errors=True)
    model, head = make_tiny(ck, fmt, seed=1 if fmt == "fp8" else 2)
    a = eng_args(ck)
    NS, Smax = 4, 112
    eng = D.Engine(a, NS, Smax)
    g = torch.Generator().manual_seed(5)
    Ls = [100, 23, 71]                       # full lengths; ragged
    pre = [60, 23 - 6, 33]                   # prefill lengths (seq 0 crosses the prune threshold inside the prefill)
    seqs = [torch.randint(0, 150000, (n,), generator=g).tolist() for n in Ls]
    refs = [hf_ref(model, head, s) for s in seqs]
    slots = [0, 2, 3]                        # slot 1 = free dummy slot
    wv = D.Wave(slots, [np.asarray(s[:p]) for s, p in zip(seqs, pre)])
    last = eng.prefill(wv)
    errs = [rel(last[j], refs[j][0][pre[j] - 1]) for j in range(3)]
    st = D.StepState(NS, [torch.device("cpu")])
    for j, s in enumerate(slots):
        st.pos_h[s] = pre[j]
    dec_err, steps = [], 0
    agree, tot = 0, 0
    while True:
        tok = torch.zeros(NS, dtype=torch.long)
        live = False
        for j, s in enumerate(slots):
            if st.pos_h[s] < Ls[j]:
                tok[s] = seqs[j][st.pos_h[s]]
                live = True
        if not live:
            break
        z = eng.decode(tok, st)
        for j, s in enumerate(slots):
            p = st.pos_h[s]
            if p < Ls[j]:
                dec_err.append(rel(z[s], refs[j][0][p]))
                agree += int(z[s].argmax() == refs[j][0][p].argmax())
                tot += 1
                st.pos_h[s] += 1
            else:
                st.pos_h[s] = min(p, Smax - 1)            # finished rows keep decoding garbage at a frozen pos
        st.pos_h[1] = 0
        steps += 1
    mism, wdiff = 0, 0.0
    for j, s in enumerate(slots):
        m, w = cmp_routing(eng.records(s, Ls[j]), refs[j][1], Ls[j], name)
        mism += m
        wdiff = max(wdiff, w)
    ok = max(errs) < 2e-3 and max(dec_err) < 2e-3 and mism == 0 and wdiff < 1e-3
    report(name, ok, f"prefill last-logit rel err max {max(errs):.2e}; decode {steps} steps rel err max "
           f"{max(dec_err):.2e} (argmax agree {agree}/{tot}); routing id-set mismatches {mism} rows, max |dw|/xn rel "
           f"{wdiff:.1e}")
    return model, head, ck


def t_driver(model, head, ck):
    name = "t_driver"
    out = f"{TMP}/drv"
    shutil.rmtree(out, ignore_errors=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(min(16, os.cpu_count())))
    cmd = [sys.executable, f"{HERE}/dec37.py", "--ckpt", ck, "--devices", "cpu", "--dtype", "fp32", "--smoke",
           "--slots", "3", "--max-new", "20", "--temp", "0", "--out", out, "--expert-block", "4", "--kda-piece", "64",
           "--wave-tokens", "80", "--shard-rows", "150", "--threads", "8", "--log-every", "5"]
    t0 = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-5000:])
        report(name, False, f"driver exit {r.returncode}")
        return
    print("\n".join(r.stdout.strip().splitlines()[-4:]))
    sys.path.insert(0, os.path.join(HERE, "..", "jf"))
    fs = sorted(glob.glob(f"{out}/seqs.r*of*.json"))
    W = len(fs)
    probs = []
    seen = set()
    tot_rows, n_seq, gtot, gagree = 0, 0, 0, 0
    mism_t, wd = 0, 0.0
    gen = {json.loads(l)["id"]: json.loads(l) for l in open(f"{out}/gen.jsonl")}
    for rk in range(W):
        j = json.load(open(f"{out}/seqs.r{rk}of{W}.json"))
        tk = np.load(f"{out}/tok.r{rk}of{W}.npy")
        if not (len(tk) == j["N"] == sum(q["rows"] for q in j["seqs"])):
            probs.append(f"rank {rk} N mismatch")
        npz = {int(os.path.basename(f).split(".")[0][1:]): np.load(f) for f in glob.glob(f"{out}/L*.r{rk}of{W}.npz")}
        for L, z in npz.items():
            if not (z["ids"].shape == (j["N"], 4) and z["ids"].dtype == np.uint16 and z["w"].dtype == np.float16
                    and z["xn"].shape == (j["N"],)):
                probs.append(f"L{L} r{rk} bad arrays")
        o = 0
        for q in j["seqs"]:
            n, p = q["rows"], q["prompt_len"]
            gi = gen[q["id"]]["gen"]
            if n != p + len(gi) - 1 or q["group"] != q["id"]:
                probs.append(f"{q['id']} rows {n} != {p}+{len(gi)}-1 or group")
            full = tk[o:o + n].tolist() + [gi[-1]]
            if full[p:] != gi:
                probs.append(f"{q['id']} tok rows != gen")
            logits, rec_h = hf_ref(model, head, full[:-1] if False else full)
            am = logits.argmax(-1)
            for t in range(p - 1, n):
                gtot += 1
                gagree += int(am[t] == full[t + 1])
            rec_e = {L: (z["ids"][o:o + n], z["w"][o:o + n], z["xn"][o:o + n]) for L, z in npz.items()}
            m, w = cmp_routing(rec_e, {L: tuple(x[:n] for x in v) for L, v in rec_h.items()}, n, name)
            mism_t += m
            wd = max(wd, w)
            seen.add(q["id"])
            o += n
            tot_rows += n
            n_seq += 1
    # --- tf mode on the same sequences: prefill top-64 + forced decode top-64 vs HF log_softmax
    tasks = []
    for rk in range(W):
        j = json.load(open(f"{out}/seqs.r{rk}of{W}.json"))
        tk = np.load(f"{out}/tok.r{rk}of{W}.npy")
        o = 0
        for q in j["seqs"][:2]:
            full = tk[o:o + q["rows"]].tolist() + [gen[q["id"]]["gen"][-1]]
            tasks.append(dict(id=q["id"], tokens=full, split=q["prompt_len"] - 5))
            o += q["rows"]
    json.dump(tasks, open(f"{TMP}/tf_tasks.json", "w"))
    tfo = f"{TMP}/tf"
    shutil.rmtree(tfo, ignore_errors=True)
    r = subprocess.run([sys.executable, f"{HERE}/dec37.py", "--ckpt", ck, "--devices", "cpu", "--dtype", "fp32", "--mode",
                        "tf", "--tasks", f"{TMP}/tf_tasks.json", "--slots", "3", "--out", tfo, "--expert-block", "4",
                        "--kda-piece", "64", "--threads", "8"], env=env, capture_output=True, text=True)
    tf_err, tf_top1 = 0.0, 0
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-4000:])
        probs.append("tf mode failed")
    else:
        for t in tasks:
            z = np.load(f"{tfo}/tf/{t['id']}.npz")
            lg, _ = hf_ref(model, head, t["tokens"])
            v, i = F.log_softmax(lg, -1).topk(64, -1)
            if z["v"].shape != (len(t["tokens"]), 64):
                probs.append(f"tf {t['id']} shape {z['v'].shape}")
                continue
            tf_err = max(tf_err, float(np.abs(z["v"].astype(np.float32) - v.numpy()).max()))
            tf_top1 += int((z["i"][:, 0] == i[:, 0].numpy()).sum()) - len(t["tokens"])
    # --- sampling run (temp 1 / top-p 0.95, 2 samples per prompt -> group = prompt id)
    out2 = f"{TMP}/drv_s"
    shutil.rmtree(out2, ignore_errors=True)
    r = subprocess.run(cmd[:cmd.index("--temp")] + ["--temp", "1.0", "--n-samples", "2", "--out", out2, "--slots", "4",
                       "--expert-block", "4", "--kda-piece", "64"], env=env, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-4000:])
        probs.append("sampling run failed")
    else:
        sq = [q for f in glob.glob(f"{out2}/seqs.r*of*.json") for q in json.load(open(f))["seqs"]]
        grp = {}
        for q in sq:
            grp.setdefault(q["group"], []).append(q["id"])
        if len(sq) != 16 or any(len(v) != 2 or not all(x.startswith(k + "#s") for x in v) for k, v in grp.items()):
            probs.append(f"sampling groups wrong: {len(sq)} seqs")
    msg_tf = f"tf mode max |dlogprob| top-64 {tf_err:.1e} (fp16 storage), top-1 mismatches {-tf_top1 if tf_top1 < 0 else tf_top1}"
    idx = json.load(open(f"{out}/index.json"))
    ok = tf_err < 1e-2 and not probs and n_seq == 8 and len(seen) == 8 and gagree >= gtot - 1 and mism_t == 0 and wd < 5e-3 and idx["W"] == W
    report(name, ok, f"{W} shards, {n_seq} seqs, {tot_rows} rows in {time.time() - t0:.0f}s; greedy tokens == HF argmax "
           f"{gagree}/{gtot}; routing id-set mismatches {mism_t}, max dw (fp16) {wd:.1e}; {msg_tf}; sampling run 16 seqs / 8 groups OK {not any('sampl' in p for p in probs)}; problems {probs[:5]}")


# ------------------------------------------------------------------ expert parallel (--ep) / triton dequant
def ep_engine_run(ck, ep, devices, dtype="fp32", dq="torch", place=None):
    """ragged prefill wave + 30 decode steps of fixed random tokens -> (prefill logits, [decode logits], records)."""
    a = eng_args(ck, devices=devices, ep=ep, dtype=dtype, dq=dq, place=place or "auto")
    NS, Smax = 4, 112
    eng = D.Engine(a, NS, Smax)
    g = torch.Generator().manual_seed(11)
    pre = [70, 9, 33]
    slots = [0, 2, 3]
    seqs = [torch.randint(0, 150000, (p + 30,), generator=g).tolist() for p in pre]
    wv = D.Wave(slots, [np.asarray(sq[:p]) for sq, p in zip(seqs, pre)])
    last = eng.prefill(wv)
    st = D.StepState(NS, sorted(set(eng.dev_of), key=str))
    for j, sl in enumerate(slots):
        st.pos_h[sl] = pre[j]
    zs = []
    for t in range(30):
        tok = torch.zeros(NS, dtype=torch.long)
        for j, sl in enumerate(slots):
            tok[sl] = seqs[j][st.pos_h[sl]]
        zs.append(eng.decode(tok, st).clone())
        for sl in slots:
            st.pos_h[sl] += 1
    recs = [eng.records(sl, pre[j] + 30) for j, sl in enumerate(slots)]
    return last.clone(), zs, recs, eng


def same_run(r1, r2):
    eq = torch.equal(r1[0], r2[0]) and all(torch.equal(x, y) for x, y in zip(r1[1], r2[1]))
    for a, b in zip(r1[2], r2[2]):
        for L in a:
            eq &= all(np.array_equal(x.view(np.uint8), y.view(np.uint8)) for x, y in zip(a[L], b[L]))
    return eq


def t_ep(ck):
    name = "t_ep"
    devs4 = "cpu,cpu,cpu,cpu"
    place = "0,0,1,1,2,2,3,3"
    t0 = time.time()
    r0 = ep_engine_run(ck, False, "cpu")
    r1 = ep_engine_run(ck, False, devs4, place=place)
    r2 = ep_engine_run(ck, True, devs4, place=place)
    lay = next(l for l in r2[3].layers if l.sparse)
    shards = [(sh.e0, sh.e1, sh.remote) for sh in lay.shards]
    eng_ok = same_run(r0, r1) and same_run(r1, r2)
    ep_ok = len(shards) == 4 and sum(s[2] for s in shards) == 3
    # driver: temp-1 sampling, 2 samples per prompt, all outputs bitwise identical with / without --ep
    outs = {}
    env = dict(os.environ, OMP_NUM_THREADS=str(min(16, os.cpu_count())))
    for tag, extra in (("noep", []), ("ep", ["--ep"])):
        out = f"{TMP}/drv_{tag}"
        shutil.rmtree(out, ignore_errors=True)
        cmd = [sys.executable, f"{HERE}/dec37.py", "--ckpt", ck, "--devices", devs4, "--place", place, "--dtype",
               "fp32", "--smoke", "--slots", "5", "--max-new", "24", "--temp", "1.0", "--top-p", "0.95",
               "--n-samples", "2", "--out", out, "--expert-block", "4", "--kda-piece", "64", "--wave-tokens", "120",
               "--shard-rows", "300", "--threads", "8", "--log-every", "5"] + extra
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if r.returncode:
            print(r.stdout[-3000:], r.stderr[-5000:])
            report(name, False, f"driver {tag} exit {r.returncode}")
            return
        outs[tag] = out
    drv_ok, nfile = True, 0
    for f in sorted(glob.glob(f"{outs['noep']}/*") + glob.glob(f"{outs['noep']}/shards/**/*", recursive=True)):
        if os.path.isdir(f):
            continue
        g = f.replace(outs["noep"], outs["ep"])
        nfile += 1
        if f.endswith(".npz"):
            za, zb = np.load(f), np.load(g)
            drv_ok &= set(za) == set(zb) and all(np.array_equal(za[k].view(np.uint8), zb[k].view(np.uint8)) for k in za)
        elif f.endswith(".npy"):
            drv_ok &= np.array_equal(np.load(f), np.load(g))
        elif f.endswith("gen.jsonl"):
            ga = [json.loads(l)["gen"] for l in open(f)]
            gb = [json.loads(l)["gen"] for l in open(g)]
            drv_ok &= ga == gb
    ok = eng_ok and ep_ok and drv_ok and nfile > 0
    report(name, ok, f"engine logits+routing bitwise: 1 dev == 4 dev == 4 dev --ep: {eng_ok} (shards {shards}); "
           f"driver temp-1 n-samples-2 outputs bitwise --ep == no-ep: {drv_ok} ({nfile} files) "
           f"[{time.time() - t0:.0f}s]")


def t_triton_child(ck):
    """runs under TRITON_INTERPRET=1: bf16 engine with the fused Triton dequant == torch dequant, bitwise (EP on)."""
    # kernel unit check: every non-NaN e4m3 byte (incl. -0, subnormals, max) x random block scales == torch, bitwise
    b = torch.tensor([v for v in range(256) if v & 0x7F != 0x7F], dtype=torch.uint8)
    kern = True
    for shp in ((2, 256, 384), (1, 128, 128), (3, 128, 256)):
        n = shp[0] * shp[1] * shp[2]
        w = b[torch.randperm(n, generator=torch.Generator().manual_seed(n)) % len(b)].view(shp).view(torch.float8_e4m3fn)
        sc = torch.rand(shp[0], shp[1] // 128, shp[2] // 128, generator=torch.Generator().manual_seed(3)) * 1e-2 + 1e-4
        ref = (w.float() * sc.repeat_interleave(128, 1).repeat_interleave(128, 2)).to(torch.bfloat16)
        got = D.fp8_dq_triton(w, sc, torch.empty(shp, dtype=torch.bfloat16))
        kern &= torch.equal(ref.view(torch.int16), got.view(torch.int16))
    print("TRITON_KERNEL", kern)
    devs4 = "cpu,cpu,cpu,cpu"
    rt = ep_engine_run(ck, True, devs4, dtype="bf16", dq="triton", place="0,0,1,1,2,2,3,3")
    used = D._DQ["triton"]
    rq = ep_engine_run(ck, True, devs4, dtype="bf16", dq="torch", place="0,0,1,1,2,2,3,3")
    print("TRITON_CHILD", used, same_run(rt, rq))


def t_triton(ck):
    name = "t_triton"
    env = dict(os.environ, TRITON_INTERPRET="1", OMP_NUM_THREADS=str(min(16, os.cpu_count())))
    t0 = time.time()
    r = subprocess.run([sys.executable, __file__, "--only", "t_triton_child", "--ck", ck], env=env,
                       capture_output=True, text=True)
    line = [l for l in r.stdout.splitlines() if l.startswith("TRITON_CHILD")]
    if r.returncode or not line:
        print(r.stdout[-2000:], r.stderr[-4000:])
        report(name, False, f"child exit {r.returncode}")
        return
    used, eq = line[0].split()[1:]
    kl = [l for l in r.stdout.splitlines() if l.startswith("TRITON_KERNEL")]
    kern = bool(kl) and kl[0].split()[1] == "True"
    report(name, used == "True" and eq == "True" and kern,
           f"kernel == torch dequant on all 254 non-NaN e4m3 bytes x random scales (3 shapes): {kern}; bf16 tiny engine (--ep, interpreter): triton dequant active {used}, logits+routing bitwise == torch "
           f"dequant {eq} [{time.time() - t0:.0f}s]")


# ------------------------------------------------------------------ real weights
class FakeEng:
    def __init__(self, ck, dtype=torch.float32):
        from transformers import AutoConfig
        self.M, self.ck, self.dtype = M, D.Ckpt(ck), dtype
        c = AutoConfig.from_pretrained(ck)
        self.cfg = c.text_config
        self.cfg.num_local_experts = self.cfg.n_routed_experts
        self.cfg._attn_implementation = "eager"
        self.G, self.tok_chunk, self.kda_piece, self.kda_tokens = 16, 4096, 256, 512
        self.attn_budget = int(0.5 * 2**30)
        self.scratch = {torch.device("cpu"): D.Scratch(16, self.cfg.hidden_size, self.cfg.moe_intermediate_size,
                                                       dtype, torch.device("cpu"))}


def t_real():
    if not os.path.exists(f"{REAL}/model.safetensors.index.json"):
        report("t_real", False, "real checkpoint missing")
        return
    eng = FakeEng(REAL)
    dev = torch.device("cpu")
    msgs = []
    ok = True
    for L, T, nd in ((0, 1300, 8), (3, 2600, 8)):
        t0 = time.time()
        lay = D.Layer(eng, L, dev)
        NS = 2
        lay.alloc(NS, T + 8)
        g = torch.Generator().manual_seed(L)
        hs = torch.randn(T, eng.cfg.hidden_size, generator=g)
        a = lay.lay.self_attn
        if lay.kda:
            ref = a(hidden_states=hs[None], attention_mask=torch.ones(1, T, dtype=torch.bool))[0]
        else:
            ref = a(hidden_states=hs[None], attention_mask=torch.ones(1, T, dtype=torch.bool), position_ids=None,
                    position_embeddings=None)[0][0]
        P = T - nd
        wv = D.Wave([1], [np.zeros(P, np.int64)])
        o = lay.attn_prefill(hs[:P], wv)
        e_pre = rel(o, ref[:P])
        st = D.StepState(NS, [dev])
        st.pos_h[:] = [0, P]
        de = []
        for t in range(P, T):
            st.push()
            x = torch.stack([torch.zeros_like(hs[0]), hs[t]])
            od = lay.attn_decode(x, st)
            de.append(rel(od[1], ref[t]))
            st.pos_h[1] += 1
        e_dec = max(de)
        good = e_pre < 1e-3 and e_dec < 1e-3
        ok &= good
        extra = ""
        if not lay.kda:
            nvis = T // lay.kp
            extra = f" ({nvis} pools > {lay.nsel}: pruned)"
        msgs.append(f"L{L} {'KDA' if lay.kda else 'DSA'} T={T}{extra}: prefill rel {e_pre:.1e}, decode rel max "
                    f"{e_dec:.1e} [{time.time() - t0:.0f}s]")
        if L == 3:
            x = torch.randn(64, eng.cfg.hidden_size, generator=g) * 0.6
            lay.RI = torch.zeros(1, 64, 8, dtype=torch.int16)
            lay.RW = torch.zeros(1, 64, 8, dtype=torch.float16)
            lay.RX = torch.zeros(1, 64)
            y = lay.mlp(x, (torch.zeros(64, dtype=torch.long), torch.arange(64)))
            _, tw, ti = lay.lay.mlp.gate(x)
            out = torch.zeros(64, eng.cfg.hidden_size)
            for e in ti.unique().tolist():
                r, k = (ti == e).nonzero(as_tuple=True)
                gu, dn = lay.ex.ref(e)
                h = F.linear(x[r], gu)
                gg, uu = h.chunk(2, -1)
                h = F.silu(gg.clamp(max=10)) * uu.clamp(-10, 10)
                out.index_add_(0, r, F.linear(h, dn) * tw[r, k, None])
            yr = out + lay.lay.mlp.shared_experts(x)
            e_moe = rel(y, yr)
            ok &= e_moe < 1e-4 and torch.equal(lay.RI[0].long(), ti)
            msgs.append(f"L3 MoE rel {e_moe:.1e}, records ids exact {torch.equal(lay.RI[0].long(), ti)}")
        del lay
    report("t_real", ok, "; ".join(msgs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-real", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--ck", default=None)
    a = ap.parse_args()
    torch.set_num_threads(min(16, os.cpu_count()))
    os.makedirs(TMP, exist_ok=True)
    u = D.host_unreclaimable_gb()
    assert u < D.HOST_LIMIT_GB, u
    run = lambda n: a.only is None or a.only == n  # noqa: E731
    if run("t_quant"):
        t_quant()
    if a.only == "t_triton_child":
        t_triton_child(a.ck)
        return
    if run("t_tiny") or run("t_driver") or run("t_ep") or run("t_triton"):
        mh = t_tiny("fp8")
        if run("t_tiny"):
            t_tiny("nvfp4")
        if run("t_tiny") or run("t_driver"):
            t_driver(*mh)
        if run("t_ep"):
            t_ep(mh[2])
        if run("t_triton"):
            t_triton(mh[2])
    if run("t_real") and not a.skip_real:
        t_real()
    print(f"peak RSS {D.rss_gb():.1f} GB")
    print("SUMMARY:", {k: v[0] for k, v in RES.items()})
    sys.exit(0 if all(v[0] for v in RES.values()) else 1)


if __name__ == "__main__":
    main()
