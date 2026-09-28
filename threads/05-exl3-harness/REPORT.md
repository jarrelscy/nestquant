# Thread 05: EXL3 harness (written by the lead from the agent's final message)

## Verdict
Harness done. Evaluation matches orbit-duet's `initial_e36_evaluation.json` to 0.001 pp. `quantize_exl3_like` regenerates the orbit-duet refits byte-identically (expert_2.bin sha 7b34ac28…, expert_4.bin sha e5434c7a…) and is bit-exact vs exllamav3 at half-bit K=2.5. Two fixes were needed: damping mean via `torch.diag(H).mean()`, and un-rotating H on CPU as exllamav3 does.

## Reproduction, GLM L16 E36 (router-weighted relative output L2 %)
| method | all F | all R | ctrl F | ctrl R | ood F | ood R |
|---|---|---|---|---|---|---|
| EXL3-2 | 40.011 | 37.411 | 38.341 | 34.934 | 42.572 | 42.757 |
| EXL3-4 | 10.457 | 9.736 | 10.004 | 9.058 | 11.152 | 11.181 |
| NVFP4 | 12.779 | 12.287 | 12.818 | 12.074 | 12.714 | 12.781 |
Refits at K=2 and K=4 give identical numbers.

Proxy tr(EHEᵀ)/tr(WHWᵀ) (undamped, original basis), gate/up/down: EXL3-2 .00939/.00971/.01620; EXL3-4 .000563/.000582/.001002; NVFP4 .00466/.00481/.00489. The proxy is a poor stand-in: at 2 bit sqrt(proxy) is ~10–13% but output L2 is 37%; at 4 bit the proxy says EXL3 beats NVFP4 by ~8x, the real gap is 1.2x (SwiGLU amplification plus train/eval mismatch).

## Captures
All GLM numbers come from `orbit-duet/runs/glm53_matched_context_pilot_v1_capture/layer_16.pt` (sha df47efad…, 5120 rows, 139 routed to E36). control = {agentic, code, instruction, medical, prose, reasoning}; ood = {encoded_bytes, fasta, scientific_telemetry, smt_bitvectors}. `native_id_control_v1_capture` and `ood_controlled_v1_capture` are MiMo (expert ids up to 383).

## EXL3 spec (exllamav3 1.5.1)
Weights as (k=in, n=out); `torch.manual_seed(seed)`, orbit-duet seed 91426.
1. H /= count; H += sigma_reg·mean(diag H)·I (default 0.025, orbit-duet 0.03); su = random ±1 per input; H ← P(SHS)Pᵀ with blockwise 128-pt Hadamard; block_ldl(H, 16) (Cholesky, 16x16 diagonal blocks inverted, unit block-lower L, diag zeroed); on failure add 2·sigma·mean(diag), up to 10 retries.
2. Source: gate/up grams[0] (6144²), down grams[1] (2048²); grams = Σ(p·x)(p·x)ᵀ with router weight p; count = 18205 = 1821 routed + 16384 context rows at 25% weight. Derivative "outputs" grams unused.
3. sv = random ±1 per output; output RMS scales only if skew (share of sqrt(diag H) in top 2% channels) < 0.15; 128-pt Hadamard over outputs; input scales su ← su·ics/(−1.24371088)+1e-10, W /= su, 128-pt Hadamard over inputs. Global scale g searched on sample tiles, target × LDLQ drift factor (K=1 1.08, 1.5 1.035, 2 1.018, 2.5 1.009, 3 1.004, else 1.0); coarse 0.1–1.9 step 0.2 on 1/3 subsample, fine ±2×0.075 + parabola. E36 g ≈ 0.78 (2 bit), 0.73 (4 bit).
4. 16x16 tiles (256 weights), `tensor_core_perm` order.
5. Bitshift trellis L=16, V=1: state ← ((state<<K)|b)&0xFFFF; tile wrap by two passes (pass 1 rolled 128, unconstrained, fixes edge; pass 2 constrained). fp16 Viterbi costs.
6. Codebooks: 3inst x=s·89226354+64248484, (x&0x8fff8fff)^0x3b603b60, two fp16 halves; mcg x=s·0xCBAC1FED then same mask; mul1 x=s·0x83DCD12D, value = fp16(1024+bytesum(x))/147.7 − 10.39. Orbit-duet uses mul1.
7. LDLQ: 16-row blocks along k, last block first, spans of 128 (buf_size_k); block input = w + prod_cache + L[bj:,blk]ᵀ·err.
8. Proxy in rotated basis with damped H, before refit.
9. Back-transform then `refit_scales`: output scales c_n = q_nᵀHw_n / q_nᵀHq_n; input scales solve ((QQᵀ)∘H) r = rowsum(Q∘HW); two alternating rounds with un-rotated damped H.
10. Storage: suh, svh fp16; trellis int16 (k/16, n/16, 16K) + mul1 marker. Decode: fp16 Q → input Hadamard → ×suh → output Hadamard → ×svh.
11. bpw: exactly K for the trellis + 16·(in+out) bits per matrix (~+0.0104). Orbit-duet .bin adds a 4096 B header, 256 B keep bits, 12 B markers; 2-bit expert = 9,490,700 B = 2.0113 bpw.

Fractional trellis (1.5.1): K = KA+0.5 alternates KA and KA+1 bit steps (mask 0xAAAA, period 16); tiles stay whole-bit (16K words); own kernel `quantize_tiles_frac` and packer `pack_trellis_frac`; mul1 only. Precedent for per-block rates (rate from step pattern only; same state machine and codebook). Harness is bit-exact at K=2.5 (down proxy 0.009155, 2.5104 bpw) and accepts a per-16-input-block K list (alternating 2/2.5: 2.2604 bpw, proxy 0.01370 vs 0.01854 at 2 and 0.00915 at 2.5).

YAQA (1.5.1, experimental): enabled by `quant_args["H_out"]` (`prepare_H_out`, `ldlq_2hess`); proxy tr(EᵀH_in E H_out); tiles along anti-diagonals from the far corner with running feedback.

Orbit-duet refits used neither: integer K, seed 91426, sigma_reg 0.03, apply_out_scales None, mul1, no H_out ("output_hessian": False in protocol), confirmed by byte-identical reimplementation.

## API
```python
import harness as h; h.gpu_cap(12)
data = h.load_expert(16, 36); data.H(proj)
q = h.load_exl3_bin(path); q = h.load_nvfp4(path, data)
h.proxy_losses(data, q); ev = h.evaluate(data, {"name": q}); h.table(ev)
Wq, info = h.quantize_exl3_like(W, H, K, count=data.count, seed=91426, sigma_reg=0.03,
    codebook="mul1"|"mcg"|"3inst"|<65536 LUT>, quantizer=callable, apply_out_scales=None,
    g_scale=True, g_scale_K=None, refit=True, ldlq=True, buf_size_k=128, fp16_scales=True)
q = h.quantize_expert_exl3_like(data, K, backend="reimpl"|"upstream", **knobs)
h.write_exl3_bin(path, infos); h.free_scratch()
h.ExtTileQuantizer(codebook); h.TorchTileQuantizer(lut); h.codebook_lut(name)
```
CLI: `./run.sh eval --layer L --expert E [...]`, `./run.sh fit --layer L --expert E --K 2 [...]`, `./run.sh selftest`.

## Cost
eval 14 s; fit+eval 12 s (K=2) / 9 s (K=4) reimpl, ~26 s upstream; PyTorch Viterbi (custom codebooks) ~80 s on down. GPU peak <3 GB (CUDA quantizer), ~10 GB (PyTorch Viterbi). exllamav3 caches Viterbi scratch per (K, codebook): 2 GB at K=2, 1 GB K=3, 0.5 GB K=4; call `free_scratch()` in sweeps. Selftest: codebook tables match kernel at K=2,4; PyTorch Viterbi within 0.1%. K=2 down proxies: mul1 .018538, mcg .018568, 3inst .018662, random Gaussian LUT .018527 (codebook choice barely matters). Env: preloads lib/libcudart.so.12 (wheel is cu128, torch is cu130).

## Recommendation
Judge on `evaluate()` over the matched capture, not the proxy. Use `quantize_exl3_like` defaults as the EXL3 control. Per-block rates via K lists including half-bit. MiMo not wired (stats at `orbit-duet/runs/full55_statistics/l55_e70.pt`, older capture format).
