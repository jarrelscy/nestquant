# T37 decode traces: serving GLM-5.3-Flash with vLLM

Prepared on 2026-10-05, around 23:30–23:45 AEDT. All work was CPU-only; no GPU was touched.

## Version choice

**vLLM 0.31.0** (PyPI, released 2026-10-05). It is in `/tmp/venv-t37v`.

- Python 3.12, `torch 2.13.0+cu130`, `triton 3.7.1`, `flashinfer-python 0.7.0.post1`, `tilelang 0.1.12`, `transformers 5.17.0`, `nvidia-nccl-cu13 2.29.7`. DeepGEMM is vendored at `vllm.third_party.deep_gemm`.
- Install recipe:
  `uv venv -p /usr/bin/python3.12 /tmp/venv-t37v && UV_CACHE_DIR=/tmp/uv-cache-t37v nice -n 10 uv pip install -p /tmp/venv-t37v/bin/python --index-url https://pypi.org/simple --extra-index-url https://flashinfer.ai/whl/ --index-strategy unsafe-best-match vllm==0.31.0`
- Model support:
  - `Glm5NextForConditionalGeneration`, `Glm5NextForCausalLM` and `Glm5NextMTPModel` have been in the registry since 0.30.0.
  - The code lives in `vllm/models/glm5next/{common,nvidia,amd}`.
  - The official recipe says "vLLM 0.29.0+" and "supports NVIDIA Hopper and newer GPUs".
- SGLang 0.5.21 also has `glm5_next`, but its cookbook lists H100+ only. It has the same sm_80 blockers (see below), so it is not used.

Verified on CPU (`CUDA_VISIBLE_DEVICES=""`):
- `ModelRegistry.get_supported_archs()` contains the 3 Glm5Next archs.
- `AutoConfig` and vLLM's `get_config` both parse `/tmp/nestquant/37-flash/fp8` as `Glm5NextConfig` / `Glm5NextTextConfig`.
- `EngineArgs(tp=8, max_model_len=32768).create_model_config()` resolves to `Glm5NextForConditionalGeneration` with `quant=fp8`, `bf16`, `is_hybrid=True`, `use_mla=True` and `multimodal=True`.
- All `serve_flash.sh` flags parse with the 0.31.0 CLI parser.
- No weights were loaded.

## Blockers on the A100 box (sm_80): stock vLLM will NOT run this model

| component | status on sm_80 | evidence |
|---|---|---|
| DSA lightning indexer (11 layers, kpool 4, top-2048) | **BLOCKED** | `nvidia/sparse_indexer.py` only calls DeepGEMM `fp8_fp4_mqa_logits` / `fp8_fp4_paged_mqa_logits`, with no torch fallback. `support_deep_gemm()` = sm90/100/120 only. |
| indexer K-cache kpool compress, Q Hadamard+fp8 quant (`nvidia/ops/kpool_compress.py`) | **BLOCKED** | These Triton kernels store `float8_e4m3fn`. A CPU AOT compile with `GPUTarget("cuda", 80)` gives `type fp8e4nv not supported in this architecture` for store, load and dot. sm90 compiles OK. Probe: `/tmp/t37v-src/probe/tri_fp8_sm80.py`. |
| sparse MLA attention with kpool tail | **BLOCKED** | The CUDA sm_80 priority list only contains `FLASH_ATTN_MLA_SPARSE` (sm90), `FLASHMLA_SPARSE` (sm90/100) and `FLASHINFER_MLA_SPARSE_SM90`. kpool-tail support exists only in the FlashInfer sparse backends (sm90/100/120) and ROCm aiter. **Expected first error: no valid attention backend at engine init.** |
| FP8 KV cache | not possible | Triton e4m3 is unavailable on sm_80, and the recipe says Hopper must also run BF16 KV for this model. Use `--kv-cache-dtype auto` (bf16). |
| FP8 block-128 linears + 288-expert MoE | OK (W8A16) | `Fp8Config.get_min_capability()=75`. Linears use `MarlinFP8ScaledMMLinearKernel`; MoE uses `--moe-backend marlin`. The Marlin path applies the SwiGLU clamp (`apply_moe_activation` with `swiglu_limit` 10). |
| KDA linear attention (34 layers) | probably OK, unverified | These are FLA-derived Triton kernels with no fp8, and the TMA paths are guarded. |
| mHC (hc_mult 4, sinkhorn 20) | probably OK, unverified | The TileLang kernels use no T.gemm or TMA (`disable_tma=True`), and there is a torch `forward_native`. They do still JIT on first use. |

SGLang 0.5.21 is blocked the same way. Its kpool indexer calls `deep_gemm.fp8_mqa_logits` / `fp8_paged_mqa_logits`; the TileLang paged path is used only on sm90 with heads not in {32, 64}, and Flash has 32. Its kpool tail is supported only on the fa3/tilelang/trtllm DSA backends.

### Ways forward (pick one; none is done)
1. **b200 box (sm100). Recommended for vLLM traces.**
   - Stock 0.31.0 supports this model natively there: FP8 KV, CUDA graphs, flashinfer sparse MLA and DeepGEMM.
   - `serve_flash.sh` picks `--kv-cache-dtype fp8` automatically on sm100.
   - You need to stage the checkpoint there (306 GiB) or fetch it from HF on that box, and build a matching venv there (no cuda-compat on b200).
   - Standing rules apply: use it only when its GPUs are free, and never kill occupant jobs.
2. **An A100 patch set for vLLM.** Estimated at several days and only testable on GPU. It needs:
   - (a) a torch/bf16 indexer that replaces the two DeepGEMM logits calls. relu(q·k)·w summed over heads, on kpool-compressed keys, then the existing top-k.
   - (b) kpool compress and Q quant rewritten to keep a bf16 (or fp8e5/int8) indexer cache.
   - (c) a sparse-MLA backend for sm80 with kpool tail. This could start from `XPU_MLA_SPARSE` (pure Triton, bf16 KV), but that only handles head size 576. Flash needs 512 (`kv_lora_rank` 512, rope 0), plus the tail tokens.
   - The numerics would deviate a little from the reference: no fp8 Hadamard indexer.
3. **A custom torch decoder, `dec37.py`.** IMPLEMENTED 2026-10-06; this is the A100 path, see "dec37" below. This is how the GLM-5.3 fp8dec traces were made: T33l `threads/33-search/ceiling/dec.py`, which is pure torch with FP8 dequant, a torch DSA indexer and routing recorded inline.
   - `capture37g.py` already runs Flash on the 8 A100s with HF `modeling_glm5_next` layers.
   - HF ships the full reference: the kpool indexer with tail (`Glm5NextTextIndexer`), the torch KDA fallback, and the hybrid `DynamicCache`.
   - This is the most faithful option on A100, and the routing capture would be native.

## Launch (once a supported GPU path exists)

```bash
cd ~/git/nestquant/threads/37-flash/decode
EAGER=1 ./serve_flash.sh          # first bring-up; waits on /tmp/nestquant/33-search/gpu.lock (never overlaps capture)
./serve_flash.sh                  # with CUDA graphs once eager works
# on A100 only after the patch set:  T37_SM80_PATCHED=1 EAGER=1 ./serve_flash.sh
```

- The effective command is:
  `vllm serve /tmp/nestquant/37-flash/fp8 --served-model-name glm53-flash --tensor-parallel-size 8 --max-model-len 32768 --max-num-seqs 64 --gpu-memory-utilization 0.90 --kv-cache-dtype {fp8 on sm100 | auto} --port 8137 --reasoning-parser glm47 --tool-call-parser glm47 --enable-auto-tool-choice --limit-mm-per-prompt '{"image":4,"video":0}' --enable-return-routed-experts [--moe-backend marlin on sm<90] [--enforce-eager]`
- Logs go to `/tmp/nestquant/37-flash/logs/serve_flash.*.log`.
- `VLLM_API_KEY` is unset and HF runs offline.

### Verify first on GPU
1. Engine init log:
   - attention backend chosen for the 11 DSA layers;
   - MoE backend (Marlin on A100, flashinfer/deep_gemm on B200);
   - KV pool size;
   - no DeepGEMM arch error;
   - the weight load is about 38 GiB per GPU at TP8.
2. Parity smoke against the HF reference (the capture37g layers) on a prompt longer than 2048 tokens. Past 2048 the indexer actually prunes; at or below 2048 the sparse path equals dense.
   - Run 1 window, teacher-forced, `max_tokens=1` + `prompt_logprobs`.
   - Compare top-1 agreement and KLD against `cap-txt` `final/` (val top-64 teacher logprobs, CE). Expect top-1 ≥ 0.95.
3. Routing sanity:
   - `routed_experts` in the response has shape `(num_tokens-1, 45, 8)`, dtype uint16 (288 > 256 experts).
   - Rows for the 3 dense layers are zero or unused.
   - Ids are in [0, 288).
   - The per-layer id histogram is roughly consistent with the cap-txt trace.
4. Sampling: use temperature 1.0 / top_p 0.95 (generation_config), as the fp8dec traces did; `reasoning_effort` defaults to max. Throughput at 64 concurrent with eager vs graphs.
5. Thinking boundaries: check that `</think>` and the end tokens are detected. The REAP boundary weighting needs think/end distances.

## Capturing per-token routing (expert ids + weights) during decode

**What GLM-5.3 did** (T33l `ceiling/dec.py`, fp8dec):
- A custom decoder recorded, for every token (prompt and decode) at every sparse layer:
  - the top-8 ids;
  - the combine weights, including `routed_scaling_factor`;
  - |x|^2 of the normalised MoE input;
  - optionally fp16 pre-sigmoid router logits (`rlog.g*.npy`, no `e_score_correction_bias`, with the bias saved separately).
- Predictor labels came from **FP8 teacher forcing**: decode tokens were re-run through the reference forward, and the user accepted int4 decode for the generation itself.
- `fp8dec_prep.py` / `dec2t32.py` turn this into 2048-token windows with a `dec_start` mask.

**What vLLM 0.31 gives for free:** `--enable-return-routed-experts`.
- `RoutedExpertsCapturer` is called from `BaseRouter._select_experts` with the logical ids, before EPLB mapping.
- Each completion returns `routed_experts`, a base64 `.npy` array `[num_tokens-1, num_layers, top_k]`. Decode it with `np.load(io.BytesIO(base64.b64decode(s)))`.
- `routed_experts_prompt_start=N` drops the first N prompt rows.
- **Ids only. No weights, no logits.**
- Monolithic MoE kernels need `supports_routing_replay_capture()`. If the B200 FlashInfer-TRTLLM path refuses, force `--moe-backend triton` or `deep_gemm`.
- Keep MTP off for trace runs.

**Proposed equivalent for Flash.**
- **(A) Preferred, mirrors GLM-5.3.**
  - Use vLLM only to *generate* on-policy text, and keep the returned ids as a cross-check.
  - Then teacher-force the generated sequences through `capture37g.py`'s per-token trace path. It already writes PRIVATE per-token ids/w/xn to `/tmp/nestquant/37-flash/private/trace`.
  - Set the window/segment to the full sequence so the DSA indexer sees the real context. capture37g currently uses ≤2048 windows, so contexts longer than 2048 need the HF indexer path with a cache.
  - This yields the labels exactly as jF was trained on them: FP8 teacher-forced ids, weights, xn and router logits.
- **(B) Weights straight from vLLM.** Extend the capturer with a parallel fp16 buffer.
  - Change the call in `base_router._select_experts` (vllm/model_executor/layers/fused_moe/router/base_router.py ~L296) from `self.capture_fn(topk_ids)` to `self.capture_fn(topk_ids, topk_weights)`.
  - Add `RoutedExpertsCapturer.capture(layer_id, topk_ids, topk_weights=None)`, writing `self.weight_buffer[:n, layer_id] = topk_weights.half()` next to `device_buffer`.
  - Add a matching `snapshot`, and route it through the same aux-output path as an extra `routed_expert_weights` field.
  - Optionally add a third buffer for `router_logits` (fp16, 288 wide, about 26 KB per token per 45 layers), for jF-R-style features.
  - Note: the vLLM `topk_weights` are sigmoid scores renormalised over the top-8 (`norm_topk_prob`). `routed_scaling_factor` (2.5) is applied later in the MoE, so multiply by 2.5 offline to match the T33 schema.
  - Under TP8 the router runs replicated, so rank 0's buffer is complete. The capturer already handles the sequence-parallel all-gather case.

Related: memory `fp8-decode-traces-priority`, `nestquant-floating-predictor-t33`; code `threads/33-search/ceiling/{dec.py,fp8dec/}`, `threads/32-gbdt-sal/fp8dec_prep.py`, `threads/37-flash/capture37g.py`.

Scratch (ephemeral): vLLM 0.30/0.31 sdists, the SGLang 0.5.21 wheel and the probe are in `/tmp/t37v-src`; the uv cache is `/tmp/uv-cache-t37v`.

## dec37: A100 HF-based decoder (the A100 decode-trace path)

Written 2026-10-06, about 00:00–01:00 AEDT. Expert parallelism (`--ep`) and the fused Triton dequant (`--dq`) were added about 01:00 AEDT. CPU-tested only; it has **not yet run on GPU**.

Files (all in `decode/`):
- `dec37.py`: the decoder.
- `run_dec37.sh`: launcher. It takes `flock /tmp/nestquant/33-search/gpu.lock`, refuses to start if host anon+shmem > 1200 GB, and logs to `/tmp/nestquant/37-flash/logs/dec37.<AEDT stamp>.log`.
- `test_dec37_cpu.py`: CPU tests.
- `tf_check37.py`: GPU parity check against capture37g.

### Design
- **Per-layer math is HF.** The modules are `transformers.models.glm5_next` (venv `/tmp/venv-t37g`), built exactly as `capture37g.py` builds them. Only the caches, the batching and the routed-expert execution are ours.
- **Layer pipeline over the 8 GPUs, single process.** Layers are split contiguously, balanced on weight bytes plus per-slot state. One batch of NS slots runs through every stage on each step. See "Expert parallelism" below for how `--ep` keeps all GPUs busy anyway.
- **Weights.**
  - The backbone, dense MLPs, shared experts, embed and lm_head are dequantised to bf16 once and stay resident.
  - Routed experts stay FP8 on the GPU and are dequantised per block of 16 experts into a bf16 scratch. The result is bitwise equal to `capture37g Experts._dq`, and this is self-checked at start-up.
  - `--ckpt` can also point at an **NVFP4** checkpoint. Both ModelOpt (`weight`/`weight_scale`/`weight_scale_2`) and compressed-tensors (`weight_packed`/`weight_global_scale`) are detected per tensor.
- **Dequant kernels (`--dq`).**
  - `torch` (the default) is the 2-pass copy plus in-place scale multiply described above, at about 7 B/weight of HBM traffic.
  - `triton` / `auto` is a fused one-pass kernel at about 3 B/weight. It decodes the e4m3 bits in integer ops, multiplies by the block scale and rounds to bf16 with explicit round-to-nearest-even, so it is bitwise equal to torch.
  - Both are self-checked at start-up against the per-expert reference on every MoE layer and shard. On a mismatch, Triton warns and falls back to torch.
  - `auto` turns Triton on only for bf16 when every device is CUDA and Triton imports.
- **MoE.** HF router (fp32 sigmoid + correction bias, top-8, renormalised, ×2.5), Flash clamps (gate ≤ 10, |up| ≤ 10), fp32 accumulation, plus the shared expert.
- **KDA (34 layers).**
  - Own conv state and fp32 recurrent state per slot.
  - Decode is the exact fp32 recurrent step (TF32 off).
  - Prefill is HF `chunk_kimi_delta_attention` run in 64-multiple pieces that chain the state; this is bitwise the same chunk partition as one full call.
- **DSA/MLA (11 layers): our own absorbed-MLA incremental cache**, chosen over the HF DynamicCache path.
  - Cached: the latent C `[NS,Smax,512]` (Flash has no RoPE part), the indexer pooled keys `[NS,Smax/4,128]` (each kpool-4 pool is computed once, in HF op order) and a 4-token ring for the open pool.
  - Scores are `(q W_uk)·C` and the output is `(p·C) W_uv`.
  - The mask is dense causal while the visible pools number ≤ 512 (length ≤ 2051). Beyond that it is the indexer's top-512 pools plus the always-selected tail, which is exact `Glm5NextTextIndexer` semantics.
  - Prefill uses HF's no-cache top-k width, and masks are filled with `finfo.min`, so tie-breaking matches HF.
- **Continuous batching.** There are NS fixed slots. Free slots are refilled by prefill waves once at least 1/8 of them are free; idle slots run a dummy token.
- **Sampling.** Temperature 1.0, top_p 0.95, thinking on via the chat template (`reasoning_effort` defaults to the template's), stop ids `[154820, 154827, 154829]`, max_new 8192. `--n-samples k` gives k samples per prompt with `group` = prompt id.
- **Routing records.** For every fed row (the prompt prefill plus each generated token except the final unfed one), at every MoE layer, it records:
  - `ids` uint16 `[n,8]`;
  - `w` fp16 (including the ×2.5);
  - `xn` fp32 (Σx² of the MoE input).

  This is the capture37g trace schema. `aux.npz` also stores per-row sampling entropy and log-prob.

### Expert parallelism (`--ep`) instead of micro-batch pipelining
The request was for G≈8 micro-batch groups overlapped across the pipeline stages. I built expert parallelism instead, because overlap does not pay here:
- **Dequant dominates.** Dequant plus the bf16 weight read is about 90% of a step: about 9 B/weight of HBM traffic, or about 43 ms per MoE layer on one GPU.
- **Micro-batches multiply that work.** A micro-batch of 32 tokens × top-8 still hits about 59% of the 288 experts. With G = 8, each stage therefore does about 4.7× the dequant work of one full batch, so the best case is about 1.7× faster than non-overlap. At B = 1024 it is about 1×.
- **Caching the bf16 dequant across groups does not fit.** It would need 14.5 GB per layer × about 5.25 layers per stage, about 76 GB per GPU.

`--ep` gets all 8 GPUs working in every MoE layer, and each expert is still dequantised once per step:
- Every layer's experts are cut into fixed blocks `[kG, (k+1)G)`. The default with `--ep` is G = ceil(288/ndev) = 36, so each device gets one block per layer.
- **Per-GPU memory is unchanged** (about 38 GB of fp8 experts), because each device now holds 1/8 of every layer instead of all of 1/8 of the layers.
- **Per MoE layer:**
  1. Route on the layer device.
  2. Argsort the assignments by expert.
  3. One host sync (`cnt.tolist()`).
  4. Copy x, the order, the weights and the expert offsets to each shard device (NVLink all-to-all, non-blocking).
  5. Each shard runs dequant + bmm on its own block, asynchronously, launched from the one Python thread.
  6. The fp32 per-assignment rows are copied back, un-permuted and summed over K in a fixed order.
- **Bitwise identical to non-EP** for the same `--expert-block`. The defaults for G differ (16 vs 36), so pass `--expert-block` explicitly when comparing.
- The attention, KDA and dense layers stay pipelined. They are not overlapped, but that part is only about 40% of the remaining step.

### Output (PRIVATE, never upload)
The default output is `/tmp/nestquant/37-flash/private/dec_trace`.
- Shards go to `shards/s{K:05d}/` and contain `L{L}.npz`, `tok.npy`, `aux.npz`, `seqs.json` and `gen.jsonl`. Each one is written to a `.part` dir and then renamed.
- The top level holds symlinks in the jF `blocks37.py` layout: `L{L}.r{K}of{W}.npz`, `tok.r{K}of{W}.npy`, `aux.r{K}of{W}.npz` and `seqs.r{K}of{W}.json` = `{"N", "seqs": [{"id", "rows", "prompt_len", "group", "n_gen", "stopped", "stop_id", "think_end"}], "decode": {...}}`.
- `index.json` lists each task's shard, offset and end.

Rows of a sequence are the prompt positions 0..P-1 followed by the generated positions P..P+G-2.

### Commands
```bash
cd ~/git/nestquant/threads/37-flash/decode
# 1. GPU smoke (8 built-in prompts; weight load time unmeasured, 306 GiB read from /tmp):
./run_dec37.sh --smoke --slots 8 --max-new 512 --out /tmp/nestquant/37-flash/private/dec_smoke
# 2. teacher-forced parity against the capture (needs cap-txt final/; see tf_check37.py):
/tmp/venv-t37g/bin/python tf_check37.py build --rank 0 --max-segs 64 --tail 256
./run_dec37.sh --mode tf --tasks /tmp/nestquant/37-flash/private/dec_tf/tasks.json --out /tmp/nestquant/37-flash/private/dec_tf --slots 64
/tmp/venv-t37g/bin/python tf_check37.py compare --rank 0        # prints TF_CHECK PASS/FAIL
# 3. generation (jsonl rows {id, messages | prompt_ids | text, [reasoning_effort], [max_new], [group]}):
./run_dec37.sh --input prompts.jsonl --n-samples 1 --slots 256 --max-new 8192
#    after GPU verification (smoke + tf_check with --ep --dq auto, and one A/B step vs no-ep with --expert-block 36):
./run_dec37.sh --ep --dq auto --input prompts.jsonl --n-samples 1 --slots 512 --max-new 8192
#    stop gracefully: touch /tmp/nestquant/37-flash/private/dec_trace/STOP   (or SIGTERM)
#    finished shards are kept; in-flight sequences are dropped
# 4. rebuild the jF symlink layout (CPU):
/tmp/venv-t37g/bin/python dec37.py --finalize /tmp/nestquant/37-flash/private/dec_trace
# CPU tests (about 11 min, of which t_triton under the Triton interpreter takes about 6; peak RSS 3.9 GB):
OMP_NUM_THREADS=16 nice -n 10 /tmp/venv-t37g/bin/python test_dec37_cpu.py [--skip-real] [--only t_driver|t_ep|t_triton]
```
Image prompts are rejected; the vision tower is not wired in.

### Throughput estimate (unmeasured; revised 2026-10-06 about 01:00 AEDT)
Per-step model at NS 256: per MoE layer the experts total about 7.25G weights; the HBM rate is taken as about 1.5 TB/s.
- **Non-EP:** about 2 s per step, so about **125 tok/s**. The earlier 150–250 tok/s was optimistic: it left out the bf16 re-read in the bmm.
- **`--ep`, torch dequant:** about 455 ms per step, so about **560 tok/s**. The parts:

  | part | time |
  |---|---|
  | experts (42 layers, 1/8 each) | ~230 ms |
  | KDA decode (34 layers; about 6 passes over 4.2 MB fp32 state per slot per layer) | ~140 ms |
  | DSA attend (reads the dense latent cache) | ~30 ms |
  | backbone | ~13 ms |
  | launch gaps / host syncs | ~42 ms |

- **`--ep --dq auto`:** the experts drop to about 128 ms, so about 355 ms per step, about **730 tok/s**.
- **`--ep`, NS 512:** this fits at Smax 16K (per-slot state is 368.5 MB, KDA 166 MB of it). Expert cost is flat in NS, while KDA and DSA double, so it is about **820 tok/s** (torch) and about **1000 tok/s** (Triton).

| config | tok/s | 6 h |
|---|---|---|
| non-EP NS 256 | ~125 | ~2.7M |
| `--ep` NS 256 | ~560 | ~12M |
| `--ep --dq auto` NS 256 | ~730 | ~16M |
| `--ep --dq auto` NS 512 | ~820–1000 | ~18–21M |

Prompt rows come on top of these decode tokens. Once EP is on, the next bottleneck is the KDA recurrent decode step; a fused Triton step kernel would be the next lever.

The memory budget is checked at start-up, and `--reserve-gb` defaults to 8. If memory is short, lower `--slots` or `--smax`.

### Tests (CPU, 2026-10-06 about 00:30 AEDT, rerun about 01:00 AEDT after EP/Triton; all PASS)

| test | result |
|---|---|
| `t_quant` | fp8 block dequant == `_dq` bitwise; nvfp4 round trip and nibble order exact |
| `t_tiny_fp8` / `t_tiny_nvfp4` | random 8-layer Flash (DSA at L3/L7, 16 experts) as fp8/nvfp4 checkpoints, ragged batch with a free slot vs HF full forward: prefill rel 2.5e-6 / 1.3e-6, 40 decode steps rel ≤ 5.7e-6, argmax 84/84, routing id-set mismatches 0 |
| `t_driver` | subprocess runs: greedy == HF argmax 160/160 with routing exact across 2 shards; tf mode top-64 \|Δlp\| ≤ 3.9e-3 (fp16 storage), top-1 identical; `--n-samples 2` groups correct |
| `t_ep` | 4 CPU "devices", expert-block 4 (1 block per device, 3 remote shards). Engine prefill + 30 decode steps: logits and routing bitwise equal for 1 device, 4 devices, and 4 devices with `--ep`. Driver at temp 1 with `--n-samples 2`: all 53 output files (tokens, L*.npz, aux, gen) bitwise equal with and without `--ep` |
| `t_triton` | `TRITON_INTERPRET=1`: the kernel equals torch dequant bitwise on all 254 non-NaN e4m3 bytes (incl. -0 and subnormals) × random scales, 3 shapes. bf16 tiny engine with `--ep`: Triton vs torch dequant gives bitwise-equal logits and routing |
| `t_real` | real Flash weights. L0 KDA (T 1300) prefill rel 0, 8 decode steps rel 1.7e-6. L3 DSA (T 2600, so the indexer actually prunes 650 → 512 pools) prefill rel 5.5e-7, decode rel 1.4e-6. L3 MoE vs the teacher math rel 6e-7, routing ids exact |

### Risks / open items
- **GPU not yet run.** Throughput, memory and the placement plan are unverified. Run the smoke and then `tf_check37` first.
- **`--ep` and `--dq triton` are GPU-unverified and default off.** CUDA cross-device stream ordering for the non-blocking peer copies has only been reasoned about; on CPU the copies are synchronous. Verify with a short A/B run, `--expert-block 36` with and without `--ep`, comparing the outputs bitwise.
- **`--tok-chunk` default is now 4096** (was 8192). The per-assignment fp32 buffer is n·K·D·4 bytes, so large prefill chunks cost memory.
- **The cap-txt `final/` does not exist yet**, so `tf_check37` cannot run until the capture's `final()` stage has written `top64.r{R}.pt`.
- **Pruning is not covered by tf_check37.** Val segments are ≤ 1897 tokens, under the 2048-token pruning threshold, so it does not exercise indexer pruning. That path is covered only by `t_real` on CPU against HF.
- **Top-k ties in the decode indexer.** These are exact score ties, mostly relu-zero. Prefill matches HF's tie-break, but in decode the top-k runs over a width of max-visible pools, so ties may resolve differently from a full forward. This is legitimate (both are valid DSA), but it is not bitwise.
- **Numerics differ from the capture.** `--tf32 all` (default) vs the capture's settings, and the absorbed-MLA bf16 rounding differs from HF's materialised K/V, cause small logit differences. Use `--tf32 none` for the parity run if KL is borderline.
- **NVFP4 is tested on a tiny synthetic checkpoint only.** The ModelOpt and compressed-tensors loaders have not been tried on a real NVFP4 Flash checkpoint.
- **Not supported:** vision; MTP. Micro-batch overlap was deliberately not built (see Expert parallelism). With `--ep`, only the non-expert part of each layer is still serialised across GPUs.
