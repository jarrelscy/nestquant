# Prompt: SM120 NestQuant serving with SSD-streamed residual bits (GLM-5.3, 4x RTX PRO 6000)

You are building the serving side of NestQuant for GLM-5.3 on a box with 4x RTX PRO 6000 Blackwell (SM120, 96 GB each, PCIe, no NVLink), running vLLM at TP4. The goal: keep every routed expert resident at 2 bits, and stream the extra 2 bits (the "P4" residual plane) **from NVMe SSD** into GPU memory for whichever ~30% of experts the model is using right now. Downgrading means dropping those bytes. Quality and speed must both be preserved.

## Why

Today's GLM-5.3 hybrid fixes a hot set (30% NVFP4, chosen offline by REAP) and gets only 26% of routed slots at ~4 bit on tb4. Your earlier analysis showed that a 10% fixed + ~20% floating set, refreshed every 64 tokens from the last 1024 tokens of routing, reaches 65.6% at ~3.2 GB/s aggregate (100 tok/s), using slightly less memory than today. That analysis assumed ARVQ stages 3–4, which haven't been fitted. NestQuant already has the fitted residual. Its 2-bit base is byte-identical whether an expert is cold or hot, so an upgrade only adds bytes, and KV from 2-bit tokens stays valid. At ~4.0 bpw its 4-bit expert output error is 20–25% lower than NVFP4 against FP8, and it reaches EXL3-4 parity at ~4.09–4.125 bpw. Its 2-bit base is ~1% better than EXL3-2. The residual bits sit on the SSD, not in host RAM, so host memory doesn't cap the 4-bit tier.

## Source of truth

Private GitHub repo `jarrelscy/nestquant` (the user's account). Read, in this order:
1. `DESIGN.md`: format, fitting, kernel and streaming summary.
2. `threads/15-level4-decode/REPORT.md`, `ref15_spec.py`, `nqk15.cu`. `ref15_spec.py` is the **bit-exact reference decoder**, and every kernel you write must match it bit for bit. `nqk15.cu` is the fastest single-expert decoder (A100: 4b 38.8–43.8 µs, 2b 26.4–30.7 µs per expert at B1–B4, vs EXL3 bare kernels 53.6–74.4 / 60.1–77.8).
3. `threads/13-moe-layer-kernel/REPORT.md`, `INTEGRATION.md`, `nqmoe.cu`, `moe.py`, `levelswitch_mbox.py`, `copybw.py`. This is the MoE layer kernel:
   - two launches per layer, B·topk ≤ 32;
   - a per-expert table read at CUDA-graph replay (levels 0/2/4, plane pointers);
   - a graph-safe mailbox level switch costing 1.4 µs per layer;
   - a pinned host-mapped `hits` export for the scheduler;
   - measured host→device upgrade copies with 0.0% MoE slowdown.

   INTEGRATION.md is written for TP8; you adapt it to TP4.
4. `threads/10-streaming-system/REPORT.md`: sizing, router statistics and format requirements.
5. `threads/14-level4-floor/`: the pattern-rate residual trellis (per-projection K 1.875 on gate/up and 2.25 on down; step i shifts in KA + bit(i mod 16) of MASK bits). The final artifact may use it, so the decoder must support it.

The quantized model is `jarrelscy/GLM-5.3-NestQuant-2-4bit` on Hugging Face (public). It is being encoded layer by layer now. Until the layers you need exist, use:
- random planes (decode timing doesn't depend on values), and
- a few real experts encoded with `threads/12-reference-encoder/nq_encode.py` for correctness.

The final P4 rate isn't fixed yet (4.0–4.125 bpw). Keep the slot size a parameter.

## Numbers (GLM-5.3 routed experts: H 6144, I 2048, 256 experts × 75 MoE layers + MTP, top-8)

- One expert is 37.75 M weights. 1 bpw = 4.72 MB per expert = 1.125 MiB per TP4 shard (I/4 = 512).
- Base (2.0143 bpw, all metadata included): ~2.27 MiB per TP4 expert-shard, ~183 GB total, ~46 GB per GPU.
- P4 + δ (+ ~2.0–2.1 bpw): ~2.3–2.4 MiB per TP4 expert-shard, ~9.8 MB per expert. The full 4-bit tier on SSD is ~190 GB.
- A 30% upgrade pool is ~57 GB total, ~14 GB per GPU. Non-expert weights are ~38 GB. Size the pool against the KV budget you need (1M context with fp8_ds_mla needs ~54 GB). Report the trade-off and don't guess.
- TP4 sharding: each TP4 rank holds I-rows 512·r..512·r+511, which is two adjacent TP8 shards. All I-side rotations are per 128-wide Hadamard block, so nothing crosses the shard. Confirm the kernel runs at I = 512 (constraint: I a multiple of 128, ≤ 64·128), and retune the tile configs.

## Work items

### A. SM120 kernels (accuracy and speed)
1. Port `nqmoe.cu` with the RM_P decoder from `nqk15.cu`, per-projection fractional K and the pattern-rate trellis to sm_120a. Use mma.sync m16n8k16 (SM120 has no tcgen05/TMEM), keep shared memory under ~99 KB per block, and tune occupancy for 188 SMs and GDDR7.
2. Correctness gate before any timing:
   - bit-exact against `ref15_spec.py` on every plane type, level and K variant;
   - MoE-layer output against the dense reference at max relative error ~1e-4, at both levels and with mixed levels in one launch.
3. Speed gate: beat EXL3 (exllamav3 kernels on SM120) at B1–B4, at both 2 and 4 bit, per expert and per MoE layer at the TP4 shape. Report µs tables in the same format as threads 04, 13 and 15.
4. Prefill (B > 4) needs a dense-decode + grouped-GEMM path. Only a torch reference exists today, so build a real kernel.
5. Native bf16-in and int32-ids variants, to remove the three cast kernels (INTEGRATION.md §3).

### B. SSD streaming engine
1. **On-disk layout.** Write a repack tool that turns the HF artifact into one file per TP4 rank. For each (layer, expert), store that rank's P4 + δ (+ flags) as one contiguous, 4 KiB-aligned record (64 KiB-aligned if it helps), with every record in a layer the same size, plus an offset index. The base planes load into GPU memory at startup and are never streamed. One read per upgrade.
2. **Read path.** Measure options on the real NVMe(s) and pick the fastest:
   - (a) GPUDirect Storage (cuFile) straight into the GPU slot, if the driver and filesystem support it on RTX PRO. Check this; don't assume.
   - (b) io_uring + O_DIRECT into a pinned host bounce ring, then `cudaMemcpyAsync` on a per-GPU side stream.

   Record queue depth, record size, throughput, and p50/p99 latency, with 1 and 4 GPUs streaming at once, and with the model running. Needed: ~3.2 GB/s aggregate at 100 tok/s. Headroom is what matters for concurrency and bursts. If one drive can't sustain it, measure RAID0 or striping records across drives.
3. **Optional host-RAM cache.** An LRU of recently evicted P4 records in pinned RAM, sized by a flag, so re-upgrades skip the SSD. Report the hit rate on the tb4 tasks.
4. **Upgrade protocol.** Reuse thread 13's mailbox:
   - order: read into slot (or bounce, then H2D) → write the stage row → bump `seq`;
   - an expert only goes live after its bytes land;
   - an upgrade that arrives late means the expert stays at level 2 for that step, which is always correct;
   - downgrade: post the level-2 row, and the slot is free once `applied == seq`;
   - at most one outstanding op per expert;
   - levels identical across the 4 ranks: one scheduler decides, and each rank reads its own shard file.
5. **Scheduler.** Implement your 10% fixed + ~20% floating policy:
   - refresh every 64 tokens from the last 1024 tokens of routing, read from the `hits` export;
   - the fixed set is chosen by usage across tasks and loaded at startup;
   - a per-step byte cap at the measured SSD rate;
   - instant downgrade under KV pressure;
   - no upgrades during a step that touches more than ~half the experts (large concurrency).

   Next-layer prefetch is optional; thread 10 measured top-16 at 90–95% recall on MiMo.
6. **Graph safety.** Everything must work under vLLM full CUDA graphs with MTP. The compute stream never waits on I/O, and the MoE step time must not regress while streaming (thread 13 measured 0.0% ± 0.3% on A100 H2D; re-measure on SM120 with SSD reads in flight).

### C. End-to-end validation
1. With the fixed set only, the floating set, and all-4-bit (if memory allows), on the 12 tb4 tasks, report:
   - the measured 4-bit share of routed slots, against your simulation (65.6% single stream; 63→56% at 1→8 concurrent);
   - SSD bytes/s, upgrade latency p50/p99, and the fraction of upgrades that land within one refresh;
   - tok/s at 1, 2, 4 and 8 concurrent, against all-2-bit and against today's NVFP4/ARVQ hybrid.
2. Quality:
   - KLD against the FP8 reference (or top-50 logprob goldens) for all-2, all-4 and dynamic;
   - tb4 task scores for dynamic against today's hybrid.

   Note that output under streaming depends on when upgrades land, so report the variance over repeated runs.
3. Failure behaviour: SSD slow or unavailable → stays at level 2; slot pool full → no upgrade. Neither case may ever produce wrong output.

## Rules
- The GPUs may be busy with MiMo jobs. Only use a GPU that is free, never kill or signal processes you didn't start, and never OOM host RAM or VRAM.
- Push code to `jarrelscy/nestquant` under `sm120/` (kernels, correctness, benches) and `streaming/` (repack, I/O engine, scheduler, vLLM plugin glue). Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never print or commit tokens.
- Don't modify thread 13's or thread 15's files. Work from copies, so the A100 path stays the reference.
- Deliver a one-command bench (`sm120/bench_sm120.sh`) and a one-command serve script, with JSON results committed.
- Report to the user in plain tables: correctness, µs vs EXL3, SSD throughput and latency, 4-bit share, tok/s, KLD. List what's still open.
