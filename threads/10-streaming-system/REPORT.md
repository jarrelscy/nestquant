# Thread 10: streaming system (written by the lead from the agent's final message)

Scripts and raw JSON in this directory: sizing.py, xfer_bench.py, xfer_bench2.py, router_stats.py, predict_next.py.

Correction to BRIEF: `orbit-duet/runs/native_id_control_v1_capture` is a MiMo capture (384 experts), not GLM.

## Sizing
One expert = 37.75 M weights; 1 bpw = 4.72 MB/expert = 576 KiB per TP8 shard.

| | experts | 2 bit | 3 bit | 4 bit | non-expert |
|---|---|---|---|---|---|
| GLM (75 MoE + MTP, 256 experts, top-8) | 19,456 | 184 GB | 275 GB | 367 GB | ~38 GB |
| MiMo (69 MoE, 384 experts, top-8) | 26,496 | 250 GB | 375 GB | 500 GB | ~37 GB |

- 8x A100-80GB: GLM fits at 4 bit. MiMo at 4 bit leaves ~13 GB/GPU for KV and workspace, so it needs a ~3–3.5 bit average mix.
- 8x B200: both fit at 4 bit; streaming is only a speed/quality knob there.
- Host RAM needs only the extra planes: 184 GB (GLM), 250 GB (MiMo).

## Transfer (A100, PCIe gen4 x16, pinned; GPU 3 was 73–99% busy, so conservative)
- ~11 µs per copy. 196 KB: 8.5 GB/s. 576 KiB: 13–16 GB/s. ≥4.5 MB: 16–19 GB/s. 512 x 4 KB copies: 0.37 GB/s. Pageable: 7–14 GB/s.
- Triton zero-copy gather through a device offset table: 8–16 GB/s even at 4 KB granularity; works inside a captured CUDA graph with the table changed after capture. Uses SMs, so fallback only.
- Device-to-device copy: 1.5–1.7 TB/s. GPU pairs share a PCIe switch, so plan 12–17 GB/s per GPU with all 8 streaming.
- Not measured: NVLink peer (~230–270 GB/s spec), B200 host link (~50 GB/s spec).

## Time budget
GLM-5.2 on B200 TP8 + MTP: ~27 ms per verify step, ~0.35 ms per layer. On A100 TP8 that allows ~0.4 GB/GPU/step, ~340 full 2→4 upgrades per step or ~4 per layer hidden behind one layer. With whole experts per GPU (EP), one upgrade takes 0.55 ms, longer than a layer.

## Router statistics
- Skew: top 25% of experts take 49–56% of routes (GLM), ~61% (MiMo L65–67); hottest expert 7–21x the mean.
- A static hot set picked on control text catches only 29–34% of GLM OOD routes (71–80% if picked in-domain).
- Single-stream recency: the next token reuses 28–43% of experts (3% random). A 4-token MTP window touches ~21 distinct experts per layer. Keeping the most recently used 25% at 4 bit catches 70–83% of routes vs 44–50% for a static set.
- Concurrency kills recency (MiMo L66, 96 experts at 4 bit):

| streams | routes already at 4 bit | upgrades per layer per step |
|---|---|---|
| 1 | 82% | 1.4 |
| 4 | 72% | 7.8 (exceeds link) |
| 16 | 52% | 40 |
| 64 | 0% | 188 |

- Next-layer prediction (MiMo, layer L input into layer L+1 router): top-8 predicted holds 72–79% of the true 8; top-16 90–95%; top-32 98–99%. Two layers ahead, top-16 holds 83%.

## Policy
- Streaming unit = one (layer, expert, plane, GPU shard). Per-16-column-block rates are fixed at fit time.
- Static 4-bit set chosen by benefit per byte to fit the memory budget; for 1–4 streams add recency-based upgrades with one step lag; optional next-layer prefetch of top 12–16 under TP only.
- Downgrade instantly under KV pressure by freeing planes. Cap bytes per step at link rate; disable upgrades when a step touches more than ~half the experts.

## Format requirements
1. Separate planes: base (2 bit) + P3 + P4 (~1 bit each). Each level reads only its planes. The 4-bit decoder may reinterpret base bits in place; the base is never rewritten and always decodes alone.
2. Strict nesting: 3 = base+P3, 4 = base+P3+P4.
3. One contiguous chunk per (layer, expert, plane, shard): 576 KiB at TP8, 1.15 MiB at TP4, 4.72 MB under EP. 64 KiB-aligned chunks, 4 KiB-aligned sub-sections. File keeps each shard's experts together for direct load into a GPU-local pinned buffer.
4. Nothing crosses a TP shard boundary (tiles, trellis state, shared bits, offset tables). vLLM splits gate/up along output rows and down along input columns (2048/TP). Down's input rotation must be ≤256 wide at TP8 or it forces an extra all-gather. **This constrains the codec threads.**
5. Constant bytes per (layer, plane, shard) across experts, so the GPU uses a pool of identical slots. Variable per-block rate only inside that total, via a ~1.5 KB offset table in the chunk header.
6. Level-specific metadata (scales, LUT selectors, rate tables) travels with its plane. Shared codebooks/LUTs stay resident.
7. Identical tiling, launch shape and output layout at 2/3/4 bit; only the inner decode branches per expert, so one launch serves mixed levels.
8. CUDA graphs: per-layer device tables of level, base pointer and plane pointers read at replay. All slots preallocated. Upgrade: side-stream copy, event, main-stream wait, flip level. Downgrade: flip level first, reuse the slot after a main-stream event.
9. Artifact carries per (layer, expert, plane) calibration error reduction, routing mass and bytes for benefit-per-byte ranking, plus default allocations for a few budgets.
10. vLLM: orbit-duet's plugin, manifest and graph-safe set-precision bank cover the basics. Missing: non-blocking export of chosen expert IDs (or a GPU-built miss list) to a CPU scheduler, per-GPU pinned buffer with batched copies, per-step byte budget and eviction tied to KV pressure, multi-GPU support.

## Open
NVLink peer and all-8 simultaneous PCIe bandwidth unmeasured. GLM next-layer prediction untested (captures are not adjacent layers). GLM recency based on 5,120 tokens.
