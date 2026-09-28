# T24: stage-1 capture speed (threads/19-full-capture/capture_fwd_fast.py)

Drop-in for `capture_fwd.py`: same CLI plus `--super-rows` (default 262144), same protocol.json, byte-identical outputs,
so a shard can be resumed by either implementation. For driver.sh, replace `capture_fwd.py` with `capture_fwd_fast.py`.

## Profile of the original (1M-token c512 shard, T19 progress.json, ~50-75 s per MoE layer)
front 11-14 s | moe 17-47 s | write 7-10 s | ckpt ~13 s every 4 layers | load 1.2 s | ~10 s pageable S copies.
Isolated (prof.py, L7, 65,536-row chunk, shared GPU): moe 3.9 s, of which the weight H2D takes 2.3 s. All 256 experts'
FP8 weights are re-copied synchronously for every chunk: 9 chunks per 1M tokens, 87 GB of PCIe per layer. Dequant is
0.7 s and is also redone per chunk; the GEMMs take 0.8 s and the scatter 0.6 s. Pageable S H2D and D2H take
0.41 s each per 65k rows, and pageable acts D2H takes 0.49 s (pinned: 0.05-0.1 s). front is ~5 ms/window and partly
launch-bound.
Not bitwise-safe, so left alone: batching windows in front (the router fp32 GEMM changes p even at B=2), and changing
the 4096-row expert batches (expert GEMM rows depend on M).

## Changes (all arithmetic-preserving)
super-chunks (experts loaded and dequantised once per super-chunk; each original chunk keeps its own argsort and
batches), double-buffered per-expert H2D prefetch on a side stream, block-view dequant, pinned S with async D2H,
residual parked in pinned S, a pinned acts ring (3 slots) plus a writer thread, a background checkpoint that the next
layer waits on per chunk (progress.json advances only when the file is complete), and segments preloaded to the GPU.

## Validation (c512 fit windows 1280..1535 = 131,072 tokens, layers 0-6, --ckpt-every 4)
acts L3-L6 x/ids/p, done.json, state_0/state_1 (+matched.pt) and protocol are all byte-identical to capture_fwd.py
(`cmp`). Scratch: /tmp/nestquant/24-capture-speed/{orig_c512,fast_c512}.

## Speed (GPU 3, shared with other jobs, so noisy)
| slice 131k tokens, per MoE layer | orig | fast |
|---|---|---|
| seconds | 11.6-17.6 | 4.6-5.2 |
| moe | 5.9-8.6 | 2.0-2.5 |
| write | 0.9-1.1 | 0.0 |
| front | 1.5-2.1 | 1.3-1.7 |
| peak CUDA alloc | 4.19 GB | 5.12 GB |
Extrapolated to 1M tokens per layer: ~27-30 s vs 50-75 s, about 2x per slot. The 1M run is unmeasured. front
(~11 s) is now the largest fixed part.

## Memory per slot at 1M tokens
- VRAM: allocated ≈ orig + 1.6 GB × (super-chunks − 1 extra sub-chunk). With super 262144 this is ~8.3 GB allocated
  (~9 GB in nvidia-smi); with 393216 it is ~9.9 GB (3 expert passes instead of 5).
- Host pinned: S 12.9 GB (pageable before), ExpertCache 9.7 GB (unchanged) and acts ring 4.8 GB (new), 27.4 GB in
  total. RSS is ~40 GB (orig 34.7). Pinned pages cannot be reclaimed, so budget ~40 GB of host RAM per slot.
- Slots per A100 80 GB: VRAM allows ~8, but GPU compute saturates at ~2-3 slots per GPU. Recommend 2-3.
