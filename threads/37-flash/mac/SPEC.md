# T37 Mac serving spec: GLM-5.3-Flash NestQuant 1.5/4 on a 128 GB M5 Mac (96 GB usable), with layer-major prefill

Status: design, 2026-10-06 00:30 AEDT. No Mac measurements yet.

- **Numbers:** from `sizes37.py` in this directory. It is pure Python and reads only safetensors headers. Output is in /tmp/nestquant/37-flash/mac/sizes37.json.
- **Format formulas:** these re-implement sm120/moe.py and streaming/p4rec.py. The script asserts they reproduce the measured b175 TP4 artifact: rec_bytes 2,854,912, every segment offset, and the per-expert planes of res/rank0/L10.pt.
- **Units:** GB means 1e9 bytes unless it says GiB.
- **Release:** jarrelscy/GLM-5.3-Flash-NestQuant-1.5-4bit. Per expert it has a 1.5-bit base (1, 0xAAAA), a gate/up residual of 2.5 bits (2, 0xAAAA), and a down residual of 2.8125 bits (2, bres 13 = 0xFBDE). The down residual is a new RKP code that the kernels must add.
- **Model:** 42 MoE layers (L3-44) × 288 experts, top-8, with 19 fixed and N floating experts at 4 bit. There are 34 KDA and 11 DSA/MLA layers, all with mHC (hc_mult 4).

## 0. Summary of decisions

1. **Layout:** a single TP1 rank built by a CPU repack of `layers/`, with no re-encode.
   - `mac/rank0.bin` holds the level-4 records. Each record is 8,323,072 B, aligned to 16 KiB. Records are layer-contiguous and expert-addressable, 100.7 GB in total.
   - `mac/rank0.json` is the index.
   - `mac/res/L{L}.safetensors` holds the 1.5-bit resident planes: 4,919,296 B per expert, 59.5 GB in total.
   - The backbone is converted at load from the shipped fp8/bf16 to MLX q8 affine (9.57 GB).
2. **Budget at 96 GiB** (the macOS default GPU wired limit on a 128 GB machine, 75% of RAM; this is the release target):
   - At 32K context: 19 fixed + **63 floating** = 82 hot experts per layer.
   - At 128K context: 59 floating (78 hot).
   - The predictor in `serving/predictor/` is trained for 63 floating; `serving/predictor_nf48/` is for smaller budgets.
   - At a strict 96 GB (1e9) it would be 42 floating at 32K and 39 at 128K (table 1.1).
3. **LMPF on the Mac** is an expert-group-streaming variant of the CUDA design:
   - Every layer runs attention over the window's sub-chunks, then routes the whole window.
   - The MoE runs expert-group-major over the whole window, while groups of 32 records stream through a 3-deep ring of 0.8 GB.
   - Reads go from `pread` with `F_NOCACHE` straight into a shared, page-aligned MTLBuffer (v0), or through `MTLIOCommandQueue` (v1). Neither path does a bounce copy.
   - On the Mac, prefill compute is slow (about 250-600 tok/s) compared with SSD speed (6-13 GB/s). So a full 4-bit pass is hidden behind compute for any window of at least about 4K tokens. **The CUDA 32K cutoff is wrong for the Mac.** Full mode should start at about 3-6K new tokens, computed at boot from the measured P and R. Budget mode covers 1K up to that point, and the plain path covers prompts under 1K.
4. **Prefill chunking, the answer for the jF eval (`J37_PF_CHUNK`):**
   - Attention sub-chunk: 2048 tokens.
   - MoE: whole window, expert-major.
   - Window: 32K by default (16K minimum, 64K at 112 GB).
   - Plain path: prompts under 1024 new tokens run as one chunk with the floating set frozen.
   - Set **J37_PF_CHUNK=1024**, which is the only case where a live floating set serves prefill tokens. The set never changes inside a prefill. One seeded refresh (`step_chunk` + `target_seed`, hm 0) happens at the decode handoff.

## 1. Memory budget

**Per-unit sizes (sizes37.py):**
- Level-4 record, TP1: 8,306,944 B raw, 8,310,784 at 4 KiB alignment, 8,323,072 at 16 KiB alignment.
  - Segments: gu.p4 5,242,880 · gu.d4 32,768 · dn.p4 2,949,124 · dn.d4 16,384 · lr4 65,536.
- Resident 1.5-bit planes per expert, TP1: 4,919,296 B.
  - gu_base 3,145,728 · dn_base 1,572,864 · var 12,288 · sc2+sc4 73,728 · lr 114,688.
- TP8 shard record: 1,069,056 B. TP8 shard resident: 715,264 B.
- One hot slot across all 42 layers costs 349.6 MB.

**Backbone (release headers, 3,532 tensors):**
- As shipped: 15.23 GB (bf16 KDA q/k/v/o, bf16 embed and lm_head, fp8 elsewhere).
- MLX q8 affine (group 64, 8.5 bpw) on every 2-D linear including embed and lm_head: **9.57 GB**. Of that, KDA is 4.98, DSA 1.47, shared 1.12, dense 0.48, embed 0.67, lm_head 0.67, and router/mHC/norms 0.17.
- With embed kept bf16: 10.16 GB.
- Apple GPUs have no fp8 datapath, so "fp8 backbone" becomes MLX q8 on the Mac. q8 affine at group 64 is finer than e4m3 with 128×128 block scales, so it is at least as accurate.
- Vision tower: 1.13 GB bf16, or 0.62 GB in q8.

**State (bf16 KV):**
- KV per token: 11 DSA layers × (512 latent × 2 B + 128 × 2 B indexer key / kpool 4) ≈ 11.97 KB.
- KDA state is a fixed 0.15 GB: 34 × 64 × 128 × 128 fp32, plus conv tails.
- Total: 0.54 GB at 32K and 1.72 GB at 128K.
- A q8 KV cache would save about 0.8 GB at 128K. The MLA latent tolerates it; measure KLD before relying on it.

### 1.1 Budget table at 96 GB (1e9), 19 fixed experts, without MTP

| Item | 32K ctx | 128K ctx | Notes |
|---|---|---|---|
| Routed experts, 1.5-bit resident planes (42 × 288) | 59.50 | 59.50 | res/L*.safetensors |
| Fixed 4-bit residual records (19 × 42) | 6.64 | 6.64 | |
| Backbone, MLX q8 | 9.57 | 9.57 | 15.23 as shipped |
| Vision tower, bf16 | 1.13 | 1.13 | 0.62 in q8 (saves 1.5 slots per layer) |
| MTP | 0 | 0 | off at 96 (see 1.3) |
| KV + KDA/conv state | 0.54 | 1.72 | single sequence |
| MLX activation workspace + Metal command buffers | 1.00 | 1.00 | dedicated. Decode is about 0.2; the 2048-token attention sub-chunk is about 0.45; command buffers are under 64 MB |
| Process runtime (Python, MLX, tokenizer, jF CPU side, GBDT) | 1.50 | 1.50 | dedicated, counted inside the 96 |
| Allocator slack / fragmentation (`mx.set_cache_limit`) | 1.00 | 1.00 | dedicated |
| jF state + prefill routing stash | 0.10 | 0.10 | 256 blocks × 42 × 288 × 12 B = 37 MB, plus model |
| **Floating pool** | **14.68 (42/layer)** | **13.63 (39/layer)** | solved |
| LMPF ring (3 × 32 records) | 0.80 | 0.80 | **borrowed** from the floating pool during prefill |
| LMPF window buffers (W=32K: mHC streams 32 KiB + MoE in/out 24 KiB per token) | 1.88 | 1.88 | **borrowed** (0.94 at W=16K) |
| Spare | 0.34 | 0.21 | |
| **Hot experts per layer** | **61** | **58** | |

**OS slack:** the user's 96 covers everything this process uses. macOS, the window server and other apps live in the remaining 32 GB. The Metal wired cap has to be raised: `sudo sysctl iogpu.wired_limit_mb=…` sets the wired GPU limit on recent macOS, and the default `recommendedMaxWorkingSetSize` is about 75% of RAM, so about 96 GiB on a 128 GB machine. Without that change, 96 GiB (= 103.1 GB) is the most that can run. The runtime should read `MTLDevice.recommendedMaxWorkingSetSize` at boot and size the floating pool from whichever is lower: that value or the configured budget.

**Borrowed vs dedicated:**
- Only the LMPF ring and window buffers are borrowed. They are plain byte ranges in the floating-pool MTLBuffer that our own kernels address.
- MLX-managed activations cannot live inside a borrowed range, so they are dedicated.
- Borrowing 2.7 GB at W=32K is 0.8 + 1.9 GB, which is 7.7 slots per layer, about 18% of the floating pool. The CUDA cap is 50%.
- After prefill, the borrowed slots are refilled from SSD in jF order, as in CUDA `bw_return`. That reads 2.7 GB, about 0.27 s at 10 GB/s, and overlaps the first decode tokens.
- In full mode the ring has just passed every expert of the last layers through memory. Most of the refill is a GPU blit from the ring (section 4), not an SSD read.

### 1.2 Variants

| Budget | Ctx | Floating / hot per layer | Notes |
|---|---|---|---|
| 96 GB | 32K / 128K | 42 / 61, 39 / 58 | table above |
| 96 GiB (103.1 GB) | 32K / 128K | 63 / 82, 59 / 78 | the default wired limit, if the user meant GiB |
| 104 GB | 32K / 128K | 65 / 84, 62 / 81 | |
| 104 GiB | 32K / 128K | 87 / 106, 84 / 103 | |
| 112 GB, MTP on | 32K / 128K | 76 / 95, 73 / 92 | MTP experts as MLX q4 (4.08 GB) + MTP non-expert q8 (0.20 GB) |
| 112 GiB, MTP on | 32K / 128K | 100 / 119, 96 / 115 | |

**More floating slots at the same budget:**
- vision in q8 saves 0.5 GB (+1.5 slots per layer);
- a q8 KV cache saves 0.2-0.8 GB;
- the process runtime trimmed to 0.75 GB saves +2 slots.

The release README uses the 96 GiB row (63 / 59 floating), with the runtime overhead above counted.

### 1.3 MTP

- The release ships MTP experts as fp8, 7.25 GB. There is no NestQuant encode of them.
- On the Mac they have to stay resident: streaming 8 × 25 MB per drafted token would cost about 20 ms per draft at 10 GB/s, which wipes out the gain.
- Options:
  - (a) Off at 96 and 104.
  - (b) Requantize to MLX q4 at load (4.08 GB) at 112. Acceptance has to be measured: 4-bit MTP experts change the drafts but not the verification.
  - (c) A future NestQuant encode of L45 at 1.5/4, ≈1.4 GB resident: 288 × 4.92 MB.

## 2. Layer-major prefill (LMPF) on Apple unified memory

### 2.1 Why the Mac numbers flip the CUDA trade-off

**CUDA (sm120 nq_lmpf):**
- Prefill runs at tens of thousands of tok/s and expert reads are slow by comparison. A full 4-bit pass is only worth it for prompts of 32K tokens or more, over 64K windows.
- The 1K-32K range gets a 2 s budget mode.

**Mac estimates (not measured):**
- Prefill FLOPs per token ≈ 2 × 17B active parameters plus 3-6 GFLOP of attention, about 38 GFLOP per token in total.
- Published M5 Max 8-bit MLX prefill on Qwen3-Coder-Next (about 3B active) is 754-1887 tok/s. Scaled by active parameters, that is **P ≈ 250-600 tok/s** for Flash. The design value is 400.
- SSD rate R:
  - M5 Max: 12.7-13.6 GB/s measured (Apple claims up to 14.5).
  - Base M5: about 6.3 GB/s.
  - M4 Max: about 5.2-7 GB/s.
  - Spec values used here: 6 / 10 / 13 GB/s.
- A full per-layer read is the non-resident records of that layer:
  - 269 × 8.32 MB = 2.24 GB if only the fixed set is resident;
  - 221 × 8.32 MB = 1.84 GB if the floating set (48 at 104 GB) stays resident and is skipped.
  - A whole window pass reads 77-94 GB.
- Reads are fully hidden when per-layer compute W/(45·P) is at least the per-layer read time. That gives the break-even window W* = 45 · P · B_L / R:

| P (tok/s) | R = 6 GB/s | R = 10 | R = 13 |
|---|---|---|---|
| 250 | 4.2K | 2.5K | 1.9K |
| 400 | 6.7K | 4.0K | 3.1K |
| 600 | 10.1K | 6.0K | 4.7K |

The table assumes only the fixed set is resident. With 48 floating resident, every value is about 18% lower.

So the window size on the Mac is driven by memory and SSD energy, not by TTFT. Any W ≥ 16K hides the reads at every grid point.

### 2.2 TTFT estimates

TTFT = prefill time to the first token, single request, design point **P = 400 tok/s**, only the fixed set resident (conservative):

| New tokens | Plain (no LMPF, frozen set) | Budget mode (+2 s) | Full LMPF R=6 | Full R=10 | Full R=13 | Mode (R=10) |
|---|---|---|---|---|---|---|
| 2K | 5.1 s | 7.1 s (x = 125 of 269 per layer) | 16.4 s | 10.0 s | 7.7 s | budget |
| 8K | 20.5 s | 22.5 s | 20.9 s | 20.7 s | 20.7 s | full |
| 32K | 81.9 s | 83.9 s | 82.3 s | 82.1 s | 82.1 s | full |
| 128K | 327.7 s | 329.7 s | 328.1 s | 327.9 s | 327.9 s | full |

- **At P = 250:** 2K / 8K / 32K / 128K take 8.2 / 32.8 / 131 / 524 s. Full mode costs only +0.2-0.4 s from 8K up.
- **At P = 600:** 3.4 / 13.7 / 54.6 / 218.5 s. Full mode at 8K costs +3.3 s at R=6 and +0.2 s at R=10.
- **Window size** (16K / 32K / 64K) changes the total reads, not TTFT. At 128K that is 752 / 376 / 188 GB. 376 GB over 328 s is a 12% SSD duty cycle at 10 GB/s.
- **Prompts over 128K** are dominated by compute. LMPF costs the same 0.2 s overhead.

**Mode rule.** It replaces CUDA `decide()`; the inputs are measured at boot:
- **Full** if new ≥ bud_min and the predicted exposed read time `prime + Σ_windows 42 · max(0, t_r − t_c(W))` ≤ budget_s (2 s). At P=400 and R=10 that is new ≥ about 4-5K tokens.
- **Budget** if 1024 ≤ new < that point.
- **Plain** below 1024 new tokens.
- P and R are EMAs updated from each request, as with the CUDA `rate`.
- The CUDA 32K threshold must not be ported. On the Mac it would push 4K-32K prompts, the usual agent tool-result size, onto the lower-quality budget path for no TTFT gain.

**Budget mode on the Mac:**
- Selection needs layer L's router counts, so reads can only overlap layer L's own MoE compute, which is about half the layer.
- x(L) = (0.5 · t_c(L) + budget_s/42) · R / rec_bytes experts, taken in descending window-count order (CUDA `bud_select`). That is 125 of 269 at 2K, P400, R10.
- Groups stream in count order, and the MoE for each group starts as soon as it lands. The heaviest experts are computed first, so a late group never stalls the earlier ones.
- Measured on CUDA, budget mode recovered most of the gap: KLD 0.0570 → 0.0224 at 2K, against 0.0138 for an all-4-bit prefill.

### 2.3 Loop structure: expert-group streaming

One window of W tokens (W a multiple of 2048):

```
for L in 0..44:
  for each 2048-token sub-chunk s of the window:            # attention part, MLX + custom kernels
    x = hc_pre_attn(h[s])                                    # mHC: 4 streams -> 1
    a = KDA(x, kda_state[L]) | DSA(x, kv_cache[L], indexer[L])  # state carried across sub-chunks and windows
    h[s] = hc_post_attn(h[s], a)
    m[s] = hc_pre_ffn(h[s]); router(m[s]) -> topk ids/weights; hist[L] += counts
  if L in MoE layers:
    order = full: every non-resident expert in file order | budget: top-x by hist[L]
    for g in groups(order, 32):                              # 3-deep ring, about 0.8 GB
      wait(ring slot of g)                                   # issued 2 groups ahead
      grouped GEMM over the window tokens routed to g (gather, 4-bit decode, SwiGLU clamp 10, scatter-add into y fp32)
      [last window] blit records in jF target T_L from the slot into floating slots (section 4)
      signal slot free -> issue g+3 (or the next layer's first groups once this layer's list is done)
    resident experts (fixed + level-4 floating + level-2 base) run in the same grouped pass, with no reads
    y += shared_expert(m); h = hc_post_ffn(h, y)
```

**Why it is expert-major over the window, not per sub-chunk as on CUDA:**
- The ring then only needs to hold 3 groups (0.8 GB), not 2 × 269 records (4.48 GB, the CUDA double layer ring).
- Every streamed expert is decoded once per window against all its tokens. That gives the best ALU amortisation for the tile-decode GEMM.
- The cost is the window-wide MoE input and output buffers: 8 + 16 KB per token, 0.75 GB at 32K, borrowed.

**Read issue order:**
- In full mode the list is the next unread records in file order across layer boundaries.
- So while layer L's MoE consumes groups, layer L+1's first groups are already landing, and they are in flight during L+1's attention. That covers the "prime" except at the very first MoE layer, L3, which is prefetched during dense L0-2.
- Reads within a layer are an almost sequential scan of one layer region, a 2.2 GB stretch of rank0.bin with resident experts skipped. That is the best SSD access pattern.

### 2.4 SSD to buffer with no bounce copy

All of these land the bytes directly in GPU-visible memory. Apple-silicon unified memory has no separate VRAM, so a shared-mode MTLBuffer is the destination.

**v0: `pread` with `F_NOCACHE` into a page-aligned shared buffer.**
- The pool is allocated with `mmap(MAP_ANON)` or `vm_allocate`. It is wrapped once with `makeBuffer(bytesNoCopy:length:options:.storageModeShared)`; Apple requires a page-aligned pointer and a length that is a multiple of the page size, and the Apple-silicon page size is 16 KiB.
- Alternatively, take MLX's own buffer pointer for an `mx.array` pool.
- The file is opened once with `fcntl(fd, F_NOCACHE, 1)` and `fcntl(fd, F_RDAHEAD, 0)`. The fcntl(2) man page: F_NOCACHE "turns data caching off/on"; F_RDAHEAD with 0 "disables read ahead".
- I/O size is 8 MB, one record. Queue depth is 4-8 using a small pthread pool or `dispatch_io`.
- Records are 16 KiB aligned in the file, which is why the layout pads to 16 KiB. Each read is then a whole number of pages, with page-aligned offsets.
- Completion: the thread writes an `MTLSharedEvent` value, and the GPU waits on it with `encodeWait(for:value:)` before the group's GEMM.

**v1: `MTLIOCommandQueue` (Metal 3 fast resource loading, WWDC22 10104).**
- `device.makeIOFileHandle(url:)`, then `ioCommandBuffer.load(buffer, offset:, size:, sourceHandle:, sourceHandleOffset:)`. One load per record, or one per contiguous run of records.
- `encodeSignalEvent` sets an `MTLSharedEvent` that the compute command buffer waits on, so there is no CPU round trip.
- Use a concurrent-type IO queue at high priority for decode refills and normal priority for LMPF streams.
- Store uncompressed. Apple's IO compression would cost GPU/CPU decode, and the NestQuant planes are near-incompressible codes.

**Not mmap.** Page faults on first touch run at 16 KiB granularity, and the unified page cache would double-count. Every streamed byte would also stay in the page cache and compete with the 96 GB budget.

**Boot-time loads** of the backbone and resident planes also use `F_NOCACHE` reads into their final buffers, so the 75 GB boot does not fill the page cache. With safetensors, read the header, then `pread` each tensor into an MLX array's buffer. Do not use `mx.load` mmap for the 59.5 GB planes.

**Residency:** put the pool, ring and resident planes in one `MTLResidencySet` (macOS 15+), attached to the queue. That avoids per-encoder `useResource` and keeps the wired set explicit.

### 2.5 State across windows

- **KDA (34 layers):** the per-layer recurrent state, 64 × 128 × 128 fp32 (4 MiB), and the conv tails (last 3 inputs of q/k/v) persist from window to window and from sub-chunk to sub-chunk. That is 0.15 GB total, dedicated.
  - Chunked prefill uses the WY/UT form with chunk 64. Windows and sub-chunks are multiples of 64, so no partial chunk ever crosses a boundary.
  - This is exactly the decode recurrence, so decode resumes from the same state.
- **DSA (11 layers):** the KV cache (latent 512 + kpool indexer keys) for the whole prompt is allocated up front, as the decode KV cache, and filled layer by layer.
  - In window w, layer L reads keys from windows 0..w of layer L only. Under layer-major order those are complete when window w reaches L, because earlier windows finished every layer.
  - Indexer kpool groups of 4 tokens do not straddle boundaries, since W is a multiple of 4. `always_select_tail` covers the open group at the window tail exactly as at decode.
  - Each layer has its own indexer, as does MTP L45. There is **no cross-layer topk stash**, unlike GLM-5.3, which removes the CUDA `stash` machinery.
- **mHC:** the 4 hidden streams for the window, 4 × 4096 bf16 per token, live in the window buffer (borrowed). They are carried layer to layer, and only the window's final h goes to the final norm and lm_head (the last token only).
- **Vision:** image tokens are embedded by the vision tower before window 0. Image embeddings are ordinary rows of h.

### 2.6 Decode resume

1. **Before the last window ends, per MoE layer L:** after L's router has run on the last window, fold the last up to 256 16-token blocks of prompt routing into jF for layer L (`step_chunk` on the per-layer state slice) and compute the seed target T_L (`target_seed`, hm 0). This is CPU work of a few ms per layer and overlaps the layer's MoE.
2. **During L's streamed MoE:** any record in T_L that passes through the ring is blitted into one of layer L's non-borrowed floating slots before its ring slot is released. A GPU blit of 8.3 MB takes about 20 µs. In full mode T_L ⊂ streamed ∪ resident, so the whole seed costs **zero extra SSD reads**. In budget mode, T_L members outside the top-x are queued for step 3.
3. **After the last layer:** return the borrowed ranges (ring and window buffers) to the pool, which is 7.7 slots per layer at W=32K. Refill them from SSD in jF target order (score descending, layers interleaved) at high IO priority.
   - That is 2.7 GB, about 0.27 s at 10 GB/s, overlapped with decode.
   - Slots not yet refilled run at level 2. The table row flips to level 4 when its read completes, through the MTLSharedEvent and a host table update between decode steps, matching the CUDA refill watcher.
   - A failed read leaves the slot at level 2.
4. **Decode** continues with `step()` every token and a refresh every 16 tokens, with no history gap.

## 3. Release file layout

**What ships today.** The GLM-5.3 2-4bit layout is `layers/L{L}/tp{0..7}.safetensors` plus a manifest. The b175 serve-ready repack adds `rank{r}.json`/`.bin` and `res/rank{r}/L.pt` per TP rank: the nq-p4rec-v1 records and nq-res-v2 torch pickles, with an artifact_stamp.

**Compared with what the Mac needs:**

| | TP8 shards (as uploaded) | TP4 serve repack (b175 style) | **Mac TP1 repack (recommended)** |
|---|---|---|---|
| Record unit | 1.07 MB per expert-shard | 2.85 MB (b175) | **8.32 MB per whole expert** |
| Decode read per miss | 8 reads (one per shard) | n/a on Mac | **1 read** |
| LMPF stream | 8 files interleaved | n/a | **one sequential range per layer** |
| Resident planes | safetensors per shard | torch pickle (.pt) | **safetensors per layer, MLX-native** |
| Size | ≈ the same bytes | | rank0.bin 100.68 GB + res 59.50 GB + backbone 15.2 + vision 1.1 ≈ **176.5 GB** |

**Recommended Mac layout (`mac/`):**
- **`rank0.bin`:** record (L, E) sits at `((L-3)·288 + E) · 8,323,072`.
  - Segments are gu.p4 @0, gu.d4 @5,242,880, dn.p4 @5,275,648, dn.d4 @8,225,024, lr4 @8,241,408.
  - All are 256 B aligned, the same SEG_ALIGN as p4rec v1. Only the record alignment changes, from 4 KiB to 16 KiB, so offsets and reads are whole Apple pages.
- **`rank0.json`:** {format "nq-p4rec-v1", align 16384, tp 1, rank 0, L0 3, NE 288, rec_bytes, seg, per-layer fixed set, floating default, rg/rd per expert}.
- **`res/L{L}.safetensors`:** gu_base [288, …] u32, gu_var u8, dn_base, dn_var, sc2/sc4 fp16, lr fp16, plus the rk_gu, rk_dn, rg, rd, bk_gu and bk_dn small int tensors. This is nq-res-v2 content in safetensors instead of pickle.
- **`artifact_stamp`:** source layer manifest sha256s and the repack git sha.

**Buildable from `layers/` alone, with no re-encode.**
- `nqload.group_art` already merges the 8 TP shards of an expert:
  - down: V concatenated along I, U2/U4 from shard 0;
  - gate/up: V from shard 0, U2/U4 concatenated along I;
  - units shard-major.
- `RankLayer(root, L, rank=0, tp=1)` therefore yields a self-contained I=2048 expert.
- streaming/repack.py needs three edits:
  - NE 288 instead of the hard-coded 256;
  - ALIGN as a parameter (16384);
  - a CPU path with the CUDA build stubbed, as was done for b175.
- It also needs safetensors output for res.

Cost: a single CPU pass that reads 160 GB and writes 160 GB. Run it on the Mac, one layer at a time, from the HF `layers/` tree. Peak scratch is one layer's 8 shards ≈ 3.8 GB, so the Mac download is about 160 GB plus 176 GB of output, or roughly 180 GB if each layer's shards are deleted after conversion.

**Recommendation:**
- Do **not** upload a second 160 GB copy to HF. Quota is tight; see the b175 memory.
- Ship `tools/mac_repack.py`, pure numpy plus safetensors, so it runs on macOS without torch-CUDA, together with its expected sha256 per output file.
- If disk on the Mac is the constraint, the converter can stream: download layer L, convert, delete.

**The record order within a layer** is the expert id. Do not reorder by popularity. Full-mode reads are a near-sequential scan anyway, and expert-id order keeps the address formula trivial.

## 4. Where jF plugs in

**Decode.** This is unchanged from CUDA/Spark.
- `step(counts[42,288], ntok=1, …)` runs every token. A refresh happens every 16 tokens, giving the floating target at hm 0.7.
- Swap reads use the decode IO queue at high priority. At 25 tok/s they run at about 0.76 GB/s, about 8% of the SSD.
- T36 found that landing delay barely matters and the predictor is the limit, so the Mac does not need a special low-latency path. Keep `order_score` ordering and the k=0.25 top-up.

**Prefill, and the reconciliation with the jF port agent's `step_chunk` proposal:**
- **Full mode** (new tokens ≥ about 4-5K): every routed expert is 4-bit while its window is processed. The floating set is irrelevant to prefill quality, so no per-chunk refresh is needed and none is done.
- **Budget mode** (1K to about 4-5K): ring selection uses the window's actual router counts for that layer. That is oracle information and strictly better than any prediction. The floating set is frozen during the prefill: the executor is paused, and its slots are partly lent to the ring. A per-chunk refresh would act on slots that are not available.
- **Plain path** (< 1024 new tokens, the common multi-turn follow-up): one chunk with the floating set frozen at the predictor's current state. That state is warm from the previous turn's decode, or from the previous prompt's handoff seed. **This is the only case where jF's set serves prefill tokens. Its chunk is at most 1024, so `J37_PF_CHUNK=1024`** replaces 512 in eval37 `prefill_cK`.
- **Handoff, in every mode with ≥ 16 prompt tokens:** `step_chunk` over the last ≤ 256 blocks (4096 tokens; the longest EMA horizon is 2048), then a single `target_seed()`, the refresh at hm 0. The port agent has implemented this as the handoff seeder in gpu_predictor37.py and joint_predictor37.py.
  - The Mac calls it **per layer**, as soon as that layer's router has run on the final window (2.6 step 1). That is valid because every state tensor is per (layer, expert), the feature normalisation is per layer, and the residual net is a transformer over the 288 experts of one layer.
  - `_pos` is global. It is set to the full prompt's block count, not 256.
  - The Metal/MLX port should expose `step_chunk_layer(L, counts[nb,288], sal[nb,288])`. The numpy version can loop over layers with a slice view. Parity is required against the all-layer call.
- **Routing stash:** counts and sal per 16-token block for the last 256 blocks of every layer, a ring of 37 MB. The router kernel writes it directly, using histogram per block.

**What the 20% prefill rows in training are for.** They are not used to pick experts for prefill tokens. They teach the predictor what a state built from prompt routing looks like:
1. the seed refresh after `step_chunk` at the handoff;
2. the first decode refreshes, while EMAs are still dominated by prompt blocks;
3. plain-path follow-up prompts processed under the decode-trained set.

Model selection uses decode val chains, which is correct. **Requested eval, in place of prefill_c512:** on chains whose prompt rows precede decode rows, measure decode sal-hot over the first 1, 4 and 16 blocks after the handoff for two arms:
- seeded: `step_chunk` over the prompt tail, then `target_seed`;
- cold: `floating_default`.

Also evaluate `prefill_c1024` for the plain path. Write the result into jf/SERVE.md, which does not exist yet; the jF port agent owns it.

## 5. Metal kernel and runtime work list

MLX already provides, and should be used as-is:
- q8/q4 affine `quantized_matmul` (backbone, vision, MTP q4), RMSNorm, RoPE (vision only: the text model is NoPE), SDPA (vision);
- `gather_mm` / `gather_qmm` (grouped expert GEMM pattern for q4 affine, a reference for ours);
- the `mx.fast.metal_kernel` custom-kernel path, `mx.compile`, safetensors I/O, and the allocator cache limit.

mlx-vlm has `glm5_next` (PR #2030, merged 2026-08-26, unreleased) with a fused KDA decode kernel (#2105) and batch-invariant sparse attention (#2245).

**Known porting bugs to avoid**, found by PipeNetwork/glm53-flash-mlx:
- swiglu_limit ignored;
- mHC base/scale cast to bf16 (must stay fp32);
- epsilons;
- a bf16 router (must be fp32).

| # | Kernel / component | Status | Notes |
|---|---|---|---|
| K1 | **NestQuant decode GEMV (decode)**: base bk (1, 0xAAAA); residual rk 2 (2, 0xAAAA) and **new rk 10 (2, 0xFBDE)**; d4 Mb\|N; var signs; sc2/sc4; Had-128 both sides; low-rank lr/lr4; levels 2 and 4 from the per-(L,E) table row (offsets into pool arrays, not raw pointers) | new | port of the sm120 moe.py/CUDA kernel; the split_bits paths uint4/uint/ushort/tail (180 = 128+32+16+4) must all be covered |
| K2 | **NestQuant tile-decode + simdgroup-matrix GEMM (prefill, expert-major)**: decode one 128-column block into threadgroup memory, then MMA against the gathered tokens; SwiGLU clamp 10, routed_scaling 2.5, scatter-add fp32 | new | the prefill hot loop. M5 neural accelerators through Metal 4 tensor ops where available, with a simdgroup_matrix fallback for M4 |
| K3 | FWHT-128 (activation side), fused into K1/K2 prologue and epilogue | new | |
| K4 | KDA chunked prefill, vector gate (per-channel decay, lower bound −5), chunk 64 WY/UT form, + short conv 4 + o_norm/gating | partial | MLX PR #4020 has scalar-gate gated-delta only (and a Kimi vector-gate bug); mlx-lm #1870 is training-only with no vector gate. We need the vector-gate version |
| K5 | KDA decode step + conv tail | exists | mlx-vlm #2105; verify the gate dtype |
| K6 | DSA indexer: wq_b/wk, weights_proj, kpool-4 compress (gate + ape), always-select-tail, top-2048 | partial | oMLX #3985 (tensor-unit indexer), llama.cpp #27754/#27752 |
| K7 | Sparse absorbed NoPE MLA (latent 512, qk 256, v 256, 64 heads) prefill + decode | partial | oMLX #3986: 42-54 ms per layer per 2K chunk at 10-13 TFLOPS |
| K8 | mHC sinkhorn (20 iterations, fp32) pre/post for attention and FFN | small | must stay fp32 |
| K9 | Router: fp32 sigmoid + correction bias, noaux_tc top-8, plus a fused per-16-token-block histogram into the jF stash and per-window counts for budget selection | new | |
| K10 | Grouped-token gather/scatter for expert-major windows (sort the window's (token, slot) pairs by expert once per layer) | new | |
| K11 | IO engine: pread/F_NOCACHE thread pool (v0), MTLIOCommandQueue (v1), MTLSharedEvent sync, ring/slot table, borrow/return, refill watcher | new | Swift/ObjC++ extension or a C++ MLX extension |
| K12 | jF on Mac: per-layer state update (elementwise), GBDT v2 score (LightGBM CPU, ~12k rows per refresh) + 2-layer transformer over 288 × 22 (MLX or CPU), and `step_chunk_layer` | port | parity against jf/parity37.py |
| K13 | Backbone loader: fp8 e4m3 + 128×128 scale_inv dequant → MLX q8 affine at load; bf16 KDA → q8 | small | one-time at boot, or a cached converted safetensors on local disk |
| K14 | Vision: mlx-vlm glm5_next tower + merger | exists | |
| K15 | MTP (112 GB only): fp8 → q4 experts, draft and verify loop | later | |

The TensorFold decode-step profile (PR #9/#39) is a reference breakdown: 14.5 ms per step, of which MoE 7.5, KDA 2.9, HC 2.1 and sparse MLA 1.7. Our decode adds NestQuant decode cost to the MoE share.

**Decode throughput ceiling.** Weights touched per token: the q8 backbone without embed is 8.9 GB, plus 8 × 42 routed experts at 4.92 MB base + h × 8.31 MB residual. That totals 11.9-12.8 GB per token for a 4-bit hit rate h of 0.5-0.8, giving **≈34-36 tok/s at 70% of 614 GB/s**. Expect about 20-25 tok/s in practice. Keeping KDA at bf16 would add 4.4 GB per token and cost about 27%.

## 6. Risks and open questions

1. **96 GB or 96 GiB?** The difference is about 21 floating slots per layer: 42 vs 63. The default macOS wired limit at 128 GB is about 96 GiB. Confirm with the user and check `recommendedMaxWorkingSetSize` on the target machine.
2. **The prefill rate P is unmeasured.** Every TTFT and cutoff above scales as 1/P.
   - The NestQuant tile decode (K2) could be ALU-bound on the GPU. That would lower P but make LMPF even more hidden.
   - If P is above about 1500 tok/s, the analysis reverts toward CUDA behaviour: a higher full-mode cutoff and larger windows. The mode rule is computed from measured P and R for that reason.
3. **Thermals and SSD power on laptops.** A 128K prompt reads 188-752 GB over about 5 minutes. Reads cause no wear, but MacBook Pro SSD throughput may drop when hot. The design does not depend on it, because reads are hidden whenever W ≥ W*.
4. **Single-request assumption.** All of this assumes one active request, like the CUDA borrow path. Concurrent decode during another request's LMPF needs the CUDA "pause executor" semantics or a smaller borrow.
5. **The 2.8125-bit down residual is a new RKP code (2, 0xFBDE).** CUDA moe.py's RKP table stops at 8, plus 9 = (2, 0xD5AA) for b1.75. Kernels and the repack must add it, or read the rk value per expert from the manifest. Check which code the encoder actually writes.
6. **The README budget is optimistic.** It says 48 floating at 96 GB, against 42 here with 3.6 GB of runtime reserved. jF is trained with n_float 48. Serving 39-42 takes a top-k prefix of the same ranking, which is fine, but re-run the KLD/sal-hot eval at n_float 42.
7. **Expert-major MoE over 32K windows** needs a sort/gather of about 262K (token, slot) pairs per layer. That is cheap on GPU, but the ordering changes fp32 accumulation order versus decode; batch invariance of the results is not guaranteed.
8. **MLX integration of borrowed ranges.** MLX arrays cannot alias sub-ranges of one MTLBuffer without a custom allocator. Ring, window and pool must be addressed only by our kernels (K1, K2, K10, K11), with offsets in table rows. The MLX ops that touch h must be our kernels or copies.
9. **MTLIOCommandQueue throughput at 8 MB loads** has not been characterised against pread/F_NOCACHE. v0 is pread for that reason. Benchmark both on the target machine before choosing v1.
10. **Disk.** The download needs about 160 GB, plus about 177 GB of Mac output (with per-layer streaming, about 180 GB total), on a 128 GB Mac that probably has a 2-4 TB SSD. The HF quota rules out uploading the Mac layout.
11. **MTP** stays off at 96 and 104. Whether a q4 MTP at 112 pays depends on acceptance with 1.5/4 targets; measure it.
12. **mlx-vlm `glm5_next` is unreleased**, and the PipeNetwork port has known fidelity bugs. Run a CPU-reference logit parity check (teacher windows) before trusting KLD numbers taken on the Mac.

## Sources

- SSD: [PetaPixel M5 Max review](https://petapixel.com/2026/03/13/macbook-pro-with-m5-max-review-this-is-better-than-your-desktop/), [Greg Benz M5 review](https://gregbenzphotography.com/photography-reviews/a-photographers-review-of-the-new-m5-macbook-pro/), [Tom's Hardware M5 SSD](https://www.tomshardware.com/laptops/macbooks/m5-macbook-pros-ssd-is-2-5x-faster-on-average-than-last-gen-m4-exceeding-apples-own-claims-m5-achieves-6-000-mb-s-across-both-read-and-write-speeds), [Apple newsroom M5 Pro/Max](https://www.apple.com/newsroom/2026/03/apple-introduces-macbook-pro-with-all-new-m5-pro-and-m5-max/), [MacRumors review](https://www.macrumors.com/review/macbook-pro-m5-pro-m5-max/), [HotHardware M4 Max](https://hothardware.com/reviews/apple-mac-studio-m4-max-vs-mac-mini-m4-pro?page=2), [SlashGear M4 Max](https://www.slashgear.com/1723675/apple-macbook-pro-m4-max-2024-review-specs-pricing-details/)
- Compute: [Apple MLR: LLMs with MLX on M5](https://machinelearning.apple.com/research/exploring-llms-mlx-m5), [arXiv 2607.19438](https://arxiv.org/html/2607.19438v1)
- Metal I/O: [MTLIOCommandQueue](https://developer.apple.com/documentation/metal/mtliocommandqueue), [MTLIOCommandBuffer load](https://developer.apple.com/documentation/metal/mtliocommandbuffer/load(_:offset:size:sourcehandle:sourcehandleoffset:)), [makeIOFileHandle](https://developer.apple.com/documentation/metal/mtldevice/4172893-makeiofilehandle), [WWDC22 10104 fast resource loading](https://developer.apple.com/videos/play/wwdc2022/10104/), [makeBuffer(bytesNoCopy:)](https://developer.apple.com/documentation/metal/mtldevice/makebuffer(bytesnocopy:length:options:deallocator:)), [MTLResidencySet](https://developer.apple.com/documentation/metal/mtlresidencyset), [fcntl(2) F_NOCACHE / F_RDAHEAD](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/fcntl.2.html)
- MLX ecosystem: [PipeNetwork/glm53-flash-mlx](https://github.com/PipeNetwork/glm53-flash-mlx), [HF pipenetwork MLX 4bit](https://huggingface.co/pipenetwork/GLM-5.3-Flash-MLX-4bit), [MLX #4020](https://github.com/ml-explore/mlx/pull/4020), [mlx-lm #1870](https://github.com/ml-explore/mlx-lm/pull/1870), [oMLX #3985](https://github.com/jundot/omlx/pull/3985), [oMLX #3986](https://github.com/jundot/omlx/pull/3986), [TensorFold #9](https://github.com/ashhart/TensorFold/pull/9), [llama.cpp #27754](https://github.com/ggml-org/llama.cpp/pull/27754), [llama.cpp #27752](https://github.com/ggml-org/llama.cpp/pull/27752)
- Internal: origin/main sm120/serve/nq_lmpf.py + nq_lmpf_engine.py (CUDA LMPF, measured KLD and TTFT), streaming/p4rec.py, streaming/repack.py, sm120/moe.py, sm120/nqload.py, origin/spark-b175 spark/README.md, threads/36-spark-land, threads/37-flash/jf
