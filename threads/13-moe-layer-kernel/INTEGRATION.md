# Plugging the NestQuant MoE layer kernel into vLLM (GLM-5.3, TP8)

vLLM is not modified. Everything below goes in an out-of-tree plugin package that is loaded through the
`vllm.general_plugins` entry point and registers a quantization config with
`@register_quantization_config("nestquant")` (both exist in the local checkout, /home/coder/git/glm52/vllm @ f32e283).
The in-repo `nvfp4_aqlm_hybrid.HybridExpertsMoEMethod` is the structural template: a `FusedMoEMethodBase` subclass with
its own `create_weights` / `process_weights_after_loading` / `apply`, `get_fused_moe_quant_config -> None`,
`supports_eplb = False`, and a gemv path whenever `torch.cuda.is_current_stream_capturing()` is true.

## 1. What a TP8 rank holds

GLM-5.3 routed experts: H = 6144, I = 2048, 256 experts, top-8. vLLM's TP MoE splits I, so each rank holds all
256 experts at I/8 = **256**. Each shard is a self-contained NestQuant "expert" of shape (H = 6144, I = 256):

- **gate|up** [2·256 × 6144] and **down** [6144 × 256].
- Rotations are per 128-wide Hadamard block, so every I-side rotation stays inside the 256-wide shard. This covers
  sv_g, sv_u and su_d (DESIGN.md format item 1).
- The H-side signs (su_in, sv_o) are identical on all ranks.
- The kernel's output for a shard is a partial sum of the full expert output. The output rotation
  (WHT128 · sv_o, over H) is linear and identical per rank, so partial sums stay valid. vLLM's MoE runner then does
  the usual `tensor_model_parallel_all_reduce` (the method must not set `skip_final_all_reduce`).
- The routed-weight combine is fused into K2, so each rank returns out[b] = Σ_k rw[b,k] · partial_e(x_b).

Per expert-shard sizes (bytes):

| plane | gate\|up | down | total |
|---|---|---|---|
| base (2 bpw, uint4 per lane record) | 786,432 | 393,216 | 1,179,648 (= 2 × 576 KiB) |
| P4 residual, K = 2 (level 4 adds) | 786,432 | 393,216 | 1,179,648 |
| P4 residual, fractional K = 1.75 gu / 2.5 down | 688,128 | 491,520 | 1,179,648 (same total) |
| **P4 production (T14 pattern) 1.9375 gu / 2.3125 down (4.0846 bpw)** | 761,856 | 454,656 | 1,216,512 |
| P4 production option 2 / 2.3125 (4.1263 bpw) | 786,432 | 454,656 | 1,241,088 |
| d4 block word (u32 Mb \| N<<8 per 16×128 unit) | 6,144 | 3,072 | 9,216 |
| base-variant signs (T12 `base_var` 'sign', uint8 per unit) | 1,536 | 768 | 2,304 |
| scales, per level (half[H su_g \| I sv_g \| I sv_u \| I su_d \| H sv_o \| H su_u]) | | | 38,400 (× 2 levels) |

(The tail sub-arrays add a 4-byte pad each. T12's `plane_bytes` counts the same payload without it.)

Plane layout (the decoder is thread 15's RM_P int-fold, `nqdec::` in nqmoe.cu; bit-level spec `ref15_spec.py`):

- **Record** = one lane's bits of one 16×128 unit, index rec = (strip·C + chunk)·32 + lane, C = K/128.
- **Base**: one uint4 (128 bits = 64 weights × K 2) per record. It is the same for every residual K.
- **P4**: RBITS = 4·(16·KA + popc(MASK)) bits per record (K = 2: 128, 1.75: 112, 2.5: 160, 2.25: 144, 3: 192,
  1.5: 96, 1.9375: 124, 2.3125: 148). Ring step p has KA + ((MASK >> (p%16)) & 1) bits, the same as
  `nq_decode.PATTERNS`. It is stored as sub-arrays `uint4[n4] | uint2 | uint | ushort | tail` (n4 = RBITS/128, then
  whichever 64/32/16-bit remainders apply). Each is contiguous over all records of the projection, so every lane load
  is coalesced. `tail` exists iff T = RBITS % 16 ≠ 0 (12 bits for 1.9375, 4 for 2.3125). It is bit-packed over
  records: record r's tail is at bit r·T, and it holds stream bits [RBITS − T, RBITS) of the record. It is followed by a
  4-byte pad. A sub-array base is `p4 + nrec·(bytes of the earlier sub-arrays)`, with nrec = S·C·32 (dense) or
  S·nm·32 (mask mode).
- **Base variant** (T12 production `base_var` 'sign'): optional uint8 per unit (dense strip·C + chunk, also in mask
  mode). Bit g is the sign of ring g, i.e. lanes 4g..4g+3. a = −1 negates that ring's level-2 and level-4 values
  exactly; the kernel XORs the sign bit into the final HFMA2 constants. Table [12] is gate|up and [13] is down; 0 means
  no variants. G = 4 only.
- **Scales**: T12 gives every projection its own suh/svh. Gate and up have different input scales (su_g ≠ su_u), and
  level 2 and level 4 have separate vectors (`planes["base"]` vs `planes["p4"]` suh/svh). The kernel takes one half
  vector per (expert, level), `[H su_g | I sv_g | I sv_u | I su_d | H sv_o | H su_u]`. Gate row-blocks rotate x with
  su_g and up row-blocks with su_u, which is a block-uniform choice at no extra cost. A level switch must also repoint
  table [9] to the vector of the new level; the mailbox row carries it.
- **d4**: one u32 per unit, Mb in bits 0–7, N in bits 8–15 (δ = N/Mb, 1 ≤ Mb ≤ 255, Mb + N ≤ 257).
- **Rings**: LSB-first, tail-biting over G = 4 lanes (lanes 4g..4g+3 = one 256-weight ring; thread 12/15 format).
  G = 2 is still compiled and verified, but G = 4 is the default and the encoder format. G is a launch argument and
  must match how the planes were encoded.
- Repacking a thread 12 artifact (`nq_encode.encode_expert`) into this layout is `verify_t12.repack`:
  - ring streams → lane words (the inverse of `ref15_spec.rings_from_lane_words`), then `moe.pack_words`;
  - the u16 block words go into d4 unchanged;
  - `base["var"]` becomes the variant byte plane;
  - storage order (shard, strip, chunk) becomes kernel order strip·C + chunk;
  - gate strips then up strips form the fused gate|up.
- Both planes are laid out per (strip, chunk), so TP8 shards are unit ranges. Gate|up shards are contiguous strip
  ranges. Down shards are chunk ranges (2 of 48 chunks per strip), i.e. strided in rec: repack per rank offline.

**Low-rank plane** (T12 f128e41 "eigen plane", DESIGN.md format item 7; per expert r ≤ 4, mostly 1, often 0).
T12 stores V fp16 [r, in] (unrotated input basis) and U2 / U4 fp16 [r, out]; L2 = W2 + U2ᵀV, L4 = W4 + (U2 + U4)ᵀV.
The kernel computes z = x·Vᵀ and adds z·U2 (+ z·U4 at level 4) in fp32, in the unrotated output basis:
- gate/up: x = the hidden state (unrotated, fp16 as given). K1 blocks whose rows start a 128-column gate group add their
  k-slice's partial z (atomicAdd into z_gu). The group's epilogue adds z·U after WHT · sv, then re-zeroes z_gu.
- down: x = the unrotated SwiGLU output silu(g)·u, taken before su_d, WHT and the fp16 store of h. Each K1 epilogue
  stores the partial z over its 128 columns, once per call (no atomics). The K2 combine sums the I/128 partials,
  adds z·U2_d (+ U4_d) after WHT · sv_o, then applies rw.
- One fp16 block per expert (shard), pointed to by table [14], in halves:
  `V_g [r_gu,H] | U2_g [r_gu,I] | U2_u [r_gu,I] | U4_g [r_gu,I] | U4_u [r_gu,I] | V_d [r_dn,I] | U2_d [r_dn,H] | U4_d [r_dn,H]`,
  with I the shard's I and each part 8-byte aligned. `moe.Expert.set_lr` packs it, and `verify_t12.load_expert` packs it from an artifact.
- gate and up share one V (T12 `lrV_from="gate"`, `meta.lr.shared_V`). Older encodes with separate V (pat9 `*lr.pt`)
  pack as V = [V_g; V_u] with zero-padded U (U_g = [U_g; 0], U_u = [0; U_u]), which needs r_g + r_u ≤ 4.
- r = 0 reads nothing and costs nothing. The level-4 U4 read follows force_level like the other level-4 planes. The
  block holds U4 even at level 2 (resident, 13 KB per r=1 expert-shard). Splitting U4 into the P4 slot would need a
  second pointer, which is not done.
- **TP8**: gate/up are shard_axis n, so V is replicated and U is sliced by output. Each rank's gate/up term is exact
  locally. down is shard_axis k: V_d is sliced by input columns and U_d is replicated. Each rank computes a partial
  z_s = h_s·V_dsᵀ and adds z_s·U_d to its partial output. Because Σ_s z_s·U_d = (Σ_s z_s)·U_d = z·U_d, **the existing
  row-parallel output all-reduce already covers it**. No extra all-reduce of z is needed. DESIGN.md item 7 and
  nq_layer's comment ("all-reduce z") describe the unneeded extra reduce. The nq_layer tp{s}.pt fields map 1:1:
  gate lrV, gate/up lrU2 / lrU4 → V_g, U*_g, U*_u, and down lrV / lrU2 / lrU4 → V_d, U2_d, U4_d.
- Bytes per expert-shard at I = 256: 2·(r_gu·(H + 4·256) + r_dn·(256 + 2·H)), i.e. 39,424 at r = 1/1 (1.6% of a
  4.13-bpw expert-shard) and 157,696 at r = 4/4.
- Cost (`bench_lr.py`, PER=1 NQ_TIMING=idle NQ_STAT=min, GPU 6 shared).
  - Setup: µs per layer, cold recency routing, top-8. Every expert has r_gu = r_dn = r.
  - "pre" is the pre-lr kernel build. 4q is production level 4 (4.1263 bpw).
  - Every number includes a constant 6.6 µs graph-launch offset (the null graph).

  TP8 shard I = 256:

  | B | EXL3 2 | L2 pre | L2 r0 | L2 r1 | L2 r4 | EXL3 4 | 4q pre | 4q r0 | 4q r1 | 4q r4 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 1 | 108.9 | 41.3 | 40.8 | 41.2 | 48.9 | 111.3 | 53.4 | 53.0 | 56.3 | 61.4 |
  | 2 | 123.0 | 50.3 | 49.3 | 51.2 | 61.1 | 126.3 | 69.1 | 69.5 | 72.9 | 82.6 |
  | 3 | 138.4 | 65.1 | 66.1 | 69.4 | 82.2 | 142.4 | 98.1 | 100.1 | 102.9 | 116.2 |
  | 4 | 153.1 | 83.8 | 85.0 | 90.4 | 104.3 | 158.1 | 121.6 | 124.0 | 127.7 | 142.5 |

  Full I = 2048:

  | B | EXL3 2 | L2 pre | L2 r0 | L2 r1 | L2 r4 | EXL3 4 | 4q pre | 4q r0 | 4q r1 | 4q r4 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 1 | 255.9 | 151.2 | 154.3 | 157.0 | 168.6 | 262.7 | 235.0 | 235.2 | 238.0 | 251.3 |
  | 2 | 341.5 | 212.2 | 215.4 | 219.5 | 235.5 | 346.6 | 340.6 | 342.0 | 345.6 | 358.6 |
  | 3 | 440.8 | 274.2 | 279.0 | 284.0 | 305.4 | 449.9 | 435.2 | 439.1 | 445.2 | 470.4 |
  | 4 | 538.9 | 332.4 | 339.2 | 345.9 | 370.8 | 550.7 | 547.6 | 551.7 | 557.8 | 582.4 |

  - r = 0 vs pre: −1.0 to +2.4 µs at the shard (noise level), and +0.2 to +6.8 µs (≤ 2%) at the full shape.
  - r = 1: +0.4 to +5.4 µs (1–6%) at the shard and +2.7 to +6.7 µs (≈ 1–2%) at full.
  - r = 4: +8 to +19 µs (≈ 15%) at the shard and +13 to +31 µs (5–9%) at full.
  - Every NQ variant stays below EXL3 at the same level, except full 4q at B3–B4, which was already at parity before lr.
- Registers (cuobjdump, vs pre-lr): 64 everywhere, unchanged.
  - Grid kernels: stack unchanged, except <2,1,1> 0 → 8 and <2,2,0>/<4,2,0> 72 → 64.
  - Persistent kernels: stack 0 to +40 bytes, e.g. <2,1,1> 176 → 216 and <4,2,1> 440 → 464.

Mask mode (128×128 per-block 2b/4b mixing inside the 4-bit tier) adds a uint64 per 16-row strip and compacts P4/d4
to the flagged chunks. With the shard's K = 256 for down, that is only 2 chunks per strip.

Per rank per MoE layer: the base for all 256 experts is **288 MiB**, and each level-4 expert adds 1.13 MiB.

Placement per rank:

- **Base planes: always resident**, one pool per layer (constant size per (layer, plane, shard), 64 KiB aligned,
  DESIGN.md format item 5). At TP8 every expert is resident at level ≥ 2. Level 0 ("not resident, contributes 0")
  is only for EP or pruning. It must then be all-or-none across the 8 ranks, because one rank at level 0 would
  drop one shard of the expert.
- **P4 slots**: a preallocated per-layer (or global) slot pool of fixed-size slots (P4 + d4 [+ flags] ≈ 1.13 MiB per
  expert-shard). The static 4-bit set and the dynamic upgrades both live in slots.
- **Host**: pinned P4 + d4 images of every expert-shard, per rank. This is the full 4-bit tier: 256 × 1.13 MiB per
  layer per rank. Keep it in NUMA-local pinned memory of each GPU's socket.

## 2. Device tables (per rank, per layer)

| tensor | shape | notes |
|---|---|---|
| `table` | int64 [256][16] | [0] level 0/2/4; [1..4] gate\|up base, p4, d4, flags; [5..8] down same; [9] scales of the current level; [10] gate\|up residual K code, [11] down residual K code (0: K 2, 1: 1.75, 2: 2.5, 3: 2.25, 4: 3, 5: 1.5, 6: 1.9375, 7: 2.3125); [12] gate\|up, [13] down base-variant plane (0 = none); [14] low-rank block pointer (fp16, §1 "Low-rank plane"; 0 = none); [15] lr ranks r_gu \| r_dn << 8 (0 = none). Read on device at every replay. |
| mailbox `stage`, `seq`, `applied` | int64 [256][16], int32 [256] ×2 | graph-safe updates (§4) |
| `applied_host` | int32 [256], pinned host-mapped | the scheduler polls this without syncing |
| `hits` | int32 [256], pinned host-mapped | routing-hit export (§5) |

- `table` is 32 KiB per layer.
- The residual K code is per (expert, projection) and is read at replay, so a slot can hold any K. Only the codes in
  the build masks are compiled, and gate|up (K1) and down (K2) each get their own mask:
  - Default build: gate|up {K 2, 1.9375} (`NQ_RK_GU=0x41`), down {K 2, 2.3125} (`NQ_RK_DN=0x81`). This covers both T14
    production options, 4.0846 (1.9375 / 2.3125) and 4.1263 (2 / 2.3125).
  - `-DNQ_RK_CODES=m` gives both kernels the whole mask `m` (0x7 for the older 1.75 / 2.5 config, 0xff for all eight
    codes). `NQ_RK_GU` / `NQ_RK_DN` narrow it further.
  - The split exists for speed. Each extra code per kernel adds stack in the CPW = 2 and persistent variants.
    One mask for both kernels (0xC7 or 0xC1) was 2–8% slower on 4p than the split.
  - A code outside a kernel's mask silently decodes as K = 2. The binding `rk_codes()` returns the compiled {gate|up, down}
    masks, and `MoELayer.set` asserts against them.
- Levels should be identical across ranks at any step. This is not needed for correctness, but it keeps the model a
  single well-defined quantization. The mailbox lag is at most one step, and a per-rank lag difference only mixes
  shards of the same expert at two levels for that step.
- The workspace (acc_gu [32, 2I] f32, h [32, I] f16, acc_d [32, H] f32, counters, wq int32 [8], zws f32
  [2·32·(I/128)·4]) can be one shared set per rank. Layers run in stream order, and the kernels leave the workspace
  zeroed on exit. zws is the low-rank z: the z_gu half returns to zero, while the z_dn partials are overwritten every
  call and never need zeroing. At I = 256 the workspace is about 0.9 MB. `moe_forward` takes zws as its last argument.
- It must not be shared with a concurrently running shared-expert stream. Leave `mk_can_overlap_shared_experts`
  False, as the hybrid method does.

## 3. The `apply` path

```
apply(layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
    if x.shape[0] <= 4:                                   # decode / MTP verify (B*topk <= 32)
        mbox.apply()                                      # captured: stage -> table for pending ops
        moe_forward(x16, ids64, rw16, table, out32, ws..., I=256, nm_gu, nm_dn,
                    cfg_gu, cfg_dn, G=4, force_level=0, which=3, hits_ptr)
        return out32.to(x.dtype)
    else:                                                 # prefill
        dense-decode experts per group + grouped GEMM (as HybridExpertsMoEMethod._apply_grouped)
```

Dtype and shape notes:

- The kernel takes x as **fp16** [B, H], sel as **int64** [B, 8], rw as **fp16** [B, 8], and writes out as
  **fp32** [B, H].
- vLLM hands over bf16 x, int32 ids and fp32 weights. Today that means three small cast kernels, which are
  graph-capturable. Native bf16-in and int32-ids variants are an open item (a few lines in the prologue and route).
- Check whether GLM's `routed_scaling_factor` is already folded into `topk_weights` on the chosen router path.
  The kernel applies rw exactly as given.

Tuned configs:

| shape | B | gate\|up | down |
|---|---|---|---|
| full I = 2048 (tune2_I2048.json) | 1 | [1,8,8] | [1,4,8] |
| full I = 2048 | 2–4 | [1,8,8] | [1,8,8] |
| TP8 shard I = 256, level 2 (tune2_I256.json + bench_shard.json) | 1–4 | [1,8,8] | [2,8,1] or [1,8,2] |
| TP8 shard I = 256, level 4 | 1 | [1,4,6] | [1,8,2] |
| TP8 shard I = 256, level 4 | 2–3 | [1,4,4] or [1,8,8] | [1,8,2] |
| TP8 shard I = 256, level 4 | 4 | [1,8,6] | [1,8,2] |

cfg = [chunks per warp, strips per block, stages]. For a mixed 2/4-bit shard layer use the level-4 row (the 4-bit
experts dominate the time). Near-ties are within ~2%; any of the listed values is fine.

Kernel constraints: B·topk ≤ 32, H and I multiples of 128 and ≤ 64·128, and K % (cpw·nst·128) == 0.

Prefill (B > 4) needs a dense-decode kernel. The decode math is `nqdec::` in nqmoe.cu, but only the torch reference
(`moe.dense_W`) exists as a whole-matrix decoder today (open item).

## 4. Level switching under vLLM's CUDA graphs

Under full CUDA graphs the host never gets to enqueue work on the compute stream between layers, so the table flip is
done by a tiny captured kernel, `mailbox()` (nqmoe.cu `nq_mailbox`, python `moe.Mailbox`). It runs at the start of each
MoE layer, before K1. The scheduler only ever touches its own side stream (one per rank).

**Upgrade 2→4.** On the side stream, in order:
1. `cudaMemcpyAsync` pinned P4 + d4 (+ flags) into a free slot, as 576 KiB chunks.
2. Write `stage[e]` = new row (level 4, slot pointers).
3. Bump `seq[e]`.

The next `mailbox()` that sees `seq != applied` copies the row into the table and publishes `applied[e]`. Because
the side stream orders copy → stage → seq, a row is never live before its P4 bytes are.

**Downgrade 4→2.** Post a level-2 row (P4 pointers ignored). Once `applied_host[e] == seq[e]`, no later kernel can
reference the slot, so the slot is free and may be overwritten immediately.

**Rule:** at most one outstanding op per expert.

Measured on A100 (levelswitch_mbox.py; graph = [mailbox, MoE] captured once, side-stream ops racing the replays):
- 306 ops over 300 steps, all applied in the same or the next replay.
- Every replay matched the dense reference of the state it saw (max rel err 7.0e-5). The stale state would have
  shown ≥ 4.1% error.
- Freed slots were scribbled with junk right after release without affecting any output.
- The host-stream-wait variant (levelswitch.py: event + `wait_event` + table copy on the main stream) passed
  399 switches over 200 steps, max rel err 6.3e-5, stale ≥ 4.7%.
- Mailbox cost: 1.4 µs per layer (35.4 → 36.8 µs, I = 256 B1 2b). Hit export adds 0.3 µs.
- Re-run on the thread 15 decoder with G = 4 and half the experts at fractional residual K (1.75 gu / 2.5 down),
  slots sized for the largest K: mailbox 311 ops / 300 steps, max lag 1, max rel err 1.2e-4 (stale ≥ 2.6%); host-event
  variant 399 switches, max rel err 6.2e-5 (stale ≥ 4.4%).
- Re-run with the production residual (odd experts at 1.9375 gu / 2.3125 down, with base-variant sign planes; default build):
  levelswitch_mbox.py max rel err 1.0e-4 (stale ≥ 9.0%).

## 5. Miss list and routing export to the CPU scheduler

- Set `MoELayer.hits_ptr` (the `moe_forward` arg `hits_ptr`) to a **pinned host-mapped** int32 [256] buffer per
  layer. One warp of one K1 block `atomicAdd`s one count per (token, expert) pick with nonzero weight, straight
  over PCIe. The pointer is fixed, so this is graph-safe, needs no extra graph node, and involves no sync.
- The host owns the levels, so the **miss / upgrade-candidate list** for a step is simply {e : hits[e] increased and
  level[e] < 4}.
- The scheduler thread polls the counters (monotone, so it diffs against the previous snapshot) and applies the
  thread 10 policy:
  - static 4-bit set by benefit per byte;
  - recency upgrades with a one-step lag, which matches the mailbox's ≤ 1-step latency;
  - next-layer top-12–16 prefetch;
  - instant downgrade under KV pressure;
  - a per-step byte cap at link rate.
- All 8 ranks see identical routing, so one scheduler process can decide for all ranks and push per-rank ops to each
  rank's side stream. Alternatively, each rank decides deterministically from the same counters.
- Only the grid-mapped kernel (the one used) exports hits; the persistent variant does not.

**Copy budget** (copybw.py, A100, pinned host → device, 576 KiB `cudaMemcpyAsync` chunks):
- Idle: 20–21 GB/s (short run). Concurrent with the MoE graph: 22.5–22.8 GB/s.
- That is 26 µs per 576 KiB chunk, and one expert-shard upgrade (1.13 MiB) takes about 52 µs.
- The MoE kernel slowed by 0.0% ± 0.3% during copies (2b/4b, B1/B4).
- Divide the link rate among the GPUs that share a PCIe switch on the target box.

## 6. Files

- `nqmoe.cu`: kernels (K1 gate|up + SwiGLU, K2 down + fused combine, persistent variant, mailbox) and the
  `moe_forward` / `mailbox` / `occ` bindings.
- `nqdec::`: the decoder (thread 15 RM_P int-fold + greedy funnels, per-projection fractional residual K).
- `ref15_spec.py`: bit-level spec (copied from thread 15). `verify15.py`: spec == torch reference == kernel-decoded
  weights (NQ_WDUMP build), bitwise, all 8 K codes, with and without base variants.
- `verify_t12.py ART.pt`: repacks a thread 12 encoded expert into the kernel layout. Checks that `moe.dense_W`, the
  kernel decode and `nq_decode` agree bitwise at levels 2 and 4, then checks the forward pass against
  `nq_decode.decode_expert`.
- `verify_lr.py [ROOT] [L]`: low-rank plane on the T12 f128e41 artifacts (smoke_lr L30 E168 / E169, text + mmself).
  Kernel vs decode_expert and vs Expert.ref at L2 / L4, with U × 64 (lr-dominated), mixed levels in one launch
  (grid + persistent), and the TP8 shard fields: reassembly, per-rank kernel, Σ_s partial z = full z.
  `lrtest.py`: synthetic ranks {0/0, 1/0, 0/1, 1/1, 2/1, 4/4} × levels × configs × B1–4, I = 2048 and 256, workspace clean.
  `bench_lr.py`: lr cost, r = 0 / 1 / 4 vs the pre-lr build and EXL3.
- `smoke_layer.py [LAYER_DIR]`: real-layer smoke on an uploaded nestquant-v1 layer. It loads all 256 experts (lr
  included, 5.1 GB GPU) and compares the kernel with Σ rw·Expert.ref, under the default allocation, all L4 and all L2.
  - Routing: B1–B4 random batches, plus a sweep that routes every expert. Tuned, persistent and grid configs are all used.
  - L38 (/tmp/nestquant/nq-encode-v1/L38): 156 launches, all 256 experts routed, max rel err 1.43e-4 (gate 1.5e-4),
    workspace clean, peak 9.1 GB. The kernel vs nq_decode dense forward gives 7.7–9.1e-4, the fp16-activation floor.
  - Lr ranks g/u/d: (1,1,0) 214, (1,1,1) 34, (2,2,0) 5, (2,2,1) 2, (2,2,2) 1.
- `abk.py`: A/B of build variants (`NQ_RK_CODES` / `NQ_RK_GU,NQ_RK_DN` / other defines) at levels 2, 4, 4p (1.9375 / 2.3125)
  and 4q (2 / 2.3125).
- `timing.py`: `NQ_STAT=min NQ_BLOCKS=150` reports the min over blocks instead of the median. Use it on a shared,
  time-sliced GPU. `NQ_TIMING=idle` times one replay started from an idle GPU, which gets a fresh time slice. The
  default mode times the second of two back-to-back replays. On GPU 6 (slice about 2.1 ms), any variant whose replay is
  longer than about 1.05 ms then always crosses a slice boundary, which shows up as a fake step of 1.6–4x (e.g. lr r3 at
  B4). Keep each timed replay under about 1 ms, e.g. `bench_lr.py PER=1` (one graph per routing).
- `moe.py`: pools, `entry()` table row builder, `MoELayer`, `Mailbox`, dense reference decode (`dense_W`,
  `Expert.ref`), residual packing (`Proj`, `pack_words`, `RKP`).
- `build.py`: JIT build (`NQ_DEFS` for variants).
- `tune2.py` (per-stage config sweep across build variants → `tune2_I2048.json`, `tune2_I256.json`), `abtest.py`
  (old nqk2 A4 decoder vs nqdec, one process), `bench_full.py` / `bench_shard.py` (vs EXL3; `bench_shard.json` has an
  EXL3 default run and an `EXL3_MOE_COOP_KSPLIT=2` run), `exl3_sweep.py` (EXL3 coop launch-option sweep →
  `exl3_sweep_I*.json`).
- EXL3 comparison note: exllamav3 1.5.1's coop MoE kernel is fastest at the shard shape with
  `EXL3_MOE_COOP_KSPLIT=2` for B1–B3 (default for B4). This needs scratch ≥ ksplit·slots rows (`EXL3MoE(smax=...)`).
  The shard speedups quoted against EXL3 use its best option per B.
