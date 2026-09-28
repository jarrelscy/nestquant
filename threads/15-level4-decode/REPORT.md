# Thread 15: level-4 decode cost (A4 below 8 ops/weight, fractional K per projection)

## Question
Can the thread-04 A4 decoder (2-bit mul1 trellis base plus a 2-bit mul1 residual, 10.3 ops/weight by the whole-kernel count) get below about 8 ops/weight?
Can the base and the residual take fractional K per projection? What does each trick cost in µs, and what quality constraint does it imply?
Per the user, only levels 2 and 4 exist; there is no level 3.

## Method
- **Kernel:** nqk15.cu, derived from thread 04's nqk2.cu.
  - mma.m16n8k16 fp16 with fp32 accumulate. The batch sits in N.
  - Tile layout: 16-row strip × 128-k chunk. Each lane owns 64 weights per tile.
  - Ring G=2 (a tail-biting ring over lane pairs).
- **Fused-B chain:** 2 launches.
  - gate/up launch: sign + WHT prologue.
  - down launch: SwiGLU + WHT prologue and output WHT.
- **Correctness:** every variant checks against a numpy reference (check15.py). P and fold modes are bit-exact on W.
- **ops/weight:** static SASS count vs UNI4 (plain 4-bit LUT-free baseline = 1.75), two ways.
  - "loop": nvdisasm line-attributed decode-loop instructions only, with -DNO_WDBG (sassline.py).
  - "whole": thread 04's whole-kernel diff (sass15.py). It is inflated by prologues.
- **Timing:** bench15.py, which is thread 04's method.
  - One process, CUDA graphs, shuffled replay.
  - NC=6 distinct expert copies, so L2 is cold.
  - (cpw, split, stages) tuned per projection and per batch (tune15.py → tune15.json).
  - EXL3 comes from the matched L16 E36 refits (expert_{2,4}.bin). "Adapter" is EXL3Expert.forward_batch. "Bare" is the three bc.run kernels plus silu*mul.
- **Data:** synthetic random codes at the real shapes. Timing does not depend on the data.

## Results (A100, GPU 2, µs per expert, full gate/up + SwiGLU + down)

| variant (id) | ops/w loop-conv | ops/w whole | B1 | B2 | B3 | B4 |
|---|---|---|---|---|---|---|
| **EXL3-4 bare kernels** | ~4-5 (3INST) | | 53.6 | 64.9 | 74.3 | 74.4 |
| EXL3-4 adapter | | | 67.3 | 78.4 | 89.0 | 89.4 |
| A4 thread-04 nqk2 (reference) | | 9.77/10.3 | 42.5 | 45.0 | 46.0 | 46.8 |
| A4 in nqk15 (3) | 8.70 | 9.77 | 42.3 | 44.2 | 45.9 | 46.6 |
| A4 unfused, 5 launches | 8.70 | | 46.4 | 48.4 | 49.7 | 50.6 |
| A4 + shared funnels (4) | 7.98 | 8.81 | 39.3 | 41.0 | 42.9 | 43.6 |
| fp32 fold, RM_F (6) | 7.48 | 8.29 | 40.6 | 42.3 | 44.2 | 44.8 |
| **int-fold RM_P + greedy funnels (33)** | **7.15** | **7.95** | **39.2** | **40.9** | **42.6** | **43.3** |
| RM_P unfused | 7.15 | | 43.9 | 45.5 | 46.4 | 47.2 |
| RM_P RAW, fp32 affine per chunk (41) | 6.87 | | 39.6 | 41.5 | 43.5 | 44.2 |
| RM_P2, V2 2D residual (37) | 6.22 | 6.83 | 37.1 | 38.7 | 40.6 | 41.3 |
| RM_P2 RAW (42) | 5.92 | | 37.5 | 39.2 | 41.4 | 42.0 |
| P2 unfused | 6.22 | | 41.2 | 42.9 | 43.7 | 44.6 |
| **frac: P, residual K 1.75 gu / 2.5 down (29/30)** | 7.23 / 7.16 | | **39.4** | 41.3 | 43.1 | 44.0 |
| frac: P RAW, same mix (43/44) | 6.88 / 6.86 | | 40.1 | 41.9 | 44.0 | 44.8 |
| frac: P2 RAW, same mix (46/45) | ~5.9 | | 38.4 | 40.1 | 42.2 | 42.9 |
| **EXL3-2 bare kernels** | | | 60.1 | 77.8 | 72.0 | 72.5 |
| EXL3-2 adapter | | | 73.6 | 91.7 | 86.2 | 86.6 |
| B2 thread-04 funnel (0) | 4.27 | 4.70 | 28.0 | 29.4 | 30.7 | 32.0 |
| **B2 greedy funnels (34)** | **3.91** | 4.26 | **26.4** | **28.1** | **29.8** | **30.7** |
| B2 RAW (40) | 3.62 | | 27.5 | 29.3 | 30.6 | 32.0 |

Fractional-K cost in the decode loop (ops/weight, loop-conv):

| setting | K=2 (int) | 1.5 | 1.75 | 2.25 | 2.5 | 3 |
|---|---|---|---|---|---|---|
| P residual | 7.15 | 7.16 | 7.23 | | 7.16 | 7.19 |
| RAW P residual | 6.87 | | 6.88 | | 6.86 | |
| base (B2 greedy) | 3.91 | | | 3.93 | 3.92 | |

- Without greedy funnels (the shared-funnel WOPT1), fractional K cost +0.3-0.5 ops/weight.
- Measured µs cost of the gu 1.75 / down 2.5 mix vs integer K: +0.2 µs at B1 and +0.7 µs at B4. This includes the extra bytes (2.5 > 2 on down).

Observations:
- Ops/weight converts to time only weakly. A4 → P cuts 1.55 ops/weight and saves 3.1 µs at B1. The kernel is partly latency/issue bound, so small ALU cuts show up only partly in time.
- RAW (the mma on the raw fp16 codes, with the affine done in fp32 per chunk) removes 0.3 ops/weight in the loop. It costs a per-block xsum prologue and extra accumulator registers, and it is never faster in time. Reject it for speed. It does keep one precision advantage (C in fp32).
- The fused chain beats the unfused one by 3.5-4.5 µs in every case.

## Recommended decoder: RM_P int-fold with greedy funnels, V1 residual (id 33; fractional residual = ids 29/30)

### Per pair of weights (2 fp16 outputs, one lane)
1. Take base window xb and residual window xr, each 16 bits.
   - Direct extraction (1 op) when the window offset O%32 ∈ {0, 8, 16}.
   - Otherwise one greedy funnel shift. It is shared by up to 3 windows per residue mod 8 (the offsets are constexpr).
2. Hash each window: xb·0x83DCD12D and xr·0x83DCD12D (the standard mul1 hash).
3. Per weight e ∈ {lo, hi}, using the lo or hi byte pair of the hash:
   - `t = dp4a(xr_e, Nrep, dp4a(xb_e, Mbrep, 0x640080))`
   - This gives t = 0x640080 + Mb·Sb + N·Sr, where Sb and Sr are the byte sums of the two hashed windows.
4. `h = PRMT(t_lo, t_hi, 0x6521)`. This puts the fp16 pair 1024+F into the mma operand directly, where F = floor((Mb·Sb + N·Sr + 128)/256).
5. `w = HFMA2(h, A'A', CC)`, with A' = 256·A/Mb and C = (1+δ)·K0 − 1024·A'.
   - A = fp16 0x1eee (1774·2^-18), B = fp16 0xc931, K0 = 1024A + B = −3.453125.
   - A' comes from `__constant__ c_rcp[256]` = 1/Mb.

### Storage (per projection, per plane; the base plane and the residual plane are separate so the 2-bit base streams on its own)
- **Record** = one lane's bits for one 16×128 tile: rec = (strip·nchunks + chunk)·32 + lane.
- **Bits per record:**
  - V1: 64·K_eff. For example K=2 → 128, 1.75 → 112, 2.5 → 160, 2.25 → 144, 1.5 → 96, 3 → 192.
  - V2 (P2): 32·K_pair.
- **Plane layout (sub-arrays):** `uint4[n4] | uint2 | uint | ushort`.
  - n4 = BITS/128. The 64/32/16-bit remainders follow as they apply.
  - Each sub-array is contiguous over rec, so every lane load is coalesced.
- **Bitstream convention: LSB-first.**
  - Window j = bits [c_j, c_j+16) of the lane stream, with bit c_j as the LSB.
  - c_j = (j>>4)·(16·KA + popc(MASK)) + Σ_{i<j%16} (KA + ((MASK>>i)&1)).
  - This is the EXL3 period-16 pattern, but EXL3 is MSB-first.
  - Fractional K patterns:

    | K | KA | MASK |
    |---|---|---|
    | 1.5 | 1 | 0xAAAA |
    | 1.75 | 1 | 0xEEEE |
    | 2.25 | 2 | 0x8888 |
    | 2.5 | 2 | 0xAAAA |
    | 2.75 | 2 | 0xEEEE |
    | 3 | 3 | 0 |

- **Tail-biting ring:** G is a compile-time parameter, and both values were verified bit-exact (check15.py with NQ15_G=4).
  - **G=4 (recommended; matches thread 12):** 256 weights, lanes 4g..4g+3, next lane = (lane&~3)|((lane+1)&3).
  - **G=2:** 128 weights, next lane = lane^1.
  - One shfl per plane per chunk, inside the same 16×128 tile.
  - Timing is the same within noise (chain15_g4.json, µs B1/B2/B3/B4):

    | G=4 variant | B1 | B2 | B3 | B4 |
    |---|---|---|---|---|
    | P | 38.8 | 40.3 | 42.6 | 43.8 |
    | P2 | 36.9 | 38.3 | 40.5 | 41.5 |
    | frac | 39.5 | 41.0 | 43.3 | 44.3 |
    | B2 greedy | 26.5 | 27.9 | 29.9 | 30.9 |
    | EXL3-4 bare | 53.7 | 64.8 | 74.1 | 74.2 |
- **Block word:** one u32 per 16×128 block, in the residual plane: `Mb | N<<8`, i.e. Mb in bits 0-7 and N in bits 8-15.
  - δ = N/Mb.
  - Constraint: Mb + N ≤ 257 and Mb ≤ 255.
  - It fits in u16. The base-only (2-bit) view ignores it.
- **Alignment:** every 16×128 block is self-contained.
  - TP8 gate/up shards are contiguous strip ranges.
  - TP8 down shards are chunk ranges. These are strided slices of rec; repack them offline or index them with the chunk offset.
- **Mixed K:** K can differ per projection (for example residual 1.75 on gate/up and 2.5 on down). K is a compile-time template parameter per projection.
  - Per-16-column-block K within one projection was not built; it would need a runtime offset table.

### Bit-level spec of the V1 (RM_P) and V2 (RM_P2) codes
The executable spec is `ref15_spec.py`: numpy only, bit-exact with the kernel. xcheck15.py checks it on random G=4 units for ids 33, 29, 30, 38, 39 (V1, residual K 2 / 1.75 / 2.5 / 3 / 1.5), 37, 31, 32 (V2) and 28 (base 2.5). All units match, 0 mismatches.

- **Unit and rings:** as in thread 12's nq_decode.
  - A unit is 16 rows × 128 k, split into 8 rings of 256 weights.
  - Ring g, position p = t4·64 + j ↔ lane 4g+t4, weight j ↔ row g + 8(r&1), k = 16t + 2t4 + 8(r>>1) + e, where pp = j>>1, e = j&1, t = pp>>2, r = pp&3.
- **Stream:** an LSB-first bit stream per ring.
  - Lane t4's kernel record = ring bits [t4·Lbits, (t4+1)·Lbits).
  - V1: Lbits = 64K. V2: Lbits = 32·Kpair.
- **Offsets:** step_off(p) = (p>>4)(16KA + popc(MASK)) + Σ_{i<p%16} (KA + ((MASK>>i)&1)).
  - Integer K gives K·p.
  - The fractional patterns are listed in the Storage section.
- **State:** ring bits [step_off(p), +16), taken as a 16-bit window with wrap-around (tail-biting).
- **Hash:** x = state·0x83DCD12D mod 2³², with bytes b0..b3.
  - S = b0+b1+b2+b3.
  - S2 = 510 + b0−b1+b2−b3 (V2 only).
- **Level 2:** Q2 = fp16(A(1024+S(sb)) + B).
  - Bit-identical to harness.codebook_lut('mul1'): all 65536 states verified.
- **Level 4, V1:**
  - Block word u16 = Mb | N<<8, with 1 ≤ Mb ≤ 255 and Mb + N ≤ 257.
  - F = (Mb·S(sb) + N·S(sr) + 128) >> 8.
  - A' = fp16(1.732421875f·(1.0f/Mb)).
  - C = fp16(fp32(N·(1.0f/Mb))·K0 + K0 − 1024A').
  - Q4 = fp16(A'(1024+F) + C).
  - This approximates Q2 + (N/Mb)·mul1(sr).
- **Level 4, V2:**
  - One residual state per weight pair, at pair ring position q = t4·32 + P, using step_off(q) with the per-pair pattern. Examples: (KA=4, 0) = 2 bpw; (5, 0xAAAA) = 2.75 bpw.
  - Sr(2P) = S(state), Sr(2P+1) = S2(state). Then apply the V1 formula, with N ≤ 127.
  - For the encoder this is a 2D codebook: each state emits the pair (A·S + K0, A·S2 + K0)·δ.
    - A 1D Viterbi (exllamav3 quantize_tiles or TorchTileQuantizer with a 65536 LUT) cannot fit it directly.
    - It needs a pair-step Viterbi over 128 steps per ring, with a 2D squared-error cost.
- **δ choice:** `delta_to_MbN(δ)` picks Mb = min(255, floor(257/(1+δ))) and N = rint(δ·Mb). The δ quantisation error is ≤ 0.004 for δ ≤ 1.
  - An LS search over neighbouring (Mb, N) pairs against the actual fold output is better.

### V1 vs thread 12's encoder (nq_decode.py / nq_encode.py): bit-identical where, and the differences
**Identical:**
- unit geometry and ring_index (G=4);
- LSB-first ring streams (thread 12's pack_bits = the concatenated kernel lane records);
- integer-K windows at bit K·p;
- the mul1 hash;
- the whole of **level 2 (Q2 bit-identical)**;
- TP8 shard units (gate/up: 16 strips; down: 2 chunks);
- level-specific su/sv vectors. The kernel prologue takes them as arguments, so passing su4/sv4 at level 4 is free.

**Differs (level 4 only):**
1. **Q4 value.**
   - Thread 12: Q4 = fp16(Q2 + δ_fp16·mul1(sr)).
   - RM_P: the int fold above.
   - They are not bit-identical. With δ mapped by delta_to_MbN, measured on 1M random states:

     | δ range | rms diff | max diff |
     |---|---|---|
     | 0.1-0.5 | 0.0034-0.0041 | 0.017 |
     | 0.5-1 | 0.0060 | 0.024 |

   - That is (1.1-2.3)e-5 of E[Q4²], or about 0.4-0.8% of thread 12's 4-bit proxy MSE (≈0.0028) if the encoder does not model it.
   - About 90% of it is the fold rounding. The δ quantisation is the rest.
   - Fix: take the encoder's final Q4 from `ref15_spec.fold`, and choose (Mb, N) per unit by LS on the actual fold output.
   - A Viterbi pass that sees the rounding needs per-position values (they depend on S(sb) at that position), so a fixed LUT cannot do it. The post-hoc (Mb, N) refit is the practical route.
2. **δ representation:** fp16 per unit becomes the rational N/Mb (u16, same size).
   - The per-ring δ option ("nq_ring_delta", 9.384 vs 9.395) is not implemented. It would be a per-lane block word, costing about 0 ops/weight.
3. **Residual codebook:** mul1 only.
   - Thread 12's per-unit mcg/3inst choice ("cb6") cannot fold into the dp4a, so it is unsupported.
   - Thread 12 measured only 0.03 pp for it, so drop it.
4. **Per-unit K3 residual units** (k3 fraction plus the p4x hi-bit plane) are not supported. The kernel has per-projection compile-time (fractional) K.
   - This is the biggest gap: thread 12's k3_0.25 is its strongest level-4 variant (8.33 vs 9.395 routed, at more bits).
   - Kernel equivalent at the same bits: uniform residual K=2.25 per projection (MASK 0x8888, +0.01 ops/weight). Thread 12/14 should A/B "k3 on the chosen 25% of units" against "uniform 2.25".
   - If per-unit selection wins clearly, the kernel needs a warp-uniform per-chunk branch between 2 decode paths plus a per-unit offset table. That is not built; I expect under 1 µs but have not measured it.
   - The separate p4x plane would be interleaved offline into one 3-bit ring stream. This is lossless because level 4 reads both planes.
5. **Base variants** (per-ring sign/sg4) are not implemented.
   - They are foldable: A' → 256gA/Mb and C per ring, with effective δ = gN/Mb.
   - The level-2 kernel would need per-ring (gA, gB) HFMA constants, giving Q2 = fp16(gA(1024+S) + gB). That is not bit-identical to thread 12's fp16(lut)·g.
   - Gain in thread 12 is 0.03 pp. Recommend dropping unless it combines better with K3.
6. **Storage order:** thread 12 stores each unit as its 8 ring streams back to back; the kernel wants per-lane sub-arrays (uint4[n4] | uint2 | uint | ushort over rec).
   - This is an offline repack. `rings_from_lane_words` is its inverse and documents the mapping.
   - Unit order within a shard (strip, chunk) is the same.
7. **Fractional K:** thread 12 uses integer K only.
   - If it adopts fractional K, its offsets must equal step_off above (LSB-first, position p).
   - The harness's half-integer K comes from EXL3, which is MSB-first with Viterbi step i = 255−p. I have not verified that it produces the same offsets; check it with ref15_spec.states before use.

### Quality constraints for thread 14 (these define the codebook the encoder must fit)
1. **Greedy and shared funnels:** decode-only, bit-identical codes. No constraint.
2. **Int fold (RM_P):**
   - w = A'·(1024 + round((Mb·Sb + N·Sr)/256)) + C, i.e. base + δ·residual, where δ = N/Mb takes a quantized scale: Mb ≤ 255, Mb + N ≤ 257.
   - The rounding step is about 256/Mb units of A. For Mb=190 that is ≈0.009 with rms ≈0.0026. Measured against thread 12's Q4 (xcheck15.py), the rms difference is 0.004, about 0.4-0.8% of 4-bit proxy MSE if the encoder does not model it.
   - The value is a deterministic function of (xb, xr, Mb, N), so Viterbi can model it exactly.
   - C is stored as fp16 (|C| ≤ ~17), which gives ≤0.004 bias per block. This is the same order as A4's fp16 constant.
   - The 2-bit base view is unchanged standard mul1.
3. **fp32 fold (RM_F), if P rounding is unwanted:** δ = N/128 with N ≤ 255. Exact, and 0.3 ops/weight more than P.
4. **V2 residual (P2):** this is a codebook change and has not been quality-tested.
   - One 16-bit window per weight pair. Pair value = (byte-sum, 510 + b0−b1+b2−b3). The two parts are uncorrelated with the same marginal.
   - The residual trellis runs one step per 2 weights, so its rate is K bits per pair: V2 needs 2K bits of window per pair for the same bits/weight.
   - For the fold, N ≤ 127.
   - Adopt it only if a Viterbi fit on iid Gaussian (for example thread 03's trellis.py) is within about 0.05 dB of 1D mul1.
5. **Fractional K:** use the LSB-first period-16 pattern above. The encoder's trellis must use the same window offsets c_j.

## Verdict
- **ADOPT RM_P int-fold + greedy funnels (V1 residual).**
  - 7.15 ops/weight by the loop count (7.95 by the whole-kernel count), vs A4 at 8.70 / 9.77.
  - Full expert at 4 bit: 39.2 / 40.9 / 42.6 / 43.3 µs at B1-B4, vs EXL3-4 bare at 53.6 / 64.9 / 74.3 / 74.4. That is 27-42% faster.
  - With per-projection fractional residual K (1.75 gu, 2.5 down) it is 39.4 / 41.3 / 43.1 / 44.0 µs.
  - Codebook cost: only the quantized δ = N/Mb and a modelled rounding.
- **At 2 bit, adopt greedy funnels for the base (id 34).** It gives 26.4 / 28.1 / 29.8 / 30.7 µs vs EXL3-2 bare at 60.1 / 77.8 / 72.0 / 72.5. Fractional base K (2.25/2.5) is free: +0.01-0.02 ops/weight.
- **Needs more: V2 residual (P2).** It is 6.22 ops/weight and 2.1 µs faster than P at B1. Adopt it only if thread 14's Viterbi test shows no quality loss.
- **Reject RAW and the LUT decoder.** RAW gives no speed gain. The LUT decoder was dropped by the lead.

## Files (threads/15-level4-decode/)
- nqk15.cu, build15.py: the kernel and its build.
- nq15.py: the packer, the Proj class and the numpy references.
- check15.py: correctness checks.
- sassline.py (loop count) and sass15.py (whole-kernel count), with outputs sass15.json / sass15.txt.
- tune15.py → tune15.json.
- bench15.py → chain15.json (G=2) and chain15_g4.json (G=4): the paired chain bench.
- ref15_spec.py: the standalone numpy bit-level spec and reference decoder for V1/V2 (thread-12 unit format).
- xcheck15.py: spec vs kernel reference, Q2 vs the harness LUT, and thread 12's Q4 vs the fold.
- The G=4 build is `NQ15_G=4 python build15.py`.
