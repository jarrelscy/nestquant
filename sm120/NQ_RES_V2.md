# nq-res-v2: pattern-rate base (b1.75) + residual code 9

This spec covers what the serving kernel (nqmoe.cu, owned by SM120) must add so it can serve the threads/35 b1.75/4
build. The host side is already in place: moe.py, nqload.py, streaming/resident.py, verify_res2.py and
threads/35-nq15/res2_testvec.py. Nothing changes for existing artifacts. All-K=2 layers still repack byte-identically
(gate (a) below). v1 res files load as base code 0, and table field [19] is 0 for them.

## What the b1.75/4 build is

| | base (level 2) | gate/up residual | down residual | total (L4) |
|---|---|---|---|---|
| v1 2-4 (today) | K=2, ring base (code 0) | K=2 (rk 0) | K=2.3125 (2,0x9248) (rk 7) | 4.10 |
| **b1.75/4** | **K=1.75 (1,0xEEEE) (bk 1)** | K=2.25 (2,0x8888) (rk 3, already compiled) | **K=2.5625 (2,0xD5AA) (rk 9, new)** | 4.10 |

L3-L6 also have in_had_down = 512 (threads/29 down rotation, exactly as shipped v1 L3-6). The kernel already supports
this through table [18], and it is carried in the res `.pt` as `in_had_down`.

(KA, MASK) works as everywhere else: step p of a ring uses KA + ((MASK >> (p % 16)) & 1) bits. The 16-bit window,
LSB-first tail-biting ring streams and greedy funnel are the same as the residual decode (nq_decode.PATTERNS,
ref15_spec.states). 0xD5AA = bres(9), the Bresenham 9-of-16 mask (threads/35 nq15.py).

## 1. New residual code 9: K = 2.5625 = (2, 0xD5AA)

- `RK<9> { KA = 2, M = 0xD5AA }`, RBITS = 4 * (16*2 + 9) = **164** bits per record.
- The P4 sub-array split is unchanged: n4 = 1 (uint4), then 36 bits remain, giving one uint, no uint2, no ushort, and
  a **4-bit tail** (T = 4, bit-packed over records, record r at bit 4r, +4 B pad).
  - Code 7 (148 bits) is uint4 + ushort + 4-bit tail. Code 9 is uint4 + **uint** + 4-bit tail, so it exercises the
    uint sub-array and the tail together.
- Plane bytes: S*R*32*164/8 + 4.
- Compile: down needs NQ_RK_DN |= 1 << 9. Gate/up uses code 3, which is already in the default 0x1C9. The host
  defaults were left unchanged. `MoELayer.set` asserts against `rk_codes()`, so a build without code 9 fails loudly at
  level 4 instead of decoding as code 0.

## 2. Base K code: table field [19] (previously reserved)

`e[19] = bk_gu | bk_dn << 8` (moe.entry). It uses the same numbering as RKP:

| bk | base | bits per record | base plane layout |
|---|---|---|---|
| 0 | K=2 ring base (today) | 128 | uint4 per record, unchanged. This equals the P4 sub-array layout at 128 bits (verify_res2 (c)). |
| 1 | K=1.75 (1, 0xEEEE) | 112 | **P4 sub-array layout**: uint2 [nrec], then uint [nrec], then ushort [nrec], each contiguous over all nrec = S*C*32 records. Record index ((strip*C + c)*32 + lane), as for the base today. No tail. Plane bytes S*C*32*14. |

Decode for bk = 1. Only the base state walk changes:

- **Level 2.** Base state at lane weight j: the 16-bit window starting at bit step_off(j, 1, 0xEEEE) of the
  tail-biting ring of G = 4 lanes (ring g = lanes 4g..4g+3, 4*112 = 448 bits = 256 weights x 1.75). This is
  `funnel_greedy` / `step_off` with KA = 1, MASK = 0xEEEE, the same template code the residual uses for RK<1>.
  `Q2 = fp16(A (1024 + S(sb)) + B)` (mul1) is unchanged.
- **Base-variant sign plane ([12]/[13]).** Unchanged: uint8 per unit, bit g = sign of ring g, applied to Q2 and Q4.
- **Level 4.** The RM_P int-fold `F = (Mb S(sb) + N S(sr) + 128) >> 8` and the rest are unchanged. They only consume
  S(sb), which is K-independent.
- **lr plane, scales, Had128/Had512 transforms.** Unchanged.
- **Compile.** Suggested: an `NQ_BK_CODES` mask (bit c = base code c, code 0 always) and an exported
  `bk_codes() -> [gu_mask, dn_mask]`. `MoELayer.set` already calls `M.bk_codes()` when the extension has it. Without
  it, set() assumes [1, 1] (base code 0 only) and refuses bk != 0 experts.

Reference implementation (bit-exact spec of the above): `moe.lane_vals` / `moe.dense_W` (the bk branch) and
`moe.unpack_words` (the inverse of `pack_words`). They are checked against the independent `ref15_spec.decode_unit`
(Kb = (1, 0xEEEE), Kr = (2, 0xD5AA)) by `sm120/verify_res2.py`.

## 3. Files

- **Record file `rank{r}.bin` / `rank{r}.json` (nq-p4rec-v1): unchanged format.** Same segments (gu.p4 | gu.d4 |
  dn.p4 | dn.d4 | lr4) and the same addressing ((L - L0)*256 + E) * rec_bytes. The P4 sub-arrays of codes 3 / 9 are
  just longer:
  - b1.75: **rec_bytes = 2,854,912** (v1: 2,560,000).
  - seg (TP4, I = 512): gu.p4 (0, 1769472), gu.d4 (1769472, 12288), dn.p4 (1781760, 1007620), dn.d4 (2789632, 6144),
    lr4 (2795776, 57344).
  - rank{r}.bin = 75 x 256 x rec_bytes = 54.8 GB (51.05 GiB) per rank. That is under HF's recommended 200 GB per
    file and the 500 GB hard limit.
- **Resident file `res/rank{r}/L{L}.pt`: `nq-res-v2`.** It is written only when some expert of the layer has a base
  code != 0; otherwise it is `nq-res-v1`, byte-identical to before. Differences from v1:
  - `format = 'nq-res-v2'`.
  - `bk_gu`, `bk_dn`: per-expert int lists (RKP numbering).
  - `gu_base` / `dn_base` [E, n_int32]: the bk layout above (bk 1: S*C*32*14 bytes per projection, 112/128 of v1).
  - Everything else (rk_gu / rk_dn, var, sc2 / sc4, lr, rg / rd, in_had_down) is as in v1.
  - `resident.load` accepts v1 and v2 and sets `p.bk` (v1 -> 0), and `moe.entry` writes [19].
  - A server older than this commit refuses a v2 file at the format assert. It never silently decodes a 1.75 base
    as K=2.
- `artifact_stamp.json`: unchanged (key = sha256 over the per-layer manifest.json, as in sm120/eval/run_c2.sh).

## 4. Test vectors: `/tmp/nestquant/35-nq15/testvec/`

Each `L{L}_E{E}_r{r}[tag].pt` is a torch.save dict. A `.json` copy of `meta` sits next to it.

- `meta`: layer / expert / rank, H = 6144, I = 512, rk_gu / rk_dn, bk_gu / bk_dn, rg / rd, in_had_down, base_K, res_K,
  `seg`, `rec_bytes`, res_format, and sha256 of each expected W.
- `record`: uint8 [rec_bytes], the exact bytes `repack.py` writes for this (L, E) on this rank.
- `res`: this expert's row of the res file: gu_base, gu_var, dn_base, dn_var, sc2, sc4, lr.
- `W`: expected dense decoded weights, fp16, rotated domain, i.e. what the NQ_WDUMP build dumps.
  - `gu2`, `gu4` [2I, H]: gate rows 0..I-1, up rows I..2I-1.
  - `dn2`, `dn4` [H, I].

| file | what it exercises |
|---|---|
| `L10_E0_r0.pt` | b1.75: **bk 1** (gate/up + down) + rk 3 (gate/up) + **rk 9** (down), Had128 |
| `L3_E0_r0.pt` | b1.75 + **in_had_down 512** (T29), bk 1, rk 3 / 9. Written when the campaign reaches L3 (it encodes L3-6 last). |
| `L7_E0_r0.pt` | b1.75, same codes as L10 (first layer finished; extra vector) |
| `L10_E0_r0_v1.pt` | shipped v1 L10 (bk 0, rk 0 / 7, nq-res-v1): regression vector for the unchanged path |

Kernel test: put `res` into a level-2 row and `record` into a slot (p4rec.row), dump the decode (NQ_WDUMP), and compare
to `W` bitwise at level 2 and 4.

How they were produced: `threads/35-nq15/res2_testvec.py`. It checks two things before writing:

1. Host dense decode of the nqload kernel planes == the encoder-side decoder (nq_decode.ring_levels, base_K-aware via
   threads/35 nq15), bitwise.
2. The res-file + record bytes round trip decodes to the same W.

## 5. Validation status (host side)

- `sm120/verify_res2.py` (CPU): pack/unpack round trip for codes 0-9; torch ref == ref15_spec on 128 units x 2 levels
  for bk {0, 1} x rk {0, 3, 7, 9} with base-variant signs and exhaustive (Mb, N); the bk 0 layout is unchanged.
- **Gate (a), regression.** origin/main code vs this code, `streaming/repack.py` on shipped v1 L3 (Had512) and L10,
  all 4 ranks: rank{r}.json, both layers' record regions and res files are byte-identical. Run on CPU with the CUDA
  build stubbed; repack never launches the kernel.
- `res2_testvec.py`: checks 1 and 2 above pass on b1.75 L7 and L10, and on v1 L10.
- Not done (SM120): the kernel implementation of [19] / RK<9> and the WDUMP check against these vectors.
