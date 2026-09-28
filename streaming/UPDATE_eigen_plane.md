# Update for the SM120 agent: the low-rank "eigen plane" is in production (2026-09-28)

Starting with T12 commit f128e41, every expert artifact can carry a low-rank correction per projection. Your kernel
has to add it, or the artifacts decode to the wrong weights. The format is DESIGN.md item 7, and the reference is
`threads/12-reference-encoder/nq_decode.py` (`lr_term`, `apply_ocol`). The T13 port is in
`threads/13-moe-layer-kernel/nqmoe.cu`, and INTEGRATION.md §1 has a "Low-rank plane" section.

## Math (exact spec)
For each projection p in gate, up, down, with rank r_p ∈ {0..4} (mostly 1, max 2/2/1 seen so far, r = 0 on many experts):
- V fp16 [r, in], in the **unrotated** input basis. U2 fp16 [r, out] belongs to the base plane, U4 fp16 [r, out] to the P4 plane.
- Dense decode, in fp32: `L2 = W2 + f32(U2)ᵀ f32(V)` and `L4 = W4 + f32(U2)ᵀ f32(V) + f32(U4)ᵀ f32(V)`. W2 and W4
  here are the usual dense weights after `dense_from_rotated`, in the original basis.
- GEMV form (what a kernel does): `z = x · Vᵀ` (r dot products per token, fp32), then `y += z · U2` (+ `z · U4` at level 4).
  - **gate / up**: x = the layer input hidden state, unrotated, i.e. before su · WHT. Add the term to the gate/up
    output in the unrotated output basis: after the inverse rotation (WHT · sv) and **before SwiGLU**.
  - **down**: x = the **unrotated SwiGLU output** silu(g) · u, i.e. before su_d · WHT and before any fp16 rounding
    for the down GEMV input. Add the term after the output rotation (WHT · sv_o) and before the routing weight.
- gate and up **share V**: up's meta has `lr.shared_V = True`, and the shard files store `lrV_from = "gate"` instead
  of a V. Only U differs, so one set of z dots serves both.
  - Older encodes (e.g. the pat9 `*lr.pt` test artifacts) have separate V. T13 handles them by stacking
    V = [V_g; V_u] and zero-padding U, which needs r_g + r_u ≤ 4. Production encodes are shared.
- Keep U2 + U4 in fp32. On L30 E169 down, U4 largely cancels U2 (|U2 + U4| = 0.23 |U2|), so the level-4 term is a
  small difference.

## Where it lives in the files
- Artifact (`experts/E*.pt`): `P["base"]["lr"] = {V, U2}` and `P["p4"]["lr"] = {U4}`, with `P["meta"]["lr"]` = {r, shared_V, ...}.
  No `lr` key means r = 0.
- Shard files (`nq_layer.split_expert`, `tp{s}.pt`): per projection, `lrV` / `lrU2` / `lrU4`. The manifest records
  `per_expert.lr_rank` and `lr_bytes_per_shard`.
  - gate / up are **out-sharded**: V is full (up has `lrV_from="gate"`), and U2 / U4 are this shard's output columns.
  - down is **in-sharded**: V is this shard's input columns, and U2 / U4 are full (replicated).
- The files are built for NSH = 8. For TP4, slice the same way, or concatenate pairs of tp shards: V_d along columns,
  gate/up U along columns.

## TP (the important part)
- gate/up: each rank has the full V and its own U columns. The term is exact locally.
- down: each rank computes a **partial** z_s = h_s · V_sᵀ over its own I columns and adds z_s · U_d to its partial
  output. Σ_s z_s · U_d = (Σ_s z_s) · U_d, so **the existing row-parallel output all-reduce already sums it**.
  - No extra all-reduce of z is needed. DESIGN.md item 7 and the nq_layer comment ("all-reduce z") describe a reduce
    you do not need.
  - T13 checked this on the real tp0..7 shard fields of L30 E169: Σ_s kernel z_s matches z from the full V_d to
    1e-5 (L2) and 1e-4 (L4).

## Kernel notes from the T13 port (A100, fused, no extra launch)
- Per-expert ranks go in the expert table (T13: [15] = r_gu | r_dn << 8), along with one pointer to an fp16 block
  `V_g [r_gu,H] | U2_g | U2_u | U4_g | U4_u [r_gu,I] | V_d [r_dn,I] | U2_d | U4_d [r_dn,H]`. With r = 0, nothing is read.
- gate/up z: one row block per 128-column gate group adds its k-slice's partial dot, reduced with a warp sum and
  atomicAdd. The group's epilogue applies z · U and re-zeroes z.
- down z: each 128-column SwiGLU epilogue stores its partial, once per call. The down combine sums the I/128 partials.
  Storing partials instead of using atomics avoids a zeroing counter, which cost about 1 µs per layer even at r = 0.
- Issue the z and U loads **before** the accumulator load / zero-store in each epilogue. Otherwise the compiler cannot
  reorder them across the store, and each token pays about 1 µs of serial latency. At the shard this was about 40% of the down-lr cost.
- Measured cost (TP8 shard I = 256, production 4.1263 level 4, every expert at the given rank, cold L2): see the table
  below. Real checkpoints are mostly r = 1 with many r = 0, so the average cost is lower.

| µs per layer (A100, top-8, B = tokens) | B1 | B2 | B3 | B4 |
|---|---|---|---|---|
| shard 4q, pre-lr build | 53.4 | 69.1 | 98.1 | 121.6 |
| shard 4q, r = 0 | 53.0 | 69.5 | 100.1 | 124.0 |
| shard 4q, r = 1 | 56.3 | 72.9 | 102.9 | 127.7 |
| shard 4q, r = 4 | 61.4 | 82.6 | 116.2 | 142.5 |
| shard EXL3 4-bit | 111.3 | 126.3 | 142.4 | 158.1 |
| full 4q, r = 0 / 1 / 4 | 235.2 / 238.0 / 251.3 | 342.0 / 345.6 / 358.6 | 439.1 / 445.2 / 470.4 | 551.7 / 557.8 / 582.4 |

- The level-2 shard costs are the same size: r = 1 adds 0.4 to 5.4 µs.
- r = 1 is roughly 1–6% at the shard and 1–2% at the full shape. r = 4 is roughly 15% and 5–9%.
- The full table is in INTEGRATION.md.
- **Timing trap on a time-sliced shared GPU.** Timing the second of two back-to-back graph replays gave a fake 1.6–4x
  step as soon as a replay passed about 1.05 ms, i.e. two replays crossing one time slice of about 2.1 ms. Time
  replays of under 1 ms, each started from idle.

## Bytes
2 · (r_gu · (H + 4·I_shard) + r_dn · (I_shard + 2·H)) per expert-shard, for H = 6144:

| r_gu = r_dn = 1 | per expert-shard |
|---|---|
| full expert (I 2048) | 57,344 B |
| TP4 shard (I 512) | 41,984 B |
| TP8 shard (I 256) | 39,424 B |

- DESIGN.md quotes +0.0135 bpw mean (max 0.027) on dist48, and 12–25 KB per expert per TP8 shard. The resident block
  in the T13 kernel holds both levels and replicates the down U, so it is larger: 39,424 B at r = 1/1.
- The U4 part (r_gu · 2 · I_shard + r_dn · H halves) belongs to the P4 plane. It can ride with the level-4 upgrade record
  if you stream P4 from SSD, or stay resident. T13 keeps the whole block resident.

## Test
Use the real artifacts at `/tmp/nestquant/12-reference-encoder/smoke_lr/{text,mmself}/L30/`: E168 (ranks 1/1/0) and
E169 (2/2/1), with tp0..7.pt and the manifest.
- Compare against `nq_decode.decode_expert(art, L)`, which includes lr through `apply_ocol`.
- The real lr term is only 5e-4 to 1e-2 of |y|. To test the lr path itself, scale all U2/U4 by 64 (exact in fp16).
  T13 gets 1e-4 rel err vs a same-rounding reference and about 9e-4 vs decode_expert, the fp16-activation floor
  (unchanged from before lr).
- Also run one launch with mixed levels (E168 at L2 and E169 at L4, and the reverse).
