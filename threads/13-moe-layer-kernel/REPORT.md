# Thread 13: NestQuant grouped MoE layer kernel (written by the lead from the agent's final message)

Code: nqmoe.cu, moe.py, build.py, check.py, bench_full.py/.json, bench_shard.py/.json, tune.py/tune_I2048.json, levelswitch.py/.json, levelswitch_mbox.py/.json, copybw.py/.json, probe_pers.py, t_mbox.py, INTEGRATION.md. Scratch /tmp/nestquant/13-moe-layer-kernel/.

## Kernel
- Two graph-capturable launches per layer. K1 gate|up: dedups experts across tokens, input rotation, tensor-core split-K GEMV, SwiGLU + down input rotation fused in the epilogue, writes h fp16. K2 down: GEMV, output rotation and routed-weight combine fused.
- Per-expert table read at replay: level 0 (not resident), 2 or 4. Level 3 removed. Optional per-128x128-block 2b/4b mask, free when unused. Decoder is thread 04's nqk2 additive decoder as a swappable device function.
- __launch_bounds__(256,4) gave ~1.5x. Persistent work-queue variant correct but 3-12% slower; not used.
- check.py: rel err 3.7e-5..7.9e-5 vs dense decode for mixed levels, B1-4, several routings, I=2048 and I=256. Workspace zero on exit.

## Latency vs EXL3 1.5.1 exl3_moe_coop (µs per layer, NQ / EXL3, cold L2, recency routing p=0.4, top-8, random weights)

Full shape H=6144, I=2048:

| B | distinct | 2b | 4b | 50/50 mix |
|---|---|---|---|---|
| 1 | 8 | 143.7 / 248.8 (1.73x) | 235.1 / 255.1 (1.09x) | 185.7 / 321.1 (1.73x) |
| 2 | 12.25 | 213.4 / 335.1 (1.57x) | 380.5 / 340.1 (0.89x) | 299.8 / 399.2 (1.33x) |
| 3 | 16 | 273.4 / 434.1 (1.59x) | 484.1 / 443.5 (0.92x) | 393.2 / 510.8 (1.30x) |
| 4 | 20.5 | 341.1 / 532.6 (1.56x) | 602.4 / 543.9 (0.90x) | 471.9 / 600.6 (1.27x) |

TP8 shard shape I=256, 256 experts resident:

| B | 2b | 4b |
|---|---|---|
| 1 | 34.6 / 100.3 (2.90x) | 49.7 / 102.6 (2.06x) |
| 2 | 45.7 / 116.6 (2.55x) | 66.0 / 119.8 (1.81x) |
| 3 | 59.1 / 130.7 (2.21x) | 85.7 / 134.4 (1.57x) |
| 4 | 74.2 / 145.8 (1.96x) | 110.2 / 150.3 (1.36x) |

Caveat: I=256 NQ config is best-of-sweep on the same run; EXL3's coop kernel looks poorly tuned for small I.

## Level switching under one graph
- Host-event version (levelswitch.py): 200 async steps, 201 upgrades / 198 downgrades, all replays match dense ref (max 6.3e-5).
- Mailbox version for vLLM full graphs (levelswitch_mbox.py): captured kernel at layer start applies rows staged by a side stream; 306 ops / 300 steps, each within one step, max 7.0e-5, 1.4 µs per layer.
- Copy bandwidth 576 KiB pinned chunks: 22.5-22.8 GB/s during MoE replay (~52 µs per expert-shard upgrade of 1.13 MiB), MoE slowdown 0.0% ± 0.3%.
- Routing-hit export: exact, +0.3 µs.

## Interface
See INTEGRATION.md: moe_forward(x, sel, rw, table[E,16], out, workspaces, I, nm_gu, nm_dn, cfgs, G=2, ...), mailbox(...). Table row: [0] level, [1..4] gate|up base/p4/d4/flags, [5..8] down, [9] signs. Limits B·topk ≤ 32, H, I multiples of 128. TP8: 256 experts per rank at I=256, base resident (288 MiB per layer per rank), P4 slot pool, pinned host holds 4b tier; out-of-tree FusedMoEMethodBase plugin modelled on nvfp4_aqlm_hybrid.

## Open items
1. 4b at B2-B4 full shape ~10% behind EXL3 (issue-bound, ~10.1 ops/weight). Lead: thread 15's int-fold decoder (7.15 ops/weight) is the fix to port.
2. Fitters must emulate fp16 δ-folding rounding (moe.dense_W EMU=True).
3. Split-K fp32 atomics: not bitwise repeatable.
4. vLLM dtype casts (bf16/int32/fp32 inputs), check routed_scaling_factor.
5. No GPU prefill (B > 4) decode kernel yet.
6. Mask flags uniform per 128-row group; at I=256 down has only 2 chunks per strip.
7. I=256 configs untuned; EXL3 I=256 baseline may be weak.
8. Persistent variant lacks hits export.
9. PCIe link sharing across GPUs on a switch.

## Update: RM_P decoder port (lead, from the agent's final message, 2026-09-28 21:50 AEST)
Thread 15's int-fold RM_P decoder and greedy-funnel 2-bit base are ported into nqmoe.cu (`nqdec::`). The per-block (Mb, N) value is one 32-bit word. The residual plane is split into uint4/uint2/uint/ushort arrays so loads stay coalesced at any residual K. Table slots [10] and [11] set the residual K for gate|up and down (codes 0=2, 1=1.75, 2=2.5, 3=2.25, 4=3, 5=1.5). `NQ_RK_CODES` picks which codes get compiled (default 0x7); a code missing from the build silently decodes as K=2. 4-lane rings (256 weights) are now the default.

Verification: verify15.py matches ref15_spec on 96 blocks, all K codes and both levels, with 0 mismatches. A decoded-weight dump matches the reference bit for bit on all 64 projections at I=2048 and I=256. check.py relative error is ≤1.15e-4. Split-K fp32 atomics are still not bit-reproducible run to run.

Full shape, I=2048, recency routing, NQ vs EXL3 µs:

| B | 2b | 4b | mixed |
|---|---|---|---|
| 1 | 132.7 vs 248.8 (1.87x) | 220.5 vs 255.1 (1.16x) | 1.86x |
| 2 | 192.6 (1.74x) | 320.7 vs 340.0 (1.06x) | 1.56x |
| 3 | 249.5 (1.74x) | 412.7 vs 444.2 (1.08x) | 1.52x |
| 4 | 313.7 (1.70x) | 516.7 vs 544.1 (1.05x) | 1.47x |

4-bit at B2–B4 was 0.89–0.92x before the port. Fractional K (gate|up 1.75, down 2.5) runs 1.03–1.15x.

TP8 shard, I=256, against EXL3's best option (`EXL3_MOE_COOP_KSPLIT=2` at B1–B3):

| B | 2b | 4b |
|---|---|---|
| 1 | 32.8 vs 74.9 (2.28x) | 46.2 vs 77.2 (1.67x) |
| 2 | 42.9 vs 98.0 (2.28x) | 65.3 vs 100.6 (1.54x) |
| 3 | 54.7 vs 128.9 (2.35x) | 85.7 vs 130.7 (1.53x) |
| 4 | 71.6 vs 146.0 (2.04x) | 109.2 vs 151.2 (1.38x) |

Level switching still works on the new layout: mailbox version 311 switches with at most 1 step of lag, host-event version 399 switches, max relative error ≤1.2e-4.

Open: the encoder must use `ref15_spec.fold` with a least-squares refit of (Mb, N) per block, and write the new residual layout. The kernel must be compiled with every K code the checkpoint uses. There is no dense prefill kernel yet. New files: verify15.py, tune2.py, abtest.py, exl3_sweep.py plus JSON results. Scratch is in /tmp/nestquant/13-moe-layer-kernel/.
