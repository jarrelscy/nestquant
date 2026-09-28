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
