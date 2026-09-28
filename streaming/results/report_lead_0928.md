# SM120 streaming: results for the lead (2026-09-28)

All numbers are from the tb4 routing logs (/data/Jarrel/routing_logs/glm5.3-arvq-v2), with the thread-22 fixed set used as is (26 experts per layer) and a 51-expert floating set, refreshed every 64 tokens with a 1-window lag unless stated. GB/s is SSD read at 111 tok/s single stream with P4 = 9.97 MB per expert. It scales linearly with tok/s.

## 1. Floating-set selection sweep (select_sweep.log)

| score | window | refresh | route share | activated hot | GB/s @111 |
|---|---|---|---|---|---|
| box | 256 | 64 | 61.25% | 41.1% | 10.55 |
| box | 512 | 64 | 61.62% | 41.1% | 6.04 |
| box | 1024 | 64 | 61.58% | 41.1% | 3.43 |
| box | 2048 | 64 | 61.29% | 40.9% | 1.93 |
| box | 4096 | 64 | 60.91% | 40.7% | 1.08 |
| EMA | 128 | 64 | 62.30% | 41.5% | 10.03 |
| EMA | 256 | 64 | 62.56% | 41.5% | 5.93 |
| EMA | 512 | 64 | 62.43% | 41.4% | 3.40 |
| EMA | 1024 | 64 | 62.04% | 41.2% | 1.92 |
| box | 1024 | 32 | 62.17% | 45.7% | 4.37 |
| box | 1024 | 16 | 62.60% | 50.3% | 5.69 |
| EMA | 256 | 32 | 63.59% | 46.6% | 7.69 |
| EMA | 256 | 16 | 64.37% | 51.7% | 9.96 |

Under the 6 GB/s cap:
- At 111 tok/s, EMA 256 @64 has the highest share (62.56%, 5.93 GB/s).
- EMA 512 @64 is 0.13 pt behind at 3.40 GB/s.
- The best fixed window, box 512, is 0.94 pt behind EMA 256. That is outside the 0.5 pt rule, so EMA wins.

Recommendation: EMA 512 @64. It is within 0.13 pt of the best, and it stays under 6 GB/s up to about 195 tok/s. EMA 256 crosses 6 GB/s at about 112 tok/s.

Faster refresh raises activated-hot a lot (41% → 50% at @16) for about 1–2 pt of route share, but costs 1.7–3x the SSD bandwidth.

## 2. Start of request (seed_streams.json)

Over 149 requests (137 with ≥1024 computed prefill tokens), measured on the first 1024 decode tokens:

| start | route share, first 1024 decode tokens |
|---|---|
| seeded from the last 1024 prefill tokens | 54.5% |
| floating_default only (top 51 non-fixed by n_routed) | 60.7% |

Seeding from prefill is worse in every 64-token window until about token 640, and it is better for 0% of requests. Prefill tokens (prompt, tool output, code) route differently from the decode tokens that follow.

Recommendation: start every request from floating_default and let the decode window take over. Don't seed from prefill.

## 3. Several streams, one shared pool on summed counts (seed_streams.log)

| streams | mean share | lowest stream (mean) | lowest stream (min window) | GB/s | aggregate tok/s |
|---|---|---|---|---|---|
| 1 | 62.3% | 62.3% | 49.5% | 3.28 | 111 |
| 2 | 61.8% | 59.8% | 54.5% | 1.88 | 148 |
| 4 | 62.3% | 58.5% | 42.6% | 0.94 | 185 |
| 8 | 62.2% | 54.7% | 39.9% | 0.44 | 214 |

The mean holds up, but the lowest stream drops by 7.6 pt at c = 8. SSD bandwidth falls with more streams because the summed counts change more slowly.

## 4. Boundary tokens (boundary_share.json, 263 requests)

| tokens | fixed | fixed + floating |
|---|---|---|
| all | 13.2% | 60.9% |
| 1 before `</think>` | 34.3% | 70.4% |
| 2–4 before `</think>` | 20.4% | 62.1% |
| 1 before end of turn | 28.9% | **49.5%** |
| 2–4 before end of turn | 30.4% | **50.2%** |

End of turn is low: 11 pt under the average, even though the fixed set covers 2x more there than on average. The floating set misses the experts used at end of turn, because they are rare in the preceding window. If end-of-turn quality matters, add the top end-of-turn experts to the fixed set.

## 5. Format and kernel

- The kernel now has the T12 low-rank plane (f128e41).
  - Gate/up: z = x Vᵀ, added in K1's finisher before SwiGLU.
  - Down: each rank adds its own partial z_s U. U is replicated, so the MoE all-reduce sums the partials. This matches the nq_layer note ("all-reduce partial z before U") by linearity, with no extra all-reduce.
  - Table is now 20 words: [14] resident lr, [15] U4, [16]/[17] ranks.
  - The kernel needs gate and up to share V (true when their H is the same).
- Production-format smoke, sm120/smoke_prod.py (results/smoke_prod_lr.log), passes 75/75 checks on L3 E0–4 with lr ranks 0–4 on each side:
  - shard sizes match INTEGRATION.md;
  - manifest lr_rank is correct;
  - assemble decode matches;
  - kernel decode is bitwise equal at TP8/TP4/TP1;
  - forward vs the dense expert with lr has rel err 0.9–1.3e-3.
- P4 slot per TP4 expert-rank is 2,554,112 B (2.44 MiB), against 2.38 MiB in update 03. The extra comes from:
  - block words, which are u32 in the kernel (+9 KB);
  - U4 of the lr plane (up to 56 KB at r = 4), kept in the slot so that level 2 needs no U4.
- A dummy production-format fit of all 75 layers is running on the SM120 box (/rawdata/Jarrel/nq-glm53-prod). It uses H = I, a random dummy V with rank 0–4, and the exact nestquant-v1 layout, and should finish around 03:00 UTC on 2026-09-29.
