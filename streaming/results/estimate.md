# GLM-5.3 NestQuant + SSD streaming: end-to-end tok/s estimate (2026-09-28)

Inputs (all measured on this box unless noted):
- NQ MoE layer us, TP4 shard, 30% pool at level 4, recency routing: `sm120/results/est_bench.json`
- prod non-MoE ms/step: 23.9 at c=1 (prod profile), +5.8 ms per extra stream (derived from prod c=2 66 ms and c=4 113 ms steps; c=8 extrapolated)
- MTP acceptance: prod code tok/step 3.51 (c=1), 3.2 (c>1)
- prefill: prod TTFT rates (131K 1663, 512K 1563 tok/s), prod trace MoE share 26%, NCCL 39%
- SSD 13.94 GB/s stripe nvme0+nvme1, 4 GPUs, QD2; NCCL 8192-token collectives +24% under that load; pinned H2D 169 GB/s (4 GPUs) but NCCL x2.8 under it
- prefill routing windows from tb4 routing logs: `streaming/results/prefill_share.json`

## Decode (code, MTP ns=3)
| streams | all 2-bit | NQ + SSD (share 65.6..56%) | all 4-bit (no fit) | prod hybrid |
|---|---|---|---|---|
| 1 | 122 | 111 | 110 | 84 |
| 2 | 163 | 148 | 141 | 97 |
| 4 | 213 | 185 | 172 | 113 |
| 8 | 251 | 214 | 194 | - |

## Prefill, 8192-token chunks
| strategy | 4-bit share | tok/s @131K | TTFT 131K | tok/s @512K | TTFT 512K |
|---|---|---|---|---|---|
| prod hybrid | - | 1663 | 79 s | 1563 | 335 s |
| resident levels only | 59% | 1705 | 77 s | 1600 | 328 s |
| budgeted SSD streaming | 91-92% | 1547 | 85 s | 1460 | 359 s |
| full 4-bit, 8192 chunks | 100% | 878 | 149 s | 878 | 597 s |
| full 4-bit, 16384 chunks | 100% | 1547 | 85 s | 1460 | 359 s |
