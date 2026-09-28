# Update 03 for the SM120 agent: the hot (level-4) rate is 4.1263 bpw (2026-09-28)

## Production level 4
- **Base:** 2.0143 bpw. Unchanged; it is the level-2 plane.
- **Residual P4:** gate/up K = 2 (uniform mul1, no mask); down K = 2.3125 (KA 2, MASK 0x9248, 592 bits per 256-weight ring). With δ scales, level 4 totals **4.1263 bpw**.

## Why not 4.02
4.0221 bpw was a thread-14 test point: gate/up K 1.875 (KA1, 0xFEFE) with down K 2.25 (KA2, 0x8888), costing the same bytes as uniform K2. On the 9-expert check it trails EXL3-4 by 3.7% routed, 4.7% forced and 5.2% OOD (geomean), so it was rejected. 4.0846 (gate/up 1.9375 + down 2.3125) also fails; its E36-only win didn't hold on the other experts.

4.1263 with the single-pass encode beats EXL3-4 on every one of the 9 experts, on routed, OOD forced and OOD routed. Mean / worst %: −3.17 / −1.43, −2.33 / −1.75, −2.32 / −0.83. It is now the default. Source: T12 9-expert check, threads/12-reference-encoder; background in threads/14-level4-floor/REPORT.md.

## What changes for you
| | at 4.02 | at 4.1263 |
|---|---|---|
| P4 + δ, bpw | 2.008 | 2.112 |
| per TP4 expert-shard | 2.26 MiB | 2.38 MiB |
| per expert | 9.48 MB | 9.97 MB |
| 30% upgrade pool | 55 GB (13.8 GB/GPU) | 58 GB (14.5 GB/GPU) |

- SSD bytes per upgrade go up ~5%. Redo the bandwidth and pool-vs-KV sizing at 2.38 MiB per record.
- Kernel residual codes needed: K2 and 2.3125 (0x9248) now. Keep 1.875 / 1.9375 / 2.25 compiled too, because the artifact may choose the split per expert later (48-expert run in progress). Size the slot for the largest code and keep it a parameter.
- Bit-exactness with 0x9248 is the check that catches mask-indexing bugs: width at ring position p = `KA + ((MASK >> (p % 16)) & 1)`.
