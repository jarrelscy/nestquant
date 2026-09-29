# Kernel variants NQ_PREFETCH / NQ_PDL (not adopted)

This is a snapshot of the dev-tree decode kernel with the NQ_PREFETCH and NQ_PDL compile-time variants, plus the bench script and the L10 results from 2026-09-29 08:35 UTC.

Measured on L10 (26 or 128 experts at level 4, B = 1/4/8):

| Variant | Change |
|---|---|
| PREFETCH | -0.5 to +1.1% |
| PDL | -3.2 to +1.4% |
| PREFETCH + PDL | +2.8 to +5.2% |

None was adopted. The file is based on the kernel as of 2026-09-29 00:56, before the prefill kernels, so diff it against main rather than dropping it in.
