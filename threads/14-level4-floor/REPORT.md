# Thread 14: nested level 4 at a fixed rate (written by the lead from the agent's final message)

## Result
The fixed-rate pattern residual does not reach EXL3-4 parity across the 9 experts. On T12's frozen level-2 stack at 4.0221 bpw (g/u K 1.875 = KA1/0xFEFE, down K 2.25 = KA2/0x8888), against T12's uniform nested residual:

| geomean over 9 experts | pattern / nq | pattern / EXL3-4 | experts improved |
|---|---|---|---|
| routed | 0.985 | 1.037 | 6/9 |
| forced | 0.9985 | 1.047 | 5/9 |
| OOD | 1.0045 | 1.052 | 3/9 |

It helps all three L16 experts on every metric (−1.7 to −4.4% routed). At L49 and L66 it mostly moves error from forced/OOD to routed, and it is worse everywhere on 49:36, 66:36 and 66:165. The best split depends on the expert, so the earlier E36 result over-generalised. Every variant keeps the base plane and L2 byte-identical and decodes bit-exactly against ref15_spec (0 mismatches).

E36 only, g/u 1.9375 (KA1, 0xFFFE) + down 2.3125 (KA2, 0x9248) at 4.0846 bpw: 8.801 / 9.107 / 9.637 routed/forced/OOD, vs EXL3-4 8.897 / 9.172 / 9.699 (4.0104 bpw) and EXL3-4 at matched bytes 8.626 / 8.900 / 9.441. Not run on the other 8 experts; parity at L49/L66 isn't expected without a per-expert split.

T12's frozen base is over the 0.5% L2 limit on some experts: L2 vs EXL3-2 +3.2% (16:165), +1.9% (49:165), +0.8% (66:92).

## Pattern-rate trellis rule
Width at ring position p: `w_p = KA + ((MASK >> (p % 16)) & 1)`. T12's Viterbi step i maps to p = (−i) mod 256, so the step into i shifts in `KA + ((MASK >> ((-i) % 16)) & 1)` bits. Using bit(i mod 16) is wrong for asymmetric masks (0x8888: 32,580/32,768 state mismatches); 0xAAAA, 0xEEEE, 0xFEFE, 0xFFFE are symmetric and hide the bug.

| K | (KA, MASK) | bits per 256-weight ring |
|---|---|---|
| 1.875 | (1, 0xFEFE) | 480 |
| 1.9375 | (1, 0xFFFE) | 496 |
| 2.25 | (2, 0x8888) | 576 |
| 2.3125 | (2, 0x9248) | 592 |

LDLQ_DRIFT: 1.875 → 1.02225, 1.9375 → 1.0201, 2.25 → 1.0135, 2.3125 → 1.0124. Decode cost is the same as K2 (compile-time widths).

## Dead ends (E36, thread-02 testbed)
- Conditional residual codebook and distribution-matched LUT: the residual is Gaussian (kurtosis 3.02) and independent of base state and neighbours, so ≤0.01 dB is available.
- Residual/base gain scaling and per-projection λ: no gain beyond noise.
- Interleaved half-bit blocks per projection: 9.410, worse than uniform, because mixing rates across blocks costs a Jensen penalty. A pattern trellis spreads the fractional rate uniformly.
- λ 0.7: L4 8.95 but breaks the L2 limit (35.5–35.7).
- Pattern base 1.875/2.25 + r2 improves L2 itself to 33.99 (−1.3%); an option only if L2 is unfrozen.

Down's share of output MSE is ~49% at both 2 and 4 bit, which puts the optimal extra rate near +0.25–0.3 bit on down (thread 16 estimated +0.5). Down's share is smaller on forced/OOD tokens and at deeper layers.

## Recommendation
Keep the pattern masks in the kernel and use them instead of interleaved K mixes. Choose the per-projection split per expert (a few header bytes); the sensitivity shares are the likely predictor, untested. Parity with EXL3-4 looks reachable around 4.08 bpw, but EXL3 at the same bytes stays 2–4% ahead.

Files: t12pat.py (pattern residual on T12's stack), patvit.py (pattern Viterbi), results_t12pat/L{L}_E{E}.json; earlier experiments t14.py, alloc.py, cond_diag.py, gain.py, sens.py, lamp.py, nine.py. Scratch in /tmp/nestquant/14-level4-floor/t12pat/.
