# Update 1 for the SM120 agent (lead, 2026-09-28 ~23:15 AEST)

Pull `main` first. Changes since PROMPT_sm120_ssd.md:

1. **Port `threads/13-moe-layer-kernel/nqmoe.cu`, not nqk15.** It now contains thread 15's RM_P decoder (`nqdec::`), bit-exact against `ref15_spec.py`. It is 4b 1.05–1.16x and 2b 1.70–1.87x vs EXL3 on A100 at the full shape, and 1.38–2.35x at the TP8 shard. INTEGRATION.md is updated.
2. **The residual layout has changed.** The per-16x128-block value is one 32-bit word `Mb | N<<8`. The residual plane is split into `uint4 | uint2 | uint | ushort` arrays so loads stay coalesced at any K. 4-lane rings (256 weights) are the default.
3. **The production 4-bit level is the pattern-rate residual.** gate/up uses K 1.9375 (KA1, MASK 0xFFFE, 124 bits/record), and down uses K 2.3125 (KA2, MASK 0x9248, 148 bits/record), for 4.0846 bpw total. E36 L4 is 8.801 routed vs EXL3-4 8.897. The alternative at 4.1263 bpw is gate/up K2 + down 2.3125. The mask rule at kernel ring position p is `w_p = KA + ((MASK >> (p%16)) & 1)`. The prompt's old "bit(i mod 16)" wording was wrong and is now corrected. See `threads/12-reference-encoder/nq_decode.py` PATTERNS. Thread 13 is adding these two codes to nqmoe.cu now.
4. **Gotcha:** table slots [10]/[11] carry the residual K code per projection, and `NQ_RK_CODES` picks which codes get compiled. A code missing from the build **silently decodes as K=2**, so compile every code the checkpoint uses and assert it at load time.
5. **Sizes:** at 4.0846 bpw, P4 + δ is ~2.07 bpw on top of the 2.0143 base, ~2.33 MiB per TP4 expert-shard. Keep the slot size a parameter. The final rate isn't locked until the 9-expert check finishes.
6. **Fair EXL3 baseline:** exllamav3 1.5.1's coop MoE kernel reads `EXL3_MOE_COOP_KSPLIT`. KSPLIT=2 is much faster for B1–B3 at small I (TP8 B1: 100→75 µs), so compare against EXL3's best setting per batch size. Re-check this on SM120.
7. **HF repo:** `jarrelscy/GLM-5.3-NestQuant-2-4bit` has no layers yet. The encoder is being sped up first, and the calibration is being redone in GLM's native chat format with boundary weighting. None of this changes the format. Keep using random planes plus `nq_encode.py` experts.
8. **Still open on A100 too:** there is no dense prefill kernel yet. Split-K fp32 atomics are not bit-reproducible run to run, which is expected.
