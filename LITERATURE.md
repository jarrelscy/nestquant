# Prior work relevant to NestQuant (web search, 2026-09-28)

## Multi-precision / nested formats for LLMs
- Any-Precision LLM (Park et al., ICML 2024, arXiv 2402.10517): SqueezeLLM seed at 3 bit, each extra bit splits every cluster in two (weighted k-means). Lower bits are a prefix of higher bits. Stored as bitplanes, so a 3-bit read only fetches 3 planes. Matches independently quantized models at 3–8 bit. Scalar non-uniform codes, far from trellis quality at 2 bit.
- AnyBCQ (ICLR 2026, arXiv 2510.10467): binary-coded bitplanes, earlier planes frozen, new plane + new scales fitted on the residual. Best at 2 bit; the shared-binary constraint costs quality at 3–4 bit.
- MatQuant (DeepMind, arXiv 2502.06786): int8 MSB slicing with a joint loss over int8/4/2 (QAT/OmniQuant). int2 gains, int4/int8 at parity.
- MatGPTQ (IST, arXiv 2602.03537): one-shot GPTQ with a joint multi-precision objective and cross-bit error compensation; ~5x GPTQ cost; EvoPress bit allocation; kernels released (github.com/IST-DASLab/MatGPTQ). Closest published analogue of thread 02.
- Drop-by-Drop (arXiv 2606.12876): AQLM additive codebooks (8-dim groups, 256-entry codebooks, ~1 bpw per stage), joint loss Σ λ_k D_k with λ on the endpoints (λ3=λ5=0.5) working best. Proves Gaussian W is successively refinable under the ||(W−Ŵ)X||² metric (nested reverse water-filling). Plain codebook dropping from AQLM/QuIP# collapses (QuIP# 4→2: ppl 95). Lowest level tested is 3 bit.
- RRQ (arXiv 2608.04048): 2-bit base + 2-bit RTN residuals. At 4 bit it usually loses to one-shot 4-bit RTN (−0.4 to −1.3 task points). No kernels. Evidence that a naive base+residual loses at the top end.
- AWSRC (arXiv 2608.23144): residual repair file with seed-generated bases, globally ranked progressive prefixes by activation/Fisher-weighted error per byte (+0.16 bpw over RTN-INT4 closes 88% of the ppl gap). Relevant to byte-ranked streaming.

## Successively refinable trellis coding (signal processing)
- Multistage TCQ (quantize the residual with a second TCQ) loses ~2 dB vs single-stage TCQ.
- Jafarkhani & Tarokh, "Design of successively refinable trellis-coded quantizers", IEEE T-IT 45(5) 1999 (US6125149): multi-level trellis in which each transition of the level-k trellis is replaced by a trellis at level k+1; hierarchical set partitioning; codevectors optimised jointly for all levels. Good results on memoryless sources, but heavy computation.
- Brunk & Farvardin, Embedded TCQ (DCC 1998): beats MS-TCQ; loss vs TCQ shrinks with more states.
- Universal refinable TCQ (IEEE 4976475): scalar quantizer + improved E-TCQ for many stages.
- No published successively refinable version of the QTIP/EXL3 bitshift trellis was found. This is an open gap.

## Trellis / lattice LLM quantization, 2025–2026
- QTIP (NeurIPS 2024) remains the 2-bit PTQ state of the art per BCJR-QAT (May 2026).
- BCJR-QAT (arXiv 2605.10655): soft trellis via forward-backward at temperature T, differentiable; fused Triton. Useful for thread 09 (tune codes/LUT end-to-end).
- HARP (arXiv 2605.29843): learned rotation replacing the random Hadamard in QTIP; 2-bit Llama-2-7B ppl 6.87→6.62 at +0.10 bpw.
- YAQA / Model-Preserving Adaptive Rounding (Tseng et al., arXiv 2505.22988): Kronecker-factored full-layer Hessian (input and output side) from the model's KL; ~30% lower error than LDLQ with any quantizer including QTIP. exllamav3 1.5.1 (the installed version) has "highly experimental YAQA support". Relevant to threads 06 and 08.
- GPTQ-2D (arXiv 2607.27042) and BaKron (arXiv 2608.06291): exact two-sided (Kronecker) rounding in cubic time.
- Price of metric universality (arXiv 2602.05790): a single universal codebook loses at most 0.11 bit/dim vs a codebook tailored to the input covariance. Bounds how much Hessian-specific codebooks can buy; supports robustness under OOD covariance.
- GLQ (github.com/cnygaard/glq): QTIP-derived trellis + E8 lattice, vLLM kernels; E8 4 bpw = 2+2 residual; trellis 5–8 bpw = K4 + residual trellis costs 1.9–2.7x decode at B1. Evidence that two-pass trellis decode is expensive unless fused carefully.
- Leech-lattice kernels (arXiv 2609.02652): large-codebook lattice decode is 2.27x slower than QTIP at 2 bit; bitplane layouts beat one-hot masks.
- HyperQuant (arXiv 2606.23406): RHT + E8/D4 lattice + Rice entropy coding, beats HIGGS at 3–5 bits.
- NestQuant (Savkin et al., ICML 2025, arXiv 2502.09720): nested Gosset lattice for W+A+KV. **Name collision with this project.**

## MoE dynamic precision systems
- HOBBIT (arXiv 2411.01433): loads INT2/INT4 experts on cache miss, importance from gating magnitude.
- DynaExq (arXiv 2511.15015): hotness from router traces, per-layer high-precision resident set, asynchronous promotion/demotion; Qwen3-80B 73.1→77.6% at equal memory.
- DyMoE (arXiv 2603.19172), MxMoE (arXiv 2505.05799), router-norm bit allocation (arXiv 2604.06515).
- exllamav3 PR #392: pinned-arena CPU MoE, gathers selected experts over PCIe into staging and runs the fused MoE kernel through pointer tables; reaches link rate (25 GB/s gen4).
- exllamav3 1.5.0 (installed 1.5.1): faster MoE decode (+4% GLM5.3-Flash 2 bpw) and a fractional-trellis mode in 1.5.1.

## Implications for the design
1. The literature supports the joint objective (MatGPTQ, Drop-by-Drop λ on endpoints) over a greedy base + residual (RRQ, codebook dropping).
2. Hierarchical trellis (Jafarkhani–Tarokh) is the direct precedent for a successively refinable bitshift trellis; nobody has built one for LLMs.
3. Output-side/Kronecker Hessians (YAQA) are the largest known algorithmic gain on top of any quantizer, and the installed exllamav3 already experiments with it. Any EXL3 comparison must state whether YAQA was used.
4. Two-pass decode costs ~2x in published kernels; the refinement must share the base's decode pass.
