# Thread 18: full-model KLD / top-1 eval of the NestQuant GLM-5.3 encode

## Verdict (2026-09-29, first full-model measurement)
- Every expert at level 4 (nq4) comes within about 0.02-0.03 nats of FP8. In-domain perplexity moves by less than 0.4%, wikitext by 1%, and top-1 agreement is 94-96.5%.
- Every expert at level 2 (nq2) costs about 0.10-0.14 nats in-domain and 0.31 on wikitext. Perplexity rises 7% in-domain and 23% on wikitext.
- The manifest default mix (nqdef) puts 26 experts per layer at level 4 (1950 of 19200, about 10%) and leaves the rest at level 2. It removes about a quarter of the nq2 KLD in-domain (0.130 -> 0.096 on nq-tail) and 20% on github, but only 7% on wikitext.
- L3-L6 are not what limits the mix at full-model level:
  - L3-L6 at level 2 on their own, with everything else FP8 (nq2_early), give 0.014-0.015 KLD.
  - Moving all of L3-L6 to level 4 inside the mix (nqdef_e4) changes nqdef's KLD by at most 0.0007, which is within the standard error. Errors from different layers do not add up.
- There were no fallbacks: every expert call in every candidate used its decoded weights.

## Results (KLD in nats vs the FP8 reference; ppl ref is FP8)
| cand | corpus | tokens | ppl ref | ppl cand | KLD mean | ± se | p99 | top-1 agree |
|---|---|---|---|---|---|---|---|---|
| nq4 | nq-tail | 262016 | 2.3009 | 2.3069 | 0.0227 | 0.0019 | 0.389 | 96.53% |
| nq4 | vllm-docs | 161713 | 2.5817 | 2.5897 | 0.0242 | 0.0012 | 0.305 | 95.43% |
| nq4 | wikitext | 135102 | 2.9398 | 2.9687 | 0.0334 | 0.0015 | 0.461 | 94.35% |
| nq4 | github | 51175 | 2.7390 | 2.7350 | 0.0202 | 0.0012 | 0.251 | 95.83% |
| nqdef | nq-tail | 262016 | 2.3009 | 2.4070 | 0.0965 | 0.0081 | 1.810 | 92.21% |
| nqdef | vllm-docs | 161713 | 2.5817 | 2.7139 | 0.1114 | 0.0069 | 1.489 | 90.23% |
| nqdef | wikitext | 135102 | 2.9398 | 3.5556 | 0.2874 | 0.0139 | 3.443 | 82.80% |
| nqdef | github | 51175 | 2.7390 | 2.7680 | 0.0807 | 0.0051 | 1.020 | 91.80% |
| nq2 | nq-tail | 262016 | 2.3009 | 2.4582 | 0.1305 | 0.0105 | 2.311 | 90.62% |
| nq2 | vllm-docs | 161713 | 2.5817 | 2.7598 | 0.1392 | 0.0084 | 1.784 | 89.08% |
| nq2 | wikitext | 135102 | 2.9398 | 3.6223 | 0.3105 | 0.0148 | 3.609 | 81.95% |
| nq2 | github | 51175 | 2.7390 | 2.7856 | 0.1016 | 0.0073 | 1.307 | 90.93% |

The two attribution arms below both concern L3-L6:
- nq2_early: L3-L6 at level 2, all other layers exact FP8.
- nqdef_e4: nqdef with all of L3-L6 moved to level 4.

| cand | corpus | ppl cand | KLD mean | ± se | p99 | top-1 agree |
|---|---|---|---|---|---|---|
| nq2_early | nq-tail | 2.3023 | 0.0149 | 0.0016 | 0.218 | 97.25% |
| nq2_early | vllm-docs | 2.5809 | 0.0147 | 0.0005 | 0.171 | 96.40% |
| nq2_early | wikitext | 2.9446 | 0.0136 | 0.0006 | 0.174 | 96.29% |
| nq2_early | github | 2.7368 | 0.0141 | 0.0008 | 0.161 | 96.65% |
| nqdef_e4 | nq-tail | 2.4064 | 0.0958 | 0.0080 | 1.779 | 92.21% |
| nqdef_e4 | vllm-docs | 2.7177 | 0.1110 | 0.0069 | 1.483 | 90.24% |
| nqdef_e4 | wikitext | 3.5547 | 0.2878 | 0.0139 | 3.428 | 82.74% |
| nqdef_e4 | github | 2.7692 | 0.0807 | 0.0049 | 1.017 | 91.80% |

## Router and per-layer divergence, by layer band
The columns are:
- route: router top-8 agreement with the FP8 stream at the same layer, over all tokens.
- local err: relative L2 error of the routed-expert output on the candidate's own input.
- dDiv: growth of the relative hidden-state divergence across the band.

| cand | L3-6 route / local err / dDiv | L7-40 | L41-77 |
|---|---|---|---|
| nq4 | 99.27% / 0.039 / 0.013 | 93.75% / 0.071 / 0.111 | 85.68% / 0.086 / 0.065 |
| nqdef | 98.35% / 0.166 / 0.036 | 87.92% / 0.270 / 0.185 | 75.76% / 0.306 / 0.127 |
| nq2 | 98.03% / 0.185 / 0.041 | 86.79% / 0.298 / 0.201 | 73.38% / 0.348 / 0.140 |
| nq2_early | 98.03% / 0.185 / 0.041 | 93.49% / (ref) / 0.080 | 87.19% / (ref) / 0.044 |
| nqdef_e4 | 99.27% / 0.039 / 0.013 | 88.27% / 0.270 / 0.210 | 75.79% / 0.306 / 0.124 |

Layers contributing most divergence (largest per-layer increments):
- nq2: L30, L7, L32, L37, L38, L6, L45, L39, L42, L31. Each adds +0.013 to +0.018.
- nqdef: L30, L7, L37, L45, L32, L38, L42, L39, L6, L47.
- nq4: L29 (+0.012), L45, L30, L47, L42, L37, L38, L39, L32, L43.

Readings:
- L29-L32, L37-L39, L42 and L45/L47 appear in every candidate, so they are the consistently sensitive layers.
- At level 2, L6 and L7 also stand out. The L7 step sits right after the weak L3-L6 block.
- Router agreement falls steadily with depth (73% by L41-77 at level 2). Most of that fall is downstream drift, not router error at each layer: nq2_early, with FP8 experts from L7 on, still loses 13 points of routing by L41-77.

## Default 4-bit set used
- The manifests' default_allocation.level4_experts at run time: "top 26 experts per layer by token-weighted REAP", 26 per layer for L3-L77, 1950 in total.
- Snapshot sha256 051853ffac7b0dfd..., stored in results/defset.json with the per-layer manifest shas.
- Thread 25's v3 global rescore (16-128 per layer) had not been written to the manifests, so it was not evaluated. The harness re-snapshots with `./run_full.sh snap` and is ready for it.

## Method
- Reference:
  - FP8 weights dequantised to bf16, bf16 math, fp32 head.
  - Non-overlapping 2048-token windows over all 78 layers (MTP excluded). The DSA indexer is skipped, which is exact because window = index_topk.
  - Windows are split win[RANK::8] over 8 independent GPUs, with no NCCL.
- Only routed experts differ. Each candidate carries its own residual stream and routing, with several candidates per pass.
- The inline reference was bitwise equal to the cached pass-B reference on every rank in both A passes.
- Candidate weights:
  - thread-12 nq_decode (rotated_levels -> decode_matrix -> apply_ocol, which includes the low-rank term), from /tmp/nestquant/nq-encode-v1/L{L}/experts/E{E}.pt;
  - predecoded to fp16 safetensors. fp16 storage adds about 2e-4 relative weight error, negligible next to level-2's 0.19 and level-4's 0.058.
- Held-out:
  - The calibration corpus (glm53_calib_glmfmt_v1) drops the nq-tail documents and is 13-gram decontaminated against all eval texts and GPQA.
  - The 4 corpora are disjoint and reported separately. Token counts are listed in the Results table; the corpus manifest shas are in /tmp/nestquant/18-e2e/corpora/manifest.json.
- MoE path (nq_e2e.moe_multi):
  - Routing and shared experts run in 16k-token slabs.
  - Each routed expert gathers its own tokens across the whole window set, and each expert's weights are read once per layer.
- Resources: peak VRAM 8.1-9.0 GB and host RSS about 2.7 GB per process. Each pass took about 25-40 min, bound by disk. Predecode ran at about 50 s per (layer, rank) file of 32 experts and is CPU-bound in nq_decode.

## Files
- Code: nq_e2e.py (prep / run / merge / predecode-nq), quantisers.py (plug-ins incl. `mix:`), nq_defset.py (default-set snapshot), run_full.sh (staged full run), nq_io.py, run.sh.
- results/: passB.json (nq4), passA1.json (nq2, nqdef), passA2.json (nq2_early, nqdef_e4), defset.json. Each pass file holds the per-corpus tables and the per-layer rel_div, route agreement, local error and band stats.
- Scratch: /tmp/nestquant/18-e2e.

## Earlier smoke results (kept for reference)
Identity KLD was 0 and top-1 100% on L0-5, and the cached-ref rerun was bitwise equal to the inline one. With only 3 experts replaced, every quantiser sat on a ~0.04 KLD floor, so few-expert end-to-end runs can't rank quantisers.

## Follow-ups (2026-09-29): fast predecoder, 4-bit count sweep, nqadapt, T27 dump

**Batched GPU decoder** (`nq_fastdec.py`, `predecode-nq --fast`). It reads the TP8 shard containers and decodes many
experts per GPU call.
- Gate (`nq_fastdec_gate.py`, results/fastdec/): fp32 `torch.equal` against `nq_decode.decode_expert` on every expert
  of L3, L30 and L77 at both levels, including the lr term and a synthetic inter_perm. 0 mismatches.
- Full arm (nq2 all + nq4 default set + L3-6): byte-identical to the old predecoded_A (parsed header + data bytes) on
  all 1183 files.
- Timing: ~7-9 s decode + ~1.5-3 s write per (layer, rank), so ~10 min for the full model including verify (old path
  ~45 min). A 10.8k-expert level-4 subset takes ~4 min.

**4-bit count sweep** (full model, KLD / top1%). Level-4 share is the share of routed slots on eval tokens, measured on
each arm's own routing.

| arm | L4/layer | L4 share | nq-tail | vllm-docs | wikitext | github |
|---|---|---|---|---|---|---|
| nq2 | 0 | 0 | 0.1305 / 90.62 | 0.1392 / 89.08 | 0.3105 / 81.95 | 0.1016 / 90.93 |
| nqdef (v2) | 26 | 0.149 | 0.0965 / 92.21 | 0.1114 / 90.23 | 0.2874 / 82.80 | 0.0807 / 91.80 |
| nqdef64 (nested top-K) | 64 | - | 0.0742 / 93.35 | 0.0884 / 91.41 | 0.2491 / 84.17 | 0.0654 / 92.72 |
| nqfloat0 (26 U floating_default 51) | 77 | 0.411 | 0.0704 / 93.44 | 0.0861 / 91.40 | 0.2199 / 85.18 | 0.0662 / 92.56 |
| nqdef128 (nested top-K) | 128 | 0.571 | 0.0480 / 94.79 | 0.0595 / 92.94 | 0.1682 / 87.17 | 0.0470 / 93.78 |
| **nqadapt** (reset per window) | 26+51 | 0.570 | 0.0536 / 94.19 | 0.0592 / 92.98 | 0.0992 / 90.41 | 0.0502 / 93.65 |
| **nqadapt_chain** | 26+51 | 0.579 | 0.0526 / 94.27 | 0.0547 / 93.30 | 0.0900 / 90.87 | 0.0467 / 93.78 |
| nq4 | 256 | 1 | 0.0227 / 96.53 | 0.0242 / 95.43 | 0.0334 / 94.35 | 0.0202 / 95.83 |

**nqadapt** (`quantisers.Adapt`, spec `adapt:lo=,hi=,hi2=,manifest=[,chain=1]`) is a causal replay of
streaming/scheduler.py with its defaults.
- Per layer: fixed = default_allocation (26), always level 4. The floating set is 51 experts, starting from
  floating_default.
- Score: the stream's own top-8 routing counts, EMA with a half-life of 512 tokens. Refresh every 64 tokens; ties go
  to the lower id.
- Lag: one refresh. Chunk k is served by the set chosen at refresh k-1; chunks 0 and 1 use floating_default.
- Validated against the real `Scheduler` class, stepped token by token on real routing ids, for both reset and chain
  modes: 0 mismatches.
- **Ignored vs scheduler.py:**
  - SSD byte-budget deferral: every upgrade lands exactly one refresh later.
  - The big_frac guard: it never fires at 8 experts/token.
  - Downgrades: they get the same one-refresh lag. The real scheduler drops them at once, so its transition chunk
    serves old∩new.
- reset: state is reset for every 2048-token window, so the scheduler is only warm for about the second half of each
  sequence.
- chain: runs with NQ_SHARD=contig. Each rank takes a contiguous block of every corpus in document order, and the
  EMA and floating set carry across that block. Chains are 3-16 windows (mean 9.3). They are not whole-corpus chains.
- Churn (experts entering the floating set per refresh):
  - first refresh away from floating_default: ~34
  - later refreshes: 4.7 (reset) / 3.4 (chain)
- The static nqfloat0 set covers 0.411 of slots; the fixed 26 alone cover 0.149.
- KLD by position (all corpora, token-weighted), positions 0-255 / 256-1023 / 1024-2046:

  | arm | 0-255 | 256-1023 | 1024-2046 |
  |---|---|---|---|
  | nqadapt | 0.110 | 0.065 | 0.054 |
  | nqadapt_chain | 0.081 | 0.064 | 0.054 |
  | nqfloat0 | 0.131 | 0.113 | 0.097 |
  | nqdef128 | 0.096 | 0.081 | 0.071 |

  Per-corpus buckets are in results/adapt_report.json.
- Shard layout changes GEMM batch shapes, so the refs differ in the 4th decimal: ppl_ref 2.3003 (S3), 2.3001 (S4) vs
  2.3009.

**T27 hidden-state dump.** `run --dump-layers/--dump-what`, output in /tmp/nestquant/18-e2e/hdump.
- Contents: L29-32, fp8 + nqdef streams, x / ids / p / shared / moe_out / h_in / h_mid / h_out / d_ref.
- Data: calib-fit, 128 windows from the fit split.
- `--expert-override L=DIR` decodes tuned E{E}.pt files through nq_fastdec at the mix's level. Overriding with the
  original encode gives bitwise-identical dumps.

Files: results/passS1-S4.json, results/adapt_report.json, results/defset_{top64,top128,float0}.json.
Tags: passS1 (nqdef64 + nqfloat0), passS2 (nqdef128), passS3 (nqadapt reset/chain), passS4 (eval-token shares:
nqdef/nqfloat0/nqdef128).
