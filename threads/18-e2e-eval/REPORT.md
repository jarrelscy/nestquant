# Thread 18: end-to-end KLD / top-1 eval for GLM-5.3 (written by the lead from the agent's final message)

## Verdict
Harness built and smoke-tested on all 78 layers. NestQuant (thread 12 nq_decode), EXL3 and NVFP4 plug in through one interface; EXL3/NVFP4 decodes are bit-identical to thread 05's loaders. Blocked on data: full-model calibration capture and the 75 x 256 expert encodes (plus same-H EXL3/NVFP4 full fits).

## Files
nq_e2e.py (prep / run / merge / predecode), quantisers.py (plug-in interface), nq_io.py (low-memory safetensors + FP8 dequant), make_dir_cand.py, run.sh. Scratch /tmp/nestquant/18-e2e.

## Method
- Reference: FP8 dequantised to bf16, bf16 math, fp32 head. Earlier campaign's eval3_kld protocol: non-overlapping 2048-token windows, DSA indexer skipped (exact because window = index_topk), windows split over 8 GPUs, each GPU streams 78 layers (MTP excluded). Only routed experts differ.
- One pass carries the reference and several candidates, each with its own residual stream and routing; backbone and FP8 experts read once per layer.
- Metrics per candidate and corpus: KLD mean/stderr/p50/p90/p99, top-1 agreement, ppl ref/candidate, fallback count; per layer residual divergence, router top-8 agreement, optional local routed-expert error.
- Interface: expert(layer, e, ref) -> {gate_proj, up_proj, down_proj} [out, in] un-rotated, or None for reference. Specs: ref, rtn:, dir:, nestquant:root=,level=2|4, exl3:root=,bits=, nvfp4:root=, py:file:Class, optional layers=a-b.

## Corpora (report separately)
| name | tokens / windows | role |
|---|---|---|
| nq-tail | 262,144 / 128 | in-distribution, tail of the 15M calibration corpus |
| vllm-docs | 162,942 / 79 | code/docs held-out from earlier campaign |
| wikitext | 136,733 / 66 | OOD prose, calibration-bias canary |
| github | 52,528 / 25 | OOD code after model cutoff |
Fits must use only 512-token windows with index < 28784 of the 15M corpus (or declare another disjoint split).

## Memory/time (8x A100, 8 independent processes, no NCCL)
~1 GB backbone + 3.8 GB fp32 head + ~3 GB per stream; host RSS <1 GB per process. NestQuant decode is Python-bound (0.37 s/expert rotation + 0.02 s/level), so pre-decode to fp16 (1.45 TB per level, ~16 min per level on 8 GPUs). Estimates for 298 windows: ref + 1 candidate 10-15 min; ref + 5 candidates one pass 30-50 min; plus pre-decode ~30 min; total ~1-1.5 h.

## Smoke results
- Layers 0-5: identity KLD 0 / 100% top-1; cached-ref rerun bitwise equal to inline.
- All 78 layers, one window per corpus:

| cand | corpus | ppl ref | ppl cand | KLD | top-1 |
|---|---|---|---|---|---|
| rtn4 | nq-tail | 2.147 | 2.132 | 0.0110 | 96.6% |
| rtn4 | wikitext | 1.597 | 1.660 | 0.0567 | 94.6% |
| rtn4 | github | 2.529 | 2.519 | 0.0254 | 95.6% |
| rtn2 | nq-tail | 2.147 | 2.443 | 0.218 | 84.8% |
| rtn2 | wikitext | 1.597 | 4.553 | 1.111 | 68.3% |
| rtn2 | github | 2.529 | 3.131 | 0.304 | 84.6% |

- Pilot artifacts (3 experts) local routed-expert error: L16 nq2 0.354 / exl3_2 0.399 / nq4 0.096 / exl3_4 0.104 / nvfp4 0.123; L49 0.343 / 0.406 / 0.093 / 0.106 / 0.117. (Lead's note: the EXL3 pilot files may use the original calibration, so this is not the fair same-H comparison.) With only 3 experts replaced every candidate sits on a ~0.04 KLD floor, so few-expert e2e can't rank quantisers.

## Needs from thread 12 / the full run
1. All 75 x 256 artifacts at ROOT/layer_{L:03d}/expert_{E:03d}.pt (nq_run.nq_variant(..., keep_artifact=True)), ~19 MB each, ~366 GB.
2. Full-model calibration capture (Hessians exist only for L16/32/49/66) and same-H full EXL3-2/4 and NVFP4 fits.
3. Held-out discipline as above. 4. Zero fallbacks in merge. 5. Optional batched rotated_levels to skip pre-decode.

## How to run
See the agent's command block: predecode nq2/nq4 on 8 ranks, then one `run.sh run` per GPU with --corpora nq-tail,vllm-docs,wikitext,github --local-err --save-ref --tag full and candidates nq2, nq4, exl3_2, exl3_4, nvfp4, then `run.sh merge --tag full`. Go signal: NestQuant wins in-distribution without a worse wikitext/in-distribution ratio than EXL3.
