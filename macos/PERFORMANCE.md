# Mac performance measurements

The first tested configuration is too slow for interactive use. The complete model runs, but refinement loading and the reference execution path remain expensive. Resident hot-set comparisons are in progress.

Measured on 2026-11-04: Apple M5 Max, 40 GPU cores, 128 GiB memory, macOS 27.2. The model is GLM-5.3-Flash NestQuant 1.5/4 at revision `4023870a6f2c6097e37fa5d2e54040db64b011e9`.

| Configuration | Decode tokens/second | Time to first token | Process peak |
|---|---|---|---|
| Canonical 2,630 hot slots, resident base, 4 GiB on-demand refinement cache, MLX wiring | 0.227 median, 0.199-0.237 range | 131.03 seconds median | 81.67 GiB |

The published JSON is a composite of the separate summary, measured requests, and warm-up files. Source and checkpoint revisions were added when this historical result was published. Later driver versions capture the source revision and dirty state directly.

[The recorded results](results/2026-11-04-canonical-cache4.json) include the source revision, package versions, prompt, token IDs, phase timings, disk reads, and memory measurements.

The driver discarded one warm-up and measured three requests in one process. Each request used the same 29-token coding prompt and generated 16 tokens with greedy sampling. Decode throughput includes 15 decode steps per request. It excludes prefill and the first generated token. All measured token sequences matched. The outputs are truncated coding responses, so this benchmark does not grade code correctness.

The 16-token measurement window includes initial refinement-cache misses. It does not establish sustained throughput or performance with long context. Process peak memory includes model loading and earlier requests. Background model servers and model downloads were paused during measurement. Other desktop applications remained open.

Confidence is high for these recorded measurements. Their applicability to longer requests is unmeasured. No claim of speed improvement from memory wiring follows from this result: the earlier unwired smoke tests used different prompts and output lengths.
