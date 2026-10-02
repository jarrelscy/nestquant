# NestQuant

Dynamic 2–4 bit quantisation for MoE models, in one artifact. A 2-bit trellis base stays resident in
VRAM; the 3- and 4-bit refinement planes stream from SSD on demand, so only the experts that matter this
step are held at 4-bit. 4-bit quality where it counts, ~2-bit memory. Running on GLM-5.3 (vision), 4× RTX
PRO 6000 (SM120, 96 GB each), TP4 + DCP4, MTP ns=3.

## How it works

- **One nested artifact.** Each routed expert is a 2-bit trellis base plus stacked 3/4-bit refinement
  bytes. The base is always loaded; refinement planes promote an expert to 3 or 4 bit.
- **Only the hot experts float.** A few experts per layer carry most of each token. A predictor picks that
  hot set every decode step and the streamer promotes/demotes planes to match — the rest sit at the 2-bit
  base.
- **Two predictors.** `gbdt` (default) — a GBDT on v2 salience features, 51 floating + 26 fixed per layer.
  `jF` (joint) — a small GPU transformer on top of those features, 77 floating per layer, refreshed every
  16 decode tokens on a background thread that overlaps decode.
- **Hot experts trade against KV.** The floating pool and the KV cache share one VRAM budget, so you pick
  more hot experts (quality) or more context. ~77 hot at ~1M context; push to ~120 hot and context falls
  to ~400k.

## Quality

KLD against the FP8 reference. NestQuant beats AQLM by **~1.5× / 5× / 1.6×** (id / wiki / code) and beats
our own ARVQ hybrid. On 64K prefill, router lookahead lifts the served share of correctly-promoted 4-bit
experts from 0.24→0.60 (wiki) and 0.29→0.52 (code), within 0.02 of oracle.

Measured 2026-09-30, single stream, 1024-tok decode:

| Predictor | tok/s | ms/step | iso-bpw KLD |
|---|---|---|---|
| GBDT (default) | 94.0 | 28.82 | 0.0268 |
| jF (joint)     | 94.2 | 28.61 | 0.0259 |

jF matches GBDT decode speed at slightly lower KLD, for ~8% more swap traffic. Prefill ~1850 tok/s at 4K,
~2100 at 64K; 1M-token cold prefill ramps 408→1488 tok/s (needle intact). Caveats: concurrency 1,
SSD-bandwidth bound (~11 GB/s plane ceiling), terminal-bench 4.0 pending.

## Run it

One command from the repo root. No repacking — `start.sh` pulls the serving image from Docker Hub and the
repack from Hugging Face on first run.

```bash
./start.sh            # pull image + repack, build kernels, start, wait for /v1/models (up is default)
./start.sh smoke      # "The capital of France is ..." coherence check
./start.sh logs
./start.sh down
```

OpenAI-compatible on `:8001`, served as `glm-5.3-nq`. Weights live at
[huggingface.co/jarrelscy/GLM-5.3-NestQuant-2-4bit](https://huggingface.co/jarrelscy/GLM-5.3-NestQuant-2-4bit).

**You supply (local artifacts, not shipped):** an SM120 host (4× RTX PRO 6000, 96 GB each); the GLM-5.3
base checkpoint at `NQ_MODEL_DIR`; the predictor dir at `NQ_PREDICTOR_DIR` (`joint/jF.pt`,
`joint/v2_sal_tweedie1.5.txt`, `delta_table.json`); a `liburing` install at `NQ_LIBURING_DIR`.

**`start.sh` fetches (once):** the serving image (`NQ_IMAGE`, ~20 GB) — the SM120 vLLM fork with the
GLM-5.3 ARVQ/MLA kernels and NestQuant hooks baked in; the repack (`NQ_REPACK_DIR`, ~366 GB) —
per-rank `rankN.json` + `res/` resident planes + `rankN.bin` streamed planes. Non-repack MoE layers fall
back to the image's ARVQ experts.

Every host path is an env var with a default matching the reference box; set what differs:

```bash
NQ_IMAGE=my/glm53-sm120:tag \
NQ_MODELS_ROOT=/mnt/models NQ_MODEL_DIR=/data/models/glm-5.3-base \
NQ_REPACK_DIR=/mnt/nq-repack NQ_PREDICTOR_DIR=/mnt/nq-predictor \
NQ_LIBURING_DIR=/opt/liburing ./start.sh
```

Select the predictor with `NQ_PREDICTOR` (`ema` | `gbdt` | `joint`/`jf`). Streaming budget is
`NQ_SLOTS_PER_LAYER` (slot pool) and `NQ_CAP_GBPS` (upgrade budget, `0` = uncapped against the ~11 GB/s
SSD ceiling). API key via `VLLM_API_KEY` in the environment or a gitignored `.env` at the repo root
(`NQ_ENV_FILE` to point elsewhere); no key = no auth. `start.sh` and
`sm120/serve/docker-compose.standalone.yaml` carry the full env-var surface.

### Trading KV cache for hot experts

`NQ_SLOTS_PER_LAYER` (how many experts are hot) and `NQ_MAXLEN` (max context) draw from the same VRAM
budget, so set them together. Per GPU: one floating plane ≈ 2.5 MiB/expert, so one more hot expert on
*every* layer ≈ **+190 MiB**; KV ≈ 14 KiB/token (MLA, DCP4), so **~14k tokens of context per hot
expert per layer**. At the default ~1M context the KV pool is nearly full at 77 hot.

```bash
# push capability into the hot set: ~120 hot, context down to 400k
NQ_SLOTS_PER_LAYER=124 NQ_MAXLEN=400000 NQ_UTIL=0.92 ./start.sh
```

120 floating needs ~124 slots (headroom for in-flight swaps). The +43 hot experts/layer ≈ +8 GiB/GPU ≈
~570k tokens of KV. Approximate — confirm at boot: the prefill-peak log line should stay a few GiB under
97.9 GB/GPU; if not, trim `NQ_SLOTS_PER_LAYER`/`NQ_MAXLEN` or nudge `NQ_UTIL`. More hot experts lowers
per-layer KLD; more KV extends usable context.

## Repo layout

- `BRIEF.md` goals/constraints · `DESIGN.md` format spec and adopted decisions · `LITERATURE.md` notes.
- `threads/NN-*/`: one directory per research thread, each with code, results JSON and `REPORT.md`.

Fitting: `12-reference-encoder` (reference encode/decode), `02-feedback-conflict` (nested trellis, dual
feedback), `03-nested-trellis-code` (2+2 mul1 code), `06-expert-objective` (two-sided gate/up rounding),
`08-ood-robustness` (mixed Hessian calibration), `05-exl3-harness` (EXL3/NVFP4 baselines + eval harness).
Inference: `04-decode-kernel` (SM120 tensor-core decode GEMV, additive A4/B2 + fused-B chain),
`13-moe-layer-kernel` (grouped MoE layer kernel + vLLM integration), `10-streaming-system` (plane layout
and streaming policy).

Model weights are not in this repo; fitting scripts expect the GLM FP8 experts at
`/tmp/nestquant/glm53-fp8-experts` (override with `NQ_GLM_SOURCE`).
