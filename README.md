# NestQuant

Dynamic 2-4 bit MoE expert quantisation in one artifact: a 2-bit trellis base, with extra refinement bytes streamed on top for 3 and 4 bit. Targets GLM-5.3 and MiMo-V2.6, compared against EXL3 and NVFP4 using the FP8 reference. Work in progress.

- `BRIEF.md`: goals and constraints.
- `DESIGN.md`: the current format spec and adopted decisions.
- `LITERATURE.md`: literature notes.
- `threads/NN-*/`: one directory per research thread, with code, results JSON and `REPORT.md`.

## Where the code is
Fitting (encoder side):
- `threads/12-reference-encoder/`: reference encoder/decoder (`nq_encode.py`, `nq_decode.py`), in progress.
- `threads/02-feedback-conflict/`: nested trellis fitting with blended dual feedback (`fbt.py`, `gam2.py`, `eval2.py`).
- `threads/03-nested-trellis-code/`: the nested 2+2 mul1 trellis code.
- `threads/06-expert-objective/`: two-sided rounding for gate/up.
- `threads/08-ood-robustness/`: mixed routed/uniform Hessian calibration.
- `threads/05-exl3-harness/`: EXL3/NVFP4 baselines and the expert-output eval harness (`harness.py`, `run.sh`).

Inference:
- `threads/04-decode-kernel/`: A100 tensor-core decode GEMV kernels. `nqk2.cu`, `nq2.py` and `bench_chain2.py` hold the adopted additive A4/B2 decoders with fused-B chain; see `REPORT2.md`.
- `threads/13-moe-layer-kernel/`: grouped MoE layer kernel and vLLM integration, in progress.
- `threads/10-streaming-system/`: plane layout and streaming policy.

Model weights are not included. Scripts expect the GLM FP8 experts in `/tmp/nestquant/glm53-fp8-experts` (override with `NQ_GLM_SOURCE`). The CUDA 12 runtime used by the harness is expected in `threads/*/lib/` and is not committed.

## Serving (SM120, vLLM)

`sm120/serve/` runs the quantised GLM-5.3 as an OpenAI-compatible endpoint on `:8001`. Target box is 4x RTX PRO 6000 (SM120, 96 GB each), TP4 + DCP4, MTP `ns=3`. The routed experts are served from the NestQuant repack: the 2-bit trellis base stays resident, 3/4-bit refinement planes stream from SSD (P4) into a per-layer slot pool. Layers absent from the repack fall back to the production ARVQ experts.

One-command serve from the repo root:

```bash
./start.sh            # build kernels, start the container, wait for /v1/models (up is the default)
./start.sh smoke      # "The capital of France is ..." coherence check
./start.sh logs
./start.sh down
```

`start.sh` is self-contained — it uses only `sm120/serve/docker-compose.standalone.yaml` (the resolved,
single-file equivalent of the compose stack) and needs no files outside this repo. The served name is
`glm-5.3-nq`. GBDT predictor deps (lightgbm/narwhals/scipy) are pip-installed into a mount on first run;
nothing in the image is shadowed.

**Prerequisites (local artifacts, not shipped here):**

- An SM120 host — 4x RTX PRO 6000 Blackwell (96 GB each).
- The serving Docker image (`NQ_IMAGE`, default `glm53-arvq-sm120:fixes12-...`): the SM120 vLLM fork with
  the GLM-5.3 ARVQ/MLA kernels and the NestQuant hooks baked in. `pull_policy: never` — build/load it
  locally. Non-repack MoE layers run on this image's base (ARVQ) experts; that path lives inside the
  image and its single overlay `sm120/serve/overlay/nvfp4_arvq_hybrid.py`, not as a separate package.
- The GLM-5.3 base checkpoint at `NQ_MODEL_DIR` (under the host dir `NQ_MODELS_ROOT` → `/data/models`).
- The predictor dir at `NQ_PREDICTOR_DIR` (`joint/jF.pt`, `joint/v2_sal_tweedie1.5.txt`, `delta_table.json`)
  and a `liburing` install at `NQ_LIBURING_DIR` (SSD streaming).

The NestQuant repack is **not** a prerequisite you build — `start.sh` downloads the serve-ready
`nq-p4rec-v1` repack (per-rank `rankN.json` + `res/` resident planes + `rankN.bin` streamed planes) from
`jarrelscy/GLM-5.3-NestQuant-2-4bit` into `NQ_REPACK_DIR` on first run (~366 GB, once; needs the `hf` CLI,
`pip install -U 'huggingface_hub[hf_transfer]'`). No repacking step. Override the source with `NQ_REPACK_REPO`
or point `NQ_REPACK_DIR` at an existing copy to skip the download.

**Overriding paths for a different host.** Every host path is an env var with a default matching the
reference box; set what differs, e.g.:

```bash
NQ_IMAGE=my/glm53-sm120:tag \
NQ_MODELS_ROOT=/mnt/models NQ_MODEL_DIR=/data/models/glm-5.3-base \
NQ_REPACK_DIR=/mnt/nq-repack NQ_PREDICTOR_DIR=/mnt/nq-predictor \
NQ_LIBURING_DIR=/opt/liburing ./start.sh
```

Supply the API key via `VLLM_API_KEY` in the environment, or put a `VLLM_API_KEY=...` line in a
gitignored `.env` at the repo root (`NQ_ENV_FILE` to point elsewhere). With no key the server runs
without auth. `start.sh {up,down,logs,smoke}` and the compose file carry the full env-var surface.

### Expert predictors

Each decode step only a subset of experts per layer is held at 4-bit ("floating"); the rest sit at the 2-bit base. A predictor chooses the floating set and the streamer promotes/demotes planes to match. Select with `NQ_PREDICTOR`:

- `ema` — static usage-ranked set, no live adaptation (baseline).
- `gbdt` (default) — v2 salience GBDT (`v2_sal_tweedie1.5`), sync refresh, 51 floating + 26 fixed per layer.
- `joint` / `jf` — GPU joint predictor: a transformer on top of the v2-GBDT features, 77 floating per layer, refreshed every 16 decode tokens on a background stream thread that overlaps decode. rank0 steps the predictor; followers track its EMA.

Both learned predictors run one continuous EMA for the serve lifetime. Streaming budget is set by `NQ_SLOTS_PER_LAYER` (slot pool) and `NQ_CAP_GBPS` (upgrade budget across the 4 ranks, `0` = uncapped against the ~11 GB/s SSD ceiling). LMCache runs CPU-RAM-tier only (`NQ_LMCACHE_CPU_GB` per rank); no disk tier — it would contend with the expert stream.

### Measured (2026-09-30, single stream, 1024-tok decode)

| Predictor | tok/s | ms/step | iso-bpw KLD |
|---|---|---|---|
| GBDT (sync) | 94.0 | 28.82 | 0.0268 |
| jF (joint) | 94.2 | 28.61 | 0.0259 |

jF matches GBDT decode speed at slightly lower KLD, for ~8% more swap traffic. See `start.sh` and `sm120/serve/docker-compose.standalone.yaml` for the full env-var surface and defaults.

### Trading KV cache for hot experts

The floating-expert pool and the KV cache draw from the same VRAM budget (~93 GB/GPU at `NQ_UTIL=0.92`, alongside the fixed 2-bit base experts and the non-expert weights). Growing one shrinks the other, so `NQ_SLOTS_PER_LAYER` (how many experts are hot) and `MAXLEN` (max context) are set together.

Rough unit costs, per GPU:

- **Floating refinement plane** ≈ 2.5 MiB/expert (14.3 GiB for the 77×75 resident set). So adding one hot expert on *every* layer ≈ **+190 MiB**.
- **KV per token** ≈ 14 KiB (MLA-compressed, DCP4-sharded; from the ~55 KiB/token total over 4 ranks). So **100k tokens of context ≈ the VRAM of ~7 extra hot experts per layer** — about **14k tokens per hot expert per layer**. At the default ~1M context the KV pool is nearly full at 77 hot, so there's only room to add ~70 hot/layer (to ~145) and only as context falls toward zero. Useful operating points trade a few hundred k of context for a few dozen more hot experts.

**Example — 120 hot experts.** From the default (~77 hot, ~1M context), push more capability into the hot set:

```bash
NQ_SLOTS_PER_LAYER=124 NQ_MAXLEN=400000 NQ_UTIL=0.92 ./start.sh
```

120 floating needs ~124 slots (the pool keeps a few slots of headroom per layer for in-flight swaps). The +43 hot experts/layer ≈ **+8 GiB/GPU** ≈ **~570k tokens** of KV, so the ~1M default context drops to ~430k; cap at **400k** for headroom. This is approximate — confirm the fit at boot: the prefill-peak line in the logs should stay a few GiB under 97.9 GB/GPU; if it doesn't, trim `NQ_SLOTS_PER_LAYER` or `NQ_MAXLEN`, or nudge `NQ_UTIL`.

Which way to lean: more hot experts lowers per-layer KLD (better quality at short-to-mid context); more KV extends usable context. The predictors (`gbdt`/`jF`) choose *which* experts are hot each step; `NQ_SLOTS_PER_LAYER` sets *how many*.
