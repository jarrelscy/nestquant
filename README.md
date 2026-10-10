# NestQuant

Dynamic 2–4 bit quantisation for MoE models, in one artifact. A 2-bit trellis base stays resident in
VRAM; the 3- and 4-bit refinement planes stream from SSD on demand, so only the experts that matter this
step are held at 4-bit. Memory stays near the 2-bit size. Running on GLM-5.3 (vision), 4× RTX
PRO 6000 (SM120, 96 GB each), TP4 + DCP4, MTP ns=3.

## How it works

- **One nested artifact.** Each routed expert is a 2-bit trellis base plus stacked 3/4-bit refinement
  bytes. The base is always loaded; refinement planes promote an expert to 3 or 4 bit.
- **Only the hot experts float.** A few experts per layer carry most of each token. A predictor picks that
  hot set and the streamer promotes/demotes planes to match. The rest sit at the 2-bit base.
- **Predictor.** Production uses GPU jF (joint), **26 fixed + 51 floating** experts per layer,
  with 56 floating/staging slots. TAP (horizon 64) schedules reads asynchronously.
  For the historical zero-fixed layout, explicitly set `NQ_JOINT_FIXED=0 NQ_SLOTS_PER_LAYER=80`.
- **Prefill.** During prefill the pool borrows free KV pages to hold 155 slots per layer and returns them
  for decode. When the KV pool is nearly full, used KV of some layers is parked in host RAM for the borrow.
- **Layer-major prefill (default on, `NQ_LMPF=1`).** Prompts with at least 32K new tokens run one layer at a
  time over 64K-token windows while every expert of the next layer streams in at 4 bit, so the whole
  prompt is prefilled at 4 bit (share 1.0). Prompts with 1K-32K new tokens get a 2 s read budget
  (`NQ_LMPF_BUDGET_S`) spent on the experts the router uses most. Shorter prompts use the borrow path
  above. `NQ_LMPF=0` turns it off. The read ring and the window state live in idle decode-expert slots for the
  length of the prefill and are refilled afterwards (`NQ_LMPF_BORROW=all`; `ring` = ring only, `0` = own memory).
- **Hot experts trade against KV.** The floating pool and the KV cache share one VRAM budget, so you pick
  more hot experts (quality) or more context. See below.

## Measured

2026-10-03, commit 8c8f9d5, historical zero-fixed config (not the current default): 1M context (KV pool 1,083,392 tokens),
`NQ_UTIL=0.925`, 80 slots/layer, jF predictor, dual-NVMe reads, concurrency 1, MTP ns=3.

| | |
|---|---|
| Decode, empty context | 83.0 tok/s (32.6 steps/s, 2.60 accepted/step; median of 3) |
| Decode, 16K context | 78.8 tok/s (31.6 steps/s) |
| Prefill | ~1900–2000 tok/s (layer-major off; see below) |
| KLD vs BF16 teacher | 0.0256 mean over 4 windows (0.0147 / 0.0612 / 0.0136 / 0.0130) |
| Hot share in decode | 0.627 of activated experts served at 4 bit (held-out generations) |
| Needles | retrieved at 43K and 947K |

Layer-major prefill, 2026-10-04 (commit e7cbde2, clean boot, same boot on/off, 2 reps). Prefill KLD is
measured on the four 2K windows above (prompt tokens, BF16 teacher); TTFT on the shown prompt lengths:

| Prompt | Mode | Prefill KLD | 4-bit share | TTFT off | TTFT on |
|---|---|---|---|---|---|
| 2K | budget 2 s | 0.0570 → 0.0224 | .37 → .74 | | +2.0 s |
| 4K | budget 2 s | | | 1.93 s | 4.0 s |
| 16K | budget 2 s | | | 7.97 s | 9.83 s |
| 64K | full | | 1.00 | 33.0 s | 35.1 s |
| 128K | full | | 1.00 | 67.5 s | 71.7 s |

All-4-bit prefill on the same 2K windows gives 0.0138. Needles found at 86K, 172K and ~947K with it on.

Decode runs range 77–89 tok/s with MTP acceptance; steps/s stays at ~31–33. KLD is full-vocab,
teacher-forced on the live server (`NQ_KLD_HOOK=1`) over the same four windows as
`threads/34-tr3/REPRODUCE.md`. Tensor-parallel all-reduce uses the b12x PCIe kernel with fused
add+RMSNorm (NCCL alone was 2.5% slower).

## Run it

The one-command default is the latest full-model TB4 allocation: **26 fixed + 51 floating**, TP4 + DCP4,
MTP3, FP8 KV, 1,048,576 context and concurrency one. TAP, compiled layer-major prefill, slot borrowing,
prefill KV offload, asynchronous decode reads, coalesced follower reads and CPU LMCache are enabled.
Explicit environment overrides remain supported. This targets four **96 GB Blackwell** cards, not RTX 6000 Ada.

Prerequisites: Docker with NVIDIA Container Toolkit, Python 3 with venv/pip, approximately 251 GB host RAM
as on the reference machine, and at least 437 GB model/record disk space plus caches and results. The optional
second-drive copy needs additional space for the rank record files. The launcher installs an isolated HF CLI
if needed. LMCache allows 72 GB host RAM across four ranks; monitor available RAM and leave more than 60 GiB
available on the reference-size host during long runs.

```bash
# Optional but recommended: choose large NVMe locations, then reuse these exports on restart.
export NQ_STATE=/mnt/nvme0/nestquant
export NQ_REPACK_ALT_DIR=/mnt/nvme1/nestquant-records
./start.sh config     # resolved settings only; does not print the API key or start GPU work
./start.sh            # downloads, prepares second copy, builds and waits for readiness
```

```bash
./start.sh            # fetch what is missing, build kernels, start, wait for /v1/models (up is default)
./start.sh smoke      # "The capital of France is ..." coherence check
./start.sh logs
./start.sh down
```

OpenAI-compatible on `:8001`, served as `glm-5.3-nq` (alias `local`). Weights:
[huggingface.co/jarrelscy/GLM-5.3-Vision-NestQuant-2-4bit](https://huggingface.co/jarrelscy/GLM-5.3-Vision-NestQuant-2-4bit).

`start.sh` fetches or builds, once:
- the serving image `NQ_IMAGE` (public on Docker Hub): SM120 vLLM fork with the GLM-5.3 kernels and
  NestQuant hooks;
- the base checkpoint `NQ_MODEL_DIR` (~44 GB from HF `base/`): the GLM-5.3 NVFP4/ARVQ hybrid without its
  routed experts, i.e. attention, shared experts, dense layers 0-2, the MTP layer, embeddings/lm_head, the
  vision tower, tokenizer and config. Downloaded to `$NQ_MODELS_ROOT/jarrelscy/GLM-5.3-NQ-base` by default;
  `NQ_BASE_REPO` selects the HF repo it comes from (default this one);
- the NestQuant records `NQ_REPACK_DIR` (~393 GB from the HF repo root: `rankN.json`, `rankN.bin`, `res/`);
- the jF predictor `NQ_PREDICTOR_DIR` (HF `serving/predictor/`);
  records and predictor come from `NQ_REPACK_REPO` (default this one);
- liburing 2.5 and lightgbm, built/installed inside the image;
- the NestQuant kernels. The first boot also builds the torch.compile cache and takes longer.

You supply:
- 4× RTX PRO 6000 Blackwell (SM120, 96 GB each), records on fast NVMe (~11 GB/s plane-read ceiling);
- optionally a directory on a second NVMe at `NQ_REPACK_ALT_DIR`; the launcher copies the records
  there automatically before starting workers. Without it, reads use one drive.

Host paths default under `$HOME/.local/share/nestquant`. Set `NQ_STATE` to a large writable NVMe filesystem. All individual path overrides still work:

```bash
NQ_MODELS_ROOT=/mnt/models NQ_MODEL_DIR=/data/models/glm-5.3-nq-base \
NQ_REPACK_DIR=/mnt/nq-repack NQ_REPACK_ALT_DIR=/mnt2/nq-repack \
NQ_PREDICTOR_DIR=/mnt/nq-predictor ./start.sh
```

All serving defaults live in `sm120/serve/docker-compose.standalone.yaml`; each can be overridden from
the environment. API key via `VLLM_API_KEY` or a gitignored `.env` at the repo root (`NQ_ENV_FILE`);
no key = no auth.

### Trading KV cache for hot experts

`NQ_SLOTS_PER_LAYER` (hot experts) and `NQ_MAXLEN` (max context) draw from the same VRAM budget, so set
them together. Per GPU, one more slot on every layer ≈ **195 MiB**; KV ≈ 13.4 KiB/token (MLA, DCP4), so
**one slot per layer ≈ 15k tokens of context**. At `NQ_UTIL=0.925`, 80 slots (77 floating + 3 for swaps
in flight) leaves a 1.08M-token KV pool.

```bash
# ~120 hot experts per layer, context down to 400k
NQ_JOINT_FIXED=0 NQ_SLOTS_PER_LAYER=124 NQ_MAXLEN=400000 ./start.sh
```

+44 slots ≈ +8.4 GiB/GPU ≈ 640k tokens, leaving ~440k. Approximate; check the KV pool size in the boot
log and keep the prefill-peak line a few GiB under 97.9 GB/GPU. If not, trim `NQ_SLOTS_PER_LAYER` or
`NQ_MAXLEN`. This trade was not measured for speed or KLD.

### Reasoning / agentic use: 26 fixed + 51 floating

This is now the default. The equivalent explicit settings keep 26 boundary-token experts per layer pinned at 4 bit and let 51 float:

```bash
NQ_JOINT_FIXED=1 NQ_SLOTS_PER_LAYER=56 ./start.sh
```

The fixed set is the top 26 experts per layer by REAP salience weighted toward the 32 tokens before
`</think>` and end-of-turn (`threads/22-boundary-experts/fixed_set.json`). The jF predictor and tap
scheduler fill the 51 floating slots around it. Memory is about the same as the default (26 + 56 = 82
records vs 80, ~30k tokens less KV).

| layout | decode KLD vs BF16, 4 windows | tb4 layout-config-recreation2 |
|---|---|---|
| default (0 fixed, 77 floating) | .0254 | fail (empty reasoning past ~100k) |
| `NQ_JOINT_FIXED=1` (26 fixed, 51 floating) | .0271 | pass, 98 turns / 189k |

Per-token KLD is slightly worse (mostly the legal window); reasoning stays intact deep into long agent
runs. One task, so treat the tb4 column as a single data point.

Decode KLD per window (4 confirmation windows of brandonmusic/GLM-5.3-BF16-full-logits):

| window | 0 fixed / 77 floating | 26 fixed / 51 floating |
|---|---|---|
| w0 | .0154 (repeat .0149) | .0155 (repeat .0162) |
| w1 (legal) | .0579 | .0650 |
| w2 | .0128 | .0136 |
| w3 | .0155 | .0142 |
| mean | .0254 | .0271 |

`threads/22-boundary-experts/fixed_set_ranked.json` ranks all 256 experts per layer by the same score, best
first; the first 26 are the fixed set. Score mass of the top K (mean over layers): 8 → .185, 16 → .269,
26 → .345, 51 → .484, 77 → .594. Fixed sizes other than 26 have not been measured.

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

### Terminal-Bench 4.0

Use the [published Harbor config and instructions](https://huggingface.co/jarrelscy/GLM-5.3-Vision-NestQuant-2-4bit/tree/main/benchmarks/tb4.0).
The latest attempts used stock Terminus-2, one task at a time, temperature 1.0, top_p 0.95,
65,536 output tokens per call, and an 8-hour agent timeout. `clear_thinking=true` excludes earlier reasoning
from later requests; it does not disable reasoning generation. Keep KLD hooks off for normal benchmarks.
Check the launch summary, startup logs and smoke output before starting Harbor. Monitor host RAM,
streaming backlog/read errors, hot-expert coverage, MTP acceptance and decode throughput.
