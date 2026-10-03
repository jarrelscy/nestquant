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

The serve path is a single configuration, **D** (below); the streaming budget is `NQ_SLOTS_PER_LAYER`. API key via `VLLM_API_KEY` in the environment or a gitignored `.env` at the repo root
(`NQ_ENV_FILE` to point elsewhere); no key = no auth. `start.sh` and
`sm120/serve/docker-compose.standalone.yaml` carry the full env-var surface.

### Serving (D)

The serve path has one configuration, D. Every flag below defaults to D in the code, and in both compose files.
`sm120/serve/tools/gate_d.sh` re-gates it against prod. The predictor is the jF joint predictor (on rank 0).
The scheduler is the tap scheduler, running in a Python host loop. Rank 0 leads; ranks 1-3 replay its ops from the oplog
using the coalescing follower. Expert planes stream from two queues per rank: `NQ_REPACK` at QD 8 and
`NQ_REPACK_ALT` at QD 4.

| flag | default | what |
|---|---|---|
| `NQ_SLOTS_PER_LAYER` | `80` | hot-expert slots per layer (77 floating + swap headroom; k0 layout, no fixed set) |
| `NQ_JOINT_NET` / `NQ_JOINT_V2` | `/nqpred/joint/jF.pt` / `…/v2_sal_tweedie1.5.txt` | jF net + LightGBM trees (`NQ_LGB_PATH=/nqlgb`) |
| `NQ_JOINT_HM` | `0.7` | jF resident hysteresis |
| `NQ_JOINT_GRAPH` | `0` | CUDA-graph the jF scoring core |
| `NQ_TAP_H` | `64` | tap horizon, tokens |
| `NQ_TAP_C` `NQ_TAP_HA` `NQ_TAP_MLA` `NQ_TAP_SVC_MS` `NQ_TAP_FAR` `NQ_TAP_TP` `NQ_TAP_RATE_GBPS` | `1.0` `0` `1.0` `2` `tail:0.1` `4` `6` | tap scheduler model (`streaming/scheduler_tap.py`) |
| `NQ_REPACK` / `NQ_REPACK_ALT` | `/nqrepack` / `/nqrepack1` | dual IO: both required, they must hold the same repack (checked at boot; mismatch = error). Standalone: `NQ_REPACK_ALT_DIR` (defaults to `NQ_REPACK_DIR`, one drive, two queues) |
| `NQ_IO_QD` / `NQ_IO_QD_ALT` | `8` / `4` | io_uring queue depth per drive |
| `NQ_PREFILL_BORROW` | `1` | phase 1: free KV blocks lent as extra expert slots during long prefills |
| `NQ_PREFILL_SLOTS` | `155` | floating experts per layer planned while borrowing |
| `NQ_PB_MIN_NEW` / `NQ_PB_MARGIN` | `8192` / `16` | borrow only for ≥ this many new tokens / KV blocks always left free |
| `NQ_PREFILL_KV_OFFLOAD` | `1` | phase 2: used KV of some MLA layers to pinned host, streamed back per layer (null block never carved; reclaim fenced on in-flight reads into lent memory) |
| `NQ_PREFILL_KV_BELOW` | `1500` | phase 2 only while phase 1 got fewer slots than this |
| `NQ_PB_KV_HOST_GB` / `NQ_PB_RAM_FLOOR_GB` | `8` / `38` | phase 2 pinned-host cap per rank / MemAvailable floor |
| `NQ_PREFILL_ADAPT` | `lookahead` | prefill router lookahead (`NQ_LA_D=1`, `NQ_LA_BUDGET=45`, `NQ_PF_RANK=gate`) |
| `NQ_SESSION_RESTORE` | `1` | per-session floating-set restore (`NQ_SR_*`) |
| `NQ_PF` / `NQ_PF_MIN` | `1` / `384` | prefill MoE path for chunks ≥ this many tokens |
| `NQ_LMCACHE` (→ `ENABLE_LMCACHE`) | `1` | LMCache KV offload/restore |
| `NUM_SPEC` | `3` | MTP speculative tokens |
| `NQ_IOSTATS` / `NQ_RANK_SHARE` | `0` / `0` | measurement only: io stats / served level-4 share |

Runtime switches (files, in-boot): `/dev/shm/nq_pb_off` (no borrow), `/dev/shm/nq_pb_kv_off` (no phase 2),
`/dev/shm/nq_pf_off`, `/dev/shm/nq_sr_ctl`, `/dev/shm/nq_la_ctl`. Debug (default off): `NQ_CHECK`, `NQ_DUMP`,
`NQ_FOLLOW_CHECK`, `NQ_SHADOW`, `NQ_FAULT_*`, `NQ_TFCAP`, `NQ_RAMTIER_GB`.
The GBDT/EMA predictors, the base `Scheduler`, and the tf predictor are offline tools (`streaming/scheduler.py`,
`threads/36-tfpred/`), and the serve never imports them.

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
