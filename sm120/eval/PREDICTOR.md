# Expert predictor used by serving and by the C2 `dyn` stream

Each MoE layer has 256 routed experts. Every expert is resident at level 2 (2-bit base). A subset is also held at
level 4 (2-bit base + 2-bit residual streamed from NVMe). The predictor decides which experts are at level 4.

## Level-4 set per layer

- **Fixed set, 26 experts (10%)**: loaded at startup, never evicted. Chosen offline by token-weighted REAP salience
  (mean gate weight x output norm, with extra weight on boundary tokens such as `</think>`), not by routing
  frequency (`streaming/fixed_set.py`, `threads/22-boundary-experts/fixed_set.json`). On C2 it covers 12% of routed
  slots, close to the 10.2% a random set would get, so it protects high-salience experts rather than busy ones.
- **Floating set, 51 experts (20%)**: chosen online from recent routing (`streaming/scheduler.py`). Production
  default since 2026-09-29: the GBDT predictor (section below); the EMA rule stays selectable (`NQ_PREDICTOR=ema`).
- Total 77 of 256 (30%) at level 4 at any time.

## Floating set, EMA rule (`scheduler.Scheduler`, `predictor='ema'`)

1. **Start**: the 51 most-routed non-fixed experts from the calibration capture (`floating_default`) are loaded at
   startup, so token 0 already has 30% of experts at level 4.
2. **Score**: per expert, `score = score * 0.5**(ntok/512) + count`, where `count` is how many of the step's routed
   slots (8 per token) went to that expert. This is an exponential moving average of routing counts with a half-life
   of 512 tokens. It is never reset, so it carries across requests.
3. **Refresh every 64 tokens**: the target set becomes the top 51 non-fixed experts by score (stable sort, ties to the
   lower index). Layers with no counts yet keep `floating_default`.
4. **Downgrades** (experts that left the target) are immediate: the kernel switches back to the resident level-2 rows,
   with no I/O.
5. **Upgrades** (experts that joined the target) are issued highest score first under a byte budget of
   `NQ_CAP_GBPS` GB/s (12 since 2026-09-29, 6 before; see the GBDT section) at an assumed 111 tok/s, accrued per token and capped at a 64-token burst. The rest wait at
   level 2 and are retried on the next step. Each upgrade reads one 2.56 MB record on every rank and is charged
   rec_bytes x tp = 10.24 MB, so the cap is aggregate over the 4 ranks (6 GB/s = 1.5 GB/s per rank). (This file said
   "6 GB/s per rank" before 2026-09-29; the code has always charged the aggregate.)
6. **Guard**: no upgrades in a step that touches more than half of a layer's experts (prefill-sized steps), so a long
   prompt updates the scores without flooding the SSD.
7. **Slots**: at most 56 streamed experts per layer in flight, landed or draining.

In serving, rank 0 runs the policy and the other ranks replay its op log from /dev/shm (the followers never run the
predictor).

## How C2 emulates it (`eval_fp8.py: sim_dyn`)

- The `dyn` stream has its own residual stream and its own routing, so the predictor sees the routing that the
  quantised model produces rather than the FP8 reference's routing.
- One `Scheduler` per layer, with 1/75 of the byte budget, 51 floating experts, 56 slots and `floating_default`
  preloaded at level 4.
- Tokens are fed in steps of 3 (`--step-tok 3`, about the ns3 tokens per step in serving). A step's levels are fixed
  before its counts are seen (causal), and upgrades land one step later.
- The 64 windows are chained in corpus order, like one long session, so the score carries from window to window.
- Hindsight oracle, reported per layer: the best 51 non-fixed experts for each 64-token block, chosen with the block's
  own counts. This upper bound can't be reached causally.

## c2-full-v1 (2026-09-29)

- Level-4 share of routed slots: dyn 0.61 (layer 40: 0.65, oracle 0.84), fixed 0.12, ARVQ hot set 0.35.
- Bits per expert: dyn 2.82 resident / 3.49 served; ARVQ 2.79 / 2.91.
- KLD vs FP8 (id / wikitext / code): dyn 0.220 / 0.074 / 0.043, ARVQ 0.436 / 0.785 / 0.106.
  Full tables: `sm120/results/c2/c2-full-v1.md`.

## Alternatives tested offline (routing-log simulator, held-out tasks)

- EMA 256, refresh 16, hysteresis 0.1: +0.8 pt share over the default on val, better in all 21 out-of-domain probe
  domains, ~81 us per refresh on the host.
- Separate answer-state memory (after `</think>`): a further +0.2 pt.
- Linear, low-rank NN and REAP-weighted scores: +0.2-0.8 pt or worse. History-based ranking tops out near 64% share
  against ~80% for a perfect-foresight oracle; the gap is routing that can't be predicted from history.

## GBDT predictor (production default since 2026-09-29, `predictor='gbdt'`)

From the expert-predict study (`/data/Jarrel/expert-predict/PROGRESS.md`). Only the floating-set target and the
upgrade order change; the budget, big-step guard, slots, fixed set and kv_pressure are the same as above.

- **Model**: `streaming/gbdt_p64_s5.txt`, LightGBM 4.7.0 Poisson regression on routed hits in the next 64 decode
  tokens, 60 trees x 15 leaves. Trained on ARVQ-served agent decode routing (fin-saccr-rwa, pretrain-shard-corruption),
  early-stopped on formal-crypto and embedding-drift-monitor. Loaded by `streaming/gbdt_predictor.py`.
- **Features** per (layer, expert), from the routing counts only: ema32, ema128 (decayed hit rates), mem_cur_state
  (own-clock memory of the current think/answer segment, from `</think>` in the token ids), tok_since_hit, hits16.
  The layer id is not a feature, so one model serves all layers.
- **Candidates**: the non-fixed experts ranked 20..120 by EMA256 are scored (ranks < 20 always kept, > 120 get 0);
  the top 51 by score become the target. Hysteresis: a resident expert's score x 1.5. Upgrades are issued in
  descending hysteresis-adjusted score.
- **Cadence**: features per 16-token decode block, a refresh at every block boundary. Mode `next_refresh`: the
  prediction runs on a worker thread (`NQ_GBDT_THREADS`, 4) and is applied at the next boundary, 16 tokens late.
  Steps with more than 16 tokens (prefill chunks) are skipped.
- **Inputs actually fed**: the serve and C2 pass counts only, no token ids, so the segment state stays 'think' (an
  offline replay shows +0.03 pt share from feeding the ids, see below).
- **Parity**: `streaming/test_gbdt_parity.py`. The online module equals the offline pipeline bitwise over 2000
  refreshes; next_refresh equals sync shifted by one refresh.
- **Serve deps**: lightgbm + narwhals + scipy in `NQ_LGB_DIR` (default `/data/Jarrel/nq-dev/pylgb`, `serve_nq.sh`
  installs it), bind-mounted at /nqlgb and appended to sys.path, so nothing in the image is shadowed.

### Results (2026-09-29)

C2 `c2-gbdt-v1` (same 64 windows and FP8 reference as c2-full-v1; the EMA rows reproduce c2-full-v1 bitwise).
KLD vs FP8, level-4 share of routed slots, streamed traffic:

| stream | KLD id | KLD wikitext | KLD code | share4 id / wiki / code | GB/s @111 |
|---|---|---|---|---|---|
| dyn (EMA) | 0.21950 | 0.07444 | 0.04255 | 0.599 / 0.678 / 0.564 | 3.84 |
| dyn_gbdt | 0.22105 (+0.7%) | 0.07575 (+1.8%) | 0.04454 (+4.7%) | 0.585 / 0.664 / 0.542 | 5.97 |
| dyn_reset (EMA, per-window reset) | 0.21947 | 0.08702 | 0.04678 | 0.581 / 0.641 / 0.531 | 4.84 |
| dyn_gbdt_reset | 0.22256 | 0.08865 | 0.04841 | 0.566 / 0.624 / 0.508 | 5.95 |

Serve (TP4, L3-77, ns3, 2 x 1024 decode tokens, same box, `q5-*` in `/data/Jarrel/nq-serve/val.jsonl`):

| predictor | ms/step | tok/s | rank-0 level-4 hit share (decode minutes) | upgrades per run | rank-0 host loop us/iter | read errors |
|---|---|---|---|---|---|---|
| EMA | 28.33 / 28.41 | 92.9 / 91.5 | 0.417 / 0.394 | 17.3K | 78-108 | 0 |
| GBDT | 28.37 / 28.42 | 95.4 / 94.0 | 0.415 / 0.395 | 35.4K | 84-181 | 0 |

(standing-serve baseline 28.29 ms/step; GBDT late_waits 0, follower backlog 0, coherence pass.)

Promoted under the rule "GBDT unless KLD > 25% worse on a corpus, ms/step > 10% worse or a hard failure". At the
current budget it is a small regression on C2 (KLD +0.7-4.7%, share4 -1.4 to -2.2 pt) and level in the serve.

### Why the offline gain (+1-2 pt share) does not show up: the byte budget

The offline study simulated a 24 GB/s aggregate cap (it took the "6 GB/s per rank" wording above literally); the
serve and C2 charge 6 GB/s aggregate. GBDT refreshes 4x as often and churns ~2.5x as many experts, so at 6 GB/s it is
budget-bound. Replay of the held-out sound-change-cascade decode stream (first 60K tokens, all 75 layers, 3-token
steps) through `scheduler.Scheduler` itself (`/tmp/nq_gapcheck.py`, slots 56 x 75, upgrades land one step later):

| predictor | cap GB/s (aggregate) | share | target landed / in flight / pending | GB/s used | steps with deferred upgrades |
|---|---|---|---|---|---|
| EMA | 6 | 0.6556 | 99.4 / 0.2 / 0.4% | 3.5 | 1.6% |
| EMA | 24 | 0.6565 | 99.7 / 0.3 / 0.0% | 3.6 | 0.2% |
| GBDT | 6 | 0.6446 | 90.6 / 0.4 / 9.0% | 6.0 | 99% |
| GBDT | 12 | 0.6710 | 98.5 / 0.7 / 0.8% | 9.6 | 17% |
| GBDT | 24 | 0.6737 | 99.3 / 0.7 / 0.1% | 9.9 | 0.3% |

- (a) At 6 GB/s, 9% of the GBDT target sits at level 2 waiting for budget; the 56-slot cap never binds.
- (b) The inputs match the offline sim: the EMA replay equals the offline libsim (0.6556 vs 0.6564 at lead 1; 0.6530
  vs 0.6527 at lead 13). Feeding token ids (think/answer state): 0.6740 vs 0.6737. Landing 4 steps late (~lead 13):
  GBDT 0.6658 vs EMA 0.6530 at 24 GB/s, still +1.3 pt. Fixed experts are excluded in both.
- (c) On the NQ routing of C2 the same happens: dyn_gbdt uses 5.97 of the 6 GB/s with deferred upgrades in most steps
  (table above). With a larger budget the offline gain comes back on NQ routing (next section), so the ARVQ-to-NQ
  routing shift is not what held it back.

### Budget raised: NQ_CAP_GBPS 6 -> 12 (2026-09-29)

C2 `c2-gbdt24-aqlm-v1` (dyn EMA at 6, dyn_gbdt at `--gbdt-cap 24`; C2 emulation, one Scheduler per layer with cap/75):

| stream | cap GB/s | KLD id | KLD wikitext | KLD code | share4 id / wiki / code | GB/s used |
|---|---|---|---|---|---|---|
| dyn (EMA) | 6 | 0.21950 | 0.07444 | 0.04255 | 0.599 / 0.678 / 0.564 | 3.84 |
| dyn_gbdt | 24 | 0.21358 (-2.7%) | 0.06962 (-6.5%) | 0.04182 (-1.7%) | 0.630 / 0.688 / 0.580 | 11.2 |

Serve at `NQ_CAP_GBPS=24` (`q5b-*`): GBDT 28.36 / 28.43 ms/step, 95.3 / 93.6 tok/s, rank-0 hit share 0.414 / 0.403,
op p50 104-105 ms; EMA 28.37 / 28.42 ms/step, hit share 0.410 / 0.393. Read errors 0, backlog 0, coherence pass. The
serve made about as many upgrades at 24 as at 6 (35.2K vs 35.4K per run), so its effective budget was already looser
than the nominal 6 (not investigated).

Record store measured directly (2.56 MB O_DIRECT random reads over rank0-3.bin, root Samsung 9100 PRO, measured by the
coordinator during a C2 run): 11.6 GB/s at 4 concurrent reads (p50 0.9 ms), 10.7 GB/s at 16-64 (p50 3.8-10.8 ms,
p99 up to 50 ms). About 11 GB/s aggregate is the practical ceiling, so 24 would promise more than the drive delivers,
and C2 (upgrades land one step later whatever the volume) can't model saturation: treat the 24 GB/s C2 row as an upper
bound. The production default is therefore `NQ_CAP_GBPS=12`: the replay above gives GBDT 0.6710 at 12 vs 0.6737 at
24, with 17% of steps deferring. The serve op p50 of ~105-115 ms is ~100x the raw read latency, so it is pipeline
overhead, not the drive.

## Comparison: AQLM hybrid (prod glm-5.3 checkpoint)

`aqlm` stream = the production GLM-5.3-Vision-NVFP4-AQLM-hybrid-1m experts (`nvfp4_aqlm_hybrid`: hot experts donor
NVFP4, cold 2-bit AQLM 1x16 codebook, group 8, PV-tuned), weights only, dequantised by `aqlmeff.py` exactly as the vLLM
kernels do (cold path bitwise equal to `nvfp4_aqlm_hybrid._dequant_reference`; hot bytes identical to the ARVQ
checkpoint's hot experts). Level 4 = the static hot NVFP4 set (30% of experts resident). Same 64 windows and FP8
reference; the NQ and ARVQ rows are from c2-full-v1 / c2-gbdt24-aqlm-v1 (not rerun).

| stream | KLD id | KLD wikitext | KLD code | level-4 share (routed) | bits/expert resident / served | weight rel. error |
|---|---|---|---|---|---|---|
| all4 (NQ 4-bit everywhere) | 0.17225 | 0.03057 | 0.02322 | 1.000 | 4.334 / 4.334 | 0.068 |
| dyn_gbdt @24 (NQ) | 0.21358 | 0.06962 | 0.04182 | 0.632 | 2.818 / 3.536 | |
| dyn EMA @6 (NQ) | 0.21950 | 0.07444 | 0.04255 | 0.610 | 2.818 / 3.488 | |
| fixed (NQ, no streaming) | 0.28426 | 0.26862 | 0.08081 | 0.120 | 2.386 / 2.427 | |
| all2 (NQ 2-bit everywhere) | 0.31648 | 0.28652 | 0.10108 | 0.000 | 2.165 / 2.165 | 0.265 |
| AQLM hybrid (prod) | 0.32280 | 0.36022 | 0.06888 | 0.346 | 2.753 / 2.870 | 0.253 (all experts) |
| ARVQ hybrid | 0.43643 | 0.78450 | 0.10607 | 0.349 | 2.792 / 2.912 | 0.349 (all experts) |

AQLM cold 2.007 bpw incl. codebooks, hot 4.5. AQLM beats ARVQ on every corpus (KLD -26% id, -54% wikitext, -35% code),
but NQ dyn beats AQLM on every corpus (-32% id, -79% wikitext, -38% code) at about the same resident bits.
Full tables: `sm120/results/c2/c2-gbdt24-aqlm-v1.md`.

