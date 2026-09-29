# Expert predictor used by serving and by the C2 `dyn` stream

Each MoE layer has 256 routed experts. Every expert is resident at level 2 (2-bit base). A subset is also held at
level 4 (2-bit base + 2-bit residual streamed from NVMe). The predictor decides which experts are at level 4.

## Level-4 set per layer

- **Fixed set, 26 experts (10%)**: loaded at startup, never evicted. Chosen offline by token-weighted REAP salience
  (mean gate weight x output norm, with extra weight on boundary tokens such as `</think>`), not by routing
  frequency (`streaming/fixed_set.py`, `threads/22-boundary-experts/fixed_set.json`). On C2 it covers 12% of routed
  slots, close to the 10.2% a random set would get, so it protects high-salience experts rather than busy ones.
- **Floating set, 51 experts (20%)**: chosen online from recent routing (`streaming/scheduler.py`).
- Total 77 of 256 (30%) at level 4 at any time.

## Floating set (`scheduler.Scheduler`)

1. **Start**: the 51 most-routed non-fixed experts from the calibration capture (`floating_default`) are loaded at
   startup, so token 0 already has 30% of experts at level 4.
2. **Score**: per expert, `score = score * 0.5**(ntok/512) + count`, where `count` is how many of the step's routed
   slots (8 per token) went to that expert. This is an exponential moving average of routing counts with a half-life
   of 512 tokens. It is never reset, so it carries across requests.
3. **Refresh every 64 tokens**: the target set becomes the top 51 non-fixed experts by score (stable sort, ties to the
   lower index). Layers with no counts yet keep `floating_default`.
4. **Downgrades** (experts that left the target) are immediate: the kernel switches back to the resident level-2 rows,
   with no I/O.
5. **Upgrades** (experts that joined the target) are issued highest score first under a byte budget: 6 GB/s per rank
   at an assumed 111 tok/s, accrued per token and capped at a 64-token burst. The rest wait at level 2 and are retried
   on the next step. Each upgrade reads one 2.56 MB per-rank record.
6. **Guard**: no upgrades in a step that touches more than half of a layer's experts (prefill-sized steps), so a long
   prompt updates the scores without flooding the SSD.
7. **Slots**: at most 56 streamed experts per layer in flight, landed or draining.

All ranks run the same policy. In serving, rank 0 schedules and the other ranks replay its op log from /dev/shm.

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
