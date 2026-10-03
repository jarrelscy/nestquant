# T36: does SSD landing time hurt b175 on 2x DGX Spark?

CPU replay of b175 (1.75-bit base + 4-bit residual) with H=45 hot slots per layer, fp8 KV at 128K, and the jF
predictor (k0, hm 0.7). The residuals stream from NVMe and take real time to land. Inputs are the private sm120tf
decode traces (75 layers, 6 chains, 3.52M tokens, 220k blocks of 16); per-token routing and salience
(w²·|x|²) were rebuilt from trace_sm120 and match the stored block counts exactly on all 75 layers. jF scores are
out-of-fold per chain. Inputs stay local; this directory holds the code and aggregate results only.

**Model.** TP2: each node reads its half of each residual (5.71 MB). Each node has one FIFO across all layers:
service = 5.71 MB / B, plus 0.2 ms latency. A residual counts from the first token that starts after it lands.
Evictions free the slot immediately. Swaps are issued at block start in layer order.

**Measured:** pooled 4-bit salience share (`sal`), routed-call share, fetches/layer/block, landing delay,
SSD utilisation and staging. **Estimated:** KLD, from the linear rule (slope 0.084 per unit salience), anchored so
arm A = 0.0432 (the measured b175 H45 KLD). These are not KLD measurements.

## Main case: 6.6 GB/s per node, 15 tok/s

| arm | sal | calls | fetch/L/blk | delay ms | SSD util | est. KLD |
|---|---|---|---|---|---|---|
| A perfect landing (parta replay) | 0.7355 | 0.557 | 1.43 | 0 | 0.09 | 0.0432 |
| B today: issue at block start, real landing | 0.7317 | 0.554 | 1.43 | 80 | 0.09 | 0.0435 |
| C lead 1 block (stale scores), H45, staging free | 0.7205 | 0.545 | 1.43 | 80 | 0.09 | 0.0445 |
| C lead 2 blocks, H45 | 0.7111 | 0.537 | 1.43 | 80 | 0.09 | 0.0453 |
| C lead 1, H44 (pays for staging) | 0.7154 | 0.539 | 1.41 | 79 | 0.09 | 0.0449 |
| D top-up k=0.25/layer/token (causal in-block routing) | **0.7375** | 0.559 | 6.78 | 64 | 0.41 | **0.0430** |
| D top-up k=0.5 | 0.7369 | 0.559 | 11.2 | 97 | 0.68 | 0.0431 |
| D top-up k=1 | 0.7235 | 0.549 | 16.4 | 190 | 1.00 | 0.0442 |
| D top-up k=2 | 0.7209 | 0.539 | 16.4 | 216 | 1.00 | 0.0444 |
| E oracle, ≤8 swaps/L/blk, issued at block start | 0.7893 | 0.633 | 8.0 | 260 | 0.49 | 0.0387 |
| E oracle, ≤14 (SSD max), issued at block start | 0.7657 | 0.675 | 14.0 | 454 | 0.85 | 0.0407 |
| E oracle, ≤14, one block ahead (14 H-slots of staging) | 0.9323 | 0.778 | 14.0 | 454 | 0.85 | 0.0267 |
| E oracle, unlimited swaps, perfect landing | 0.9651 | 0.848 | 23.9 | 0 | 1.45 | 0.0239 |

## Arm B and the best arm D at each SSD speed

| SSD GB/s/node | tok/s | B sal | B delay ms | D k0.25 sal | D k0.5 sal | B est. KLD | best D est. KLD |
|---|---|---|---|---|---|---|---|
| 3.5 | 10 / 15 / 20 | 0.7311 / 0.7294 / 0.7277 | 152-168 | 0.7364 / 0.7323 / 0.7281 | 0.7342 / 0.7288 / 0.7247 | 0.0436-0.0439 | 0.0431-0.0438 |
| 6.6 | 10 / 15 / 20 | 0.7326 / 0.7317 / 0.7309 | 80 | 0.7389 / 0.7375 / 0.7359 | 0.7392 / 0.7369 / 0.7333 | 0.0434-0.0436 | 0.0429-0.0432 |
| 11 | 10 / 15 / 20 | 0.7333 / 0.7328 / 0.7323 | 48 | 0.7400 / 0.7393 / 0.7383 | 0.7413 / 0.7396 / 0.7381 | 0.0434-0.0435 | 0.0427-0.0430 |

Arm C is the same at every speed and rate (0.720 lead 1, 0.711 lead 2 at H45). The full grid, including C at
H42-45 and E budgets 2/4/8/max for lead 0 and lead 1, is in `results/t36_results.json`. `kld_est_rel` is the
anchored estimate.

## Takeaways

- **Landing delay barely matters for today's schedule.** jF changes ~1.4 experts per layer per block, which is
  ~9% of a 6.6 GB/s SSD. The swaps land in ~80 ms (~1.2 of 16 tokens). Cost vs perfect landing: −0.004 sal,
  about +0.0003 est. KLD; −0.006 at 3.5 GB/s.
- **Leading with stale scores loses** (−0.011 sal per block of lead) before paying for staging. Mean staging is
  ~107 experts/node per block of lead (~0.6 GB, 1.4 H-slots); the peak is 1803 (10 GB) at chain/topic changes.
  Don't prefetch early.
- **Small per-token top-ups help a little.** k=0.25-0.5 swaps/layer/token, driven by the experts routed so far in
  the block and evicting the lowest in-block salience (then lowest jF rank), beats even perfect landing:
  +0.002 to +0.006 sal over A, est. KLD 0.0430 vs 0.0435 for B. k≥1 saturates the SSD and delays the block-start
  swaps, so it loses. Top-ups load in place, so no staging is needed.
- **The SSD is not the limit, the predictor is.** A bandwidth-limited oracle with 8 swaps/layer/block issued at
  block start gets 0.79 (est. KLD 0.039). Knowing the set one block ahead gets 0.93, but needs ~14 H-slots of
  staging.

Run: `prep.py` (private inputs → /tmp/nestquant/36-spark-land/private), then `drive.py` (A-D + C at H42-45) and
`drive_e.py` (budgeted oracle E + D k0.25). numba, CPU only, ~45 min on 18 procs.
