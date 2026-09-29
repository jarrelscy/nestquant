# Thread 31: per-expert upgrade-benefit table (nq-delta-v1)

The SM120 serve ranks experts for the 2-bit to 4-bit upgrade within each layer using
`benefit_e ~ delta_e * sum_t w_{t,e}^2 ||x_t||^2`. This table supplies `delta` per (layer, expert). The value is
independent of TP rank.

## Files

| file | public? | content |
|---|---|---|
| `delta_table.json`, `delta_table.npz` | yes (HF serving/predictor/) | schema `nq-delta-v1`, `version`, `definition`, `manifest_sha256` per layer, and 256 values per layer for `e2 e4 drel G delta n_routed` (npz: [75, 256], layers 3..77) |
| `private/delta_extra.npz` | no | the columns above plus G_plain, delta_plain, kappa, e2/e4/drel/delta_comp, reap, proxy per projection, raw traces |
| `private/provenance.json` | no | manifest and capture paths, extra-column definitions |
| `private/sanity.{txt,json}` | no | sanity report |

## Definition

- **e2, e4.** Computed as `sqrt(mean_{gate,up,down} proxy_rot[v])`, the same combination `nq25_rescore` uses (rms3). proxy_rot comes from the served manifests; for L3-6 these are the T29 h512 refit.
- **drel.** `e2^2 - e4^2`.
- **G.** `tr(W_d D2 W_d^T) / tr(A2)`, which equals `E_{p^2}||y||^2 / E_{p^2}||x||^2`. The tokens are the calibration tokens routed to the expert, each weighted by p^2, the same weighting as the serve's `w^2 ||x||^2`. This is exact, with no mean-||y|| approximation:
  - A2 = sum p^2 x x^T. Only its diagonal is read.
  - D2 = sum p^2 h h^T.
  - W_d is the bf16-cast FP8 teacher.
- **delta.** `drel * G`.
- **Unweighted variant.** `G_plain` uses A0/D0 instead of A2/D2. It is private only; its within-layer Spearman with delta is 0.91-0.96.

## Recompute (CPU, about 12 min, about 1 GB RSS per worker)

```
cd threads/31-delta
OMP_NUM_THREADS=12 python -W ignore nq31_gain.py --layers 3-77 --workers 8 --threads 12   # -> gain/L{L}.npz
python nq31_table.py                                                                    # -> table + private/
```

## Sanity checks and extras

See `private/sanity.txt` for the numbers.

- **Composed estimator (`e*_comp`, private).** `ec^2 = p_down + kappa (p_gate + p_up)`:
  - The gate/up proxies are output-weighted, so each measures the relative h-error it causes.
  - kappa is W_d's gain on channel-uncorrelated h-noise relative to its gain on the signal, and comes out at about 0.9-1.0.
  - The composed ranking matches rms3 almost exactly (within-layer Spearman 0.999, 1-2 of the top-45 swapped).
  - It matches the absolute scale of the measured L7-77 spot errors better (measured/pred 0.86 vs 1.48), but L3-6 worse, and its rank correlation is slightly lower.
  - Default stays rms3.
- **Mean-||y|| / layer-||x||^2 approximation.** It would have given within-layer Spearman with the exact delta of only 0.55 (L3-6), 0.81 (L7-40) and 0.92 (L41-77). The exact traces are worth it.
