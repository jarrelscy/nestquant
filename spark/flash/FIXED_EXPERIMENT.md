# Fixed-share U experiment

Opt in using `NQ_FLASH_FIXED_FRACTION=0.2` with
`spark_256K_U_2630_flat50`. All layer budgets retain their previous totals.
Largest-remainder rounding (ties by layer ID) allocates526 fixed and2104 floating
experts across42 layers, with8 separate spare slots. The same fraction applies
to each layer before rounding; it is not20% of all288 experts per layer.

Fixed IDs are the highest published `fixed_set.json:score` salience values in each layer,
with expert-ID tie breaking. The shipped top19 `fixed_set` is validated against that ranking. Remaining floating slots initialize by `n_routed` frequency, excluding fixed IDs. They remain wanted across request resets and all
refreshes. Existing startup drain loads them before serving. The pool's existing
wanted mask makes them ineligible for demotion or stale-read cancellation.
Selection still uses the shipped causal block prediction, EMA, refresh cadence
and hysteresis, but the fixed IDs are excluded before ranking the remaining
floating slots. Full288-expert normalization is retained. Predictor state only
advances on committed rows as before. This changes allocation policy; it has no
claim to the zero-fixed research KLD or benchmark scores.

Tests cover zero-fixed byte-for-byte mask/state parity, total budgets and fraction
rounding, fixed IDs surviving zero salience across refreshes, request reset, and
actual Pool operations never demoting fixed experts. Runtime startup reports
fixed totals and per-layer counts. Other allocation totals are included within
2630, not additional memory. Fixed records still use the existing acknowledged
slot path; no new expert tensor/kernel format is introduced.

This is a local experiment. Published U2630/FP4 defaults are not changed.
The last10 benchmark tasks restart in separate artifacts, with identical sampling,
MTP2, FP4 cache, pacing and seeds, after a fresh server boot.

Correction: earlier fixed-share experiments used routing frequency to pin experts. They are frequency-selected baselines, not results for this salience-based policy. Zero-fixed behavior is unchanged. Full80 rerun uses separate artifacts.
