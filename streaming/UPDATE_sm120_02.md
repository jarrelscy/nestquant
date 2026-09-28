# Update 02 for the SM120 agent: fixed set chosen by boundary-weighted REAP (2026-09-28, revised)

The user wants experts used just before boundary tokens (`</think>`, and end of turn `<|user|>` / `<|observation|>`) to be at 4 bit, because overthinking and failing to stop are known failure modes.

## What changes for you
1. **How the fixed set is chosen.** It is the top 26 experts per layer (10%) by token-weighted REAP salience, Σ_t w_t·p·‖y‖. Weights depend on the token's distance before a boundary: 50 at 1 token, 20 at 2–4, 5 at 5–16, 2 at 17–32, and 1 everywhere else. The artifact ships this as `fixed_set.json` (`{layer: [experts]}`). Load it at startup in place of your own usage-based fixed set.
2. **Scheduler.** No other change. The floating policy works as before.
3. **Report.** In the tb4 runs, report the 4-bit share of routed slots at 1 and 2–4 tokens before boundaries separately from the overall share.

## Expected effect (chunk 0, share of routes that land in the fixed set)
| | all tokens | end, 1 before | end, 2–4 before | think, 1 before |
|---|---|---|---|---|
| unweighted REAP | 0.207 | 0.330 | 0.193 | 0.249 |
| weighted REAP | 0.207 | 0.447 | 0.224 | 0.281 |

Code: `threads/22-boundary-experts/reap_weighted.py`.
