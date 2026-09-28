# Update 02 for the SM120 agent: boundary experts are always hot (2026-09-28)

The user wants experts that fire consistently just before boundary tokens (`</think>`, and end of turn `<|user|>` / `<|observation|>`) to always be at 4 bit. Overthinking and failing to stop are known failure modes, and the tokens leading up to a boundary are where the stop decision is made.

## What changes for you
1. **Fixed set.** The fixed ~10% always-4-bit set (~26 experts per layer) must include every expert listed in the artifact's `boundary_hot.json`. Fill the rest of the fixed set by usage, as before. Boundary experts are never downgraded by the floating policy. Under KV pressure, drop floating experts first, and drop boundary experts only as the very last step.
2. **Size.** Partial run on the new corpus: 342 experts across 75 layers (median 5 per layer, max 12). The final list comes out when the full capture finishes, and should be similar in size. Budget up to 12 per layer; no per-layer list is published yet.
3. **Format.** `boundary_hot.json` is `{layer: [{e, kind, bucket, hit, lift}]}`, shipped in the HF repo next to the manifest. The manifest also flags `boundary_hot: true` per (layer, expert). Load it at startup, together with the fixed set.
4. **Report.** In the tb4 runs, report the 4-bit share of routed slots in the 32 tokens before each boundary separately from the overall share. It should be close to 100% for the boundary experts.

## How the set is chosen
Take rows 1 and 2–4 tokens before each boundary. Split them into two halves by document. An expert counts if it is picked on ≥30% of those rows AND at ≥3x its usual rate, in both halves. Code and data are in `threads/22-boundary-experts/`.

End-of-turn preference is very reproducible (split-half rank correlation 0.84–0.95 per layer). The `</think>` data is still thin (219 events per layer) until the reasoning-trace capture lands.
