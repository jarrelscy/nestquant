# T27: PV-style continuous post-tuning of the GLM-5.3 NestQuant encode

**Verdict: no-go for a full layer-wise PV run.**
- Per-expert continuous tuning works and is cheap. It gives a clear, reproducible gain in local routed error.
- It does not turn into a full-model KLD gain that justifies a 75-layer run:
  - Of four single-layer tests (L10, L29, L50, L70), only L29 shows a clear wikitext gain (-0.00078 ± 0.00020).
  - L10 regresses (wikitext +0.00064 ± 0.00048, github +0.00111 ± 0.00051).
  - L50 and L70 are null.
  - Extrapolated over 75 layers, the total is far below the 0.003 wikitext gate, and early layers may be net negative.
- The remaining gap is not a per-expert weight-fit problem:
  - The layer target is dominated by upstream drift: ‖D‖/‖target‖ is 0.51-0.53 at L10/L29/L30 and 0.79-0.86 at L50/L70.
  - **See T18's routing attribution** (threads/18-e2e-eval/results/passR1*, passR2*, commit e7c9b17). FP8-oracle routing on the nqdef model gives wikitext dKLD -0.049 ± 0.002, about 60x the best PV layer. That is where the remaining KLD is.
- Nothing was uploaded to HF, and nq-encode-v1 was not modified.

All code is in this directory. Small result files (eval/val tables, layer evals, per-arm summaries) are in `results/`. Large artifacts are in /tmp/nestquant/27-pv-tune (section 6).

## 1. Method

- **What is tuned.** Only continuous parameters, over the fixed discrete codes:
  - su/sv, stored per level: p/base = L2, p/p4 = L4.
  - The fp16 U2, U4 and V factors. U2 and V are shared by both levels. U4 is L4-only. gate and up share V.
- **Format.** The layout is unchanged: identical tree, dtypes, shapes and code bytes, checked by `layout_check`. Tuned artifacts decode through `nq_decode` deterministically at L2 and L4, and match the fp32 tuning model to about 5e-4 relative (fp16 rounding).
- **Loss.** `a*err(L2) + (1-a)*err(L4)` with a = 0.5. The error is the p²-weighted relative SwiGLU output error against the FP8 teacher on routed rows.
- **Robust version.** Each level is normalised by its untuned error. Row weights are capped at the 0.99 energy quantile. Warmup, lr 3e-3, early stop on held-out blocks, and a divergence fallback to lr/3 and lr/10.
- **Forced regulariser γ.** Adds γ times the error on uniform non-routed rows (expert applied at weight 1).
- **Data.**
  - Per-expert pilot: T19 chunk-0 routed activations. The held-out split is chunk-0 blocks; eval/val is untouched and used for scoring.
  - Band: T18 hdump. The held-out split is window % 10 == 0.
- **Files.**
  - `nq27_tune.py`: model and loss.
  - `nq27_run.py`: per-expert driver.
  - `nq27_eval.py`, `nq27_table.py`, `nq27_ess.py`: eval/val scoring against the nq25_spot EXL3 refs.
  - `nq27_rows.py`, `nq27_frows.py`: row extraction.
  - `nq27_band.py`: band pilot. Subcommands `prep` / `tune` / `eval` / `down`.
  - `nq27_pv.py`: the V step, re-encode with propagated H through the pinned T23 path, imported read-only.

## 2. Per-expert pilot (L3-L6 plus L30 control, 48 experts, eval/val)

Mean change in relative error, tuned vs untuned, in %, at L2 / L4. The routed metric is all/routed; the forced metric is all/forced. Per-expert tables with ESS are in `results/table_*.md`.

| arm | set | routed dL2 / dL4 | forced dL2 / dL4 |
|---|---|---|---|
| act_c99 (γ=0) | L3-6 (40) | -4.02 / -2.33 | **+1.97 / +4.92** (worst +17.6 / +69.0, L3 E138) |
| act_c99 | L3-6 flagged (8) | -3.60 / -1.66 | +1.81 / +2.55 |
| act_c99 | L30 control (8) | -1.54 / -0.28 | +0.38 / +0.42 |
| **act_c99_g1 (γ=1)** | L3-6 (40) | **-3.23 / -1.69** | **-0.79 / -0.62** (every expert improves; max -0.25 / -0.09) |
| act_c99_g1 | L3-6 flagged (8) | -3.11 / -1.51 | -0.95 / -0.83 |
| act_c99_g1 | L30 control (8) | -1.47 / -0.31 | -0.09 / +0.02 |

**Gap to EXL3 at L3-6, routed.**

| | L2 | L4 |
|---|---|---|
| untuned | +20.3% | +5.2% |
| act_c99 | +15.2% | +2.6% |
| γ=1 | +16.3% | +3.3% |

**Findings.**
- **ESS.** On many experts the p²-weighted metric has an effective sample size of only 5-70 tokens: a few massive-output tokens carry more than 50% of the energy.
  - A raw L2 objective at lr 1e-2 overfits or diverges (L5 E134, L3 E134). The robust loss fixes this.
  - Gains are not an ESS artefact. For act_c99, ESS_val ≥ 100 gives -4.02 / -2.71 and ESS_val < 100 gives -4.03 / -1.70.
- **Per-level scales confirmed.** su/sv are stored per level and tuned separately. U2 and V are shared across levels, which is why the loss is joint.
- **Forced regression and fix.** Routed-only objectives let su/sv move freely on channels with little routed signal (max |Δ| reaches 3.4). This hurts under routing drift.
  - γ = 1 removes the regression at about 0.8 points of routed dL2.
  - γ = 0.25 sat in between, with slightly worse forced numbers. **γ = 1 was chosen.**
- **a trade-off** (flagged 8, dL2 / dL4). a = 0.5 was kept.

| a | dL2 / dL4 |
|---|---|
| 0.25 | -3.58 / -1.77 |
| **0.5** | **-3.60 / -1.66** |
| 0.75 | -3.82 / -0.90 (one expert +6.8% L4) |
| 1.0 | -3.90 / +80% L4 |

- **H objective** (full-stats Hessian proxy instead of activations). Weaker: routed -3.04 / -1.83. It has the same forced regression (+2.5 / +3.9).
- **Speed.** 500 steps TF32 (fast_tf32) keeps about 95% of the gain: L3-6 -3.41 / -1.59 vs -3.59 / -1.66 on the same 16 experts; L30 -1.48 / -0.26 vs -1.54 / -0.28.

## 3. Band pilot on propagated inputs (L29, L30; T18 hdump, nqdef upstream)

**Targets.**
- **same:** R_fp8 = Σ p_k f^FP8_{E_k}(x_nqdef), i.e. FP8 experts on the propagated input.
- **ref:** f^FP8_E(x_nqdef) + p_E·D/Σp², with D = (d_ref - shared) - R_fp8. d_ref = h_out_fp8 - h_mid_nqdef, so this target also asks each expert to absorb the upstream residual drift D.

**Layer eval.** Held-out routed-MoE relative error of the deployed nqdef mix (defset: 26 L4 experts, the rest L2). Tuning: 800 steps TF32, γ=1, a=0.5, all 256 experts.

| layer / dump | arm | drift ‖D‖/‖tgt‖ | mix vs same | mix vs ref | pure L2 vs same | pure L4 vs same |
|---|---|---|---|---|---|---|
| L29 / hdump | base | 0.53 | 0.2275 | 0.5605 | 0.2728 | 0.0617 |
| | band_same | | 0.2234 | 0.5594 | 0.2668 | 0.0614 |
| | band_ref | | 0.2240 | 0.5588 | 0.2675 | 0.0620 |
| | pvV (V only) | | 0.2269 | 0.5609 | 0.2720 | 0.0618 |
| | pvVP_ref (V then P) | | 0.2235 | 0.5592 | 0.2669 | 0.0621 |
| L30 / ovr29_same | base | 0.52 | 0.2325 | 0.5573 | 0.2660 | 0.0608 |
| | band_same | | 0.2279 | 0.5562 | 0.2606 | 0.0605 |
| L30 / ovr29_ref | base | 0.52 | 0.2324 | 0.5569 | 0.2660 | 0.0608 |
| | band_ref | | 0.2284 | 0.5554 | 0.2612 | 0.0609 |

- **Per-expert held-out** (chunk-0 windows), mean dL2 / dL4 / forced dF2 / dF4 in %:

| layer | same target | ref target |
|---|---|---|
| L29 | -1.23 / -0.51 / -0.02 / +0.07 | -0.58 / -0.10 / -0.13 / -0.12 |
| L30 | -1.20 / -0.49 / -0.03 / +0.07 | -0.54 / -0.07 / -0.15 / -0.14 |

  L30 (tuned after a tuned L29, on its own re-dump) replicates L29. The gain neither compounds nor decays. L30's layer gain vs same is -2.0% for both targets, a little more than L29's -1.5 to -1.8%.
- **Propagation is tiny.** Tuning L29 moves the L30 input error by only -0.25% (0.0428 -> 0.0427). The next-layer re-dump is therefore close to the base dump, which is why layer-parallel tuning was acceptable.
- **Downstream at L30-32** (held-out, `down`):
  - h_out error -0.20 to -0.33%.
  - Per-token median -0.3%.
  - Top-1 routing +0.07 to +0.23%.
  - Router overlap about 0.88 at base.
  - ref is marginally better than same.
- **Massive-norm tokens dominate global metrics.** At L29, moe_out global error is 4.6% but the per-token median is 29%, so medians are reported alongside.

**L29-only full-model KLD** (T18, paired vs nqdef, ± se; T18 results/passO29_*, commit 0c32827):

| arm | nq-tail | vllm-docs | wikitext | github |
|---|---|---|---|---|
| band_same | -0.00027 ±0.00019 | -0.00057 ±0.00016 | -0.00034 ±0.00020 | -0.00086 ±0.00036 |
| band_ref | -0.00025 ±0.00021 | -0.00059 ±0.00017 | **-0.00078 ±0.00020** | -0.00042 ±0.00026 |

Baseline nqdef KLD: 0.0965 / 0.1114 / 0.2874 / 0.0807. The two targets are statistically indistinguishable; ref was chosen because wikitext is the final metric. Sequential L29-32 KLD was not run: the full run was replaced by the single-layer typicality test in section 4.

### P+V arm (L29): re-encode with propagated H, then P

- **V step (`nq27_pv.py`).** H_text' = nt(H_T19) + nt(Ĥ_nqdef) - nt(Ĥ_fp8), PSD-clamped.
  - Ĥ uses the thread-08 recipe (alpha 0.25, ctx_mass 0.25, 25% context sample) on the dump.
  - G and the vision side are the campaign's (BlendCapture w = 0.25).
  - The encode goes through the pinned T23 `encode_group` with the production config.
- **Sanity check.** With propagation off, the output is bit-identical to nq-encode-v1.
- **H shift.** Relative shift is 0.058 for gate/up and 0.105 for down.
- **Cost.** About 7 s for H plus 38 s encode per expert under contention.
- **Result.** The V step alone barely moves the layer (mix vs same -0.3%, mix vs ref +0.07%). V then P (pvVP_ref) matches P alone (band_ref): mix vs ref 0.5592 vs 0.5588, mix vs same 0.2235 vs 0.2240.
- **Conclusion.** Re-encoding with propagated Hessians adds nothing over P alone at about 2.5x the cost, so it was not sent for KLD.

## 4. Is L29 typical? L10, L50, L70 each alone (ref, γ=1, 500 steps)

Tuned on T18's hdump_L10_50_70 (nqdef upstream, T29 h512 for L3-6). The base is nq-encode-h512, which is v1 for L7+. All 256 experts per layer, about 41 s per expert under contention, deterministic decode.

| layer | drift ‖D‖/‖tgt‖ | mix vs ref | mix vs same | pure L2 / L4 vs ref | pure L2 / L4 vs same | per-expert dL2 / dL4 |
|---|---|---|---|---|---|---|
| L10 | 0.51 | 0.5389 -> 0.5318 (-1.33%) | 0.1816 -> 0.1787 (-1.63%) | -2.8% / -0.7% | **+1.6% / +4.6%** | -1.58 / -0.54 |
| L29 | 0.53 | 0.5605 -> 0.5588 (-0.31%) | 0.2275 -> 0.2240 (-1.54%) | -0.45% / -0.08% | -1.9% / +0.5% | -0.58 / -0.10 |
| L50 | 0.79 | 0.7955 -> 0.7953 (-0.03%) | 0.1941 -> 0.1929 (-0.62%) | -0.04% / 0.00% | -0.6% / +0.1% | -0.10 / -0.01 |
| L70 | 0.86 | 0.8670 -> 0.8668 (-0.02%) | 0.2203 -> 0.2192 (-0.51%) | -0.03% / -0.01% | -0.7% / +0.8% | -0.11 / -0.03 |

**Single-layer full-model KLD** (T18 passOw/passOr, paired vs h512 nqdef, ± se, fraction of windows improved in brackets):

| layer | wikitext | nq-tail | vllm-docs | github |
|---|---|---|---|---|
| L10 | **+0.00064 ±0.00048** (0.48) | +0.00011 ±0.00026 | -0.00032 ±0.00023 | **+0.00111 ±0.00051** (0.32) |
| L29 | -0.00078 ±0.00020 | -0.00025 ±0.00021 | -0.00059 ±0.00017 | -0.00042 ±0.00026 |
| L50 | -0.00013 ±0.00013 (0.56) | -0.00009 ±0.00007 | -0.00010 ±0.00007 | +0.00004 ±0.00007 |
| L70 | -0.00000 ±0.00006 (0.45) | +0.00008 ±0.00004 | +0.00001 ±0.00005 | +0.00005 ±0.00005 |

**Reading.**
- **Layer-level gain does not predict KLD.** L10 has the largest gain on the ref target and is the only regression.
  - At L10 the ref target pulls the pure levels away from the same-input target (L4 +4.6%): the layer absorbs upstream drift that downstream layers apparently deal with anyway, so the absorbed correction hurts.
  - At L50/L70 the target is 79-86% unreachable drift, so the ref objective is flat and nothing moves.
- **L29 is the exception, not the rule.** A crude 75-layer extrapolation from the four points is about -0.0002 to -0.0003 per middle layer, zero for deep layers, and possibly positive for early layers. That is far from the 0.003 wikitext gate.
- **Planned full run and why it was cancelled.** The plan was layer-parallel on one base dump: 500 steps, ref, γ=1, about 12 h on 8 GPUs, plus about 480-950 GB of dumps in bands. The lead and the user cancelled it on this evidence.
- **Where to look instead.** T18's routing attribution (passR1/R2, e7c9b17): FP8-oracle routing on nqdef recovers wikitext -0.049 ± 0.002 (-17%). The remaining KLD is a routing and drift problem, not a per-expert weight-fit problem.

## 5. Costs

| item | cost |
|---|---|
| per-expert pilot tune, act_c99 1500 steps fp32 | about 359 s/expert contended |
| per-expert tune, 500 steps TF32 | 25.7 s/expert (17.3 s tune), about 139 GPU-h (93 tune-only) for 19,456 experts |
| band tune, 800 steps TF32 | about 60 s/expert-proc contended, about 2 GPU-h/layer |
| band tune, 500 steps TF32 | about 41 s/expert-proc contended, about 1.2 GPU-h/layer |
| V step (propagated-H re-encode) | about 45 s/expert contended, about 3 GPU-h/layer |
| T18 full-forward re-dump | about 17-20 min on 8 GPUs, about 40-64 GB/layer |

## 6. Scratch data (/tmp/nestquant/27-pv-tune)

Everything needed for this report is in `results/`, so the scratch can be deleted.

| path | size | content |
|---|---|---|
| band/ | 59 GB | band arms (band_same, band_ref, lp_ref, pvV, pvVP_ref), D prep tensors (19 GB) |
| rows/ | 20 GB | T19 chunk-0 routed rows for the 48 pilot experts |
| arms/ | 4.0 GB | pilot arms (all a/γ/H/lr variants) |
| frows/ | 2.4 GB | forced-regulariser rows |
| logs/, val/, tables | about 2 MB | logs, eval/val JSONs |

T18 dumps used only by T27: /tmp/nestquant/18-e2e/hdump/L29 (40 GB), hdump_ovr29_ref (64 GB) and hdump_ovr29_same (64 GB).
