# Thread 19 capture format (GLM-5.3, all routed experts of MoE layers 3..77)

Two roots, same schema (Python: `import nq19_load as C`, PYTHONPATH as in `run.sh`):

| root | corpus | status |
|---|---|---|
| `/tmp/nestquant/19-capture-glmfmt` | thread-21 GLM-native `glm53_calib_glmfmt_v1` (c512 + c2048 + c2048_traces groups) | **`stats0` = chunk 0; `stats1` = chunk 0 + traces (fixed set, T12 A/Bs); `stats` after shards 12-24 = production encode (15.4M tokens)** |
| `/tmp/nestquant/19-capture` (default, `NQ19_OUT`) | old orbit `glm53_training_15m_v2` (512-token windows) | chunk 0 = fallback; shards 1-13 held |

Pass the root explicitly: `C.Capture(root="/tmp/nestquant/19-capture-glmfmt", stats="stats0")`.

```python
cap  = C.Capture()                    # cumulative stats (all shards merged so far)
cap0 = C.Capture(stats="stats0")      # frozen chunk-0 snapshot (1.05M tokens)
cap1 = C.Capture(stats="stats1")      # frozen chunk-0 + traces snapshot (2.61M tokens): fixed set, T12 A/Bs
HG   = cap.glm_H(L, E)                # thread-12 glm_H dict: {"H": [Hx, Hx, Ha], "G": [Gg, Gu, None], "meta": {...}}
st   = cap.pilot_stats(L, E)          # orbit statistics-style {"grams": [Wx, Wd], "metadata": {training_rows, mass}}
c    = cap.components(L, E)           # raw sums (below)
cb   = cap.components_bnd(L, E)       # {(kind, bucket): same sums restricted to boundary rows} (below)
HGb  = cap.glm_H(L, E, bnd_w=50)      # boundary-weighted recipe: experiments only, NOT the encode (see below)
sal  = cap.salience(L)                # REAP / usage per expert per category (weights=... only for fixed-set style scores)
U    = cap.uniform_H(L)               # per-layer all-token gate/up Gram / T_fit   (which="C_ctx" for context rows)
ev   = cap.eval_capture(L, "val")     # harness capture dict (also "matched")
data = cap.expert_data(L, E, "val")   # harness.ExpertData (FP8 teacher, pilot_stats, eval capture)
```
**Encode = weight 1 everywhere:** `glm_H(L, E)` with no `bnd_w`. Boundary weights only enter `fixed_set.json`.

Every layer resolves to one immutable version directory the first time it's accessed, and all of that
directory's files are opened (mmapped) at that point. A concurrent merge, which swaps the symlink atomically
and then deletes the superseded version, can't change what an existing `Capture` object sees. Create a new
`Capture` to pick up newer shards.

The damping is not stored in H; the encoder adds it. Thread-08 selection: sigma = 0.5 for gate/up and 1.0 for
down, with count = 1 (`C.GU_SIGMA`, `C.DOWN_SIGMA`). These are the same arguments as thread 12's
`NE.prep(W, H, 1, sigma, G=G)` and harness `quantize_exl3_like(W, H, K, count=1, sigma_reg=sigma)`.

## Calibration data

- **Corpus:** orbit `runs/glm53_training_15m_v2` (training only, 29,296 windows of 512 tokens).
  - Fit windows: 0..28655. Val windows: 28656..28783.
  - Windows ≥ 28784 are thread 18's nq-tail and are never read.
- **Progressive shards:** shard k covers fit windows [2048k, min(2048(k+1), 28656)). That gives 14 shards
  (13 × 1,048,576 + 1 × 1,040,384 tokens = 14,671,872 fit tokens).
- **Stats are additive:** every stage-2 job adds the sums of one or more shards of one layer into `stats/L{L}`.
  Shard 0 always merges alone; later jobs merge up to 2 pending shards at once (`--max-shards`), which halves the
  cumulative-stats IO. The result matches sequential merges to within fp32 summation order (≤5e-8 relative).
  `meta.json["shards"]` lists the shards merged into that version; `n` in `L{L}.v{n}` is their count.
- **Forward arithmetic:** the orbit pilot's, reproduced bitwise at shard-0 chunk 0 = the pilot's 65,536 tokens.
  - Attention is document-local per 512-token window.
  - MoE accumulation is expert-index-ordered bf16.
  - Details are in `capture_fwd.py`'s docstring.
- **Context rows per shard:** `ctx_k = randperm(T_k, generator seed 20260925 + k)[:T_k // 4]`.
  Shard 0 at T = 65,536 is exactly orbit's CalibrationBatches context set.

## `stats/L{L}` → `L{L}.v{n}` (symlink; n = shards merged). Schema `nestquant-19-stats-v2`

| file | shape / dtype | content |
|---|---|---|
| `A2.f32` | raw [256, stride] f32, packed row = npk(6144) | Σ_routed p² x xᵀ (gate/up input) |
| `A0.f32` | same | Σ_routed x xᵀ |
| `D2.f32` | raw [256, stride], npk(2048) | Σ_routed p² h hᵀ (down input) |
| `D0.f32` | same | Σ_routed h hᵀ |
| `Dc.f32` | same | Σ_ctx h_e h_eᵀ (all context rows pushed through expert e) |
| `C_ctx.npy` | [npk(6144)] f32 | Σ_ctx x xᵀ (layer) |
| `C_all.npy` | [npk(6144)] f32 | Σ over all fit rows x xᵀ (layer, "uniform all-token") |
| `gdiag.npy` | [256, 6, 2048] f64 | rows: Σ_r p² cg², Σ_r cg², Σ_r p² cu², Σ_r cu², Σ_ctx cg², Σ_ctx cu² |
| `scalars.npy` | [256, 4] f64 | n_routed, Σp, Σp², Σp⁴ |
| `meta.json` | | shards (T, n_ctx, n_dc, dc_scale, seed), T_fit, n_ctx, ess[256], n_routed[256], timings |

Notes on the table:
- **Packing:** each row is the row-major upper triangle (i ≤ j) of the symmetric matrix; `nq19.unpack(v, n)`
  inverts it.
- **Raw files:** have no header. Row e starts at byte `e * stride_bytes`, where stride_bytes = npk(n)·4 rounded
  up to 4096 (these are O_DIRECT writes). `meta.json["files"]["raw"]` records n, packed and stride_bytes.
- **Router weight:** p is the sigmoid-router weight including routed_scaling_factor 2.5, stored in fp32.
- **ESS:** (Σp²)² / Σp⁴.
- **Down input:** h = bf16(silu(bf16(x g_bf16)) · bf16(x u_bf16)), where g_bf16 and u_bf16 are the FP8-dequantized
  teacher cast to bf16, as in the pilot.
- **Output-weight terms:** cg = ux·σ(gx)·(1 + gx(1 − σ(gx))) and cu = silu(gx), with gx, ux = x·g and x·u.
  - On routed rows these use the fp32 teacher, computed as a bf16 hi+lo split with fp32 accumulation.
  - On ctx rows they come from the bf16 linear outputs, which gives a relative G error of ~1.3e-4.
- **Dc and the ctx gdiag rows:** these use the first n_dc = min(n_ctx_k, 131072) context rows of each shard,
  scaled by n_ctx_k / n_dc. This is an unbiased estimate; the scaling is recorded per shard.
- **Arithmetic:** Grams are bf16 tensor-core GEMMs with fp32 output per 2048-row sub-chunk, accumulated in fp32.
  The p²-weighted Grams are computed from a bf16 hi/lo split of p²x. There's no TF32 and no reduced-precision
  split-K.

`stats0/L{L}` → `../stats/L{L}.v1` is the frozen shard-0 version (1,048,576 fit tokens, 262,144 context rows).
It is never auto-deleted; remove it by hand once the shard-0 encode is finished. `SHARD0_READY` in the root
marks it complete for all 75 layers.

## Recipe reconstruction (thread 08 "unif0.75"; = `Capture.glm_H`)

```
cp²   = 0.25 · Σp² / n_ctx                         (context rows carry 1/4 of the routed p² mass)
W_x   = A2 + cp² C_ctx,      U_x = A0 + C_ctx
H_x   = 0.25 · W_x/mean(diag W_x) + 0.75 · U_x/mean(diag U_x)          (gate and up share it, count 1)
W_a   = D2 + cp² Dc,         U_a = D0 + Dc,      H_a (down) = same mix
G_gate = diag( dmix(g0 + cp² g4, g1 + g4) )^½,  G_up = diag( dmix(g2 + cp² g5, g3 + g5) )^½
dmix(A, B) = 0.25 A/mean(A) + 0.75 B/mean(B)
```
- The pilot/orbit gram format is W_x and W_a with training_rows = n_routed + n_ctx and
  mass = 1.25 · Σp². Harness `ExpertData.H(proj)` = grams / training_rows.
- Other recipes (alpha, context mass, uniform-only, `C_all`) can be built from the same sums without
  re-capturing: `glm_H(L, E, alpha=..., ctx_mass=...)`.

## Held-out evaluation rows (harness.evaluate format), from shard 0

- **`eval/val/layer_{L}.pt`:** windows 28656..28783 (65,536 tokens, the last 128 windows below the nq-tail).
  - These are disjoint from every fit window.
  - Same document-local forward as the fit rows.
  - Keys: x [65536, 6144] bf16, ids [65536, 8], p [65536, 8] f32, document_ids (window-local segment ordinal,
    globally numbered), token_positions, domains ("val:{category}" per document from windows.jsonl), layer,
    protocol.role = "evaluation only".
- **`eval/matched/layer_{L}.pt`:** orbit's 20 frozen matched-context documents (`glm53_matched_context_pilot_v1`),
  replayed with capture_glm.py arithmetic (batch-1 documents, no mask, index_add_).
  - Byte-identical to orbit's capture at L16.
  - Domains are "control:*" / "ood:*", so `evaluate` gives control/ood groups.
- `eval` → `shards/s00/eval`.

## Stage-1 activations

`shards/s{kk}/acts/L{L}/{x.bf16 [T_k, 6144], ids.u8 [T_k, 8], p.f32 [T_k, 8], done.json}` hold the normalized
MoE inputs of the fit rows.
- The `merged` marker means stage 2 has folded them in.
- x of shard 0 is kept: ~12.9 GB/layer, 0.97 TB total, usable for alternative recipes and re-derivation.
- For other shards, x is deleted after merging; ids and p are kept.

## Precision (L16 E36/E92/E165 vs orbit pilot / thread-12 cache)

- A2 vs orbit fp32 grams: 3–4e-6.
- H_gate/H_up vs thread 12: 4–5e-6.
- G: 1.3e-4.
- H_down vs pilot/thread 12: 1.5e-3. This is because the pilot's and thread 12's h came from small-M bf16
  F.linear with cuBLAS reduced-precision split-K, so 40–55% of h elements are off by one bf16 ulp. Using thread
  12's own h arithmetic reproduces their cached H_down bitwise. This capture's h is 8.7e-5 from exactly-rounded h.
- Harness evaluate on the matched set (L16 E36), this capture vs thread 12's results (routed rel-L2 %):
  nq/L2 34.49 vs 34.42, nq/L4 9.34 vs 9.33, EXL3-4 8.91 vs 8.90, NVFP4 12.29 vs 12.29.

## Group corpora and `plan.json` (new-corpus root)

- **Groups:** `{corpus}/{c512,c2048}/tokens.npy, segments.npy [W, C] int32, split.json {fit, val}, windows.jsonl,
  bnd_think_d.npy / bnd_end_d.npy [W, C] int8, manifest.json`. The corpus sha256s used are in `ROOT/corpus_sha256.txt`.
  - Attention is segment-local within a window, and positions restart per segment.
  - A segment id is the doc_index. Non-contiguous pieces of one document in a window share one id, so they form
    one attention segment.
  - C = 2048 is exactly the DSA index_topk, so the dense replay is exact.
- **`ROOT/plan.json`:** `{"chunk0": [ids], "shards": {"<id>": {corpus, fit_start, fit_windows, val_windows (-1 = whole
  val split), matched}}}`. Stage 1 runs via `driver.sh plan <gpu> <ids..>`. Shard ids set the context seed
  (20260925 + id). `"snapshots": {"stats1": [0..11]}` names further frozen sets. `stats0` (= `chunk0`) and every
  snapshot `ROOT/<name>/L{L}` are linked when the merged shard set of a layer equals the set, and those versions
  are never auto-deleted. Stage 2 completes each snapshot's set (in order) before merging any other shard.
  Merge dedup is by shard id (`protocol.json["shard_id"]`); fit_start alone collides across groups.
- **New-corpus chunk 0:** c512 fit windows [0,1280) (655,360 tokens) plus c2048 fit windows [0,192) (393,216
  tokens), 1,048,576 fit tokens in total.
  - Shards 0-3 are c512 [0,153), [153,529), [529,905), [905,1280).
  - Shards 4-5 are c2048 [0,75), [75,192).
  - Fit windows are pre-shuffled by thread 21, so these prefixes are random samples.
- **Traces (`c2048_traces`, thread 21):** shards 6-11 = fit windows [0,117), [117,246), [246,375), [375,504),
  [504,633), [633,761) (1,558,528 tokens; 512 think / 578 end boundaries). s06 writes the traces val split
  (10 × 2048), merged separately into `ROOT/eval/val_traces/layer_L.pt` (20,480 rows, `VAL_TRACES_READY`), so
  `eval/val` is unchanged. Traces are summed with plain weight 1 (lead, 2026-09-28).
- **Full rest (production):** shards 12-19 = c512 fit from 1280 in 2048-window pieces (19 = 1295 windows),
  20-24 = c2048 fit from 192 in 512-window pieces (24 = 273). 12,756,480 tokens; with shards 0-11 the final
  `stats` = 15,363,584 fit tokens (c512 8,658,432 + c2048 5,146,624 + traces 1,558,528).
- **Val:** s00 writes the whole c512 val split (223 × 512) and s04 the whole c2048 val split (42 × 2048).
  `merge_eval.py --root R --shards 0,4` concatenates them into `ROOT/eval/val/layer_L.pt` (200,192 rows,
  document_ids renumbered). `eval/matched` comes from s00.

## Boundary rows (lead's spec 2026-09-28; `bnd19.py`)

- **Boundaries:**
  - `think` = a `</think>` whose think block is non-empty (empty `<think></think>` stubs are skipped).
  - `end` = the first token of the `<|im_end|>` text, or `<|endoftext|>`.
- **Rows:** positions t with 1 ≤ b − t ≤ 32 before a boundary b, in the same segment.
- **Buckets:** `d1`, `d2_4`, `d5_16`, `d17_32`, kept separately per kind.
- **Exclusivity:** a row near both kinds counts only toward the nearer one; ties go to `end`.
- **Calibration weight: 1 everywhere (lead decision 2026-09-28).** Boundary upweighting is NOT used in the encode
  H: thread 12's A/B showed 50x costs +3% on all tokens. `glm_H` / `salience` default to `bnd_w=None` /
  `weights=None` (= weight 1); encode consumers must not pass them. Boundary weighting is used only for
  `fixed_set.json` (REAP 50/20/5/2, below). The boundary rows stay captured for analysis.

`bnd_rows/s{kk}/L{L}/` holds the raw rows of shard k (the per-bucket grams are rebuilt on the GPU; materialised
bucket grams would be about 360 GB/layer):

| file | content |
|---|---|
| `x.bf16` | [Nb, 6144] bf16, the same normalized MoE-input rows as the stats |
| `rows.npz` | row (shard-local fit row), kind (1 think, 2 end), d (1..32), bucket (0..3), is_ctx, ids [Nb, 8] u8, p [Nb, 8] f32 |
| `sal.npy` | this shard's salience sums |
| `meta.json` | per-(kind/bucket) counts |

`meta.json["shards"][i]["bnd_rows"]` of a stats version points at them.

- **`components_bnd(L, E, device, groups=None, ctx=True)`:** returns `{(kind, bucket): {A2, A0, D2, D0, C_ctx, Dc, g,
  n_routed, sum_p2, sum_p4, ess, n_ctx_rows}}`, unpacked fp32 ([6144²] / [2048²]), with g [6, 2048] f64 in the
  gdiag row layout.
  - It uses capture_stats arithmetic (accurate fp32 routed gx/ux; bf16 on ctx rows).
  - The ctx sums use every boundary context row, with no n_dc subsampling.
  - Checked at pilot L16: A0 7e-8, A2 2.4e-6, C_ctx 5e-7 against a direct computation.
- **`glm_H(bnd_w=w)`:** computes W' = W + Σ_g (w_g − 1) W_g, and likewise U', D2/D0/Dc and g, then applies the
  unchanged recipe. cp² stays at its unweighted value. `bnd_w=1` is bitwise the base H; 50 changes H by about 50%
  (gate/up) and 113% (down) at L16 E36.

## Salience / usage: `sal.npy` in every stats version, [256, 9, 6] f64

- **Categories** (`bnd19.SAL_CATS`): `all`, `think/{d1,d2_4,d5_16,d17_32}`, `end/{...}`.
- **Columns:** n, Σp, Σp², Σp⁴, Σp‖y‖, Σ‖y‖, where y = bf16 down(h) is the expert output before p.
- **REAP:** Σp‖y‖ / n.
- `salience(L, weights=50)` returns the boundary-weighted totals.
- Check: n for `all` equals `meta.json["n_routed"]`.

## Val boundary flags

`eval/val/layer_L.pt` has `bnd_think` / `bnd_end` int8 [rows] tensors: the distance (1..32) to the next boundary of
that kind in the same segment, 0 = none. They are exclusive under the same rule as the fit rows.

## Flashblade backup / restore (restart insurance: /tmp is wiped on container restart)

| root | S3 prefix | content |
|---|---|---|
| `19-capture-glmfmt` | `s3://annalise-shared-prod/jarrel/nestquant/19-capture-glmfmt/` | stats0 + stats1 small-only; final stats full grams (≈ 44.5 GB/layer, ≈ 3.35 TB) |
| `19-capture` | `s3://annalise-shared-prod/jarrel/nestquant/19-capture/` | `--small-only` (no A0/A2/D0/D2/Dc grams), 1.83 GB/layer, 137 GB |

- **Sets:** `--set stats0` (default; markers `done/`, `latest.json`), `--set stats1 --small-only` (`done_stats1/`,
  `latest_stats1.json`; the traces delta is small-only per the lead) and `--set full` (all 25 plan shards merged;
  `done_full/`, `latest_full.json`). `fb_restore.py --set <same>` restores one; restore stats0 first.
  `fixed_set.json`, `MANIFEST.json` and `eval/val_traces` are global files.
- **What `fb_backup19.py` uploads:** per final layer (the set's version plus `eval/VAL_READY`), the stats version
  dir, the boundary rows of its shards, `eval/val` and `eval/matched`.
  - Every object carries `sha256` and `size` metadata and is HEAD-verified.
  - After that it writes `done/L{L}.json` (file list) and `latest.json`.
  - Global small files (plan, corpus shas, shard protocol/progress, markers, boundary flags) go to `global/`.
  - It does not upload raw x or stage-1 hidden-state checkpoints; those are recomputable (chunk 0: ≈ 30 min
    stage 1 on 6 GPUs).
- **Budget (user rule): flashblade under 5 TB at all times.** `fb_backup19.py` lists `--budget-scope`
  (default all of `s3://annalise-shared-prod/jarrel/`) before each pass and aborts if total + still-to-upload
  bytes ≥ `--budget-tb` (4.8). The final set is uploaded with `--bnd-max-shard 11`: boundary rows only of
  chunk 0 + traces; those of shards 12-24 (≈ 14 GB/layer) are not backed up (not used by the weight-1 encode;
  recomputable by stage 1). `components_bnd` on a restored root therefore covers shards 0-11 only. The stats0
  gram objects are deleted from S3 once the final full set is verified (the local /tmp copy stays).
- **Environment:** `AWS_PROFILE=flashblade`, `AWS_REQUEST_CHECKSUM_CALCULATION=when_required` and
  `AWS_RESPONSE_CHECKSUM_VALIDATION=when_required` (the scripts set these themselves). The endpoint is
  https://fb.harrisonai.io.
- **Restore:**
  `./run.sh fb_restore.py --prefix s3://annalise-shared-prod/jarrel/nestquant/19-capture-glmfmt --root /tmp/nestquant/19-capture-glmfmt`
  - It sha256-verifies every file and rebuilds the `stats/L{L}` → `L{L}.vN` and `stats0/L{L}` links.
  - It then works with `Capture(root=..., stats="stats0")`.
  - Speed is about 1 GB/s, so a full restore takes about 1 h.

## Fixed set: `ROOT/fixed_set.json` (`fixed_set19.py`, lead's final spec 2026-09-28)

- **Score:** token-weighted REAP, S_e = Σp‖y‖[all] + Σ_c (w_c − 1) Σp‖y‖[c] over the 8 boundary categories
  (sal column 4), w = d1 50, d2_4 20, d5_16 5, d17_32 2 for both think and end, 1 elsewhere. Un-normalized, so
  routing frequency counts.
- **Set:** top 26 per layer by S_e (ties → lower id), computed on `stats1` (chunk 0 + traces).
- **Keys:** `fixed_set{L: [26 ids]}`, `S_e`, `reap_sum` (w = 1), `reap_mean` (classic REAP), `n_routed`,
  `coverage{L: {all, think, end, think_d1, end_d1}}` (share of routes landing in the set), `changed_vs_unweighted`,
  `stats_version`, `weights`, `definition`. The sha256 and weights are recorded in `ROOT/MANIFEST.json["fixed_set"]`.
- Copy: `threads/22-boundary-experts/fixed_set.json`.
