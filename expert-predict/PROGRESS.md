# Expert-prediction research for NestQuant SSD streaming (GLM-5.3 ARVQ-v2)

Offline CPU only. Workspace /data/Jarrel/expert-predict. Live scheduler files in /data/Jarrel/nestquant are read-only here.

## Log
- 2026-09-29 01:35 start. Index 77 in the logs = main-model layer L77 (config: 78 hidden layers, 3 dense, 1 MTP layer that is NOT logged). MoE layers = L3..L77 (75).
- build_data.py -> data/<task>.npz: per-task deduped streams (budget_pertask_norecompute convention: keep last row per (req,pos), drop re-prefilled 8-grams), with a decode flag (row from a step with <=16 rows for its request).
  - Replay trials (cmpl-replay-*) are prefill-only (0 decode rows). Live decode exists for 6 tasks: formal-crypto 0.90M, embedding-drift-monitor 0.18M, fin-saccr-rwa 2.33M, freight-dispatch-shift 0.76M, pretrain-shard-corruption 1.96M, sound-change-cascade 0.49M decode tokens (kept after dedupe).
  - 7 prefill-only replay tasks: ks-solver-cpp, lake-temp-glm, layout-config-recreation2, photonic-waveguide-routing, react-lead-form, satb-audio-transcription, takens-embedding-lean. Probes: 21 domains x ~3 chunks of 8K prefill tokens.
- 02:10 Harness: src/sim.c (token-exact C sim of scheduler.py semantics: refresh every R, top-51 non-fixed wanted, immediate downs (lazy=0) or evict-on-demand (lazy=1), per-token byte budget 6 GB/s at 95 tok/s with 64-token burst, 9.97 MB/upgrade all ranks, upgrade serves from t+lead). Fixed set = thread-22 fixed_set.json (26/layer = 1950), start = floating_default. Decode stream only (the tokens the live scheduler serves at 95 tok/s); scheduler state carried across a task's requests (c1).
- Baseline check: EMA-512/R64/lead13 on decode = 62.3-67.3% per task (sched_sim.py got 63.4% on a 716K-token subset: consistent).
- Oracles (results/baseline_*.log): perfect next-64 knowledge, lead 13, NO I/O cap = 75.9-78.1% but needs ~24-26 GB/s. Same oracle under the 6 GB/s cap = 50-55% (immediate downs) / 67-71% (lazy); future-256 oracle capped+lazy = 71.4-74.1% (best realistic ceiling found). Lead-0 uncapped block oracle = 79.8-81.6% (prior 86.2% was all-token incl. prefill with a usage-ranked 30% set).
  => The I/O cap, not prediction, bounds the gap: the realistic ceiling is ~72-74%, not 86%.
- Linear per-layer multi-horizon EMA + per-state (think/answer) memory + static prior (fit_linear.py, train fin+pretrain, val formal+embedding): +0.3-0.8 pt only with lazy eviction; saturates the 6 GB/s cap and loses without lazy (churn). EMA512 uses only ~2.9 GB/s.
- Coordinator request (REAP): sal.npy files NOT found (not in /data/Jarrel, /rawdata/Jarrel, HF repo jarrelscy/GLM-5.3-NestQuant-2-4bit; /tmp copy gone). Using instead: fixed_set.json reap_mean (= sal col4/col0, mean p||y|| per routed row, 15.4M-token calib) and nq-glm53-hf per-expert proxy_rot (2-bit vs 4-bit relative error, real encode) -> GAIN = reap_mean * (err2-err4). Added salience-weighted (REAP) and gain-weighted coverage to the sim.
- 02:40 EMA sweep on val (sweep_ema.py, formal+embedding token-weighted, lead 13, 6 GB/s): baseline EMA512/R64 = 63.42% slot (sal 69.72%) at 2.89 GB/s. Best within cap = EMA256/R16/hysteresis 0.1 = 64.18% (sal 70.7%) at 4.78 GB/s (+0.76 pt). EMA128/R16 = 64.22% but needs 14.8 GB/s uncapped.
- REAP-weighted scores (EMA x reap_mean, EMA x GAIN): sal share ~72.9-73.0% (+3 pt) but slot share ~61.5-62% (-1.7 pt). Circular: the salience metric uses the same static weight as the score. gain_share tracks sal_share (GAINREL varies only 0.035-0.116, corr with REAP 0.21).
- Low-rank NN (train_nn.py, softplus(diag + W2 relu(W1 sqrt x))), 4 EMA horizons + state, Poisson on next-64: held-out loss slightly better than EMA512 (formal 7.739 vs 7.757) but sim share worse (best 63.18% vs 63.42%).
- Uncapped ranking recall (recall.py: fixed + top-51 of score vs next-64 window, no I/O): formal EMA128 63.9 / EMA512 63.5 / NN 63.9 / oracle 79.9; embedding 64.4 / 63.9 / 64.2 / 80.2. In answer state NN +3.5-4 pt over EMA512 (58.0 vs 54.4, 55.3 vs 51.2). => ranking headroom from features is ~0.5 pt overall; the oracle gap is unpredictable-from-history routing.
- Miss breakdown EMA512 (where_miss.py): answer tokens (after </think>, 10-14% of decode) served 50.7-53.4% vs think 64.5-65.8%; request starts on embedding 57-58% in first 64 tokens. Next: </think>-aware answer-state predictor.
- 03:30 Answer-segment memory (answer_pred.py): in answer state score = 0.5*global EMA + 0.5*own-clock answer EMA (half-life 2048 answer tokens, carried across requests); think state unchanged. Val: EMA256/R16/hm0.1 + ans0.5 = formal 64.24% / embedding 64.69% (answer tokens 57.1/54.5 vs 54.7/52.9 without) at ~4.7 GB/s. +0.2 pt over EMA256/R16/hm0.1 alone; a=0.8 and adding the train answer prior (b=0.2) are worse; think-state own-clock memory (c=0.3) churns to the cap and loses.
- Probe OOD (probe_ood.py, 21 domains, 6-41K prefill tokens each, streamed through the same sim): EMA256/R16/hm0.1 beats EMA512/R64 in every domain (+0.4 to +1.2 pt); EMA x REAP loses 1.5-2.5 pt slot share, gains ~2.5 pt salience share; capped future-256 oracle 65-74%.
- Host cost (hostcost.py, numpy 1 thread, [75,256]): per-step EMA update 3.7 us (7.6 us with the answer memory); refresh (argsort top-51) 81 us EMA, 98 us answer-state, 85 us REAP, 378 us linear 12-feature, 776 us NN low-rank.
- Test eval launched (test_eval.py): 4 test tasks, predictors fixed from val (no fitted weights -> no fold issue), R 16/32/64 x lead 0/13/32/64.
- 05:00 CAP CORRECTION (coordinator): the live budget is per rank (rec_bytes 2.56 MB, 6 GB/s per rank) = 24 GB/s aggregate; my earlier runs used 6 GB/s aggregate. The sim's cap argument is aggregate GB/s with 9.97 MB/upgrade, so 24 = the live configured cap. cap_curve.py sweeps 3/6/12/24/48/uncapped.
  - The EMA family is demand-limited: it never uses more than 3-15 GB/s, and share saturates by 12 GB/s (EMA512/R64 uses 2.6-2.9 GB/s at any cap). At 24 GB/s, the capped oracle F64+lazy = 75.3% and uncapped = 75.7% (embedding) vs EMA best 64.8%. => At the real cap, the gap to the oracle is a PREDICTION gap (~11 pt), not an I/O gap. The earlier "I/O cap bounds the gap" conclusion was an artefact of the 4x too-low cap.
- Lead/refresh sweep (test_eval.py, 4 test tasks, 6 GB/s aggregate; EMA runs are demand-limited so this holds at 24): lead 0 -> +0.8-1.0 pt, lead 32 -> -1.1 pt, lead 64 -> -2.7 pt vs lead 13; R16 beats R32 by 0.2-0.3 pt and R64 by 0.6-0.7 pt.
- GBDT (LightGBM 4.7.0 in /data/Jarrel/expert-predict/venv): rows = (16-token refresh, layer, expert) for EMA256 ranks 20-120 among non-fixed experts (ranks <20 kept). 23 features. Train fin+pretrain (every 32nd refresh, 63.5M rows), early-stop on formal+embedding (15.9M rows). Poisson on next-64 hits is best on uncapped recall.
  - Uncapped recall: embedding 66.6 vs EMA256 64.9; formal 66.2 vs 64.4; freight 68.9 vs 67.2.
  - Sim at 24 GB/s with hysteresis 0.5: formal 65.7, embedding 66.0, freight 68.4 vs best EMA 64.5 / 64.8 / 67.4 (about +1.0-1.4 pt), using 7-8 GB/s.
  - Feature gain importance: ema128 48%, own-clock state memory 22%, ema32 21%, tokens-since-hit 2%, hits16 2%; co-activation features 2% total; static prior, REAP, layer and rank each ~0.1-0.4%.
  - Probe OOD (19 domains): GBDT p64 hm0.5 @24 = 62.65% vs EMA128/R16/hm0.1 61.95% and EMA512/R64 61.02%; positive in all 19 domains (+0.3 to +1.0). At 6 GB/s it loses unless hysteresis is raised.
  - Host cost: 400 trees x 63 leaves = ~10 ms per refresh on 16 threads (over the 5 ms budget); the 120x31 compact model costs 23 ms on 1 thread / 6.4 ms on 4 threads and matches the full model on the probes. Ablations with 3-5 features are training.

## 2026-09-29 ~07:30 — held-out fin/pretrain (swapped fold), late-by-one, online module (DELIVERED)
Cap 24 aggregate, lead 13, share %. fin/pretrain = swapped-fold model (train freight+sound); freight/sound = shipped model (train fin+pretrain).
| predictor | fin | pretrain | freight | sound | mean(4) |
|---|---|---|---|---|---|
| ema512_R64 (current) | 62.28 | 64.38 | 65.85 | 67.33 | 64.96 |
| ema256_R16_hm0.1 | 63.35 | 65.14 | 66.91 | 67.91 | 65.83 |
| ema128_R16_hm0.1 | 63.77 | 65.32 | — | — | — |
| GBDT s5 p64 hm0.5 sync | 64.75 | 66.35 | 68.28 | 69.08 | 67.12 |
| GBDT s5 p64 hm0.5 late-by-1 refresh | 63.89 | 65.64 | — | 68.45 | — |
| GBDT full p64 hm0.5 | 64.86 | 66.37 | 68.41 | 69.22 | 67.22 |
GBDT GB/s 7.3-8.5 (hm0.5); 12-13 (hm0.25); 18-19 (hm0.1, and worse share). Late-by-1 costs 0.6-0.9 pt.
Pretrain EMA at 24: results/ema24_pretrain.jsonl (src/ema24.py).
Online module: /data/Jarrel/nestquant/streaming/gbdt_predictor.py + gbdt_p64_s5.txt; parity test streaming/test_gbdt_parity.py
(copy of src/parity_gbdt.py). Parity (sound): S bitwise equal over 2000 refreshes; sync sim 0.69084 == offline; next_refresh
== sync shifted one refresh (0.68447). Online host cost 7-10 ms/refresh on a loaded box (4 thr). Handed to NQ agent; not committed.
Skipped: p256 swapped fold (p64 chosen), fin/pretrain cap_curve restart (EMA demand-limited; 24 GB/s rows computed directly).
