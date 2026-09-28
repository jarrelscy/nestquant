# Thread 23: NestQuant expert encoder throughput (bit-identical to thread 12)

**Result:** `nq_layer_batch.py` (group 4) encodes the production format ("p4126 single-pass inner0" plus the T12
f128e41 low-rank plane) **bit-identically** to T12's `nq_layer.py`.
- Speed: **15.1 s/expert with 1 process per GPU and 12.8 s/expert aggregate with 3 processes**, against a 45-54 s
  median for the reference on the same contended GPUs, so a **3.5-4.2x** speedup.
- The ≤8 s/expert (5x) target is not reached on today's shared GPUs.
- Encoder code: commit 8c74fd2, used by the T25 campaign ("t23b", t23_id c149a800ed4e17c9). The gate passed against
  pinned T12 f128e41 (nq_encode sha ff150c25).

## Steps (each gated byte for byte against the T12 reference)
1. **Layer-batched encoder (`nq_encode_batch.py`).**
   - Calls T12's own functions: `NE.prep`, `NE.encode_projection` and the lr/ocol helpers. `encode_rotated` is
     batched: G experts × (gate, up) and then × down share one batched LDLQ plus Viterbi pass.
   - Also: memoised g-scale/LDL work, a deduplicated candidate search, and batched segment costs with an exact loop
     fallback wherever batching could change float order.
   - Result: 45 → 31.7 s/expert.
2. **Exact K2 mul1 Viterbi kernel (`k2vit.py`, `csrc/nq_k2vit.cu`).**
   - Bit for bit equal to exllamav3 `quantize_tiles` K=2 mul1: the same fp16 hsub2/hmul2/hfma2 order, strict-less
     selection from k=0, the argmin rank tie order, and two-pass tail-biting.
   - Uses 2-bit history (1 MB per block instead of 8 MB), persistent blocks, a register decode cache, and an hmin2
     select with a per-tile NaN fallback to the exact sequential select.
   - Kernel speed: 8.7 vs 20.3 µs/ring. End to end: 31.7 → 17.9 s/expert.
3. **Mirrored T12 f128e41's low-rank eigen plane** (lr_detect/lr_H in prep, lr_apply after encode, meta fields),
   plus `--stats-mm/--mm-w/--lr-tau/--lr-rmax/--no-lr`.
   - The layer driver uses T12's `open_stats`/`expert_HG`, and finalize is T12's own `nq_layer.main()`, so the tp
     shard format (lrV/lrU2/lrU4) and the manifest are T12's code.
   - The lr fit costs ~2.3 s/expert (eigh 2.0 s). On L30 E169 (the rank-2 case) the reference took 47.8 s; the 348 s
     reported earlier was a CUDA-extension build wait.

## Bit-identity gate: ALL PASS (2026-09-29 ~02:25 Melbourne; gate.sh, resume_test.sh)
| test | result |
|---|---|
| (1) 27 experts: eigen cases L30 E169, L3 E2, L3 E60, L43 E180, L76 E108, L56 E96; T22 fixed-set 3:8, 20:9, 43:13, 56:42, 66:20, 77:12; 15 ordinary L3-L77 | PASS 27/27 (tensors + meta) |
| (2) group composition: groups 1, 2 (shuffled), 3 and 4 (reversed), different mates, GPUs 0/2/6; refs split over GPUs 0/6 | PASS, all arms |
| (3) layer path, L3 E0:16: nq_layer_batch --group 4 vs T12 nq_layer.py | PASS: tp0-7 sha256-identical, manifest identical apart from "time" |
| (4) resume: SIGKILL mid second group (E0-E3 on disk), then resume | PASS: tp0-7 + manifest identical |
| (5) --stats-mm self-blend, L30 E166:170 | PASS: tp0-7 + manifest identical |
| lr coverage | lr on throughout. Ranks: gate/up 1-2; down 0-3, with down rank 0 in 12/27 |

- Also passed earlier: 10/10 experts on the pre-lr code (347b6d03), and two concurrent processes (8/8).
- Expert E*.pt files differ at the byte level even between two reference runs, because of the meta.info.*.time
  fields. Compare tensors and meta with time skipped (common.compare, T25 nq25_finalize.py), or compare the tp files.
- Group 8 (and 6) exceeds the 12 GB per-process cap. It OOMs inside the process
  (`set_per_process_memory_fraction(12/80)`), so production is group 4.

## Throughput (GPU 4, co-tenant: T19's two 11 GB capture jobs, GPU ~95% busy)
| variant | s/expert | GPU memory / proc | notes |
|---|---|---|---|
| T12 reference nq_layer.py | 45 median (10 exp, pre-lr); 54 median (27 exp, lr) | 5.0 GB | one expert at a time |
| batched, no Viterbi kernel (step 2), g4 | 31.7 | 10.3 GB | pre-lr |
| + exact K2 kernel (step 3a), g4 | 17.9 | 9.95 GB | pre-lr |
| lr code (f128e41), 1 proc, g4 | **15.1** | 9.95 GB | 12 experts, wall incl. startup |
| 2 procs, g4 | 13.5 aggregate | 9.95 GB each | |
| **3 procs, g4** | **12.8 aggregate** | 9.95 GB each (~31 GB) | best |
| 4 procs, g4 | 13.0 aggregate | 9.95 GB each | GPU saturated |
| group 6 | OOM | >12 GB | not allowed |

- **Projection (8×A100, 19,456 experts):** 3 procs/GPU at 282 experts/h/GPU gives 2256/h, so **~8.6 h** at today's
  contention. At 1 proc/GPU it is ~10.2 h.
- No idle GPU was available to measure.

## Step 4 (candidate, not yet in production): exact frac K2.3125 kernel (commit 20f75a5)
This covers `csrc/nq_frac23.cu` and `frac23.py`.

**What it replaces:** the down-residual pattern-rate Viterbi, which is exllamav3 `quantize_tiles_frac_kernel<2, 0x2492>`
reached through T12 `nq_patvit.patq_cuda`. The new kernel matches it bit for bit:
- the same fp16 hsub2/hmul2/hfma2 order and pre_state inf mask;
- a strict-less select from k=0 in the reference candidate order;
- the same argmin rank tie order over edges_last;
- the same two-pass tail-biting.

**What differs (storage and scheduling only):**
- Costs are in shared memory.
- History keeps the 2/3-bit winning branch instead of a uint16 edge: 1 MB per tile instead of 8 MB.
- The thread-to-window mapping doesn't depend on the step widths, so each thread's 128 windows are the same in the
  2-bit and 3-bit steps. That lets the decoded codebook values be cached in registers.
- An hmin2 select is used, with a per-tile NaN fallback.

**Unit test (`test_frac.py`):** 0 mismatching rings, in both indices and values, against the reference kernel.
- 12 input classes: normal at 3 scales, zeros, const, quantized ties, huge, ±inf, NaN, all-NaN, sparse.
- K 2.3125 and 2.25.
- Speed: 26.7 vs 133 µs/ring on a contended GPU, i.e. **5x**.

**Encoder gate (`gate1.sh`):** the candidate copy `nq_encode_batch_f23.py` is loaded via the `f23_run.py` alias shim,
so no pinned file is edited. Compared against the f128e41 reference arms on 1 GPU: ALL PASS.
| arm | result |
|---|---|
| 27 experts, group 4 | ALL BIT-IDENTICAL |
| 27 experts, group 3 reversed | ALL BIT-IDENTICAL |
| 8 experts, group 1 | ALL BIT-IDENTICAL |
| L3 E0:16 layer path | tp0-7 + manifest identical |
| --stats-mm L30 E166:170 | tp0-7 + manifest identical |

- Every arm logs its frac23 call count, as proof that the new kernel actually ran.
- The A/B off and on arms are also identical to each other.

**Speed in the encoder:**
- Measured A/B on a GPU shared with 3 campaign workers: off 39.1 / 37.6 vs on 35.4 / 38.7 s/expert. That is
  within the noise.
- By construction the saving is device time, ~60k frac rings per expert × ~0.1 ms, which is roughly 2 s of GPU
  work per expert when uncontended.
- Estimate (not measured): ~10% more campaign throughput at GPU saturation.
- Peak memory: 10.06 vs 9.95 GB at group 4.

**Swap-in:** `/tmp/nestquant/23-encode-throughput/frac23_swapin.patch`, 2 hunks on nq_encode_batch.py
(`frac23=True` + the patched() hook). `NQ23_FRAC23=0` switches it off. The lead decides on adoption.

## Remaining time
- **Device time, contended:** ~14.8 s/expert in total.
  - K2 Viterbi (k2vit): ~5.4 s.
  - Frac K2.3125 (exllamav3 generic kernel): ~2.1 s.
  - LDLQ/candidate matmuls: ~3.3 s.
  - lu: 0.4 s.
  - lr eigh: ~0.5 s.
- **At 1 process** the GPU is idle ~50% of the time (launch/sync-bound: 784 syncs and 12.7k pageable H2D copies per
  4 experts). Multiple processes per GPU recover that, and the GPU saturates at 3.
- **Bit-identical options remaining:**
  - An exact frac K2.3125 kernel: done as a candidate (step 4), awaiting the lead's swap-in decision.
  - k2vit tuning and fewer syncs (skipped by the lead).
- **In-process overlap** (next-group prep during the Viterbi, streams for gate/up vs down) is unsafe as built: the
  k2vit history buffer and the PV scratch are shared, and Python threads don't beat the GIL. Multiple processes give
  the same overlap safely.
- **Batched eigh** is not guaranteed bit-identical, so it was not done.

## Files (threads/23-encode-throughput/)
- `nq_encode_batch.py`: batched encoder.
- `nq_layer_batch.py`: drop-in for nq_layer.py (+ --group).
- `k2vit.py`, `csrc/nq_k2vit.cu`: exact K2 kernel.
- `frac23.py`, `csrc/nq_frac23.cu`: exact frac K2.3125 kernel (candidate).
- `nq_encode_batch_f23.py`, `f23_run.py`: candidate encoder copy and the alias shim.
- `common.py`: production config and the NQ23_T12 pin.
- Tests:
  - `check_bitid.py`: ref/batch/cmp/pair.
  - `cmp_layer.py`, `ref_layer.py`.
  - `gate.sh`: widened gate.
  - `resume_test.sh`.
  - `gate1.sh`: single-GPU re-gate against existing reference arms.
  - `test_frac.py`: kernel exactness and speed.
  - `tput.sh`: procs × group scan.
- Scratch in /tmp/nestquant/23-encode-throughput/:
  - `t12pin_f128/`: pinned T12.
  - `capsnap/`: frozen stats0 root.
  - `logs/f128_/summary.txt`: gate.
  - `logs/tput_scan.log`: throughput scan.
