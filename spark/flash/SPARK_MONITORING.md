# U2630 + FP4 serving: native Spark checklist

This is the published serving default, not a claim of native DGX Spark validation.
SM120 tests used a discrete 96 GiB GPU and host-mapped residuals across PCIe.
GB10 has unified memory. A TPS cap cannot simulate that hardware or guarantee an
identical quality score. Run the same benchmark locally to establish the result.

## Reproduce the configuration

Use the pinned serving revision recorded in the Hugging Face
`serving/default.json`. Default launcher settings:

- TP1/DCP1, one sequence, eager, prefix caching and KV connectors disabled.
- `spark_256K_U_2630`: 2,630 active floating slots, zero fixed, eight spare slots.
  Per-layer counts are in `u_distribution_2630.json`; retain them exactly.
  21.74% of the 12,096 expert-layer pairs are resident at 4 bit, averaging
  62.62 slots/layer. Residency is not activated-expert coverage.
- Published jT weights/policy: refresh8 committed tokens, horizon16, mix0.5,
  EMA half-life64, hysteresis4, causal routing alignment and rejected-token ledger.
- FP8 backbone attention/lm_head and MTP weights; FP4 E2M1 MLA cache with FP16
  scales per16 latent values. DSA indexer remains FP8; KDA state is unchanged.
- 262,144 maximum context, probabilistic MTP2, temperature0.7, top_p0.95,
  thinking-off template, async throughput-aware prefetch, no artificial TPS cap
  (`NQ_FLASH_TPS=0`). The SM120 benchmark used35; native uncapped behavior must
  be measured separately.
- The native launcher starts with an explicit2 GiB KV budget. vLLM must verify
  capacity at startup. This budget is a candidate for GB10, not measured fit.
  SM120 instead used profiled1.91 GiB KV and reported307,341-token capacity.

The CLI still says `--kv-cache-dtype fp8` to select the existing backend/indexer.
Confirm the startup message `EXPERIMENTAL Flash FP4 MLA`, not the CLI flag alone.
Confirm ready telemetry says `active_slots:2630`, `spare_slots:8`, `fixed:0`.
Requests can override sampling: send temperature0.7/top_p0.95 explicitly.
The launcher forces the tested thinking-off template. Do not pass a reasoning
budget or change the template during a comparison. Keep concurrency1.

## Watch continuously

```bash
python3 spark/flash/monitor_serving.py "$NQ_FLASH_DATA/routing.json" --interval 10
# Disk latency, queue depth, read bandwidth, saturation:
iostat -dx 5
# Whole-system shared RAM, swap activity and CPU pressure:
free -h
vmstat 5
cat /proc/pressure/memory
# Serving errors, cache capacity, requests and speculative statistics:
docker logs --tail 100 -f nq-spark-flash
```

Do not print environment variables or authorization headers. If querying an
API-key-protected endpoint, load the key privately; never put it in traces.

| Signal | Interpretation and action |
|---|---|
| `decode_committed.hit_rate`, `mean_hot_of_8` | Actual routed experts served hot. Read decode separately from prefill. The SM120 run was around54% /4.3 of8; this is an observation, not an acceptance threshold. |
| `hot_salience_coverage` vs `desired_salience_coverage` | We care about routing-weight² × normalized-input-norm² coverage. SM120 was around72–73%, with desired roughly0.7–1 percentage point higher. A sustained larger gap on matched prompts suggests upgrades arrive late; investigate loading before changing jT. |
| Per-layer coverage | Global totals can hide one starved layer. Compare `total.decode_committed.layers` on identical prompts and similar context lengths; U intentionally gives different budgets. |
| `desired_not_yet_admitted`, `pending`, `outstanding_reads`, `mailbox_pending`, `pending_demotions` | These are different stages, not interchangeable. Brief spikes are normal; a growing queue plus falling delivered coverage matters. Mailbox/demotion stalls with idle SSD suggest acknowledgement/CPU/GPU scheduling, not bandwidth alone. |
| `ssd_GBps`, `delivered_GBps`, `drive_read_ms`; iostat `r_await`, queue and bandwidth | Separate read completion from GPU-visible upgrades. Poor bandwidth with high latency can be small/random-read latency; sustained saturation can be bandwidth. Check the actual SSD/interface and thermal throttling. |
| Actual client output tok/s, TTFT, prefill tok/s; scheduler `estimated_committed_tps` | The scheduler number is an EWMA estimate, not an endpoint measurement. Record prompt/output counts, time to first token and completion time. Compare early and steady decode, short and long prompts. |
| Speculative drafts and accepted tokens per position | Use before/after Prometheus counter deltas around matched requests. For MTP2, emitted/step=1+(accepted_pos0+accepted_pos1)/drafts. High acceptance alone is not quality; repeated/runaway output can also accept well. |
| Whole-system MemAvailable, swap-in/out, memory PSI, process/cgroup RAM and CUDA usage | On Spark CPU and GPU share physical RAM: do NOT add GPU usage to host totals as if they were independent. Track load/prefill/decode peaks and OS headroom. File cache may be reclaimable; pinned/shmem allocations are not free capacity. Swap or sustained pressure can slow upgrades and change effective precision. |
| I/O errors, stale telemetry, NaNs, repetition, invisible output, malformed tools, missing EOS | Fail the validation if these appear. Inspect outputs and logs, not just benchmark aggregates. |

The monitor shows lifetime decode totals. Use deltas from before/after each test
(or current-request telemetry when available) for fair comparisons. Do not
compare a prose-heavy cumulative run against a code-heavy request. Preserve raw
telemetry timestamps and logs alongside benchmark transcripts.

## If Spark delivers fewer upgrades

First verify checkpoint/code revisions, exact U counts, predictor files, no I/O
errors, memory headroom and unthrottled SSD. Then check whether desired coverage
is healthy but delivered coverage lags. If so, try a lower explicit
`NQ_FLASH_TPS` in a separate measured run to give async loads time to land.
Do not silently increase pool sizes beyond measured RAM fit, switch to strict
waits, or retune predictor parameters. A lower cap might help coverage but does
not guarantee it; it also reduces latency-based benchmark scores. Report changes.
If desired and delivered coverage are both low, slowing the SSD path is not the
explanation; examine workload, per-layer allocation and predictor behavior.

## What is actually validated

SM120: independent FP4 packed-byte oracle, GPU attention parity (FP16/BF16),
padded/permuted/high-position pages, reused slots, real MTP2 serving,10K and32K
needle requests plus request reuse. Synthetic positions near262K were tested;
full262K model quality and memory peak were not. SM121 kernels compiled offline;
native GB10 execution, compute-sanitizer and broad FP8-versus-FP4 KLD are untested.
FP4 is lossy. The older research KLD0.0691 belongs to102/74 with FP8 KV and must
not be attached to U2630/FP4.

The provisional SM120 Spark Bench snapshot at70/80 scenarios (two repeats) had
quality84.43/100 and TrueScore86.49/100. It used thinking off, temperature0.7,
top_p0.95, MTP2,35 tok/s cap. This is neither a final score nor a native Spark
result. TrueScore includes latency. No monitoring threshold guarantees matching
quality: use identical grader revision, harness, tools, prompts, seeds/repeats,
sampling and timeout policy; complete all80 scenarios and compare uncertainty.
Keep predictor-reset semantics and request ordering identical. State and upgrade
timing can change which experts actually use their4-bit residuals.

## Throughput-aware prefetch and demotion safeguards

All async fixes from `e601770`, `580e577` and `98517d6` are included in the
published runtime; U and FP4 do not replace them. The launcher explicitly enables
`NQ_FLASH_PREFETCH=throughput` and leaves strict decode upgrade waits OFF.

- Read admission adapts to measured service capacity and committed-token speed;
  lookahead is0.1 seconds, with an8–64 read-admission range and16-token horizon.
- SSD read admission is counted separately from GPU mailbox acknowledgments.
  A backlog of acknowledgments must not masquerade as saturated SSD reads.
- At most64 total transitions and32 pending demotions across layers; at most
  one demotion per layer. This avoids draining a layer while replacements queue.
- Blocked layers are filtered before candidate-priority scoring, avoiding the
  repeated CPU scan that previously introduced a large scheduling gap.
- Slot memory is reused only after executor/GPU acknowledgment. Cancellation
  requests alone do not free a slot. Late upgrades use the resident low-bit base.
- jT has a private predictor stream; request reset and committed/rejected-token
  handling remain in place. This is asynchronous serving, not strict wait mode.

Verify `io.prefetch.mode=throughput`, `maximum_pending=64`, and
`maximum_demotions=32` in routing.json. Keep the complete pinned serving code:
copying just the allocation JSON or FP4 module into an older runtime omits fixes.
