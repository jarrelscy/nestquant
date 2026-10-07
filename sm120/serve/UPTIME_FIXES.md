# Serving uptime correctness fixes

Baseline: `434e5289150b78ad5ed6010295937deeb7a41487`. Changes are review-branch code, not a production deployment. A CPU-reproduced failure mechanism does not establish that it caused a benchmark failure.

| Area | Correction |
|---|---|
| First-N decode waits | Register new-request prompt lengths during prefill; count generated tokens rather than total context. |
| TAP liveness | Preserve per-rank throughput estimates when that rank has no queued demand. Keep backpressure for busy ranks; reject invalid initial rates. No unconditional probing that bypasses follower queues. |
| Follower log retention | Retain generations until every configured follower acknowledges consuming them. Missing committed history raises rather than silently stalling or skipping expert decisions. Serve configures all TP followers explicitly. |
| Borrow/ring recovery | Refuse unsafe ownership changes after pause timeout; propagate reclaim/close failures; runtime ring failures require worker restart instead of permanently lowering prefill precision. |
| Background failures | Worker execute hooks on all ranks check sticky streaming/async-wait errors, including graph-replay steps. Failures arriving after an execute returns are caught at a later boundary. |
| Graph capture | A busy executor prevents capture; failed capture begin releases the pause counter. |
| Session restore | Reject unsupported joint/GPU predictor snapshots before mutation. Supported CPU predictor arrays restore in place. |
| Lookahead buffers | A pinned buffer returns to the free pool only after completed DMA and detached host snapshot. Exhaustion drops optional lookahead work rather than overwriting in-flight data. |
| Routing counters | Host differences use unsigned modulo-2^32 arithmetic across signed int32 counter wrap. Assumes fewer than 2^32 hits per expert between polls. |
| Loader/routing | Validate all resident expert metadata and complete expert coverage before publishing rows, including future streamed residual formats. All-inactive batches clear reused output buffers in both kernel dispatches. |
| FP8 output projection | Fused decode stages BF16 inputs into FP32 and accumulates in FP32, writing BF16 directly. Diagnostic fallback also avoids FP16 intermediates. This changes rounding and may cost throughput; performance is unmeasured. |
| Long-running bookkeeping | Bound completed-upgrade latency history to4096 entries; publish native I/O statistics from the owning worker under a mutex. |

## Validation

CPU tests (set `CUDA_VISIBLE_DEVICES=` and `OPENBLAS_NUM_THREADS=1`):

```
python -m unittest discover -s tests -p 'test_*uptime*.py' -v
python -m unittest discover -s tests -p test_serving_recovery.py -v
python tests/test_prefill_borrow_cpu.py
python tests/test_slotborrow_cpu.py
git diff --check
```

22 new unit tests pass:5 serving integration,4 recovery/session,2 numerical/loader,11 streaming/log/latency. Existing prefill-borrow simulation passes12 randomized seeds; slot-borrow suite passes13 cases. Tests extract actual serving function ASTs where importing the complete module would require CUDA/vLLM. Numerical tests exercise the actual CPU-compatible fallback, not the fused CUDA implementation.

**Untested:** native CUDA compilation/execution, SM120 output parity, TP4 fault recovery, GPU racecheck, performance, and BF16-teacher KL/needle quality gates. No host `nvcc` was available during this pass. Do not treat CPU mocks or source review as those validations.

Acknowledged logs intentionally retain more history when a follower is delayed or never starts. Unknown membership retains all history. Monitor shared-memory usage; do not delete records to free space behind an unacknowledged follower. Existing sessions cannot be hot-migrated to the new protocol; start all workers together when deploying.

Session restore for GPU/joint predictors is explicitly unsupported until a complete snapshot protocol exists. Recovery favors explicit failure over silent reduced precision; distributed shutdown behavior must be exercised before deployment. The FP8 projection fix is stateless; the old half-overflow counterexample was not observed in live model activations.
