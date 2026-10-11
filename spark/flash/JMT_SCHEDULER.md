# Experimental jMT admission and eviction

This branch supplies the previously local jMT serving adapter, bounded prefetcher,
spare-slot publication, trace instrumentation, and a cost-benefit scheduling
candidate. Production defaults remain jT; the new policy is opt-in. This is a
training/integration handoff, not a recommended production scheduler.

## Contract for the predictor trainer

The probability scheduler consumes two arrays of shape [16, 42, 288]:

- `activation[h-1,L-3,e]` is the marginal probability that expert e is selected
  at future committed token n+h, for h=1..16 and MoE layers L=3..44.
- `first_use[h-1,L-3,e]` is the probability of at least one selection in n+1..n+h.
  It is a cumulative union probability, not a per-step hazard or normalized share.

The latest predictor input uses token n and routing from n-1. Forecast origin,
training labels and serving positions must agree exactly. Rejected speculative
rows must not advance predictor state. Requests reset incremental state. Tests
must cover full-sequence versus incremental inference and actual commit alignment.

Use calibrated independent activation probabilities, not an expert softmax:
there are eight selected experts per layer per token. The sum of well-calibrated
marginals should therefore be approximately eight, not one. Expected hit count
is the sum of marginal probabilities; it needs no independence assumption.

The cumulative head must satisfy F1=p1, monotonicity, and union bounds
max(F[h-1],p[h]) <= F[h] <= min(1,F[h-1]+p[h]). Do not reconstruct it using
1-product(1-p): expert uses across tokens are correlated. The adapter on this
branch matches the original activation-firstuse release (44bfc132), not the later
calibration-v2 projection. Do not silently substitute that newer release.

For the unweighted hit-count objective all experts have benefit one. Keep any
upgrade-quality weights separate from probability predictions. Optional published
Flash 1.5-to-4 benefits are conditional local expert-output teacher-error reduction,
already routing-weighted; they are not measured language-model loss reductions.
Never multiply them by routing salience a second time.

Useful trainer reports: probability calibration and ranking by horizon, and
same-trace realized hits under finite delivery capacity. Include uncertain cases,
repeated uses, long inter-use gaps, and non-reasoning contexts. Evaluation must
separate forecast quality from admission, delivery, and eviction losses.

## Decision rules and implementation

`jmt_queuecost.QueueCostPlanner` adapts the common-currency cost-benefit idea from
[CMU TIP](https://www.pdl.cmu.edu/TIP/cost_benefit.shtml). It is not a reproduction
of TIP-2, nor an optimal stochastic cache policy.

1. Estimate arrival lead from measured end-to-end delivery latency and queued
   records divided by observed delivery rate. Keep all sixteen forecast heads.
2. Score an incoming expert only for uses after estimated arrival.
3. With a spare slot, compare against the victim's uses after the planned swap:
   the resident stays usable while its replacement loads.
4. Without a spare, immediate eviction costs include the resident's earlier
   uses as well. The existing additional early-reuse risk coefficient remains
   on this immediate-demotion path.
5. Once a payload is ready, compare both experts using the full current forecast;
   the newcomer no longer pays another arrival delay. Revalidate the victim.
6. Issue only positive-net-value replacements, ordered by net gain within the
   existing bounded read/transition limits. GPU acknowledgements own reclamation.

All forecast uses lost after eviction are charged through the end of the horizon.
This deliberately does not assume a free future reload. A later reload must
compete for read capacity again. It is a conservative one-exchange approximation,
not a joint optimization of an explicit multi-swap/reload schedule. Uncertainty
beyond sixteen tokens is not modeled. The arrival estimator uses observed p90 and
an observed peak rate; it is not an M/M/1 model or a guarantee of individual
completion times. Queueing-capacity optimization remains experimental work.

Near-use urgency discounting and a paired-only replacement policy were separately
screened and regressed; neither is included in this branch.

Main files:

- `jmt_activation.py`: calibrated activation/first-use adapter and benefit checks.
- `jmt_probability_planner.py`: previous full-horizon reference policy.
- `jmt_queuecost.py`: prospective paired-victim and ready-publication costs.
- `nq_flash_prefetch.py`: net replacement admission, bandwidth limits, spare reads.
- `preload_executor.py`: paired publication and acknowledgement-based reclamation.
- `jmt_joint.py`: committed-token alignment, measured delay and eviction regret.
- `hit_trace.py`, `delivery_timing.py`: opt-in private diagnostic capture.

## Enabling the candidate

Use the existing Flash launcher/model setup, with these variables forwarded into
its serving container (setting host variables alone is insufficient if a launcher
does not forward them):

```
NQ_FLASH_JMT=/path/to/activation-firstuse-release
NQ_FLASH_JT_PARAMS={"mix":1.0,"G":1,"hm":4.0,"half_life":64.0}
NQ_FLASH_JMT_JOINT=1
NQ_FLASH_JMT_OBJECTIVE=hits
NQ_FLASH_QUEUE_COST=1
NQ_FLASH_JOINT_FULL_HORIZON=1
NQ_FLASH_JOINT_RELOAD_PENALTY=0.5
NQ_FLASH_PREFETCH=throughput
NQ_FLASH_SPARE_PRELOAD=1
NQ_FLASH_EARLY_PUBLISH=1
```

Current matched experiment: 2,630 floating slots, zero fixed, eight spares,
U050 per-layer budgets, host-mapped upgrade slots on GPU0. Actual serving uses
MTP2 probabilistic, 35 TPS cap, temperature 0.7, top_p 0.95, thinking off,
262144 context, max_num_seqs=1, tracing enabled. Do not compare this as an
allocation-matched causal test against historical 20%-fixed jT results.

## Validation as of 2026-10-11

CPU command (NumPy required; PyTorch enables the optional GPU-related imports):

```
cd tests
python -m unittest test_flash_queuecost_cpu test_flash_probability_cpu \
  test_flash_netgain_cpu test_flash_preload_cpu test_flash_prefetch_cpu \
  test_flash_joint_cpu test_delivery_timing_cpu test_flash_hit_trace_cpu \
  test_flash_jmt_heads_cpu
```

55 tests passed. New regressions cover paired pre-arrival loss, immediate eviction
without spares, future reuse protection, ready-payload scoring, reset, ties,
in-flight pinning and malformed contracts. The six-window modeled CPU replay was
unchanged at 86.1125% hot: those timing assumptions yielded lead=1, so they do not
exercise the delayed-arrival distinction. CPU modeled coverage is not hardware
coverage.

Completed GPU0 oracle: six captured 512-token windows, exact next-16 routing,
real SSD/transfer/mailbox/expert kernels, synthetic expert inputs, uniform routing
weights, ideal initial residency, no attention or backbone.

| Policy | Hot activations | Achieved replay tokens/s |
|---|---:|---:|
| Saved net-gain, with miss tracing | 80.2312% | 14.0826 |
| Queue cost-benefit candidate | 79.3202% | 14.2233 |

Both had zero slot-pointer and CPU-hot/GPU-cold mismatches. Requested aggregate
pace was 15.2598 tokens/s; some windows fell behind. Capture overhead and pacing
differ, so this is not a clean speed/quality improvement claim. Replay token pace
is not generation TPS. The candidate has not demonstrated better coverage.

The six-request live jMT comparison is in progress; no completed live result is
claimed here. No checkpoint, production default or Spark benchmark was changed.
No private token/routing traces, model weights or calibration data are included.
