Flash routing telemetry is written atomically every two seconds of active execution
to `NQ_FLASH_ROUTING_STATS` (default `/artifacts/nq-flash-routing.json`). On the local
server this is `/tmp/nestquant/flash-artifacts/nq-flash-routing.json`.

`total` covers the current server lifetime; `current_request` resets at position 0.
Both contain `prefill_committed`, `decode_committed`, `prefill_executed`, and
`decode_executed` as those phases occur. Executed includes rejected target verify
rows; committed excludes them. MTP's separate FP8 expert layer is not counted.

- `mean_hot_of_8`: mean number of the eight selected experts that actually used
  level 4 with nonzero FP16 routing weight, averaged over token/layer observations.
- `hit_rate`: hot activations divided by all nonzero-weight activations.
- `hot_count_histogram`: counts of token/layer observations with 0 through 8 hot experts.
- `layers`: individual layer counts and means.

The hot mask is gathered from the actual GPU expert table on the execution stream,
after mailbox application and MoE execution, before the next mailbox application.
It does not infer readiness from the predictor's desired pool or pending loads.
The owned mask is included in the existing batched routing snapshot. No prompt,
token IDs, or per-token routing are exported. Timestamps identify stale idle stats.

Schema v2 also reports `desired_pool_hit_rate`, `desired_but_cold`, and
`outside_desired_and_cold` (counts and fractions of active routes). The latter two
partition actual cold activations. The desired mask is the policy's selected pool
for this forward, before its next update; it is not a prediction fitted to these
observed routes. A hot expert can still be outside the current desired pool while
its demotion is pending, so desired minus actual alone is not the loading-miss rate.

`io` holds executor bandwidth, per-drive read time/queue depth, outstanding ops,
free slots, mailbox backlog and recent latency quantiles. The first interval is
primed after initial loading. Read latency includes engine queuing, while the
per-drive service timing runs from submission to observed completion. These are
not proof of SSD bandwidth saturation; other disk users and software polling can
also affect them. Known vLLM startup warmups are excluded.

## Throughput-aware delivery

`NQ_FLASH_PREFETCH=throughput` (default) leaves the shipped jT policy and
102/74 desired pools unchanged. It ranks pending desired upgrades by committed
0.5 EMA + 0.5 block scores and limits admitted replacements using measured
completed-record throughput. `NQ_FLASH_PREFETCH_MAX_PENDING=64` bounds outstanding
loads plus demotions; the adaptive floor is eight. `NQ_FLASH_PREFETCH_SECONDS=0.1`
sets the delivery window, also bounded by the 16-token predictor horizon.
Initialization fills the published pool under the same 64-operation bound.

Stale queued reads are cancelled best-effort. Only executor/mailbox acknowledgments
release slots, including when an already-started cancelled read finishes normally.
Old residents are demoted only as replacements are admitted; there is no bulk
pool eviction or unbounded slot-wait queue. Slot storage and spare counts do not
increase. `NQ_FLASH_PREFETCH=legacy` retains the prior delivery behavior.

`io.prefetch` reports admission limit, pending operations, desired upgrades not
yet admitted, estimated record throughput and cancellation requests. Cancellation
requests are not necessarily successful cancellations. The decoder still falls
back to the base when an upgrade has not landed. Different delivery timing can
therefore change outputs even though the predictor policy is unchanged; this is
not a lossless scheduling claim or a reproduced KLD result.

## Salience coverage (schema v3)

`hot_salience_coverage` is sum(w² × xn over actual hot active routes) divided by
sum(w² × xn over all active routes), where w is the captured FP32 routing weight
and xn is the squared norm of normalized MoE input. FP64 CPU accumulation uses
existing predictor snapshots; no additional GPU transfer is required.
`desired_salience_coverage` uses jT's desired mask at execution. The
`desired_but_cold_salience_fraction` and `outside_desired_and_cold_salience_fraction`
partition the remaining salience. Values are ratios of sums across tokens and
layers, not averages of per-token ratios. Per-layer values are also exported.
Rejected verification rows are excluded from committed coverage, retained in
executed coverage. Prefill and decode remain separate. Zero total salience yields
null, not zero. `salience_token_rows` identifies the measured population.

Compare with the reported held-out harness reference of 81.8% salience and 67.1%
activation coverage cautiously: workload and delivery assumptions differ. Existing
v2 counters cannot reconstruct salience retrospectively. Changes load only on a
server restart; editing the module alone does not activate the counter.
