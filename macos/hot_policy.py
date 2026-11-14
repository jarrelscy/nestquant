"""Hot-set policy settings shared by the runtime (native_refinements.py, native_dflash.py) and the offline
simulator (simulate_hot_policy.py). No MLX imports, so the simulator runs without the GPU.

The settings only change which experts are held at 4 bits and when their refinements are read and
installed; weights, precision and the memory budget are unchanged. Defaults were chosen with
simulate_hot_policy.py on recorded coding-benchmark and agentic traces (20% hot set, expert-major archive):
simulated 4-bit share of routed calls 0.460 -> 0.516 (coding traces 0.436 -> 0.493) from the 46e443d settings
to these defaults; in-model on the coding benchmark 0.40 -> 0.46 (mean per task) at unchanged memory.
"""
import json
import os

import numpy as np


DEFAULTS = dict(
    # jT FloatingSet parameters: score = (1 - mix) * EMA share + mix * jT block, EMA half-life in rows, pool
    # refresh every G rows, wanted experts' score x (1 + hm) when re-ranking. Was hm=4.0, half_life=64.0.
    jt=dict(mix=0.75, hm=1.5, G=1, half_life=32.0),
    # Victim among resident experts outside the wanted set when a new refinement is installed: 'score' (lowest
    # loading priority, the jT/EMA mix) or 'fifo' (oldest install, the previous rule).
    evict='score',
    # Wait for the refinement copies started at the last publish before encoding the next verify, so reads that
    # completed during a round join it one round earlier. Off: it raised the 4-bit share by ~0.02 (simulated and
    # in-model) but the ~60 copies per publish (~1 ms each per install thread) stalled each round by ~16 ms,
    # about 15% of decode speed.
    verify_wait=False,
    # Decode reads in flight with staged installs. 64 records of at most 8.34 MB fit the 512 MiB staging budget.
    pending=64,
    # Decode delivery: 'spare' reads each refinement straight into a free slot (one preadv, no staging buffer and
    # no install copy) and spends the staging budget on spare slots instead; 'staged' reads into a staging
    # buffer and copies into a slot reserved at the next publish (the previous behaviour); 'direct' reserves at
    # publish and then reads into the slot.
    delivery='spare',
    # Spare delivery: reads in flight, and how many extra resident experts outside the wanted set each layer may
    # evict at a publish (beyond its spare slots) to make room for missing wanted ones.
    direct_inflight=128,
    evict_ahead=2,
)


def settings():
    """DEFAULTS overridden by NQ_FLASH_JT_PARAMS (JSON, any subset of the jT parameters), NQ_HOT_EVICT,
    NQ_HOT_VERIFY_WAIT (0/1), NQ_REFINEMENT_PENDING, NQ_REFINEMENT_DIRECT (spare / staged or 0 / direct or 1),
    NQ_DIRECT_INFLIGHT and NQ_EVICT_AHEAD."""
    jt = dict(DEFAULTS['jt'])
    raw = os.environ.get('NQ_FLASH_JT_PARAMS')
    if raw:
        jt.update(json.loads(raw))
    evict = os.environ.get('NQ_HOT_EVICT', DEFAULTS['evict'])
    if evict not in ('fifo', 'score'):
        raise ValueError(f'Unknown NQ_HOT_EVICT {evict!r}')
    verify_wait = os.environ.get('NQ_HOT_VERIFY_WAIT', '1' if DEFAULTS['verify_wait'] else '0') != '0'
    pending = int(os.environ.get('NQ_REFINEMENT_PENDING', DEFAULTS['pending']))
    delivery = {'0': 'staged', '1': 'direct'}.get(os.environ.get('NQ_REFINEMENT_DIRECT', ''),
                                                    os.environ.get('NQ_REFINEMENT_DIRECT') or DEFAULTS['delivery'])
    if delivery not in ('spare', 'staged', 'direct'):
        raise ValueError(f'Unknown NQ_REFINEMENT_DIRECT {delivery!r}')
    inflight = int(os.environ.get('NQ_DIRECT_INFLIGHT', DEFAULTS['direct_inflight']))
    ahead = int(os.environ.get('NQ_EVICT_AHEAD', DEFAULTS['evict_ahead']))
    return dict(jt=jt, evict=evict, verify_wait=verify_wait, pending=pending, delivery=delivery,
                direct_inflight=inflight, evict_ahead=ahead)


def spare_rows(budgets, total):
    """Split `total` spare slots over the layers in proportion to their pools (largest remainder)."""
    budgets = np.asarray(budgets, dtype=float)
    raw = total * budgets / budgets.sum()
    rows = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - rows), kind='stable')[:int(total - rows.sum())]:
        rows[i] += 1
    return rows.tolist()


def jt_parameters_json():
    """The NQ_FLASH_JT_PARAMS value the runtime installs: the full jT parameter set after overrides."""
    return json.dumps(settings()['jt'])


def seed_from_prompt(predictor, salience, weight):
    """Seed each layer's hot pool from whole-prompt routing salience (NQ_SEED_PROMPT=1; off by default).

    The prompt's salience share is blended into the policy's moving average (weight 1.0 = equal mass to the
    current average) and the pool is reset to the top experts by that score, keeping fixed experts. The
    predictor's usual updates and hysteresis take over from there. Returns the number of pool changes."""
    changed = 0
    for index, layer in enumerate(predictor.layers):
        if layer not in salience:
            continue
        share = salience[layer] / max(float(salience[layer].sum()), 1e-30)
        policy = predictor.policies[index]
        inner = getattr(policy, 'inner', policy)
        mass = float(inner.state.sum()) or 1.0
        inner.state = inner.state + weight * mass * share
        fixed = getattr(policy, 'fixed', np.zeros(288, dtype=bool))
        score = inner.state / max(float(inner.state.sum()), 1e-30)
        order = np.argsort(-score, kind='stable')
        candidates = order[~fixed[order]]
        new = fixed.copy()
        new[candidates[:inner.nf - int(fixed.sum())]] = True
        changed += int((new & ~inner.cur).sum())
        inner.cur = new
    return changed


def choose_victim(slots, wanted, scores=None):
    """The resident expert to evict for a new wanted one, or None if every resident expert is wanted.

    slots: experts in insertion order (oldest install first); wanted: container supporting `in`;
    scores: [288] loading priorities for the 'score' rule (lowest goes; ties keep insertion order), or None
    for the 'fifo' rule (oldest install outside the wanted set)."""
    if scores is None:
        return next((key for key in slots if key not in wanted), None)
    victim, best = None, None
    for key in slots:
        if key in wanted:
            continue
        value = scores[key]
        if best is None or value < best:
            victim, best = key, value
    return victim
