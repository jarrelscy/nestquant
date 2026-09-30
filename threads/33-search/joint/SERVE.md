# T33i joint floating-set predictor: serve note

The code is `gpu_predictor.GPUJointPredictor`. Its numpy reference is `joint_predictor.JointPredictor`, and parity is checked by `parity_gpu.py` and `parity_stream.py`. The interface is the same as `streaming/gbdt_predictor_v2.GBDTPredictorV2` in `mode='sync'`:

    p = GPUJointPredictor(layers=range(3, 78), fixed, net_path, n_float=77, hm=H, device="cuda", graph=True)
    p.step(counts, ntok, token_ids, new_request, sal)   # -> True at a refresh (every 16 tokens)
    want = p.target(resident)                           # bool [75, 256] floating set to be resident

## Inputs per step, for each decode step or chunk of ≤16 tokens

- `counts [75, 256]`: routed-slot hits per layer and expert from this step's top-8 routing. It can be a numpy array or a device tensor, and a device tensor avoids the H2D copy.
- `sal [75, 256]`: the sum over this step's routed slots of `w^2 * xn`. Here `w` is the final gate weight used in the MoE combine, and `xn = sum(x^2)` in fp32 of the normalised MoE input. This is the same definition as GBDTPredictorV2.
- `ntok`: the number of tokens in this step. `token_ids` are used only for the think/answer state (`</think>` switches to answer). `new_request=True` resets that state.
- Build a fresh instance at the start of every request or chain. Features are trained with the state starting at zero, and the floating set starts at `floating_default`.

## State

All state lives on the device:

- Per layer and expert:
  - 7 hit EMAs and 7 salience EMAs, both f64
  - 2 serve-v2 hit EMAs, f32
  - think and answer EMAs, f32
  - last-hit block, i64
  - block accumulators: counts f32, answer counts f32, salience f64
  - last-block hits f32 and salience f64
- That is about 164 B per (layer, expert), about 3.2 MB for 75 x 256.
- Weights: the net has about 2.0 M parameters, about 8 MB in fp32. v2's 60 trees are held as 5 small [60, 29] tables.
- Measured peak VRAM with the CUDA graph and its memory pool: 109 MiB, well under the 2 GB budget.
- Host state is a few scalars: token counts in the block, the think/answer segment, the think and answer weight sums, and the block index.

## Work per refresh (once every 16 tokens)

1. `_close_block`: in-place updates of all EMAs and block state. This is about 40 small kernels, outside the graph. It then mirrors the host scalars into 0-dim device buffers (`b_wt`, `b_wa`, `b_state`, `b_nblk`, `b_pos`).
2. `_core`: a CUDA graph replay of the following steps.
   - The 22 features.
   - v2's 60 trees, a batched traversal over 75 x 256 rows. Comparisons are in f64, matching LightGBM's `double(x) <= thr`.
   - The 2-layer transformer residual over the 256 experts of each layer, batch 75, bf16 autocast.
   - `S = exp(log v2 + r)`.
3. `S` goes D2H, about 77 KB. `target()` then runs on the host: top-77 of `S * (1 + hm)` for residents, with stable ties, the same as v2.

Graph setup: `graph=True` warms `_core` up three times on a side stream, then captures it with `torch.cuda.graph`. `_core` is pure: it only reads state tensors and scalar buffers whose addresses never change, because all updates are in-place. Replay is therefore valid for the whole request. Capture happens once per instance. For serving, keep one instance per request slot and reset its state in place rather than re-capturing. That reset is not written yet (see Open items).

## Stream and ordering assumptions

- One dedicated CUDA stream for the predictor. Each refresh's inputs (`counts`, `sal`) must be complete before `step()` is enqueued. With model tensors, record an event after the MoE layer's routing and have the predictor stream wait on it.
- In sync mode (lag 0), the set computed at the end of block k serves block k+1, matching the offline metric. At serve time the upgrade DMAs for that set must land before block k+1's tokens use them; otherwise use the scheduler's next-refresh semantics. All metric numbers here are sync.
- The D2H of `S` synchronises the predictor stream only. The host-side `target()` (stable argsort of 75 x 256) is **not** included in the 2.99 ms. It measures 0.77 ms on one CPU thread, so the end-to-end host path is about 3.8 ms. A device-side top-k inside the graph would remove most of that (see Open items).
- This is a single stream: the 75 layers are batched in one call, with no per-layer launches.

## Timing context: 2.99 ms per refresh

Measured with `parity_gpu.py` (GRAPH=1 BF16=1 NOREF=1) on one idle A100-SXM4-80GB, with nothing else on the GPU and 4 CPU threads. The 0.77 ms host `target()` is excluded:

- Median 2.99 ms, p90 3.49 ms, over 192 refreshes on heldout chains 0 and 5.
- The window covers H2D of one step's counts and salience, `_close_block`, graph replay, and D2H of `S`.
- Non-refresh accumulate step: 0.046 ms.
- Without the graph: 5.43 ms in fp32. Graph with an fp32 net: 4.04 ms.

Context for these numbers:

- They are launch- and latency-bound, not FLOP-bound. The graph removes most of the Python and launch overhead, and `_close_block` (about 40 launches) is still outside it.
- Running concurrently with the model will add contention. That includes SM sharing with the decode kernels and CPU jitter on the launching thread. This has not been measured.
- Because the predictor runs async, once per 16 tokens, the budget is a 16-token decode window, not a per-token cost.
- Precision: bf16 vs fp32 changes individual top-77 sets near the boundary, but not the metric. A heldout re-score in CPU fp32 gave 78.24/78.39, identical to the GPU bf16 scores.

## Open items

- Capture `_close_block` into the graph as well. Its python-float factors depend on per-block token counts, so they would have to move to device buffers too.
- Add a device-side `target()` and an in-place reset for reusing an instance across requests.
- Measure under a live decode alongside the model.
