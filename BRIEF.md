# NestQuant: shared brief for all research threads

## Goal (from the user)
Design a new weight quantization format from scratch for MoE experts in **GLM 5.3** (block FP8 source) and **MiMo 2.6** (MXFP4 source). Expert shapes: gate/up [2048, 6144], down [6144, 2048], SwiGLU.

1. **Dynamic 2–4 bit.** One fitted artifact. A 2-bit base runs on its own; extra bytes can be streamed in at inference to reach 3 and 4 bits without refitting. The higher-bit decoder MAY reinterpret the base bits (it does not have to be "base + residual").
2. **Quality.** Beat EXL3 at 2 bit and at 4 bit on relative expert-output error, with identical calibration data (ceteris paribus), on as many experts as possible.
   - UPDATE (user, 2026-09-28): per model. GLM must beat EXL3 at 2 bit AND 4 bit, and at 4 bit also beat NVFP4, all measured against the FP8 reference. MiMo only needs to beat EXL3 at 2 bit. MiMo 3/4-bit levels only have to exist and be sane; do not spend effort optimising MiMo 4 bit.
3. **4-bit vs NVFP4.** On GLM, the 4-bit endpoint should beat calibrated NVFP4 (ModelOpt W4A16) against the FP8 reference, including OOD tokens.
4. **Speed.** Beat EXL3 decode latency for batches of 1–4 tokens (MTP verification) at both 2 and 4 bit, with the same artifact that gives the quality.

## What already failed (orbit-duet, ~1.5 days of work by another agent)
Repo: `/home/coder/git/orbit-duet` (READ ONLY, another agent is actively working in it). "Duet": 4-weight vectors from an 8192-entry dictionary, 10 address bits shared with neighbouring groups + 3 private bits = 2 bpw, message-passing encoder, Hadamard + LDLQ-style feedback. 4-bit = frozen 2-bit base + a separately fitted refinement byte per 4 weights. Result: 2 bit roughly ties EXL3; 4 bit loses everywhere (e.g. GLM L16 E36 routed 12.96% vs EXL3 9.74% vs NVFP4 12.29%), and it is 1.6–1.9x slower than EXL3 at 4 bit (B1: 117 vs 63 µs; B4: 154 vs 92 µs on A100). ~30 structural variants (E8, Golay, parity, tree, low-rank, dictionary tweaks) gave <0.5 pp. An early EXL3 "2-bit trellis + 2-bit trellis residual" prototype was only ~2.5% worse in weight L2 than native 4-bit on tiles, but 33–40% slower in a research kernel. See `docs/research-history.md` and `results/*/REPORT.md` there.

## Working hypotheses from the lead (test, don't assume)
- H1 **Rate allocation over LDL innovations.** With LDLQ the proxy loss is ~ Σ_j D_jj·η_j² (D from the block LDL of the rotated Hessian). Uniform bits per column wastes rate if AM(D)/GM(D) is large. A progressive format naturally supports per-16-column-block rates that grow with the stream level.
- H2 **Feedback conflict.** The 2-bit base is fitted with 2-bit error feedback, which pushes Q2 away from W; the 4-bit view then has to undo that. Blending the base's target between the 2-bit and 4-bit feedback targets may trade a little 2-bit quality for a lot of 4-bit quality.
- H3 **Refinement must be as strong a code as the base** (trellis-grade), and decode ops/weight at 4 bit must not exceed EXL3's (~4–5 int ops/weight with 3INST). 2D-output LUT codes (QTIP "HYB", V=2) may halve decode cost.
- H4 Speed is partly kernel engineering: 63 µs for a 4-bit expert is ~300 GB/s effective, far below A100 bandwidth, so EXL3 has slack to beat.

## Shared data (read only)
- Python: `source /home/coder/git/glm52/artifacts/shared-bit-graphs/runtime/env.sh` then `/home/coder/git/glm52/.venv/bin/python`. CUDA needs that env (driver is old; compat lib). `PYTHONPATH=/home/coder/git/orbit-duet` gives you its loaders/evaluators (import, do not edit). exllamav3 is installed in that venv.
- GLM L16/L49/L66 experts 36, 92, 165 pilot statistics (65,536-token pilot, 18k rows for E36): `/home/coder/git/orbit-duet/runs/glm53_pilot_matched_l{16,49,66}/statistics/l{L}_e{E}.pt` and `..._training_sample.pt`. Matched EXL3 refits: `.../exl3_e{E}/expert_{2,4}.bin` (+ results json). Calibrated NVFP4: `.../nvfp4_e{E}/weights.pt`. Evaluation receipts: `.../*evaluation*.json`.
- CORRECTION: `runs/native_id_control_v1_capture` and `runs/ood_controlled_v1_capture` are both MiMo captures (expert ids up to 383), NOT GLM. All GLM reference numbers come from ONE capture: `runs/glm53_matched_context_pilot_v1_capture/layer_16.pt` (5120 rows, 139 routed to E36; control = domains prefixed control:, ood = prefixed ood:). Never fit on it.
- Shared GLM harness (thread 05): `threads/05-exl3-harness/harness.py`, run via its `run.sh`; reproduces the EXL3/NVFP4 numbers to 0.001 pp and refits EXL3 byte-identically; `quantize_exl3_like` accepts per-16-block K lists and half-integer K. See its REPORT.md.
- GLM FP8 source: box restart 2026-09-28 wiped /tmp. Per-expert copies of L16/49/66 × E36/92/165 are in `/tmp/nestquant/glm53-fp8-experts` (re-fetch with `orbit-duet/benchmarks/fetch_glm53_experts.py --output ... --layer L --experts ...`). The harness defaults to it. MiMo source (/tmp/mimo-a100) is gone.
- MiMo L55 E70 (408,663 training rows) prepared tensors: `runs/full55_e70/prepared/`; statistics path is recorded in the receipts under `results/full55_matched/` and `results/matched_native/`.
- Evaluator: `python -m benchmarks.matched_native_expert` (see `docs/glm53.md`) computes forced/routed/OOD relative output L2 for EXL3/NVFP4/Duet from the same statistics.
- Reference numbers (relative expert-output L2, lower is better): GLM L16 E36 routed: EXL3-2 37.41, EXL3-4 9.74, NVFP4 12.29; forced control: EXL3-2 40.01, EXL3-4 10.46, NVFP4 12.78. MiMo L55 E70 routed: EXL3-2 15.99, EXL3-4 4.00.

## Resource rules (hard)
- This box is shared with a live agent. **Never OOM host RAM or GPU VRAM. Never kill or signal processes you did not start.**
- Host RAM: keep your processes under **48 GB RSS** total. Stream/mmap large tensors. Check `free -g` before big allocations.
- GPU: use only your assigned GPU (`CUDA_VISIBLE_DEVICES=<n>`). Cap VRAM at **12 GB**: call `torch.cuda.set_per_process_memory_fraction(12/80)` at start. Check `nvidia-smi` first; if free VRAM on your GPU < 20 GB, wait or go CPU.
- CPU: `OMP_NUM_THREADS`/`MKL_NUM_THREADS` ≤ 16.
- Disk: home NFS is ~97% full. Keep writes in `/home/coder/git/nestquant/threads/<your-thread>/` under **1 GB** (code, small JSON, report). Large scratch goes to `/tmp/nestquant/<your-thread>/` (ephemeral, may be wiped).
- Do not write into `/home/coder/git/orbit-duet` or `/tmp/orbit-duet-*`. Do not push to GitHub.
- Wall-clock: aim to finish within ~3 hours. Prefer small, decisive experiments over broad builds.

## Deliverable
`threads/<your-thread>/REPORT.md`: question, method, numbers (with baselines on the same data), verdict (adopt / reject / needs more), and the concrete recommendation for the NestQuant design. Keep your final message to the lead short: key numbers + recommendation.

## Note on reports
Some subagents found REPORT.md writes blocked. If that happens, put the full report in your final message; the lead will write the file.

UPDATE 2026-09-28 (user): a 1-2% margin over EXL3-2 at 2 bit is sufficient. Level-2 encoder frozen at per-tile sign + two-sided G + blend 0.3 (thread 17). Only levels 2 and 4 are required (no level 3). One joint fit must yield both levels; promotion streams only the P4 plane + δ onto a byte-identical base.
