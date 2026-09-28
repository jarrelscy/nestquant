# NestQuant

Dynamic 2-4 bit MoE expert quantisation in one artifact: a 2-bit trellis base, with extra refinement bytes streamed on top for 3 and 4 bit. Targets GLM-5.3 and MiMo-V2.6, compared against EXL3 and NVFP4 using the FP8 reference. Work in progress.

- `BRIEF.md`: goals and constraints.
- `DESIGN.md`: the current format spec and adopted decisions.
- `LITERATURE.md`: literature notes.
- `threads/NN-*/`: one directory per research thread, with code, results JSON and `REPORT.md`.

## Where the code is
Fitting (encoder side):
- `threads/12-reference-encoder/`: reference encoder/decoder (`nq_encode.py`, `nq_decode.py`), in progress.
- `threads/02-feedback-conflict/`: nested trellis fitting with blended dual feedback (`fbt.py`, `gam2.py`, `eval2.py`).
- `threads/03-nested-trellis-code/`: the nested 2+2 mul1 trellis code.
- `threads/06-expert-objective/`: two-sided rounding for gate/up.
- `threads/08-ood-robustness/`: mixed routed/uniform Hessian calibration.
- `threads/05-exl3-harness/`: EXL3/NVFP4 baselines and the expert-output eval harness (`harness.py`, `run.sh`).

Inference:
- `threads/04-decode-kernel/`: A100 tensor-core decode GEMV kernels. `nqk2.cu`, `nq2.py` and `bench_chain2.py` hold the adopted additive A4/B2 decoders with fused-B chain; see `REPORT2.md`.
- `threads/13-moe-layer-kernel/`: grouped MoE layer kernel and vLLM integration, in progress.
- `threads/10-streaming-system/`: plane layout and streaming policy.

Model weights are not included. Scripts expect the GLM FP8 experts in `/tmp/nestquant/glm53-fp8-experts` (override with `NQ_GLM_SOURCE`). The CUDA 12 runtime used by the harness is expected in `threads/*/lib/` and is not committed.
