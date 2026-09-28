# Thread 11: MatQuant / MatGPTQ nested scalar baseline (written by the lead from the agent's final message)

Files: matgptq.py, sweep.py, run.sh, runall.sh, gen.py, show.py, cfg/; raw results e36.json, e92.json, e165.json, damp.json, s03.json (control and OOD for every config). Scratch/logs: /tmp/nestquant/11-matquant-baseline/.

**Verdict: reject.** MSB-sliced nesting is not competitive at either endpoint on GLM. Even non-nested, this scalar GPTQ pipeline loses to EXL3 at 2 and 4 bit and at best ties NVFP4 routed (loses forced/OOD). Nesting adds ~4–5 pp at 4 bit.

## Method
- GPTQ over input columns, Hessian damped as EXL3 (sigma_reg 0.03; extra rows at 0.3 and 1.0). Group parameters fitted on the fully updated group.
- Nesting: one 4-bit code c per weight, 2-bit value = c>>2.
- Grids: `centre` (uniform 4-bit cells, 2-bit value = centre of its 4 sub-cells, 32 bits/group shared); `msb` (MatQuant integer grid, plain truncation); `sep` (separately stored 2- and 4-bit grids); `nu` (Any-Precision style: 2-bit Gaussian Lloyd centroids each split into 4, 16 bits/group).
- Code choice minimises λ2(w2−v2)² + λ4(w4−v4)².
- Feedback: `dual` (separate 2- and 4-bit GPTQ targets coupled through the shared code) vs `shared` (one target fed the λ-weighted mean error; MatGPTQ uses the unweighted mean).
- h0/h1 = without/with random sign + 128-block Hadamard on both sides. rX = λ4/λ2. bpw includes fp16 group params and sign bits. Harness reproduces reference numbers exactly.

## GLM L16 E36 (all-domain routed / forced, %)

| method | bpw (2 / 4) | 2-bit | 4-bit |
|---|---|---|---|
| EXL3 (refs, sigma 0.03) | 2.0 / 4.0 | **37.41 / 40.01** | **9.74 / 10.46** |
| EXL3 refit, sigma 0.3 | 2.0 / 4.0 | 34.97 / 37.05 | 9.11 / 9.64 |
| NVFP4 | 4.5 | – | 12.29 / 12.78 |
| native centre g128 h1 | 2.25 / 4.25 | 48.23 / 51.33 | 13.68 / 14.63 |
| native centre g64 h0 | 2.5 / 4.5 | 47.55 / 50.29 | 12.84 / 13.66 |
| native centre g128 h1, sigma 0.3 | 2.25 / 4.25 | 45.22 / 47.79 | 12.74 / 13.50 |
| native centre g128 h1, sigma 1.0 | 2.25 / 4.25 | 45.14 / 47.17 | 12.56 / 13.26 |
| native nu g128 h1 | 2.125 / 4.125 | 47.81 / 50.95 | 12.97 / 13.89 |
| RTN g128 h1 | 2.25 / 4.25 | 50.87 / 51.95 | 14.38 / 14.80 |
| nested centre dual h1 r1 | 2.25 / 4.25 | 47.23 / 50.12 | 21.16 / 22.52 |
| nested centre dual h1 r3 | 2.25 / 4.25 | 47.12 / 49.94 | 18.27 / 19.52 |
| nested centre dual h1 r10 | 2.25 / 4.25 | 48.94 / 51.48 | 15.44 / 16.50 |
| nested centre dual h1 r30 | 2.25 / 4.25 | 52.06 / 54.36 | 14.08 / 15.09 |
| nested centre dual h1 r3, sigma 0.3 | 2.25 / 4.25 | 46.11 / 48.44 | 15.78 / 16.70 |
| nested centre dual h1 r10, sigma 0.3 | 2.25 / 4.25 | 48.80 / 51.09 | 13.76 / 14.61 |
| nested centre dual g64 h0 r10 | 2.5 / 4.5 | 46.43 / 48.81 | 14.39 / 15.32 |
| nested nu dual h1 r3 | 2.125 / 4.125 | 45.47 / 47.93 | 17.05 / 18.16 |
| nested nu dual h1 r10 | 2.125 / 4.125 | 46.64 / 48.43 | 14.39 / 15.39 |
| nested centre shared h1 r3 | 2.25 / 4.25 | 48.79 / 50.89 | 18.26 / 19.29 |
| nested centre shared h1 r10 | 2.25 / 4.25 | 52.83 / 54.72 | 14.62 / 15.67 |
| nested sep dual h1 r3 | 2.25 / 4.5 | 47.61 / 49.96 | 19.43 / 20.70 |
| nested msb dual h1 r1 | 2.25 / 4.25 | 56.44 / 61.40 | 32.24 / 35.34 |
| nested msb dual h1 r10 | 2.25 / 4.25 | 252.7 / 327.8 | 19.04 / 20.36 |

OOD routed: EXL3-4 11.18, NVFP4 12.78, native-4 centre h1 15.67 (13.85 at sigma 1.0), nested r10 17.66.

## E92 / E165 (h1, g128, sigma 0.3; routed / forced)

| | E92 2-bit | E92 4-bit | E165 2-bit | E165 4-bit |
|---|---|---|---|---|
| EXL3 | 35.78 / 40.93 | 9.24 / 10.68 | 30.01 / 39.90 | 7.79 / 10.42 |
| NVFP4 | – | 11.60 / 12.63 | – | 10.65 / 12.62 |
| native centre | 43.56 / 48.49 | 12.11 / 13.74 | 36.38 / 47.31 | 10.13 / 13.38 |
| native nu | 43.23 / 48.06 | 11.49 / 13.01 | 36.52 / 46.91 | 9.56 / 12.68 |
| nested centre r3 | 43.81 / 48.90 | 15.08 / 16.99 | 37.42 / 47.95 | 12.75 / 16.62 |
| nested nu r3 | 43.40 / 47.42 | 13.53 / 15.32 | 37.46 / 46.60 | 11.55 / 14.98 |

## Findings
1. The scalar codebook is the limit, not the nesting. Native 2-bit is 6–11 pp worse than EXL3-2 at +0.125–0.5 bpw; native 4-bit is 2–4 pp worse than EXL3-4 and only ties NVFP4 routed (E36 best 12.56 vs 12.29), losing forced/OOD. `nu` beats NVFP4 routed on E92/E165 (11.49 vs 11.60, 9.56 vs 10.65) but loses OOD.
2. Nesting cost: with dual feedback the 2-bit view is free (r0.3–r3 match or beat native 2-bit). The 4-bit view pays +4.6 pp at r3 and +0.4 pp at r30, where 2-bit is ~4 pp worse. No λ keeps both endpoints within 1 pp of native.
3. Design details: MSB truncation is catastrophic at 2 bit (~250%); the coarse value must be the centre/centroid of its sub-cells. Dual per-precision feedback beats MatGPTQ-style shared mean-error feedback at 2 bit (47.1 vs 48.8 at r3; 48.9 vs 52.8 at r10), 4-bit unchanged. `sep` doesn't help. `nu` is the best nested variant (~1–1.5 pp better at both ends, half the scale bits). Hadamard gains only 0.3–0.5 pp.
4. Damping 0.3–1.0 helps scalar GPTQ by 1–3 pp and EXL3 equally (EXL3-2 34.97, EXL3-4 9.11), so the gap does not close. A small nonzero λ4 regularises the 2-bit fit (~1 pp better than native 2-bit at sigma 0.03), consistent with H2 and thread 01.

## Recommendation
Do not build on a scalar integer container; nest inside a trellis/vector code whose 2-bit base reaches EXL3-2. Carry over: decode the coarse level as the centroid of its refinement cells, never by truncation; keep a separate feedback target per precision coupled through the shared code.
