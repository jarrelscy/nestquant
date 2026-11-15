# T39: Inkling-2-Small NestQuant 1.75/4

Build of thinkingmachines/Inkling-2-Small (48 layers, 384 experts, top-8) for 2x RTX PRO 6000.
Released: https://huggingface.co/jarrelscy/Inkling-2-Small-NestQuant-1.75-4bit (public). `HF_README.md` is a copy of its card.

Work dir on the b200 box was `/tmp/nq_inkling` (ephemeral). Paths in the scripts point there.
Encoder runs in `/home/coder/git/glm52/.venv`; capture and eval in `/tmp/nq_inkling/venv` (transformers main, has inkling2).

## Files

| File | Purpose |
|---|---|
| `inkdq.py` | Dequantize the native checkpoint (NVFP4 L0-39, MXFP8 L40-47) |
| `corpus/` | Re-template the GLM calibration corpus to Inkling's chat format |
| `corpus_mm/` | Image calibration corpus (1,200 fit / 200 val images, 640 px, 256 tokens each) and vision-tower loader |
| `cap_ink.py` | Capture of expert stats and routing traces; KLD eval (`--no-stats --wdir --wsel 2/4/hot:N/float`) |
| `nq_ink_env.py`, `ink_layer.py`, `ink_fin.py`, `ink_campaign.py` | NestQuant encode driver (cfg `b175mm` = 75/25 text/image blend) |
| `predecode_ink.py`, `pre_*.sh` | Predecode L2/L4 experts to bf16 for the eval |
| `pred/` | Floating hot-set predictor: `blocks.py` (block tables from traces), `train.py` (fit / eval) |
| `*.sh` | Eval and campaign runners (`kld_run.sh`, `milestones.sh`, `mm_ab.sh`, `final_eval.sh`) |
| `upload_one.py` | Single-commit HF upload (avoids the 128 commits/hour limit) |

The traces, corpora and predictor block tables are private and not in this repo.

## Results (full model, KLD vs native, 64 hot experts per layer)

| Setting | Text KLD | Caption KLD |
|---|---|---|
| All L2 | 0.0361 | 0.0779 |
| Static hot 64 | 0.0179 | 0.0663 |
| Floating hot 64, salience EMA | 0.0114 | 0.0398 |
| Floating hot 64, predictor | 0.0114 | 0.0365 |
| All L4 | 0.0060 | 0.0227 |

Eval: `cap_ink.py --wsel float --pred-mode {batch,stream,ema4}`; `stream` runs the published `serving/predictor/predictor.py`.
