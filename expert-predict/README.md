# expert-predict

This directory holds offline training and evaluation for the NestQuant decode floating-set predictor. The online module is `streaming/gbdt_predictor.py` (with `gbdt_p64_s5.txt`), plus `streaming/gbdt_predictor_v2.py`.

- `build_data.py` turns the raw routing logs into per-task deduplicated streams, `data/<task>.npz` (`ex`, `tok`, `dec`, `req`). The processed traces are published at hf://jarrelscy/GLM-5.3-NestQuant-2-4bit/serving/predictor/traces/.
- `src/gbdt_feats.py`, `src/gbdt_train*.py` and `src/gbdt_eval.py` compute the GBDT features and run training and held-out evaluation.
- `src/parity_gbdt.py` checks that the online module matches the offline simulation.
- The EMA, linear and NN baselines and the capacity curves are in `src/` too.
- `PROGRESS.md` is the results log.

The scripts expect the data under /data/Jarrel/expert-predict/{data,feat} and use a venv with numpy, lightgbm and torch.
