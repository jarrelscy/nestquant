# T40: Inkling-2-Mini NestQuant 1.75/4 (b175mm)

Target: one 12 GB card, 131,072-token context, FP8 KV cache, about 56 of 128 experts per layer at L4.
Same recipe as T39 (Inkling-2-Small): 1.75-bit base, 4-bit nested level, 75/25 text/image calibration blend.

Mini runs layers 6-17 three times per token with shared weights. `cap_mini.py` runs the official HF model per
segment and hooks every MoE call; statistics for a looped layer pool the rows of all three passes. Other files are
T39's with Mini dimensions (D 2304, F 1152, 128 experts, NSH 1, 24 layers).

| File | Purpose |
|---|---|
| `cap_mini.py` | capture (`--phase fwd`, `--phase stats`) and KLD eval (`--phase eval --wsel 2/4/hot:N/float`) |
| `cap_txt.sh`, `cap_mm.sh` | text and image captures |
| `ink_campaign.py`, `nq_ink_env.py`, `inkdq.py` | encode driver and environment (other encoder files from T39) |
| `pre_mini.sh`, `caps.sh`, `kld_mini*.sh`, `trim.py` | predecode, eval runs, trimmed KLD |
| `pred/` | predictor: `blocks_mini.py` (passes summed per layer), `train.py` (NL 24, NE 128) |

## Results (KLD vs native, 2026-11-23)

Trimmed = without tokens that exceed 0.1 nats when the eval runs with the native weights (2,162 of 118,951 text
tokens, 169 of 14,274 caption tokens).

| Setting | Text raw | Text trimmed | Caption raw | Caption trimmed |
|---|---|---|---|---|
| Native weights (floor) | 0.0287 | 0.0022 | 0.0123 | 0.0044 |
| All L2 | 0.1104 | 0.0748 | 0.1345 | 0.1204 |
| Static hot 56 | 0.0606 | 0.0255 | 0.0739 | 0.0622 |
| Floating hot 56 | 0.0469 | 0.0163 | 0.0322 | 0.0240 |
| All L4 | 0.0360 | 0.0088 | 0.0230 | 0.0171 |

Predictor coverage at 56 hot per layer: 92.2% (EMA heuristic 91.3%, static 76.0%).
