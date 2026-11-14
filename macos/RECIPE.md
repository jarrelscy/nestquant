# GLM-5.3-Flash NestQuant 1.5/4 on a 128 GB Apple silicon Mac

This recipe runs GLM-5.3-Flash at NestQuant 1.5/4 bits on a single Mac with an OpenAI-compatible
server. It uses the native MLX runtime in this directory with DFlash2 speculative decoding. Measured on
an M5 Max (40-core GPU, 128 GiB) on AC power, macOS 27.2:

| Workload | Decode tok/s | Notes |
|---|---:|---|
| DFlash2, coding benchmark, 6 tasks (average) | ~32 | edit, refactor and fix tasks 35–45, new code 23–27, prose ~21 |
| Plain decode, coding benchmark (average) | ~25.5 | |
| GSM8K, first 100 test problems | 34.0 | 99/100 correct (earlier build, 15% hot set) |
| Agent turns (Terminal-Bench, 2K–20K context) | ~27 median | earlier build; prefix reuse, new turns prefill in ~1–8 s |

Peak memory is about 86 GiB. Quality settings (the defaults): the full 1.5-bit expert base stays resident, 20%
of expert slots hold 4-bit refinements, and every verified token uses the same precision as plain decoding.
About 46% of expert calls run at 4 bits on coding work and about 52% on agent-style prompts (the jT predictor
chooses which experts to upgrade; selection and delivery settings are in `hot_policy.py`). Upgrades are read
straight into spare slots and become usable in the next forward pass; `NQ_REFINEMENT_DIRECT=staged` restores
the older staged copies (about 3% faster decode, about 3 points fewer 4-bit calls). With
`--hot-fraction 0.15 --budget-gib 16` the share is about 30% and memory about 84 GiB.

### Presets

| Preset | Settings | 4-bit expert calls (coding) | DFlash2 decode (coding avg) | Peak memory |
|---|---|---:|---:|---:|
| **Quality (default)** | `--hot-fraction 0.2 --budget-gib 22`, archive, spare-slot delivery | ~47% | ~32 tok/s (plain ~25.5) | ~86 GiB |
| Speed | `--hot-fraction 0.15 --budget-gib 16` and `NQ_REFINEMENT_DIRECT=staged` | ~30% | ~36–39 tok/s | ~84 GiB |

A 4-bit expert call decodes two trellises instead of one, so more 4-bit calls means slower verify rounds.
Single benchmark runs vary by about ±15% because generated text, and with it draft acceptance, changes.

### Sustained speed

The numbers above are from short runs (192 tokens, a few minutes of GPU work). All measurements used the
96 W USB-C adapter. Under sustained load the GPU slows down while macOS still reports a normal thermal state:

- A plain fp16 matmul runs at about 54 TFLOPS when the GPU is fresh, about 21 TFLOPS under continuous load,
  and at worst about 5 TFLOPS.
- The same 192-token benchmark repeated six times in one process gave 52, 44, 39, 40, 35 and 11 tok/s.
  Acceptance and the 4-bit share did not change.
- When the server runs nonstop (MMLU-Pro, 130–1,024-token answers), decode is mostly 25–34 tok/s, with dips to
  7–16 tok/s on long answers.

Expect lower and more variable decode than the table when the server runs for a long time. The most likely
reason is the adapter's power limit, but this is not confirmed. To check on your machine, run
`sudo powermetrics --samplers gpu_power,thermal -i 2000` during decode and compare GPU frequency and power
with decode speed. The Speed preset does less GPU work per token and should slow down less.

## 1. Requirements

- Apple silicon Mac with 128 GB of unified memory. Close other large applications; the process needs
  about 86 GiB with the default 20% hot set.
- Use a proper power adapter. On a low-wattage charger the GPU throttles and decode becomes erratic.
- About 360 GB of free disk: 176 GiB checkpoint, 55 GiB base archive, 94 GB refinement archive, 5.5 GiB
  backbone, 2.3 GB drafter.
- Xcode Command Line Tools (`xcode-select --install`) and [uv](https://docs.astral.sh/uv/).

## 2. Code and environment

```sh
git clone -b macos-fast-decode https://github.com/jarrelscy/nestquant.git
cd nestquant
uv venv --python 3.12 .venv-macos
uv pip install --python .venv-macos/bin/python -r macos/requirements.txt 'mlx-vlm==0.7.12'
```

## 3. Model files

Download the pinned checkpoint (about 176 GiB without the experiment and evaluation folders):

```sh
export MODEL="$HOME/Models/GLM/GLM-5.3-Flash-NestQuant-1.5-4bit"
.venv-macos/bin/hf download jarrelscy/GLM-5.3-Flash-NestQuant-1.5-4bit \
  --revision 4023870a6f2c6097e37fa5d2e54040db64b011e9 \
  --exclude 'experiments/*' --exclude 'evaluations/*' --local-dir "$MODEL"
.venv-macos/bin/python macos/verify_checkpoint.py --model "$MODEL" --output /tmp/checkpoint-verification.json
```

Prepare the native files: the resident 1.5-bit base banks and the 4-bit backbone with 8-bit embeddings.

```sh
.venv-macos/bin/python macos/native_assets.py --model "$MODEL" --output-dir "$HOME/Models/GLM/resident-banks-v1"
.venv-macos/bin/python macos/native_backbone.py --model "$MODEL" --bits 4 --embedding-bits 8 \
  --output-dir "$HOME/Models/GLM/backbone-affine4-e8-v1"
```

Download the DFlash2 drafter (Apache-2.0):

```sh
mkdir -p "$HOME/Models/GLM/drafters/dflash2" && cd "$HOME/Models/GLM/drafters/dflash2"
for f in config.json model.safetensors; do
  curl -fL -o "$f" "https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/resolve/main/$f"
done
cd -
```

Optional but recommended: convert the jT expert predictor to Core ML so it runs on the Neural Engine. This
frees CPU time for the refinement readers. It uses a separate environment so the runtime's packages stay
unchanged; the runtime finds it automatically and falls back to PyTorch if it is missing.

```sh
uv venv --python 3.12 .venv-coreml
uv pip install --python .venv-coreml/bin/python coremltools torch==2.15.0 numpy safetensors
.venv-coreml/bin/python macos/coreml_predictor.py convert
```

This writes `~/Models/GLM/predictor-coreml-v1/` (about 144 MB).

Recommended: repack the 4-bit refinements expert by expert. The runtime then fetches each expert's upgrade
with one read instead of assembling it from 8 shard files. The archive is byte-identical to the checkpoint's
refinements (the converter verifies every record). It takes about 5 minutes and 94 GB of disk.

```sh
.venv-macos/bin/python macos/refinement_archive.py convert --model "$MODEL" \
  --base "$HOME/Models/GLM/resident-banks-v1" --output "$HOME/Models/GLM/refinements-expert-major-v1"
.venv-macos/bin/python macos/refinement_archive.py verify --model "$MODEL" \
  --base "$HOME/Models/GLM/resident-banks-v1" --output "$HOME/Models/GLM/refinements-expert-major-v1"
```

The runtime uses the archive automatically when it exists at that path (set `NQ_REFINEMENT_ARCHIVE` to use
another location, or to `0` to use the shard reader).

Measured with a 15% hot set on the coding benchmark: prefill is about 2× faster (7–10 s instead of 14–17 s for
360–1,700-token prompts), decode uses about 25% less CPU and about 3 GiB less memory, and more expert calls run
at 4 bits (35% instead of 30%; late upgrades 8% instead of 14%). Because more calls take the 4-bit path, decode
is about 5–10% slower. To keep today's decode speed and 4-bit share while still getting the faster prefill,
also set `NQ_REFINEMENT_RESERVE=3x`. Leave `NQ_REFINEMENT_CACHE` unset (page caching made decode slower).

The server's default paths match these locations.

## 4. Run the server

```sh
.venv-macos/bin/python macos/dflash_server.py --port 18090 --max-context 32768
```

Loading takes about 2 minutes. By default the server binds to 127.0.0.1 (this Mac only). To serve other
machines, add `--host 0.0.0.0` and set an API key with `--api-key <key>` (or `NQ_SERVER_API_KEY`); clients
then send `Authorization: Bearer <key>`. The server serves `POST /v1/chat/completions` (streaming
and non-streaming) and `GET /v1/models`, and reports the model as `glm-nestquant`. Any API key is accepted.

```sh
curl http://127.0.0.1:18090/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "glm-nestquant", "max_tokens": 256,
  "messages": [{"role": "user", "content": "Write a Python function that reverses a linked list."}]}'
```

With LiteLLM-based tools, use model `openai/glm-nestquant` and base URL `http://127.0.0.1:18090/v1`.

What to expect:
- One request at a time; others queue.
- Greedy decoding only (`temperature` and `top_p` are accepted and ignored). Thinking is off.
- Multi-turn conversations reuse the previous turn's state: only the new text is prefilled. New text up to
  256 tokens prefills in about 1.5 s, up to 8,192 tokens at about 250–280 tok/s. Fresh prompts and longer
  additions prefill at about 260–400 tok/s with every routed expert at 4 bits.
- Context up to 32K tokens is tested (`--max-context` accepts up to 65,536).

## 5. Benchmarks

```sh
# Six-task coding benchmark (prompts in a directory of .txt files)
.venv-macos/bin/python macos/native_dflash.py --model "$MODEL" --archive "$HOME/Models/GLM/resident-banks-v1" \
  --backbone "$HOME/Models/GLM/backbone-affine4-e8-v1" --drafter "$HOME/Models/GLM/drafters/dflash2" \
  --prompt-dir prompts/ --tokens 192 --blocks 0,-8 --batched-prefill --output codebench.json

# GSM8K (download test.jsonl from openai/grade-school-math first)
.venv-macos/bin/python macos/eval_gsm8k.py --model "$MODEL" --archive "$HOME/Models/GLM/resident-banks-v1" \
  --backbone "$HOME/Models/GLM/backbone-affine4-e8-v1" --drafter "$HOME/Models/GLM/drafters/dflash2" \
  --problems test.jsonl --limit 100 --output gsm8k.json

# MMLU-Pro through a running server (questions: a JSON list of rows from TIGER-Lab/MMLU-Pro, test split)
.venv-macos/bin/python macos/eval_mmlu_pro.py --questions mmlu_pro.json \
  --url http://127.0.0.1:18090/v1/chat/completions --output mmlu-pro.json   # add --resume to continue a run
```

`--blocks 0` is plain decode and `-8` is DFlash2 with the adaptive verify block (up to 8).

## How it works

- **Experts:** a custom Metal kernel decodes the NestQuant trellis on the GPU. Each lane loads its own 32-bit
  words and decodes 16 consecutive weights; fp16 values are built from bits (`0x6400 + code`); rows routed to
  the same expert share one decode. The 4-bit hot set is predicted by the checkpoint's jT predictor and
  streamed from SSD in the background.
- **Backbone:** fused hyper-connection kernels and block attention for 1–8 token steps.
- **Speculative decoding:** DFlash2 drafts up to 7 tokens per round; the main model verifies them in one
  forward pass at full precision. An adaptive policy picks the block size from measured acceptance and round
  cost.
- **Prefill:** batched layer-major prefill decodes each used expert once per layer and runs dense matmuls.

Details, measurements and history: [README.md](README.md) and [PERFORMANCE.md](PERFORMANCE.md).
