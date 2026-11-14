# GLM-5.3-Flash NestQuant 1.5/4 on a 128 GB Apple silicon Mac

This recipe runs GLM-5.3-Flash at NestQuant 1.5/4 bits on a single Mac with an OpenAI-compatible
server. It uses the native MLX runtime in this directory with DFlash2 speculative decoding. Measured on
an M5 Max (40-core GPU, 128 GiB) on AC power, macOS 27.2:

| Workload | Decode tok/s | Notes |
|---|---:|---|
| Plain decode, coding benchmark (average) | ~26 | ~25 at 8K–30K context |
| DFlash2, coding benchmark, 6 tasks (average) | ~39 | edit, refactor and fix tasks 41–56, new code 27–34, prose ~27 |
| GSM8K, first 100 test problems | 34.0 | 99/100 correct (earlier build) |
| Agent turns (Terminal-Bench, 2K–20K context) | ~27 median | earlier build; prefix reuse, new turns prefill in ~1–8 s |

Peak memory is about 84–86 GiB. Quality settings: the full 1.5-bit expert base stays resident, 15% of expert
slots hold 4-bit refinements (about half of all expert calls run at 4 bits), and every verified token uses
the same precision as plain decoding.

## 1. Requirements

- Apple silicon Mac with 128 GB of unified memory. Close other large applications; the process needs
  about 86 GiB.
- Use a proper power adapter. On a low-wattage charger the GPU throttles and decode becomes erratic.
- About 260 GB of free disk: 176 GiB checkpoint, 55 GiB base archive, 5.5 GiB backbone, 2.3 GB drafter.
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

This writes `~/Models/GLM/predictor-coreml-v1/` (about 144 MB). The server's default paths match these
locations.

## 4. Run the server

```sh
.venv-macos/bin/python macos/dflash_server.py --port 18090 --max-context 32768
```

Loading takes about 2 minutes. The server binds to 127.0.0.1, serves `POST /v1/chat/completions` (streaming
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
