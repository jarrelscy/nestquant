# GLM-5.3-Flash NestQuant 1.5/4 on a 128 GB Apple silicon Mac

This recipe runs GLM-5.3-Flash at NestQuant 1.5/4 bits on a single Mac with an OpenAI-compatible
server. It uses the native MLX runtime in this directory with DFlash2 speculative decoding. Measured on
an M5 Max (40-core GPU, 128 GiB) on AC power, macOS 27.2:

| Workload | Decode tok/s | Notes |
|---|---:|---|
| DFlash2, coding benchmark, 6 tasks (average) | ~37 | edit, refactor and fix tasks 41–52, new code and tests 30–34, prose ~26 |
| Plain decode, coding benchmark (average) | ~25.5 | |
| GSM8K, first 100 test problems | 34.0 | 99/100 correct (earlier build, 15% hot set) |
| MMLU-Pro, first 152 test questions, through the server | 26.2 (median 30) | 128/152 correct (84.2%), sustained load |
| Terminal-Bench 2.1 with Claude Code, thinking on, first 20 of 89 tasks | 39.7 (median request 30) | 19/20 passed (95%); 1,051 requests, 1.08M output tokens, 7.6 h of decode; context median 28K, up to 78K |

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
| **Quality (default)** | `--hot-fraction 0.2 --budget-gib 22`, archive, spare-slot delivery | ~47% | ~37 tok/s (plain ~25.5) | ~86 GiB |
| Speed | `--hot-fraction 0.15 --budget-gib 16` and `NQ_REFINEMENT_DIRECT=staged` | ~30% | ~36–39 tok/s | ~84 GiB |

A 4-bit expert call decodes two trellises instead of one, so more 4-bit calls means slower verify rounds.
Single benchmark runs vary by about ±15% because generated text, and with it draft acceptance, changes.
The DFlash2 numbers were measured before the MMLU-Pro run below, on short runs with a rested GPU.

### Sustained speed

The numbers above are from short runs (192 tokens, a few minutes of GPU work). All measurements used the
96 W USB-C adapter. Under sustained load the GPU slows down while macOS still reports a normal thermal state:

- A plain fp16 matmul runs at about 54 TFLOPS when the GPU is fresh, about 21 TFLOPS under continuous load,
  and at worst about 5 TFLOPS.
- The same 192-token benchmark repeated six times in one process gave 52, 44, 39, 40, 35 and 11 tok/s.
  Acceptance and the 4-bit share did not change.
- When the server runs nonstop (MMLU-Pro, 136 requests over about an hour, 130–1,024-token answers), decode
  averaged 26.2 tok/s: median 30, 10th percentile 24, 90th percentile 34, with dips to 7–16 tok/s on some long
  answers. That run predates draft pruning.
- Terminal-Bench 2.1 with Claude Code (thinking on, about 14 hours of agent work over the first 20 tasks):
  1,080,425 output tokens at 39.7 tok/s overall. Per request the median was 30 tok/s (10th percentile 25,
  90th percentile 39); long replies decode faster than short ones because their drafts are accepted more often.
  93.5% of prompt tokens came from the cache.

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

Start and stop the server with the scripts in `macos/scripts/`:

```sh
macos/scripts/startnq.sh     # starts the server on 0.0.0.0:8000 and waits until it answers (about 2 minutes)
macos/scripts/stopnq.sh      # stops it
```

Both are safe to run twice. `startnq.sh` refuses to start when another model process is running or when less
than 95 GiB of memory is free or reclaimable (starting without that headroom can freeze macOS). It starts the
server with these settings, each of which an environment variable can change:

| Setting | Default | Variable |
|---|---|---|
| Address | 0.0.0.0:8000 | `NQ_HOST`, `NQ_PORT` |
| Context (prompt plus reply) | 131,072 tokens | `NQ_MAX_CONTEXT` |
| Sampling | temperature 1.0, top_p 0.95 | `NQ_TEMPERATURE`, `NQ_TOP_P` |
| Thinking for Claude Code requests | high effort | `NQ_THINKING` (off, low, high, max) |
| Thinking budget | 16,384 tokens | `NQ_THINKING_BUDGET` (0 = no limit) |
| API key | none | `NQ_API_KEY` |
| Model names | `glm-nestquant` and `local` | `NQ_MODEL_ALIAS` |
| Cached conversations besides the current one | 4 | `NQ_CACHE_SLOTS` (0–64) |

The log is `~/.nestquant/server.log`. Replies of 16,384 tokens or more are saved to `~/.nestquant/long-replies/`.

To watch the server, run `macos/scripts/nqstat.py -w` (refreshes every 2 s; `--json` prints the raw data). It shows
requests running and queued, the current request's phase and decode speed, the last request's prefill and decode
speed, recent averages, GPU utilisation and memory, and the server's and the system's memory. The same data is at
`GET /metrics`.

The server keeps the live conversation plus up to 4 parked ones (`NQ_CACHE_SLOTS`, about 25 MB per 1,000
tokens each; the oldest are dropped when the server reaches 100 GiB), so a client that switches between
conversations (Claude Code's main session and its side requests or subagents) resumes each from cache instead of
re-reading it. A prompt sent again unchanged is answered from the saved state with no prefill.
Without an API key, anyone on your network can use the server; set `NQ_API_KEY` to require
`Authorization: Bearer <key>` (OpenAI clients) or `x-api-key: <key>` (Anthropic clients).

The server serves `POST /v1/chat/completions` (OpenAI), `POST /v1/messages` (Anthropic, used by Claude Code)
and `GET /v1/models`, and lists the model as `glm-nestquant` and `local`. Requests may name either (any other
name is also accepted and answered by the same model). The examples below use port 8000.

```sh
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "glm-nestquant", "max_tokens": 256,
  "messages": [{"role": "user", "content": "Write a Python function that reverses a linked list."}]}'
```

With LiteLLM-based tools, use model `openai/glm-nestquant` and base URL `http://127.0.0.1:8000/v1`.

Images work the OpenAI way: add `image_url` parts (base64 `data:` URLs or `http(s)://` links) to a message.

```sh
IMG=$(base64 -i photo.png)
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "glm-nestquant", "max_tokens": 256,
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,'"$IMG"'"}},
    {"type": "text", "text": "What is in this image?"}]}]}'
```

The checkpoint's bf16 vision tower (1.1 GB) loads with the server; `--no-vision` skips it. A 640×400 image
is about 340 prompt tokens. Follow-up turns reuse the image from the cache, and a different image never
matches a cached one. Up to 16 images per request, 20 MiB each.

What to expect:
- One request at a time; others queue.
- Sampling: requests may set `temperature` and `top_p`; requests without them use the server defaults
  (`--temperature`, `--top-p`; 0 = greedy). `~/startnq.sh`-style launches use temperature 1.0, top_p 0.95.
  Speculative decoding stays exact under sampling. Thinking is off unless a request sets `reasoning_effort`
  (or Anthropic `thinking`).
- The backbone (attention, dense layers, shared experts, router, embeddings) runs as MLX 4-bit with 8-bit
  embeddings. The checkpoint ships it as FP8/bf16 and the model card's reference runtime loads it as MLX 8-bit,
  so this recipe's backbone is lower precision than the reference. The routed experts are NestQuant as shipped.
- Multi-turn conversations reuse the previous turn's state: only the new text is prefilled. New text up to
  256 tokens prefills in about 1.5 s, up to 8,192 tokens at about 250–280 tok/s. Fresh prompts and longer
  additions prefill at about 260–400 tok/s with every routed expert at 4 bits.
- Context: `--max-context` accepts up to 131,072. A 77,604-token prompt prefilled in 213 s (364 tok/s), decoded at
  24 tok/s and peaked at 94 GiB; a full 131K context would reach about 98 GiB, so keep clients near 110K.
- Thinking: OpenAI `reasoning_effort` per request; Anthropic `thinking` (Claude Code sends it every turn) maps to
  `--anthropic-thinking off|low|high|max` (startnq.sh uses high; max over-plans hard tasks past 65K tokens on this hardware).
- Thinking budget: `--thinking-budget N` closes the reasoning after N tokens (appending a short sentence and
  `</think>`) and the model continues to its answer or tool call. startnq.sh uses 16,384.

### Claude Code

Point Claude Code at the server:

```sh
ANTHROPIC_BASE_URL=http://127.0.0.1:8000 ANTHROPIC_API_KEY=local ANTHROPIC_MODEL=glm-nestquant \
CLAUDE_CODE_ATTRIBUTION_HEADER=0 CLAUDE_CODE_MAX_CONTEXT_TOKENS=110000 CLAUDE_CODE_MAX_OUTPUT_TOKENS=65536 \
API_TIMEOUT_MS=7200000 CLAUDE_STREAM_IDLE_TIMEOUT_MS=3600000 claude
```

- Claude Code asks for thinking on every turn; the server answers at the effort set by `NQ_THINKING` and streams
  the reasoning live, so long turns do not time out. A turn can take several minutes.
- `CLAUDE_CODE_ATTRIBUTION_HEADER=0` keeps the start of the prompt identical between requests, so the server
  reuses its cache. The server also removes Claude Code's per-request billing line for the same reason.
- `CLAUDE_CODE_MAX_CONTEXT_TOKENS=110000` keeps the conversation below the server's 131,072-token limit and
  memory peak.

### Terminal-Bench 2.1

The Terminal-Bench run uses Harbor with Claude Code against this server. `macos/tbench/run_claude_code.sh` holds
the exact settings and `macos/tbench/local_claude_code.py` installs Claude Code in the task containers.

1. Install Docker through Colima with Rosetta (the task images are x86-64; Claude Code crashes under QEMU):
   `sudo softwareupdate --install-rosetta --agree-to-license`, then `colima start --vm-type vz --vz-rosetta`.
2. Install Harbor in `~/src/tbench/.venv` and download the dataset:
   `harbor dataset download terminal-bench/terminal-bench-2-1 -o ~/src/tbench/datasets/terminal-bench-2-1`.
3. Start the server (`macos/scripts/startnq.sh`), then run all 89 tasks:
   `macos/tbench/run_claude_code.sh <job-name>` (add `-i <task>` to run single tasks).

Result so far (thinking on at high effort, 16K thinking budget, temperature 1.0): **19 of the first 20 tasks
passed (95%)**. The one failure, caffe-cifar-10, passed 5 of its 6 checks but did not set the `CPU_ONLY` CMake
flag. The thinking budget closed runaway reasoning 20 times. Z.ai reports 84.3% for the full-precision model
over all 89 tasks; the first 20 are the alphabetical start of the list, not a random sample. Each task takes
about 40 minutes on average, so all 89 take about 2.5 days. On the 96 W adapter the
battery drains under this load; pause runs when it falls low.

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
  --url http://127.0.0.1:8000/v1/chat/completions --output mmlu-pro.json   # add --resume to continue a run
```

MMLU-Pro uses a zero-shot chain-of-thought prompt, greedy decoding and up to 1,024 answer tokens. On the first
152 test questions (all in the business category) the default preset scored 128/152 (84.2%). 24 answers hit the
1,024-token limit and 6 gave no extractable answer, so a higher `--max-tokens` may score higher.

`--blocks 0` is plain decode and `-8` is DFlash2 with the adaptive verify block (up to 8).

## How it works

- **Experts:** a custom Metal kernel decodes the NestQuant trellis on the GPU. Each lane loads its own 32-bit
  words and decodes 16 consecutive weights; fp16 values are built from bits (`0x6400 + code`); rows routed to
  the same expert share one decode. The 4-bit hot set is predicted by the checkpoint's jT predictor and
  streamed from SSD in the background.
- **Backbone:** fused hyper-connection kernels and block attention for 1–8 token steps.
- **Speculative decoding:** DFlash2 drafts 7 tokens per round; the main model verifies them in one forward
  pass at full precision. Each verified row costs about 20 ms (it pulls in more distinct experts), so the
  runtime verifies only the prefix of the draft that maximises expected tokens per millisecond, using the
  drafter's own per-token confidence (calibrated against measured acceptance). This added about 11% on coding
  and MMLU-Pro prompts, most on prose and new code. `NQ_DRAFT_PRUNE=0` restores the older adaptive block policy.
- **Prefill:** batched layer-major prefill decodes each used expert once per layer and runs dense matmuls.

Details, measurements and history: [README.md](README.md) and [PERFORMANCE.md](PERFORMANCE.md).
