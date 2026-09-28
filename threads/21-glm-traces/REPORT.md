# Thread 21: GLM-format calibration corpus and trace group (lead, from the agent's final message, 2026-09-29)

Corpus: /tmp/nestquant/corpus/glm53_calib_glmfmt_v1 (backed up to s3://annalise-shared-prod/jarrel/nestquant/corpus/).

- 26,910 ChatML docs were re-rendered with GLM-5.3's chat_template and 12,855 raw docs kept as-is, with 0 errors. 584 nq-tail docs and 24 overlap docs were excluded, and 166 docs forced into val.
- An assistant answer ends with <|user|> and a tool call ends with <|observation|>. Truncated traces get no end token.
- Segment ids: segment_id % 2^20 = doc index. Each later non-contiguous piece of the same doc adds k·2^20.
- Per-token boundary arrays: bnd_think_d and bnd_end_d (1..32, 0 = none). They are exclusive: the nearer boundary wins, ties go to end, and only positions in the same segment count.

| group | windows | fit tokens | think ends | notes |
|---|---|---|---|---|
| c512 | 17,134 | 8.66M | 0 | manifest a47fb37c |
| c2048 | 2,555 | 5.15M | 2,897 | all docs with a real reasoning end; manifest cdf25c64 |
| c2048_traces | 771 | 1.56M | 512 (+16 val) | on-box traces only; manifest e93cbec1 |

Traces, by source (the user excluded Qwen and gpt-oss; paid or external generation was cancelled and nothing was spent):

| source | tokens | docs | think | end |
|---|---|---|---|---|
| TB2.1 GLM-5.3 ARVQ trajectories | 773,702 | 10 | 144 | 144 |
| claude-opus mdmathena thinking | 332,563 | 128 | 216 | 245 |
| glm-5.3f companion sessions | 218,367 | 9 | 143 | 172 |
| GPQA GLM-5.2 hybrid traces | 216,631 | 6 | 3 | 3 |
| DeepSeek v4.1-flash sessions | 36,057 | 2 | 22 | 37 |

GPQA: question indices 7, 10 and 13 are in the calibration set (manifest `gpqa_included`), so exclude them from later GPQA evals. 3 of the traces are runaways, flagged as truncated. For the other 3, the </think> position was lost and had to be restored heuristically.

13-gram dedup against the corpus, GPQA, tb4, ICH and the thread-18 eval sets dropped nothing. The generation scripts (build_prompts.py, gen_traces.py) are committed but were never run.
