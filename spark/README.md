# NestQuant GLM-5.3 b1.75/4 on 2x DGX Spark

Serves [jarrelscy/GLM-5.3-NestQuant-1.75-4bit](https://huggingface.co/jarrelscy/GLM-5.3-NestQuant-1.75-4bit) on two DGX Spark (GB10) nodes as one TP2 vLLM instance over the ConnectX-7 link. Each node holds the 1.75-bit base of every routed expert for its half of the tensor-parallel split, and streams 4-bit upgrades for the jF-predicted hot experts from its NVMe.

**Status: untested on Spark hardware.** Every part that could be tested without a GB10 was tested (see "What was verified"). The list under "Hardware TODO" has to be checked on the first real run.

## Quick start

On each node, from a checkout of this repo (branch `spark-b175`):

```bash
# 1. build the image natively on the Spark (full vLLM source build for sm_121a, ~1-2 h; arm64 QEMU builds take days)
docker build -f spark/Dockerfile -t nestquant-spark:b175 spark/

# 2. download + merge + kernel build, then serve. HEAD_IP = the head node's CX-7 address on both nodes.
HEAD_IP=192.168.100.10 spark/start_spark.sh worker      # node 1
HEAD_IP=192.168.100.10 spark/start_spark.sh head        # node 0, OpenAI API on :8001
HEAD_IP=192.168.100.10 spark/start_spark.sh head smoke
```

Each node's first `up` downloads ~255 GB into `NQ_DATA` (default `~/nq-spark`): the backbone's non-expert files (58 GB), the jF predictor, and the node's two TP4 rank files (196 GB). It then converts the MTP layer's experts to fp8 (`spark/mtp_fp8.py`, +12 GB), merges the two TP4 ranks into one TP2 rank with `streaming/tp4to2.py` (~10 min) and builds the NestQuant kernels. Peak disk use is ~460 GB, or ~270 GB with `NQ_DROP_TP4=1`. `start_spark.sh <role> fetch` does the download, merge and kernel build without serving.

Check the CX-7 names with `ibdev2netdev` and set `NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA` if they differ from the defaults (`enp1s0f1np1`, `rocep1s0f1,roceP2p1s0f1`).

## Presets

| `NQ_PRESET` | context | MTP | 4-bit hot experts/layer (`NQ_JF_NFLOAT`) | slots/layer | est. KLD |
|---|---|---|---|---|---|
| `speed` (default) | 128K | ns=1 | 18 | 21 | ~0.064 |
| `quality` | 128K | off | 29 | 32 | ~0.055 |

`NUM_SPEC`, `NQ_MAXLEN`, `NQ_JF_NFLOAT` and `NQ_SLOTS_PER_LAYER` override the preset. Slots must be at least the floating count + 3. At 64K context each node has 3.35 GiB more, about 8 more slots per layer.

The KLD estimates interpolate linearly between two live measurements on 4x RTX PRO 6000 (BF16 teacher, full vocabulary, windows 0000-0003, fp8 KV): 18 floating gave 0.0618-0.0665 and 45 floating gave 0.0434. Neither preset has been measured on GB10.

## Speed (estimates, not measured on GB10)

Decode is bound by LPDDR5x bandwidth (273 GB/s per node). Per decode token each node reads ~11.6 GiB: ~8.7 GiB of non-expert weights and ~2.9 GiB of NestQuant experts (8 active × 75 layers). Converting the BF16 linears to fp8 cut this from ~15.6 GiB.

| | decode, 1 request | prefill |
|---|---|---|
| `speed` (MTP ns=1) | ~27-33 tok/s | ~800-1,400 tok/s up to ~16K, ~500-900 tok/s at 128K |
| `quality` (no MTP) | ~16-18 tok/s | same |

Prefill is compute bound (~75 GFLOP per token across the two nodes); the fp8 linears run on the fp8 tensor cores. The NestQuant expert kernel's prefill launch configs for 48 SMs are untuned guesses, so prefill is the least certain number. Several concurrent requests share each weight read, so total throughput rises well above the single-request rate.

## Memory budget (per node, GiB)

| item | GiB | notes |
|---|---|---|
| usable LPDDR5x | 114 | measured by howtospark (GB10 128 GB) |
| expert base + resident planes (`res/rank{r}`) | 80.8 | every routed expert at 1.75 bits, TP2 half (161.6 GiB total, measured) |
| non-expert weights, layers 0-77 + embed/head + vision | 10.7 | see below; 14.8 in BF16 before the load-time conversion. MLA q_a/kv_a, indexer and router are replicated on both ranks |
| MTP layer 78 | 4.5 | routed experts e4m3 with 128x128 block scales (`spark/mtp_fp8.py`, 9.0 in BF16); only loaded when MTP is on |
| KV cache, fp8_ds_mla | 6.7 @128K / 3.35 @64K | ~55 KB/token; the MLA latent is replicated on both ranks |
| runtime (CUDA context, graphs, activations, NCCL) | ~3 | estimate |
| 4-bit upgrade slots | rest | 5.40 MiB per slot (TP2 record, 5,660,672 B) x 75 layers = 0.395 GiB per slot/layer |

`speed`: 114 - 80.8 - 10.7 - 4.5 - 6.7 - 3 = 8.3 GiB → 21 slots/layer.
`quality`: 114 - 80.8 - 10.7 - 6.7 - 3 = 12.8 GiB → 32 slots/layer.

Non-expert weight formats:
- o_proj, layers 0-77: NVFP4, converted at load (`VLLM_ENABLE_NVFP4_P4_O_PROJ=1`, as on SM120).
- MLA q_a/kv_a (fused), q_b, shared experts, dense MLP (layers 0-2) and the MTP layer's o_proj: e4m3 per output channel (W8A16), converted at load by the fork's `fp8_w8a16` method. Decode uses SM120's fused gemv (`sm120/serve/csrc/nq_fp8o.cu`, one weight read for up to 8 rows, which covers MTP verify); prefill uses fp8 `_scaled_mm`. The list is `NQ_DBG_FP8_TARGETS` (`NQ_FP8_TARGETS` in `start_spark.sh`; `none` keeps them BF16).
- kv_b (absorbed into MLA at load), indexer, router, embed and lm_head stay BF16.

Each layer's linears are converted just before that layer's NestQuant planes load, and the slot pool is allocated after the last layer, so the BF16 copies are gone before the slots exist. Peak memory during load is below the steady state.

The slot pool is allocated while weights load, so vLLM's memory profile already counts it when sizing the KV cache from `NQ_UTIL` (0.95). On the first boot, check the log's KV-cache token count. It must cover `NQ_MAXLEN`; if it doesn't, lower `NQ_SLOTS_PER_LAYER` (and `NQ_JF_NFLOAT`).

More memory and speed, not done here (both need a KLD gate):
- NVFP4 for the shared experts and q_b (~2.2 GiB less per token per node, ~1.2x decode).
- kv_b in fp8 after MLA absorption (1.07 GiB per token per node).

## What changed vs the SM120 (4x RTX PRO 6000) serve

- **b1.75 kernel** (`sm120/nqmoe.cu`): base K-code 1 (1.75 bit, 112-bit base) via `e[19]` and residual code 9 (down 2.5625). The build mask is `NQ_DEFS=NQ_RK_CODES=0x209,NQ_RK_GU=0x9,NQ_RK_DN=0x201,NQ_BK_CODES=0x3`. At serve time a guard refuses any layer whose codes the build doesn't have. `sm120/verify_b175.py` checks it.
- **Arch**: `sm120/build.py` takes `TORCH_CUDA_ARCH_LIST` from the environment (Spark: `12.1a`).
- **TP2 from the TP4 release** (`streaming/tp4to2.py`): each node merges TP4 ranks 2r and 2r+1 into TP2 rank r, so nothing new is stored on HF.
- **Cross-node op log** (`NQ_OPLOG=dist`, `streaming/oplog_net.py`): rank 0 schedules and sends its upgrade/downgrade ops and I/O stats to rank 1 over TCP (`NQ_OPLOG_ADDR`, port 29611), in place of the `/dev/shm` log.
- **Unified memory** (`NQ_UNIFIED`, auto-on for an integrated GPU): io_uring O_DIRECT reads land straight in host-mapped slot memory, with no bounce buffer and no H2D copy. The RAM tier and prefill-borrow are off (`NQ_PREFILL_BORROW=0`).
- **One drive per node** (`NQ_REPACK_ALT=none`). `NQ_JF_NFLOAT` sets the jF floating count (77 on SM120).
- **fp8 backbone linears and fp8 MTP experts** (above). The quant-config overlay is `spark/overlay/nvfp4_arvq_hybrid.py`, based on the fork's `serving/arvq-v4-v5` file (the ARVQ-v2 backbone's format `rvq256_256x8_expert_fp16block` needs it; `sm120/serve/overlay` predates it) plus the NestQuant hook and the `NQ_MTP_FP8` path.
- **SM120 step 2** (origin/main 67a0d5d), on by default in `start_spark.sh`: decode-wait mode 3 (`NQ_DEC_ASYNC=3`, `csrc/nq_decwait.cu`, `nq_pfblock.py`; the first 64 tokens wait for landed planes, then 0.04 ms), prefill block (`NQ_PF_BLOCK=1`), hit carry (`NQ_HIT_CARRY=1`), predictor inputs 3, tap `NQ_TAP_QREAL=1` / `NQ_TAP_TODO_FIX=2`, `NQ_DBG_NO_BF16_RED=1`. Each is overridable by env. `start_spark.sh` deletes the `/dev/shm` knob files left by an earlier boot (the container runs with `--ipc host`).
- **pfblock markers over the dist channel**: on SM120 the follower reads rank 0's `{iokey}_pfblk.bin` / `_decblk.bin` from `/dev/shm`. With `NQ_OPLOG=dist` rank 0 sends each marker in-band on the op-log TCP stream (`put_mark`), and rank 1 stores it with its byte position in the stream. A follower is ready once it has replayed past that position, the same test as the file path.
- **Dropped flags**: PCIe/b12x all-reduce, DCP, LMCache, the AQLM path, `NCCL_P2P_LEVEL`.

## What was verified (no GB10 available)

- Kernel decode for b1.75 is bit-exact vs the reference (`moe.dense_W`) and the T35 test vectors at levels 2 and 4, for decode and prefill paths, at I=512 and I=1024: 160/160 projections on sm_80. The extension compiles with `-arch=sm_121a`.
- `tp4to2.py` output (records, index and `res` files) is byte-identical to `repack.py ... 2` from the encode shards on layers 10, 40 and 77, both ranks. The full 75-layer merge ran end to end (108.68 GB bin + 78 GiB res per rank, ~8 min per rank at 8 threads).
- `tests/test_oplog_net.py`: the TCP op log gives the same records as the file log, across processes, including io-stats exchange.
- `tests/test_oplog_net.py` also sends random markers between records: every marker arrives after the records sent before it, with the right stream position (2,887 records, 146 markers).
- `tests/test_pfblock_dist.py`: leader and follower `PFBlock` over localhost TCP. Prefill readiness before and after the marker, a later chunk not ready, an in-flight level-4 read blocking, and the follower's decode wait gated on the leader's step marker (9/9).
- `tests/test_leader_hits.py` with synthetic routing (the SM120 routing logs are not on this box): hit carry gets 99.97% of hits to the scheduler vs 64% without, identical to SM120's origin/main.
- `csrc/nq_decwait.cu` compiles with `-arch=sm_121a`.
- `streaming/smoke_unified.py`: the direct-to-slot path is byte-exact against the bounce path (A100, mapped memory over PCIe).
- The fork's ARVQ CUDA kernels compile for sm_121a with CUDA 13.
- Backbone: all 49 non-expert files of ARVQ-v2-hybrid match the sha256s recorded at build time, and v2 was seeded from a full v1 copy. v1 is what SM120 serves.

## Hardware TODO

- The image build on aarch64. Some optional `requirements/cuda.txt` packages (PyNvVideoCodec, tokenspeed-mla, humming-kernels, tilelang) may have no aarch64 wheels and are skipped. None of them are on this serve path.
- sm_121 runtime for the fork: FlashInfer sparse MLA, the DSA indexer, CUTLASS DSL kernels. Several fast paths check `capability == (12, 0)` and fall back to the generic path on sm_121: `vllm/v1/attention/ops/dcp_bytepack.py:57`, `raw_kv_gather.py:92`, `vllm/model_executor/warmup/fa4_cutedsl_config.py:202`.
- NCCL over CX-7 (RoCE) with vLLM `--nnodes 2`. `VLLM_GLM_COMM_OVERLAP=1` is untested across nodes; set it to 0 if the all-reduce hangs.
- Unified-memory streaming speed and the SSD rate (`NQ_TAP_RATE_GBPS`, default 6.6 GB/s).
- fp8 W8A16 on the MLA q_a/kv_a and q_b projections: SM120 runs fp8 on o_proj only. The model calls these through the linear method (no raw weight reads outside `apply`, checked in the fork), but they have not run end to end.
- Decode-wait cost across nodes: rank 1 waits on rank 0's step marker over CX-7, so its wait includes one TCP hop. Compare `NQ_DEC_ASYNC=3` against 0 on tok/s and KLD.
- Kernel launch configs at I=1024 on 48 SMs (`NQ_CFG_GU` / `NQ_CFG_DN` JSON overrides).
- KV-cache size vs `NQ_UTIL` (see memory budget).
- KLD of each preset (live server logprobs on the BF16-teacher windows), and decode tok/s.
