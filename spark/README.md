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

Each node's first `up` downloads ~255 GB into `NQ_DATA` (default `~/nq-spark`): the backbone's non-expert files (58 GB), the jF predictor, and the node's two TP4 rank files (196 GB). It then merges the two TP4 ranks into one TP2 rank with `streaming/tp4to2.py` (~10 min) and builds the NestQuant kernels. Peak disk use is ~450 GB, or ~255 GB with `NQ_DROP_TP4=1`. `start_spark.sh <role> fetch` does the download, merge and kernel build without serving.

Check the CX-7 names with `ibdev2netdev` and set `NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA` if they differ from the defaults (`enp1s0f1np1`, `rocep1s0f1,roceP2p1s0f1`).

## Presets

| `NQ_PRESET` | context | MTP | 4-bit hot experts/layer (`NQ_JF_NFLOAT`) | slots/layer |
|---|---|---|---|---|
| `quality` (default) | 128K | off | 26 | 29 |
| `speed` | 64K | ns=1 | 12 | 15 |

`NUM_SPEC`, `NQ_MAXLEN`, `NQ_JF_NFLOAT` and `NQ_SLOTS_PER_LAYER` override the preset. Slots must be at least the floating count + 3.

The KLD target sheet (`threads/35-nq15`) sized the Spark at H=45 hot experts per layer, which measured 0.0432 against the BF16 teacher. That sizing assumed 12 GiB/node of non-expert weights. The served backbone is larger (next section), so H45 does not fit. In the same harness H32 gave 0.0446-0.0449. Expect ~0.045-0.046 for `quality`, and higher for `speed`. Neither preset has been KLD-measured yet.

## Memory budget (per node, GiB)

| item | GiB | notes |
|---|---|---|
| usable LPDDR5x | 114 | measured by howtospark (GB10 128 GB) |
| expert base + resident planes (`res/rank{r}`) | 78 | every routed expert at 1.75 bits, TP2 half |
| non-expert weights, layers 0-77 + embed/head + vision | 14.7 | BF16 checkpoint; attention o_proj goes to NVFP4 at load (`VLLM_ENABLE_NVFP4_P4_O_PROJ=1`, as on SM120). MLA q_a/kv_a, indexer and router are replicated on both ranks |
| MTP layer 78 routed experts | 9.0 | BF16 in the checkpoint (256 experts, not in the NestQuant repack); only loaded when MTP is on |
| KV cache, fp8_ds_mla | 6.7 @128K / 3.35 @64K | ~55 KB/token; the MLA latent is replicated on both ranks |
| runtime (CUDA context, graphs, activations, NCCL) | ~3 | estimate |
| 4-bit upgrade slots | rest | 5.40 MiB per slot (TP2 record, 5,660,672 B) x 75 layers |

`quality`: 114 - 78 - 14.7 - 6.7 - 3 = 11.6 GiB → 29 slots/layer.
`speed`: 114 - 78 - 14.7 - 9.0 - 3.35 - 3 = 6.0 GiB → 15 slots/layer.
MTP at 128K leaves ~2.6 GiB (6 slots/layer), so it isn't offered as a preset.

The slot pool is allocated while weights load, so vLLM's memory profile already counts it when sizing the KV cache from `NQ_UTIL` (0.95). On the first boot, check the log's KV-cache token count. It must cover `NQ_MAXLEN`; if it doesn't, lower `NQ_SLOTS_PER_LAYER` (and `NQ_JF_NFLOAT`).

Ways to free more memory, not done here:
- Store the MTP experts in fp8 or NVFP4 (9 → 4.5 / 2.5 GiB).
- Quantize the rest of the BF16 attention (q_b, kv_b, shared experts). Both need a KLD gate.

## What changed vs the SM120 (4x RTX PRO 6000) serve

- **b1.75 kernel** (`sm120/nqmoe.cu`): base K-code 1 (1.75 bit, 112-bit base) via `e[19]` and residual code 9 (down 2.5625). The build mask is `NQ_DEFS=NQ_RK_CODES=0x209,NQ_RK_GU=0x9,NQ_RK_DN=0x201,NQ_BK_CODES=0x3`. At serve time a guard refuses any layer whose codes the build doesn't have. `sm120/verify_b175.py` checks it.
- **Arch**: `sm120/build.py` takes `TORCH_CUDA_ARCH_LIST` from the environment (Spark: `12.1a`).
- **TP2 from the TP4 release** (`streaming/tp4to2.py`): each node merges TP4 ranks 2r and 2r+1 into TP2 rank r, so nothing new is stored on HF.
- **Cross-node op log** (`NQ_OPLOG=dist`, `streaming/oplog_net.py`): rank 0 schedules and sends its upgrade/downgrade ops and I/O stats to rank 1 over TCP (`NQ_OPLOG_ADDR`, port 29611), in place of the `/dev/shm` log.
- **Unified memory** (`NQ_UNIFIED`, auto-on for an integrated GPU): io_uring O_DIRECT reads land straight in host-mapped slot memory, with no bounce buffer and no H2D copy. The RAM tier and prefill-borrow are off (`NQ_PREFILL_BORROW=0`).
- **One drive per node** (`NQ_REPACK_ALT=none`). `NQ_JF_NFLOAT` sets the jF floating count (77 on SM120).
- **Dropped flags**: PCIe/b12x all-reduce, DCP, LMCache, the AQLM path, `NCCL_P2P_LEVEL`.

## What was verified (no GB10 available)

- Kernel decode for b1.75 is bit-exact vs the reference (`moe.dense_W`) and the T35 test vectors at levels 2 and 4, for decode and prefill paths, at I=512 and I=1024: 160/160 projections on sm_80. The extension compiles with `-arch=sm_121a`.
- `tp4to2.py` output (records, index and `res` files) is byte-identical to `repack.py ... 2` from the encode shards on layers 10, 40 and 77, both ranks. The full 75-layer merge ran end to end (108.68 GB bin + 78 GiB res per rank, ~8 min per rank at 8 threads).
- `tests/test_oplog_net.py`: the TCP op log gives the same records as the file log, across processes, including io-stats exchange.
- `streaming/smoke_unified.py`: the direct-to-slot path is byte-exact against the bounce path (A100, mapped memory over PCIe).
- The fork's ARVQ CUDA kernels compile for sm_121a with CUDA 13.
- Backbone: all 49 non-expert files of ARVQ-v2-hybrid match the sha256s recorded at build time, and v2 was seeded from a full v1 copy. v1 is what SM120 serves.

## Hardware TODO

- The image build on aarch64. Some optional `requirements/cuda.txt` packages (PyNvVideoCodec, tokenspeed-mla, humming-kernels, tilelang) may have no aarch64 wheels and are skipped. None of them are on this serve path.
- sm_121 runtime for the fork: FlashInfer sparse MLA, the DSA indexer, CUTLASS DSL kernels. Several fast paths check `capability == (12, 0)` and fall back to the generic path on sm_121: `vllm/v1/attention/ops/dcp_bytepack.py:57`, `raw_kv_gather.py:92`, `vllm/model_executor/warmup/fa4_cutedsl_config.py:202`.
- NCCL over CX-7 (RoCE) with vLLM `--nnodes 2`. `VLLM_GLM_COMM_OVERLAP=1` is untested across nodes; set it to 0 if the all-reduce hangs.
- Unified-memory streaming speed and the SSD rate (`NQ_TAP_RATE_GBPS`, default 6.6 GB/s).
- Kernel launch configs at I=1024 on 48 SMs (`NQ_CFG_GU` / `NQ_CFG_DN` JSON overrides).
- KV-cache size vs `NQ_UTIL` (see memory budget).
- KLD of each preset (live server logprobs on the BF16-teacher windows), and decode tok/s.
