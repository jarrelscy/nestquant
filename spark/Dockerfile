# NestQuant b1.75/4 serving image for DGX Spark (GB10: aarch64, sm_121, CUDA 13).
#
# Contents: our vLLM fork (jarrelscy/vllm-glm52-sm120) built from source for sm_121a, the fork's ARVQ ctypes kernels
# rebuilt for sm_121a, liburing (NestQuant io_uring streaming), lightgbm (jF trees), hf CLI. The nestquant repo itself
# is NOT baked in: spark/start_spark.sh mounts the checkout at /nq (same as the SM120 compose), and builds the
# NestQuant kernels (nqmoe / nqsal / nqstream) on first start, on a GPU, into a persistent build dir.
#
# Build natively on a Spark (an arm64 QEMU build of vLLM from source takes days):
#   docker build -f spark/Dockerfile -t nestquant-spark:b175 spark/
# The fork changes csrc/, so precompiled upstream vLLM wheels do not apply: this is a full source build (~1-2 h on
# a GB10 with MAX_JOBS=16; lower MAX_JOBS if the build OOMs).
ARG CUDA_IMAGE=nvidia/cuda:13.0.2-devel-ubuntu24.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential ninja-build cmake pkg-config libnuma-dev \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# liburing (static, for streaming/nqstream.cu; stream_engine.py links $LIBURING/lib/liburing.a)
ARG LIBURING_REF=liburing-2.7
RUN git clone --depth 1 -b ${LIBURING_REF} https://github.com/axboe/liburing.git /tmp/liburing \
    && cd /tmp/liburing && ./configure --prefix=/opt/liburing && make -j8 -C src && make install \
    && rm -rf /tmp/liburing
ENV LIBURING=/opt/liburing

# vLLM fork. serving/arvq-v4-v5 (d7ada6d) = ARVQ v4/v5 serving + the MTP buffer / proposal-RNG fixes that the SM120
# image (glm53-nestquant-sm120:fixes12-mtp-buffer-rng-20260917) carries.
ARG VLLM_REPO=https://github.com/jarrelscy/vllm-glm52-sm120.git
ARG VLLM_REF=serving/arvq-v4-v5
ARG TORCH_CUDA_ARCH_LIST=12.1a
ARG MAX_JOBS=16
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} CUDA_HOME=/usr/local/cuda MAX_JOBS=${MAX_JOBS}
RUN git clone --filter=blob:none ${VLLM_REPO} /opt/vllm && cd /opt/vllm && git checkout ${VLLM_REF}
WORKDIR /opt/vllm

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130
RUN uv venv --python 3.12 .venv && . .venv/bin/activate \
    && uv pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url ${TORCH_INDEX}

# Packages in requirements/cuda.txt that may have no aarch64 wheel: installed one by one, skipped if unavailable.
# None of them is on the NestQuant GLM-5.3 serve path (video codec, alternative MLA / quant gemm kernels).
ARG OPTIONAL_REQS="PyNvVideoCodec tokenspeed-mla humming-kernels tilelang"
RUN . .venv/bin/activate && python use_existing_torch.py \
    && for p in ${OPTIONAL_REQS}; do grep -i "^$p" requirements/cuda.txt >> /tmp/optional.txt || true; \
         sed -i "/^$p/Id" requirements/cuda.txt; done \
    && uv pip install -r requirements/build.txt \
    && uv pip install --no-build-isolation -e . \
    && while read -r r; do uv pip install "$r" || echo "optional requirement skipped on aarch64: $r"; done < /tmp/optional.txt \
    && python -c "import vllm, vllm._C; print('vllm', vllm.__version__)"

# DeepGEMM JIT names the fp8 MQA-logits headers impls/sm{arch}_*.cuh, so on GB10 it looks for sm121_*, which DeepGEMM
# does not ship (the indexer crashes on its first eager call, e.g. cudagraph_mode NONE). The sm120 kernels are
# guarded by __CUDA_ARCH__ >= 1200: copy them under the sm121 name (start_spark.sh also mounts spark/overlay/deep_gemm).
RUN cd vllm/third_party/deep_gemm/include/deep_gemm/impls \
    && for k in fp8_mqa_logits fp8_paged_mqa_logits; do \
         [ -f sm121_$k.cuh ] || sed "s/void sm120_$k(/void sm121_$k(/" sm120_$k.cuh > sm121_$k.cuh; done

# ARVQ ctypes kernels (ship as .cu + build scripts hard-coded to sm_120a). Only the MTP layer and non-routed parts use
# the hybrid quant path on Spark (NestQuant serves every routed expert), but the module imports the libraries at load.
RUN . .venv/bin/activate && cd vllm/model_executor/layers/quantization/arvq \
    && sed -i "s/compute_120a,code=sm_120a/compute_${TORCH_CUDA_ARCH_LIST%.*}${TORCH_CUDA_ARCH_LIST#*.},code=sm_${TORCH_CUDA_ARCH_LIST%.*}${TORCH_CUDA_ARCH_LIST#*.}/g" build*.sh \
    && grep -h gencode build*.sh | sort -u \
    && for f in build*.sh; do NVCC=$CUDA_HOME/bin/nvcc ARVQ_PYTHON=/opt/vllm/.venv/bin/python bash "$f"; done \
    && ls *.so

# jF predictor trees (LightGBM) + HF CLI for start_spark.sh downloads
RUN . .venv/bin/activate && uv pip install lightgbm==4.7.0 narwhals scipy "huggingface_hub[cli,hf_xet]"

ENV FLASHINFER_DISABLE_VERSION_CHECK=1 \
    VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256 \
    VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
    VLLM_DISABLE_FP8_W8A16=1 \
    CUTE_DSL_ARCH=sm_121a \
    PATH="/opt/vllm/.venv/bin:${PATH}"
EXPOSE 8001
ENTRYPOINT ["/opt/vllm/.venv/bin/vllm"]
