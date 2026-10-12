# Native aarch64 build on GB10. Build from repository root. Not validated on GB10.
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-dev python3-pip python3-venv git curl ca-certificates build-essential ninja-build cmake pkg-config libnuma-dev libssl-dev && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=12.1a MAX_JOBS=2 LIBURING=/opt/liburing
RUN git clone --depth 1 --branch liburing-2.7 https://github.com/axboe/liburing.git /tmp/liburing && cd /tmp/liburing && ./configure --prefix=/opt/liburing && make -j4 -C src && make install
RUN pip install --upgrade pip && pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
ARG VLLM_REV=487ecf187d3dfe74d2cf6119a92881dba403c219
ADD spark/flash/vllm-pinned.tar /opt/vllm/
RUN cd /opt/vllm && test "$(git rev-parse HEAD)" = "${VLLM_REV}" && git reset --mixed HEAD && git diff --exit-code HEAD
COPY spark/flash/glm5next.patch /tmp/glm5next.patch
RUN git init /opt/flashmla && cd /opt/flashmla && git fetch --depth=1 https://github.com/JaredforReal/FlashMLA.git 8447acbcb558db892bf7c1197d225be1c95b168c && git checkout --detach FETCH_HEAD && test "$(git rev-parse HEAD)" = "8447acbcb558db892bf7c1197d225be1c95b168c" && git submodule update --init --recursive --depth=1
ENV FLASH_MLA_SRC_DIR=/opt/flashmla
RUN git init /opt/flashattn && cd /opt/flashattn && git fetch --depth=1 https://github.com/JaredforReal/flash-attention.git 2b84100f50e1d2a8726a86c86d14c2f9c9e5a67c && git checkout --detach FETCH_HEAD && test "$(git rev-parse HEAD)" = "2b84100f50e1d2a8726a86c86d14c2f9c9e5a67c" && git submodule update --init --recursive --depth=1
ENV VLLM_FLASH_ATTN_SRC_DIR=/opt/flashattn
RUN cd /opt/vllm && git apply --check /tmp/glm5next.patch && git apply /tmp/glm5next.patch && python use_existing_torch.py && pip install -r requirements/build/cuda.txt && pip install --no-build-isolation -e .
# Unlike the old b175 recipe, TileLang is REQUIRED by Flash's mHC path.
RUN python -c 'import torch, tilelang, flashinfer, vllm; import vllm._C_stable_libtorch; import vllm._moe_C_stable_libtorch'
# Generate from this pinned vLLM's headers, not an older fork's implementation.
RUN cd /opt/vllm/vllm/third_party/deep_gemm/include/deep_gemm/impls && \
    for k in fp8_mqa_logits fp8_paged_mqa_logits; do \
      test -s "sm120_${k}.cuh" && grep -q "sm120_${k}" "sm120_${k}.cuh" && \
      sed "s/sm120_${k}/sm121_${k}/g" "sm120_${k}.cuh" > "sm121_${k}.cuh"; \
    done
COPY . /nq
RUN pip install --no-deps /nq/spark/flash && pip install huggingface_hub safetensors numpy
ENV PYTHONPATH=/nq/spark/flash NQ_HOME=/nq NQ_FLASH_ENABLE=1 NQ_FLASH_NATIVE_FP8=1 VLLM_PLUGINS=nq_flash VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
WORKDIR /nq
ENTRYPOINT ["vllm"]
