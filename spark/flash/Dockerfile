# Native aarch64 build on GB10. Build from repository root. Not validated on GB10.
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-dev python3-pip python3-venv git curl ca-certificates build-essential ninja-build cmake pkg-config libnuma-dev libssl-dev && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=12.1a MAX_JOBS=4 LIBURING=/opt/liburing
RUN git clone --depth 1 --branch liburing-2.7 https://github.com/axboe/liburing.git /tmp/liburing && cd /tmp/liburing && ./configure --prefix=/opt/liburing && make -j4 -C src && make install
RUN pip install --upgrade pip && pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
ARG VLLM_REV=487ecf187d3dfe74d2cf6119a92881dba403c219
RUN git clone https://github.com/vllm-project/vllm.git /opt/vllm && cd /opt/vllm && git checkout ${VLLM_REV}
COPY spark/flash/glm5next.patch /tmp/glm5next.patch
RUN cd /opt/vllm && git apply --check /tmp/glm5next.patch && git apply /tmp/glm5next.patch && python use_existing_torch.py && pip install -r requirements/build.txt && pip install --no-build-isolation -e .
# Unlike the old b175 recipe, TileLang is REQUIRED by Flash's mHC path.
RUN python -c 'import torch, tilelang, flashinfer, vllm; import vllm._C'
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
