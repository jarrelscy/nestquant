/*
 * nq_fp8o.cu (nq-kld) -- fused variant of vllm csrc/quantization/aqlm_moe/fp8_linear_v4.cu for the FP8 o_proj decode
 * path: takes the bf16 activation directly (the 2^-ACT_SHIFT prescale + fp16 cast happen while staging the tile to
 * shared memory: bf16 -> fp32 -> *2^-6 (exact) -> fp16 RN, bit-identical to the host prescale) and writes bf16
 * (fp32 -> fp16 RN -> bf16 RN, bit-identical to the old .half() output + .to(bf16)). One kernel per o_proj instead of
 * cast/mul/cast/gemv/cast. Registered as torch op nq_fp8o::gemv_bf16.
 */

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#ifndef ACT_SHIFT
#define ACT_SHIFT 6  // host multiplies activations by 2^-6, epilogue undoes it
#endif

namespace nq_fp8o {

constexpr int THREAD_M = 16;
inline int ceildiv(int a, int b) { return (a + b - 1) / b; }

__device__ __forceinline__ half2 e4m3x2_to_half2(uint16_t v) {
  return half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)v, __NV_E4M3));
}

// C[N, M] = (act_prescaled[N, K] @ w_e4m3[M, K]^T) * (scale[M] * 2^ACT_SHIFT)
__global__ void Fp8W8A16Gemv(const int4* __restrict__ packed,   // e4m3 [M,K]
                             const float* __restrict__ scale,   // [M]
                             const int4* __restrict__ B_all,    // bf16 [N,K] (unscaled)
                             __nv_bfloat16* __restrict__ C,     // bf16 [N,M]
                             const int prob_m, const int prob_k) {
  const int slot = blockIdx.y;
  const int row = (blockDim.x / 32) * blockIdx.x + (threadIdx.x / 32);
  const bool pred = row < prob_m;
  const int lane = threadIdx.x % 32;

  const int a_gl_stride = prob_k / 16;  // uint4 (16 e4m3) per weight row
  const uint4* a_row = reinterpret_cast<const uint4*>(packed) +
                       (int64_t)row * a_gl_stride;
  const int4* B = B_all + (int64_t)slot * (prob_k / 8);

  // Per tile: 32 lanes * 2 int4 of activation (each lane consumes 16 half).
  __shared__ int4 sh_b[32 * 2];
  float res = 0;

  int iters = (prob_k / 8 + 2 * 32 - 1) / (2 * 32);
  int b_gl_rd = 0;
  int a_rd = lane;
  bool have = pred && a_rd < a_gl_stride;
  uint4 w_cur = have ? a_row[a_rd] : uint4{};

  while (iters--) {
    __syncthreads();
    for (int i = threadIdx.x; i < 32 * 2; i += blockDim.x) {
      if (b_gl_rd + i < prob_k / 8) {
        int4 raw = B[b_gl_rd + i];
        const __nv_bfloat16* rb = reinterpret_cast<const __nv_bfloat16*>(&raw);
        int4 o;
        half* oh = reinterpret_cast<half*>(&o);
#pragma unroll
        for (int j = 0; j < 8; j++) oh[j] = __float2half_rn(__bfloat162float(rb[j]) * (1.0f / (float)(1 << ACT_SHIFT)));
        sh_b[i] = o;
      }
    }
    __syncthreads();
    b_gl_rd += 32 * 2;

    if (have) {
      const int a_nx = a_rd + 32;
      const bool have_nx = a_nx < a_gl_stride;
      uint4 w_nx = have_nx ? a_row[a_nx] : uint4{};  // prefetch next chunk

      const uint16_t* wp = reinterpret_cast<const uint16_t*>(&w_cur);
      const half2* bb = reinterpret_cast<const half2*>(&sh_b[lane * 2]);
      half2 acc = {};
#pragma unroll
      for (int i = 0; i < 8; i++) acc = __hfma2(e4m3x2_to_half2(wp[i]), bb[i], acc);
      res += __half2float(acc.x) + __half2float(acc.y);

      a_rd = a_nx;
      w_cur = w_nx;
      have = have_nx;
    }
  }

  if (pred) {
#pragma unroll
    for (int i = 16; i > 0; i /= 2) res += __shfl_down_sync(0xffffffff, res, i);
    if (lane == 0) {
      const float s = scale[row] * (float)(1 << ACT_SHIFT);
      C[(int64_t)slot * prob_m + row] = __float2bfloat16_rn(__half2float(__float2half(res * s)));
    }
  }
}

static void pick_grid(int prob_m, int n, dim3& blocks, int& threads) {
  int dev, sms;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  int waves = 0, thread_m;
  do {
    waves++;
    thread_m = ceildiv(prob_m, waves * sms);
  } while (thread_m > THREAD_M);
  blocks = dim3(ceildiv(prob_m, thread_m), n);
  threads = 32 * thread_m;
}

}  // namespace nq_fp8o

// x:      [N, K] half (already prescaled by 2^-ACT_SHIFT on host)
// packed: [M, K] uint8 (e4m3 bits)
// scale:  [M] f32 per-output-channel dequant scale
// returns [N, M] half
torch::Tensor gemv_bf16(const torch::Tensor& x,
                             const torch::Tensor& packed,
                             const torch::Tensor& scale) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kBFloat16 && x.is_contiguous());
  TORCH_CHECK(packed.dtype() == torch::kUInt8 && packed.is_contiguous());
  TORCH_CHECK(scale.dtype() == torch::kFloat32 && scale.is_contiguous());
  const int64_t n = x.size(0);
  const int64_t k = x.size(1);
  const int64_t m = packed.size(0);
  TORCH_CHECK(packed.size(1) == k, "packed K mismatch");
  TORCH_CHECK(scale.size(0) == m, "scale M mismatch");
  TORCH_CHECK(k % 16 == 0, "K must be a multiple of 16");

  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  auto out = torch::empty({n, m}, x.options());
  if (n == 0) return out;
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  dim3 blocks;
  int threads;
  nq_fp8o::pick_grid((int)m, (int)n, blocks, threads);
  nq_fp8o::Fp8W8A16Gemv<<<blocks, threads, 0, stream>>>(
      (const int4*)packed.data_ptr(), scale.data_ptr<float>(),
      (const int4*)x.data_ptr(), (__nv_bfloat16*)out.data_ptr(), (int)m, (int)k);
  return out;
}

TORCH_LIBRARY(nq_fp8o, m) {
  m.def("gemv_bf16(Tensor x, Tensor packed, Tensor scale) -> Tensor");
}
TORCH_LIBRARY_IMPL(nq_fp8o, CUDA, m) { m.impl("gemv_bf16", &gemv_bf16); }
TORCH_LIBRARY_IMPL(nq_fp8o, Meta, m) {
  m.impl("gemv_bf16", [](const torch::Tensor& x, const torch::Tensor& p, const torch::Tensor& s) {
    return torch::empty({x.size(0), p.size(0)}, x.options()); });
}
