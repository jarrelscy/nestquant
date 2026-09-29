// Decode salience export for the GBDT x mps128 floating-set predictor (streaming/gbdt_predictor_v2.py).
// One CTA per call (T <= 8 decode tokens, topk 8), graph-capturable (static pointers, no host sync):
//   xn_t   = sum_i x[t,i]^2 in fp32 (x = post_attention_layernorm output = the MoE input, identical on every TP rank)
//   sal[e] += sum over slots (t,k) with ids[t,k]==e of (rsf * w[t,k])^2 * xn_t   (w = combine weight, rsf =
//             routed_scaling_factor applied by the runner after the experts; slots with w == 0 are skipped like the hits)
// acc is a device fp64 [E] running sum; every touched entry is also stored to host (pinned fp64 [E], UVA pointer), which
// the host loop diffs like the routing-hit counters. Calls on one layer are stream-ordered, so acc needs no atomics;
// duplicate experts within a call are summed in shared memory and written once (by their first slot).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename T> __device__ __forceinline__ float f32(T v);
template <> __device__ __forceinline__ float f32<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float f32<half>(half v) { return __half2float(v); }

template <typename T>
__global__ void __launch_bounds__(256) nq_sal_k(const T* __restrict__ x, int64_t sx, const half* __restrict__ w,
                                                const int64_t* __restrict__ ids, int Tn, int H, int K, float rsf,
                                                double* acc, double* host)
{
    __shared__ float red[8][8];                 // [token][warp]
    __shared__ float xn[8];
    __shared__ int se[64];
    __shared__ double sv[64];
    const int tid = threadIdx.x, lane = tid & 31, wp = tid >> 5;
    for (int t = 0; t < Tn; ++t) {
        const T* xr = x + (int64_t)t * sx; float s = 0.f;
        if ((H & 7) == 0 && ((reinterpret_cast<uintptr_t>(xr) & 15) == 0)) {
            const uint4* v4 = reinterpret_cast<const uint4*>(xr);
            for (int i = tid; i < H / 8; i += 256) {
                uint4 u = v4[i]; const T* p = reinterpret_cast<const T*>(&u);
#pragma unroll
                for (int j = 0; j < 8; ++j) { float f = f32(p[j]); s += f * f; }
            }
        } else {
            for (int i = tid; i < H; i += 256) { float f = f32(xr[i]); s += f * f; }
        }
#pragma unroll
        for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0) red[t][wp] = s;
    }
    __syncthreads();
    if (tid < Tn) { float s = 0.f; for (int j = 0; j < 8; ++j) s += red[tid][j]; xn[tid] = s; }
    __syncthreads();
    const int n = Tn * K;
    if (tid < n) {
        float ww = __half2float(w[tid]);
        se[tid] = ww != 0.f ? (int)ids[tid] : -1;
        ww *= rsf; sv[tid] = (double)(ww * ww * xn[tid / K]);
    }
    __syncthreads();
    if (tid < n && se[tid] >= 0) {
        const int e = se[tid]; double tot = 0.0; bool first = true;
        for (int j = 0; j < n; ++j)
            if (se[j] == e) { if (j < tid) { first = false; break; } tot += sv[j]; }
        if (first) { const double a = acc[e] + tot; acc[e] = a; host[e] = a; }
    }
}

void sal(torch::Tensor x, torch::Tensor w, torch::Tensor ids, double rsf, torch::Tensor acc, int64_t host_ptr)
{
    const int Tn = x.size(0), H = x.size(1), K = ids.size(1);
    TORCH_CHECK(x.dim() == 2 && x.stride(1) == 1 && Tn <= 8 && K <= 8 && w.scalar_type() == at::kHalf && w.is_contiguous() &&
                ids.scalar_type() == at::kLong && ids.is_contiguous() && w.numel() == (int64_t)Tn * K && ids.numel() == (int64_t)Tn * K &&
                acc.scalar_type() == at::kDouble && host_ptr, "nqsal: bad args");
    if (!Tn) return;
    auto st = at::cuda::getCurrentCUDAStream();
    if (x.scalar_type() == at::kBFloat16)
        nq_sal_k<__nv_bfloat16><<<1, 256, 0, st>>>((const __nv_bfloat16*)x.data_ptr(), x.stride(0), (const half*)w.data_ptr(),
            (const int64_t*)ids.data_ptr(), Tn, H, K, (float)rsf, (double*)acc.data_ptr(), (double*)host_ptr);
    else if (x.scalar_type() == at::kHalf)
        nq_sal_k<half><<<1, 256, 0, st>>>((const half*)x.data_ptr(), x.stride(0), (const half*)w.data_ptr(),
            (const int64_t*)ids.data_ptr(), Tn, H, K, (float)rsf, (double*)acc.data_ptr(), (double*)host_ptr);
    else TORCH_CHECK(false, "nqsal: x must be bf16 or fp16");
    cudaError_t e = cudaGetLastError(); TORCH_CHECK(e == cudaSuccess, "nqsal: ", cudaGetErrorString(e));
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("sal", &sal); }
