// Thread 23: exact re-implementation of exllamav3 quantize_tiles_kernel<K=2, cb=2 (mul1), L=256> (states only).
//
// Bit-identical by construction: same fp16 decode (decode_mul1_product_2), same per-candidate fp16 ops
// (hsub2, hmul2 on the first step, hfma2 after), same sequential strict-less candidate selection from k = 0, same
// argmin_cost rank tie order, same two-pass tail-biting schedule (forward(L/2, -1), traceback to ring position 0,
// forward(0, end_state), traceback). Differences are storage/scheduling only:
//   * history: the 2-bit winning branch k per out-edge (prev edge = (k << 12) | (e >> 2)) instead of a uint16 edge,
//     8 out-edges per uint16 word -> 1 MB per tile instead of 8 MB (the reference is history-write bound);
//   * a thread owns 8 consecutive out-edges 8g..8g+7; their 4 predecessor pairs are contiguous half2 loads;
//   * persistent blocks loop over tiles (scratch = blocks x 1 MB).
// Output: output_indices[ri] = (k << 14) | edge exactly as the reference's `(prev_edge << K) | edge`.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include "util.h"
#include "util.cuh"
#include "quant/codebook.cuh"

#ifndef H_INF
#define H_INF __ushort_as_half(0x7c00)
#endif

constexpr int L = 256;
constexpr int EDGES = 16384;          // 65536 >> K
constexpr int GROUPS = EDGES / 8;     // 2048 words of history per step
constexpr int NT = 512;
constexpr int GPT = GROUPS / NT;      // groups per thread per step

__device__ __forceinline__ uint32_t warp_min_u32(uint32_t v) { return __reduce_min_sync(0xffffffff, v); }

// MODE 0: first step, no predecessor cost (err = dh^2); MODE 1: first step constrained to pre_state (reference
// masks non-matching in-edges to +inf after hmul2); MODE 2: err = fma(dh, dh, cost[in_edge]).
// HMIN: running min cost via __hmin2 (equal to the sequential strict-less select for non-NaN costs; tiles with a NaN
// input run HMIN = false). CACHE: decoded codebook values held in registers across steps and tiles.
template <int MODE, bool HMIN, bool CACHE>
__device__ __forceinline__ void step(const half* __restrict__ prev, half* __restrict__ cur, uint16_t* __restrict__ hist,
                                     half w, int pre_state, bool store_hist, int thread, const half2 (&dc)[GPT][4][4])
{
    const half2 w2 = __half2half2(w);
    #pragma unroll
    for (int gi = 0; gi < GPT; ++gi)
    {
        const int g = thread + gi * NT;
        half2 c[4];
        if constexpr (MODE == 2)
        {
            #pragma unroll
            for (int k = 0; k < 4; ++k) c[k] = reinterpret_cast<const half2*>(prev)[(k << 11) | g];
        }
        half2 outc[4];
        uint32_t bits = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            const int e = 8 * g + 2 * j;
            half2 min_err2;
            uint32_t i0 = 0, i1 = 0;
            #pragma unroll
            for (int k = 0; k < 4; ++k)
            {
                half2 dec;
                if constexpr (CACHE) dec = dc[gi][j][k];
                else
                {
                    const uint32_t s0 = ((uint32_t) k << 14) | (uint32_t) e;
                    const uint32_t p0 = s0 * 0x83DCD12Du;
                    dec = decode_mul1_product_2(p0, p0 + 0x83DCD12Du);
                }
                const half2 dh = __hsub2(dec, w2);
                half2 err;
                if constexpr (MODE == 2)
                {
                    const half cp = j < 2 ? __low2half(c[k]) : __high2half(c[k]);
                    err = __hfma2(dh, dh, __half2half2(cp));
                }
                else
                {
                    err = __hmul2(dh, dh);
                    if constexpr (MODE == 1)
                    {
                        const int in_edge = (k << 12) | (e >> 2);
                        if (in_edge != pre_state) err = __half2half2(H_INF);
                    }
                }
                if (k == 0) { min_err2 = err; }
                else if constexpr (HMIN)
                {
                    const bool lo = __hlt(__low2half(err), __low2half(min_err2));
                    const bool hi = __hlt(__high2half(err), __high2half(min_err2));
                    i0 = lo ? (uint32_t) k : i0;
                    i1 = hi ? (uint32_t) k : i1;
                    min_err2 = __hmin2(min_err2, err);
                }
                else
                {
                    if (__hlt(__low2half(err), __low2half(min_err2)))
                    {
                        min_err2 = __halves2half2(__low2half(err), __high2half(min_err2));
                        i0 = k;
                    }
                    if (__hlt(__high2half(err), __high2half(min_err2)))
                    {
                        min_err2 = __halves2half2(__low2half(min_err2), __high2half(err));
                        i1 = k;
                    }
                }
            }
            outc[j] = min_err2;
            bits |= (i0 | (i1 << 2)) << (4 * j);
        }
        uint4 o;
        o.x = *reinterpret_cast<uint32_t*>(&outc[0]); o.y = *reinterpret_cast<uint32_t*>(&outc[1]);
        o.z = *reinterpret_cast<uint32_t*>(&outc[2]); o.w = *reinterpret_cast<uint32_t*>(&outc[3]);
        reinterpret_cast<uint4*>(cur)[g] = o;
        if (store_hist) hist[g] = (uint16_t) bits;
    }
}

template <bool HMIN, bool CACHE>
__global__ __launch_bounds__(NT, CACHE ? 1 : 2)
void nq_k2vit_kernel(const float* __restrict__ input, uint16_t* __restrict__ out_idx, float* __restrict__ out_tiles,
                     uint16_t* __restrict__ hist_all, int num_tiles)
{
    extern __shared__ __align__(16) uint8_t shbuf[];
    half* costA = (half*) shbuf;
    half* costB = costA + EDGES;
    half* sh_in = costB + EDGES;
    uint32_t* sh_red = (uint32_t*) (sh_in + L);
    const int thread = threadIdx.x;
    uint16_t* hist = hist_all + (size_t) blockIdx.x * L * GROUPS;

    half2 dc[GPT][4][4];
    if constexpr (CACHE)
    {
        #pragma unroll
        for (int gi = 0; gi < GPT; ++gi)
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                #pragma unroll
                for (int k = 0; k < 4; ++k)
                {
                    const uint32_t s0 = ((uint32_t) k << 14) | (uint32_t) (8 * (thread + gi * NT) + 2 * j);
                    const uint32_t p0 = s0 * 0x83DCD12Du;
                    dc[gi][j][k] = decode_mul1_product_2(p0, p0 + 0x83DCD12Du);
                }
    }

    for (int tile = blockIdx.x; tile < num_tiles; tile += gridDim.x)
    {
        int nan_here = 0;
        for (int i = thread; i < L; i += NT)
        {
            const float x = input[(size_t) L * tile + i];
            nan_here |= (x != x);
            sh_in[i] = __float2half_rn(x);
        }
        const bool fast = !__syncthreads_or(nan_here);     // also the barrier after the sh_in fill

        auto ring = [&](int i, int roll) { int r = i + roll; return r >= L ? r - L : r; };

        auto run_steps = [&](auto fast_tag, int roll, int pre_state) -> half*
        {
            constexpr bool F = decltype(fast_tag)::value;
            half* prev = costA; half* cur = costB;
            {
                const int ri = ring(0, roll);
                const bool sh = pre_state >= 0 || ri < L / 2;
                if (pre_state >= 0) step<1, F, CACHE>(nullptr, cur, hist + ri * GROUPS, sh_in[ri], pre_state, sh, thread, dc);
                else step<0, F, CACHE>(nullptr, cur, hist + ri * GROUPS, sh_in[ri], -1, sh, thread, dc);
                __syncthreads();
            }
            for (int i = 1; i < L; ++i)
            {
                half* t = prev; prev = cur; cur = t;
                const int ri = ring(i, roll);
                const bool sh = pre_state >= 0 || ri < L / 2;
                step<2, F, CACHE>(prev, cur, hist + ri * GROUPS, sh_in[ri], pre_state, sh, thread, dc);
                __syncthreads();
            }
            return cur;
        };
        auto forward = [&](int roll, int pre_state) -> half*
        {
            if (HMIN && fast) return run_steps(std::integral_constant<bool, HMIN>{}, roll, pre_state);
            return run_steps(std::false_type{}, roll, pre_state);
        };

        auto argmin_cost = [&](const half* costs) -> int
        {
            uint32_t best = 0x7c00ffffu;
            for (int e = thread; e < EDGES; e += NT)
            {
                unsigned v = e & 1023;
                unsigned rank = ((__brev(v >> 5) >> 27) << 10) | ((__brev(v & 31) >> 27) << 5) | (e >> 10);
                unsigned key = ((uint32_t) __half_as_ushort(costs[e]) << 16) | rank;
                best = min(best, key);
            }
            best = warp_min_u32(best);
            if ((thread & 31) == 0) sh_red[thread >> 5] = best;
            __syncthreads();
            if (thread < 32)
            {
                best = thread < NT / 32 ? sh_red[thread] : 0x7c00ffffu;
                best = warp_min_u32(best);
            }
            unsigned rank = best & 65535;
            unsigned v = ((__brev(rank >> 10) >> 27) << 5) | (__brev((rank >> 5) & 31) >> 27);
            return best >= 0x7c000000u ? 0 : (int) (((rank & 31) << 10) | v);
        };

        auto backward = [&](int roll, bool write, int edge) -> int
        {
            if (thread == 0)
            {
                for (int i = L - 1; i >= 0; --i)
                {
                    const int ri = ring(i, roll);
                    const uint32_t hw = __ldcg(hist + ri * GROUPS + (edge >> 3));
                    const int k = (hw >> (2 * (edge & 7))) & 3;
                    const int prev_edge = (k << 12) | (edge >> 2);
                    if (write)
                    {
                        const int encoded = (k << 14) | edge;
                        out_idx[(size_t) L * tile + ri] = (uint16_t) encoded;
                        if (out_tiles) out_tiles[(size_t) L * tile + ri] = __half2float(decode_3inst<2>(encoded));
                    }
                    edge = prev_edge;
                    if (!write && ri == 0) break;
                }
                sh_red[32] = edge;
            }
            __syncthreads();
            const int r = sh_red[32];
            __syncthreads();
            return r;
        };

        half* fin = forward(L / 2, -1);
        const int best = argmin_cost(fin);      // only thread 0's value is used (backward runs on thread 0)
        const int end_state = backward(L / 2, false, best);
        forward(0, end_state);
        backward(0, true, end_state);
    }
}

typedef void (*kfn_t)(const float*, uint16_t*, float*, uint16_t*, int);
static kfn_t pick(int variant)
{
    switch (variant)
    {
        case 0: return nq_k2vit_kernel<false, false>;
        case 1: return nq_k2vit_kernel<true, false>;
        case 2: return nq_k2vit_kernel<false, true>;
        case 3: return nq_k2vit_kernel<true, true>;
    }
    return nullptr;
}
constexpr int SHMEM = 2 * EDGES * sizeof(half) + L * sizeof(half) + 33 * sizeof(uint32_t) + 64;

int64_t nq_k2vit_blocks(int64_t device, int64_t variant)
{
    kfn_t k = pick((int) variant);
    TORCH_CHECK(k, "nq_k2vit: bad variant");
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM);
    int bps = 0, sms = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, k, NT, SHMEM);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, (int) device);
    return (int64_t) bps * sms;
}

// tiles [n, 256] fp32 (Viterbi step order) -> out_idx [n, 256] int16 states; hist scratch int16 [nb, 256, 2048]
void nq_k2vit(at::Tensor tiles, at::Tensor out_idx, c10::optional<at::Tensor> out_tiles, at::Tensor hist, int64_t variant)
{
    const at::cuda::OptionalCUDAGuard guard(tiles.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(tiles.dim() == 2 && tiles.size(1) == L && tiles.scalar_type() == at::kFloat && tiles.is_contiguous());
    TORCH_CHECK(out_idx.sizes() == tiles.sizes() && out_idx.scalar_type() == at::kShort && out_idx.is_contiguous());
    TORCH_CHECK(hist.dim() == 3 && hist.size(1) == L && hist.size(2) == GROUPS && hist.scalar_type() == at::kShort
                && hist.is_contiguous());
    float* ot = nullptr;
    if (out_tiles.has_value())
    {
        TORCH_CHECK(out_tiles->sizes() == tiles.sizes() && out_tiles->scalar_type() == at::kFloat && out_tiles->is_contiguous());
        ot = out_tiles->data_ptr<float>();
    }
    kfn_t k = pick((int) variant);
    TORCH_CHECK(k, "nq_k2vit: bad variant");
    const int n = tiles.size(0);
    if (!n) return;
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM);
    const int nb = (int) std::min<int64_t>(hist.size(0), n);
    k<<<nb, NT, SHMEM, stream>>>(tiles.data_ptr<float>(), (uint16_t*) out_idx.data_ptr(), ot, (uint16_t*) hist.data_ptr(), n);
    TORCH_CHECK(cudaPeekAtLastError() == cudaSuccess, "nq_k2vit launch");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("k2vit", &nq_k2vit);
    m.def("blocks", &nq_k2vit_blocks);
}
