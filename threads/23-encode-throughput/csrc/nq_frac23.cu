// Thread 23: exact re-implementation of exllamav3 quantize_tiles_frac_kernel<KA = 2, MASK, L = 256> (mul1), the
// pattern-rate residual Viterbi of NestQuant (production: K 2.3125 = (2, step mask 0x2492), T12 nq_fracvit.cu).
//
// Bit-identical by construction: same fp16 decode (decode_mul1_product_2), same per-candidate fp16 ops (hsub2, hmul2
// on the first step with the pre_state inf mask, hfma2 after), same sequential strict-less candidate selection from
// k = 0 in the reference's candidate order (state = (k << (16 - KOUT)) | out_edge), same argmin_cost rank tie order
// over the same edges_last, same two-pass tail-biting schedule. Differences are storage/scheduling only:
//   * costs in shared memory (2 x 16384 halves) instead of global scratch;
//   * history: the winning branch k per out-edge (<= 3 bits) instead of a uint16 in-edge, 1 MB per tile;
//   * thread <-> state mapping independent of the step widths: a thread owns low-13-bit windows 4g..4g+3
//     (g = thread + 512 gi, gi < 4) and, in a KOUT = 2 step, both out-edges low and low | 0x2000; its 128 states
//     s = (k' << 13) | low are the same in every step, so the decoded codebook values are cached in registers;
//   * persistent blocks loop over tiles.
// History word g of a step: KOUT = 2: 2-bit k of out-edge (b << 13) | (4g + j) at bit 2 (j + 4b);
//                           KOUT = 3: 3-bit k of out-edge 4g + j at bit 4 j.
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

namespace nqf {

constexpr int L = 256;
constexpr int KA = 2;
constexpr int EDGES = 65536 >> KA;    // widest node space (16384)
constexpr int WORDS = 2048;           // history words per step
constexpr int NT = 512;
constexpr int GPT = WORDS / NT;       // 4 groups (of 4 low windows) per thread

__device__ __forceinline__ uint32_t warp_min_u32(uint32_t v) { return __reduce_min_sync(0xffffffff, v); }

__device__ __forceinline__ half2 dec_pair(uint32_t s0)
{
    const uint32_t p0 = s0 * 0x83DCD12Du;
    return decode_mul1_product_2(p0, p0 + 0x83DCD12Du);
}

// MODE 0: first step, err = dh^2; MODE 1: first step, in_edge != pre_state -> +inf; MODE 2: err = fma(dh, dh, cost)
template <int MODE, bool HMIN, bool CACHE, int KIN, int KOUT>
__device__ __forceinline__ void step(const half* __restrict__ prev, half* __restrict__ cur, uint16_t* __restrict__ hist,
                                     half w, int pre_state, bool store_hist, int thread, const half2 (&dc)[GPT][2][8])
{
    constexpr int NK = 1 << KOUT;             // candidates per out-edge
    constexpr int NB = KOUT == 2 ? 2 : 1;     // out-edge copies (b = bit 13 of the window) per low window
    const half2 w2 = __half2half2(w);
    #pragma unroll
    for (int gi = 0; gi < GPT; ++gi)
    {
        const int g = thread + gi * NT;
        const int low0 = 4 * g;
        half cst[8];
        if constexpr (MODE == 2)
        {
            #pragma unroll
            for (int kp = 0; kp < 8; ++kp) cst[kp] = prev[(kp << (13 - KIN)) | (low0 >> KIN)];
        }
        uint32_t bits = 0;
        half2 outc[NB][2];
        #pragma unroll
        for (int b = 0; b < NB; ++b)
            #pragma unroll
            for (int p = 0; p < 2; ++p)
            {
                half2 min_err2;
                uint32_t i0 = 0, i1 = 0;
                #pragma unroll
                for (int k = 0; k < NK; ++k)
                {
                    const int kp = KOUT == 2 ? 2 * k + b : k;       // window bits 15..13
                    half2 dec;
                    if constexpr (CACHE) dec = dc[gi][p][kp];
                    else dec = dec_pair(((uint32_t) kp << 13) | (uint32_t) (low0 + 2 * p));
                    const half2 dh = __hsub2(dec, w2);
                    half2 err;
                    if constexpr (MODE == 2) err = __hfma2(dh, dh, __half2half2(cst[kp]));
                    else
                    {
                        err = __hmul2(dh, dh);
                        if constexpr (MODE == 1)
                        {
                            const int in_edge = ((kp << 13) | (low0 + 2 * p)) >> KIN;
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
                outc[b][p] = min_err2;
                const int j = 2 * p;
                if constexpr (KOUT == 2) bits |= (i0 << (2 * (j + 4 * b))) | (i1 << (2 * (j + 1 + 4 * b)));
                else bits |= (i0 << (4 * j)) | (i1 << (4 * (j + 1)));
            }
        #pragma unroll
        for (int b = 0; b < NB; ++b)
        {
            uint2 o;
            o.x = *reinterpret_cast<uint32_t*>(&outc[b][0]);
            o.y = *reinterpret_cast<uint32_t*>(&outc[b][1]);
            *reinterpret_cast<uint2*>(cur + (b << 13) + low0) = o;
        }
        if (store_hist) hist[g] = (uint16_t) bits;
    }
}

template <int MODE, bool HMIN, bool CACHE>
__device__ __forceinline__ void step_any(int din, int dout, const half* prev, half* cur, uint16_t* hist, half w,
                                         int pre_state, bool sh, int thread, const half2 (&dc)[GPT][2][8])
{
    switch (din * 2 + dout)
    {
        case 0: step<MODE, HMIN, CACHE, 2, 2>(prev, cur, hist, w, pre_state, sh, thread, dc); break;
        case 1: step<MODE, HMIN, CACHE, 2, 3>(prev, cur, hist, w, pre_state, sh, thread, dc); break;
        case 2: step<MODE, HMIN, CACHE, 3, 2>(prev, cur, hist, w, pre_state, sh, thread, dc); break;
        default: step<MODE, HMIN, CACHE, 3, 3>(prev, cur, hist, w, pre_state, sh, thread, dc); break;
    }
}

template <bool HMIN, bool CACHE>
__global__ __launch_bounds__(NT, CACHE ? 1 : 2)
void frac23_kernel(const float* __restrict__ input, uint16_t* __restrict__ out_idx, float* __restrict__ out_tiles,
                   uint16_t* __restrict__ hist_all, int num_tiles, uint32_t mask)
{
    extern __shared__ __align__(16) uint8_t shbuf[];
    half* costA = (half*) shbuf;
    half* costB = costA + EDGES;
    half* sh_in = costB + EDGES;
    uint32_t* sh_red = (uint32_t*) (sh_in + L);
    const int thread = threadIdx.x;
    uint16_t* hist = hist_all + (size_t) blockIdx.x * L * WORDS;
    const int edges_last = 65536 >> (KA + (mask & 1));

    half2 dc[GPT][2][8];
    if constexpr (CACHE)
    {
        #pragma unroll
        for (int gi = 0; gi < GPT; ++gi)
            #pragma unroll
            for (int p = 0; p < 2; ++p)
                #pragma unroll
                for (int kp = 0; kp < 8; ++kp)
                    dc[gi][p][kp] = dec_pair(((uint32_t) kp << 13) | (uint32_t) (4 * (thread + gi * NT) + 2 * p));
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
        const bool fast = !__syncthreads_or(nan_here);

        auto ring = [&](int i, int roll) { int r = i + roll; return r >= L ? r - L : r; };
        auto D = [&](int ri) { return (int) ((mask >> (ri & 15)) & 1); };

        auto run_steps = [&](auto fast_tag, int roll, int pre_state) -> half*
        {
            constexpr bool F = decltype(fast_tag)::value;
            half* prev = costA; half* cur = costB;
            {
                const int ri = ring(0, roll);
                const bool sh = pre_state >= 0 || ri < L / 2;
                if (pre_state >= 0)
                    step_any<1, F, CACHE>(D(ri), D(ri + 1), nullptr, cur, hist + ri * WORDS, sh_in[ri], pre_state, sh, thread, dc);
                else
                    step_any<0, F, CACHE>(D(ri), D(ri + 1), nullptr, cur, hist + ri * WORDS, sh_in[ri], -1, sh, thread, dc);
                __syncthreads();
            }
            for (int i = 1; i < L; ++i)
            {
                half* t = prev; prev = cur; cur = t;
                const int ri = ring(i, roll);
                const bool sh = pre_state >= 0 || ri < L / 2;
                step_any<2, F, CACHE>(D(ri), D(ri + 1), prev, cur, hist + ri * WORDS, sh_in[ri], pre_state, sh, thread, dc);
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
            for (int e = thread; e < edges_last; e += NT)
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
                    const int kin = KA + D(ri), kout = KA + D(ri + 1);
                    const uint32_t hw = __ldcg(hist + ri * WORDS + ((edge & 0x1FFF) >> 2));
                    int s;
                    if (kout == 2) s = (int) (((hw >> (2 * ((edge & 3) + 4 * ((edge >> 13) & 1)))) & 3) << 14) | edge;
                    else s = (int) (((hw >> (4 * (edge & 3))) & 7) << 13) | edge;
                    const int prev_edge = s >> kin;
                    if (write)
                    {
                        const int encoded = ((prev_edge << kin) | edge) & 0xFFFF;
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
        const int best = argmin_cost(fin);
        const int end_state = backward(L / 2, false, best);
        forward(0, end_state);
        backward(0, true, end_state);
    }
}

typedef void (*kfn_t)(const float*, uint16_t*, float*, uint16_t*, int, uint32_t);
static kfn_t pick(int variant)
{
    switch (variant)
    {
        case 0: return frac23_kernel<false, false>;
        case 1: return frac23_kernel<true, false>;
        case 2: return frac23_kernel<false, true>;
        case 3: return frac23_kernel<true, true>;
    }
    return nullptr;
}
constexpr int SHMEM = 2 * EDGES * sizeof(half) + L * sizeof(half) + 33 * sizeof(uint32_t) + 64;

}  // namespace nqf

int64_t nq_frac23_blocks(int64_t device, int64_t variant)
{
    nqf::kfn_t k = nqf::pick((int) variant);
    TORCH_CHECK(k, "nq_frac23: bad variant");
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, nqf::SHMEM);
    int bps = 0, sms = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, k, nqf::NT, nqf::SHMEM);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, (int) device);
    return (int64_t) bps * sms;
}

// tiles [n, 256] fp32 (Viterbi step order), mask = EXL3 Viterbi-step mask (D(i) = 2 + bit(i mod 16))
// -> out_idx [n, 256] int16 states (+ optional fp32 decoded tiles); hist scratch int16 [nb, 256, 2048]
void nq_frac23(at::Tensor tiles, at::Tensor out_idx, c10::optional<at::Tensor> out_tiles, at::Tensor hist, int64_t mask,
               int64_t variant)
{
    using namespace nqf;
    const at::cuda::OptionalCUDAGuard guard(tiles.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(tiles.dim() == 2 && tiles.size(1) == L && tiles.scalar_type() == at::kFloat && tiles.is_contiguous());
    TORCH_CHECK(out_idx.sizes() == tiles.sizes() && out_idx.scalar_type() == at::kShort && out_idx.is_contiguous());
    TORCH_CHECK(hist.dim() == 3 && hist.size(1) == L && hist.size(2) == WORDS && hist.scalar_type() == at::kShort
                && hist.is_contiguous());
    TORCH_CHECK(mask >= 0 && mask < 65536);
    float* ot = nullptr;
    if (out_tiles.has_value())
    {
        TORCH_CHECK(out_tiles->sizes() == tiles.sizes() && out_tiles->scalar_type() == at::kFloat && out_tiles->is_contiguous());
        ot = out_tiles->data_ptr<float>();
    }
    kfn_t k = pick((int) variant);
    TORCH_CHECK(k, "nq_frac23: bad variant");
    const int n = tiles.size(0);
    if (!n) return;
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM);
    const int nb = (int) std::min<int64_t>(hist.size(0), n);
    k<<<nb, NT, SHMEM, stream>>>(tiles.data_ptr<float>(), (uint16_t*) out_idx.data_ptr(), ot, (uint16_t*) hist.data_ptr(),
                                 n, (uint32_t) mask);
    TORCH_CHECK(cudaPeekAtLastError() == cudaSuccess, "nq_frac23 launch");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("frac23", &nq_frac23);
    m.def("blocks", &nq_frac23_blocks);
}
