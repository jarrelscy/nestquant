// T35 copy of threads/12 csrc/nq_fracvit.cu + base/residual patterns 1.75 (1,0xEEEE), 1.25 (1,0x8888), 2.5625, 2.8125.
// Pattern-rate Viterbi: exllamav3's quantize_tiles_frac_kernel<KA, MASK> (generic, only 0xAAAA shipped) instantiated
// for the NestQuant production patterns. MASK here is in EXL3 VITERBI-STEP convention: D(i) = KA + bit(i mod 16);
// nq_patvit converts from the kernel ring-position mask (bit j of the step mask = bit (16 - j) mod 16 of the kernel mask).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "quant/quantize_tiles_frac_kernel.cuh"

typedef void (*fq_t)(const float*, float*, uint16_t*, half*, uint16_t*, const half2*);

#define NQ_INST(X) \
    X(1, 0xfffeu) X(1, 0xfefeu) X(2, 0x2492u) X(2, 0x2222u) X(1, 0xaaaau) X(2, 0xaaaau) \
    X(1, 0xeeeeu) X(1, 0x2222u) X(2, 0xab56u) X(2, 0xf7beu)

static fq_t pick(int ka, uint32_t mask)
{
#define NQ_CASE(A, M) if (ka == A && mask == M) return quantize_tiles_frac_kernel<A, M>;
    NQ_INST(NQ_CASE)
#undef NQ_CASE
    return nullptr;
}

void nq_quantize_tiles_frac(at::Tensor tiles, at::Tensor out_tiles, at::Tensor out_idx, at::Tensor temp_costs,
                            at::Tensor temp_edges, int64_t ka, int64_t mask)
{
    const at::cuda::OptionalCUDAGuard guard(tiles.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(tiles.dim() == 2 && tiles.size(1) == 256 && tiles.scalar_type() == at::kFloat && tiles.is_contiguous());
    TORCH_CHECK(temp_costs.size(2) == (65536 >> ka) && temp_edges.size(2) == (65536 >> ka) && temp_edges.size(1) == 256);
    fq_t k = pick((int) ka, (uint32_t) mask);
    TORCH_CHECK(k, "nq_fracvit: no instance for (KA, MASK) = (", ka, ", ", mask, ")");
    const int n = tiles.size(0);
    const int shmem = 256 * sizeof(half) + 32 * sizeof(int) + 128;
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    const int mb = (int) std::min(temp_costs.size(0), temp_edges.size(0));
    for (int b = 0; b < n; b += mb)
    {
        const int bs = std::min(mb, n - b);
        k<<<bs, 512, shmem, stream>>>(tiles.data_ptr<float>() + 256LL * b, out_tiles.data_ptr<float>() + 256LL * b,
                                      (uint16_t*) out_idx.data_ptr() + 256LL * b, (half*) temp_costs.data_ptr(),
                                      (uint16_t*) temp_edges.data_ptr(), nullptr);
        TORCH_CHECK(cudaPeekAtLastError() == cudaSuccess, "nq_fracvit launch");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("quantize_tiles_frac", &nq_quantize_tiles_frac); }
