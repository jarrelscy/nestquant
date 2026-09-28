// NestQuant decode microkernels: batch 1-4 GEMV via mma.m16n8k16 (fp16 x fp16 -> fp32),
// weights decoded in registers directly into A fragments. Random data, real shapes.
//
// Tiling (identical at every level): strip = 16 output rows, chunk = 128 k (8 mma k-tiles);
// each lane owns 64 weights of a (strip, chunk): pairs p = 0..31 -> k-tile t = p>>2, frag reg r = p&3.
// Planes (per strip, chunk, lane): base = uint4 (128 bit), P3 = uint2, P4 = uint2.
// Single-stream 4-bit formats use a 256-bit per-lane record (two uint4, interleaved).
#include <cuda_fp16.h>
#include <stdint.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

enum { UNI2 = 0, UNI4, T2, T4, H2, H4, T2R1, T2R2, T2H, SYN4, SYN2, H2B, J4, J3, NDEC };

__device__ __forceinline__ uint32_t win(const uint32_t* w, int n, int o, uint32_t mask)
{
    o &= (32 * n - 1);
    int i = o >> 5, s = o & 31;
    uint32_t v = s ? __funnelshift_r(w[i], w[(i + 1) & (n - 1)], s) : w[i];
    return v & mask;
}
__device__ __forceinline__ uint32_t mul1_pair(uint32_t v0, uint32_t v1)
{
    uint32_t x0 = v0 * 0x83DCD12Du, x1 = v1 * 0x83DCD12Du;
    uint32_t s0 = __dp4a(x0, 0x01010101u, 0x6400u), s1 = __dp4a(x1, 0x01010101u, 0x6400u);
    uint32_t h = __byte_perm(s0, s1, 0x5410);
    half2 r = __hfma2(*(half2*)&h, __half2half2(__ushort_as_half(0x1eee)), __half2half2(__ushort_as_half(0xc931)));
    return *(uint32_t*)&r;
}
__device__ __forceinline__ uint32_t hfma2u(uint32_t a, uint32_t s, uint32_t b)
{
    half2 r = __hfma2(*(half2*)&a, *(half2*)&s, *(half2*)&b);
    return *(uint32_t*)&r;
}
template <bool LUTR>
__device__ __forceinline__ uint32_t hyb(uint32_t v, const uint32_t* lut, int lane)
{
    uint32_t h = v * v + v;  // QTIP-style quadratic hash, 1 IMAD
    if constexpr (LUTR) return lut[((h >> 8) & 0xFF) * 32 + lane];   // bank-replicated Q=8
    else return lut[(h >> 7) & 0x1FF];                               // shared Q=9
}

template <int DEC> struct Cfg { static constexpr int NB = 4, N3 = 0, N4 = 0, LUT = 0; };
template <> struct Cfg<UNI4> { static constexpr int NB = 8, N3 = 0, N4 = 0, LUT = 0; };
template <> struct Cfg<T4>   { static constexpr int NB = 8, N3 = 0, N4 = 0, LUT = 0; };
template <> struct Cfg<H4>   { static constexpr int NB = 8, N3 = 0, N4 = 0, LUT = 1; };
template <> struct Cfg<SYN4> { static constexpr int NB = 8, N3 = 0, N4 = 0, LUT = 0; };
template <> struct Cfg<H2>   { static constexpr int NB = 4, N3 = 0, N4 = 0, LUT = 1; };
template <> struct Cfg<H2B>  { static constexpr int NB = 4, N3 = 0, N4 = 0, LUT = 1; };
template <> struct Cfg<T2R1> { static constexpr int NB = 4, N3 = 2, N4 = 0, LUT = 0; };
template <> struct Cfg<T2R2> { static constexpr int NB = 4, N3 = 2, N4 = 2, LUT = 0; };
template <> struct Cfg<J4>   { static constexpr int NB = 4, N3 = 2, N4 = 2, LUT = 0; };
template <> struct Cfg<J3>   { static constexpr int NB = 4, N3 = 2, N4 = 0, LUT = 0; };
template <> struct Cfg<T2H>  { static constexpr int NB = 4, N3 = 2, N4 = 2, LUT = 1; };

// Decode pair p (weights 2p, 2p+1 of the lane's 64) into a half2 (as uint32).
template <int DEC, bool LUTR, int R>
__device__ __forceinline__ uint32_t decode_pair(const uint32_t* b, const uint32_t* p3, const uint32_t* p4,
                                                int p, const uint32_t* lut, int lane, uint32_t sc)
{
    if constexpr (DEC == UNI2)
    {
        // 16 x 2-bit per word; Marlin-style magic: (q | 0x6400) - 1025.5 -> {-1.5..1.5}
        uint32_t w = b[p >> 3]; int s = (p & 7) * 2;
        uint32_t q = ((w >> s) & 0x00030003u) | 0x64006400u;   // pairs at bits s and s+16
        half2 r = __hfma2(__hsub2(*(half2*)&q, __half2half2(__ushort_as_half(0x6402))), *(half2*)&sc, __half2half2(__float2half(0.f)));
        return *(uint32_t*)&r;
    }
    if constexpr (DEC == UNI4)
    {
        uint32_t w = b[p >> 2]; int s = (p & 3) * 4;
        uint32_t q = ((w >> s) & 0x000F000Fu) | 0x64006400u;
        half2 r = __hmul2(__hsub2(*(half2*)&q, __half2half2(__ushort_as_half(0x6408))), *(half2*)&sc);
        return *(uint32_t*)&r;
    }
    if constexpr (DEC == T2)   // EXL3 mul1, 2 bpw, per-lane 128-bit tail-biting trellis
        return mul1_pair(win(b, 4, 4 * p, 0xFFFF), win(b, 4, 4 * p + 2, 0xFFFF));
    if constexpr (DEC == T4)
        return mul1_pair(win(b, 8, 8 * p, 0xFFFF), win(b, 8, 8 * p + 4, 0xFFFF));
    if constexpr (DEC == SYN4 || DEC == SYN2)
    {
        constexpr int NW = DEC == SYN4 ? 8 : 4, ST = DEC == SYN4 ? 4 : 2;
        uint32_t v0 = win(b, NW, 2 * ST * p, 0xFFFF), v1 = win(b, NW, 2 * ST * p + ST, 0xFFFF);
        #pragma unroll
        for (int i = 0; i < R; ++i)
        {
            asm volatile ("mad.lo.u32 %0, %0, %1, %2;" : "+r"(v0) : "r"(0x9E3779B1u), "r"(v1));
            asm volatile ("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(v1) : "r"(v0), "r"(0x5bd1e995u));
        }
        return mul1_pair(v0 & 0xFFFF, v1 & 0xFFFF);
    }
    if constexpr (DEC == H2)   // HYB V=2 at 2 bpw: 4 bits per pair
        return hyb<LUTR>(win(b, 4, 4 * p, 0xFFFF), lut, lane);
    if constexpr (DEC == H2B)  // HYB V=2 with sign bit folded (Q=9 + sign)
    {
        uint32_t v = win(b, 4, 4 * p, 0xFFFF);
        uint32_t h = v * v + v;
        uint32_t e = LUTR ? lut[((h >> 8) & 0xFF) * 32 + lane] : lut[(h >> 7) & 0x1FF];
        return e ^ (h & 0x80008000u);
    }
    if constexpr (DEC == H4)   // HYB V=2 at 4 bpw: 8 bits per pair
        return hyb<LUTR>(win(b, 8, 8 * p, 0xFFFF), lut, lane);
    if constexpr (DEC == T2R1) // 3 bpw progressive: base T2 + 1-bit trellis refinement from P3 (64-bit)
    {
        uint32_t base = mul1_pair(win(b, 4, 4 * p, 0xFFFF), win(b, 4, 4 * p + 2, 0xFFFF));
        uint32_t ref = mul1_pair(win(p3, 2, 2 * p, 0xFFFF), win(p3, 2, 2 * p + 1, 0xFFFF));
        return hfma2u(ref, sc, base);
    }
    if constexpr (DEC == T2R2) // 4 bpw progressive: base T2 + 2-bit refinement, window = 8b(P3) | 8b(P4) << 8
    {
        uint32_t base = mul1_pair(win(b, 4, 4 * p, 0xFFFF), win(b, 4, 4 * p + 2, 0xFFFF));
        uint32_t r0 = __byte_perm(win(p3, 2, 2 * p, 0xFF), win(p4, 2, 2 * p, 0xFF), 0x7340);
        uint32_t r1 = __byte_perm(win(p3, 2, 2 * p + 1, 0xFF), win(p4, 2, 2 * p + 1, 0xFF), 0x7340);
        return hfma2u(mul1_pair(r0, r1), sc, base);
    }
    if constexpr (DEC == J4)   // 4 bpw joint window: one mul1 decode, window = 8b base | 4b P3 | 4b P4 (reinterprets base)
    {
        uint32_t b0 = win(b, 4, 4 * p, 0xFF), b1 = win(b, 4, 4 * p + 2, 0xFF);
        uint32_t q3 = win(p3, 2, 2 * p, 0x1F), q4 = win(p4, 2, 2 * p, 0x1F);
        uint32_t v0 = b0 | ((q3 & 0xF) << 8) | ((q4 & 0xF) << 12);
        uint32_t v1 = b1 | ((q3 & 0x1E) << 7) | ((q4 & 0x1E) << 11);
        return mul1_pair(v0, v1);
    }
    if constexpr (DEC == J3)   // 3 bpw joint window: window = 8b base | 8b P3
    {
        uint32_t b0 = win(b, 4, 4 * p, 0xFF), b1 = win(b, 4, 4 * p + 2, 0xFF);
        uint32_t q3 = win(p3, 2, 2 * p, 0x1FF);
        return mul1_pair(b0 | ((q3 & 0xFF) << 8), b1 | ((q3 & 0x1FE) << 7));
    }
    if constexpr (DEC == T2H)  // 4 bpw progressive: base T2 + HYB V=2 refinement (2b P3 + 2b P4 per pair)
    {
        uint32_t base = mul1_pair(win(b, 4, 4 * p, 0xFFFF), win(b, 4, 4 * p + 2, 0xFFFF));
        uint32_t v = __byte_perm(win(p3, 2, 2 * p, 0xFF), win(p4, 2, 2 * p, 0xFF), 0x7340);
        return hfma2u(hyb<LUTR>(v, lut, lane), sc, base);
    }
    return 0;
}

__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1)
{
    asm volatile ("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// grid: (strips / SB, chunks / (WK * CPW * NST)); block 32*SB*WK threads.
// Each warp streams NST stages of CPW chunks with a 2-deep register ring (load stage s+2 while computing s+1).
// y_acc [B, N] fp32 accumulated with atomics. x [B, K] fp16. If wdbg != nullptr, writes dense decoded W [N, K].
template <int DEC, int CPW>
struct Stage
{
    using C = Cfg<DEC>;
    uint32_t wb[CPW][C::NB], w3[CPW][C::N3 ? C::N3 : 1], w4[CPW][C::N4 ? C::N4 : 1];
};
template <int DEC, int CPW, bool SPLIT>
__device__ __forceinline__ void load_stage(Stage<DEC, CPW>& S, const uint4* __restrict__ base, const uint2* __restrict__ pl3,
                                           const uint2* __restrict__ pl4, size_t rec0, int istride)
{
    using C = Cfg<DEC>;
    #pragma unroll
    for (int c = 0; c < CPW; ++c)
    {
        size_t rec = rec0 + c * 32;
        if (!SPLIT)
        {
            const uint4* pr = base + rec * istride;
            uint4 v = pr[0]; S.wb[c][0] = v.x; S.wb[c][1] = v.y; S.wb[c][2] = v.z; S.wb[c][3] = v.w;
            if constexpr (C::N3) { uint4 u = pr[1]; S.w3[c][0] = u.x; S.w3[c][1] = u.y; if constexpr (C::N4) { S.w4[c][0] = u.z; S.w4[c][1] = u.w; } }
        }
        else
        {
            if constexpr (C::NB == 8) { uint4 v = base[rec * 2]; uint4 u = base[rec * 2 + 1];
                S.wb[c][0] = v.x; S.wb[c][1] = v.y; S.wb[c][2] = v.z; S.wb[c][3] = v.w; S.wb[c][4] = u.x; S.wb[c][5] = u.y; S.wb[c][6] = u.z; S.wb[c][7] = u.w; }
            else { uint4 v = base[rec]; S.wb[c][0] = v.x; S.wb[c][1] = v.y; S.wb[c][2] = v.z; S.wb[c][3] = v.w; }
            if constexpr (C::N3) { uint2 v = pl3[rec]; S.w3[c][0] = v.x; S.w3[c][1] = v.y; }
            if constexpr (C::N4) { uint2 v = pl4[rec]; S.w4[c][0] = v.x; S.w4[c][1] = v.y; }
        }
    }
}
template <int DEC, bool LUTR, int CPW, int R>
__device__ __forceinline__ void compute_stage(const Stage<DEC, CPW>& S, float* acc, const half* xs, int kslice, int kc0,
                                              const uint32_t* lut, int lane, int B, float* wdbg, int strip, int kabs0, int K)
{
    const int g = lane >> 2, t4 = lane & 3;
    const uint32_t sc = 0x2C002C00u;
    #pragma unroll
    for (int c = 0; c < CPW; ++c)
    {
        const int kc = kc0 + c * 128;
        #pragma unroll
        for (int t = 0; t < 8; ++t)
        {
            uint32_t a[4];
            #pragma unroll
            for (int r = 0; r < 4; ++r)
                a[r] = decode_pair<DEC, LUTR, R>(S.wb[c], S.w3[c], S.w4[c], t * 4 + r, lut, lane, sc);
            if (wdbg)
            {
                #pragma unroll
                for (int r = 0; r < 4; ++r)
                {
                    int row = strip * 16 + g + (r & 1) * 8, k = kabs0 + kc + t * 16 + t4 * 2 + (r >> 1) * 8;
                    half2 h = *(half2*) &a[r];
                    wdbg[(size_t) row * K + k] = __low2float(h); wdbg[(size_t) row * K + k + 1] = __high2float(h);
                }
            }
            uint32_t b0 = 0, b1 = 0;
            if (g < B)
            {
                const half* xr = xs + g * kslice + kc + t * 16 + t4 * 2;
                b0 = *(const uint32_t*) xr; b1 = *(const uint32_t*) (xr + 8);
            }
            mma16816(acc, a, b0, b1);
        }
    }
}

template <int DEC, bool LUTR, int CPW, int R, bool SPLIT>
__global__ void __launch_bounds__(256) nq_gemv(
    const half* __restrict__ x, const uint4* __restrict__ base, const uint2* __restrict__ pl3, const uint2* __restrict__ pl4,
    const uint32_t* __restrict__ lut_g, float* __restrict__ y, int B, int N, int K, int SB, int WK, float* wdbg, int istride, int NST)
{
    using C = Cfg<DEC>;
    extern __shared__ uint32_t smem[];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int ws = warp / WK, wk = warp % WK;
    const int strip = blockIdx.x * SB + ws;
    const int nchunks = K / 128;
    const int wslice = CPW * NST * 128;         // k per warp
    const int kslice = WK * wslice;             // k per block
    const int k0 = blockIdx.y * kslice;
    half* xs = (half*) smem;
    uint32_t* lut = smem + (4 * kslice) / 2;
    constexpr int LUTN = C::LUT ? (LUTR ? 256 * 32 : 512) : 0;
    float* red = (float*) (lut + LUTN);

    const int chunk0 = (blockIdx.y * WK + wk) * CPW * NST;
    const size_t rec0 = ((size_t) strip * nchunks + chunk0) * 32 + lane;
    Stage<DEC, CPW> SA, SB_;
    load_stage<DEC, CPW, SPLIT>(SA, base, pl3, pl4, rec0, istride);
    if (NST > 1) load_stage<DEC, CPW, SPLIT>(SB_, base, pl3, pl4, rec0 + CPW * 32, istride);

    for (int i = threadIdx.x; i < B * kslice / 2; i += blockDim.x)
    {
        int bb = i / (kslice / 2), kk = i % (kslice / 2);
        ((half2*) xs)[i] = ((const half2*) (x + (size_t) bb * K + k0))[kk];
    }
    if constexpr (C::LUT)
    {
        if constexpr (LUTR) for (int i = threadIdx.x; i < 256 * 32; i += blockDim.x) lut[i] = lut_g[i >> 5];
        else for (int i = threadIdx.x; i < 512; i += blockDim.x) lut[i] = lut_g[i];
    }
    __syncthreads();

    float acc[4] = {0, 0, 0, 0};
    const half* xw = xs;   // warp's k offset within block slice = wk * wslice
    const int kw = wk * wslice;
    for (int s = 0; s < NST; s += 2)
    {
        compute_stage<DEC, LUTR, CPW, R>(SA, acc, xw, kslice, kw + s * CPW * 128, lut, lane, B, wdbg, strip, k0, K);
        if (s + 2 < NST) load_stage<DEC, CPW, SPLIT>(SA, base, pl3, pl4, rec0 + (s + 2) * CPW * 32, istride);
        if (s + 1 < NST)
        {
            compute_stage<DEC, LUTR, CPW, R>(SB_, acc, xw, kslice, kw + (s + 1) * CPW * 128, lut, lane, B, wdbg, strip, k0, K);
            if (s + 3 < NST) load_stage<DEC, CPW, SPLIT>(SB_, base, pl3, pl4, rec0 + (s + 3) * CPW * 32, istride);
        }
    }
    if (WK > 1)
    {
        float* rr = red + (ws * WK + wk) * 128;
        #pragma unroll
        for (int i = 0; i < 4; ++i) rr[lane * 4 + i] = acc[i];
        __syncthreads();
        if (wk == 0)
        {
            for (int j = 1; j < WK; ++j)
            {
                float* ro = red + (ws * WK + j) * 128;
                #pragma unroll
                for (int i = 0; i < 4; ++i) acc[i] += ro[lane * 4 + i];
            }
        }
        else return;
    }
    const int g = lane >> 2, t4 = lane & 3;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        int bb = t4 * 2 + (i & 1), row = strip * 16 + g + (i >> 1) * 8;
        if (bb < B) atomicAdd(y + (size_t) bb * N + row, acc[i]);
    }
}

// ---------------- fused epilogue: Hadamard-128 (+sign) on gate/up accumulators, SwiGLU, down-input
// Hadamard-128, write fp16 h; zero accumulators for next call. One block per (batch, 128-col group).
__device__ void had128_warp4(float* v)   // 128 values in smem, 128 threads; in-place fast WHT
{
    int t = threadIdx.x;
    for (int h = 1; h < 128; h <<= 1)
    {
        __syncthreads();
        float a = v[t]; float b = v[t ^ h];
        __syncthreads();
        v[t] = (t & h) ? (b - a) : (a + b);
    }
    __syncthreads();
}
__global__ void nq_swiglu(float* acc, half* h, const half* sv_g, const half* sv_u, const half* su_d, float* yzero, int B, int I, int Nout)
{
    __shared__ float vg[128], vu[128];
    int bb = blockIdx.y, grp = blockIdx.x, t = threadIdx.x;
    int col = grp * 128 + t;
    vg[t] = acc[(size_t) bb * 2 * I + col]; vu[t] = acc[(size_t) bb * 2 * I + I + col];
    acc[(size_t) bb * 2 * I + col] = 0.f; acc[(size_t) bb * 2 * I + I + col] = 0.f;
    had128_warp4(vg); had128_warp4(vu);
    float gg = vg[t] * __half2float(sv_g[col]) * 0.08838834764f, uu = vu[t] * __half2float(sv_u[col]) * 0.08838834764f;
    float s = gg / (1.f + __expf(-gg)) * uu;
    __syncthreads();
    vg[t] = s * __half2float(su_d[col]);
    had128_warp4(vg);
    h[(size_t) bb * I + col] = __float2half(vg[t] * 0.08838834764f);
    // zero the down accumulator (Nout = 6144 per batch row), spread across blocks
    int nb = gridDim.x * gridDim.y, bid = bb * gridDim.x + grp;
    for (int i = bid * 128 + t; i < B * Nout; i += nb * 128) yzero[i] = 0.f;
}
// input Hadamard on x (and output Hadamard on y): generic block-128 WHT with sign, fp16 out
__global__ void nq_had_in(const float* xin, half* xout, const half* su, int K)
{
    __shared__ float v[128];
    int bb = blockIdx.y, grp = blockIdx.x, t = threadIdx.x, col = grp * 128 + t;
    v[t] = xin[(size_t) bb * K + col] * __half2float(su[col]);
    had128_warp4(v);
    xout[(size_t) bb * K + col] = __float2half(v[t] * 0.08838834764f);
}

typedef void (*kfn)(const half*, const uint4*, const uint2*, const uint2*, const uint32_t*, float*, int, int, int, int, int, float*, int, int);

template <int DEC, bool LUTR, int R, bool SPLIT>
kfn pick_cpw(int cpw)
{
    if (cpw == 1) return nq_gemv<DEC, LUTR, 1, R, SPLIT>;
    if (cpw == 2) return nq_gemv<DEC, LUTR, 2, R, SPLIT>;
    if (cpw == 4) return nq_gemv<DEC, LUTR, 4, R, SPLIT>;
    return nullptr;
}
template <int DEC, bool LUTR, bool SPLIT>
kfn pick_r(int r, int cpw)
{
    if constexpr (DEC == SYN4 || DEC == SYN2)
    {
        switch (r) { case 0: return pick_cpw<DEC, LUTR, 0, SPLIT>(cpw); case 1: return pick_cpw<DEC, LUTR, 1, SPLIT>(cpw);
                     case 2: return pick_cpw<DEC, LUTR, 2, SPLIT>(cpw); case 4: return pick_cpw<DEC, LUTR, 4, SPLIT>(cpw);
                     case 6: return pick_cpw<DEC, LUTR, 6, SPLIT>(cpw); case 8: return pick_cpw<DEC, LUTR, 8, SPLIT>(cpw);
                     case 12: return pick_cpw<DEC, LUTR, 12, SPLIT>(cpw); case 16: return pick_cpw<DEC, LUTR, 16, SPLIT>(cpw); }
        return nullptr;
    }
    else return pick_cpw<DEC, LUTR, 0, SPLIT>(cpw);
}
template <int DEC> constexpr bool has_lut() { return DEC == H2 || DEC == H4 || DEC == T2H || DEC == H2B; }
template <int DEC> constexpr bool has_planes() { return DEC == T2R2 || DEC == T2H || DEC == T2R1 || DEC == J4 || DEC == J3; }
template <int DEC, bool LUTR, bool SPLIT>
kfn guard(int r, int cpw)
{
    if constexpr ((LUTR && !has_lut<DEC>()) || (!SPLIT && !has_planes<DEC>())) return nullptr;
    else return pick_r<DEC, LUTR, SPLIT>(r, cpw);
}
template <bool LUTR, bool SPLIT>
kfn pick_dec(int dec, int r, int cpw)
{
    switch (dec)
    {
        case UNI2: return guard<UNI2, LUTR, SPLIT>(r, cpw); case UNI4: return guard<UNI4, LUTR, SPLIT>(r, cpw);
        case T2: return guard<T2, LUTR, SPLIT>(r, cpw); case T4: return guard<T4, LUTR, SPLIT>(r, cpw);
        case H2: return guard<H2, LUTR, SPLIT>(r, cpw); case H4: return guard<H4, LUTR, SPLIT>(r, cpw);
        case T2R1: return guard<T2R1, LUTR, SPLIT>(r, cpw); case T2R2: return guard<T2R2, LUTR, SPLIT>(r, cpw);
        case T2H: return guard<T2H, LUTR, SPLIT>(r, cpw); case SYN4: return guard<SYN4, LUTR, SPLIT>(r, cpw);
        case SYN2: return guard<SYN2, LUTR, SPLIT>(r, cpw); case H2B: return guard<H2B, LUTR, SPLIT>(r, cpw);
        case J4: return guard<J4, LUTR, SPLIT>(r, cpw); case J3: return guard<J3, LUTR, SPLIT>(r, cpw);
    }
    return nullptr;
}

void gemv(torch::Tensor x, torch::Tensor base, torch::Tensor p3, torch::Tensor p4, torch::Tensor lut, torch::Tensor y,
          int64_t dec, bool lutr, bool split, int64_t r, int64_t cpw, int64_t sb, int64_t wk, int64_t N, int64_t K,
          torch::Tensor wdbg, int64_t istride, int64_t nst)
{
    int B = x.size(0);
    kfn f = split ? (lutr ? pick_dec<true, true>(dec, r, cpw) : pick_dec<false, true>(dec, r, cpw))
                  : (lutr ? pick_dec<true, false>(dec, r, cpw) : pick_dec<false, false>(dec, r, cpw));
    TORCH_CHECK(f, "no kernel");
    int kslice = wk * cpw * nst * 128;
    TORCH_CHECK(K % kslice == 0 && (N / 16) % sb == 0 && sb * wk <= 8);
    bool haslut = (dec == H2 || dec == H4 || dec == T2H || dec == H2B);
    int lutn = haslut ? (lutr ? 256 * 32 : 512) : 0;
    int shm = 4 * kslice * 2 + lutn * 4 + sb * wk * 128 * 4;
    static bool set[2][2][NDEC][17][5] = {};
    if (!set[lutr][split][dec][r][cpw]) { cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, 100 * 1024); set[lutr][split][dec][r][cpw] = true; }
    dim3 grid(N / 16 / sb, K / kslice);
    f<<<grid, 32 * sb * wk, shm, at::cuda::getCurrentCUDAStream()>>>(
        (const half*) x.data_ptr(), (const uint4*) base.data_ptr(), (const uint2*) p3.data_ptr(), (const uint2*) p4.data_ptr(),
        (const uint32_t*) lut.data_ptr(), (float*) y.data_ptr(), B, N, K, sb, wk,
        wdbg.numel() ? (float*) wdbg.data_ptr() : nullptr, istride, nst);
}
void swiglu(torch::Tensor acc, torch::Tensor h, torch::Tensor svg, torch::Tensor svu, torch::Tensor sud, torch::Tensor yzero)
{
    int B = h.size(0), I = h.size(1);
    nq_swiglu<<<dim3(I / 128, B), 128, 0, at::cuda::getCurrentCUDAStream()>>>((float*) acc.data_ptr(), (half*) h.data_ptr(),
        (const half*) svg.data_ptr(), (const half*) svu.data_ptr(), (const half*) sud.data_ptr(), (float*) yzero.data_ptr(), B, I, yzero.size(1));
}
void had_in(torch::Tensor xin, torch::Tensor xout, torch::Tensor su)
{
    int B = xin.size(0), K = xin.size(1);
    nq_had_in<<<dim3(K / 128, B), 128, 0, at::cuda::getCurrentCUDAStream()>>>((const float*) xin.data_ptr(), (half*) xout.data_ptr(), (const half*) su.data_ptr(), K);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gemv", &gemv); m.def("swiglu", &swiglu); m.def("had_in", &had_in);
}
